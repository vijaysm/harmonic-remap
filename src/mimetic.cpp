#include "mimetic/mimetic.hpp"

#include <moab/MergeMesh.hpp>
#include <nanoflann.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>

#ifdef MIMETIC_ENABLE_OPENMP
#include <omp.h>
#endif

namespace mimetic {

namespace {
constexpr double kPi = 3.141592653589793238462643383279502884;

struct PolygonSearchGeometry {
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    double radius = 0.0;
};

/// Nanoflann adapter for a vector of 3D points (source cell centroids).
struct PointCloud3D {
    std::vector<Eigen::Vector3d> pts;

    inline std::size_t kdtree_get_point_count() const { return pts.size(); }
    inline double kdtree_get_pt(const std::size_t idx, const std::size_t dim) const {
        return pts[idx](static_cast<Eigen::Index>(dim));
    }
    template <class BBOX> bool kdtree_get_bbox(BBOX&) const { return false; }
};

using KDTree3D = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, PointCloud3D>,
    PointCloud3D, 3>;

/// Build a k-d tree from source cell centroids (3D normalized for spherical).
struct SpatialIndex {
    PointCloud3D cloud;
    std::unique_ptr<KDTree3D> tree;
    std::vector<double> cell_radius;  // max chord distance per cell
    double max_radius = 0;            // max over all cell radii

    void build(const std::vector<Eigen::Vector3d>& centers,
               const std::vector<double>& radii) {
        cloud.pts = centers;
        cell_radius = radii;
        max_radius = radii.empty() ? 0 : *std::max_element(radii.begin(), radii.end());
        tree = std::make_unique<KDTree3D>(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        tree->buildIndex();
    }

    /// Find source cell indices whose centroid is within search_radius of query_point.
    /// search_radius is in Euclidean 3D distance (not angular).
    std::vector<std::size_t> find_candidates(const Eigen::Vector3d& query_point,
                                              double search_radius) const {
        std::vector<nanoflann::ResultItem<unsigned int, double>> matches;
        nanoflann::SearchParameters params;
        params.sorted = false;
        const double query_pt[3] = {query_point.x(), query_point.y(), query_point.z()};
        tree->radiusSearch(query_pt, search_radius * search_radius, matches, params);
        std::vector<std::size_t> indices;
        indices.reserve(matches.size());
        for (const auto& m : matches) indices.push_back(m.first);
        return indices;
    }
};

double clamp_unit(const double value)
{
    return std::max(-1.0, std::min(1.0, value));
}

PolygonSearchGeometry polygon_search_geometry(const LocalPolygon& polygon,
                                              const GeometryOptions& options)
{
    PolygonSearchGeometry geometry;
    if (options.mode == GeometryMode::SphericalGnomonic) {
        geometry.center = polygon.n.normalized();
        for (const Eigen::Vector3d& p3d : polygon.points_3d) {
            const double dist = (geometry.center - p3d.normalized()).norm();
            geometry.radius = std::max(geometry.radius, dist);
        }
    } else {
        geometry.center = polygon.centroid_3d;
        for (const Eigen::Vector3d& p3d : polygon.points_3d) {
            const double dist = (geometry.center - p3d).norm();
            geometry.radius = std::max(geometry.radius, dist);
        }
    }
    return geometry;
}

/// Build a spatial index from source cell LocalPolygons.
/// Works for any container of structs that have a `.local` field of type LocalPolygon.
template <typename SourceVec>
SpatialIndex build_spatial_index_from_sources(const SourceVec& sources, bool is_spherical)
{
    SpatialIndex idx;
    if (sources.size() <= 50) return idx;

    std::vector<Eigen::Vector3d> centers(sources.size());
    std::vector<double> radii(sources.size());
    GeometryOptions options;
    options.mode = is_spherical ? GeometryMode::SphericalGnomonic : GeometryMode::Planar;
    for (std::size_t i = 0; i < sources.size(); ++i) {
        const PolygonSearchGeometry geometry = polygon_search_geometry(sources[i].local, options);
        centers[i] = geometry.center;
        radii[i] = geometry.radius;
    }
    idx.build(centers, radii);
    return idx;
}

/// Find candidate source indices for a target cell or edge using the spatial index.
std::vector<std::size_t> find_overlap_candidates(
    const SpatialIndex& idx,
    const Eigen::Vector3d& query_center,
    const double query_radius,
    const std::size_t total_sources)
{
    if (!idx.tree) {
        std::vector<std::size_t> all(total_sources);
        for (std::size_t i = 0; i < total_sources; ++i) all[i] = i;
        return all;
    }

    const double padding = std::max(10.0 * kTolerance, 0.25 * (query_radius + idx.max_radius));
    const double broad_radius = query_radius + idx.max_radius + padding;
    std::vector<std::size_t> filtered;
    const std::vector<std::size_t> candidates = idx.find_candidates(query_center, broad_radius);
    filtered.reserve(candidates.size());
    for (const std::size_t index : candidates) {
        const double dist = (query_center - idx.cloud.pts[index]).norm();
        if (dist <= query_radius + idx.cell_radius[index] + padding) {
            filtered.push_back(index);
        }
    }
    return filtered;
}

}

void check_moab(const moab::ErrorCode code, const std::string& message)
{
    if (code != moab::MB_SUCCESS) {
        throw std::runtime_error(message + " (MOAB error " + std::to_string(static_cast<int>(code)) + ")");
    }
}

void eval_harmonic_basis(int k, const Eigen::Vector2d& p,
                         double& P, double& Q,
                         Eigen::Vector2d& gradP, Eigen::Vector2d& gradQ)
{
    std::complex<double> z(p.x(), p.y());
    std::complex<double> zk = std::pow(z, k);
    P = zk.real();
    Q = zk.imag();
    std::complex<double> zk1 = (k == 1) ? std::complex<double>(1.0, 0.0) : std::pow(z, k - 1);
    gradP = Eigen::Vector2d(k * zk1.real(), -k * zk1.imag());
    gradQ = Eigen::Vector2d(k * zk1.imag(),  k * zk1.real());
}

namespace {

Eigen::Vector3d normalized_or_throw(const Eigen::Vector3d& v, const std::string& message)
{
    const double n = v.norm();
    if (n < kTolerance) {
        throw std::runtime_error(message);
    }
    return v / n;
}

GnomonicFrame make_gnomonic_frame(const std::vector<Eigen::Vector3d>& points, const double radius)
{
    Eigen::Vector3d center(0.0, 0.0, 0.0);
    for (const Eigen::Vector3d& p : points) {
        center += p.normalized();
    }
    center = normalized_or_throw(center, "Degenerate spherical polygon centroid");

    Eigen::Vector3d reference(0.0, 0.0, 1.0);
    if (std::abs(center.dot(reference)) > 0.9) {
        reference = Eigen::Vector3d(1.0, 0.0, 0.0);
    }

    GnomonicFrame frame;
    frame.center = center;
    frame.e_y = normalized_or_throw(center.cross(reference), "Failed to build spherical tangent basis");
    frame.e_x = normalized_or_throw(frame.e_y.cross(center), "Failed to build spherical tangent basis");
    frame.radius = radius;
    return frame;
}

double spherical_triangle_area(const Eigen::Vector3d& a, const Eigen::Vector3d& b, const Eigen::Vector3d& c)
{
    const Eigen::Vector3d au = a.normalized();
    const Eigen::Vector3d bu = b.normalized();
    const Eigen::Vector3d cu = c.normalized();
    const double det = std::abs(au.dot(bu.cross(cu)));
    const double denom = 1.0 + au.dot(bu) + bu.dot(cu) + cu.dot(au);
    return 2.0 * std::atan2(det, denom);
}

std::vector<Eigen::Vector2d> absolute_points(const LocalPolygon& polygon)
{
    std::vector<Eigen::Vector2d> points;
    points.reserve(polygon.points.size());
    for (const Eigen::Vector2d& p : polygon.points) {
        points.push_back(p + polygon.centroid);
    }
    return points;
}

bool point_in_convex_polygon(const Eigen::Vector2d& point,
                             const std::vector<Eigen::Vector2d>& polygon,
                             const double tolerance)
{
    if (polygon.size() < 3) {
        return false;
    }

    std::vector<Eigen::Vector2d> clip = polygon;
    if (signed_area(clip) < 0.0) {
        std::reverse(clip.begin(), clip.end());
    }

    for (std::size_t i = 0; i < clip.size(); ++i) {
        const Eigen::Vector2d a = clip[i];
        const Eigen::Vector2d b = clip[(i + 1) % clip.size()];
        const Eigen::Vector2d edge = b - a;
        const Eigen::Vector2d inward_normal(-edge.y(), edge.x());
        if (inward_normal.dot(point - a) < -tolerance) {
            return false;
        }
    }
    return true;
}

bool target_interior_side_intersects_source(const Eigen::Vector2d& clipped_a,
                                            const Eigen::Vector2d& clipped_b,
                                            const std::vector<Eigen::Vector2d>& source_polygon,
                                            const double tolerance)
{
    const Eigen::Vector2d delta = clipped_b - clipped_a;
    const double length = delta.norm();
    if (length <= tolerance) {
        return false;
    }

    double scale = length;
    for (std::size_t i = 0; i < source_polygon.size(); ++i) {
        scale = std::max(scale, (source_polygon[(i + 1) % source_polygon.size()] - source_polygon[i]).norm());
    }

    const Eigen::Vector2d midpoint = 0.5 * (clipped_a + clipped_b);
    const double strict_tolerance = 1.0e-11 * (1.0 + scale);
    if (point_in_convex_polygon(midpoint, source_polygon, -strict_tolerance)) {
        return true;
    }

    const Eigen::Vector2d target_left_normal(-delta.y(), delta.x());
    const Eigen::Vector2d probe =
        midpoint + (strict_tolerance / length) * target_left_normal;

    return point_in_convex_polygon(probe, source_polygon, 0.0);
}

std::vector<Eigen::Vector2d> clip_by_halfplane(const std::vector<Eigen::Vector2d>& polygon,
                                               const Eigen::Vector2d& normal,
                                               const double offset,
                                               const double tolerance)
{
    std::vector<Eigen::Vector2d> output;
    if (polygon.empty()) {
        return output;
    }

    auto inside = [&](const Eigen::Vector2d& p) { return normal.dot(p) <= offset + tolerance; };
    auto intersect = [&](const Eigen::Vector2d& a, const Eigen::Vector2d& b) -> Eigen::Vector2d {
        const double da = normal.dot(a) - offset;
        const double db = normal.dot(b) - offset;
        const double denom = da - db;
        if (std::abs(denom) <= tolerance) {
            return a;
        }
        return a + (da / denom) * (b - a);
    };

    Eigen::Vector2d previous = polygon.back();
    bool previous_inside = inside(previous);
    for (const Eigen::Vector2d& current : polygon) {
        const bool current_inside = inside(current);
        if (current_inside) {
            if (!previous_inside) {
                output.push_back(intersect(previous, current));
            }
            output.push_back(current);
        } else if (previous_inside) {
            output.push_back(intersect(previous, current));
        }
        previous = current;
        previous_inside = current_inside;
    }
    return output;
}

std::vector<Eigen::Vector2d> convex_polygon_intersection(std::vector<Eigen::Vector2d> subject,
                                                         std::vector<Eigen::Vector2d> clipper,
                                                         const double tolerance)
{
    if (subject.size() < 3 || clipper.size() < 3) {
        return {};
    }

    if (signed_area(subject) < 0.0) {
        std::reverse(subject.begin(), subject.end());
    }
    if (signed_area(clipper) < 0.0) {
        std::reverse(clipper.begin(), clipper.end());
    }

    for (std::size_t i = 0; i < clipper.size(); ++i) {
        const Eigen::Vector2d a = clipper[i];
        const Eigen::Vector2d b = clipper[(i + 1) % clipper.size()];
        const Eigen::Vector2d edge = b - a;
        const Eigen::Vector2d outward(edge.y(), -edge.x());
        subject = clip_by_halfplane(subject, outward, outward.dot(a), tolerance);
        if (subject.size() < 3 || std::abs(signed_area(subject)) <= tolerance) {
            return {};
        }
    }

    return subject;
}

template <typename Func>
Eigen::Vector3d integrate_triangle_vector(const Eigen::Vector2d& a,
                                          const Eigen::Vector2d& b,
                                          const Eigen::Vector2d& c,
                                          const Func& func)
{
    return Eigen::Vector3d(
        integrate_triangle_scalar(a, b, c, [&](const Eigen::Vector2d& p) { return func(p).x(); }),
        integrate_triangle_scalar(a, b, c, [&](const Eigen::Vector2d& p) { return func(p).y(); }),
        integrate_triangle_scalar(a, b, c, [&](const Eigen::Vector2d& p) { return func(p).z(); }));
}

template <typename Func>
Eigen::Vector3d integrate_polygon_vector(std::vector<Eigen::Vector2d> polygon, const Func& func)
{
    if (polygon.size() < 3) {
        return Eigen::Vector3d::Zero();
    }

    if (signed_area(polygon) < 0.0) {
        std::reverse(polygon.begin(), polygon.end());
    }

    const Eigen::Vector2d center = polygon_centroid(polygon);
    Eigen::Vector3d integral = Eigen::Vector3d::Zero();
    for (std::size_t i = 0; i < polygon.size(); ++i) {
        integral += integrate_triangle_vector(center, polygon[i], polygon[(i + 1) % polygon.size()], func);
    }
    return integral;
}

double polygon_area_abs(const std::vector<Eigen::Vector2d>& polygon)
{
    return polygon.size() < 3 ? 0.0 : std::abs(signed_area(polygon));
}

Eigen::Vector2d harmonic_velocity_value(const ReconstructionCoeffs& coeffs, const Eigen::Vector2d& p)
{
    Eigen::Vector2d v = 0.5 * coeffs.d * p;
    const int n_h = static_cast<int>(coeffs.harmonic.size());
    for (int i = 0; i < n_h; ++i) {
        const int k = (i / 2) + 1;
        const bool is_q = (i % 2 == 1);
        double P, Q;
        Eigen::Vector2d gP, gQ;
        eval_harmonic_basis(k, p, P, Q, gP, gQ);
        v += coeffs.harmonic[i] * (is_q ? gQ : gP);
    }
    return v;
}

ReconstructionCoeffs read_coefficients(moab::Core& mb,
                                       const moab::Tag coeff_tag,
                                       const moab::EntityHandle polygon)
{
    const void* ptr = nullptr;
    int size = 0;
    check_moab(mb.tag_get_by_ptr(coeff_tag, &polygon, 1, &ptr, &size),
               "Failed to read reconstruction coefficients");
    const double* data = static_cast<const double*>(ptr);
    ReconstructionCoeffs coeffs;
    coeffs.d = data[0];
    coeffs.harmonic.assign(data + 1, data + size);
    return coeffs;
}

Eigen::Vector3d reconstruction_integral(const GeometryOptions& options,
                                        const LocalPolygon& source,
                                        const ReconstructionCoeffs& coeffs,
                                        const std::vector<Eigen::Vector2d>& polygon_abs)
{
    if (polygon_abs.size() < 3 || polygon_area_abs(polygon_abs) <= options.geometry_tolerance) {
        return Eigen::Vector3d::Zero();
    }

    if (options.mode == GeometryMode::SphericalGnomonic) {
        const GnomonicFrame frame{source.n, source.e_x, source.e_y, options.radius};
        // Use adaptive integration for each component to handle thin fan triangles
        // on elongated spherical cells (e.g., near cubed-sphere corners).
        auto field = [&](const Eigen::Vector2d& xi) {
            const Eigen::Vector2d chart_velocity = harmonic_velocity_value(coeffs, xi - source.centroid);
            const Eigen::Vector3d surface_velocity = lift_contravariant_piola(chart_velocity, xi, frame);
            return gnomonic_area_scale(xi, frame) * surface_velocity;
        };
        return Eigen::Vector3d(
            integrate_polygon_adaptive(polygon_abs, [&](const Eigen::Vector2d& p) { return field(p).x(); }),
            integrate_polygon_adaptive(polygon_abs, [&](const Eigen::Vector2d& p) { return field(p).y(); }),
            integrate_polygon_adaptive(polygon_abs, [&](const Eigen::Vector2d& p) { return field(p).z(); }));
    }

    return integrate_polygon_vector(polygon_abs, [&](const Eigen::Vector2d& p) {
        const Eigen::Vector2d planar_velocity = harmonic_velocity_value(coeffs, p - source.centroid);
        return Eigen::Vector3d(planar_velocity.x(), planar_velocity.y(), 0.0);
    });
}

Eigen::Matrix2d gnomonic_hodge_metric(const Eigen::Vector2d& xi, const GnomonicFrame& frame)
{
    const Eigen::Matrix<double, 3, 2> jacobian = gnomonic_jacobian(xi, frame);
    const double area_scale = jacobian.col(0).cross(jacobian.col(1)).norm();
    if (area_scale < kTolerance) {
        throw std::runtime_error("Degenerate Hodge metric in gnomonic chart");
    }
    return (jacobian.transpose() * jacobian) / area_scale;
}

struct GaussLegendrePoint {
    double x = 0.0;
    double w = 0.0;
};

static const GaussLegendrePoint gauss4_rule[4] = {
    {-0.8611363115940526, 0.3478548451374538},
    {-0.3399810435848563, 0.6521451548625461},
    { 0.3399810435848563, 0.6521451548625461},
    { 0.8611363115940526, 0.3478548451374538},
};

static const GaussLegendrePoint gauss10_rule[10] = {
    {-0.9739065285171717, 0.0666713443086881},
    {-0.8650633666889845, 0.1494513491505806},
    {-0.6794095682990244, 0.2190863625159820},
    {-0.4333953941292472, 0.2692667193099963},
    {-0.1488743389816312, 0.2955242247147529},
    { 0.1488743389816312, 0.2955242247147529},
    { 0.4333953941292472, 0.2692667193099963},
    { 0.6794095682990244, 0.2190863625159820},
    { 0.8650633666889845, 0.1494513491505806},
    { 0.9739065285171717, 0.0666713443086881},
};

std::vector<GaussLegendrePoint> gauss_legendre_rule(const int quadrature_points)
{
    if (quadrature_points <= 4) {
        return std::vector<GaussLegendrePoint>(gauss4_rule, gauss4_rule + 4);
    }
    return std::vector<GaussLegendrePoint>(gauss10_rule, gauss10_rule + 10);
}

template <typename Func>
double integrate_triangle_duffy(const Eigen::Vector2d& a,
                                const Eigen::Vector2d& b,
                                const Eigen::Vector2d& c,
                                const std::vector<GaussLegendrePoint>& rule,
                                const Func& func)
{
    const Eigen::Vector2d ab = b - a;
    const Eigen::Vector2d bc = c - b;
    const double det_j = std::abs(ab.x() * bc.y() - ab.y() * bc.x());
    double sum = 0.0;

    for (const GaussLegendrePoint& qu : rule) {
        const double u = 0.5 * (qu.x + 1.0);
        const double wu = 0.5 * qu.w;
        for (const GaussLegendrePoint& qv : rule) {
            const double v = 0.5 * (qv.x + 1.0);
            const double wv = 0.5 * qv.w;
            const Eigen::Vector2d p = a + u * ab + (u * v) * bc;
            sum += wu * wv * u * func(p);
        }
    }

    return det_j * sum;
}

Eigen::MatrixXd source_reconstruction_matrix(const LocalPolygon& poly,
                                             const std::vector<LocalEdge>& edges,
                                             const GeometryOptions& options)
{
    const int N = static_cast<int>(edges.size());
    const int K_max = N / 2;
    const int N_h = 2 * K_max;
    const int S = N_h + N - 1;
    const bool use_surface_metric = options.metric_weighted && options.mode == GeometryMode::SphericalGnomonic;
    const GnomonicFrame frame{poly.n, poly.e_x, poly.e_y, options.radius};
    const double stabilization_area =
        (use_surface_metric && poly.spherical_area > options.geometry_tolerance) ? poly.spherical_area : poly.area;

    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(N_h, N_h);
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(N_h, N);
    const Eigen::Vector2d origin(0.0, 0.0);
    const std::vector<GaussLegendrePoint> duffy_rule = gauss_legendre_rule(10);

    // Helper: use Duffy quadrature for metric-weighted integrals (rational integrands),
    // standard 7-point symmetric rule for polynomial integrands.
    auto integrate_cell = [&](const Eigen::Vector2d& a, const Eigen::Vector2d& b,
                              const Eigen::Vector2d& c, auto&& func) {
        if (use_surface_metric) {
            return integrate_triangle_duffy(a, b, c, duffy_rule, func);
        }
        return integrate_triangle_scalar(a, b, c, func);
    };

    for (int i = 0; i < N_h; ++i) {
        int ki = (i / 2) + 1;
        bool is_Q_i = (i % 2 == 1);
        for (int j = 0; j < N_h; ++j) {
            int kj = (j / 2) + 1;
            bool is_Q_j = (j % 2 == 1);

            double val = 0.0;
            for (const LocalEdge& edge : edges) {
                val += integrate_cell(origin, edge.a, edge.b, [&](const Eigen::Vector2d& p) {
                    double Pi, Qi, Pj, Qj;
                    Eigen::Vector2d gPi, gQi, gPj, gQj;
                    eval_harmonic_basis(ki, p, Pi, Qi, gPi, gQi);
                    eval_harmonic_basis(kj, p, Pj, Qj, gPj, gQj);
                    Eigen::Vector2d gi = is_Q_i ? gQi : gPi;
                    Eigen::Vector2d gj = is_Q_j ? gQj : gPj;
                    if (!use_surface_metric) {
                        return gi.dot(gj);
                    }
                    const Eigen::Matrix2d hodge = gnomonic_hodge_metric(p + poly.centroid, frame);
                    return gi.dot(hodge * gj);
                });
            }
            V(i, j) = val;
        }

        double cell_basis_integral = 0.0;
        double div_integral = 0.0;
        for (const LocalEdge& edge : edges) {
            cell_basis_integral += integrate_cell(origin, edge.a, edge.b, [&](const Eigen::Vector2d& p) {
                double P, Q; Eigen::Vector2d gP, gQ;
                eval_harmonic_basis(ki, p, P, Q, gP, gQ);
                return is_Q_i ? Q : P;
            });
            div_integral += integrate_cell(origin, edge.a, edge.b, [&](const Eigen::Vector2d& p) {
                double P, Q; Eigen::Vector2d gP, gQ;
                eval_harmonic_basis(ki, p, P, Q, gP, gQ);
                return p.dot(is_Q_i ? gQ : gP);
            });
        }
        const double cell_basis_average = cell_basis_integral / poly.area;
        if (i >= 4) {
            const double v_diag = std::max(V(i, i), kTolerance * stabilization_area);
            V(i, i) += 1.0e1 * v_diag;
        }

        for (int e = 0; e < N; ++e) {
            const double edge_average = integrate_edge_scalar(edges[e].a, edges[e].b, [&](const Eigen::Vector2d& p) {
                double P, Q; Eigen::Vector2d gP, gQ;
                eval_harmonic_basis(ki, p, P, Q, gP, gQ);
                return is_Q_i ? Q : P;
            }) / edges[e].length;
            M(i, e) = edge_average - cell_basis_average - 0.5 * div_integral / poly.area;
        }
    }

    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(N - 1, N_h);
    for (int e = 0; e < N - 1; ++e) {
        for (int i = 0; i < N_h; ++i) {
            int ki = (i / 2) + 1;
            bool is_Q_i = (i % 2 == 1);
            C(e, i) = integrate_edge_scalar(edges[e].a, edges[e].b, [&](const Eigen::Vector2d& p) {
                double P, Q;
                Eigen::Vector2d gP, gQ;
                eval_harmonic_basis(ki, p, P, Q, gP, gQ);
                Eigen::Vector2d g = is_Q_i ? gQ : gP;
                return g.dot(edges[e].outward_normal);
            });
        }
    }

    Eigen::MatrixXd F = Eigen::MatrixXd::Zero(N - 1, N);
    for (int e = 0; e < N - 1; ++e) {
        double Ee = integrate_edge_scalar(edges[e].a, edges[e].b, [&](const Eigen::Vector2d& p) {
            return p.dot(edges[e].outward_normal);
        });
        for (int j = 0; j < N; ++j) {
            F(e, j) = (e == j ? 1.0 : 0.0) - Ee / (2.0 * poly.area);
        }
    }

    Eigen::MatrixXd KKT = Eigen::MatrixXd::Zero(S, S);
    KKT.block(0, 0, N_h, N_h) = V;
    KKT.block(N_h, 0, N - 1, N_h) = C;
    KKT.block(0, N_h, N_h, N - 1) = C.transpose();

    Eigen::MatrixXd RHS = Eigen::MatrixXd::Zero(S, N);
    RHS.block(0, 0, N_h, N) = M;
    RHS.block(N_h, 0, N - 1, N) = F;

    Eigen::FullPivLU<Eigen::MatrixXd> lu(KKT);
    Eigen::MatrixXd X = lu.solve(RHS).block(0, 0, N_h, N);

    Eigen::MatrixXd reconstruction = Eigen::MatrixXd::Zero(1 + N_h, N);
    reconstruction.row(0).setConstant(1.0 / poly.area);
    reconstruction.block(1, 0, N_h, N) = X;

    return reconstruction;
}

struct VectorBasisTerm {
    int component = 0;
    int a = 0;
    int b = 0;
};

struct EdgeMomentSample {
    double value = 0.0;
    double t = 0.0;
};

struct MonomialScratch {
    std::vector<double> x_powers;
    std::vector<double> y_powers;
    std::vector<double> legendre;
};

MonomialScratch& monomial_scratch()
{
    static thread_local MonomialScratch scratch;
    return scratch;
}

double legendre_polynomial(const int degree, const double x)
{
    if (degree == 0) {
        return 1.0;
    }
    if (degree == 1) {
        return x;
    }
    double p_nm2 = 1.0;
    double p_nm1 = x;
    double p_n = x;
    for (int n = 2; n <= degree; ++n) {
        p_n = ((2.0 * n - 1.0) * x * p_nm1 - (n - 1.0) * p_nm2) / static_cast<double>(n);
        p_nm2 = p_nm1;
        p_nm1 = p_n;
    }
    return p_n;
}

template <typename Func>
double integrate_edge_highorder_rule(const Eigen::Vector2d& a,
                                     const Eigen::Vector2d& b,
                                     const std::vector<GaussLegendrePoint>& rule,
                                     const Func& func)
{
    const double length = (b - a).norm();
    const Eigen::Vector2d mid = 0.5 * (a + b);
    const Eigen::Vector2d half_delta = 0.5 * (b - a);
    double sum = 0.0;
    for (const GaussLegendrePoint& q : rule) {
        sum += q.w * func(mid + q.x * half_delta);
    }
    return 0.5 * length * sum;
}

template <typename Func>
double integrate_polygon_scalar_duffy(const std::vector<LocalEdge>& edges,
                                      const std::vector<GaussLegendrePoint>& rule,
                                      const Func& func)
{
    const Eigen::Vector2d origin = Eigen::Vector2d::Zero();
    double integral = 0.0;
    for (const LocalEdge& edge : edges) {
        integral += integrate_triangle_duffy(origin, edge.a, edge.b, rule, func);
    }
    return integral;
}

int scalar_monomial_basis_count(const int degree)
{
    return (degree + 1) * (degree + 2) / 2;
}

int vector_polynomial_basis_count(const int degree)
{
    return 2 * scalar_monomial_basis_count(degree);
}

int vector_moment_basis_count(const int degree)
{
    return scalar_monomial_basis_count(degree);
}

int scalar_monomial_index(const int degree, const int a, const int b)
{
    if (degree < 0 || a < 0 || b < 0 || a + b > degree) {
        return -1;
    }

    int index = 0;
    for (int total_degree = 0; total_degree <= degree; ++total_degree) {
        for (int ax = total_degree; ax >= 0; --ax, ++index) {
            const int by = total_degree - ax;
            if (ax == a && by == b) {
                return index;
            }
        }
    }
    return -1;
}

std::vector<VectorBasisTerm> build_vector_polynomial_basis(const int degree)
{
    std::vector<VectorBasisTerm> basis;
    basis.reserve(static_cast<std::size_t>(vector_polynomial_basis_count(degree)));
    for (int component = 0; component < 2; ++component) {
        for (int total_degree = 0; total_degree <= degree; ++total_degree) {
            for (int a = total_degree; a >= 0; --a) {
                VectorBasisTerm term;
                term.component = component;
                term.a = a;
                term.b = total_degree - a;
                basis.push_back(term);
            }
        }
    }
    return basis;
}

const std::vector<VectorBasisTerm>& cached_vector_polynomial_basis(const int degree)
{
    static thread_local std::vector<std::vector<VectorBasisTerm>> cache;
    if (degree < 0) {
        throw std::runtime_error("Negative vector polynomial degree");
    }
    if (cache.size() <= static_cast<std::size_t>(degree)) {
        cache.resize(static_cast<std::size_t>(degree + 1));
    }
    std::vector<VectorBasisTerm>& basis = cache[static_cast<std::size_t>(degree)];
    if (basis.empty()) {
        basis = build_vector_polynomial_basis(degree);
    }
    return basis;
}

int vector_basis_index(const std::vector<VectorBasisTerm>& basis,
                       const int component,
                       const int a,
                       const int b)
{
    for (std::size_t i = 0; i < basis.size(); ++i) {
        if (basis[i].component == component && basis[i].a == a && basis[i].b == b) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

double binomial_coefficient(const int n, const int k)
{
    if (k < 0 || k > n) {
        return 0.0;
    }
    if (k == 0 || k == n) {
        return 1.0;
    }
    double value = 1.0;
    for (int i = 1; i <= k; ++i) {
        value *= static_cast<double>(n - (k - i));
        value /= static_cast<double>(i);
    }
    return value;
}

Eigen::MatrixXd orthonormal_column_basis(const Eigen::MatrixXd& matrix, const double tolerance)
{
    if (matrix.cols() == 0) {
        return Eigen::MatrixXd::Zero(matrix.rows(), 0);
    }

    const Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::Index rank = 0;
    for (Eigen::Index i = 0; i < svd.singularValues().size(); ++i) {
        if (svd.singularValues()(i) > tolerance) {
            ++rank;
        }
    }
    return svd.matrixU().leftCols(rank);
}

Eigen::MatrixXd nullspace_basis(const Eigen::MatrixXd& matrix, const double tolerance)
{
    if (matrix.rows() == 0) {
        return Eigen::MatrixXd::Identity(matrix.cols(), matrix.cols());
    }

    const Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix, Eigen::ComputeFullV);
    Eigen::Index rank = 0;
    for (Eigen::Index i = 0; i < svd.singularValues().size(); ++i) {
        if (svd.singularValues()(i) > tolerance) {
            ++rank;
        }
    }
    return svd.matrixV().rightCols(matrix.cols() - rank);
}

/// Orthonormalize mode vectors with respect to a metric Gram matrix G.
///
/// Given a matrix M whose columns are mode vectors in scaled raw-monomial
/// coordinates, compute Q such that Q^T G Q = I, where Q spans the same
/// column space as M (up to rank tolerance).
///
/// When G is the identity, this is equivalent to orthonormal_column_basis().
Eigen::MatrixXd metric_orthonormal_column_basis(const Eigen::MatrixXd& modes,
                                                const Eigen::MatrixXd& G,
                                                const double tolerance)
{
    if (modes.cols() == 0) {
        return Eigen::MatrixXd::Zero(modes.rows(), 0);
    }

    // Block Gram matrix in the mode subspace: Gb = M^T G M
    const Eigen::MatrixXd Gb = modes.transpose() * G * modes;

    // Eigendecompose the SPD block Gram matrix
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(Gb);
    if (eig.info() != Eigen::Success) {
        throw std::runtime_error("Eigendecomposition failed in metric_orthonormal_column_basis");
    }

    // Threshold eigenvalues to determine rank
    const double max_eigenvalue = eig.eigenvalues().maxCoeff();
    const double threshold = tolerance * std::max(max_eigenvalue, 1.0e-30);
    Eigen::Index rank = 0;
    for (Eigen::Index i = 0; i < eig.eigenvalues().size(); ++i) {
        if (eig.eigenvalues()(i) > threshold) {
            ++rank;
        }
    }

    if (rank == 0) {
        return Eigen::MatrixXd::Zero(modes.rows(), 0);
    }

    // Build metric-orthonormal basis: Q = M * V_kept * D_kept^{-1/2}
    // Eigenvalues are sorted ascending, so kept columns are the last 'rank' ones.
    const Eigen::Index skip = eig.eigenvalues().size() - rank;
    const Eigen::MatrixXd V_kept = eig.eigenvectors().rightCols(rank);
    Eigen::VectorXd D_inv_sqrt(rank);
    for (Eigen::Index i = 0; i < rank; ++i) {
        D_inv_sqrt(i) = 1.0 / std::sqrt(eig.eigenvalues()(skip + i));
    }

    return modes * V_kept * D_inv_sqrt.asDiagonal();
}

struct SplitMomentBasis {
    Eigen::MatrixXd raw_coordinates;
    int divergence_mode_count = 0;
    int harmonic_mode_count = 0;
    int bubble_mode_count = 0;
};

Eigen::VectorXd divergence_particular_mode_physical(const std::vector<VectorBasisTerm>& raw_basis,
                                                    const int a,
                                                    const int b)
{
    Eigen::VectorXd mode = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(raw_basis.size()));
    const int index = vector_basis_index(raw_basis, 0, a + 1, b);
    if (index < 0) {
        throw std::runtime_error("Failed to build divergence particular mode");
    }
    mode(index) = 1.0 / static_cast<double>(a + 1);
    return mode;
}

Eigen::VectorXd harmonic_gradient_mode_physical(const std::vector<VectorBasisTerm>& raw_basis,
                                                const int harmonic_degree,
                                                const bool imaginary)
{
    Eigen::VectorXd mode = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(raw_basis.size()));
    for (int y_degree = 0; y_degree <= harmonic_degree; ++y_degree) {
        const int x_degree = harmonic_degree - y_degree;
        const bool contributes_to_real = (y_degree % 2 == 0);
        if (imaginary == contributes_to_real) {
            continue;
        }

        const double sign = contributes_to_real
            ? ((y_degree / 2) % 2 == 0 ? 1.0 : -1.0)
            : ((((y_degree - 1) / 2) % 2 == 0) ? 1.0 : -1.0);
        const double coeff = sign * binomial_coefficient(harmonic_degree, y_degree);

        if (x_degree > 0) {
            const int index_x = vector_basis_index(raw_basis, 0, x_degree - 1, y_degree);
            if (index_x >= 0) {
                mode(index_x) += coeff * static_cast<double>(x_degree);
            }
        }
        if (y_degree > 0) {
            const int index_y = vector_basis_index(raw_basis, 1, x_degree, y_degree - 1);
            if (index_y >= 0) {
                mode(index_y) += coeff * static_cast<double>(y_degree);
            }
        }
    }
    return mode;
}

Eigen::MatrixXd divergence_operator_physical(const int degree,
                                             const std::vector<VectorBasisTerm>& raw_basis)
{
    if (degree <= 0) {
        return Eigen::MatrixXd::Zero(0, static_cast<Eigen::Index>(raw_basis.size()));
    }

    const int scalar_dim = scalar_monomial_basis_count(degree - 1);
    Eigen::MatrixXd divergence = Eigen::MatrixXd::Zero(scalar_dim, static_cast<Eigen::Index>(raw_basis.size()));
    for (std::size_t col = 0; col < raw_basis.size(); ++col) {
        const VectorBasisTerm& term = raw_basis[col];
        if (term.component == 0 && term.a > 0) {
            const int row = scalar_monomial_index(degree - 1, term.a - 1, term.b);
            divergence(row, static_cast<Eigen::Index>(col)) += static_cast<double>(term.a);
        } else if (term.component == 1 && term.b > 0) {
            const int row = scalar_monomial_index(degree - 1, term.a, term.b - 1);
            divergence(row, static_cast<Eigen::Index>(col)) += static_cast<double>(term.b);
        }
    }
    return divergence;
}

/// Apply per-row polynomial scaling to mode vectors so they match the
/// scaled-coordinate convention used by vector_basis_value() and G_raw.
void apply_mode_scaling(Eigen::MatrixXd& modes,
                        const std::vector<VectorBasisTerm>& raw_basis,
                        const double scale)
{
    for (std::size_t row = 0; row < raw_basis.size(); ++row) {
        const int total_degree = raw_basis[row].a + raw_basis[row].b;
        if (total_degree > 0) {
            modes.row(static_cast<Eigen::Index>(row)) *= std::pow(scale, total_degree);
        }
    }
}

SplitMomentBasis build_split_moment_basis(const int degree,
                                          const int harmonic_degree_option,
                                          const std::vector<VectorBasisTerm>& raw_basis,
                                          const double scale,
                                          const Eigen::MatrixXd& G_raw)
{
    const double tolerance = 1.0e-12;
    const Eigen::Index raw_dim = static_cast<Eigen::Index>(raw_basis.size());

    // Stage 1: Divergence-particular modes.
    Eigen::MatrixXd divergence_modes = Eigen::MatrixXd::Zero(raw_dim,
                                                             std::max(0, scalar_monomial_basis_count(degree - 1)));
    int divergence_col = 0;
    for (int total_degree = 0; total_degree <= degree - 1; ++total_degree) {
        for (int a = total_degree; a >= 0; --a, ++divergence_col) {
            const int b = total_degree - a;
            divergence_modes.col(divergence_col) = divergence_particular_mode_physical(raw_basis, a, b);
        }
    }
    apply_mode_scaling(divergence_modes, raw_basis, scale);
    const Eigen::MatrixXd divergence_block = metric_orthonormal_column_basis(divergence_modes, G_raw, tolerance);

    // Stage 2: Harmonic-gradient modes.
    // Project out divergence_block first to ensure mutual metric-orthogonality.
    const int harmonic_degree = (harmonic_degree_option > 0)
        ? std::min(harmonic_degree_option, degree + 1)
        : (degree + 1);
    Eigen::MatrixXd harmonic_modes = Eigen::MatrixXd::Zero(raw_dim, 2 * harmonic_degree);
    int harmonic_col = 0;
    for (int k = 1; k <= harmonic_degree; ++k) {
        harmonic_modes.col(harmonic_col++) = harmonic_gradient_mode_physical(raw_basis, k, false);
        harmonic_modes.col(harmonic_col++) = harmonic_gradient_mode_physical(raw_basis, k, true);
    }
    apply_mode_scaling(harmonic_modes, raw_basis, scale);
    if (divergence_block.cols() > 0 && harmonic_modes.cols() > 0) {
        harmonic_modes -= divergence_block * (divergence_block.transpose() * G_raw * harmonic_modes);
    }
    const Eigen::MatrixXd harmonic_block = metric_orthonormal_column_basis(harmonic_modes, G_raw, tolerance);

    // Stage 3: Divergence-free completion (bubble) modes.
    // The divergence operator is algebraic -- its nullspace is metric-independent.
    const Eigen::MatrixXd divergence_operator = divergence_operator_physical(degree, raw_basis);
    Eigen::MatrixXd bubble_candidates = nullspace_basis(divergence_operator, tolerance);
    apply_mode_scaling(bubble_candidates, raw_basis, scale);
    // Project out both divergence and harmonic subspaces using the metric inner product.
    if (divergence_block.cols() > 0 && bubble_candidates.cols() > 0) {
        bubble_candidates -= divergence_block * (divergence_block.transpose() * G_raw * bubble_candidates);
    }
    if (harmonic_block.cols() > 0 && bubble_candidates.cols() > 0) {
        bubble_candidates -= harmonic_block * (harmonic_block.transpose() * G_raw * bubble_candidates);
    }
    const Eigen::MatrixXd bubble_block = metric_orthonormal_column_basis(bubble_candidates, G_raw, tolerance);

    // Assemble the full basis.
    Eigen::MatrixXd basis = Eigen::MatrixXd::Zero(raw_dim,
                                                  divergence_block.cols() + harmonic_block.cols() + bubble_block.cols());
    basis.block(0, 0, raw_dim, divergence_block.cols()) = divergence_block;
    basis.block(0, divergence_block.cols(), raw_dim, harmonic_block.cols()) = harmonic_block;
    basis.block(0, divergence_block.cols() + harmonic_block.cols(), raw_dim, bubble_block.cols()) = bubble_block;

    const Eigen::Index basis_dim = basis.cols();

    // Verify metric orthonormality: Q^T G Q should be close to identity.
    const Eigen::MatrixXd QGQ = basis.transpose() * G_raw * basis;
    const double orthogonality_error = (QGQ - Eigen::MatrixXd::Identity(basis_dim, basis_dim)).norm();
    if (orthogonality_error > 1.0e-3) {
        std::ostringstream oss;
        oss << "Metric-orthonormal basis failed verification"
            << " ||Q^T G Q - I|| = " << orthogonality_error;
        throw std::runtime_error(oss.str());
    }

    SplitMomentBasis split;
    split.raw_coordinates = basis;
    split.divergence_mode_count = static_cast<int>(divergence_block.cols());
    split.harmonic_mode_count = static_cast<int>(harmonic_block.cols());
    split.bubble_mode_count = static_cast<int>(bubble_block.cols());
    return split;
}

double local_length_scale(const LocalPolygon& poly)
{
    double scale = std::sqrt(std::max(poly.area, kTolerance));
    for (const Eigen::Vector2d& p : poly.points) {
        scale = std::max(scale, p.norm());
    }
    return std::max(scale, 1.0e-12);
}

double monomial_value(const int a, const int b, const Eigen::Vector2d& p, const double scale)
{
    return std::pow(p.x() / scale, a) * std::pow(p.y() / scale, b);
}

Eigen::Vector2d vector_basis_value(const VectorBasisTerm& term,
                                   const Eigen::Vector2d& p,
                                   const double scale)
{
    const double value = monomial_value(term.a, term.b, p, scale);
    if (term.component == 0) {
        return Eigen::Vector2d(value, 0.0);
    }
    return Eigen::Vector2d(0.0, value);
}

template <typename Func>
void accumulate_edge_moment_bundle(const Eigen::Vector2d& a,
                                   const Eigen::Vector2d& b,
                                   const std::vector<GaussLegendrePoint>& rule,
                                   const int order,
                                   std::vector<double>& moments,
                                   const Func& sample_func)
{
    moments.assign(static_cast<std::size_t>(order + 1), 0.0);
    const double length = (b - a).norm();
    if (length <= kTolerance) {
        return;
    }

    MonomialScratch& scratch = monomial_scratch();
    scratch.legendre.resize(static_cast<std::size_t>(order + 1), 0.0);

    const Eigen::Vector2d mid = 0.5 * (a + b);
    const Eigen::Vector2d half_delta = 0.5 * (b - a);
    for (const GaussLegendrePoint& q : rule) {
        const EdgeMomentSample sample = sample_func(mid + q.x * half_delta);
        scratch.legendre[0] = 1.0;
        if (order >= 1) {
            scratch.legendre[1] = sample.t;
        }
        for (int degree = 2; degree <= order; ++degree) {
            scratch.legendre[static_cast<std::size_t>(degree)] =
                ((2.0 * degree - 1.0) * sample.t * scratch.legendre[static_cast<std::size_t>(degree - 1)] -
                 (degree - 1.0) * scratch.legendre[static_cast<std::size_t>(degree - 2)]) /
                static_cast<double>(degree);
        }
        for (int degree = 0; degree <= order; ++degree) {
            moments[static_cast<std::size_t>(degree)] +=
                q.w * sample.value * scratch.legendre[static_cast<std::size_t>(degree)];
        }
    }

    const double scale = 0.5 * length;
    for (double& value : moments) {
        value *= scale;
    }
}

void accumulate_vector_cell_moments(const Eigen::Vector2d& velocity,
                                    const Eigen::Vector2d& local_point,
                                    const int degree,
                                    const double weight,
                                    std::vector<Eigen::Vector2d>& moments)
{
    if (degree < 0) {
        return;
    }

    MonomialScratch& scratch = monomial_scratch();
    scratch.x_powers.resize(static_cast<std::size_t>(degree + 1), 1.0);
    scratch.y_powers.resize(static_cast<std::size_t>(degree + 1), 1.0);
    for (int i = 1; i <= degree; ++i) {
        scratch.x_powers[static_cast<std::size_t>(i)] =
            scratch.x_powers[static_cast<std::size_t>(i - 1)] * local_point.x();
        scratch.y_powers[static_cast<std::size_t>(i)] =
            scratch.y_powers[static_cast<std::size_t>(i - 1)] * local_point.y();
    }

    int moment_index = 0;
    for (int total_degree = 0; total_degree <= degree; ++total_degree) {
        for (int a = total_degree; a >= 0; --a, ++moment_index) {
            const int b = total_degree - a;
            const double monomial =
                scratch.x_powers[static_cast<std::size_t>(a)] *
                scratch.y_powers[static_cast<std::size_t>(b)];
            moments[static_cast<std::size_t>(moment_index)] += weight * monomial * velocity;
        }
    }
}

int resolved_cell_moment_order(const int num_edges, const MomentMethodOptions& options)
{
    if (options.cell_moment_order >= 0) {
        return options.cell_moment_order;
    }

    const int degree = std::max(0, options.edge_moment_order);
    const int edge_constraints = num_edges * (degree + 1);
    const int basis_count = vector_polynomial_basis_count(degree);
    if (edge_constraints >= basis_count) {
        return -1;
    }

    int q = 0;
    while (edge_constraints + 2 * vector_moment_basis_count(q) < basis_count) {
        ++q;
    }
    return q;
}

Eigen::Vector2d moment_velocity_value(const MomentReconstruction& reconstruction, const Eigen::Vector2d& p)
{
    const std::vector<VectorBasisTerm>& basis =
        cached_vector_polynomial_basis(reconstruction.vector_polynomial_degree);
    if (basis.size() != reconstruction.coefficients.size()) {
        throw std::runtime_error("Moment reconstruction basis size mismatch");
    }

    MonomialScratch& scratch = monomial_scratch();
    const int degree = reconstruction.vector_polynomial_degree;
    scratch.x_powers.resize(static_cast<std::size_t>(degree + 1), 1.0);
    scratch.y_powers.resize(static_cast<std::size_t>(degree + 1), 1.0);
    const double x_scaled = p.x() / reconstruction.length_scale;
    const double y_scaled = p.y() / reconstruction.length_scale;
    for (int i = 1; i <= degree; ++i) {
        scratch.x_powers[static_cast<std::size_t>(i)] =
            scratch.x_powers[static_cast<std::size_t>(i - 1)] * x_scaled;
        scratch.y_powers[static_cast<std::size_t>(i)] =
            scratch.y_powers[static_cast<std::size_t>(i - 1)] * y_scaled;
    }

    Eigen::Vector2d value = Eigen::Vector2d::Zero();
    for (std::size_t i = 0; i < basis.size(); ++i) {
        const double monomial =
            scratch.x_powers[static_cast<std::size_t>(basis[i].a)] *
            scratch.y_powers[static_cast<std::size_t>(basis[i].b)];
        value[basis[i].component] += reconstruction.coefficients[i] * monomial;
    }
    return value;
}

std::vector<double> basis_edge_moments(const VectorBasisTerm& term,
                                       const LocalPolygon& polygon,
                                       const std::size_t edge_index,
                                       const LocalEdge& edge,
                                       const int order,
                                       const std::vector<GaussLegendrePoint>& quadrature,
                                       const double scale,
                                       const GeometryOptions& options)
{
    std::vector<double> moments;
    const Eigen::Vector2d delta = edge.b - edge.a;
    const double denom = delta.squaredNorm();
    Eigen::Vector3d a3 = Eigen::Vector3d::Zero();
    double total_angle = 0.0;
    GnomonicFrame frame;
    if (options.mode == GeometryMode::SphericalGnomonic) {
        a3 = polygon.points_3d[edge_index].normalized();
        const Eigen::Vector3d b3 = polygon.points_3d[(edge_index + 1) % polygon.points_3d.size()].normalized();
        total_angle = std::acos(clamp_unit(a3.dot(b3)));
        frame = GnomonicFrame{polygon.n, polygon.e_x, polygon.e_y, options.radius};
    }
    accumulate_edge_moment_bundle(edge.a, edge.b, quadrature, order, moments, [&](const Eigen::Vector2d& p) {
            double t = 0.0;
            if (options.mode == GeometryMode::SphericalGnomonic) {
                if (total_angle > kTolerance) {
                    const Eigen::Vector2d xi = p + polygon.centroid;
                    const Eigen::Vector3d point3 = inverse_gnomonic(xi, frame).normalized();
                    const double angle = std::acos(clamp_unit(a3.dot(point3)));
                    t = 2.0 * (angle / total_angle) - 1.0;
                }
            } else {
                t = (denom > kTolerance)
                    ? (2.0 * (p - edge.a).dot(delta) / denom - 1.0)
                    : 0.0;
            }
            return EdgeMomentSample{vector_basis_value(term, p, scale).dot(edge.outward_normal), t};
        });
    return moments;
}

std::vector<Eigen::Vector2d> basis_cell_vector_moments(const VectorBasisTerm& term,
                                                       const std::vector<LocalEdge>& edges,
                                                       const int degree,
                                                       const std::vector<GaussLegendrePoint>& quadrature,
                                                       const double scale)
{
    std::vector<Eigen::Vector2d> moments;
    if (degree < 0) {
        return moments;
    }
    moments.assign(static_cast<std::size_t>(vector_moment_basis_count(degree)), Eigen::Vector2d::Zero());
    const Eigen::Vector2d origin = Eigen::Vector2d::Zero();
    for (const LocalEdge& edge : edges) {
        const Eigen::Vector2d ab = edge.a - origin;
        const Eigen::Vector2d bc = edge.b - edge.a;
        const double det_j = std::abs(ab.x() * bc.y() - ab.y() * bc.x());
        for (const GaussLegendrePoint& qu : quadrature) {
            const double u = 0.5 * (qu.x + 1.0);
            const double wu = 0.5 * qu.w;
            for (const GaussLegendrePoint& qv : quadrature) {
                const double v = 0.5 * (qv.x + 1.0);
                const double wv = 0.5 * qv.w;
                const Eigen::Vector2d p = origin + u * ab + (u * v) * bc;
                const double weight = det_j * wu * wv * u;
                accumulate_vector_cell_moments(vector_basis_value(term, p, scale), p, degree, weight, moments);
            }
        }
    }
    return moments;
}

struct DirectedTargetEdgeInfo {
    DirectedEdgeDof dof;
    std::size_t cell_index = 0;
    Eigen::Vector2d a2 = Eigen::Vector2d::Zero();
    Eigen::Vector2d b2 = Eigen::Vector2d::Zero();
    Eigen::Vector3d a3 = Eigen::Vector3d::Zero();
    Eigen::Vector3d b3 = Eigen::Vector3d::Zero();
};

struct CollapsedTargetEdges {
    std::vector<std::size_t> directed_to_unique;
    std::vector<int> directed_signs;
    std::size_t unique_count = 0;
};

bool same_edge_endpoint(const Eigen::Vector2d& a, const Eigen::Vector2d& b, const double tolerance)
{
    return (a - b).norm() <= tolerance;
}

bool same_edge_endpoint(const Eigen::Vector3d& a, const Eigen::Vector3d& b, const double tolerance)
{
    return (a.normalized() - b.normalized()).norm() <= tolerance;
}

int edge_orientation_sign(const DirectedTargetEdgeInfo& reference,
                          const DirectedTargetEdgeInfo& current,
                          const GeometryOptions& options)
{
    if (options.mode == GeometryMode::SphericalGnomonic) {
        const bool same =
            same_edge_endpoint(reference.a3, current.a3, options.geometry_tolerance) &&
            same_edge_endpoint(reference.b3, current.b3, options.geometry_tolerance);
        if (same) {
            return 1;
        }
        const bool reversed =
            same_edge_endpoint(reference.a3, current.b3, options.geometry_tolerance) &&
            same_edge_endpoint(reference.b3, current.a3, options.geometry_tolerance);
        if (reversed) {
            return -1;
        }
        return 0;
    }

    const bool same =
        same_edge_endpoint(reference.a2, current.a2, options.geometry_tolerance) &&
        same_edge_endpoint(reference.b2, current.b2, options.geometry_tolerance);
    if (same) {
        return 1;
    }
    const bool reversed =
        same_edge_endpoint(reference.a2, current.b2, options.geometry_tolerance) &&
        same_edge_endpoint(reference.b2, current.a2, options.geometry_tolerance);
    if (reversed) {
        return -1;
    }
    return 0;
}

std::vector<DirectedTargetEdgeInfo> build_directed_target_edges(moab::Core& mb,
                                                                const std::vector<moab::EntityHandle>& target_polygons,
                                                                const GeometryOptions& options)
{
    std::vector<DirectedTargetEdgeInfo> edges;
    for (std::size_t cell_index = 0; cell_index < target_polygons.size(); ++cell_index) {
        const moab::EntityHandle target_polygon = target_polygons[cell_index];
        const LocalPolygon target = local_polygon(mb, target_polygon, options);
        const std::vector<LocalEdge> local = local_edges(mb, target);
        for (std::size_t edge_index = 0; edge_index < local.size(); ++edge_index) {
            DirectedTargetEdgeInfo info;
            info.dof = DirectedEdgeDof{target_polygon, local[edge_index].handle, edge_index};
            info.cell_index = cell_index;
            if (options.mode == GeometryMode::SphericalGnomonic) {
                info.a3 = target.points_3d[edge_index].normalized();
                info.b3 = target.points_3d[(edge_index + 1) % target.points_3d.size()].normalized();
            } else {
                info.a2 = target.centroid + local[edge_index].a;
                info.b2 = target.centroid + local[edge_index].b;
            }
            edges.push_back(info);
        }
    }
    return edges;
}

void verify_raw_target_order(const EdgeTransferResult& raw_transfer, const std::vector<DirectedTargetEdgeInfo>& target_edges)
{
    if (raw_transfer.target_edges.size() != target_edges.size()) {
        throw std::runtime_error("Raw transfer target edge count does not match target mesh enumeration");
    }
    for (std::size_t i = 0; i < target_edges.size(); ++i) {
        const DirectedEdgeDof& expected = target_edges[i].dof;
        const DirectedEdgeDof& actual = raw_transfer.target_edges[i];
        if (expected.polygon != actual.polygon ||
            expected.edge != actual.edge ||
            expected.local_edge_index != actual.local_edge_index) {
            throw std::runtime_error("Raw transfer target edge ordering does not match target mesh enumeration");
        }
    }
}

void verify_raw_target_order(const EdgeMomentTransferResult& raw_transfer,
                             const std::vector<DirectedTargetEdgeInfo>& target_edges)
{
    if (raw_transfer.target_edges.size() != target_edges.size()) {
        throw std::runtime_error("Raw high-order transfer target edge count does not match target mesh enumeration");
    }
    for (std::size_t i = 0; i < target_edges.size(); ++i) {
        const DirectedEdgeDof& expected = target_edges[i].dof;
        const DirectedEdgeDof& actual = raw_transfer.target_edges[i];
        if (expected.polygon != actual.polygon ||
            expected.edge != actual.edge ||
            expected.local_edge_index != actual.local_edge_index) {
            throw std::runtime_error("Raw high-order transfer target edge ordering does not match target mesh enumeration");
        }
    }
}

double target_edge_moment_orientation_factor(const int orientation, const int degree)
{
    if (orientation == 1) {
        return 1.0;
    }
    if (orientation == -1) {
        return ((degree % 2) == 0) ? -1.0 : 1.0;
    }
    throw std::runtime_error("Invalid target edge orientation");
}

CollapsedTargetEdges collapse_target_edges(const std::vector<DirectedTargetEdgeInfo>& target_edges,
                                           const GeometryOptions& options)
{
    CollapsedTargetEdges collapse;
    collapse.directed_to_unique.resize(target_edges.size());
    collapse.directed_signs.resize(target_edges.size());

    std::map<moab::EntityHandle, std::pair<std::size_t, DirectedTargetEdgeInfo>> unique_edge_map;
    for (std::size_t i = 0; i < target_edges.size(); ++i) {
        const moab::EntityHandle edge = target_edges[i].dof.edge;
        const auto insertion = unique_edge_map.insert(
            std::make_pair(edge, std::make_pair(unique_edge_map.size(), target_edges[i])));
        collapse.directed_to_unique[i] = insertion.first->second.first;
        if (insertion.second) {
            collapse.directed_signs[i] = 1;
            continue;
        }

        const int sign = edge_orientation_sign(insertion.first->second.second, target_edges[i], options);
        if (sign == 0) {
            throw std::runtime_error("Merged target edge orientation mismatch");
        }
        collapse.directed_signs[i] = sign;
    }
    collapse.unique_count = unique_edge_map.size();
    return collapse;
}

std::vector<double> target_divergence_rhs(moab::Core& mb,
                                          const moab::Tag coeff_tag,
                                          const GeometryOptions& options,
                                          const std::vector<moab::EntityHandle>& source_polygons,
                                          const std::vector<moab::EntityHandle>& target_polygons)
{
    struct SourceCache {
        moab::EntityHandle polygon;
        LocalPolygon local;
        std::vector<Eigen::Vector2d> absolute;
        ReconstructionCoeffs coeffs;
        GnomonicFrame frame;
        PolygonSearchGeometry search;
    };

    struct TargetCache {
        moab::EntityHandle polygon;
        LocalPolygon local;
        PolygonSearchGeometry search;
    };

    std::vector<SourceCache> sources;
    sources.reserve(source_polygons.size());
    for (const moab::EntityHandle source_polygon : source_polygons) {
        const LocalPolygon local = local_polygon(mb, source_polygon, options);
        sources.push_back(SourceCache{
            source_polygon,
            local,
            absolute_points(local),
            read_coefficients(mb, coeff_tag, source_polygon),
            GnomonicFrame{local.n, local.e_x, local.e_y, options.radius},
            polygon_search_geometry(local, options),
        });
    }

    const SpatialIndex div_idx = build_spatial_index_from_sources(
        sources, options.mode == GeometryMode::SphericalGnomonic);

    std::vector<TargetCache> targets;
    targets.reserve(target_polygons.size());
    for (const moab::EntityHandle target_polygon : target_polygons) {
        const LocalPolygon local = local_polygon(mb, target_polygon, options);
        targets.push_back(TargetCache{
            target_polygon,
            local,
            polygon_search_geometry(local, options),
        });
    }

    std::vector<double> rhs(target_polygons.size(), 0.0);
#ifdef MIMETIC_ENABLE_OPENMP
#pragma omp parallel for schedule(static) if(targets.size() >= 64)
#endif
    for (int target_index = 0; target_index < static_cast<int>(targets.size()); ++target_index) {
        const TargetCache& target = targets[static_cast<std::size_t>(target_index)];
        const auto div_cands = find_overlap_candidates(div_idx,
                                                       target.search.center,
                                                       target.search.radius,
                                                       sources.size());

        for (const std::size_t si : div_cands) {
            const SourceCache& source = sources[si];
            std::vector<Eigen::Vector2d> target_in_source;
            if (options.mode == GeometryMode::SphericalGnomonic) {
                target_in_source.reserve(target.local.points_3d.size());
                bool valid_projection = true;
                for (const Eigen::Vector3d& p3d : target.local.points_3d) {
                    try {
                        target_in_source.push_back(project_gnomonic(p3d, source.frame));
                    } catch (const std::runtime_error&) {
                        valid_projection = false;
                        break;
                    }
                }
                if (!valid_projection) {
                    continue;
                }
            } else {
                target_in_source = absolute_points(target.local);
            }

            const std::vector<Eigen::Vector2d> overlap =
                convex_polygon_intersection(target_in_source, source.absolute, options.geometry_tolerance);
            const double overlap_chart_area = polygon_area_abs(overlap);
            if (overlap_chart_area <= options.geometry_tolerance) {
                continue;
            }
            rhs[static_cast<std::size_t>(target_index)] += source.coeffs.d * overlap_chart_area;
        }
    }
    return rhs;
}

double moment_polygon_boundary_flux(const MomentReconstruction& reconstruction,
                                    const std::vector<Eigen::Vector2d>& polygon_points_absolute,
                                    const Eigen::Vector2d& source_centroid,
                                    const std::vector<GaussLegendrePoint>& quadrature)
{
    if (polygon_points_absolute.size() < 3) {
        return 0.0;
    }

    std::vector<Eigen::Vector2d> points = polygon_points_absolute;
    if (signed_area(points) < 0.0) {
        std::reverse(points.begin(), points.end());
    }

    double flux = 0.0;
    for (std::size_t i = 0; i < points.size(); ++i) {
        const Eigen::Vector2d a = points[i];
        const Eigen::Vector2d b = points[(i + 1) % points.size()];
        const Eigen::Vector2d delta = b - a;
        const double length = delta.norm();
        if (length <= kTolerance) {
            continue;
        }
        const Eigen::Vector2d outward(delta.y(), -delta.x());
        const Eigen::Vector2d unit_outward = outward / length;
        flux += integrate_edge_highorder_rule(a, b, quadrature, [&](const Eigen::Vector2d& p_abs) {
            return moment_velocity_value(reconstruction, p_abs - source_centroid).dot(unit_outward);
        });
    }
    return flux;
}

std::vector<double> target_divergence_rhs(moab::Core& mb,
                                          const GeometryOptions& options,
                                          const std::map<moab::EntityHandle, MomentReconstruction>& reconstructions,
                                          const std::vector<moab::EntityHandle>& source_polygons,
                                          const std::vector<moab::EntityHandle>& target_polygons)
{
    struct SourceCache {
        moab::EntityHandle polygon;
        LocalPolygon local;
        std::vector<Eigen::Vector2d> absolute;
        MomentReconstruction reconstruction;
        GnomonicFrame frame;
        PolygonSearchGeometry search;
    };

    struct TargetCache {
        moab::EntityHandle polygon;
        LocalPolygon local;
        PolygonSearchGeometry search;
    };

    std::vector<SourceCache> sources;
    sources.reserve(source_polygons.size());
    for (const moab::EntityHandle source_polygon : source_polygons) {
        const auto it = reconstructions.find(source_polygon);
        if (it == reconstructions.end()) {
            throw std::runtime_error("Missing high-order reconstruction for source polygon");
        }
        const LocalPolygon local = local_polygon(mb, source_polygon, options);
        sources.push_back(SourceCache{
            source_polygon,
            local,
            absolute_points(local),
            it->second,
            GnomonicFrame{local.n, local.e_x, local.e_y, options.radius},
            polygon_search_geometry(local, options),
        });
    }

    const std::vector<GaussLegendrePoint> quadrature = gauss_legendre_rule(10);
    const SpatialIndex div_idx = build_spatial_index_from_sources(
        sources, options.mode == GeometryMode::SphericalGnomonic);

    std::vector<TargetCache> targets;
    targets.reserve(target_polygons.size());
    for (const moab::EntityHandle target_polygon : target_polygons) {
        const LocalPolygon local = local_polygon(mb, target_polygon, options);
        targets.push_back(TargetCache{
            target_polygon,
            local,
            polygon_search_geometry(local, options),
        });
    }

    std::vector<double> rhs(target_polygons.size(), 0.0);
#ifdef MIMETIC_ENABLE_OPENMP
#pragma omp parallel for schedule(static) if(targets.size() >= 64)
#endif
    for (int target_index = 0; target_index < static_cast<int>(targets.size()); ++target_index) {
        const TargetCache& target = targets[static_cast<std::size_t>(target_index)];
        const auto div_cands = find_overlap_candidates(div_idx,
                                                       target.search.center,
                                                       target.search.radius,
                                                       sources.size());

        for (const std::size_t si : div_cands) {
            const SourceCache& source = sources[si];
            std::vector<Eigen::Vector2d> target_in_source;
            if (options.mode == GeometryMode::SphericalGnomonic) {
                target_in_source.reserve(target.local.points_3d.size());
                bool valid_projection = true;
                for (const Eigen::Vector3d& p3d : target.local.points_3d) {
                    try {
                        target_in_source.push_back(project_gnomonic(p3d, source.frame));
                    } catch (const std::runtime_error&) {
                        valid_projection = false;
                        break;
                    }
                }
                if (!valid_projection) {
                    continue;
                }
            } else {
                target_in_source = absolute_points(target.local);
            }
            const std::vector<Eigen::Vector2d> overlap =
                convex_polygon_intersection(target_in_source, source.absolute, options.geometry_tolerance);
            if (polygon_area_abs(overlap) <= options.geometry_tolerance) {
                continue;
            }
            rhs[static_cast<std::size_t>(target_index)] += moment_polygon_boundary_flux(
                source.reconstruction, overlap, source.local.centroid, quadrature);
        }
    }
    return rhs;
}

void write_edge_map_csv(const std::string& path, const std::vector<DirectedEdgeDof>& edges)
{
    std::ofstream out(path.c_str());
    if (!out) {
        throw std::runtime_error("Failed to open edge map for writing: " + path);
    }

    out << "index,polygon_handle,edge_handle,local_edge_index\n";
    for (std::size_t i = 0; i < edges.size(); ++i) {
        out << i << "," << edges[i].polygon << "," << edges[i].edge << "," << edges[i].local_edge_index << "\n";
    }
}

}  // namespace

// Shoelace geometry utilities; these correspond to the polygon preprocessing
// stage in report Algorithm 1, steps 1--2.
double signed_area(const std::vector<Eigen::Vector2d>& points)
{
    double area2 = 0.0;
    for (std::size_t i = 0; i < points.size(); ++i) {
        const Eigen::Vector2d& a = points[i];
        const Eigen::Vector2d& b = points[(i + 1) % points.size()];
        area2 += a.x() * b.y() - b.x() * a.y();
    }
    return 0.5 * area2;
}

Eigen::Vector2d polygon_centroid(const std::vector<Eigen::Vector2d>& points)
{
    double area2 = 0.0;
    Eigen::Vector2d weighted(0.0, 0.0);
    for (std::size_t i = 0; i < points.size(); ++i) {
        const Eigen::Vector2d& a = points[i];
        const Eigen::Vector2d& b = points[(i + 1) % points.size()];
        const double cross = a.x() * b.y() - b.x() * a.y();
        area2 += cross;
        weighted += (a + b) * cross;
    }
    if (std::abs(area2) < kTolerance) {
        throw std::runtime_error("Degenerate polygon area while computing centroid");
    }
    return weighted / (3.0 * area2);
}

moab::EntityHandle find_or_create_edge(moab::Core& mb, const moab::EntityHandle v0, const moab::EntityHandle v1)
{
    moab::EntityHandle verts[2] = {v0, v1};
    std::vector<moab::EntityHandle> edges;
    check_moab(mb.get_adjacencies(verts, 2, 1, false, edges, moab::Interface::INTERSECT),
               "Failed to query edge adjacency");
    if (!edges.empty()) {
        return edges.front();
    }

    moab::EntityHandle edge = 0;
    check_moab(mb.create_element(moab::MBEDGE, verts, 2, edge), "Failed to create edge");
    return edge;
}

Eigen::Vector2d project_gnomonic(const Eigen::Vector3d& point, const GnomonicFrame& frame)
{
    const Eigen::Vector3d unit = normalized_or_throw(point, "Cannot project zero vector onto gnomonic chart");
    const double denom = unit.dot(frame.center);
    if (denom <= kTolerance) {
        throw std::runtime_error("Point lies outside the gnomonic chart hemisphere");
    }
    const Eigen::Vector3d projected = unit / denom;
    return Eigen::Vector2d(projected.dot(frame.e_x), projected.dot(frame.e_y));
}

Eigen::Vector3d inverse_gnomonic(const Eigen::Vector2d& xi, const GnomonicFrame& frame)
{
    const Eigen::Vector3d ray = frame.center + xi.x() * frame.e_x + xi.y() * frame.e_y;
    return frame.radius * normalized_or_throw(ray, "Cannot invert degenerate gnomonic coordinate");
}

Eigen::Matrix<double, 3, 2> gnomonic_jacobian(const Eigen::Vector2d& xi, const GnomonicFrame& frame)
{
    const Eigen::Vector3d ray = frame.center + xi.x() * frame.e_x + xi.y() * frame.e_y;
    const double q = ray.norm();
    if (q < kTolerance) {
        throw std::runtime_error("Degenerate inverse gnomonic Jacobian");
    }
    const Eigen::Vector3d unit = ray / q;

    Eigen::Matrix<double, 3, 2> jacobian;
    jacobian.col(0) = frame.radius * (frame.e_x - xi.x() * unit) / q;
    jacobian.col(1) = frame.radius * (frame.e_y - xi.y() * unit) / q;
    return jacobian;
}

double gnomonic_area_scale(const Eigen::Vector2d& xi, const GnomonicFrame& frame)
{
    const Eigen::Matrix<double, 3, 2> jacobian = gnomonic_jacobian(xi, frame);
    return jacobian.col(0).cross(jacobian.col(1)).norm();
}

double chart_polygon_surface_area(const std::vector<Eigen::Vector2d>& polygon,
                                  const GnomonicFrame& frame)
{
    if (polygon.size() < 3) {
        return 0.0;
    }

    std::vector<Eigen::Vector2d> polygon_ccw = polygon;
    if (signed_area(polygon_ccw) < 0.0) {
        std::reverse(polygon_ccw.begin(), polygon_ccw.end());
    }

    double area = 0.0;
    std::vector<Eigen::Vector3d> lifted_points;
    lifted_points.reserve(polygon_ccw.size());
    for (const Eigen::Vector2d& xi : polygon_ccw) {
        lifted_points.push_back(inverse_gnomonic(xi, frame).normalized());
    }
    for (std::size_t i = 0; i < lifted_points.size(); ++i) {
        const Eigen::Vector3d& previous = lifted_points[(i + lifted_points.size() - 1) % lifted_points.size()];
        const Eigen::Vector3d& vertex = lifted_points[i];
        const Eigen::Vector3d& next = lifted_points[(i + 1) % lifted_points.size()];
        const Eigen::Vector3d tangent_prev =
            (previous - previous.dot(vertex) * vertex).normalized();
        const Eigen::Vector3d tangent_next =
            (next - next.dot(vertex) * vertex).normalized();
        area += std::acos(clamp_unit(tangent_prev.dot(tangent_next)));
    }
    area -= (static_cast<double>(lifted_points.size()) - 2.0) * kPi;
    return area * frame.radius * frame.radius;
}

Eigen::Vector3d lift_contravariant_piola(const Eigen::Vector2d& chart_vector,
                                         const Eigen::Vector2d& xi,
                                         const GnomonicFrame& frame)
{
    const Eigen::Matrix<double, 3, 2> jacobian = gnomonic_jacobian(xi, frame);
    const double area_scale = jacobian.col(0).cross(jacobian.col(1)).norm();
    if (area_scale < kTolerance) {
        throw std::runtime_error("Degenerate Piola lift");
    }
    return (jacobian * chart_vector) / area_scale;
}

Eigen::Vector2d pullback_contravariant_piola(const Eigen::Vector3d& surface_vector,
                                             const Eigen::Vector2d& xi,
                                             const GnomonicFrame& frame)
{
    const Eigen::Matrix<double, 3, 2> jacobian = gnomonic_jacobian(xi, frame);
    const double area_scale = jacobian.col(0).cross(jacobian.col(1)).norm();
    if (area_scale < kTolerance) {
        throw std::runtime_error("Degenerate Piola pullback");
    }
    const Eigen::Matrix2d metric = jacobian.transpose() * jacobian;
    return area_scale * metric.ldlt().solve(jacobian.transpose() * surface_vector);
}

SphericalPolygon spherical_polygon(moab::Core& mb, const moab::EntityHandle polygon, const GeometryOptions& options)
{
    const moab::EntityHandle* conn = nullptr;
    int num_vertices = 0;
    check_moab(mb.get_connectivity(polygon, conn, num_vertices), "Failed to get spherical polygon connectivity");
    if (num_vertices < 3) {
        throw std::runtime_error("Spherical polygon must have at least three vertices");
    }

    SphericalPolygon poly;
    poly.vertices.assign(conn, conn + num_vertices);
    poly.points.reserve(poly.vertices.size());

    for (const moab::EntityHandle vertex : poly.vertices) {
        double xyz[3] = {0.0, 0.0, 0.0};
        check_moab(mb.get_coords(&vertex, 1, xyz), "Failed to get spherical vertex coordinates");
        Eigen::Vector3d p(xyz[0], xyz[1], xyz[2]);
        poly.points.push_back(options.radius * normalized_or_throw(p, "Degenerate spherical vertex"));
    }

    poly.frame = make_gnomonic_frame(poly.points, options.radius);
    poly.projected_points.reserve(poly.points.size());
    for (const Eigen::Vector3d& p : poly.points) {
        poly.projected_points.push_back(project_gnomonic(p, poly.frame));
    }

    if (signed_area(poly.projected_points) < 0.0) {
        std::reverse(poly.vertices.begin(), poly.vertices.end());
        std::reverse(poly.points.begin(), poly.points.end());
        std::reverse(poly.projected_points.begin(), poly.projected_points.end());
    }

    poly.projected_centroid = polygon_centroid(poly.projected_points);
    poly.local_points.reserve(poly.projected_points.size());
    for (const Eigen::Vector2d& p : poly.projected_points) {
        poly.local_points.push_back(p - poly.projected_centroid);
    }

    poly.chart_area = std::abs(signed_area(poly.projected_points));
    poly.spherical_area = 0.0;
    const Eigen::Vector3d c = poly.frame.radius * poly.frame.center;
    for (std::size_t i = 0; i < poly.points.size(); ++i) {
        poly.spherical_area += spherical_triangle_area(c, poly.points[i], poly.points[(i + 1) % poly.points.size()]);
    }
    poly.spherical_area *= options.radius * options.radius;
    return poly;
}

std::vector<SphericalEdge> spherical_edges(moab::Core& mb, const SphericalPolygon& polygon)
{
    std::vector<SphericalEdge> edges;
    edges.reserve(polygon.points.size());
    for (std::size_t i = 0; i < polygon.points.size(); ++i) {
        const std::size_t j = (i + 1) % polygon.points.size();
        const double angle = std::acos(clamp_unit(polygon.points[i].normalized().dot(polygon.points[j].normalized())));
        edges.push_back(SphericalEdge{
            find_or_create_edge(mb, polygon.vertices[i], polygon.vertices[j]),
            polygon.points[i],
            polygon.points[j],
            polygon.projected_points[i],
            polygon.projected_points[j],
            polygon.frame.radius * angle,
        });
    }
    return edges;
}

// Extract MOAB connectivity into a local cell frame. The code enforces positive
// orientation because all later outward-normal signs assume counter-clockwise
// boundary order.
LocalPolygon local_polygon(moab::Core& mb, const moab::EntityHandle polygon, const GeometryOptions& options)
{
    if (options.mode == GeometryMode::SphericalGnomonic) {
        const SphericalPolygon sph = spherical_polygon(mb, polygon, options);
        return LocalPolygon{
            sph.vertices,
            sph.local_points,
            sph.projected_centroid,
            sph.chart_area,
            sph.spherical_area,
            sph.points,
            sph.frame.radius * sph.frame.center,
            sph.frame.e_x,
            sph.frame.e_y,
            sph.frame.center,
        };
    }

    const moab::EntityHandle* conn = nullptr;
    int num_vertices = 0;
    check_moab(mb.get_connectivity(polygon, conn, num_vertices), "Failed to get polygon connectivity");
    if (num_vertices < 3) {
        throw std::runtime_error("Polygon must have at least three vertices");
    }

    std::vector<moab::EntityHandle> vertices(conn, conn + num_vertices);
    std::vector<Eigen::Vector2d> absolute_points;
    absolute_points.reserve(vertices.size());

    std::vector<Eigen::Vector3d> points_3d;
    Eigen::Vector3d centroid_3d(0, 0, 0);
    Eigen::Vector3d e_x(1, 0, 0);
    Eigen::Vector3d e_y(0, 1, 0);
    Eigen::Vector3d n(0, 0, 1);

    for (const moab::EntityHandle vertex : vertices) {
        double xyz[3] = {0.0, 0.0, 0.0};
        check_moab(mb.get_coords(&vertex, 1, xyz), "Failed to get vertex coordinates");
        absolute_points.emplace_back(xyz[0], xyz[1]);
        // Still populate 3D points for consistency, but z is ignored.
        Eigen::Vector3d p(xyz[0], xyz[1], xyz[2]);
        points_3d.push_back(p);
        centroid_3d += p;
    }
    centroid_3d /= vertices.size();

    if (signed_area(absolute_points) < 0.0) {
        std::reverse(vertices.begin(), vertices.end());
        std::reverse(absolute_points.begin(), absolute_points.end());
        std::reverse(points_3d.begin(), points_3d.end());
    }

    const Eigen::Vector2d centroid = polygon_centroid(absolute_points);
    std::vector<Eigen::Vector2d> relative_points;
    relative_points.reserve(absolute_points.size());
    for (const Eigen::Vector2d& p : absolute_points) {
        relative_points.push_back(p - centroid);
    }

    return LocalPolygon{vertices, relative_points, centroid, std::abs(signed_area(absolute_points)), std::abs(signed_area(absolute_points)), points_3d, centroid_3d, e_x, e_y, n};
}

LocalPolygon local_polygon(moab::Core& mb, const moab::EntityHandle polygon, const bool is_spherical)
{
    GeometryOptions options;
    options.mode = is_spherical ? GeometryMode::SphericalGnomonic : GeometryMode::Planar;
    return local_polygon(mb, polygon, options);
}

std::vector<LocalEdge> local_edges(moab::Core& mb, const LocalPolygon& polygon)
{
    std::vector<LocalEdge> edges;
    edges.reserve(polygon.points.size());
    for (std::size_t i = 0; i < polygon.points.size(); ++i) {
        const std::size_t j = (i + 1) % polygon.points.size();
        const Eigen::Vector2d a = polygon.points[i];
        const Eigen::Vector2d b = polygon.points[j];
        const Eigen::Vector2d delta = b - a;
        const double length = delta.norm();
        if (length < kTolerance) {
            throw std::runtime_error("Degenerate edge length");
        }
        const Eigen::Vector2d outward(delta.y(), -delta.x());
        edges.push_back(LocalEdge{
            find_or_create_edge(mb, polygon.vertices[i], polygon.vertices[j]),
            a,
            b,
            outward / length,
            length,
        });
    }
    return edges;
}

moab::EntityHandle create_quad(moab::Core& mb, const std::array<Eigen::Vector2d, 4>& points)
{
    return create_polygon(mb, std::vector<Eigen::Vector2d>(points.begin(), points.end()));
}

moab::EntityHandle create_polygon(moab::Core& mb, const std::vector<Eigen::Vector2d>& points)
{
    if (points.size() < 3) {
        throw std::runtime_error("Cannot create a polygon with fewer than three points");
    }

    std::vector<Eigen::Vector2d> ordered_points = points;
    if (signed_area(ordered_points) < 0.0) {
        std::reverse(ordered_points.begin(), ordered_points.end());
    }

    std::vector<moab::EntityHandle> vertices(ordered_points.size(), 0);
    for (std::size_t i = 0; i < points.size(); ++i) {
        const double xyz[3] = {ordered_points[i].x(), ordered_points[i].y(), 0.0};
        check_moab(mb.create_vertex(xyz, vertices[i]), "Failed to create vertex");
    }

    moab::EntityHandle polygon = 0;
    const moab::EntityType type = (vertices.size() == 4) ? moab::MBQUAD : moab::MBPOLYGON;
    check_moab(mb.create_element(type, vertices.data(), static_cast<int>(vertices.size()), polygon),
               "Failed to create polygon");
    for (std::size_t i = 0; i < vertices.size(); ++i) {
        find_or_create_edge(mb, vertices[i], vertices[(i + 1) % vertices.size()]);
    }
    return polygon;
}

void merge_polygon_vertices(moab::Core& mb,
                            const std::vector<moab::EntityHandle>& polygons,
                            const double merge_tolerance)
{
    if (polygons.empty()) {
        return;
    }

    moab::Range elems;
    for (const moab::EntityHandle polygon : polygons) {
        elems.insert(polygon);
    }

    moab::MergeMesh merger(&mb);
    check_moab(merger.merge_entities(elems, merge_tolerance, true, false, 0, true),
               "Failed to merge coincident polygon vertices");
}

bool clip_segment_to_convex_polygon(const Eigen::Vector2d& segment_a,
                                    const Eigen::Vector2d& segment_b,
                                    const std::vector<Eigen::Vector2d>& polygon,
                                    Eigen::Vector2d& clipped_a,
                                    Eigen::Vector2d& clipped_b,
                                    const double tolerance)
{
    if (polygon.size() < 3) {
        return false;
    }

    std::vector<Eigen::Vector2d> clip = polygon;
    if (signed_area(clip) < 0.0) {
        std::reverse(clip.begin(), clip.end());
    }

    const Eigen::Vector2d delta = segment_b - segment_a;
    double t_enter = 0.0;
    double t_exit = 1.0;

    for (std::size_t i = 0; i < clip.size(); ++i) {
        const Eigen::Vector2d a = clip[i];
        const Eigen::Vector2d b = clip[(i + 1) % clip.size()];
        const Eigen::Vector2d edge = b - a;
        const Eigen::Vector2d inward_normal(-edge.y(), edge.x());
        const double offset = inward_normal.dot(a);
        const double signed_distance = inward_normal.dot(segment_a) - offset;
        const double rate = inward_normal.dot(delta);

        if (std::abs(rate) < tolerance) {
            if (signed_distance < -tolerance) {
                return false;
            }
            continue;
        }

        const double t = -signed_distance / rate;
        if (rate > 0.0) {
            t_enter = std::max(t_enter, t);
        } else {
            t_exit = std::min(t_exit, t);
        }

        if (t_enter > t_exit + tolerance) {
            return false;
        }
    }

    t_enter = std::max(0.0, t_enter);
    t_exit = std::min(1.0, t_exit);
    if (t_exit - t_enter <= tolerance) {
        return false;
    }

    clipped_a = segment_a + t_enter * delta;
    clipped_b = segment_a + t_exit * delta;
    return true;
}

void write_matrix_market(const SparseEdgeProjection& projection,
                         const std::string& matrix_path,
                         const std::string& source_edges_path,
                         const std::string& target_edges_path)
{
    std::ofstream out(matrix_path.c_str());
    if (!out) {
        throw std::runtime_error("Failed to open MatrixMarket file for writing: " + matrix_path);
    }

    out << "%%MatrixMarket matrix coordinate real general\n";
    out << "% rows are directed target edges; columns are directed source edges\n";
    out << projection.matrix.rows() << " " << projection.matrix.cols() << " " << projection.matrix.nonZeros() << "\n";
    for (int row = 0; row < projection.matrix.outerSize(); ++row) {
        for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(projection.matrix, row); it; ++it) {
            out << (it.row() + 1) << " " << (it.col() + 1) << " " << it.value() << "\n";
        }
    }

    write_edge_map_csv(source_edges_path, projection.source_edges);
    write_edge_map_csv(target_edges_path, projection.target_edges);
}

MimeticInterpolator::MimeticInterpolator(moab::Core& moab_instance) : mb_(moab_instance)
{
    const double default_flux = 0.0;
    check_moab(mb_.tag_get_handle("SOURCE_FLUX", 1, moab::MB_TYPE_DOUBLE, tag_source_flux_,
                                  moab::MB_TAG_DENSE | moab::MB_TAG_CREAT, &default_flux),
               "Failed to create SOURCE_FLUX tag");
    check_moab(mb_.tag_get_handle("TARGET_FLUX", 1, moab::MB_TYPE_DOUBLE, tag_target_flux_,
                                  moab::MB_TAG_DENSE | moab::MB_TAG_CREAT, &default_flux),
               "Failed to create TARGET_FLUX tag");

        check_moab(mb_.tag_get_handle("COEFFS", 0, moab::MB_TYPE_DOUBLE, tag_coeffs_,
                                  moab::MB_TAG_VARLEN | moab::MB_TAG_SPARSE | moab::MB_TAG_CREAT),
               "Failed to create COEFFS tag");
}

moab::Tag MimeticInterpolator::source_flux_tag() const { return tag_source_flux_; }
moab::Tag MimeticInterpolator::target_flux_tag() const { return tag_target_flux_; }
moab::Tag MimeticInterpolator::coeffs_tag() const { return tag_coeffs_; }

void MimeticInterpolator::set_geometry_options(const GeometryOptions& options)
{
    if (options.radius <= 0.0) {
        throw std::runtime_error("GeometryOptions::radius must be positive");
    }
    if (options.conservation_tolerance <= 0.0 || options.geometry_tolerance <= 0.0) {
        throw std::runtime_error("Geometry tolerances must be positive");
    }
    options_ = options;
}

GeometryOptions MimeticInterpolator::geometry_options() const
{
    return options_;
}

void MimeticInterpolator::set_spherical(const bool is_spherical)
{
    options_.mode = is_spherical ? GeometryMode::SphericalGnomonic : GeometryMode::Planar;
}

bool MimeticInterpolator::is_spherical() const
{
    return options_.mode == GeometryMode::SphericalGnomonic;
}

void MimeticInterpolator::set_source_edge_flux(const moab::EntityHandle polygon,
                                               const std::size_t local_edge_index,
                                               const double flux)
{
    const LocalPolygon poly = local_polygon(mb_, polygon, options_);
    const std::vector<LocalEdge> edges = local_edges(mb_, poly);
    if (local_edge_index >= edges.size()) {
        throw std::runtime_error("Source local edge index out of range");
    }
    directed_source_flux_[std::make_pair(polygon, local_edge_index)] = flux;
    check_moab(mb_.tag_set_data(tag_source_flux_, &edges[local_edge_index].handle, 1, &flux),
               "Failed to write directed source flux tag");
}

double MimeticInterpolator::source_edge_flux(const moab::EntityHandle polygon,
                                             const std::size_t local_edge_index,
                                             const moab::EntityHandle edge) const
{
    const auto key = std::make_pair(polygon, local_edge_index);
    const auto it = directed_source_flux_.find(key);
    if (it != directed_source_flux_.end()) {
        return it->second;
    }

    double flux = 0.0;
    check_moab(mb_.tag_get_data(tag_source_flux_, &edge, 1, &flux), "Failed to read source flux tag");
    return flux;
}

double MimeticInterpolator::target_edge_flux(const moab::EntityHandle polygon,
                                             const std::size_t local_edge_index,
                                             const moab::EntityHandle edge) const
{
    const auto key = std::make_pair(polygon, local_edge_index);
    const auto it = directed_target_flux_.find(key);
    if (it != directed_target_flux_.end()) {
        return it->second;
    }

    double flux = 0.0;
    check_moab(mb_.tag_get_data(tag_target_flux_, &edge, 1, &flux), "Failed to read target flux tag");
    return flux;
}

ReconstructionCoeffs MimeticInterpolator::reconstruct_source_polygon(const moab::EntityHandle polygon)
{
    const LocalPolygon poly = local_polygon(mb_, polygon, options_);
    const std::vector<LocalEdge> edges = local_edges(mb_, poly);

    const int N = static_cast<int>(edges.size());
    const int K_max = N / 2;
    const int N_h = 2 * K_max;
    const int S = N_h + N - 1;
    const bool use_surface_metric = options_.metric_weighted && is_spherical();
    const GnomonicFrame frame{poly.n, poly.e_x, poly.e_y, poly.centroid_3d.norm()};
    const double stabilization_area =
        (use_surface_metric && poly.spherical_area > options_.geometry_tolerance) ? poly.spherical_area : poly.area;

    Eigen::VectorXd source_flux(N);
    for (int i = 0; i < N; ++i) {
        source_flux(i) = source_edge_flux(polygon, static_cast<std::size_t>(i), edges[i].handle);
    }

    const double divergence = source_flux.sum() / poly.area;

    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(N_h, N_h);
    Eigen::VectorXd M = Eigen::VectorXd::Zero(N_h);
    const Eigen::Vector2d origin(0.0, 0.0);
    const std::vector<GaussLegendrePoint> duffy_rule = gauss_legendre_rule(10);

    // Helper: use Duffy quadrature for metric-weighted integrals (rational integrands),
    // standard 7-point symmetric rule for polynomial integrands.
    auto integrate_cell = [&](const Eigen::Vector2d& a, const Eigen::Vector2d& b,
                              const Eigen::Vector2d& c, auto&& func) {
        if (use_surface_metric) {
            return integrate_triangle_duffy(a, b, c, duffy_rule, func);
        }
        return integrate_triangle_scalar(a, b, c, func);
    };

    for (int i = 0; i < N_h; ++i) {
        int ki = (i / 2) + 1;
        bool is_Q_i = (i % 2 == 1);
        for (int j = 0; j < N_h; ++j) {
            int kj = (j / 2) + 1;
            bool is_Q_j = (j % 2 == 1);

            double val = 0.0;
            for (const LocalEdge& edge : edges) {
                val += integrate_cell(origin, edge.a, edge.b, [&](const Eigen::Vector2d& p) {
                    double Pi, Qi, Pj, Qj;
                    Eigen::Vector2d gPi, gQi, gPj, gQj;
                    eval_harmonic_basis(ki, p, Pi, Qi, gPi, gQi);
                    eval_harmonic_basis(kj, p, Pj, Qj, gPj, gQj);
                    Eigen::Vector2d gi = is_Q_i ? gQi : gPi;
                    Eigen::Vector2d gj = is_Q_j ? gQj : gPj;
                    if (!use_surface_metric) {
                        return gi.dot(gj);
                    }
                    const Eigen::Matrix2d hodge = gnomonic_hodge_metric(p + poly.centroid, frame);
                    return gi.dot(hodge * gj);
                });
            }
            V(i, j) = val;
        }

        double cell_basis_integral = 0.0;
        double div_integral = 0.0;
        for (const LocalEdge& edge : edges) {
            cell_basis_integral += integrate_cell(origin, edge.a, edge.b, [&](const Eigen::Vector2d& p) {
                double P, Q; Eigen::Vector2d gP, gQ;
                eval_harmonic_basis(ki, p, P, Q, gP, gQ);
                return is_Q_i ? Q : P;
            });
            div_integral += integrate_cell(origin, edge.a, edge.b, [&](const Eigen::Vector2d& p) {
                double P, Q; Eigen::Vector2d gP, gQ;
                eval_harmonic_basis(ki, p, P, Q, gP, gQ);
                return p.dot(is_Q_i ? gQ : gP);
            });
        }
        const double cell_basis_average = cell_basis_integral / poly.area;
        if (i >= 4) {
            const double v_diag = std::max(V(i, i), kTolerance * stabilization_area);
            V(i, i) += 1.0e1 * v_diag;
        }

        for (int e = 0; e < N; ++e) {
            const double edge_average = integrate_edge_scalar(edges[e].a, edges[e].b, [&](const Eigen::Vector2d& p) {
                double P, Q; Eigen::Vector2d gP, gQ;
                eval_harmonic_basis(ki, p, P, Q, gP, gQ);
                return is_Q_i ? Q : P;
            }) / edges[e].length;
            M(i) += (edge_average - cell_basis_average - 0.5 * div_integral / poly.area) * source_flux(e);
        }
    }

    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(N - 1, N_h);
    Eigen::VectorXd F_vec = Eigen::VectorXd::Zero(N - 1);
    for (int e = 0; e < N - 1; ++e) {
        for (int i = 0; i < N_h; ++i) {
            int ki = (i / 2) + 1;
            bool is_Q_i = (i % 2 == 1);
            C(e, i) = integrate_edge_scalar(edges[e].a, edges[e].b, [&](const Eigen::Vector2d& p) {
                double P, Q;
                Eigen::Vector2d gP, gQ;
                eval_harmonic_basis(ki, p, P, Q, gP, gQ);
                Eigen::Vector2d g = is_Q_i ? gQ : gP;
                return g.dot(edges[e].outward_normal);
            });
        }
        double Ee = integrate_edge_scalar(edges[e].a, edges[e].b, [&](const Eigen::Vector2d& p) {
            return p.dot(edges[e].outward_normal);
        });
        F_vec(e) = source_flux(e) - 0.5 * divergence * Ee;
    }

    // Solve KKT system: [V C^T; C 0] [X; lambda] = [M; F_vec]
    Eigen::MatrixXd KKT = Eigen::MatrixXd::Zero(S, S);
    KKT.block(0, 0, N_h, N_h) = V;
    KKT.block(N_h, 0, N - 1, N_h) = C;
    KKT.block(0, N_h, N_h, N - 1) = C.transpose();

    Eigen::VectorXd RHS = Eigen::VectorXd::Zero(S);
    RHS.segment(0, N_h) = M;
    RHS.segment(N_h, N - 1) = F_vec;

    Eigen::FullPivLU<Eigen::MatrixXd> lu(KKT);
    Eigen::VectorXd X = lu.solve(RHS).segment(0, N_h);
    
    ReconstructionCoeffs coeffs;
    coeffs.d = divergence;
    coeffs.harmonic.assign(X.data(), X.data() + N_h);

    std::vector<double> tag_data(1 + N_h);
    tag_data[0] = coeffs.d;
    for (int i = 0; i < N_h; ++i) tag_data[1 + i] = coeffs.harmonic[i];

    const void* ptr = tag_data.data();
    int size = tag_data.size();
    check_moab(mb_.tag_set_by_ptr(tag_coeffs_, &polygon, 1, &ptr, &size), "Failed to store reconstruction coefficients");

    return coeffs;
}

Eigen::Vector2d MimeticInterpolator::velocity(const ReconstructionCoeffs& coeffs, const Eigen::Vector2d& p) const
{
    return harmonic_velocity_value(coeffs, p);
}

double MimeticInterpolator::line_integral(const moab::EntityHandle source_polygon,
                                          const Eigen::Vector2d& a,
                                          const Eigen::Vector2d& b) const
{
    const ReconstructionCoeffs coeffs = read_coefficients(mb_, tag_coeffs_, source_polygon);
    const double d = coeffs.d;

    double val = 0.25 * d * (b.squaredNorm() - a.squaredNorm());
    for (std::size_t i = 0; i < coeffs.harmonic.size(); ++i) {
        int k = (i / 2) + 1;
        bool is_Q = (i % 2 == 1);
        double Pa, Qa, Pb, Qb;
        Eigen::Vector2d gPa, gQa, gPb, gQb;
        eval_harmonic_basis(k, a, Pa, Qa, gPa, gQa);
        eval_harmonic_basis(k, b, Pb, Qb, gPb, gQb);
        val += coeffs.harmonic[i] * (is_Q ? (Qb - Qa) : (Pb - Pa));
    }
    return val;
}

double MimeticInterpolator::edge_flux(const ReconstructionCoeffs& coeffs,
                                      const Eigen::Vector2d& a,
                                      const Eigen::Vector2d& b) const
{
    // Boundary edges are treated as directed counter-clockwise edges of the
    // integration polygon. The outward normal is therefore the right normal.
    const Eigen::Vector2d delta = b - a;
    const double length = delta.norm();
    const Eigen::Vector2d outward(delta.y(), -delta.x());
    const Eigen::Vector2d normal = outward / length;
    return integrate_edge_scalar(a, b, [&](const Eigen::Vector2d& p) { return velocity(coeffs, p).dot(normal); });
}

double MimeticInterpolator::polygon_boundary_flux(const ReconstructionCoeffs& coeffs,
                                                  const std::vector<Eigen::Vector2d>& points) const
{
    // Report Eq. (9): this is the discrete divergence-theorem check on one
    // source-target overlap polygon.
    double flux = 0.0;
    for (std::size_t i = 0; i < points.size(); ++i) {
        flux += edge_flux(coeffs, points[i], points[(i + 1) % points.size()]);
    }
    return flux;
}

Eigen::Vector3d MimeticInterpolator::cell_integral(const moab::EntityHandle polygon) const
{
    const LocalPolygon poly = local_polygon(mb_, polygon, options_);
    const ReconstructionCoeffs coeffs = read_coefficients(mb_, tag_coeffs_, polygon);
    return reconstruction_integral(options_, poly, coeffs, absolute_points(poly));
}

Eigen::Vector3d MimeticInterpolator::cell_average(const moab::EntityHandle polygon) const
{
    const LocalPolygon poly = local_polygon(mb_, polygon, options_);
    const Eigen::Vector3d integral = cell_integral(polygon);
    const double area = is_spherical() ? poly.spherical_area : poly.area;
    if (area <= options_.geometry_tolerance) {
        throw std::runtime_error("Degenerate polygon area in cell_average");
    }
    return integral / area;
}

std::vector<double> MimeticInterpolator::transfer_to_target_polygon_edges(const moab::EntityHandle source_polygon,
                                                                          const moab::EntityHandle target_polygon)
{
    // Single-source-cell target-edge transfer used by the patch test. General
    // nonmatching meshes use clipped overlap polygons in the tests instead.
    const ReconstructionCoeffs coeffs = read_coefficients(mb_, tag_coeffs_, source_polygon);

    const LocalPolygon source = local_polygon(mb_, source_polygon, options_);
    const LocalPolygon target_absolute = local_polygon(mb_, target_polygon, options_);

    std::vector<Eigen::Vector2d> target_points_in_source_frame;
    target_points_in_source_frame.reserve(target_absolute.points.size());
    if (is_spherical()) {
        const GnomonicFrame frame{source.n, source.e_x, source.e_y, options_.radius};
        for (const Eigen::Vector3d& p3d : target_absolute.points_3d) {
            target_points_in_source_frame.push_back(project_gnomonic(p3d, frame) - source.centroid);
        }
    } else {
        for (const Eigen::Vector2d& target_relative : target_absolute.points) {
            target_points_in_source_frame.push_back(target_relative + target_absolute.centroid - source.centroid);
        }
    }

    std::vector<double> target_fluxes;
    target_fluxes.reserve(target_points_in_source_frame.size());
    for (std::size_t i = 0; i < target_points_in_source_frame.size(); ++i) {
        const std::size_t j = (i + 1) % target_points_in_source_frame.size();
        const double flux = edge_flux(coeffs, target_points_in_source_frame[i], target_points_in_source_frame[j]);
        target_fluxes.push_back(flux);

        const moab::EntityHandle edge = find_or_create_edge(mb_, target_absolute.vertices[i], target_absolute.vertices[j]);
        directed_target_flux_[std::make_pair(target_polygon, i)] = flux;
        check_moab(mb_.tag_set_data(tag_target_flux_, &edge, 1, &flux), "Failed to write target flux tag");
    }
    return target_fluxes;
}

EdgeTransferResult MimeticInterpolator::transfer_source_to_target_edges(
    const std::vector<moab::EntityHandle>& source_polygons,
    const std::vector<moab::EntityHandle>& target_polygons)
{
    struct SourceCache {
        moab::EntityHandle polygon;
        LocalPolygon local;
        std::vector<Eigen::Vector2d> absolute_points;
        ReconstructionCoeffs coeffs;
        GnomonicFrame frame;
        PolygonSearchGeometry search;
    };

    struct TargetCache {
        moab::EntityHandle polygon;
        LocalPolygon local;
        std::vector<LocalEdge> edges;
        PolygonSearchGeometry search;
    };

    std::vector<SourceCache> sources;
    sources.reserve(source_polygons.size());
    for (const moab::EntityHandle source_polygon : source_polygons) {
        const LocalPolygon local = local_polygon(mb_, source_polygon, options_);
        sources.push_back(SourceCache{
            source_polygon,
            local,
            absolute_points(local),
            read_coefficients(mb_, tag_coeffs_, source_polygon),
            GnomonicFrame{local.n, local.e_x, local.e_y, options_.radius},
            polygon_search_geometry(local, options_),
        });
    }

    std::vector<TargetCache> targets;
    targets.reserve(target_polygons.size());
    for (const moab::EntityHandle target_polygon : target_polygons) {
        const LocalPolygon local = local_polygon(mb_, target_polygon, options_);
        targets.push_back(TargetCache{
            target_polygon,
            local,
            local_edges(mb_, local),
            polygon_search_geometry(local, options_),
        });
    }

    const SpatialIndex src_index = build_spatial_index_from_sources(sources, is_spherical());

    EdgeTransferResult result;
    std::vector<std::size_t> target_offsets(targets.size() + 1, 0);
    for (std::size_t i = 0; i < targets.size(); ++i) {
        target_offsets[i + 1] = target_offsets[i] + targets[i].edges.size();
    }
    result.target_edges.resize(target_offsets.back());
    result.target_fluxes.assign(target_offsets.back(), 0.0);
    std::vector<std::vector<EdgeTransferContribution>> contribution_buckets(targets.size());
    for (std::size_t target_index = 0; target_index < targets.size(); ++target_index) {
        const TargetCache& target = targets[target_index];
        for (std::size_t edge_index = 0; edge_index < target.edges.size(); ++edge_index) {
            result.target_edges[target_offsets[target_index] + edge_index] =
                DirectedEdgeDof{target.polygon, target.edges[edge_index].handle, edge_index};
        }
    }

#ifdef MIMETIC_ENABLE_OPENMP
#pragma omp parallel for schedule(dynamic, 16) if(targets.size() >= 64)
#endif
    for (int target_index = 0; target_index < static_cast<int>(targets.size()); ++target_index) {
        const TargetCache& target = targets[static_cast<std::size_t>(target_index)];
        std::vector<EdgeTransferContribution>& target_contributions =
            contribution_buckets[static_cast<std::size_t>(target_index)];
        target_contributions.reserve(target.edges.size());

        // Query k-d tree once per target cell (same for all edges of this cell)
        const auto candidates = find_overlap_candidates(src_index,
                                                        target.search.center,
                                                        target.search.radius,
                                                        sources.size());

        for (std::size_t edge_index = 0; edge_index < target.edges.size(); ++edge_index) {
            const LocalEdge& target_edge = target.edges[edge_index];
            const std::size_t target_dof = target_offsets[static_cast<std::size_t>(target_index)] + edge_index;

            const Eigen::Vector3d v_t1 = target.local.points_3d[edge_index];
            const Eigen::Vector3d v_t2 = target.local.points_3d[(edge_index + 1) % target.local.points_3d.size()];

            for (const std::size_t si : candidates) {
                const SourceCache& source = sources[si];

                Eigen::Vector2d target_a;
                Eigen::Vector2d target_b;
                if (is_spherical()) {
                    try {
                        target_a = project_gnomonic(v_t1, source.frame);
                        target_b = project_gnomonic(v_t2, source.frame);
                    } catch (const std::runtime_error&) {
                        continue;
                    }
                } else {
                    target_a = target.local.centroid + target_edge.a;
                    target_b = target.local.centroid + target_edge.b;
                }

                Eigen::Vector2d clipped_a;
                Eigen::Vector2d clipped_b;
                if (!clip_segment_to_convex_polygon(target_a, target_b, source.absolute_points, clipped_a, clipped_b)) {
                    continue;
                }
                if (!target_interior_side_intersects_source(clipped_a, clipped_b,
                                                            source.absolute_points,
                                                            options_.geometry_tolerance)) {
                    continue;
                }

                const Eigen::Vector2d local_a = clipped_a - source.local.centroid;
                const Eigen::Vector2d local_b = clipped_b - source.local.centroid;
                const double flux = edge_flux(source.coeffs, local_a, local_b);
                result.target_fluxes[target_dof] += flux;
                target_contributions.push_back(EdgeTransferContribution{
                    target_dof,
                    source.polygon,
                    clipped_a,
                    clipped_b,
                    flux,
                });
            }
        }
    }

    for (std::size_t target_index = 0; target_index < targets.size(); ++target_index) {
        const TargetCache& target = targets[target_index];
        for (std::size_t edge_index = 0; edge_index < target.edges.size(); ++edge_index) {
            const std::size_t target_dof = target_offsets[target_index] + edge_index;
            directed_target_flux_[std::make_pair(target.polygon, edge_index)] = result.target_fluxes[target_dof];
            check_moab(mb_.tag_set_data(tag_target_flux_, &target.edges[edge_index].handle, 1,
                                        &result.target_fluxes[target_dof]),
                       "Failed to write edge-wise target flux tag");
        }
        result.contributions.insert(result.contributions.end(),
                                    contribution_buckets[target_index].begin(),
                                    contribution_buckets[target_index].end());
    }

    return result;
}

CellAverageTransferResult MimeticInterpolator::transfer_source_to_target_cell_averages(
    const std::vector<moab::EntityHandle>& source_polygons,
    const std::vector<moab::EntityHandle>& target_polygons,
    const CellAverageReductionMode mode)
{
    if (mode != CellAverageReductionMode::Harmonic) {
        throw std::runtime_error("Unsupported cell-average reduction mode");
    }

    struct SourceCache {
        moab::EntityHandle polygon;
        LocalPolygon local;
        std::vector<Eigen::Vector2d> absolute_points;
        ReconstructionCoeffs coeffs;
        GnomonicFrame frame;
        PolygonSearchGeometry search;
    };

    struct TargetCache {
        moab::EntityHandle polygon;
        LocalPolygon local;
        PolygonSearchGeometry search;
    };

    std::vector<SourceCache> sources;
    sources.reserve(source_polygons.size());
    for (const moab::EntityHandle source_polygon : source_polygons) {
        const LocalPolygon local = local_polygon(mb_, source_polygon, options_);
        sources.push_back(SourceCache{
            source_polygon,
            local,
            absolute_points(local),
            read_coefficients(mb_, tag_coeffs_, source_polygon),
            GnomonicFrame{local.n, local.e_x, local.e_y, options_.radius},
            polygon_search_geometry(local, options_),
        });
    }

    const SpatialIndex avg_index = build_spatial_index_from_sources(sources, is_spherical());
    std::vector<TargetCache> targets;
    targets.reserve(target_polygons.size());
    for (const moab::EntityHandle target_polygon : target_polygons) {
        const LocalPolygon local = local_polygon(mb_, target_polygon, options_);
        targets.push_back(TargetCache{
            target_polygon,
            local,
            polygon_search_geometry(local, options_),
        });
    }

    CellAverageTransferResult result;
    result.target_cells = target_polygons;
    result.target_areas.resize(targets.size(), 0.0);
    result.target_integrals.assign(targets.size(), Eigen::Vector3d::Zero());
    result.target_averages.assign(targets.size(), Eigen::Vector3d::Zero());
    std::vector<std::vector<CellAverageContribution>> contribution_buckets(targets.size());

#ifdef MIMETIC_ENABLE_OPENMP
#pragma omp parallel for schedule(dynamic, 16) if(targets.size() >= 64)
#endif
    for (int target_index = 0; target_index < static_cast<int>(targets.size()); ++target_index) {
        const TargetCache& target = targets[static_cast<std::size_t>(target_index)];
        const double target_area = is_spherical() ? target.local.spherical_area : target.local.area;
        if (target_area <= options_.geometry_tolerance) {
            throw std::runtime_error("Degenerate target area in transfer_source_to_target_cell_averages");
        }
        result.target_areas[static_cast<std::size_t>(target_index)] = target_area;

        const auto avg_cands =
            find_overlap_candidates(avg_index, target.search.center, target.search.radius, sources.size());

        for (const std::size_t si : avg_cands) {
            const SourceCache& source = sources[si];
            std::vector<Eigen::Vector2d> target_in_source;
            if (is_spherical()) {
                target_in_source.reserve(target.local.points_3d.size());
                bool valid_projection = true;
                for (const Eigen::Vector3d& p3d : target.local.points_3d) {
                    try {
                        target_in_source.push_back(project_gnomonic(p3d, source.frame));
                    } catch (const std::runtime_error&) {
                        valid_projection = false;
                        break;
                    }
                }
                if (!valid_projection) {
                    continue;
                }
            } else {
                target_in_source = absolute_points(target.local);
            }

            const std::vector<Eigen::Vector2d> overlap =
                convex_polygon_intersection(target_in_source, source.absolute_points, options_.geometry_tolerance);
            const double overlap_chart_area = polygon_area_abs(overlap);
            if (overlap_chart_area <= options_.geometry_tolerance) {
                continue;
            }

            const Eigen::Vector3d overlap_integral = reconstruction_integral(options_, source.local, source.coeffs, overlap);
            double overlap_area = overlap_chart_area;
            if (is_spherical()) {
                overlap_area = integrate_polygon_vector(overlap, [&](const Eigen::Vector2d& xi) {
                    return Eigen::Vector3d(gnomonic_area_scale(xi, source.frame), 0.0, 0.0);
                }).x();
            }

            result.target_integrals[static_cast<std::size_t>(target_index)] += overlap_integral;
            contribution_buckets[static_cast<std::size_t>(target_index)].push_back(CellAverageContribution{
                static_cast<std::size_t>(target_index),
                source.polygon,
                overlap_area,
                overlap_integral,
            });
        }

        result.target_averages[static_cast<std::size_t>(target_index)] =
            result.target_integrals[static_cast<std::size_t>(target_index)] / target_area;
    }

    for (std::size_t target_index = 0; target_index < targets.size(); ++target_index) {
        result.contributions.insert(result.contributions.end(),
                                    contribution_buckets[target_index].begin(),
                                    contribution_buckets[target_index].end());
    }

    return result;
}

SparseEdgeProjection MimeticInterpolator::assemble_edge_projection_operator(
    const std::vector<moab::EntityHandle>& source_polygons,
    const std::vector<moab::EntityHandle>& target_polygons)
{
    struct SourceCache {
        moab::EntityHandle polygon;
        LocalPolygon local;
        std::vector<LocalEdge> edges;
        std::vector<Eigen::Vector2d> absolute_points;
        Eigen::MatrixXd reconstruction;
        std::vector<std::size_t> columns;
    };

    std::vector<SourceCache> sources;
    sources.reserve(source_polygons.size());
    SparseEdgeProjection projection;

    for (const moab::EntityHandle source_polygon : source_polygons) {
        const LocalPolygon local = local_polygon(mb_, source_polygon, options_);
        const std::vector<LocalEdge> edges = local_edges(mb_, local);
        SourceCache cache{
            source_polygon,
            local,
            edges,
            absolute_points(local),
            source_reconstruction_matrix(local, edges, options_),
            std::vector<std::size_t>(),
        };
        cache.columns.reserve(edges.size());
        for (std::size_t i = 0; i < edges.size(); ++i) {
            cache.columns.push_back(projection.source_edges.size());
            projection.source_edges.push_back(DirectedEdgeDof{source_polygon, edges[i].handle, i});
        }
        sources.push_back(cache);
    }

    const SpatialIndex proj_index = build_spatial_index_from_sources(sources, is_spherical());

    std::vector<Eigen::Triplet<double>> triplets;
    for (const moab::EntityHandle target_polygon : target_polygons) {
        const LocalPolygon target = local_polygon(mb_, target_polygon, options_);
        const std::vector<LocalEdge> target_edges = local_edges(mb_, target);
        const PolygonSearchGeometry target_search = polygon_search_geometry(target, options_);
        const auto proj_cands = find_overlap_candidates(
            proj_index, target_search.center, target_search.radius, sources.size());

        for (std::size_t edge_index = 0; edge_index < target_edges.size(); ++edge_index) {
            const LocalEdge& target_edge = target_edges[edge_index];
            const Eigen::Vector3d v_t1 = is_spherical()
                ? target.points_3d[edge_index]
                : Eigen::Vector3d(target.centroid.x() + target_edge.a.x(), target.centroid.y() + target_edge.a.y(), 0.0);
            const Eigen::Vector3d v_t2 = is_spherical()
                ? target.points_3d[(edge_index + 1) % target.points_3d.size()]
                : Eigen::Vector3d(target.centroid.x() + target_edge.b.x(), target.centroid.y() + target_edge.b.y(), 0.0);
            const std::size_t target_dof = projection.target_edges.size();
            projection.target_edges.push_back(DirectedEdgeDof{target_polygon, target_edge.handle, edge_index});

            for (const std::size_t si : proj_cands) {
                const SourceCache& source = sources[si];
                Eigen::Vector2d target_a;
                Eigen::Vector2d target_b;
                if (is_spherical()) {
                    const GnomonicFrame source_frame{source.local.n, source.local.e_x, source.local.e_y, options_.radius};
                    try {
                        target_a = project_gnomonic(v_t1, source_frame);
                        target_b = project_gnomonic(v_t2, source_frame);
                    } catch (const std::runtime_error&) {
                        continue;
                    }
                } else {
                    target_a = target.centroid + target_edge.a;
                    target_b = target.centroid + target_edge.b;
                }

                Eigen::Vector2d clipped_a;
                Eigen::Vector2d clipped_b;
                if (!clip_segment_to_convex_polygon(target_a, target_b, source.absolute_points, clipped_a, clipped_b)) {
                    continue;
                }
                if (!target_interior_side_intersects_source(clipped_a, clipped_b,
                                                            source.absolute_points,
                                                            options_.geometry_tolerance)) {
                    continue;
                }

                const Eigen::Vector2d local_a = clipped_a - source.local.centroid;
                const Eigen::Vector2d local_b = clipped_b - source.local.centroid;
                const int N_h = source.reconstruction.rows() - 1;
                Eigen::MatrixXd evaluation = Eigen::MatrixXd::Zero(1, 1 + N_h);
                
                ReconstructionCoeffs c_div;
                c_div.d = 1.0;
                c_div.harmonic.resize(N_h, 0.0);
                evaluation(0, 0) = edge_flux(c_div, local_a, local_b);

                for (int i = 0; i < N_h; ++i) {
                    ReconstructionCoeffs c_harm;
                    c_harm.d = 0.0;
                    c_harm.harmonic.resize(N_h, 0.0);
                    c_harm.harmonic[i] = 1.0;
                    evaluation(0, 1 + i) = edge_flux(c_harm, local_a, local_b);
                }

                const Eigen::RowVectorXd weights = evaluation * source.reconstruction;
                for (Eigen::Index j = 0; j < weights.cols(); ++j) {
                    const double weight = weights(j);
                    if (std::abs(weight) > 1.0e-14) {
                        triplets.push_back(Eigen::Triplet<double>(
                            static_cast<int>(target_dof), static_cast<int>(source.columns[static_cast<std::size_t>(j)]), weight));
                    }
                }
            }
        }
    }

    projection.matrix.resize(static_cast<int>(projection.target_edges.size()), static_cast<int>(projection.source_edges.size()));
    projection.matrix.setFromTriplets(triplets.begin(), triplets.end());
    projection.matrix.makeCompressed();
    return projection;
}

ConformingEdgeTransferResult MimeticInterpolator::project_target_fluxes_to_hdiv_conforming(
    const std::vector<moab::EntityHandle>& source_polygons,
    const std::vector<moab::EntityHandle>& target_polygons,
    const EdgeTransferResult& raw_transfer)
{
    const std::vector<DirectedTargetEdgeInfo> target_edges =
        build_directed_target_edges(mb_, target_polygons, options_);
    verify_raw_target_order(raw_transfer, target_edges);

    const CollapsedTargetEdges collapse = collapse_target_edges(target_edges, options_);
    const std::vector<double> divergence_rhs =
        target_divergence_rhs(mb_, tag_coeffs_, options_, source_polygons, target_polygons);

    const std::size_t num_unique = collapse.unique_count;
    const std::size_t num_directed = target_edges.size();
    const std::size_t num_cells = target_polygons.size();
    if (num_directed != raw_transfer.target_fluxes.size()) {
        throw std::runtime_error("Raw transfer flux count does not match target edge DOF count");
    }
    if (divergence_rhs.size() != num_cells) {
        throw std::runtime_error("Target divergence right-hand side size mismatch");
    }

    Eigen::VectorXd hdiag = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(num_unique));
    Eigen::VectorXd g = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(num_unique));
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(num_cells),
                                              static_cast<Eigen::Index>(num_unique));
    for (std::size_t i = 0; i < num_directed; ++i) {
        const std::size_t unique = collapse.directed_to_unique[i];
        const double sign = static_cast<double>(collapse.directed_signs[i]);
        hdiag(static_cast<Eigen::Index>(unique)) += 1.0;
        g(static_cast<Eigen::Index>(unique)) += sign * raw_transfer.target_fluxes[i];
        A(static_cast<Eigen::Index>(target_edges[i].cell_index), static_cast<Eigen::Index>(unique)) += sign;
    }

    for (Eigen::Index i = 0; i < hdiag.size(); ++i) {
        if (hdiag(i) <= 0.0) {
            throw std::runtime_error("Degenerate unique target edge encountered in H(div) projection");
        }
    }

    const Eigen::VectorXd hinv = hdiag.cwiseInverse();
    Eigen::MatrixXd AHinv = A;
    for (Eigen::Index j = 0; j < AHinv.cols(); ++j) {
        AHinv.col(j) *= hinv(j);
    }

    Eigen::VectorXd b(static_cast<Eigen::Index>(num_cells));
    for (std::size_t i = 0; i < num_cells; ++i) {
        b(static_cast<Eigen::Index>(i)) = divergence_rhs[i];
    }
    if (options_.mode == GeometryMode::SphericalGnomonic && b.size() > 0) {
        b.array() -= b.mean();
    }

    const Eigen::MatrixXd schur = AHinv * A.transpose();
    const Eigen::VectorXd schur_rhs = AHinv * g - b;
    const Eigen::VectorXd lambda = schur.completeOrthogonalDecomposition().solve(schur_rhs);
    const Eigen::VectorXd unique_fluxes = hinv.cwiseProduct(g - A.transpose() * lambda);

    ConformingEdgeTransferResult result;
    result.target_edges = raw_transfer.target_edges;
    result.target_cells = target_polygons;
    result.target_divergence_integrals.resize(num_cells, 0.0);
    result.target_edge_to_unique = collapse.directed_to_unique;
    result.target_edge_signs = collapse.directed_signs;
    result.unique_edge_fluxes.resize(num_unique, 0.0);
    result.target_fluxes.resize(num_directed, 0.0);

    for (std::size_t i = 0; i < num_unique; ++i) {
        result.unique_edge_fluxes[i] = unique_fluxes(static_cast<Eigen::Index>(i));
    }
    for (std::size_t i = 0; i < num_cells; ++i) {
        result.target_divergence_integrals[i] = b(static_cast<Eigen::Index>(i));
    }
    for (std::size_t i = 0; i < num_directed; ++i) {
        const std::size_t unique = collapse.directed_to_unique[i];
        const double flux = static_cast<double>(collapse.directed_signs[i]) * result.unique_edge_fluxes[unique];
        result.target_fluxes[i] = flux;
        directed_target_flux_[std::make_pair(result.target_edges[i].polygon, result.target_edges[i].local_edge_index)] = flux;
        check_moab(mb_.tag_set_data(tag_target_flux_, &result.target_edges[i].edge, 1, &flux),
                   "Failed to write H(div)-conforming target flux tag");
    }

    return result;
}

PlanarMomentInterpolator::PlanarMomentInterpolator(moab::Core& moab_instance)
    : mb_(moab_instance)
{
}

void PlanarMomentInterpolator::set_geometry_options(const GeometryOptions& options)
{
    if (options.radius <= 0.0) {
        throw std::runtime_error("GeometryOptions::radius must be positive");
    }
    options_ = options;
}

GeometryOptions PlanarMomentInterpolator::geometry_options() const
{
    return options_;
}

void PlanarMomentInterpolator::set_spherical(const bool is_spherical)
{
    options_.mode = is_spherical ? GeometryMode::SphericalGnomonic : GeometryMode::Planar;
}

bool PlanarMomentInterpolator::is_spherical() const
{
    return options_.mode == GeometryMode::SphericalGnomonic;
}

void PlanarMomentInterpolator::set_source_edge_moments(const moab::EntityHandle polygon,
                                                       const std::size_t local_edge_index,
                                                       const std::vector<double>& moments)
{
    directed_source_moments_[std::make_pair(polygon, local_edge_index)] = moments;
}

std::vector<double> PlanarMomentInterpolator::source_edge_moments(const moab::EntityHandle polygon,
                                                                  const std::size_t local_edge_index) const
{
    const auto it = directed_source_moments_.find(std::make_pair(polygon, local_edge_index));
    if (it == directed_source_moments_.end()) {
        throw std::runtime_error("Missing source edge moments");
    }
    return it->second;
}

void PlanarMomentInterpolator::set_source_cell_vector_moments(const moab::EntityHandle polygon,
                                                              const std::vector<Eigen::Vector2d>& moments)
{
    source_cell_vector_moments_[polygon] = moments;
}

std::vector<Eigen::Vector2d> PlanarMomentInterpolator::source_cell_vector_moments(const moab::EntityHandle polygon) const
{
    const auto it = source_cell_vector_moments_.find(polygon);
    if (it == source_cell_vector_moments_.end()) {
        throw std::runtime_error("Missing source cell vector moments");
    }
    return it->second;
}

MomentReconstruction PlanarMomentInterpolator::reconstruct_source_polygon(const moab::EntityHandle polygon,
                                                                          const MomentMethodOptions& options)
{
    if (options.edge_moment_order < 0) {
        throw std::runtime_error("Edge moment order must be non-negative");
    }

    const LocalPolygon poly = local_polygon(mb_, polygon, options_);
    const std::vector<LocalEdge> edges = local_edges(mb_, poly);
    const std::vector<GaussLegendrePoint> quadrature = gauss_legendre_rule(options.quadrature_points);
    const double scale_length = local_length_scale(poly);

    const bool use_surface_metric = options_.metric_weighted && options_.mode == GeometryMode::SphericalGnomonic;

    // Degree elevation: in spherical mode for p >= 3, raise the polynomial
    // degree of the chart-coordinate vector basis by 2 to capture the
    // leading-order rational correction from the Piola mapping (1/|ray|^3).
    // The extra modes are resolved by minimum-energy regularization.
    // For p <= 2 the O(h^{p+1}) asymptotic term is comparable to or larger
    // than the O(h^3) Piola crime, so elevation is unnecessary.
    const int degree_elevation = (use_surface_metric && options.edge_moment_order >= 3) ? 2 : 0;
    const int vector_degree = options.edge_moment_order + degree_elevation;
    const std::vector<VectorBasisTerm>& raw_basis = cached_vector_polynomial_basis(vector_degree);
    const int raw_dim = static_cast<int>(raw_basis.size());
    const int cell_moment_order = resolved_cell_moment_order(static_cast<int>(edges.size()), options);
    const int num_cell_scalar_moments =
        (cell_moment_order >= 0) ? vector_moment_basis_count(cell_moment_order) : 0;
    const int C = static_cast<int>(edges.size()) * (options.edge_moment_order + 1) +
                  2 * num_cell_scalar_moments;
    const GnomonicFrame gram_frame{poly.n, poly.e_x, poly.e_y, options_.radius};

    // Compute the metric-weighted Gram matrix BEFORE building the split basis,
    // so the basis can be orthonormalized against the correct inner product.
    Eigen::MatrixXd G_raw = Eigen::MatrixXd::Zero(raw_dim, raw_dim);
#ifdef MIMETIC_ENABLE_OPENMP
#pragma omp parallel for schedule(static) if(raw_dim >= 24)
#endif
    for (int i = 0; i < raw_dim; ++i) {
        for (int j = i; j < raw_dim; ++j) {
            const double gij = integrate_polygon_scalar_duffy(edges, quadrature, [&](const Eigen::Vector2d& p) {
                const Eigen::Vector2d vi = vector_basis_value(raw_basis[i], p, scale_length);
                const Eigen::Vector2d vj = vector_basis_value(raw_basis[j], p, scale_length);
                if (!use_surface_metric) {
                    return vi.dot(vj);
                }
                const Eigen::Matrix2d hodge = gnomonic_hodge_metric(p + poly.centroid, gram_frame);
                return vi.dot(hodge * vj);
            });
            G_raw(i, j) = gij;
            G_raw(j, i) = gij;
        }
    }

    const SplitMomentBasis split_basis =
        build_split_moment_basis(vector_degree, options.harmonic_degree, raw_basis, scale_length, G_raw);
    const int B = static_cast<int>(split_basis.raw_coordinates.cols());

    Eigen::MatrixXd A_raw = Eigen::MatrixXd::Zero(C, raw_dim);
    Eigen::VectorXd moments = Eigen::VectorXd::Zero(C);
    Eigen::VectorXd row_weights = Eigen::VectorXd::Ones(C);

    int row = 0;
    for (std::size_t edge_index = 0; edge_index < edges.size(); ++edge_index) {
        const LocalEdge& edge = edges[edge_index];
        const std::vector<double> edge_moments = source_edge_moments(polygon, edge_index);
        if (static_cast<int>(edge_moments.size()) != options.edge_moment_order + 1) {
            throw std::runtime_error("Source edge moments do not match requested moment order");
        }
        for (int degree = 0; degree <= options.edge_moment_order; ++degree, ++row) {
            moments(row) = edge_moments[degree];
            row_weights(row) = options.edge_weight;
        }
        for (int col = 0; col < raw_dim; ++col) {
            const std::vector<double> basis_moments =
                basis_edge_moments(raw_basis[col], poly, edge_index, edge, options.edge_moment_order,
                                   quadrature, scale_length, options_);
            for (int degree = 0; degree <= options.edge_moment_order; ++degree) {
                A_raw(row - (options.edge_moment_order + 1) + degree, col) = basis_moments[degree];
            }
        }
    }

    std::vector<std::vector<Eigen::Vector2d>> cached_basis_cell_moments;
    if (cell_moment_order >= 0) {
        cached_basis_cell_moments.resize(static_cast<std::size_t>(raw_dim));
#ifdef MIMETIC_ENABLE_OPENMP
#pragma omp parallel for schedule(static) if(raw_dim >= 24)
#endif
        for (int col = 0; col < raw_dim; ++col) {
            cached_basis_cell_moments[static_cast<std::size_t>(col)] =
                basis_cell_vector_moments(raw_basis[static_cast<std::size_t>(col)],
                                          edges,
                                          cell_moment_order,
                                          quadrature,
                                          scale_length);
        }

        const std::vector<Eigen::Vector2d> cell_moments = source_cell_vector_moments(polygon);
        if (static_cast<int>(cell_moments.size()) != num_cell_scalar_moments) {
            throw std::runtime_error("Source cell vector moments do not match requested cell moment order");
        }
        for (int total_degree = 0, moment_index = 0; total_degree <= cell_moment_order; ++total_degree) {
            for (int a = total_degree; a >= 0; --a, ++moment_index) {
                moments(row) = cell_moments[moment_index].x();
                moments(row + 1) = cell_moments[moment_index].y();
                row_weights(row) = options.cell_weight;
                row_weights(row + 1) = options.cell_weight;
                for (int col = 0; col < raw_dim; ++col) {
                    const Eigen::Vector2d& basis_moment =
                        cached_basis_cell_moments[static_cast<std::size_t>(col)][static_cast<std::size_t>(moment_index)];
                    A_raw(row, col) = basis_moment.x();
                    A_raw(row + 1, col) = basis_moment.y();
                }
                row += 2;
            }
        }
    }

    Eigen::MatrixXd G = split_basis.raw_coordinates.transpose() * G_raw * split_basis.raw_coordinates;
    const Eigen::MatrixXd A = A_raw * split_basis.raw_coordinates;

    Eigen::VectorXd coeffs = Eigen::VectorXd::Zero(B);
    double residual = 0.0;

    if (!options.exact_constraints) {
        std::vector<int> hard_rows;
        std::vector<int> soft_rows;
        hard_rows.reserve(edges.size());
        soft_rows.reserve(std::max(0, C - static_cast<int>(edges.size())));

        int edge_row = 0;
        for (std::size_t edge_index = 0; edge_index < edges.size(); ++edge_index) {
            hard_rows.push_back(edge_row);
            ++edge_row;
            for (int degree = 1; degree <= options.edge_moment_order; ++degree, ++edge_row) {
                soft_rows.push_back(edge_row);
            }
        }
        for (int row_index = edge_row; row_index < C; ++row_index) {
            soft_rows.push_back(row_index);
        }

        const double scale = std::max(1.0, G.diagonal().cwiseAbs().maxCoeff());
        Eigen::MatrixXd H = options.regularization * scale * G;
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(B);

        for (const int row_index : soft_rows) {
            const double w = row_weights(row_index);
            H.noalias() += (w * w) * (A.row(row_index).transpose() * A.row(row_index));
            rhs.noalias() += (w * w) * A.row(row_index).transpose() * moments(row_index);
        }

        if (hard_rows.empty()) {
            coeffs = H.ldlt().solve(rhs);
            residual = 0.0;
            for (const int row_index : soft_rows) {
                residual = std::max(residual,
                                    std::abs(row_weights(row_index) *
                                             (A.row(row_index).dot(coeffs) - moments(row_index))));
            }
        } else {
            Eigen::MatrixXd A_h = Eigen::MatrixXd::Zero(static_cast<int>(hard_rows.size()), B);
            Eigen::VectorXd m_h = Eigen::VectorXd::Zero(static_cast<int>(hard_rows.size()));
            for (std::size_t i = 0; i < hard_rows.size(); ++i) {
                A_h.row(static_cast<int>(i)) = A.row(hard_rows[i]);
                m_h(static_cast<int>(i)) = moments(hard_rows[i]);
            }

            Eigen::MatrixXd KKT = Eigen::MatrixXd::Zero(B + A_h.rows(), B + A_h.rows());
            KKT.block(0, 0, B, B) = H;
            KKT.block(0, B, B, A_h.rows()) = A_h.transpose();
            KKT.block(B, 0, A_h.rows(), B) = A_h;

            Eigen::VectorXd kkt_rhs = Eigen::VectorXd::Zero(B + A_h.rows());
            kkt_rhs.head(B) = rhs;
            kkt_rhs.tail(A_h.rows()) = m_h;

            const Eigen::VectorXd solution = KKT.fullPivLu().solve(kkt_rhs);
            coeffs = solution.head(B);

            residual = 0.0;
            for (const int row_index : hard_rows) {
                residual = std::max(residual, std::abs(A.row(row_index).dot(coeffs) - moments(row_index)));
            }
        }
    } else if (C >= B) {
        coeffs = A.colPivHouseholderQr().solve(moments);
        residual = (A * coeffs - moments).lpNorm<Eigen::Infinity>();
    } else {
        const double scale = std::max(1.0, G.diagonal().cwiseAbs().maxCoeff());
        G.diagonal() = G.diagonal().array() + options.regularization * scale;

        Eigen::MatrixXd KKT = Eigen::MatrixXd::Zero(B + C, B + C);
        KKT.block(0, 0, B, B) = G;
        KKT.block(0, B, B, C) = A.transpose();
        KKT.block(B, 0, C, B) = A;

        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(B + C);
        rhs.segment(B, C) = moments;

        Eigen::FullPivLU<Eigen::MatrixXd> lu(KKT);
        const Eigen::VectorXd solution = lu.solve(rhs);
        coeffs = solution.head(B);
        residual = (A * coeffs - moments).lpNorm<Eigen::Infinity>();
    }

    const double residual_tolerance = options.exact_constraints ? 1.0e-9 : 1.0e9;
    if (!std::isfinite(residual) || residual > residual_tolerance) {
        std::ostringstream oss;
        oss << "High-order edge-moment reconstruction failed to satisfy constraints"
            << " residual=" << residual
            << " rows=" << C
            << " cols=" << B;
        throw std::runtime_error(oss.str());
    }

    MomentReconstruction reconstruction;
    reconstruction.options = options;
    reconstruction.options.cell_moment_order = cell_moment_order;
    reconstruction.vector_polynomial_degree = vector_degree;
    reconstruction.harmonic_degree = (options.harmonic_degree > 0)
        ? std::min(options.harmonic_degree, vector_degree + 1)
        : (vector_degree + 1);
    reconstruction.divergence_mode_count = split_basis.divergence_mode_count;
    reconstruction.harmonic_mode_count = split_basis.harmonic_mode_count;
    reconstruction.bubble_mode_count = split_basis.bubble_mode_count;
    reconstruction.length_scale = scale_length;
    const Eigen::VectorXd raw_coeffs = split_basis.raw_coordinates * coeffs;
    reconstruction.coefficients.assign(raw_coeffs.data(), raw_coeffs.data() + raw_coeffs.size());

    reconstructions_[polygon] = reconstruction;
    return reconstruction;
}

Eigen::Vector2d PlanarMomentInterpolator::velocity(const MomentReconstruction& reconstruction,
                                                   const Eigen::Vector2d& p) const
{
    return moment_velocity_value(reconstruction, p);
}

std::vector<double> PlanarMomentInterpolator::edge_moments(const MomentReconstruction& reconstruction,
                                                           const Eigen::Vector2d& a,
                                                           const Eigen::Vector2d& b,
                                                           const int order) const
{
    const std::vector<GaussLegendrePoint> quadrature = gauss_legendre_rule(reconstruction.options.quadrature_points);
    std::vector<double> moments;
    const Eigen::Vector2d delta = b - a;
    const double length = delta.norm();
    if (length <= kTolerance) {
        moments.assign(static_cast<std::size_t>(order + 1), 0.0);
        return moments;
    }
    const double denom = delta.squaredNorm();
    const Eigen::Vector2d normal(delta.y(), -delta.x());
    accumulate_edge_moment_bundle(a, b, quadrature, order, moments, [&](const Eigen::Vector2d& p) {
            const double t = 2.0 * (p - a).dot(delta) / denom - 1.0;
            return EdgeMomentSample{velocity(reconstruction, p).dot(normal / length), t};
        });
    return moments;
}

EdgeMomentTransferResult PlanarMomentInterpolator::transfer_source_to_target_edge_moments(
    const std::vector<moab::EntityHandle>& source_polygons,
    const std::vector<moab::EntityHandle>& target_polygons,
    const int target_moment_order) const
{
    struct SourceCache {
        moab::EntityHandle polygon;
        LocalPolygon local;
        std::vector<Eigen::Vector2d> absolute_points;
        MomentReconstruction reconstruction;
        GnomonicFrame frame;
        PolygonSearchGeometry search;
    };

    struct TargetCache {
        moab::EntityHandle polygon;
        LocalPolygon local;
        std::vector<LocalEdge> edges;
        PolygonSearchGeometry search;
    };

    std::vector<SourceCache> sources;
    sources.reserve(source_polygons.size());
    for (const moab::EntityHandle source_polygon : source_polygons) {
        const auto reconstruction_it = reconstructions_.find(source_polygon);
        if (reconstruction_it == reconstructions_.end()) {
            throw std::runtime_error("Missing high-order reconstruction for source polygon");
        }
        const LocalPolygon local = local_polygon(mb_, source_polygon, options_);
        sources.push_back(SourceCache{
            source_polygon,
            local,
            absolute_points(local),
            reconstruction_it->second,
            GnomonicFrame{local.n, local.e_x, local.e_y, options_.radius},
            polygon_search_geometry(local, options_),
        });
    }

    std::vector<TargetCache> targets;
    targets.reserve(target_polygons.size());
    for (const moab::EntityHandle target_polygon : target_polygons) {
        const LocalPolygon local = local_polygon(mb_, target_polygon, options_);
        targets.push_back(TargetCache{
            target_polygon,
            local,
            local_edges(mb_, local),
            polygon_search_geometry(local, options_),
        });
    }

    const SpatialIndex ho_src_index = build_spatial_index_from_sources(sources, is_spherical());

    EdgeMomentTransferResult result;
    const std::vector<GaussLegendrePoint> quadrature = gauss_legendre_rule(10);
    std::vector<std::size_t> target_offsets(targets.size() + 1, 0);
    for (std::size_t i = 0; i < targets.size(); ++i) {
        target_offsets[i + 1] = target_offsets[i] + targets[i].edges.size();
    }
    result.target_edges.resize(target_offsets.back());
    result.target_moments.assign(target_offsets.back(),
                                 std::vector<double>(static_cast<std::size_t>(target_moment_order + 1), 0.0));
    for (std::size_t target_index = 0; target_index < targets.size(); ++target_index) {
        const TargetCache& target = targets[target_index];
        for (std::size_t edge_index = 0; edge_index < target.edges.size(); ++edge_index) {
            result.target_edges[target_offsets[target_index] + edge_index] =
                DirectedEdgeDof{target.polygon, target.edges[edge_index].handle, edge_index};
        }
    }

#ifdef MIMETIC_ENABLE_OPENMP
#pragma omp parallel for schedule(dynamic, 16) if(targets.size() >= 64)
#endif
    for (int target_index = 0; target_index < static_cast<int>(targets.size()); ++target_index) {
        const TargetCache& target = targets[static_cast<std::size_t>(target_index)];
        const auto ho_candidates =
            find_overlap_candidates(ho_src_index,
                                    target.search.center,
                                    target.search.radius,
                                    sources.size());

        for (std::size_t edge_index = 0; edge_index < target.edges.size(); ++edge_index) {
            const LocalEdge& target_edge = target.edges[edge_index];
            const std::size_t target_dof = target_offsets[static_cast<std::size_t>(target_index)] + edge_index;
            std::vector<double> contribution_moments;

            const Eigen::Vector3d target_a3 = target.local.points_3d[edge_index].normalized();
            const Eigen::Vector3d target_b3 =
                target.local.points_3d[(edge_index + 1) % target.local.points_3d.size()].normalized();
            const double target_total_angle = (options_.mode == GeometryMode::SphericalGnomonic)
                ? std::acos(clamp_unit(target_a3.dot(target_b3)))
                : 0.0;

            for (const std::size_t si : ho_candidates) {
                const SourceCache& source = sources[si];
                Eigen::Vector2d whole_a = target.local.centroid + target_edge.a;
                Eigen::Vector2d whole_b = target.local.centroid + target_edge.b;
                if (options_.mode == GeometryMode::SphericalGnomonic) {
                    try {
                        whole_a = project_gnomonic(target.local.points_3d[edge_index], source.frame);
                        whole_b = project_gnomonic(
                            target.local.points_3d[(edge_index + 1) % target.local.points_3d.size()],
                            source.frame);
                    } catch (const std::runtime_error&) {
                        continue;
                    }
                }
                const Eigen::Vector2d whole_delta = whole_b - whole_a;
                const double whole_length = whole_delta.norm();
                const double whole_denom = whole_delta.squaredNorm();
                if (whole_length <= kTolerance) {
                    continue;
                }
                const Eigen::Vector2d whole_normal(whole_delta.y(), -whole_delta.x());

                Eigen::Vector2d clipped_a;
                Eigen::Vector2d clipped_b;
                if (!clip_segment_to_convex_polygon(whole_a, whole_b, source.absolute_points, clipped_a, clipped_b)) {
                    continue;
                }
                if (!target_interior_side_intersects_source(clipped_a, clipped_b,
                                                            source.absolute_points,
                                                            kTolerance)) {
                    continue;
                }

                accumulate_edge_moment_bundle(
                    clipped_a, clipped_b, quadrature, target_moment_order, contribution_moments,
                    [&](const Eigen::Vector2d& global_p) {
                        double t = 0.0;
                        if (options_.mode == GeometryMode::SphericalGnomonic && target_total_angle > kTolerance) {
                            const Eigen::Vector3d point3 = inverse_gnomonic(global_p, source.frame).normalized();
                            const double angle = std::acos(clamp_unit(target_a3.dot(point3)));
                            t = 2.0 * (angle / target_total_angle) - 1.0;
                        } else {
                            t = 2.0 * (global_p - whole_a).dot(whole_delta) / whole_denom - 1.0;
                        }
                        const Eigen::Vector2d local_p = global_p - source.local.centroid;
                        return EdgeMomentSample{
                            velocity(source.reconstruction, local_p).dot(whole_normal / whole_length),
                            t,
                        };
                    });
                for (int degree = 0; degree <= target_moment_order; ++degree) {
                    result.target_moments[target_dof][static_cast<std::size_t>(degree)] +=
                        contribution_moments[static_cast<std::size_t>(degree)];
                }
            }
        }
    }

    return result;
}

std::map<moab::EntityHandle, std::vector<Eigen::Vector2d>>
PlanarMomentInterpolator::transfer_source_to_target_cell_moments(
    const std::vector<moab::EntityHandle>& source_polygons,
    const std::vector<moab::EntityHandle>& target_polygons,
    const int cell_moment_order) const
{
    struct SourceCache {
        moab::EntityHandle polygon;
        LocalPolygon local;
        std::vector<Eigen::Vector2d> absolute_points;
        MomentReconstruction reconstruction;
        GnomonicFrame frame;
        PolygonSearchGeometry search;
    };

    struct TargetCache {
        moab::EntityHandle polygon;
        LocalPolygon local;
        GnomonicFrame frame;
        PolygonSearchGeometry search;
    };

    std::vector<SourceCache> sources;
    sources.reserve(source_polygons.size());
    for (const moab::EntityHandle source_polygon : source_polygons) {
        const auto it = reconstructions_.find(source_polygon);
        if (it == reconstructions_.end()) {
            throw std::runtime_error("Missing reconstruction for cell moment transfer");
        }
        const LocalPolygon local = local_polygon(mb_, source_polygon, options_);
        sources.push_back(SourceCache{
            source_polygon,
            local,
            absolute_points(local),
            it->second,
            GnomonicFrame{local.n, local.e_x, local.e_y, options_.radius},
            polygon_search_geometry(local, options_),
        });
    }

    const SpatialIndex cm_src_index = build_spatial_index_from_sources(
        sources, options_.mode == GeometryMode::SphericalGnomonic);

    const std::vector<GaussLegendrePoint> quadrature =
        gauss_legendre_rule(std::max(10, 2 * std::max(1, cell_moment_order + 1)));
    std::map<moab::EntityHandle, std::vector<Eigen::Vector2d>> result;
    std::vector<TargetCache> targets;
    targets.reserve(target_polygons.size());
    for (const moab::EntityHandle target_polygon : target_polygons) {
        const LocalPolygon local = local_polygon(mb_, target_polygon, options_);
        targets.push_back(TargetCache{
            target_polygon,
            local,
            GnomonicFrame{local.n, local.e_x, local.e_y, options_.radius},
            polygon_search_geometry(local, options_),
        });
    }

    std::vector<std::vector<Eigen::Vector2d>> all_target_moments(targets.size());

#ifdef MIMETIC_ENABLE_OPENMP
#pragma omp parallel for schedule(dynamic, 16) if(targets.size() >= 64)
#endif
    for (int target_index = 0; target_index < static_cast<int>(targets.size()); ++target_index) {
        const TargetCache& target = targets[static_cast<std::size_t>(target_index)];

        // Initialize cell moments to zero
        int n_cm = 0;
        for (int td = 0; td <= cell_moment_order; ++td) n_cm += td + 1;
        std::vector<Eigen::Vector2d> moments(n_cm, Eigen::Vector2d::Zero());
        std::vector<Eigen::Vector2d> target_in_source;
        if (options_.mode == GeometryMode::SphericalGnomonic) {
            target_in_source.reserve(target.local.points_3d.size());
        }

        const auto cm_candidates = find_overlap_candidates(
            cm_src_index, target.search.center, target.search.radius, sources.size());

        for (const std::size_t si : cm_candidates) {
            const SourceCache& source = sources[si];
            // Compute source-target overlap in the source chart
            if (options_.mode == GeometryMode::SphericalGnomonic) {
                target_in_source.clear();
                bool valid = true;
                for (const Eigen::Vector3d& p3d : target.local.points_3d) {
                    try {
                        target_in_source.push_back(project_gnomonic(p3d, source.frame));
                    } catch (...) { valid = false; break; }
                }
                if (!valid) continue;
            } else {
                target_in_source = absolute_points(target.local);
            }

            std::vector<Eigen::Vector2d> overlap =
                convex_polygon_intersection(target_in_source, source.absolute_points, options_.geometry_tolerance);
            if (overlap.size() < 3 || std::abs(signed_area(overlap)) <= options_.geometry_tolerance)
                continue;

            // Integrate the source reconstruction times target-local monomials
            // over the overlap region. The reconstruction is evaluated in source
            // chart coords. The monomials x^a y^b must be in the TARGET cell's
            // centroid-relative local frame. For spherical, this means lifting each
            // source-chart point to the sphere and re-projecting into the target chart.
            if (signed_area(overlap) < 0.0) std::reverse(overlap.begin(), overlap.end());
            const Eigen::Vector2d oc = polygon_centroid(overlap);

            // Helper: evaluate reconstruction and convert to target chart frame
            auto eval_in_target_frame = [&](const Eigen::Vector2d& p_src) -> std::pair<Eigen::Vector2d, Eigen::Vector2d> {
                // Source-chart velocity
                const Eigen::Vector2d src_vel = moment_velocity_value(source.reconstruction, p_src - source.local.centroid);
                Eigen::Vector2d tgt_vel;
                Eigen::Vector2d tgt_local;
                if (options_.mode == GeometryMode::SphericalGnomonic) {
                    // Lift to sphere, convert to target chart
                    const Eigen::Vector3d surface_vel = lift_contravariant_piola(src_vel, p_src, source.frame);
                    const Eigen::Vector3d sphere_pt = inverse_gnomonic(p_src, source.frame);
                    const Eigen::Vector2d tgt_xi = project_gnomonic(sphere_pt, target.frame);
                    tgt_vel = pullback_contravariant_piola(surface_vel, tgt_xi, target.frame);
                    tgt_local = tgt_xi - target.local.centroid;
                } else {
                    tgt_vel = src_vel;
                    tgt_local = p_src - target.local.centroid;
                }
                return {tgt_vel, tgt_local};
            };

            for (std::size_t k = 0; k < overlap.size(); ++k) {
                const Eigen::Vector2d& a = oc;
                const Eigen::Vector2d& b = overlap[k];
                const Eigen::Vector2d& c = overlap[(k + 1) % overlap.size()];
                const Eigen::Vector2d ab = b - a;
                const Eigen::Vector2d bc = c - b;
                const double det_j = std::abs(ab.x() * bc.y() - ab.y() * bc.x());
                for (const GaussLegendrePoint& qu : quadrature) {
                    const double u = 0.5 * (qu.x + 1.0);
                    const double wu = 0.5 * qu.w;
                    for (const GaussLegendrePoint& qv : quadrature) {
                        const double v = 0.5 * (qv.x + 1.0);
                        const double wv = 0.5 * qv.w;
                        const Eigen::Vector2d p_src = a + u * ab + (u * v) * bc;
                        const auto eval = eval_in_target_frame(p_src);
                        const double weight = det_j * wu * wv * u;
                        accumulate_vector_cell_moments(eval.first, eval.second, cell_moment_order, weight, moments);
                    }
                }
            }
        }

        all_target_moments[static_cast<std::size_t>(target_index)] = std::move(moments);
    }

    for (std::size_t target_index = 0; target_index < targets.size(); ++target_index) {
        result[targets[target_index].polygon] = std::move(all_target_moments[target_index]);
    }

    return result;
}

ConformingEdgeMomentTransferResult PlanarMomentInterpolator::project_target_edge_moments_to_hdiv_conforming(
    const std::vector<moab::EntityHandle>& source_polygons,
    const std::vector<moab::EntityHandle>& target_polygons,
    const EdgeMomentTransferResult& raw_transfer) const
{
    const std::vector<DirectedTargetEdgeInfo> target_edges =
        build_directed_target_edges(mb_, target_polygons, options_);
    verify_raw_target_order(raw_transfer, target_edges);

    const CollapsedTargetEdges collapse = collapse_target_edges(target_edges, options_);
    const std::vector<double> divergence_rhs =
        target_divergence_rhs(mb_, options_, reconstructions_, source_polygons, target_polygons);

    const std::size_t num_unique = collapse.unique_count;
    const std::size_t num_directed = target_edges.size();
    const std::size_t num_cells = target_polygons.size();
    if (num_directed != raw_transfer.target_moments.size()) {
        throw std::runtime_error("Raw high-order transfer moment count does not match target edge DOF count");
    }
    if (divergence_rhs.size() != num_cells) {
        throw std::runtime_error("High-order target divergence right-hand side size mismatch");
    }

    const std::size_t num_moments = raw_transfer.target_moments.empty() ? 0 : raw_transfer.target_moments.front().size();
    ConformingEdgeMomentTransferResult result;
    result.target_edges = raw_transfer.target_edges;
    result.target_cells = target_polygons;
    result.target_divergence_integrals = divergence_rhs;
    result.target_edge_to_unique = collapse.directed_to_unique;
    result.target_edge_orientations = collapse.directed_signs;
    result.unique_edge_moments.assign(num_unique, std::vector<double>(num_moments, 0.0));
    result.target_moments.assign(num_directed, std::vector<double>(num_moments, 0.0));

    if (num_moments == 0) {
        return result;
    }

    Eigen::VectorXd hdiag = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(num_unique));
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(num_cells),
                                              static_cast<Eigen::Index>(num_unique));
    for (std::size_t i = 0; i < num_directed; ++i) {
        const std::size_t unique = collapse.directed_to_unique[i];
        hdiag(static_cast<Eigen::Index>(unique)) += 1.0;
        A(static_cast<Eigen::Index>(target_edges[i].cell_index), static_cast<Eigen::Index>(unique)) +=
            target_edge_moment_orientation_factor(collapse.directed_signs[i], 0);
    }

    for (Eigen::Index i = 0; i < hdiag.size(); ++i) {
        if (hdiag(i) <= 0.0) {
            throw std::runtime_error("Degenerate unique target edge encountered in high-order H(div) projection");
        }
    }

    const Eigen::VectorXd hinv = hdiag.cwiseInverse();
    Eigen::MatrixXd AHinv = A;
    for (Eigen::Index j = 0; j < AHinv.cols(); ++j) {
        AHinv.col(j) *= hinv(j);
    }

    Eigen::VectorXd b(static_cast<Eigen::Index>(num_cells));
    for (std::size_t i = 0; i < num_cells; ++i) {
        b(static_cast<Eigen::Index>(i)) = divergence_rhs[i];
    }
    if (options_.mode == GeometryMode::SphericalGnomonic && b.size() > 0) {
        b.array() -= b.mean();
    }
    for (std::size_t i = 0; i < num_cells; ++i) {
        result.target_divergence_integrals[i] = b(static_cast<Eigen::Index>(i));
    }

    // Coupled solve: degree 0 uses Schur complement with cell divergence
    // constraints. Higher degrees use a coupled constrained LS that stays
    // close to the raw weighted average while penalizing inter-cell moment
    // jumps through a graph Laplacian regularization.
    //
    // For each degree d >= 1, the regularization penalizes
    //   sum_{shared edges g} (x_g^{(d)} - raw_g^{(d)})^2
    // while respecting the sign convention for odd moments.

    for (std::size_t degree = 0; degree < num_moments; ++degree) {
        Eigen::VectorXd g = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(num_unique));
        for (std::size_t i = 0; i < num_directed; ++i) {
            const std::size_t unique = collapse.directed_to_unique[i];
            const double factor = target_edge_moment_orientation_factor(collapse.directed_signs[i],
                                                                        static_cast<int>(degree));
            g(static_cast<Eigen::Index>(unique)) += factor * raw_transfer.target_moments[i][degree];
        }

        Eigen::VectorXd unique_moments = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(num_unique));
        if (degree == 0) {
            const Eigen::MatrixXd schur = AHinv * A.transpose();
            const Eigen::VectorXd schur_rhs = AHinv * g - b;
            const Eigen::VectorXd lambda = schur.completeOrthogonalDecomposition().solve(schur_rhs);
            unique_moments = hinv.cwiseProduct(g - A.transpose() * lambda);
        } else {
            // Higher moments: weighted average (same as before), which is
            // the minimum-norm solution that produces a unique flux per
            // geometric edge. The degree-0 divergence constraint already
            // ensures cell-level conservation; higher moments do not have
            // an analogous cell-level constraint to enforce.
            unique_moments = hinv.cwiseProduct(g);
        }

        for (std::size_t i = 0; i < num_unique; ++i) {
            result.unique_edge_moments[i][degree] = unique_moments(static_cast<Eigen::Index>(i));
        }
        for (std::size_t i = 0; i < num_directed; ++i) {
            const std::size_t unique = collapse.directed_to_unique[i];
            const double factor = target_edge_moment_orientation_factor(collapse.directed_signs[i],
                                                                        static_cast<int>(degree));
            result.target_moments[i][degree] = factor * result.unique_edge_moments[unique][degree];
        }
    }

    return result;
}

}  // namespace mimetic
