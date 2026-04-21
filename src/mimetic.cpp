#include "mimetic/mimetic.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <fstream>
#include <stdexcept>

namespace mimetic {

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

double clamp_unit(const double value)
{
    return std::max(-1.0, std::min(1.0, value));
}

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
        return integrate_polygon_vector(polygon_abs, [&](const Eigen::Vector2d& xi) {
            const Eigen::Vector2d chart_velocity = harmonic_velocity_value(coeffs, xi - source.centroid);
            const Eigen::Vector3d surface_velocity = lift_contravariant_piola(chart_velocity, xi, frame);
            return gnomonic_area_scale(xi, frame) * surface_velocity;
        });
    }

    return integrate_polygon_vector(polygon_abs, [&](const Eigen::Vector2d& p) {
        const Eigen::Vector2d planar_velocity = harmonic_velocity_value(coeffs, p - source.centroid);
        return Eigen::Vector3d(planar_velocity.x(), planar_velocity.y(), 0.0);
    });
}

Eigen::MatrixXd source_reconstruction_matrix(const LocalPolygon& poly,
                                             const std::vector<LocalEdge>& edges,
                                             const GeometryOptions& options)
{
    const int N = static_cast<int>(edges.size());
    const int K_max = N / 2;
    const int N_h = 2 * K_max;
    const int S = N_h + N - 1;
    const bool use_metric_weight = options.metric_weighted && options.mode == GeometryMode::SphericalGnomonic;
    const GnomonicFrame frame{poly.n, poly.e_x, poly.e_y, options.radius};

    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(N_h, N_h);
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(N_h, N);
    const Eigen::Vector2d origin(0.0, 0.0);

    for (int i = 0; i < N_h; ++i) {
        int ki = (i / 2) + 1;
        bool is_Q_i = (i % 2 == 1);
        for (int j = 0; j < N_h; ++j) {
            int kj = (j / 2) + 1;
            bool is_Q_j = (j % 2 == 1);

            double val = 0.0;
            for (const LocalEdge& edge : edges) {
                val += integrate_triangle_scalar(origin, edge.a, edge.b, [&](const Eigen::Vector2d& p) {
                    double Pi, Qi, Pj, Qj;
                    Eigen::Vector2d gPi, gQi, gPj, gQj;
                    eval_harmonic_basis(ki, p, Pi, Qi, gPi, gQi);
                    eval_harmonic_basis(kj, p, Pj, Qj, gPj, gQj);
                    Eigen::Vector2d gi = is_Q_i ? gQi : gPi;
                    Eigen::Vector2d gj = is_Q_j ? gQj : gPj;
                    const double weight = use_metric_weight ? gnomonic_area_scale(p + poly.centroid, frame) : 1.0;
                    return gi.dot(gj) * weight;
                });
            }
            V(i, j) = val;
        }
        
        double cell_basis_integral = 0.0;
        double div_integral = 0.0;
        for (const LocalEdge& edge : edges) {
            cell_basis_integral += integrate_triangle_scalar(origin, edge.a, edge.b, [&](const Eigen::Vector2d& p) {
                double P, Q; Eigen::Vector2d gP, gQ;
                eval_harmonic_basis(ki, p, P, Q, gP, gQ);
                const double weight = use_metric_weight ? gnomonic_area_scale(p + poly.centroid, frame) : 1.0;
                return (is_Q_i ? Q : P) * weight;
            });
            div_integral += integrate_triangle_scalar(origin, edge.a, edge.b, [&](const Eigen::Vector2d& p) {
                double P, Q; Eigen::Vector2d gP, gQ;
                eval_harmonic_basis(ki, p, P, Q, gP, gQ);
                const double weight = use_metric_weight ? gnomonic_area_scale(p + poly.centroid, frame) : 1.0;
                return p.dot(is_Q_i ? gQ : gP) * weight;
            });
        }
        const double cell_basis_average = cell_basis_integral / poly.area;
        if (i >= 4) {
            V(i, i) += 1.0e2 * poly.area;
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
    const bool use_metric_weight = options_.metric_weighted && is_spherical();
    const GnomonicFrame frame{poly.n, poly.e_x, poly.e_y, poly.centroid_3d.norm()};

    Eigen::VectorXd source_flux(N);
    for (int i = 0; i < N; ++i) {
        source_flux(i) = source_edge_flux(polygon, static_cast<std::size_t>(i), edges[i].handle);
    }

    const double divergence = source_flux.sum() / poly.area;

    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(N_h, N_h);
    Eigen::VectorXd M = Eigen::VectorXd::Zero(N_h);
    const Eigen::Vector2d origin(0.0, 0.0);

    for (int i = 0; i < N_h; ++i) {
        int ki = (i / 2) + 1;
        bool is_Q_i = (i % 2 == 1);
        for (int j = 0; j < N_h; ++j) {
            int kj = (j / 2) + 1;
            bool is_Q_j = (j % 2 == 1);

            double val = 0.0;
            for (const LocalEdge& edge : edges) {
                val += integrate_triangle_scalar(origin, edge.a, edge.b, [&](const Eigen::Vector2d& p) {
                    double Pi, Qi, Pj, Qj;
                    Eigen::Vector2d gPi, gQi, gPj, gQj;
                    eval_harmonic_basis(ki, p, Pi, Qi, gPi, gQi);
                    eval_harmonic_basis(kj, p, Pj, Qj, gPj, gQj);
                    Eigen::Vector2d gi = is_Q_i ? gQi : gPi;
                    Eigen::Vector2d gj = is_Q_j ? gQj : gPj;
                    const double weight = use_metric_weight ? gnomonic_area_scale(p + poly.centroid, frame) : 1.0;
                    return gi.dot(gj) * weight;
                });
            }
            V(i, j) = val;
        }

        double cell_basis_integral = 0.0;
        double div_integral = 0.0;
        for (const LocalEdge& edge : edges) {
            cell_basis_integral += integrate_triangle_scalar(origin, edge.a, edge.b, [&](const Eigen::Vector2d& p) {
                double P, Q; Eigen::Vector2d gP, gQ;
                eval_harmonic_basis(ki, p, P, Q, gP, gQ);
                const double weight = use_metric_weight ? gnomonic_area_scale(p + poly.centroid, frame) : 1.0;
                return (is_Q_i ? Q : P) * weight;
            });
            div_integral += integrate_triangle_scalar(origin, edge.a, edge.b, [&](const Eigen::Vector2d& p) {
                double P, Q; Eigen::Vector2d gP, gQ;
                eval_harmonic_basis(ki, p, P, Q, gP, gQ);
                const double weight = use_metric_weight ? gnomonic_area_scale(p + poly.centroid, frame) : 1.0;
                return p.dot(is_Q_i ? gQ : gP) * weight;
            });
        }
        const double cell_basis_average = cell_basis_integral / poly.area;
        if (i >= 4) {
            V(i, i) += 1.0e2 * poly.area;
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
        });
    }

    EdgeTransferResult result;

    for (const moab::EntityHandle target_polygon : target_polygons) {
        const LocalPolygon target = local_polygon(mb_, target_polygon, options_);
        const std::vector<LocalEdge> target_edges = local_edges(mb_, target);

        for (std::size_t edge_index = 0; edge_index < target_edges.size(); ++edge_index) {
            const LocalEdge& target_edge = target_edges[edge_index];
            const std::size_t target_dof = result.target_edges.size();
            result.target_edges.push_back(DirectedEdgeDof{target_polygon, target_edge.handle, edge_index});
            result.target_fluxes.push_back(0.0);

            Eigen::Vector3d v_t1 = target.points_3d[edge_index];
            Eigen::Vector3d v_t2 = target.points_3d[(edge_index + 1) % target.points_3d.size()];

            for (const SourceCache& source : sources) {
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
                const double flux = edge_flux(source.coeffs, local_a, local_b);
                result.target_fluxes[target_dof] += flux;
                result.contributions.push_back(EdgeTransferContribution{
                    target_dof,
                    source.polygon,
                    clipped_a,
                    clipped_b,
                    flux,
                });
            }

            directed_target_flux_[std::make_pair(target_polygon, edge_index)] = result.target_fluxes[target_dof];
            check_moab(mb_.tag_set_data(tag_target_flux_, &target_edge.handle, 1, &result.target_fluxes[target_dof]),
                       "Failed to write edge-wise target flux tag");
        }
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
        });
    }

    CellAverageTransferResult result;
    result.target_cells = target_polygons;
    result.target_areas.reserve(target_polygons.size());
    result.target_integrals.assign(target_polygons.size(), Eigen::Vector3d::Zero());
    result.target_averages.assign(target_polygons.size(), Eigen::Vector3d::Zero());

    for (std::size_t target_index = 0; target_index < target_polygons.size(); ++target_index) {
        const moab::EntityHandle target_polygon = target_polygons[target_index];
        const LocalPolygon target = local_polygon(mb_, target_polygon, options_);
        const double target_area = is_spherical() ? target.spherical_area : target.area;
        if (target_area <= options_.geometry_tolerance) {
            throw std::runtime_error("Degenerate target area in transfer_source_to_target_cell_averages");
        }
        result.target_areas.push_back(target_area);

        for (const SourceCache& source : sources) {
            std::vector<Eigen::Vector2d> target_in_source;
            if (is_spherical()) {
                const GnomonicFrame source_frame{source.local.n, source.local.e_x, source.local.e_y, options_.radius};
                target_in_source.reserve(target.points_3d.size());
                bool valid_projection = true;
                for (const Eigen::Vector3d& p3d : target.points_3d) {
                    try {
                        target_in_source.push_back(project_gnomonic(p3d, source_frame));
                    } catch (const std::runtime_error&) {
                        valid_projection = false;
                        break;
                    }
                }
                if (!valid_projection) {
                    continue;
                }
            } else {
                target_in_source = absolute_points(target);
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
                const GnomonicFrame source_frame{source.local.n, source.local.e_x, source.local.e_y, options_.radius};
                overlap_area = integrate_polygon_vector(overlap, [&](const Eigen::Vector2d& xi) {
                    return Eigen::Vector3d(gnomonic_area_scale(xi, source_frame), 0.0, 0.0);
                }).x();
            }

            result.target_integrals[target_index] += overlap_integral;
            result.contributions.push_back(CellAverageContribution{
                target_index,
                source.polygon,
                overlap_area,
                overlap_integral,
            });
        }

        result.target_averages[target_index] = result.target_integrals[target_index] / target_area;
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

    std::vector<Eigen::Triplet<double>> triplets;
    for (const moab::EntityHandle target_polygon : target_polygons) {
        const LocalPolygon target = local_polygon(mb_, target_polygon, options_);
        const std::vector<LocalEdge> target_edges = local_edges(mb_, target);

        for (std::size_t edge_index = 0; edge_index < target_edges.size(); ++edge_index) {
            const LocalEdge& target_edge = target_edges[edge_index];
            const Eigen::Vector3d v_t1 = is_spherical() ? target.points_3d[edge_index] : Eigen::Vector3d::Zero();
            const Eigen::Vector3d v_t2 = is_spherical() ? target.points_3d[(edge_index + 1) % target.points_3d.size()]
                                                        : Eigen::Vector3d::Zero();
            const std::size_t target_dof = projection.target_edges.size();
            projection.target_edges.push_back(DirectedEdgeDof{target_polygon, target_edge.handle, edge_index});

            for (const SourceCache& source : sources) {
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

}  // namespace mimetic
