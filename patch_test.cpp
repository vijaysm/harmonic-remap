#include <Eigen/Dense>
#include <moab/Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::VectorXd;

namespace {

constexpr double kTolerance = 1.0e-12;

void check_moab(const moab::ErrorCode code, const std::string& message)
{
    if (code != moab::MB_SUCCESS) {
        throw std::runtime_error(message + " (MOAB error " + std::to_string(static_cast<int>(code)) + ")");
    }
}

double p1(const Vector2d& p) { return p.x(); }
double q1(const Vector2d& p) { return p.y(); }
double p2(const Vector2d& p) { return p.x() * p.x() - p.y() * p.y(); }
double q2(const Vector2d& p) { return 2.0 * p.x() * p.y(); }

Vector2d grad_p1(const Vector2d&) { return Vector2d(1.0, 0.0); }
Vector2d grad_q1(const Vector2d&) { return Vector2d(0.0, 1.0); }
Vector2d grad_p2(const Vector2d& p) { return Vector2d(2.0 * p.x(), -2.0 * p.y()); }
Vector2d grad_q2(const Vector2d& p) { return Vector2d(2.0 * p.y(), 2.0 * p.x()); }

Vector2d exact_u(const Vector2d&)
{
    return Vector2d(1.0, 1.0);
}

template <typename Func>
double integrate_edge_scalar(const Vector2d& a, const Vector2d& b, const Func& func)
{
    const double xi = 1.0 / std::sqrt(3.0);
    const double length = (b - a).norm();
    const Vector2d midpoint = 0.5 * (a + b);
    const Vector2d half_delta = 0.5 * (b - a);
    return 0.5 * length * (func(midpoint - xi * half_delta) + func(midpoint + xi * half_delta));
}

template <typename Func>
double integrate_triangle_scalar(const Vector2d& a, const Vector2d& b, const Vector2d& c, const Func& func)
{
    const double signed_twice_area = (b - a).x() * (c - a).y() - (b - a).y() * (c - a).x();
    const double area = 0.5 * std::abs(signed_twice_area);
    const std::array<std::array<double, 3>, 3> bary = {{
        {{1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0}},
        {{1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0}},
        {{2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0}},
    }};

    double sum = 0.0;
    for (const auto& w : bary) {
        const Vector2d p = w[0] * a + w[1] * b + w[2] * c;
        sum += func(p);
    }
    return area * sum / 3.0;
}

struct ReconstructionCoeffs {
    double c;
    double d;
    double a1;
    double b1;
    double a2;
    double b2;
};

struct LocalPolygon {
    std::vector<moab::EntityHandle> vertices;
    std::vector<Vector2d> points;
    Vector2d centroid;
    double area;
};

struct LocalEdge {
    moab::EntityHandle handle;
    Vector2d a;
    Vector2d b;
    Vector2d outward_normal;
    double length;
};

double signed_area(const std::vector<Vector2d>& points)
{
    double area2 = 0.0;
    for (std::size_t i = 0; i < points.size(); ++i) {
        const Vector2d& a = points[i];
        const Vector2d& b = points[(i + 1) % points.size()];
        area2 += a.x() * b.y() - b.x() * a.y();
    }
    return 0.5 * area2;
}

Vector2d polygon_centroid(const std::vector<Vector2d>& points)
{
    double area2 = 0.0;
    Vector2d weighted(0.0, 0.0);
    for (std::size_t i = 0; i < points.size(); ++i) {
        const Vector2d& a = points[i];
        const Vector2d& b = points[(i + 1) % points.size()];
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

LocalPolygon local_polygon(moab::Core& mb, const moab::EntityHandle polygon)
{
    const moab::EntityHandle* conn = nullptr;
    int num_vertices = 0;
    check_moab(mb.get_connectivity(polygon, conn, num_vertices), "Failed to get polygon connectivity");
    if (num_vertices < 3) {
        throw std::runtime_error("Polygon must have at least three vertices");
    }

    std::vector<moab::EntityHandle> vertices(conn, conn + num_vertices);
    std::vector<Vector2d> absolute_points;
    absolute_points.reserve(vertices.size());

    for (const moab::EntityHandle vertex : vertices) {
        double xyz[3] = {0.0, 0.0, 0.0};
        check_moab(mb.get_coords(&vertex, 1, xyz), "Failed to get vertex coordinates");
        absolute_points.emplace_back(xyz[0], xyz[1]);
    }

    if (signed_area(absolute_points) < 0.0) {
        std::reverse(vertices.begin(), vertices.end());
        std::reverse(absolute_points.begin(), absolute_points.end());
    }

    const Vector2d centroid = polygon_centroid(absolute_points);
    std::vector<Vector2d> relative_points;
    relative_points.reserve(absolute_points.size());
    for (const Vector2d& p : absolute_points) {
        relative_points.push_back(p - centroid);
    }

    return LocalPolygon{vertices, relative_points, centroid, std::abs(signed_area(absolute_points))};
}

std::vector<LocalEdge> local_edges(moab::Core& mb, const LocalPolygon& polygon)
{
    std::vector<LocalEdge> edges;
    edges.reserve(polygon.points.size());
    for (std::size_t i = 0; i < polygon.points.size(); ++i) {
        const std::size_t j = (i + 1) % polygon.points.size();
        const Vector2d a = polygon.points[i];
        const Vector2d b = polygon.points[j];
        const Vector2d delta = b - a;
        const double length = delta.norm();
        if (length < kTolerance) {
            throw std::runtime_error("Degenerate edge length");
        }
        const Vector2d outward(delta.y(), -delta.x());
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

class MimeticInterpolator {
  public:
    explicit MimeticInterpolator(moab::Core& moab_instance) : mb_(moab_instance)
    {
        const double default_flux = 0.0;
        check_moab(mb_.tag_get_handle("SOURCE_FLUX", 1, moab::MB_TYPE_DOUBLE, tag_source_flux_,
                                      moab::MB_TAG_DENSE | moab::MB_TAG_CREAT, &default_flux),
                   "Failed to create SOURCE_FLUX tag");
        check_moab(mb_.tag_get_handle("TARGET_FLUX", 1, moab::MB_TYPE_DOUBLE, tag_target_flux_,
                                      moab::MB_TAG_DENSE | moab::MB_TAG_CREAT, &default_flux),
                   "Failed to create TARGET_FLUX tag");

        const std::array<double, 6> default_coeffs = {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
        check_moab(mb_.tag_get_handle("COEFFS", 6, moab::MB_TYPE_DOUBLE, tag_coeffs_,
                                      moab::MB_TAG_DENSE | moab::MB_TAG_CREAT, default_coeffs.data()),
                   "Failed to create COEFFS tag");
    }

    moab::Tag source_flux_tag() const { return tag_source_flux_; }
    moab::Tag target_flux_tag() const { return tag_target_flux_; }

    ReconstructionCoeffs reconstruct_source_polygon(const moab::EntityHandle polygon)
    {
        const LocalPolygon poly = local_polygon(mb_, polygon);
        const std::vector<LocalEdge> edges = local_edges(mb_, poly);

        VectorXd source_flux(edges.size());
        for (std::size_t i = 0; i < edges.size(); ++i) {
            double flux = 0.0;
            check_moab(mb_.tag_get_data(tag_source_flux_, &edges[i].handle, 1, &flux),
                       "Failed to read source flux tag");
            source_flux(static_cast<Eigen::Index>(i)) = flux;
        }

        const double divergence = source_flux.sum() / poly.area;

        std::array<double (*)(const Vector2d&), 4> basis = {{p1, q1, p2, q2}};
        std::array<Vector2d (*)(const Vector2d&), 4> gradients = {{grad_p1, grad_q1, grad_p2, grad_q2}};

        MatrixXd v = MatrixXd::Zero(4, 4);
        VectorXd rhs = VectorXd::Zero(4);
        const Vector2d origin(0.0, 0.0);

        for (int i = 0; i < 4; ++i) {
            double cell_basis_integral = 0.0;
            double div_integral = 0.0;
            for (std::size_t e = 0; e < edges.size(); ++e) {
                const Vector2d& a = edges[e].a;
                const Vector2d& b = edges[e].b;
                cell_basis_integral += integrate_triangle_scalar(origin, a, b, basis[i]);
                div_integral += integrate_triangle_scalar(origin, a, b, [&](const Vector2d& p) {
                    return p.dot(gradients[i](p));
                });
            }
            const double cell_basis_average = cell_basis_integral / poly.area;
            rhs(i) -= 0.5 * divergence * div_integral;

            for (std::size_t e = 0; e < edges.size(); ++e) {
                const double edge_average =
                    integrate_edge_scalar(edges[e].a, edges[e].b, basis[i]) / edges[e].length;
                rhs(i) += source_flux(static_cast<Eigen::Index>(e)) * (edge_average - cell_basis_average);
            }

            for (int j = 0; j < 4; ++j) {
                for (const LocalEdge& edge : edges) {
                    v(i, j) += integrate_triangle_scalar(origin, edge.a, edge.b, [&](const Vector2d& p) {
                        return gradients[i](p).dot(gradients[j](p));
                    });
                }
            }
        }

        const VectorXd harmonic_coeffs = v.ldlt().solve(rhs);
        ReconstructionCoeffs coeffs = {
            0.0,
            divergence,
            harmonic_coeffs(0),
            harmonic_coeffs(1),
            harmonic_coeffs(2),
            harmonic_coeffs(3),
        };
        check_moab(mb_.tag_set_data(tag_coeffs_, &polygon, 1, &coeffs), "Failed to store reconstruction coefficients");
        return coeffs;
    }

    Vector2d velocity(const ReconstructionCoeffs& coeffs, const Vector2d& p) const
    {
        return 0.5 * coeffs.d * p + coeffs.a1 * grad_p1(p) + coeffs.b1 * grad_q1(p) +
               coeffs.a2 * grad_p2(p) + coeffs.b2 * grad_q2(p);
    }

    double line_integral(const moab::EntityHandle source_polygon, const Vector2d& a, const Vector2d& b) const
    {
        ReconstructionCoeffs coeffs{};
        check_moab(mb_.tag_get_data(tag_coeffs_, &source_polygon, 1, &coeffs),
                   "Failed to read reconstruction coefficients");
        return 0.25 * coeffs.d * (b.squaredNorm() - a.squaredNorm()) + coeffs.a1 * (p1(b) - p1(a)) +
               coeffs.b1 * (q1(b) - q1(a)) + coeffs.a2 * (p2(b) - p2(a)) + coeffs.b2 * (q2(b) - q2(a));
    }

    double edge_flux(const ReconstructionCoeffs& coeffs, const Vector2d& a, const Vector2d& b) const
    {
        const Vector2d delta = b - a;
        const double length = delta.norm();
        const Vector2d outward(delta.y(), -delta.x());
        const Vector2d normal = outward / length;
        return integrate_edge_scalar(a, b, [&](const Vector2d& p) { return velocity(coeffs, p).dot(normal); });
    }

    std::vector<double> transfer_to_target_polygon_edges(const moab::EntityHandle source_polygon,
                                                         const moab::EntityHandle target_polygon)
    {
        ReconstructionCoeffs coeffs{};
        check_moab(mb_.tag_get_data(tag_coeffs_, &source_polygon, 1, &coeffs),
                   "Failed to read source reconstruction coefficients");

        const LocalPolygon source = local_polygon(mb_, source_polygon);
        const LocalPolygon target_absolute = local_polygon(mb_, target_polygon);

        std::vector<Vector2d> target_points_in_source_frame;
        target_points_in_source_frame.reserve(target_absolute.points.size());
        for (const Vector2d& target_relative : target_absolute.points) {
            target_points_in_source_frame.push_back(target_relative + target_absolute.centroid - source.centroid);
        }

        std::vector<double> target_fluxes;
        target_fluxes.reserve(target_points_in_source_frame.size());
        for (std::size_t i = 0; i < target_points_in_source_frame.size(); ++i) {
            const std::size_t j = (i + 1) % target_points_in_source_frame.size();
            const double flux = edge_flux(coeffs, target_points_in_source_frame[i], target_points_in_source_frame[j]);
            target_fluxes.push_back(flux);

            const moab::EntityHandle edge =
                find_or_create_edge(mb_, target_absolute.vertices[i], target_absolute.vertices[j]);
            check_moab(mb_.tag_set_data(tag_target_flux_, &edge, 1, &flux), "Failed to write target flux tag");
        }
        return target_fluxes;
    }

  private:
    moab::Core& mb_;
    moab::Tag tag_source_flux_ = 0;
    moab::Tag tag_target_flux_ = 0;
    moab::Tag tag_coeffs_ = 0;
};

moab::EntityHandle create_quad(moab::Core& mb, const std::array<Vector2d, 4>& points)
{
    std::array<moab::EntityHandle, 4> vertices = {{0, 0, 0, 0}};
    for (std::size_t i = 0; i < points.size(); ++i) {
        const double xyz[3] = {points[i].x(), points[i].y(), 0.0};
        check_moab(mb.create_vertex(xyz, vertices[i]), "Failed to create vertex");
    }

    moab::EntityHandle quad = 0;
    check_moab(mb.create_element(moab::MBQUAD, vertices.data(), static_cast<int>(vertices.size()), quad),
               "Failed to create quad");
    for (std::size_t i = 0; i < vertices.size(); ++i) {
        find_or_create_edge(mb, vertices[i], vertices[(i + 1) % vertices.size()]);
    }
    return quad;
}

void set_source_fluxes_from_exact_field(moab::Core& mb, const moab::Tag source_flux_tag, const moab::EntityHandle polygon)
{
    const LocalPolygon poly = local_polygon(mb, polygon);
    const std::vector<LocalEdge> edges = local_edges(mb, poly);
    for (const LocalEdge& edge : edges) {
        const double flux =
            integrate_edge_scalar(edge.a, edge.b, [&](const Vector2d& p) { return exact_u(p).dot(edge.outward_normal); });
        check_moab(mb.tag_set_data(source_flux_tag, &edge.handle, 1, &flux), "Failed to set source flux");
    }
}

bool near(const double actual, const double expected, const double tolerance, const std::string& label)
{
    const double error = std::abs(actual - expected);
    std::cout << "  " << std::left << std::setw(34) << label << " actual=" << std::right << std::setw(22)
              << std::setprecision(15) << actual << " expected=" << std::setw(22) << expected
              << " error=" << error << "\n";
    return error <= tolerance;
}

}  // namespace

int main()
{
    try {
        std::cout << "--- Mimetic Polygon Patch Test (Level 2) ---\n\n";

        moab::Core mb;
        MimeticInterpolator interpolator(mb);

        const moab::EntityHandle source_quad = create_quad(mb, {{
                                                                    Vector2d(0.0, 0.0),
                                                                    Vector2d(1.0, 0.0),
                                                                    Vector2d(1.0, 1.0),
                                                                    Vector2d(0.0, 1.0),
                                                                }});
        set_source_fluxes_from_exact_field(mb, interpolator.source_flux_tag(), source_quad);

        const ReconstructionCoeffs coeffs = interpolator.reconstruct_source_polygon(source_quad);

        std::cout << "Reconstructed coefficients:\n";
        std::cout << std::fixed << std::setprecision(15);
        std::cout << "  c  = " << coeffs.c << "\n";
        std::cout << "  d  = " << coeffs.d << "\n";
        std::cout << "  a1 = " << coeffs.a1 << "\n";
        std::cout << "  b1 = " << coeffs.b1 << "\n";
        std::cout << "  a2 = " << coeffs.a2 << "\n";
        std::cout << "  b2 = " << coeffs.b2 << "\n\n";

        bool ok = true;
        ok = near(coeffs.d, 0.0, kTolerance, "zero divergence") && ok;
        ok = near(coeffs.a1, 1.0, kTolerance, "constant x velocity coeff") && ok;
        ok = near(coeffs.b1, 1.0, kTolerance, "constant y velocity coeff") && ok;
        ok = near(coeffs.a2, 0.0, kTolerance, "zero P2 coefficient") && ok;
        ok = near(coeffs.b2, 0.0, kTolerance, "zero Q2 coefficient") && ok;

        const Vector2d segment_a(-0.2, -0.3);
        const Vector2d segment_b(0.4, 0.2);
        const double exact_line = exact_u(segment_a).dot(segment_b - segment_a);
        const double reconstructed_line = interpolator.line_integral(source_quad, segment_a, segment_b);
        std::cout << "\nInterior target line integral:\n";
        ok = near(reconstructed_line, exact_line, kTolerance, "line integral") && ok;

        const moab::EntityHandle target_quad = create_quad(mb, {{
                                                                    Vector2d(0.25, 0.20),
                                                                    Vector2d(0.80, 0.20),
                                                                    Vector2d(0.80, 0.75),
                                                                    Vector2d(0.25, 0.75),
                                                                }});
        const std::vector<double> target_fluxes =
            interpolator.transfer_to_target_polygon_edges(source_quad, target_quad);

        std::cout << "\nNon-matching target edge fluxes:\n";
        const std::array<double, 4> exact_target_fluxes = {{-0.55, 0.55, 0.55, -0.55}};
        double target_flux_sum = 0.0;
        for (std::size_t i = 0; i < target_fluxes.size(); ++i) {
            target_flux_sum += target_fluxes[i];
            ok = near(target_fluxes[i], exact_target_fluxes[i], kTolerance,
                      "target edge " + std::to_string(i) + " flux") &&
                 ok;
        }
        ok = near(target_flux_sum, 0.0, kTolerance, "closed target conservation") && ok;

        if (!ok) {
            std::cout << "\n[FAILED] Conservative patch test failed.\n";
            return 1;
        }

        std::cout << "\n[SUCCESS] Recovered the constant conservative interpolant and target edge fluxes.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 2;
    }
}
