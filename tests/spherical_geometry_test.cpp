#include "mimetic/mimetic.hpp"
#include "test_utils.hpp"

#include <Eigen/Dense>
#include <moab/Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <iostream>
#include <vector>

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

moab::EntityHandle create_spherical_polygon(moab::Core& mb, const std::vector<Eigen::Vector3d>& points)
{
    std::vector<moab::EntityHandle> vertices(points.size(), 0);
    for (std::size_t i = 0; i < points.size(); ++i) {
        const Eigen::Vector3d p = points[i].normalized();
        const double xyz[3] = {p.x(), p.y(), p.z()};
        mimetic::check_moab(mb.create_vertex(xyz, vertices[i]), "Failed to create spherical test vertex");
    }

    moab::EntityHandle polygon = 0;
    const moab::EntityType type = (vertices.size() == 4) ? moab::MBQUAD : moab::MBPOLYGON;
    mimetic::check_moab(mb.create_element(type, vertices.data(), static_cast<int>(vertices.size()), polygon),
                        "Failed to create spherical test polygon");
    for (std::size_t i = 0; i < vertices.size(); ++i) {
        mimetic::find_or_create_edge(mb, vertices[i], vertices[(i + 1) % vertices.size()]);
    }
    return polygon;
}

moab::EntityHandle create_chart_polygon(moab::Core& mb, const std::vector<Eigen::Vector2d>& chart_points)
{
    const mimetic::GnomonicFrame frame{
        Eigen::Vector3d(0.0, 0.0, 1.0),
        Eigen::Vector3d(1.0, 0.0, 0.0),
        Eigen::Vector3d(0.0, 1.0, 0.0),
        1.0,
    };
    std::vector<Eigen::Vector3d> points;
    points.reserve(chart_points.size());
    for (const Eigen::Vector2d& p : chart_points) {
        points.push_back(mimetic::inverse_gnomonic(p, frame));
    }
    return create_spherical_polygon(mb, points);
}

double spherical_interior_angle(const Eigen::Vector3d& previous,
                                const Eigen::Vector3d& vertex,
                                const Eigen::Vector3d& next)
{
    const Eigen::Vector3d v = vertex.normalized();
    const Eigen::Vector3d a = (previous.normalized() - previous.normalized().dot(v) * v).normalized();
    const Eigen::Vector3d b = (next.normalized() - next.normalized().dot(v) * v).normalized();
    return std::acos(std::max(-1.0, std::min(1.0, a.dot(b))));
}

double spherical_excess_area(const std::vector<Eigen::Vector3d>& points)
{
    double angle_sum = 0.0;
    for (std::size_t i = 0; i < points.size(); ++i) {
        angle_sum += spherical_interior_angle(points[(i + points.size() - 1) % points.size()],
                                              points[i],
                                              points[(i + 1) % points.size()]);
    }
    return angle_sum - (static_cast<double>(points.size()) - 2.0) * kPi;
}

double cross2(const Eigen::Vector2d& a, const Eigen::Vector2d& b)
{
    return a.x() * b.y() - a.y() * b.x();
}

}  // namespace

int main()
{
    try {
        std::cout << "--- Spherical Geometry Unit Test ---\n\n";

        bool ok = true;
        mimetic::GeometryOptions spherical;
        spherical.mode = mimetic::GeometryMode::SphericalGnomonic;
        spherical.conservation_tolerance = mimetic::kConservationTolerance;

        moab::Core mb;

        {
            const moab::EntityHandle triangle = create_spherical_polygon(mb, {
                Eigen::Vector3d(1.0, 0.0, 0.0),
                Eigen::Vector3d(0.0, 1.0, 0.0),
                Eigen::Vector3d(0.0, 0.0, 1.0),
            });
            const mimetic::SphericalPolygon poly = mimetic::spherical_polygon(mb, triangle, spherical);
            ok = mimetic::test::near(poly.spherical_area, 0.5 * kPi, 2.0e-15,
                                     "octant triangle area") &&
                 ok;
        }

        {
            const moab::EntityHandle quad = create_chart_polygon(mb, {
                Eigen::Vector2d(-0.22, -0.18),
                Eigen::Vector2d( 0.21, -0.16),
                Eigen::Vector2d( 0.20,  0.23),
                Eigen::Vector2d(-0.19,  0.22),
            });
            const mimetic::SphericalPolygon poly = mimetic::spherical_polygon(mb, quad, spherical);
            const double excess_area = spherical_excess_area(poly.points);
            ok = mimetic::test::near(poly.spherical_area, excess_area, 2.0e-15,
                                     "spherical quad area") &&
                 ok;
        }

        {
            const mimetic::GnomonicFrame frame{
                Eigen::Vector3d(0.0, 0.0, 1.0),
                Eigen::Vector3d(1.0, 0.0, 0.0),
                Eigen::Vector3d(0.0, 1.0, 0.0),
                1.0,
            };
            const Eigen::Vector2d xi(0.31, -0.17);
            const Eigen::Vector3d lifted = mimetic::inverse_gnomonic(xi, frame);
            const Eigen::Vector2d round_trip = mimetic::project_gnomonic(lifted, frame);
            ok = mimetic::test::near((round_trip - xi).norm(), 0.0, 1.0e-15,
                                     "gnomonic project/inverse") &&
                 ok;
        }

        {
            const mimetic::GnomonicFrame frame{
                Eigen::Vector3d(0.0, 0.0, 1.0),
                Eigen::Vector3d(1.0, 0.0, 0.0),
                Eigen::Vector3d(0.0, 1.0, 0.0),
                1.0,
            };
            const Eigen::Vector3d a = mimetic::inverse_gnomonic(Eigen::Vector2d(-0.35, 0.08), frame);
            const Eigen::Vector3d b = mimetic::inverse_gnomonic(Eigen::Vector2d(0.26, 0.31), frame);
            const Eigen::Vector3d midpoint = (a + b).normalized();
            const Eigen::Vector2d pa = mimetic::project_gnomonic(a, frame);
            const Eigen::Vector2d pb = mimetic::project_gnomonic(b, frame);
            const Eigen::Vector2d pm = mimetic::project_gnomonic(midpoint, frame);
            const double straightness = std::abs(cross2(pm - pa, pb - pa)) / (pb - pa).norm();
            ok = mimetic::test::near(straightness, 0.0, 1.0e-15,
                                     "great circle projects to line") &&
                 ok;
        }

        {
            const moab::EntityHandle pentagon = create_chart_polygon(mb, {
                Eigen::Vector2d(-0.30, -0.14),
                Eigen::Vector2d( 0.05, -0.28),
                Eigen::Vector2d( 0.32, -0.02),
                Eigen::Vector2d( 0.17,  0.25),
                Eigen::Vector2d(-0.25,  0.20),
            });
            mimetic::MimeticInterpolator interpolator(mb);
            interpolator.set_geometry_options(spherical);
            const std::vector<double> fluxes = {0.17, -0.08, 0.11, -0.04, 0.03};
            for (std::size_t i = 0; i < fluxes.size(); ++i) {
                interpolator.set_source_edge_flux(pentagon, i, fluxes[i]);
            }
            const mimetic::ReconstructionCoeffs coeffs = interpolator.reconstruct_source_polygon(pentagon);
            const mimetic::LocalPolygon poly = mimetic::local_polygon(mb, pentagon, spherical);
            const std::vector<mimetic::LocalEdge> edges = mimetic::local_edges(mb, poly);
            for (std::size_t i = 0; i < edges.size(); ++i) {
                const double reintegrated = interpolator.edge_flux(coeffs, edges[i].a, edges[i].b);
                ok = mimetic::test::near(reintegrated, fluxes[i], mimetic::kConservationTolerance,
                                         "KKT edge reintegration " + std::to_string(i)) &&
                     ok;
            }
        }

        if (!ok) {
            std::cout << "\n[FAILED] Spherical geometry test failed.\n";
            return 1;
        }

        std::cout << "\n[SUCCESS] Spherical chart geometry and KKT flux matching passed.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 2;
    }
}
