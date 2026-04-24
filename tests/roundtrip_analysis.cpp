// Quick diagnostic: test p=1 vs p=3 one-way and round-trip
// on CS→Voronoi (no RLL) to isolate the cell-moment bootstrap issue.

#include "mimetic/mimetic.hpp"
#include "spherical_transfer_test_utils.hpp"
#include <iostream>
#include <cmath>

using namespace mimetic;
using namespace mimetic::test_sphere;

int main() {
    GeometryOptions spherical;
    spherical.mode = GeometryMode::SphericalGnomonic;
    spherical.metric_weighted = true;

    // Test 1: One-way CS→Voronoi at p=1
    {
        moab::Core mb;
        auto cs = generate_cubed_sphere(mb, 8);
        auto vor = generate_icosahedral_dual(mb, 8);

        MimeticInterpolator interp(mb);
        interp.set_geometry_options(spherical);
        set_conservative_source_fluxes(interp, mb, cs, spherical_harmonic_gradient);
        for (auto c : cs) interp.reconstruct_source_polygon(c);

        auto xfer = interp.transfer_source_to_target_edges(cs, vor);

        // Compute error: transferred divergence vs exact on Voronoi
        auto exact_fluxes = conservative_edge_fluxes(mb, vor, spherical_harmonic_gradient);
        double max_err = 0;
        std::size_t dof = 0;
        for (auto cell : vor) {
            LocalPolygon poly = local_polygon(mb, cell, spherical);
            double xfer_div = 0, exact_div = 0;
            for (std::size_t i = 0; i < poly.vertices.size(); ++i, ++dof) {
                xfer_div += xfer.target_fluxes[dof];
                exact_div += exact_fluxes[std::make_pair(cell, i)];
            }
            max_err = std::max(max_err, std::abs((xfer_div - exact_div) / poly.area));
        }
        std::cout << "One-way CS→Voronoi p=1 max div error: " << max_err << "\n";
    }

    // Test 2: One-way Voronoi→CS at p=1
    {
        moab::Core mb;
        auto vor = generate_icosahedral_dual(mb, 8);
        auto cs = generate_cubed_sphere(mb, 8);

        MimeticInterpolator interp(mb);
        interp.set_geometry_options(spherical);
        set_conservative_source_fluxes(interp, mb, vor, spherical_harmonic_gradient);
        for (auto c : vor) interp.reconstruct_source_polygon(c);

        auto xfer = interp.transfer_source_to_target_edges(vor, cs);

        auto exact_fluxes = conservative_edge_fluxes(mb, cs, spherical_harmonic_gradient);
        double max_err = 0;
        std::size_t dof = 0;
        for (auto cell : cs) {
            LocalPolygon poly = local_polygon(mb, cell, spherical);
            double xfer_div = 0, exact_div = 0;
            for (std::size_t i = 0; i < poly.vertices.size(); ++i, ++dof) {
                xfer_div += xfer.target_fluxes[dof];
                exact_div += exact_fluxes[std::make_pair(cell, i)];
            }
            max_err = std::max(max_err, std::abs((xfer_div - exact_div) / poly.area));
        }
        std::cout << "One-way Voronoi→CS p=1 max div error: " << max_err << "\n";
    }

    // Test 3: Round-trip CS→Voronoi→CS at p=1
    {
        moab::Core mb;
        auto cs = generate_cubed_sphere(mb, 8);
        auto vor = generate_icosahedral_dual(mb, 8);

        // Forward
        MimeticInterpolator fwd(mb);
        fwd.set_geometry_options(spherical);
        set_conservative_source_fluxes(fwd, mb, cs, spherical_harmonic_gradient);
        for (auto c : cs) fwd.reconstruct_source_polygon(c);
        auto fwd_xfer = fwd.transfer_source_to_target_edges(cs, vor);

        // Backward
        MimeticInterpolator bwd(mb);
        bwd.set_geometry_options(spherical);
        std::size_t dof = 0;
        for (auto cell : vor) {
            LocalPolygon poly = local_polygon(mb, cell, spherical);
            for (std::size_t i = 0; i < poly.vertices.size(); ++i, ++dof)
                bwd.set_source_edge_flux(cell, i, fwd_xfer.target_fluxes[dof]);
        }
        for (auto c : vor) bwd.reconstruct_source_polygon(c);
        auto bwd_xfer = bwd.transfer_source_to_target_edges(vor, cs);

        // Error on CS
        auto exact_fluxes = conservative_edge_fluxes(mb, cs, spherical_harmonic_gradient);
        double max_err = 0;
        dof = 0;
        for (auto cell : cs) {
            LocalPolygon poly = local_polygon(mb, cell, spherical);
            double rt_div = 0, exact_div = 0;
            for (std::size_t i = 0; i < poly.vertices.size(); ++i, ++dof) {
                rt_div += bwd_xfer.target_fluxes[dof];
                exact_div += exact_fluxes[std::make_pair(cell, i)];
            }
            max_err = std::max(max_err, std::abs((rt_div - exact_div) / poly.area));
        }
        std::cout << "Round-trip CS→Voronoi→CS p=1 max div error: " << max_err << "\n";
    }

    // Test 4: One-way convergence p=1 vs p=3 at multiple resolutions
    std::cout << "\n=== One-way CS→Voronoi convergence (exact source moments) ===\n";
    for (int n : {4, 8, 12}) {
        moab::Core mb1;
        auto cs1 = generate_cubed_sphere(mb1, n);
        auto vor1 = generate_icosahedral_dual(mb1, n);

        // p=1
        MimeticInterpolator p1_interp(mb1);
        p1_interp.set_geometry_options(spherical);
        set_conservative_source_fluxes(p1_interp, mb1, cs1, spherical_harmonic_gradient);
        for (auto c : cs1) p1_interp.reconstruct_source_polygon(c);
        auto p1_xfer = p1_interp.transfer_source_to_target_edges(cs1, vor1);

        auto exact1 = conservative_edge_fluxes(mb1, vor1, spherical_harmonic_gradient);
        double p1_max = 0;
        std::size_t dof = 0;
        for (auto cell : vor1) {
            LocalPolygon poly = local_polygon(mb1, cell, spherical);
            double xd = 0, ed = 0;
            for (std::size_t i = 0; i < poly.vertices.size(); ++i, ++dof) {
                xd += p1_xfer.target_fluxes[dof];
                ed += exact1[std::make_pair(cell, i)];
            }
            p1_max = std::max(p1_max, std::abs((xd - ed) / poly.area));
        }

        // p=3 (exact source moments, needs fresh MOAB)
        moab::Core mb3;
        auto cs3 = generate_cubed_sphere(mb3, n);
        auto vor3 = generate_icosahedral_dual(mb3, n);

        PlanarMomentInterpolator p3_interp(mb3);
        p3_interp.set_geometry_options(spherical);
        MomentMethodOptions opts;
        opts.edge_moment_order = 3;
        opts.cell_moment_order = 2;
        opts.quadrature_points = 10;
        opts.regularization = 1.0e-12;
        opts.exact_constraints = false;

        for (auto cell : cs3) {
            LocalPolygon poly = local_polygon(mb3, cell, spherical);
            auto edges = local_edges(mb3, poly);
            GnomonicFrame frame{poly.n, poly.e_x, poly.e_y, 1.0};
            for (std::size_t ei = 0; ei < edges.size(); ++ei) {
                Eigen::Vector3d a3 = poly.points_3d[ei].normalized();
                Eigen::Vector3d b3 = poly.points_3d[(ei+1)%poly.points_3d.size()].normalized();
                double ta = std::acos(std::max(-1.0,std::min(1.0,a3.dot(b3))));
                std::vector<double> mom(4, 0.0);
                for (int deg = 0; deg <= 3; ++deg) {
                    mom[deg] = integrate_edge_gauss16(edges[ei].a, edges[ei].b,
                        [&](const Eigen::Vector2d& pl) {
                            Eigen::Vector2d xi = pl + poly.centroid;
                            Eigen::Vector3d pt = inverse_gnomonic(xi, frame).normalized();
                            Eigen::Vector2d cv = pullback_contravariant_piola(spherical_harmonic_gradient(pt), xi, frame);
                            double t = (ta > 1e-12) ? 2.0*(std::acos(std::max(-1.0,std::min(1.0,a3.dot(pt))))/ta)-1.0 : 0.0;
                            double Lm = 1.0;
                            if (deg==1) Lm=t; else if (deg==2) Lm=0.5*(3*t*t-1); else if (deg==3) Lm=0.5*(5*t*t*t-3*t);
                            return cv.dot(edges[ei].outward_normal) * Lm;
                        });
                }
                p3_interp.set_source_edge_moments(cell, ei, mom);
            }
            std::vector<Eigen::Vector2d> cm;
            for (int td = 0; td <= 2; ++td) for (int a = td; a >= 0; --a) {
                int b = td - a; double mx=0, my=0;
                for (auto& edge : edges) {
                    mx += integrate_triangle_scalar(Eigen::Vector2d::Zero(), edge.a, edge.b,
                        [&](const Eigen::Vector2d& p) { auto xi=p+poly.centroid; return pullback_contravariant_piola(spherical_harmonic_gradient(inverse_gnomonic(xi,frame).normalized()),xi,frame).x()*std::pow(p.x(),a)*std::pow(p.y(),b); });
                    my += integrate_triangle_scalar(Eigen::Vector2d::Zero(), edge.a, edge.b,
                        [&](const Eigen::Vector2d& p) { auto xi=p+poly.centroid; return pullback_contravariant_piola(spherical_harmonic_gradient(inverse_gnomonic(xi,frame).normalized()),xi,frame).y()*std::pow(p.x(),a)*std::pow(p.y(),b); });
                }
                cm.push_back(Eigen::Vector2d(mx, my));
            }
            p3_interp.set_source_cell_vector_moments(cell, cm);
            p3_interp.reconstruct_source_polygon(cell, opts);
        }
        auto p3_xfer = p3_interp.transfer_source_to_target_edge_moments(cs3, vor3, 3);

        auto exact3 = conservative_edge_fluxes(mb3, vor3, spherical_harmonic_gradient);
        double p3_max = 0;
        dof = 0;
        for (auto cell : vor3) {
            LocalPolygon poly = local_polygon(mb3, cell, spherical);
            double xd = 0, ed = 0;
            for (std::size_t i = 0; i < poly.vertices.size(); ++i, ++dof) {
                xd += p3_xfer.target_moments[dof][0];
                ed += exact3[std::make_pair(cell, i)];
            }
            p3_max = std::max(p3_max, std::abs((xd - ed) / poly.area));
        }

        std::cout << "n=" << n << " CS(" << cs1.size() << ")→Vor(" << vor1.size()
                  << ")  p=1=" << p1_max << "  p=3=" << p3_max << "\n";
    }

    return 0;
}
