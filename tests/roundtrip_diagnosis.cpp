/// roundtrip_diagnosis.cpp
///
/// Diagnostic to isolate the bottleneck in round-trip convergence.
/// Test A: One-way CS → RLL (backward leg only, Piola RT source).
/// Test B: One-way RLL → CS (forward leg only, exact source).
/// This tells us which leg limits convergence and whether Piola RT
/// gives the expected O(h^{p+1}) rates on the backward leg.

#include "mimetic/mimetic.hpp"
#include "spherical_transfer_test_utils.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>

using namespace mimetic;
using namespace mimetic::test_sphere;

static constexpr double PI = 3.14159265358979323846;

// ── Utility: create lat/lon grid ──────────────────────────────────────────────
static std::vector<moab::EntityHandle>
make_latlon_grid(moab::Core& mb, int nlon, int nlat)
{
    std::vector<moab::EntityHandle> cells;
    std::vector<std::vector<moab::EntityHandle>> vgrid(nlat + 1);
    for (int j = 0; j <= nlat; ++j) {
        const double lat = PI * (0.5 - static_cast<double>(j) / nlat);
        const int nrow = (j == 0 || j == nlat) ? 1 : nlon;
        vgrid[j].resize(static_cast<std::size_t>(nrow));
        for (int i = 0; i < nrow; ++i) {
            const double lon = 2.0 * PI * i / nlon;
            const double xyz[3] = {
                std::cos(lat) * std::cos(lon),
                std::cos(lat) * std::sin(lon), std::sin(lat)};
            moab::EntityHandle v = 0;
            check_moab(mb.create_vertex(xyz, v), "v");
            vgrid[j][i] = v;
        }
    }
    {
        const moab::EntityHandle pole = vgrid[0][0];
        for (int i = 0; i < nlon; ++i) {
            moab::EntityHandle conn[3] = {pole, vgrid[1][i], vgrid[1][(i+1)%nlon]};
            moab::EntityHandle tri = 0;
            check_moab(mb.create_element(moab::MBPOLYGON, conn, 3, tri), "N tri");
            for (int k = 0; k < 3; ++k) find_or_create_edge(mb, conn[k], conn[(k+1)%3]);
            cells.push_back(tri);
        }
    }
    for (int j = 1; j < nlat - 1; ++j) {
        for (int i = 0; i < nlon; ++i) {
            const int in = (i + 1) % nlon;
            moab::EntityHandle conn[4] = {vgrid[j][i], vgrid[j][in], vgrid[j+1][in], vgrid[j+1][i]};
            moab::EntityHandle q = 0;
            check_moab(mb.create_element(moab::MBQUAD, conn, 4, q), "q");
            for (int k = 0; k < 4; ++k) find_or_create_edge(mb, conn[k], conn[(k+1)%4]);
            cells.push_back(q);
        }
    }
    {
        const moab::EntityHandle pole = vgrid[nlat][0];
        for (int i = 0; i < nlon; ++i) {
            moab::EntityHandle conn[3] = {vgrid[nlat-1][i], pole, vgrid[nlat-1][(i+1)%nlon]};
            moab::EntityHandle tri = 0;
            check_moab(mb.create_element(moab::MBPOLYGON, conn, 3, tri), "S tri");
            for (int k = 0; k < 3; ++k) find_or_create_edge(mb, conn[k], conn[(k+1)%3]);
            cells.push_back(tri);
        }
    }
    return cells;
}

// ── Set exact high-order edge moments on cells ────────────────────────────────
static void set_exact_moments(moab::Core& mb,
                               PlanarMomentInterpolator& interp,
                               const std::vector<moab::EntityHandle>& cells,
                               const GeometryOptions& geo,
                               int order)
{
    MomentMethodOptions opts;
    opts.edge_moment_order  = order;
    opts.cell_moment_order  = std::max(1, order - 1);
    opts.quadrature_points  = 10;
    opts.regularization     = 1.0e-12;
    opts.exact_constraints  = false;

    for (const moab::EntityHandle cell : cells) {
        const LocalPolygon poly = local_polygon(mb, cell, geo);
        const std::vector<LocalEdge> edges = local_edges(mb, poly);
        const GnomonicFrame frame{poly.n, poly.e_x, poly.e_y, 1.0};

        for (std::size_t ei = 0; ei < edges.size(); ++ei) {
            const Eigen::Vector3d a3 = poly.points_3d[ei].normalized();
            const Eigen::Vector3d b3 = poly.points_3d[(ei+1)%poly.points_3d.size()].normalized();
            const double total_angle = std::acos(std::max(-1.0, std::min(1.0, a3.dot(b3))));
            std::vector<double> moments(static_cast<std::size_t>(order + 1), 0.0);
            for (int deg = 0; deg <= order; ++deg) {
                moments[deg] = integrate_edge_gauss16(edges[ei].a, edges[ei].b,
                    [&](const Eigen::Vector2d& p) {
                        const Eigen::Vector2d xi = p + poly.centroid;
                        const Eigen::Vector3d pt = inverse_gnomonic(xi, frame).normalized();
                        const Eigen::Vector2d cv = pullback_contravariant_piola(
                            spherical_harmonic_gradient(pt), xi, frame);
                        double t = 0.0;
                        if (total_angle > 1e-12)
                            t = 2.0 * std::acos(std::max(-1.0, std::min(1.0, a3.dot(pt))))
                                / total_angle - 1.0;
                        double Lm = (deg==0)?1:(deg==1)?t:(deg==2)?
                            0.5*(3*t*t-1):0.5*(5*t*t*t-3*t);
                        return cv.dot(edges[ei].outward_normal) * Lm;
                    });
            }
            interp.set_source_edge_moments(cell, ei, moments);
        }

        const int cmo = std::max(1, order - 1);
        std::vector<Eigen::Vector2d> cm;
        for (int td = 0; td <= cmo; ++td)
            for (int a = td; a >= 0; --a) {
                const int b = td - a;
                double mx = 0, my = 0;
                for (const LocalEdge& edge : edges) {
                    mx += integrate_triangle_scalar(Eigen::Vector2d::Zero(), edge.a, edge.b,
                        [&](const Eigen::Vector2d& p) {
                            const Eigen::Vector2d xi = p + poly.centroid;
                            const Eigen::Vector3d pt = inverse_gnomonic(xi, frame).normalized();
                            return pullback_contravariant_piola(spherical_harmonic_gradient(pt), xi, frame).x()
                                * std::pow(p.x(), a) * std::pow(p.y(), b);
                        });
                    my += integrate_triangle_scalar(Eigen::Vector2d::Zero(), edge.a, edge.b,
                        [&](const Eigen::Vector2d& p) {
                            const Eigen::Vector2d xi = p + poly.centroid;
                            const Eigen::Vector3d pt = inverse_gnomonic(xi, frame).normalized();
                            return pullback_contravariant_piola(spherical_harmonic_gradient(pt), xi, frame).y()
                                * std::pow(p.x(), a) * std::pow(p.y(), b);
                        });
                }
                cm.emplace_back(mx, my);
            }
        interp.set_source_cell_vector_moments(cell, cm);
        interp.reconstruct_source_polygon(cell, opts);
    }
}

// ── Exact cell-div ────────────────────────────────────────────────────────────
static std::vector<double>
exact_cell_div(moab::Core& mb, const std::vector<moab::EntityHandle>& cells,
               const GeometryOptions& geo)
{
    std::vector<double> res;
    for (const moab::EntityHandle cell : cells) {
        const LocalPolygon poly = local_polygon(mb, cell, geo);
        const std::vector<LocalEdge> edges = local_edges(mb, poly);
        double div = 0;
        for (std::size_t i = 0; i < edges.size(); ++i)
            div += exact_chart_edge_flux(mb, cell, i, spherical_harmonic_gradient);
        res.push_back(div / poly.area);
    }
    return res;
}

static double maxerr(const std::vector<double>& a, const std::vector<double>& b) {
    double e = 0;
    for (std::size_t i = 0; i < a.size(); ++i) e = std::max(e, std::abs(a[i]-b[i]));
    return e;
}

int main()
{
    std::cout << std::scientific << std::setprecision(4);

    GeometryOptions geo;
    geo.mode           = GeometryMode::SphericalGnomonic;
    geo.metric_weighted = true;

    const std::vector<int> ns = {8, 16, 32, 64};

    // ── Test A: CS (exact) → RLL (backward leg with Piola RT) ─────────────────
    std::cout << "=== Test A: CS (exact) → RLL (backward leg, Piola RT) ===\n";
    std::cout << "  Measures the backward-leg convergence in isolation.\n\n";

    for (int p = 1; p <= 3; ++p) {
        std::cout << "  p=" << p << ":\n";
        double prev = 0;
        for (const int n : ns) {
            moab::Core mb;
            const auto cs  = generate_cubed_sphere(mb, n);
            // RLL is proportional: cell size ≈ CS cell size
            const auto rll = make_latlon_grid(mb, 4*n, 2*n);
            const auto exact_rll = exact_cell_div(mb, rll, geo);

            MomentMethodOptions opts;
            opts.edge_moment_order  = p;
            opts.cell_moment_order  = std::max(1, p - 1);
            opts.quadrature_points  = 10;
            opts.regularization     = 1.0e-12;
            opts.exact_constraints  = false;
            const int cmo = std::max(1, p - 1);
            int n_cm = 0; for (int td = 0; td <= cmo; ++td) n_cm += td + 1;

            // Set exact moments on CS, reconstruct with Piola RT
            PlanarMomentInterpolator bwd(mb);
            GeometryOptions bwd_geo = geo; bwd_geo.metric_weighted = false;
            bwd.set_geometry_options(bwd_geo);
            set_exact_moments(mb, bwd, cs, bwd_geo, p);

            // Apply Piola RT on CS quads
            for (const moab::EntityHandle cell : cs) {
                const LocalPolygon poly = local_polygon(mb, cell, bwd_geo);
                if (static_cast<int>(poly.vertices.size()) == 4)
                    bwd.reconstruct_source_polygon_piola_rt(cell, p);
                else {
                    MomentMethodOptions eo = opts; eo.cell_weight = 0.0;
                    bwd.set_source_cell_vector_moments(cell,
                        std::vector<Eigen::Vector2d>(
                            static_cast<std::size_t>(n_cm), Eigen::Vector2d::Zero()));
                    bwd.reconstruct_source_polygon(cell, eo);
                }
            }

            const EdgeMomentTransferResult xfer =
                bwd.transfer_source_to_target_edge_moments(cs, rll, p);

            std::vector<double> rll_div;
            std::size_t d = 0;
            for (const moab::EntityHandle cell : rll) {
                const LocalPolygon poly = local_polygon(mb, cell, geo);
                double flux = 0;
                for (std::size_t e = 0; e < poly.vertices.size(); ++e, ++d)
                    flux += xfer.target_moments[d][0];
                rll_div.push_back(flux / poly.area);
            }

            const double err = maxerr(rll_div, exact_rll);
            const double rate = (prev > 0) ? std::log2(prev / err) : 0;
            std::cout << "    n=" << std::setw(3) << n
                      << "  CS=" << std::setw(6) << cs.size()
                      << "  RLL=" << std::setw(6) << rll.size()
                      << "  L∞=" << err;
            if (prev > 0) std::cout << "  rate=" << std::fixed << std::setprecision(2) << rate;
            std::cout << "\n" << std::scientific << std::setprecision(4);
            prev = err;
        }
        std::cout << "  Expected rate: " << p+1 << "\n\n";
    }

    // ── Test B: RLL (exact) → CS (forward leg) ────────────────────────────────
    std::cout << "=== Test B: RLL (exact) → CS (forward leg only) ===\n";
    std::cout << "  Measures forward-leg convergence (RLL as source).\n\n";

    for (int p = 1; p <= 3; ++p) {
        std::cout << "  p=" << p << ":\n";
        double prev = 0;
        for (const int n : ns) {
            moab::Core mb;
            const auto rll = make_latlon_grid(mb, 4*n, 2*n);
            const auto cs  = generate_cubed_sphere(mb, n);
            const auto exact_cs = exact_cell_div(mb, cs, geo);

            MomentMethodOptions opts;
            opts.edge_moment_order  = p;
            opts.cell_moment_order  = std::max(1, p - 1);
            opts.quadrature_points  = 10;
            opts.regularization     = 1.0e-12;
            opts.exact_constraints  = false;

            PlanarMomentInterpolator fwd(mb);
            fwd.set_geometry_options(geo);
            set_exact_moments(mb, fwd, rll, geo, p);

            const EdgeMomentTransferResult xfer =
                fwd.transfer_source_to_target_edge_moments(rll, cs, p);

            // cell-div on CS from moment 0
            std::vector<double> cs_div;
            std::size_t d = 0;
            for (const moab::EntityHandle cell : cs) {
                const LocalPolygon poly = local_polygon(mb, cell, geo);
                double flux = 0;
                for (std::size_t e = 0; e < poly.vertices.size(); ++e, ++d)
                    flux += xfer.target_moments[d][0];
                cs_div.push_back(flux / poly.area);
            }

            const double err = maxerr(cs_div, exact_cs);
            const double rate = (prev > 0) ? std::log2(prev / err) : 0;
            std::cout << "    n=" << std::setw(3) << n
                      << "  RLL=" << std::setw(6) << rll.size()
                      << "  CS=" << std::setw(6) << cs.size()
                      << "  L∞=" << err;
            if (prev > 0) std::cout << "  rate=" << std::fixed << std::setprecision(2) << rate;
            std::cout << "\n" << std::scientific << std::setprecision(4);
            prev = err;
        }
        std::cout << "  Expected rate: " << p+1 << "\n\n";
    }

    // ── Test C: CS (exact) → CS (one-way, same mechanism as one-way study) ──────
    std::cout << "=== Test C: CS (exact) → CS (one-way, cell-div measure) ===\n";
    std::cout << "  Baseline: does cell-div converge at O(h^{p+1}) for CS→CS?\n\n";

    for (int p = 1; p <= 3; ++p) {
        std::cout << "  p=" << p << ":\n";
        double prev = 0;
        for (const int n : ns) {
            moab::Core mb;
            const int n_src = n;
            const int n_tgt = n + n/2;  // ~1.5× refinement of source
            const auto cs_src = generate_cubed_sphere(mb, n_src);
            const auto cs_tgt = generate_cubed_sphere(mb, n_tgt);
            const auto exact_tgt = exact_cell_div(mb, cs_tgt, geo);

            MomentMethodOptions opts;
            opts.edge_moment_order  = p;
            opts.cell_moment_order  = std::max(1, p - 1);
            opts.quadrature_points  = 10;
            opts.regularization     = 1.0e-12;
            opts.exact_constraints  = false;

            PlanarMomentInterpolator fwd(mb);
            GeometryOptions src_geo = geo; src_geo.metric_weighted = false;
            fwd.set_geometry_options(src_geo);
            set_exact_moments(mb, fwd, cs_src, src_geo, p);

            // Piola RT for CS source quads
            const int cmo = std::max(1, p - 1);
            int n_cm = 0; for (int td = 0; td <= cmo; ++td) n_cm += td + 1;
            for (const moab::EntityHandle cell : cs_src) {
                const LocalPolygon poly = local_polygon(mb, cell, src_geo);
                if (static_cast<int>(poly.vertices.size()) == 4)
                    fwd.reconstruct_source_polygon_piola_rt(cell, p);
                else {
                    MomentMethodOptions eo = opts; eo.cell_weight = 0.0;
                    fwd.set_source_cell_vector_moments(cell,
                        std::vector<Eigen::Vector2d>(
                            static_cast<std::size_t>(n_cm), Eigen::Vector2d::Zero()));
                    fwd.reconstruct_source_polygon(cell, eo);
                }
            }

            const EdgeMomentTransferResult xfer =
                fwd.transfer_source_to_target_edge_moments(cs_src, cs_tgt, p);

            std::vector<double> tgt_div;
            std::size_t d = 0;
            for (const moab::EntityHandle cell : cs_tgt) {
                const LocalPolygon poly = local_polygon(mb, cell, geo);
                double flux = 0;
                for (std::size_t e = 0; e < poly.vertices.size(); ++e, ++d)
                    flux += xfer.target_moments[d][0];
                tgt_div.push_back(flux / poly.area);
            }

            const double err = maxerr(tgt_div, exact_tgt);
            const double rate = (prev > 0) ? std::log2(prev / err) : 0;
            std::cout << "    n_src=" << std::setw(3) << n_src
                      << "  n_tgt=" << std::setw(3) << n_tgt
                      << "  L∞=" << err;
            if (prev > 0) std::cout << "  rate=" << std::fixed << std::setprecision(2) << rate;
            std::cout << "\n" << std::scientific << std::setprecision(4);
            prev = err;
        }
        std::cout << "  Expected rate: " << p+1 << "\n\n";
    }

    return 0;
}
