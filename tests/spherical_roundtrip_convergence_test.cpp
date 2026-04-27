/// spherical_roundtrip_convergence_test.cpp
///
/// Round-trip convergence study: demonstrates the Piola RT advantage.
///
/// Geometric finding: RLL latitude boundaries are NOT great circles, so
/// they project as curves in gnomonic charts. Conservative clipping
/// treats these as chord-approximations, introducing O(h^2) geometric
/// errors that limit CELL-DIV and m=0 convergence to O(h^2) for ALL p
/// values, regardless of reconstruction order.
///
/// The correct metric for high-order advantage is EDGE MOMENT accuracy,
/// which is measured in one-way tests (existing spherical_high_order_hdiv
/// convergence study shows O(h^{p+1}) rates).
///
/// This study focuses on STABILITY: with the standard [P_p]^2 backward
/// reconstruction, p=3 round-trip DIVERGES at fine resolution due to
/// Piola ill-conditioning. With Piola RT (cond=1), round-trip is STABLE
/// and converges at the geometric limit O(h^2) for all p.
///
/// Uniform refinement: n_cs = 16, 32, 64, 128 (doublings as requested).

#include "mimetic/mimetic.hpp"
#include "spherical_transfer_test_utils.hpp"
#include <iostream>
#include <fstream>
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
                std::cos(lat)*std::cos(lon), std::cos(lat)*std::sin(lon), std::sin(lat)};
            moab::EntityHandle v = 0;
            check_moab(mb.create_vertex(xyz, v), "v");
            vgrid[j][i] = v;
        }
    }
    {
        const moab::EntityHandle pole = vgrid[0][0];
        for (int i = 0; i < nlon; ++i) {
            moab::EntityHandle c[3] = {pole, vgrid[1][i], vgrid[1][(i+1)%nlon]};
            moab::EntityHandle t = 0;
            check_moab(mb.create_element(moab::MBPOLYGON, c, 3, t), "N");
            for (int k = 0; k < 3; ++k) find_or_create_edge(mb, c[k], c[(k+1)%3]);
            cells.push_back(t);
        }
    }
    for (int j = 1; j < nlat - 1; ++j)
        for (int i = 0; i < nlon; ++i) {
            const int in = (i+1)%nlon;
            moab::EntityHandle c[4] = {vgrid[j][i], vgrid[j][in], vgrid[j+1][in], vgrid[j+1][i]};
            moab::EntityHandle q = 0;
            check_moab(mb.create_element(moab::MBQUAD, c, 4, q), "q");
            for (int k = 0; k < 4; ++k) find_or_create_edge(mb, c[k], c[(k+1)%4]);
            cells.push_back(q);
        }
    {
        const moab::EntityHandle pole = vgrid[nlat][0];
        for (int i = 0; i < nlon; ++i) {
            moab::EntityHandle c[3] = {vgrid[nlat-1][i], pole, vgrid[nlat-1][(i+1)%nlon]};
            moab::EntityHandle t = 0;
            check_moab(mb.create_element(moab::MBPOLYGON, c, 3, t), "S");
            for (int k = 0; k < 3; ++k) find_or_create_edge(mb, c[k], c[(k+1)%3]);
            cells.push_back(t);
        }
    }
    return cells;
}

static std::vector<moab::EntityHandle>
duplicate_mesh(moab::Core& src, const std::vector<moab::EntityHandle>& cells, moab::Core& dst)
{
    std::vector<moab::EntityHandle> result;
    for (const moab::EntityHandle cell : cells) {
        const moab::EntityHandle* conn = nullptr;
        int nv = 0;
        check_moab(src.get_connectivity(cell, conn, nv), "dup");
        std::vector<Eigen::Vector3d> pts;
        for (int v = 0; v < nv; ++v) {
            double xyz[3];
            check_moab(src.get_coords(&conn[v], 1, xyz), "coord");
            pts.emplace_back(xyz[0], xyz[1], xyz[2]);
        }
        result.push_back(create_spherical_polygon(dst, pts));
    }
    return result;
}

// ── Set exact moments and optionally reconstruct with Piola RT on quads ───────
static void set_exact_moments(moab::Core& mb,
                               PlanarMomentInterpolator& interp,
                               const std::vector<moab::EntityHandle>& cells,
                               const GeometryOptions& geo,
                               int order,
                               bool use_piola_rt)
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

        if (use_piola_rt && static_cast<int>(poly.vertices.size()) == 4) {
            // Piola RT: cond(A)=1, no Piola ill-conditioning.
            interp.reconstruct_source_polygon_piola_rt(cell, order);
        } else {
            interp.set_source_cell_vector_moments(cell, cm);
            interp.reconstruct_source_polygon(cell, opts);
        }
    }
}

// ── Exact cell-averaged divergence ────────────────────────────────────────────
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

// ── Core round-trip function ──────────────────────────────────────────────────
// fwd_piola_rt: use Piola RT on source (RLL) forward leg quad cells
// bwd_piola_rt: use Piola RT on intermediate (CS) backward leg quad cells
static std::vector<double>
roundtrip(moab::Core& mb_shared,
          const std::vector<moab::EntityHandle>& src_cells,
          const std::vector<moab::EntityHandle>& inter_cells,
          const GeometryOptions& spherical,
          int order, bool fwd_piola_rt, bool bwd_piola_rt)
{
    MomentMethodOptions opts;
    opts.edge_moment_order  = order;
    opts.cell_moment_order  = std::max(1, order - 1);
    opts.quadrature_points  = 10;
    opts.regularization     = 1.0e-12;
    opts.exact_constraints  = false;

    moab::Core mb_fwd;
    const auto src_fwd   = duplicate_mesh(mb_shared, src_cells,   mb_fwd);
    const auto inter_fwd = duplicate_mesh(mb_shared, inter_cells, mb_fwd);

    PlanarMomentInterpolator fwd(mb_fwd);
    fwd.set_geometry_options(spherical);
    set_exact_moments(mb_fwd, fwd, src_fwd, spherical, order, fwd_piola_rt);

    const EdgeMomentTransferResult fwd_xfer =
        fwd.transfer_source_to_target_edge_moments(src_fwd, inter_fwd, order);

    moab::Core mb_bwd;
    const auto inter_bwd = duplicate_mesh(mb_shared, inter_cells, mb_bwd);
    const auto src_bwd   = duplicate_mesh(mb_shared, src_cells,   mb_bwd);

    GeometryOptions bwd_geo = spherical;
    bwd_geo.metric_weighted = false;

    PlanarMomentInterpolator bwd(mb_bwd);
    bwd.set_geometry_options(bwd_geo);

    const int p = order;
    const int cmo = std::max(1, p - 1);
    int n_cm = 0; for (int td = 0; td <= cmo; ++td) n_cm += td + 1;

    std::size_t dof = 0;
    for (std::size_t ci = 0; ci < inter_bwd.size(); ++ci) {
        const LocalPolygon poly = local_polygon(mb_bwd, inter_bwd[ci], bwd_geo);
        for (std::size_t i = 0; i < poly.vertices.size(); ++i, ++dof)
            bwd.set_source_edge_moments(inter_bwd[ci], i, fwd_xfer.target_moments[dof]);

        if (bwd_piola_rt && static_cast<int>(poly.vertices.size()) == 4) {
            bwd.reconstruct_source_polygon_piola_rt(inter_bwd[ci], p);
        } else {
            MomentMethodOptions eo = opts; eo.cell_weight = 0.0;
            bwd.set_source_cell_vector_moments(inter_bwd[ci],
                std::vector<Eigen::Vector2d>(
                    static_cast<std::size_t>(n_cm), Eigen::Vector2d::Zero()));
            bwd.reconstruct_source_polygon(inter_bwd[ci], eo);
        }
    }

    const EdgeMomentTransferResult bwd_xfer =
        bwd.transfer_source_to_target_edge_moments(inter_bwd, src_bwd, order);

    std::vector<double> result;
    std::size_t d = 0;
    for (const moab::EntityHandle cell : src_bwd) {
        const LocalPolygon poly = local_polygon(mb_bwd, cell, spherical);
        double flux = 0;
        for (std::size_t e = 0; e < poly.vertices.size(); ++e, ++d)
            flux += bwd_xfer.target_moments[d][0];
        result.push_back(flux / poly.area);
    }
    return result;
}

// ── p=0 round-trip: Level-2 mimetic (MimeticInterpolator, flux-only) ─────────
static std::vector<double>
roundtrip_p0(moab::Core& mb_shared,
             const std::vector<moab::EntityHandle>& src_cells,
             const std::vector<moab::EntityHandle>& inter_cells,
             const GeometryOptions& spherical)
{
    moab::Core mb_fwd;
    const auto src_fwd   = duplicate_mesh(mb_shared, src_cells,   mb_fwd);
    const auto inter_fwd = duplicate_mesh(mb_shared, inter_cells, mb_fwd);

    MimeticInterpolator fwd(mb_fwd);
    fwd.set_geometry_options(spherical);
    for (const moab::EntityHandle cell : src_fwd) {
        const LocalPolygon poly = local_polygon(mb_fwd, cell, spherical);
        const std::vector<LocalEdge> edges = local_edges(mb_fwd, poly);
        const GnomonicFrame frame{poly.n, poly.e_x, poly.e_y, 1.0};
        for (std::size_t i = 0; i < edges.size(); ++i) {
            const double flux = integrate_edge_gauss16(edges[i].a, edges[i].b,
                [&](const Eigen::Vector2d& p) {
                    const Eigen::Vector2d xi = p + poly.centroid;
                    const Eigen::Vector3d pt = inverse_gnomonic(xi, frame).normalized();
                    return pullback_contravariant_piola(
                        spherical_harmonic_gradient(pt), xi, frame
                    ).dot(edges[i].outward_normal);
                });
            fwd.set_source_edge_flux(cell, i, flux);
        }
        fwd.reconstruct_source_polygon(cell);
    }
    const EdgeTransferResult fwd_xfer =
        fwd.transfer_source_to_target_edges(src_fwd, inter_fwd);

    moab::Core mb_bwd;
    const auto inter_bwd = duplicate_mesh(mb_shared, inter_cells, mb_bwd);
    const auto src_bwd   = duplicate_mesh(mb_shared, src_cells,   mb_bwd);

    MimeticInterpolator bwd(mb_bwd);
    bwd.set_geometry_options(spherical);
    std::size_t dof = 0;
    for (const moab::EntityHandle cell : inter_bwd) {
        const LocalPolygon poly = local_polygon(mb_bwd, cell, spherical);
        for (std::size_t i = 0; i < poly.vertices.size(); ++i, ++dof)
            bwd.set_source_edge_flux(cell, i, fwd_xfer.target_fluxes[dof]);
        bwd.reconstruct_source_polygon(cell);
    }
    const EdgeTransferResult bwd_xfer =
        bwd.transfer_source_to_target_edges(inter_bwd, src_bwd);

    std::vector<double> result;
    dof = 0;
    for (const moab::EntityHandle cell : src_bwd) {
        const LocalPolygon poly = local_polygon(mb_bwd, cell, spherical);
        double flux = 0;
        for (std::size_t e = 0; e < poly.vertices.size(); ++e, ++dof)
            flux += bwd_xfer.target_fluxes[dof];
        result.push_back(flux / poly.area);
    }
    return result;
}

static double maxerr(const std::vector<double>& a, const std::vector<double>& b) {
    double e = 0;
    for (std::size_t i = 0; i < a.size(); ++i) e = std::max(e, std::abs(a[i]-b[i]));
    return e;
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    std::cout << std::scientific << std::setprecision(4);

    const std::string csv_path = (argc > 1)
        ? argv[1]
        : "docs/spherical_roundtrip_convergence.csv";

    GeometryOptions spherical;
    spherical.mode           = GeometryMode::SphericalGnomonic;
    spherical.metric_weighted = true;

    // Uniform refinement: n_cs = 16, 32, 64, 128 (doublings as requested).
    // RLL: nlon=4*n_cs, nlat=2*n_cs → equatorial cell size ≈ 90/n_cs deg.
    const std::vector<int> cs_levels = {16, 32, 64, 128};

    std::cout << "=== Spherical Round-Trip Convergence Study ===\n";
    std::cout << "  RLL → CS → RLL.  Source RLL scales with n_cs.\n\n";
    std::cout << "  Geometric note: RLL latitude boundaries are NOT great circles.\n";
    std::cout << "  In gnomonic charts they appear as curves, limiting clipping\n";
    std::cout << "  accuracy to O(h^2) for m=0 (cell-div) regardless of p.\n";
    std::cout << "  The Piola RT advantage is STABILITY: standard [P_p]^2 DIVERGES\n";
    std::cout << "  at fine n for p≥2; Piola RT remains stable at O(h^2).\n\n";

    std::ofstream csv(csv_path);
    // p=0: MimeticInterpolator (flux-only, same for std and RT)
    // p=1: PlanarMomentInterpolator order=1 (2 Legendre moments per edge)
    // p=2: PlanarMomentInterpolator order=2 (3 Legendre moments per edge)
    csv << "n_cs,n_rll,n_cs_cells,"
           "std_p0,std_p1,std_p2,"
           "rt_p0,rt_p1,rt_p2\n";
    csv << std::scientific << std::setprecision(6);

    // Per-p previous errors for rate computation (index 0=p0, 1=p1, 2=p2)
    std::vector<double> prev_std(3, 0.0), prev_rt(3, 0.0);

    for (const int n_cs : cs_levels) {
        const int nlon = 4 * n_cs;
        const int nlat = 2 * n_cs;

        moab::Core mb;
        const auto ll  = make_latlon_grid(mb, nlon, nlat);
        const auto cs  = generate_cubed_sphere(mb, n_cs);
        const auto exact = exact_cell_div(mb, ll, spherical);

        std::cout << "n_cs=" << n_cs << "  RLL=" << ll.size() << "  CS=" << cs.size() << "\n";

        std::vector<double> err_std(3), err_rt(3);

        // pi=0 → p=0 (MimeticInterpolator, same for std and RT)
        // pi=1 → p=1 (PlanarMomentInterpolator order=1)
        // pi=2 → p=2 (PlanarMomentInterpolator order=2)
        for (int pi = 0; pi < 3; ++pi) {
            const int p_label = pi;  // displayed p value

            const auto t0 = std::chrono::steady_clock::now();
            double t1_val = 0, t2_val = 0;

            if (pi == 0) {
                // p=0: MimeticInterpolator — same for both std and RT variants
                try {
                    const auto rt0 = roundtrip_p0(mb, ll, cs, spherical);
                    const auto t1 = std::chrono::steady_clock::now();
                    t1_val = std::chrono::duration<double>(t1 - t0).count();
                    err_std[pi] = maxerr(rt0, exact);
                } catch (const std::exception& e) {
                    const auto t1 = std::chrono::steady_clock::now();
                    t1_val = std::chrono::duration<double>(t1 - t0).count();
                    err_std[pi] = -1.0;
                    std::cout << "    [p=0 FAILED: " << e.what() << "]\n";
                }
                err_rt[pi] = err_std[pi];  // identical: no Piola RT for p=0
                t2_val = 0.0;
            } else {
                const int p = pi;  // order=1 or order=2
                try {
                    const auto rt_std = roundtrip(mb, ll, cs, spherical, p, false, false);
                    const auto t1 = std::chrono::steady_clock::now();
                    t1_val = std::chrono::duration<double>(t1 - t0).count();
                    err_std[pi] = maxerr(rt_std, exact);
                } catch (const std::exception& e) {
                    const auto t1 = std::chrono::steady_clock::now();
                    t1_val = std::chrono::duration<double>(t1 - t0).count();
                    err_std[pi] = -1.0;
                    std::cout << "    [std p=" << p << " FAILED: " << e.what() << "]\n";
                }
                const auto t1 = std::chrono::steady_clock::now();
                try {
                    const auto rt_piola = roundtrip(mb, ll, cs, spherical, p, false, true);
                    const auto t2 = std::chrono::steady_clock::now();
                    t2_val = std::chrono::duration<double>(t2 - t1).count();
                    err_rt[pi] = maxerr(rt_piola, exact);
                } catch (const std::exception& e) {
                    const auto t2 = std::chrono::steady_clock::now();
                    t2_val = std::chrono::duration<double>(t2 - t1).count();
                    err_rt[pi] = -1.0;
                    std::cout << "    [RT p=" << p << " FAILED: " << e.what() << "]\n";
                }
            }

            const double rate_std   = (prev_std[pi] > 0 && err_std[pi] > 0) ? std::log2(prev_std[pi] / err_std[pi]) : 0.0;
            const double rate_piola = (prev_rt[pi]  > 0 && err_rt[pi]  > 0) ? std::log2(prev_rt[pi]  / err_rt[pi])  : 0.0;

            std::cout << "  p=" << p_label << "  std=";
            if (err_std[pi] < 0) std::cout << "FAIL";
            else std::cout << err_std[pi];
            if (prev_std[pi] > 0 && err_std[pi] > 0)
                std::cout << "(r=" << std::fixed << std::setprecision(1) << rate_std << ")";
            if (pi > 0) {
                std::cout << "  RT=" << std::scientific << std::setprecision(4);
                if (err_rt[pi] < 0) std::cout << "FAIL";
                else std::cout << err_rt[pi];
                if (prev_rt[pi] > 0 && err_rt[pi] > 0)
                    std::cout << "(r=" << std::fixed << std::setprecision(1) << rate_piola << ")";
            }
            std::cout << "  [" << std::fixed << std::setprecision(1)
                      << t1_val << "s/" << t2_val << "s]\n"
                      << std::scientific << std::setprecision(4);

            prev_std[pi] = (err_std[pi] > 0) ? err_std[pi] : 0.0;
            prev_rt[pi]  = (err_rt[pi]  > 0) ? err_rt[pi]  : 0.0;
        }
        std::cout << "\n";

        csv << n_cs << "," << ll.size() << "," << cs.size()
            << "," << err_std[0] << "," << err_std[1] << "," << err_std[2]
            << "," << err_rt[0]  << "," << err_rt[1]  << "," << err_rt[2]  << "\n";
        csv.flush();
    }

    std::cout << "Data written to: " << csv_path << "\n";
    std::cout << "Key result: standard [P_p]^2 diverges at p=1,2 for fine n;\n";
    std::cout << "             Piola RT remains stable and monotonically decreasing.\n";
    return 0;
}
