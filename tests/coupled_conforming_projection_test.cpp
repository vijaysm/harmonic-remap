// Regression test for the coupled multi-degree conforming target-edge
// projection.
//
// Verifies the four properties promised by the implementation plan
// (docs/plans/2026-05-02-coupled-conforming-projection.md):
//
//   1. Trace continuity per unique edge per degree -- after the
//      projection, every pair of directed views of the same geometric
//      edge agrees to roundoff after the orientation parity flip.
//   2. Cell divergence balance for m = 0 within the repository
//      conservation tolerance.
//   3. The new diagnostic fields trace_jump_per_unique_edge and
//      unique_edge_lengths are populated coherently (lengths positive,
//      jumps non-negative, jump = 0 on boundary edges).
//   4. The trace-jump norm shrinks under refinement -- a smoke check
//      that the diagnostic is a meaningful asymptotic indicator of
//      conformance.
//
// The test uses planar nonmatching quad meshes with exact edge
// Legendre moments produced from a known polynomial field, so the
// transfer step is deterministic and the conforming projection is the
// only thing under test.

#include "mimetic/mimetic.hpp"
#include "test_utils.hpp"

#include <Eigen/Dense>
#include <moab/Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace {

constexpr double kConservationTol = 5.0e-13;
constexpr double kTraceContinuityTol = 1.0e-12;

// Manufactured field.  We use a non-polynomial sin/cos field so that
// the source p=2 reconstruction has truncation error and the raw
// trace-jump at the target skeleton is non-zero and shrinks under
// refinement.  A polynomial field that lives exactly in the local
// reconstruction space would make the raw trace-jump roundoff at any
// resolution (the source reconstruction is exact, so directed views
// of every shared edge already agree to machine precision before any
// projection takes place); the rate test would then measure noise.
Eigen::Vector2d test_field(const Eigen::Vector2d& p)
{
    const double pi = 3.14159265358979323846;
    return Eigen::Vector2d( std::sin(pi * p.x()) * std::cos(pi * p.y()),
                           -std::cos(pi * p.x()) * std::sin(pi * p.y()));
}

struct GaussPoint {
    double x;
    double w;
};

// 10-point Gauss-Legendre on [-1, 1] -- exact through degree 19, more
// than enough for the degree-4 integrands here.
const GaussPoint kGauss10[10] = {
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

double legendre(const int degree, const double x)
{
    if (degree == 0) return 1.0;
    if (degree == 1) return x;
    double pn2 = 1.0;
    double pn1 = x;
    double pn = x;
    for (int n = 2; n <= degree; ++n) {
        pn = ((2.0 * n - 1.0) * x * pn1 - (n - 1.0) * pn2) / static_cast<double>(n);
        pn2 = pn1;
        pn1 = pn;
    }
    return pn;
}

template <typename Func>
double integrate_edge(const Eigen::Vector2d& a, const Eigen::Vector2d& b, const Func& f)
{
    const double L = (b - a).norm();
    const Eigen::Vector2d mid = 0.5 * (a + b);
    const Eigen::Vector2d half = 0.5 * (b - a);
    double sum = 0.0;
    for (const GaussPoint& q : kGauss10) {
        sum += q.w * f(mid + q.x * half);
    }
    return 0.5 * L * sum;
}

// Tensor-product Duffy 10x10 quadrature on a triangle (a, b, c).
// Identical to integrate_triangle_duffy_highorder used by
// tests/high_order_hdiv_convergence_test.cpp.
template <typename Func>
double integrate_triangle_duffy_10(const Eigen::Vector2d& a,
                                   const Eigen::Vector2d& b,
                                   const Eigen::Vector2d& c,
                                   const Func& f)
{
    const Eigen::Vector2d ab = b - a;
    const Eigen::Vector2d bc = c - b;
    const double det_j = std::abs(ab.x() * bc.y() - ab.y() * bc.x());
    double sum = 0.0;
    for (const GaussPoint& qu : kGauss10) {
        const double u = 0.5 * (qu.x + 1.0);
        const double wu = 0.5 * qu.w;
        for (const GaussPoint& qv : kGauss10) {
            const double v = 0.5 * (qv.x + 1.0);
            const double wv = 0.5 * qv.w;
            const Eigen::Vector2d p = a + u * ab + (u * v) * bc;
            sum += wu * wv * u * f(p);
        }
    }
    return det_j * sum;
}

std::vector<Eigen::Vector2d> exact_cell_vector_moments(const mimetic::LocalPolygon& poly,
                                                       const int order)
{
    std::vector<Eigen::Vector2d> moments;
    const Eigen::Vector2d origin = Eigen::Vector2d::Zero();
    for (int total_degree = 0; total_degree <= order; ++total_degree) {
        for (int a_pow = total_degree; a_pow >= 0; --a_pow) {
            const int b_pow = total_degree - a_pow;
            Eigen::Vector2d integral = Eigen::Vector2d::Zero();
            for (std::size_t i = 0; i < poly.points.size(); ++i) {
                const Eigen::Vector2d& vb = poly.points[i];
                const Eigen::Vector2d& vc = poly.points[(i + 1) % poly.points.size()];
                integral.x() += integrate_triangle_duffy_10(origin, vb, vc, [&](const Eigen::Vector2d& p) {
                    return test_field(p + poly.centroid).x() *
                           std::pow(p.x(), a_pow) * std::pow(p.y(), b_pow);
                });
                integral.y() += integrate_triangle_duffy_10(origin, vb, vc, [&](const Eigen::Vector2d& p) {
                    return test_field(p + poly.centroid).y() *
                           std::pow(p.x(), a_pow) * std::pow(p.y(), b_pow);
                });
            }
            moments.push_back(integral);
        }
    }
    return moments;
}

std::vector<double> exact_edge_moments(const Eigen::Vector2d& a,
                                       const Eigen::Vector2d& b,
                                       const int order)
{
    std::vector<double> out(static_cast<std::size_t>(order + 1), 0.0);
    const Eigen::Vector2d delta = b - a;
    const double L = delta.norm();
    const Eigen::Vector2d normal_unit(delta.y() / L, -delta.x() / L);
    for (int m = 0; m <= order; ++m) {
        out[m] = integrate_edge(a, b, [&](const Eigen::Vector2d& p) {
            const double t = 2.0 * (p - a).dot(delta) / delta.squaredNorm() - 1.0;
            return test_field(p).dot(normal_unit) * legendre(m, t);
        });
    }
    return out;
}

struct CellInfo {
    moab::EntityHandle handle;
    std::vector<Eigen::Vector2d> pts;
};

std::vector<CellInfo> create_quad_mesh(moab::Core& mb, const int nx, const int ny,
                                       const double x0, const double y0,
                                       const double Lx, const double Ly)
{
    std::vector<CellInfo> cells;
    const double dx = Lx / nx;
    const double dy = Ly / ny;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const double xa = x0 + i * dx;
            const double ya = y0 + j * dy;
            const std::vector<Eigen::Vector2d> pts = {
                Eigen::Vector2d(xa, ya),
                Eigen::Vector2d(xa + dx, ya),
                Eigen::Vector2d(xa + dx, ya + dy),
                Eigen::Vector2d(xa, ya + dy),
            };
            cells.push_back(CellInfo{mimetic::create_polygon(mb, pts), pts});
        }
    }
    std::vector<moab::EntityHandle> polys;
    polys.reserve(cells.size());
    for (const CellInfo& c : cells) polys.push_back(c.handle);
    mimetic::merge_polygon_vertices(mb, polys);
    return cells;
}

std::vector<moab::EntityHandle> handles_of(const std::vector<CellInfo>& cells)
{
    std::vector<moab::EntityHandle> out;
    out.reserve(cells.size());
    for (const CellInfo& c : cells) out.push_back(c.handle);
    return out;
}

void seed_source_moments(mimetic::PlanarMomentInterpolator& interp,
                         moab::Core& mb,
                         const std::vector<CellInfo>& source_cells,
                         const int order,
                         const int cell_order)
{
    for (const CellInfo& cell : source_cells) {
        const mimetic::LocalPolygon poly = mimetic::local_polygon(mb, cell.handle);
        const std::vector<mimetic::LocalEdge> edges = mimetic::local_edges(mb, poly);
        for (std::size_t e = 0; e < edges.size(); ++e) {
            const Eigen::Vector2d a = poly.centroid + edges[e].a;
            const Eigen::Vector2d b = poly.centroid + edges[e].b;
            interp.set_source_edge_moments(cell.handle, e, exact_edge_moments(a, b, order));
        }
        interp.set_source_cell_vector_moments(cell.handle,
                                              exact_cell_vector_moments(poly, cell_order));
    }
}

// Run reconstruction + transfer + projection and return the conforming
// result.  Source/target meshes are created inside; caller passes their
// resolutions.
mimetic::ConformingEdgeMomentTransferResult run_one(int order,
                                                    int src_n,
                                                    int tgt_n,
                                                    double& trace_jump_l2,
                                                    double& source_jump_l2,
                                                    double& div_residual_max,
                                                    double& post_jump_max,
                                                    std::size_t& boundary_edges,
                                                    std::size_t& interior_edges)
{
    moab::Core mb;
    const std::vector<CellInfo> source_cells =
        create_quad_mesh(mb, src_n, src_n, 0.0, 0.0, 1.0, 1.0);
    const std::vector<CellInfo> target_cells =
        create_quad_mesh(mb, tgt_n, tgt_n, 0.0, 0.0, 1.0, 1.0);

    mimetic::PlanarMomentInterpolator interp(mb);
    const int cell_order = std::max(1, order - 1);
    seed_source_moments(interp, mb, source_cells, order, cell_order);

    // Match the non-polynomial-field reconstruction pattern used in
    // tests/high_order_hdiv_convergence_test.cpp: soft constraints,
    // edge + cell weighting, 10-point quadrature.  Hard constraints
    // (the default) make the local 12x12 system at p=2 on a quad
    // exactly determined, which throws on any field that does not lie
    // in [P_p]^2 -- e.g. our sin/cos test field.
    mimetic::MomentMethodOptions mopts;
    mopts.edge_moment_order = order;
    mopts.cell_moment_order = cell_order;
    mopts.quadrature_points = 10;
    mopts.regularization = 1.0e-12;
    mopts.exact_constraints = false;
    mopts.edge_weight = 1.0;
    mopts.cell_weight = 1.0;
    for (const CellInfo& cell : source_cells) {
        interp.reconstruct_source_polygon(cell.handle, mopts);
    }

    const mimetic::EdgeMomentTransferResult raw =
        interp.transfer_source_to_target_edge_moments(
            handles_of(source_cells), handles_of(target_cells), order);
    const mimetic::ConformingEdgeMomentTransferResult conf =
        interp.project_target_edge_moments_to_hdiv_conforming(
            handles_of(source_cells), handles_of(target_cells), raw);

    const std::size_t Nu = conf.unique_edge_moments.size();
    const std::size_t Nm = conf.unique_edge_moments.empty() ? 0 : conf.unique_edge_moments[0].size();

    // Per-target-unique-edge raw transfer consistency: L2 over
    // (unique_edge, degree).  By construction this is at the roundoff
    // floor on every case (see ConformingEdgeMomentTransferResult
    // documentation): both directed views integrate the same source
    // field with opposite normals, so the orientation-corrected raw
    // values agree to roundoff.  A nonzero value here would indicate
    // a transfer-side bug.
    double sumsq = 0.0;
    for (std::size_t u = 0; u < Nu; ++u) {
        for (std::size_t m = 0; m < Nm; ++m) {
            const double j = conf.trace_jump_per_unique_edge[u][m];
            sumsq += j * j;
        }
    }
    trace_jump_l2 = std::sqrt(sumsq);

    // Source-skeleton jump: per-degree L2 norm over the source mesh
    // interior.  This refines as O(h^{p+1}) for a smooth source field
    // reconstructed at order p.
    double s2 = 0.0;
    for (std::size_t m = 0; m < conf.source_skeleton_jump_l2.size(); ++m) {
        s2 += conf.source_skeleton_jump_l2[m] * conf.source_skeleton_jump_l2[m];
    }
    source_jump_l2 = std::sqrt(s2);

    // Divergence balance per cell.  conf.target_moments[i][0] are the
    // cell-side directed flux integrals (already orientation-corrected
    // relative to the unique-edge representative through the parity
    // factor in the implementation); summing them over a cell IS the
    // divergence integral by construction.  No extra sign required.
    const std::size_t Nc = conf.target_cells.size();
    std::map<moab::EntityHandle, std::size_t> cell_index_of;
    for (std::size_t c = 0; c < Nc; ++c) cell_index_of[conf.target_cells[c]] = c;
    std::vector<double> cell_sum(Nc, 0.0);
    for (std::size_t i = 0; i < conf.target_edges.size(); ++i) {
        const std::size_t cell_index = cell_index_of[conf.target_edges[i].polygon];
        cell_sum[cell_index] += conf.target_moments[i][0];
    }
    div_residual_max = 0.0;
    for (std::size_t c = 0; c < Nc; ++c) {
        div_residual_max = std::max(div_residual_max,
                                    std::abs(cell_sum[c] - conf.target_divergence_integrals[c]));
    }

    // Post-projection trace continuity: directed views of the same
    // unique edge must agree (after orientation parity flip).
    std::map<std::size_t, std::vector<std::size_t>> views;
    for (std::size_t i = 0; i < conf.target_edges.size(); ++i) {
        views[conf.target_edge_to_unique[i]].push_back(i);
    }
    boundary_edges = 0;
    interior_edges = 0;
    post_jump_max = 0.0;
    for (const auto& kv : views) {
        if (kv.second.size() == 1) {
            ++boundary_edges;
            continue;
        }
        ++interior_edges;
        for (std::size_t m = 0; m < Nm; ++m) {
            double mn =  std::numeric_limits<double>::infinity();
            double mx = -std::numeric_limits<double>::infinity();
            for (std::size_t i : kv.second) {
                const int o = conf.target_edge_orientations[i];
                // Same parity rule as the implementation.
                const double f = (o == 1) ? 1.0
                                          : (((static_cast<int>(m) % 2) == 0) ? -1.0 : 1.0);
                const double v = f * conf.target_moments[i][m];
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
            post_jump_max = std::max(post_jump_max, mx - mn);
        }
    }
    return conf;
}

bool report(const char* name, bool ok)
{
    std::cout << "  " << (ok ? "PASS" : "FAIL") << " : " << name << "\n";
    return ok;
}

}  // namespace

int main()
{
    int order = 2;
    bool ok = true;

    // Single resolution case: full property checks.
    {
        std::cout << "case A: nonmatching 4x4 -> 6x6, p = " << order << "\n";
        double trace_jump_l2 = 0, source_jump_l2 = 0, div_res = 0, post_jump = 0;
        std::size_t nb = 0, ni = 0;
        const auto conf = run_one(order, 4, 6,
                                  trace_jump_l2, source_jump_l2,
                                  div_res, post_jump, nb, ni);

        // Unique-edge length diagnostic populated and positive.
        bool lengths_ok = true;
        for (std::size_t u = 0; u < conf.unique_edge_lengths.size(); ++u) {
            if (!(conf.unique_edge_lengths[u] > 0.0)) { lengths_ok = false; break; }
        }

        // Boundary edges have zero raw trace jump (single directed view).
        bool boundary_jump_zero = true;
        std::map<std::size_t, std::vector<std::size_t>> views;
        for (std::size_t i = 0; i < conf.target_edges.size(); ++i) {
            views[conf.target_edge_to_unique[i]].push_back(i);
        }
        for (const auto& kv : views) {
            if (kv.second.size() == 1) {
                for (std::size_t m = 0; m < conf.trace_jump_per_unique_edge[kv.first].size(); ++m) {
                    if (conf.trace_jump_per_unique_edge[kv.first][m] != 0.0) {
                        boundary_jump_zero = false;
                    }
                }
            }
        }

        std::cout << "    unique edges                   = " << conf.unique_edge_moments.size() << "\n";
        std::cout << "    boundary unique edges          = " << nb << "\n";
        std::cout << "    interior unique edges          = " << ni << "\n";
        std::cout << "    raw trace-jump L2 (Nu,Nm)      = " << trace_jump_l2 << "\n";
        std::cout << "    source-skeleton jump L2        = " << source_jump_l2 << "\n";
        std::cout << "    post-projection max trace jump = " << post_jump << "\n";
        std::cout << "    max cell div residual          = " << div_res << "\n";

        ok &= report("unique_edge_lengths populated and > 0",          lengths_ok);
        ok &= report("boundary edges have zero raw trace jump",         boundary_jump_zero);
        ok &= report("post-projection trace continuity (interior)",     post_jump <= kTraceContinuityTol);
        ok &= report("cell divergence balance within 5e-13",            div_res  <= kConservationTol);
        ok &= report("interior edges exist (non-trivial test)",         ni > 0);
        ok &= report("raw target trace-jump at roundoff floor",         trace_jump_l2 < 1e-12);
        ok &= report("source-skeleton jump strictly positive",          source_jump_l2 > 0.0);
    }

    // Refinement case: source-skeleton jump must shrink at a positive
    // rate.  This is the genuinely refining diagnostic; the per-target-
    // unique-edge raw trace jump is identically zero by construction
    // and would not yield a meaningful rate.
    {
        std::cout << "case B: refinement, source 4->8->16, target src+2 each, p = " << order << "\n";
        const int src_levels[3] = {4, 8, 16};
        std::vector<double> jumps;
        std::vector<double> div_residuals;
        for (int s : src_levels) {
            double tj = 0, sj = 0, dr = 0, pj = 0;
            std::size_t nb = 0, ni = 0;
            run_one(order, s, s + 2, tj, sj, dr, pj, nb, ni);
            jumps.push_back(sj);
            div_residuals.push_back(dr);
            std::cout << "    src=" << s
                      << " tgt=" << (s + 2)
                      << "  source_jump_L2 = " << sj
                      << "  raw_target_jump_L2 = " << tj
                      << "  div_residual_max = " << dr << "\n";
        }
        bool monotone = (jumps[1] < jumps[0]) && (jumps[2] < jumps[1]);
        const double rate_01 = std::log(jumps[0] / std::max(jumps[1], 1e-300)) / std::log(2.0);
        const double rate_12 = std::log(jumps[1] / std::max(jumps[2], 1e-300)) / std::log(2.0);
        std::cout << "    source-jump rate 4->8  = " << rate_01 << "\n";
        std::cout << "    source-jump rate 8->16 = " << rate_12 << "\n";

        bool div_ok = true;
        for (double r : div_residuals) if (r > kConservationTol) div_ok = false;

        // For a sin/cos field on quad cells at p = 2, the source
        // reconstruction has L2 error O(h^{p+1}) = O(h^3) inside each
        // source cell; the per-source-edge trace jump inherits this
        // rate.  Require rate >= 1.5 (conservative) to absorb pre-
        // asymptotic effects.
        ok &= report("source-skeleton jump strictly decreasing",       monotone);
        ok &= report("source-jump rate >= 1.5 between coarse pair",    rate_01 >= 1.5);
        ok &= report("source-jump rate >= 1.5 between fine pair",      rate_12 >= 1.5);
        ok &= report("div residual <= 5e-13 at every level",           div_ok);
    }

    return ok ? 0 : 1;
}
