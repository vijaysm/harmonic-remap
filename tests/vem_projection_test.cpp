#include "mimetic/mimetic.hpp"
#include "test_utils.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace mimetic;

static constexpr double PI = 3.14159265358979323846;

using FieldFunc = Eigen::Vector2d (*)(const Eigen::Vector2d&);

struct GaussPoint {
    double x;
    double w;
};

static const GaussPoint gauss10_table[10] = {
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

template <typename Func>
double integrate_edge_highorder(const Eigen::Vector2d& a, const Eigen::Vector2d& b, const Func& func)
{
    const double length = (b - a).norm();
    const Eigen::Vector2d mid = 0.5 * (a + b);
    const Eigen::Vector2d half_delta = 0.5 * (b - a);
    double sum = 0.0;
    for (const GaussPoint& q : gauss10_table) {
        sum += q.w * func(mid + q.x * half_delta);
    }
    return 0.5 * length * sum;
}

template <typename Func>
double integrate_triangle_duffy_highorder(const Eigen::Vector2d& a,
                                          const Eigen::Vector2d& b,
                                          const Eigen::Vector2d& c,
                                          const Func& func)
{
    const Eigen::Vector2d ab = b - a;
    const Eigen::Vector2d bc = c - b;
    const double det_j = std::abs(ab.x() * bc.y() - ab.y() * bc.x());
    double sum = 0.0;
    for (const GaussPoint& qu : gauss10_table) {
        const double u = 0.5 * (qu.x + 1.0);
        const double wu = 0.5 * qu.w;
        for (const GaussPoint& qv : gauss10_table) {
            const double v = 0.5 * (qv.x + 1.0);
            const double wv = 0.5 * qv.w;
            const Eigen::Vector2d p = a + u * ab + (u * v) * bc;
            sum += wu * wv * u * func(p);
        }
    }
    return det_j * sum;
}

double legendre(const int degree, const double x)
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

double halton(int index, int base)
{
    double result = 0.0;
    double f = 1.0 / base;
    int i = index;
    while (i > 0) {
        result += f * (i % base);
        i /= base;
        f /= base;
    }
    return result;
}

std::vector<Eigen::Vector2d> halton_seeds(const int count, const int offset)
{
    std::vector<Eigen::Vector2d> seeds;
    seeds.reserve(static_cast<std::size_t>(count));
    for (int i = 1; i <= count; ++i) {
        const int k = i + offset;
        seeds.emplace_back(0.05 + 0.90 * halton(k, 2),
                           0.05 + 0.90 * halton(k, 3));
    }
    return seeds;
}

std::vector<Eigen::Vector2d> clip_by_halfplane(const std::vector<Eigen::Vector2d>& polygon,
                                               const Eigen::Vector2d& normal,
                                               const double offset)
{
    std::vector<Eigen::Vector2d> output;
    if (polygon.empty()) {
        return output;
    }

    auto inside = [&](const Eigen::Vector2d& p) { return normal.dot(p) <= offset + 1.0e-13; };
    auto intersect = [&](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
        const double da = normal.dot(a) - offset;
        const double db = normal.dot(b) - offset;
        return a + (da / (da - db)) * (b - a);
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

std::vector<Eigen::Vector2d> voronoi_cell_polygon(const Eigen::Vector2d& seed,
                                                  const std::vector<Eigen::Vector2d>& seeds)
{
    std::vector<Eigen::Vector2d> cell = {
        Eigen::Vector2d(0.0, 0.0),
        Eigen::Vector2d(1.0, 0.0),
        Eigen::Vector2d(1.0, 1.0),
        Eigen::Vector2d(0.0, 1.0),
    };
    for (const Eigen::Vector2d& other : seeds) {
        if ((other - seed).squaredNorm() < 1.0e-24) {
            continue;
        }
        const Eigen::Vector2d normal = 2.0 * (other - seed);
        const double offset = other.squaredNorm() - seed.squaredNorm();
        cell = clip_by_halfplane(cell, normal, offset);
        if (cell.size() < 3) {
            return {};
        }
    }
    return cell;
}

double polygon_area(const std::vector<Eigen::Vector2d>& polygon)
{
    return std::abs(mimetic::signed_area(polygon));
}

struct CellInfo {
    moab::EntityHandle handle;
    std::vector<Eigen::Vector2d> points;
};

std::vector<CellInfo> create_quad_mesh(moab::Core& mb, const int nx, const int ny)
{
    std::vector<CellInfo> cells;
    const double dx = 1.0 / nx;
    const double dy = 1.0 / ny;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const double x0 = i * dx;
            const double y0 = j * dy;
            const std::vector<Eigen::Vector2d> pts = {
                Eigen::Vector2d(x0, y0),
                Eigen::Vector2d(x0 + dx, y0),
                Eigen::Vector2d(x0 + dx, y0 + dy),
                Eigen::Vector2d(x0, y0 + dy),
            };
            cells.push_back(CellInfo{mimetic::create_polygon(mb, pts), pts});
        }
    }
    std::vector<moab::EntityHandle> polygons;
    polygons.reserve(cells.size());
    for (const CellInfo& cell : cells) {
        polygons.push_back(cell.handle);
    }
    mimetic::merge_polygon_vertices(mb, polygons);
    return cells;
}

std::vector<CellInfo> create_voronoi_mesh(moab::Core& mb,
                                          const std::vector<Eigen::Vector2d>& seeds)
{
    std::vector<CellInfo> cells;
    for (const Eigen::Vector2d& seed : seeds) {
        const std::vector<Eigen::Vector2d> polygon = voronoi_cell_polygon(seed, seeds);
        if (polygon.size() < 3 || polygon_area(polygon) < 1.0e-12) {
            continue;
        }
        cells.push_back(CellInfo{mimetic::create_polygon(mb, polygon), polygon});
    }
    std::vector<moab::EntityHandle> polygons;
    polygons.reserve(cells.size());
    for (const CellInfo& cell : cells) {
        polygons.push_back(cell.handle);
    }
    mimetic::merge_polygon_vertices(mb, polygons);
    return cells;
}

std::vector<moab::EntityHandle> handles(const std::vector<CellInfo>& cells)
{
    std::vector<moab::EntityHandle> result;
    result.reserve(cells.size());
    for (const CellInfo& cell : cells) {
        result.push_back(cell.handle);
    }
    return result;
}

Eigen::Vector2d smooth_variable_divergence_field(const Eigen::Vector2d& p)
{
    const double x = p.x();
    const double y = p.y();
    return Eigen::Vector2d(
        std::exp(0.20 * x) * (std::sin(PI * x) + 0.20 * y * y) + 0.05 * x * x * x,
        std::exp(0.20 * y) * (std::cos(PI * y) + 0.20 * x * y) + 0.05 * y * y * y);
}

std::vector<double> exact_edge_moments_absolute(const Eigen::Vector2d& a,
                                                const Eigen::Vector2d& b,
                                                const int order,
                                                const FieldFunc& field)
{
    std::vector<double> moments(static_cast<std::size_t>(order + 1), 0.0);
    const Eigen::Vector2d delta = b - a;
    const double length = delta.norm();
    const double denom = delta.squaredNorm();
    const Eigen::Vector2d normal(delta.y(), -delta.x());
    for (int degree = 0; degree <= order; ++degree) {
        moments[degree] = integrate_edge_highorder(a, b, [&](const Eigen::Vector2d& p) {
            const double t = (denom > 0.0) ? (2.0 * (p - a).dot(delta) / denom - 1.0) : 0.0;
            return field(p).dot(normal / length) * legendre(degree, t);
        });
    }
    return moments;
}

std::vector<Eigen::Vector2d> exact_cell_vector_moments(const mimetic::LocalPolygon& poly,
                                                       const int order,
                                                       const FieldFunc& field)
{
    std::vector<Eigen::Vector2d> moments;
    const Eigen::Vector2d origin = Eigen::Vector2d::Zero();
    for (int total_degree = 0; total_degree <= order; ++total_degree) {
        for (int a_pow = total_degree; a_pow >= 0; --a_pow) {
            const int b_pow = total_degree - a_pow;
            Eigen::Vector2d integral = Eigen::Vector2d::Zero();
            for (std::size_t i = 0; i < poly.points.size(); ++i) {
                const Eigen::Vector2d a = origin;
                const Eigen::Vector2d b = poly.points[i];
                const Eigen::Vector2d c = poly.points[(i + 1) % poly.points.size()];
                integral.x() += integrate_triangle_duffy_highorder(a, b, c, [&](const Eigen::Vector2d& p) {
                    return field(p + poly.centroid).x() * std::pow(p.x(), a_pow) * std::pow(p.y(), b_pow);
                });
                integral.y() += integrate_triangle_duffy_highorder(a, b, c, [&](const Eigen::Vector2d& p) {
                    return field(p + poly.centroid).y() * std::pow(p.x(), a_pow) * std::pow(p.y(), b_pow);
                });
            }
            moments.push_back(integral);
        }
    }
    return moments;
}

void set_source_moments_vem(moab::Core& mb,
                            mimetic::PlanarMomentInterpolator& interpolator,
                            const moab::EntityHandle polygon,
                            const int order,
                            const int cell_order,
                            const FieldFunc& field)
{
    const mimetic::LocalPolygon poly = mimetic::local_polygon(mb, polygon);
    const std::vector<mimetic::LocalEdge> edges = mimetic::local_edges(mb, poly);
    for (std::size_t edge_index = 0; edge_index < edges.size(); ++edge_index) {
        const Eigen::Vector2d a = poly.centroid + edges[edge_index].a;
        const Eigen::Vector2d b = poly.centroid + edges[edge_index].b;
        interpolator.set_source_edge_moments(polygon, edge_index,
                                             exact_edge_moments_absolute(a, b, order, field));
    }
    if (cell_order >= 0) {
        interpolator.set_source_cell_vector_moments(polygon,
                                                    exact_cell_vector_moments(poly, cell_order, field));
    }
}

// ============================================================================
// Test 1: Trace operator diagnostic
// ============================================================================

bool test_trace_operator_diagnostic()
{
    std::cout << "=== Trace operator diagnostic ===\n";
    bool pass = true;

    moab::Core mb;
    const std::vector<Eigen::Vector2d> seeds = halton_seeds(20, 0);
    const std::vector<CellInfo> cells = create_voronoi_mesh(mb, seeds);

    for (int order = 1; order <= 3; ++order) {
        std::cout << "\n  p=" << order << ":\n";
        double max_kappa = 0.0;
        double min_kappa = std::numeric_limits<double>::infinity();
        double avg_kappa = 0.0;
        int count = 0;

        for (const CellInfo& cell : cells) {
            const TraceOperatorDiagnostic diag =
                diagnose_trace_operator(mb, cell.handle, order);

            if (diag.condition_number > max_kappa) {
                max_kappa = diag.condition_number;
            }
            if (diag.condition_number < min_kappa) {
                min_kappa = diag.condition_number;
            }
            avg_kappa += diag.condition_number;
            ++count;

            if (!std::isfinite(diag.condition_number)) {
                std::cout << "    cell " << cell.handle
                          << " has infinite condition number (edges=" << diag.num_edges
                          << " rows=" << diag.constraint_rows
                          << " cols=" << diag.basis_dim << ")\n";
            }
        }
        avg_kappa /= count;

        std::cout << "    cells=" << count
                  << "  kappa_min=" << std::scientific << std::setprecision(2) << min_kappa
                  << "  kappa_avg=" << avg_kappa
                  << "  kappa_max=" << max_kappa << "\n";

        if (!std::isfinite(max_kappa) || max_kappa < 1.0) {
            std::cout << "    FAIL: unexpected condition numbers\n";
            pass = false;
        }
    }

    moab::Core mb_quad;
    const std::vector<CellInfo> quads = create_quad_mesh(mb_quad, 4, 4);
    std::cout << "\n  Quad reference (4x4):\n";
    for (int order = 1; order <= 3; ++order) {
        double max_kappa = 0.0;
        for (const CellInfo& cell : quads) {
            const TraceOperatorDiagnostic diag =
                diagnose_trace_operator(mb_quad, cell.handle, order);
            if (diag.condition_number > max_kappa) {
                max_kappa = diag.condition_number;
            }
        }
        std::cout << "    p=" << order << "  kappa_max=" << std::scientific << std::setprecision(2) << max_kappa << "\n";
    }

    return pass;
}

// ============================================================================
// Test 2: VEM projection — single-cell polynomial recovery
// ============================================================================

bool test_vem_single_cell_recovery()
{
    std::cout << "\n=== VEM single-cell polynomial recovery ===\n";
    bool pass = true;

    auto linear_field = [](const Eigen::Vector2d& p) -> Eigen::Vector2d {
        return Eigen::Vector2d(1.0 + 2.0 * p.x() + 0.5 * p.y(),
                               -0.5 + p.x() + 3.0 * p.y());
    };

    auto quadratic_field = [](const Eigen::Vector2d& p) -> Eigen::Vector2d {
        return Eigen::Vector2d(p.x() * p.x() + 0.5 * p.x() * p.y(),
                               p.y() * p.y() - 0.3 * p.x() * p.y());
    };

    struct TestCase {
        std::string name;
        FieldFunc field;
        int order;
        double tolerance;
    };

    const std::vector<TestCase> cases = {
        {"linear on pentagon, p=1", [](const Eigen::Vector2d& p) -> Eigen::Vector2d {
            return Eigen::Vector2d(1.0 + 2.0 * p.x() + 0.5 * p.y(),
                                   -0.5 + p.x() + 3.0 * p.y());
        }, 1, 1.0e-10},
        {"linear on pentagon, p=2", [](const Eigen::Vector2d& p) -> Eigen::Vector2d {
            return Eigen::Vector2d(1.0 + 2.0 * p.x() + 0.5 * p.y(),
                                   -0.5 + p.x() + 3.0 * p.y());
        }, 2, 1.0e-10},
        {"quadratic on pentagon, p=2", [](const Eigen::Vector2d& p) -> Eigen::Vector2d {
            return Eigen::Vector2d(p.x() * p.x() + 0.5 * p.x() * p.y(),
                                   p.y() * p.y() - 0.3 * p.x() * p.y());
        }, 2, 1.0e-10},
    };

    const std::vector<Eigen::Vector2d> pentagon = {
        Eigen::Vector2d(0.3, 0.2),
        Eigen::Vector2d(0.7, 0.1),
        Eigen::Vector2d(0.9, 0.5),
        Eigen::Vector2d(0.6, 0.9),
        Eigen::Vector2d(0.2, 0.6),
    };

    for (const TestCase& tc : cases) {
        std::cout << "  " << tc.name << ": ";

        moab::Core mb;
        const moab::EntityHandle poly_handle = create_polygon(mb, pentagon);
        std::vector<moab::EntityHandle> polygons = {poly_handle};
        merge_polygon_vertices(mb, polygons);

        PlanarMomentInterpolator interpolator(mb);
        MomentMethodOptions opts;
        opts.edge_moment_order = tc.order;
        opts.cell_moment_order = tc.order;
        opts.quadrature_points = 10;
        opts.reconstruction_mode = ReconstructionMode::VemProjection;

        set_source_moments_vem(mb, interpolator, poly_handle, tc.order,
                               tc.order, tc.field);

        const MomentReconstruction recon = interpolator.reconstruct_source_polygon(poly_handle, opts);
        const LocalPolygon lpoly = local_polygon(mb, poly_handle);

        double max_error = 0.0;
        const int ntest = 5;
        for (int i = 0; i < ntest; ++i) {
            for (int j = 0; j < ntest; ++j) {
                const double s = (static_cast<double>(i) + 0.5) / ntest;
                const double t = (static_cast<double>(j) + 0.5) / ntest;
                const Eigen::Vector2d test_pt = lpoly.centroid +
                    s * (lpoly.points[0]) + t * (lpoly.points[1]);

                const Eigen::Vector2d computed = interpolator.velocity(recon, test_pt - lpoly.centroid);
                const Eigen::Vector2d exact = tc.field(test_pt);
                const double err = (computed - exact).norm();
                max_error = std::max(max_error, err);
            }
        }

        std::cout << "max_error=" << std::scientific << std::setprecision(3) << max_error;
        if (max_error > tc.tolerance) {
            std::cout << " FAIL\n";
            pass = false;
        } else {
            std::cout << " OK\n";
        }
    }

    return pass;
}

// ============================================================================
// Test 3: VEM vs LS convergence comparison
// ============================================================================

struct ConvergenceMetrics {
    int order = 0;
    std::string method;
    std::string domain;
    double h = 0.0;
    double l2_flux_rel = 0.0;
};

double convergence_rate(const double e_coarse, const double e_fine, const double h_coarse, const double h_fine)
{
    if (e_coarse < 1.0e-30 || e_fine < 1.0e-30) {
        return 0.0;
    }
    return std::log(e_coarse / e_fine) / std::log(h_coarse / h_fine);
}

ConvergenceMetrics run_vem_case(const std::string& method,
                                const std::string& domain,
                                moab::Core& mb,
                                const std::vector<CellInfo>& source_cells,
                                const std::vector<CellInfo>& target_cells,
                                const int order,
                                const double h,
                                const ReconstructionMode mode)
{
    PlanarMomentInterpolator interpolator(mb);
    MomentMethodOptions options;
    options.edge_moment_order = order;
    options.cell_moment_order = (mode == ReconstructionMode::VemProjection) ? order : std::max(0, order - 2);
    options.quadrature_points = 10;
    options.regularization = 1.0e-12;
    options.reconstruction_mode = mode;

    if (mode == ReconstructionMode::SplitBasis) {
        options.exact_constraints = false;
        options.edge_weight = 1.0;
        options.cell_weight = 1.0;
    } else {
        options.exact_constraints = true;
        options.cell_weight = 1.0;
    }

    for (const CellInfo& source : source_cells) {
        set_source_moments_vem(mb, interpolator, source.handle, order,
                               options.cell_moment_order, smooth_variable_divergence_field);
        interpolator.reconstruct_source_polygon(source.handle, options);
    }

    const EdgeMomentTransferResult transfer =
        interpolator.transfer_source_to_target_edge_moments(handles(source_cells), handles(target_cells), order);

    double flux_num = 0.0;
    double flux_den = 0.0;
    std::size_t dof = 0;
    for (std::size_t cell_index = 0; cell_index < target_cells.size(); ++cell_index) {
        const CellInfo& target = target_cells[cell_index];
        const LocalPolygon poly = local_polygon(mb, target.handle);
        const std::vector<LocalEdge> edges = local_edges(mb, poly);
        for (std::size_t edge_index = 0; edge_index < edges.size(); ++edge_index) {
            const Eigen::Vector2d a = poly.centroid + edges[edge_index].a;
            const Eigen::Vector2d b = poly.centroid + edges[edge_index].b;
            const std::vector<double> exact =
                exact_edge_moments_absolute(a, b, order, smooth_variable_divergence_field);

            for (int m = 0; m <= order; ++m) {
                const double computed = transfer.target_moments[dof][m];
                const double err = computed - exact[m];
                flux_num += err * err;
                flux_den += exact[m] * exact[m];
            }
            ++dof;
        }
    }

    ConvergenceMetrics metrics;
    metrics.order = order;
    metrics.method = method;
    metrics.domain = domain;
    metrics.h = h;
    metrics.l2_flux_rel = (flux_den > 1.0e-30) ? std::sqrt(flux_num / flux_den) : std::sqrt(flux_num);
    return metrics;
}

bool test_vem_convergence()
{
    std::cout << "\n=== VEM vs SplitBasis convergence comparison ===\n";
    bool pass = true;

    const int refinements[] = {4, 8, 16};
    const int voronoi_counts[] = {16, 64, 256};

    for (int order = 1; order <= 3; ++order) {
        std::cout << "\n  p=" << order << ":\n";

        std::vector<ConvergenceMetrics> vem_quad_metrics;
        std::vector<ConvergenceMetrics> ls_quad_metrics;
        std::vector<ConvergenceMetrics> vem_vor_metrics;
        std::vector<ConvergenceMetrics> ls_vor_metrics;

        for (int ref : refinements) {
            moab::Core mb_source, mb_target;
            const int source_n = ref;
            const int target_n = ref + ref / 2;
            const double h = 1.0 / source_n;

            const std::vector<CellInfo> source = create_quad_mesh(mb_source, source_n, source_n);
            const std::vector<CellInfo> target = create_quad_mesh(mb_source, target_n, target_n);

            vem_quad_metrics.push_back(
                run_vem_case("VEM", "quad", mb_source, source, target, order, h,
                             ReconstructionMode::VemProjection));
            ls_quad_metrics.push_back(
                run_vem_case("LS", "quad", mb_source, source, target, order, h,
                             ReconstructionMode::SplitBasis));
        }

        for (int count : voronoi_counts) {
            moab::Core mb;
            const double h = 1.0 / std::sqrt(static_cast<double>(count));
            const std::vector<Eigen::Vector2d> source_seeds = halton_seeds(count, 0);
            const std::vector<Eigen::Vector2d> target_seeds = halton_seeds(count + count / 2, 100);

            const std::vector<CellInfo> source = create_voronoi_mesh(mb, source_seeds);
            const std::vector<CellInfo> target = create_voronoi_mesh(mb, target_seeds);

            vem_vor_metrics.push_back(
                run_vem_case("VEM", "voronoi", mb, source, target, order, h,
                             ReconstructionMode::VemProjection));
            ls_vor_metrics.push_back(
                run_vem_case("LS", "voronoi", mb, source, target, order, h,
                             ReconstructionMode::SplitBasis));
        }

        std::cout << std::fixed << std::setprecision(2);

        auto print_rates = [&](const std::string& label, const std::vector<ConvergenceMetrics>& metrics) {
            std::cout << "    " << std::left << std::setw(20) << label;
            for (std::size_t i = 0; i < metrics.size(); ++i) {
                std::cout << "  L2=" << std::scientific << std::setprecision(2) << metrics[i].l2_flux_rel;
                if (i > 0) {
                    const double rate = convergence_rate(
                        metrics[i - 1].l2_flux_rel, metrics[i].l2_flux_rel,
                        metrics[i - 1].h, metrics[i].h);
                    std::cout << " (r=" << std::fixed << std::setprecision(2) << rate << ")";
                }
            }
            std::cout << "\n";
        };

        print_rates("VEM quad", vem_quad_metrics);
        print_rates("LS  quad", ls_quad_metrics);
        print_rates("VEM voronoi", vem_vor_metrics);
        print_rates("LS  voronoi", ls_vor_metrics);
    }

    return pass;
}

// ============================================================================
// Main
// ============================================================================

// ============================================================================
// Test 4: Topology independence — VEM recovery on triangle, hexagon, irregular
// ============================================================================

bool test_vem_topology_independence()
{
    std::cout << "\n=== VEM topology independence ===\n";
    bool pass = true;

    auto linear_field = [](const Eigen::Vector2d& p) -> Eigen::Vector2d {
        return Eigen::Vector2d(1.0 + 2.0 * p.x() + 0.5 * p.y(),
                               -0.5 + p.x() + 3.0 * p.y());
    };

    auto quadratic_field = [](const Eigen::Vector2d& p) -> Eigen::Vector2d {
        return Eigen::Vector2d(p.x() * p.x() + 0.5 * p.x() * p.y(),
                               p.y() * p.y() - 0.3 * p.x() * p.y());
    };

    struct ShapeCase {
        std::string name;
        std::vector<Eigen::Vector2d> vertices;
    };

    const std::vector<ShapeCase> shapes = {
        {"triangle", {
            Eigen::Vector2d(0.2, 0.1),
            Eigen::Vector2d(0.8, 0.2),
            Eigen::Vector2d(0.4, 0.9),
        }},
        {"hexagon", {
            Eigen::Vector2d(0.5, 0.1),
            Eigen::Vector2d(0.85, 0.25),
            Eigen::Vector2d(0.9, 0.6),
            Eigen::Vector2d(0.55, 0.9),
            Eigen::Vector2d(0.15, 0.7),
            Eigen::Vector2d(0.1, 0.35),
        }},
        {"irregular-7", {
            Eigen::Vector2d(0.3, 0.1),
            Eigen::Vector2d(0.7, 0.05),
            Eigen::Vector2d(0.95, 0.3),
            Eigen::Vector2d(0.85, 0.7),
            Eigen::Vector2d(0.5, 0.95),
            Eigen::Vector2d(0.15, 0.8),
            Eigen::Vector2d(0.05, 0.4),
        }},
    };

    struct FieldCase {
        std::string name;
        FieldFunc field;
        int order;
    };

    const std::vector<FieldCase> fields = {
        {"linear p=1", linear_field, 1},
        {"linear p=2", linear_field, 2},
        {"quadratic p=2", quadratic_field, 2},
    };

    const double tol = 1.0e-9;

    for (const ShapeCase& shape : shapes) {
        for (const FieldCase& fc : fields) {
            std::cout << "  " << shape.name << ", " << fc.name << ": ";

            moab::Core mb;
            const moab::EntityHandle poly_handle = create_polygon(mb, shape.vertices);
            std::vector<moab::EntityHandle> polygons = {poly_handle};
            merge_polygon_vertices(mb, polygons);

            PlanarMomentInterpolator interpolator(mb);
            MomentMethodOptions opts;
            opts.edge_moment_order = fc.order;
            opts.cell_moment_order = fc.order;
            opts.quadrature_points = 10;
            opts.reconstruction_mode = ReconstructionMode::VemProjection;

            set_source_moments_vem(mb, interpolator, poly_handle, fc.order,
                                   fc.order, fc.field);

            const MomentReconstruction recon =
                interpolator.reconstruct_source_polygon(poly_handle, opts);
            const LocalPolygon lpoly = local_polygon(mb, poly_handle);

            double max_error = 0.0;
            const int ntest = 5;
            for (int i = 0; i < ntest; ++i) {
                for (int j = 0; j < ntest; ++j) {
                    const double s = (static_cast<double>(i) + 0.5) / ntest;
                    const double t = (static_cast<double>(j) + 0.5) / ntest;
                    const Eigen::Vector2d test_pt = lpoly.centroid +
                        s * (lpoly.points[0]) + t * (lpoly.points[1]);

                    const Eigen::Vector2d computed =
                        interpolator.velocity(recon, test_pt - lpoly.centroid);
                    const Eigen::Vector2d exact = fc.field(test_pt);
                    const double err = (computed - exact).norm();
                    max_error = std::max(max_error, err);
                }
            }

            std::cout << "max_error=" << std::scientific << std::setprecision(3) << max_error;
            if (max_error > tol) {
                std::cout << " FAIL (>" << tol << ")\n";
                pass = false;
            } else {
                std::cout << " OK\n";
            }
        }
    }

    return pass;
}

int main()
{
    int failures = 0;

    if (!test_trace_operator_diagnostic()) {
        ++failures;
    }

    if (!test_vem_single_cell_recovery()) {
        ++failures;
    }

    if (!test_vem_convergence()) {
        ++failures;
    }

    if (!test_vem_topology_independence()) {
        ++failures;
    }

    std::cout << "\n";
    if (failures > 0) {
        std::cout << "FAILED: " << failures << " test(s) failed\n";
        return 1;
    }
    std::cout << "ALL TESTS PASSED\n";
    return 0;
}
