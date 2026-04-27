#include "mimetic/mimetic.hpp"
#include "test_utils.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
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

double legendre(const int degree, const double x)
{
    if (degree == 0) return 1.0;
    if (degree == 1) return x;
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
        seeds.emplace_back(0.05 + 0.90 * halton(k, 2), 0.05 + 0.90 * halton(k, 3));
    }
    return seeds;
}

std::vector<Eigen::Vector2d> clip_by_halfplane(const std::vector<Eigen::Vector2d>& polygon,
                                               const Eigen::Vector2d& normal,
                                               const double offset)
{
    std::vector<Eigen::Vector2d> output;
    if (polygon.empty()) return output;

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
            if (!previous_inside) output.push_back(intersect(previous, current));
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
        Eigen::Vector2d(0.0, 0.0), Eigen::Vector2d(1.0, 0.0),
        Eigen::Vector2d(1.0, 1.0), Eigen::Vector2d(0.0, 1.0),
    };
    for (const Eigen::Vector2d& other : seeds) {
        if ((other - seed).squaredNorm() < 1.0e-24) continue;
        const Eigen::Vector2d normal = 2.0 * (other - seed);
        const double offset = other.squaredNorm() - seed.squaredNorm();
        cell = clip_by_halfplane(cell, normal, offset);
        if (cell.size() < 3) return {};
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
                Eigen::Vector2d(x0, y0), Eigen::Vector2d(x0 + dx, y0),
                Eigen::Vector2d(x0 + dx, y0 + dy), Eigen::Vector2d(x0, y0 + dy),
            };
            cells.push_back(CellInfo{mimetic::create_polygon(mb, pts), pts});
        }
    }
    std::vector<moab::EntityHandle> polygons;
    polygons.reserve(cells.size());
    for (const CellInfo& cell : cells) polygons.push_back(cell.handle);
    mimetic::merge_polygon_vertices(mb, polygons);
    return cells;
}

std::vector<CellInfo> create_voronoi_mesh(moab::Core& mb,
                                          const std::vector<Eigen::Vector2d>& seeds)
{
    std::vector<CellInfo> cells;
    for (const Eigen::Vector2d& seed : seeds) {
        const std::vector<Eigen::Vector2d> polygon = voronoi_cell_polygon(seed, seeds);
        if (polygon.size() < 3 || polygon_area(polygon) < 1.0e-12) continue;
        cells.push_back(CellInfo{mimetic::create_polygon(mb, polygon), polygon});
    }
    std::vector<moab::EntityHandle> polygons;
    polygons.reserve(cells.size());
    for (const CellInfo& cell : cells) polygons.push_back(cell.handle);
    mimetic::merge_polygon_vertices(mb, polygons);
    return cells;
}

std::vector<moab::EntityHandle> handles(const std::vector<CellInfo>& cells)
{
    std::vector<moab::EntityHandle> result;
    result.reserve(cells.size());
    for (const CellInfo& cell : cells) result.push_back(cell.handle);
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

// Build face-neighbor map: for each cell, find cells sharing an edge (via MOAB shared vertices)
std::map<moab::EntityHandle, std::vector<moab::EntityHandle>>
build_face_neighbor_map(moab::Core& mb, const std::vector<CellInfo>& cells)
{
    std::map<moab::EntityHandle, std::vector<moab::EntityHandle>> neighbors;

    // Build edge→cell map via vertex pairs
    using VertexPair = std::pair<moab::EntityHandle, moab::EntityHandle>;
    std::map<VertexPair, std::vector<moab::EntityHandle>> edge_to_cells;

    for (const CellInfo& cell : cells) {
        const moab::EntityHandle* conn;
        int num_verts;
        mb.get_connectivity(cell.handle, conn, num_verts);

        for (int i = 0; i < num_verts; ++i) {
            moab::EntityHandle v0 = conn[i];
            moab::EntityHandle v1 = conn[(i + 1) % num_verts];
            VertexPair edge_key = (v0 < v1) ? std::make_pair(v0, v1) : std::make_pair(v1, v0);
            edge_to_cells[edge_key].push_back(cell.handle);
        }
    }

    for (const CellInfo& cell : cells) {
        std::set<moab::EntityHandle> nbr_set;
        const moab::EntityHandle* conn;
        int num_verts;
        mb.get_connectivity(cell.handle, conn, num_verts);

        for (int i = 0; i < num_verts; ++i) {
            moab::EntityHandle v0 = conn[i];
            moab::EntityHandle v1 = conn[(i + 1) % num_verts];
            VertexPair edge_key = (v0 < v1) ? std::make_pair(v0, v1) : std::make_pair(v1, v0);
            for (moab::EntityHandle other : edge_to_cells[edge_key]) {
                if (other != cell.handle) {
                    nbr_set.insert(other);
                }
            }
        }

        neighbors[cell.handle] = std::vector<moab::EntityHandle>(nbr_set.begin(), nbr_set.end());
    }

    return neighbors;
}

double convergence_rate(const double e_coarse, const double e_fine,
                        const double h_coarse, const double h_fine)
{
    if (e_coarse < 1.0e-30 || e_fine < 1.0e-30) return 0.0;
    return std::log(e_coarse / e_fine) / std::log(h_coarse / h_fine);
}

struct ConvergenceMetrics {
    int order = 0;
    std::string method;
    std::string domain;
    double h = 0.0;
    double l2_flux_rel = 0.0;
};

// Workflow: single flux per edge → patch recovery → VEM → transfer → measure error
ConvergenceMetrics run_patch_vem_case(const std::string& domain,
                                      moab::Core& mb,
                                      const std::vector<CellInfo>& source_cells,
                                      const std::vector<CellInfo>& target_cells,
                                      const int order,
                                      const double h)
{
    PlanarMomentInterpolator interpolator(mb);
    MomentMethodOptions options;
    options.edge_moment_order = order;
    options.quadrature_points = 10;
    options.reconstruction_mode = ReconstructionMode::PatchRecoveryVem;

    // Step 1: Set ONLY moment-0 (single flux per edge) for ALL source cells
    for (const CellInfo& source : source_cells) {
        const LocalPolygon poly = local_polygon(mb, source.handle);
        const std::vector<LocalEdge> edges = local_edges(mb, poly);
        for (std::size_t ei = 0; ei < edges.size(); ++ei) {
            const Eigen::Vector2d a = poly.centroid + edges[ei].a;
            const Eigen::Vector2d b = poly.centroid + edges[ei].b;
            const std::vector<double> m0 =
                exact_edge_moments_absolute(a, b, 0, smooth_variable_divergence_field);
            interpolator.set_source_edge_moments(source.handle, ei, m0);
        }
    }

    // Step 2: Build face-neighbor map
    const auto neighbor_map = build_face_neighbor_map(mb, source_cells);

    // Step 3: For each source cell, run patch recovery to bootstrap high-order moments
    for (const CellInfo& source : source_cells) {
        auto it = neighbor_map.find(source.handle);
        const std::vector<moab::EntityHandle>& nbrs =
            (it != neighbor_map.end()) ? it->second : std::vector<moab::EntityHandle>{};
        interpolator.recover_moments_from_patch(source.handle, order, nbrs);
    }

    // Step 4: Reconstruct each source cell using VEM with recovered moments
    for (const CellInfo& source : source_cells) {
        interpolator.reconstruct_source_polygon(source.handle, options);
    }

    // Step 5: Transfer edge moments to target mesh
    const EdgeMomentTransferResult transfer =
        interpolator.transfer_source_to_target_edge_moments(
            handles(source_cells), handles(target_cells), order);

    // Step 6: Measure error
    double flux_num = 0.0;
    double flux_den = 0.0;
    std::size_t dof = 0;
    for (std::size_t ci = 0; ci < target_cells.size(); ++ci) {
        const CellInfo& target = target_cells[ci];
        const LocalPolygon poly = local_polygon(mb, target.handle);
        const std::vector<LocalEdge> edges = local_edges(mb, poly);
        for (std::size_t ei = 0; ei < edges.size(); ++ei) {
            const Eigen::Vector2d a = poly.centroid + edges[ei].a;
            const Eigen::Vector2d b = poly.centroid + edges[ei].b;
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
    metrics.method = "PatchVEM";
    metrics.domain = domain;
    metrics.h = h;
    metrics.l2_flux_rel = (flux_den > 1.0e-30) ? std::sqrt(flux_num / flux_den) : std::sqrt(flux_num);
    return metrics;
}

bool test_patch_recovery_convergence()
{
    std::cout << "=== Patch recovery + VEM convergence (single flux/edge → patch → VEM → transfer) ===\n";
    bool pass = true;

    const int refinements[] = {4, 8, 16};
    const int voronoi_counts[] = {16, 64, 256};

    // Recovery from single-flux-per-edge bootstraps higher moments via patch LS.
    // The recovered moments are approximate, so convergence rates degrade
    // relative to exact-moment VEM. Oracle analysis predicts ~O(h^2) for p=1
    // and ~O(h^{2-3}) for p=2 on regular patches, with rate decay at fine h.
    const double min_rates_quad[] = {0.0, 1.5, 1.3};
    const double min_rates_vor[]  = {0.0, 1.3, 1.0};

    for (int order = 1; order <= 2; ++order) {
        std::cout << "\n  p=" << order << ":\n";

        std::vector<ConvergenceMetrics> quad_metrics;
        std::vector<ConvergenceMetrics> vor_metrics;

        for (int ref : refinements) {
            moab::Core mb;
            const int source_n = ref;
            const int target_n = ref + ref / 2;
            const double h = 1.0 / source_n;

            const std::vector<CellInfo> source = create_quad_mesh(mb, source_n, source_n);
            const std::vector<CellInfo> target = create_quad_mesh(mb, target_n, target_n);

            quad_metrics.push_back(run_patch_vem_case("quad", mb, source, target, order, h));
        }

        for (int count : voronoi_counts) {
            moab::Core mb;
            const double h = 1.0 / std::sqrt(static_cast<double>(count));
            const std::vector<Eigen::Vector2d> source_seeds = halton_seeds(count, 0);
            const std::vector<Eigen::Vector2d> target_seeds = halton_seeds(count + count / 2, 100);

            const std::vector<CellInfo> source = create_voronoi_mesh(mb, source_seeds);
            const std::vector<CellInfo> target = create_voronoi_mesh(mb, target_seeds);

            vor_metrics.push_back(run_patch_vem_case("voronoi", mb, source, target, order, h));
        }

        auto print_rates = [&](const std::string& label, const std::vector<ConvergenceMetrics>& metrics) {
            std::cout << "    " << std::left << std::setw(25) << label;
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

        print_rates("PatchVEM quad", quad_metrics);
        print_rates("PatchVEM voronoi", vor_metrics);

        // Check convergence rate from last two refinements
        if (quad_metrics.size() >= 2) {
            const double rate = convergence_rate(
                quad_metrics[quad_metrics.size() - 2].l2_flux_rel,
                quad_metrics[quad_metrics.size() - 1].l2_flux_rel,
                quad_metrics[quad_metrics.size() - 2].h,
                quad_metrics[quad_metrics.size() - 1].h);
            std::cout << "    Quad rate=" << std::fixed << std::setprecision(2) << rate
                      << " (min=" << min_rates_quad[order] << ") ";
            if (rate < min_rates_quad[order]) {
                std::cout << "FAIL\n";
                pass = false;
            } else {
                std::cout << "OK\n";
            }
        }

        if (vor_metrics.size() >= 2) {
            const double rate = convergence_rate(
                vor_metrics[vor_metrics.size() - 2].l2_flux_rel,
                vor_metrics[vor_metrics.size() - 1].l2_flux_rel,
                vor_metrics[vor_metrics.size() - 2].h,
                vor_metrics[vor_metrics.size() - 1].h);
            std::cout << "    Voronoi rate=" << std::fixed << std::setprecision(2) << rate
                      << " (min=" << min_rates_vor[order] << ") ";
            if (rate < min_rates_vor[order]) {
                std::cout << "FAIL\n";
                pass = false;
            } else {
                std::cout << "OK\n";
            }
        }
    }

    return pass;
}

int main()
{
    int failures = 0;

    if (!test_patch_recovery_convergence()) {
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
