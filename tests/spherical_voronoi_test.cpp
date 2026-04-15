#include "mimetic/mimetic.hpp"
#include "spherical_transfer_test_utils.hpp"

#include <Eigen/Dense>
#include <moab/Core.hpp>

#include <algorithm>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace {

constexpr double kPatchExtent = 0.55;

struct CellInfo {
    moab::EntityHandle handle;
    std::size_t sides;
};

struct Metrics {
    std::string label;
    std::size_t source_cells;
    std::size_t target_cells;
    int source_ngons;
    int target_ngons;
    double global_conservation_residual;
    double max_direct_sparse_delta;
    double max_reintegration_residual;
    double l2_relative_flux_error;
    double max_flux_error;
};

double halton(int index, int base)
{
    double result = 0.0;
    double factor = 1.0 / static_cast<double>(base);
    int i = index;
    while (i > 0) {
        result += factor * (i % base);
        i /= base;
        factor /= static_cast<double>(base);
    }
    return result;
}

std::vector<Eigen::Vector2d> halton_seeds(const int count, const int offset)
{
    std::vector<Eigen::Vector2d> seeds;
    seeds.reserve(static_cast<std::size_t>(count));
    for (int i = 1; i <= count; ++i) {
        const int k = i + offset;
        seeds.emplace_back(-kPatchExtent + 2.0 * kPatchExtent * halton(k, 2),
                           -kPatchExtent + 2.0 * kPatchExtent * halton(k, 3));
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
        Eigen::Vector2d(-kPatchExtent, -kPatchExtent),
        Eigen::Vector2d( kPatchExtent, -kPatchExtent),
        Eigen::Vector2d( kPatchExtent,  kPatchExtent),
        Eigen::Vector2d(-kPatchExtent,  kPatchExtent),
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

std::vector<CellInfo> create_spherical_voronoi_patch(moab::Core& mb,
                                                     const int seed_count,
                                                     const int seed_offset)
{
    const std::vector<Eigen::Vector2d> seeds = halton_seeds(seed_count, seed_offset);
    std::vector<CellInfo> cells;
    for (const Eigen::Vector2d& seed : seeds) {
        const std::vector<Eigen::Vector2d> polygon = voronoi_cell_polygon(seed, seeds);
        if (polygon.size() < 3 || std::abs(mimetic::signed_area(polygon)) < 1.0e-12) {
            continue;
        }
        cells.push_back(CellInfo{
            mimetic::test_sphere::create_chart_polygon(mb, polygon),
            polygon.size(),
        });
    }
    return cells;
}

std::vector<CellInfo> create_spherical_quad_patch(moab::Core& mb, const int n)
{
    std::vector<CellInfo> cells;
    const double h = 2.0 * kPatchExtent / static_cast<double>(n);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            const double x0 = -kPatchExtent + i * h;
            const double y0 = -kPatchExtent + j * h;
            cells.push_back(CellInfo{
                mimetic::test_sphere::create_chart_polygon(mb, {
                    Eigen::Vector2d(x0, y0),
                    Eigen::Vector2d(x0 + h, y0),
                    Eigen::Vector2d(x0 + h, y0 + h),
                    Eigen::Vector2d(x0, y0 + h),
                }),
                4,
            });
        }
    }
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

int count_ngons(const std::vector<CellInfo>& cells)
{
    int count = 0;
    for (const CellInfo& cell : cells) {
        if (cell.sides > 4) {
            ++count;
        }
    }
    return count;
}

Metrics run_transfer_case(const std::string& label,
                          const std::vector<CellInfo>& source_cells,
                          const std::vector<CellInfo>& target_cells,
                          moab::Core& mb)
{
    mimetic::MimeticInterpolator interpolator(mb);
    mimetic::GeometryOptions options;
    options.mode = mimetic::GeometryMode::SphericalGnomonic;
    options.conservation_tolerance = mimetic::kConservationTolerance;
    interpolator.set_geometry_options(options);

    double total_source_divergence = 0.0;
    mimetic::test_sphere::set_conservative_source_fluxes(
        interpolator, mb, handles(source_cells), mimetic::test_sphere::spherical_harmonic_gradient);
    for (const CellInfo& source : source_cells) {
        const mimetic::LocalPolygon poly = mimetic::local_polygon(mb, source.handle, options);
        for (std::size_t i = 0; i < poly.vertices.size(); ++i) {
            const moab::EntityHandle edge =
                mimetic::find_or_create_edge(mb, poly.vertices[i], poly.vertices[(i + 1) % poly.vertices.size()]);
            total_source_divergence += interpolator.source_edge_flux(source.handle, i, edge);
        }
        interpolator.reconstruct_source_polygon(source.handle);
    }

    const std::vector<moab::EntityHandle> source_handles = handles(source_cells);
    const std::vector<moab::EntityHandle> target_handles = handles(target_cells);
    const mimetic::EdgeTransferResult transfer =
        interpolator.transfer_source_to_target_edges(source_handles, target_handles);
    const mimetic::SparseEdgeProjection projection =
        interpolator.assemble_edge_projection_operator(source_handles, target_handles);
    const std::map<std::pair<moab::EntityHandle, std::size_t>, double> target_exact_fluxes =
        mimetic::test_sphere::conservative_edge_fluxes(
            mb, target_handles, mimetic::test_sphere::spherical_harmonic_gradient);

    double total_target_divergence = 0.0;
    double max_reintegration_residual = 0.0;
    double l2_num = 0.0;
    double l2_den = 0.0;
    double max_flux_error = 0.0;
    std::size_t dof = 0;

    for (const CellInfo& target : target_cells) {
        const mimetic::LocalPolygon poly = mimetic::local_polygon(mb, target.handle, options);
        const std::vector<mimetic::LocalEdge> edges = mimetic::local_edges(mb, poly);
        for (std::size_t i = 0; i < edges.size(); ++i) {
            const double transferred = transfer.target_fluxes[dof++];
            const double exact = target_exact_fluxes.at(std::make_pair(target.handle, i));
            const double error = std::abs(transferred - exact);
            l2_num += error * error;
            l2_den += exact * exact;
            max_flux_error = std::max(max_flux_error, error);
            total_target_divergence += transferred;
            interpolator.set_source_edge_flux(target.handle, i, transferred);
        }

        const mimetic::ReconstructionCoeffs coeffs = interpolator.reconstruct_source_polygon(target.handle);
        for (std::size_t i = 0; i < edges.size(); ++i) {
            const double reintegrated = interpolator.edge_flux(coeffs, edges[i].a, edges[i].b);
            const double transferred = interpolator.source_edge_flux(target.handle, i, edges[i].handle);
            max_reintegration_residual =
                std::max(max_reintegration_residual, std::abs(reintegrated - transferred));
        }
    }

    Metrics metrics;
    metrics.label = label;
    metrics.source_cells = source_cells.size();
    metrics.target_cells = target_cells.size();
    metrics.source_ngons = count_ngons(source_cells);
    metrics.target_ngons = count_ngons(target_cells);
    metrics.global_conservation_residual = std::abs(total_source_divergence - total_target_divergence);
    metrics.max_direct_sparse_delta =
        mimetic::test_sphere::max_direct_sparse_delta(interpolator, projection, transfer);
    metrics.max_reintegration_residual = max_reintegration_residual;
    metrics.l2_relative_flux_error = (l2_den > 0.0) ? std::sqrt(l2_num / l2_den) : std::sqrt(l2_num);
    metrics.max_flux_error = max_flux_error;
    return metrics;
}

void print_metrics(const Metrics& m)
{
    std::cout << m.label << ":\n"
              << "  source_cells=" << m.source_cells
              << " target_cells=" << m.target_cells
              << " source_ngons=" << m.source_ngons
              << " target_ngons=" << m.target_ngons << "\n"
              << std::scientific << std::setprecision(6)
              << "  global_residual=" << m.global_conservation_residual
              << " direct_sparse_delta=" << m.max_direct_sparse_delta
              << " reintegration=" << m.max_reintegration_residual << "\n"
              << "  edge_flux_l2_rel=" << m.l2_relative_flux_error
              << " edge_flux_linf=" << m.max_flux_error << "\n";
}

bool invariants_pass(const Metrics& m)
{
    return m.global_conservation_residual <= mimetic::kConservationTolerance &&
           m.max_direct_sparse_delta <= mimetic::kConservationTolerance &&
           m.max_reintegration_residual <= mimetic::kConservationTolerance;
}

}  // namespace

int main()
{
    try {
        std::cout << "--- Spherical Voronoi Patch Mimetic Edge Transfer Test ---\n\n";

        moab::Core mb_vv_coarse;
        const std::vector<CellInfo> vv_source_coarse = create_spherical_voronoi_patch(mb_vv_coarse, 36, 0);
        const std::vector<CellInfo> vv_target_coarse = create_spherical_voronoi_patch(mb_vv_coarse, 45, 200);
        const Metrics vv_coarse = run_transfer_case("Voronoi -> Voronoi coarse",
                                                    vv_source_coarse, vv_target_coarse, mb_vv_coarse);

        moab::Core mb_vv_fine;
        const std::vector<CellInfo> vv_source_fine = create_spherical_voronoi_patch(mb_vv_fine, 100, 0);
        const std::vector<CellInfo> vv_target_fine = create_spherical_voronoi_patch(mb_vv_fine, 120, 500);
        const Metrics vv_fine = run_transfer_case("Voronoi -> Voronoi fine",
                                                  vv_source_fine, vv_target_fine, mb_vv_fine);

        moab::Core mb_vq;
        const std::vector<CellInfo> vq_source = create_spherical_voronoi_patch(mb_vq, 64, 50);
        const std::vector<CellInfo> vq_target = create_spherical_quad_patch(mb_vq, 8);
        const Metrics vq = run_transfer_case("Voronoi -> structured quad patch",
                                             vq_source, vq_target, mb_vq);

        moab::Core mb_qv;
        const std::vector<CellInfo> qv_source = create_spherical_quad_patch(mb_qv, 8);
        const std::vector<CellInfo> qv_target = create_spherical_voronoi_patch(mb_qv, 64, 350);
        const Metrics qv = run_transfer_case("Structured quad patch -> Voronoi",
                                             qv_source, qv_target, mb_qv);

        print_metrics(vv_coarse);
        print_metrics(vv_fine);
        print_metrics(vq);
        print_metrics(qv);

        bool ok = invariants_pass(vv_coarse) && invariants_pass(vv_fine) &&
                  invariants_pass(vq) && invariants_pass(qv);
        ok = (vv_coarse.source_ngons > 0 && vv_coarse.target_ngons > 0 &&
              vv_fine.source_ngons > 0 && vv_fine.target_ngons > 0 &&
              vq.source_ngons > 0 && qv.target_ngons > 0) && ok;
        ok = (vv_fine.l2_relative_flux_error < vv_coarse.l2_relative_flux_error) && ok;
        ok = (vv_fine.max_flux_error < vv_coarse.max_flux_error) && ok;

        if (!ok) {
            std::cout << "\n[FAILED] Spherical Voronoi transfer did not meet acceptance checks.\n";
            return 1;
        }

        std::cout << "\n[SUCCESS] Spherical unstructured transfer is conservative, sparse-consistent, and convergent.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 2;
    }
}
