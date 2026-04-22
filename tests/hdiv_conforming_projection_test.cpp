#include "mimetic/mimetic.hpp"
#include "test_utils.hpp"

#include <Eigen/Dense>
#include <moab/Core.hpp>

#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr double PI = 3.14159265358979323846;

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

using FieldFunc = Eigen::Vector2d(*)(const Eigen::Vector2d&);

Eigen::Vector2d linear_field(const Eigen::Vector2d& p)
{
    return Eigen::Vector2d(p.x(), p.y());
}

Eigen::Vector2d divergence_free_field(const Eigen::Vector2d& p)
{
    return Eigen::Vector2d(std::sin(PI * p.x()) * std::cos(PI * p.y()),
                          -std::cos(PI * p.x()) * std::sin(PI * p.y()));
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

std::vector<CellInfo> create_voronoi_mesh(moab::Core& mb, const std::vector<Eigen::Vector2d>& seeds)
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

void set_source_fluxes_highorder(moab::Core& mb,
                                 mimetic::MimeticInterpolator& interpolator,
                                 const moab::EntityHandle polygon,
                                 const FieldFunc field)
{
    const mimetic::LocalPolygon poly = mimetic::local_polygon(mb, polygon);
    const std::vector<mimetic::LocalEdge> edges = mimetic::local_edges(mb, poly);
    for (std::size_t edge_index = 0; edge_index < edges.size(); ++edge_index) {
        const mimetic::LocalEdge& edge = edges[edge_index];
        const double flux = integrate_edge_highorder(edge.a, edge.b, [&](const Eigen::Vector2d& p) {
            return field(p + poly.centroid).dot(edge.outward_normal);
        });
        interpolator.set_source_edge_flux(polygon, edge_index, flux);
    }
}

double exact_directed_edge_flux(const Eigen::Vector2d& a, const Eigen::Vector2d& b, const FieldFunc field)
{
    const Eigen::Vector2d delta = b - a;
    const Eigen::Vector2d normal(delta.y(), -delta.x());
    const double length = delta.norm();
    return integrate_edge_highorder(a, b, [&](const Eigen::Vector2d& p) {
        return field(p).dot(normal / length);
    });
}

struct ProjectionMetrics {
    double max_raw_cell_residual = 0.0;
    double max_conforming_cell_residual = 0.0;
    double max_raw_edge_jump = 0.0;
    double max_conforming_edge_jump = 0.0;
    double correction_l2_relative = 0.0;
    double raw_edge_l2_relative = 0.0;
    double conforming_edge_l2_relative = 0.0;
    double max_rhs_error = 0.0;
};

ProjectionMetrics run_projection_case(const int n_source,
                                      const int n_target,
                                      const int source_offset,
                                      const int target_offset,
                                      const FieldFunc field,
                                      const bool expect_linear_rhs)
{
    moab::Core mb;
    mimetic::MimeticInterpolator interpolator(mb);

    const std::vector<CellInfo> source_cells = create_voronoi_mesh(mb, halton_seeds(n_source, source_offset));
    const std::vector<CellInfo> target_cells = create_voronoi_mesh(mb, halton_seeds(n_target, target_offset));
    const std::vector<moab::EntityHandle> source_handles = handles(source_cells);
    const std::vector<moab::EntityHandle> target_handles = handles(target_cells);

    for (const CellInfo& source : source_cells) {
        set_source_fluxes_highorder(mb, interpolator, source.handle, field);
        interpolator.reconstruct_source_polygon(source.handle);
    }

    const mimetic::EdgeTransferResult raw =
        interpolator.transfer_source_to_target_edges(source_handles, target_handles);
    const mimetic::ConformingEdgeTransferResult conforming =
        interpolator.project_target_fluxes_to_hdiv_conforming(source_handles, target_handles, raw);

    ProjectionMetrics metrics;
    double raw_diff_l2 = 0.0;
    double conf_norm_l2 = 0.0;
    double raw_edge_num = 0.0;
    double conf_edge_num = 0.0;
    double edge_den = 0.0;
    std::size_t dof = 0;
    for (std::size_t cell_index = 0; cell_index < target_cells.size(); ++cell_index) {
        const CellInfo& target = target_cells[cell_index];
        const mimetic::LocalPolygon poly = mimetic::local_polygon(mb, target.handle);
        const std::vector<mimetic::LocalEdge> edges = mimetic::local_edges(mb, poly);
        double raw_cell_sum = 0.0;
        double conforming_cell_sum = 0.0;
        for (const mimetic::LocalEdge& edge : edges) {
            const Eigen::Vector2d a = poly.centroid + edge.a;
            const Eigen::Vector2d b = poly.centroid + edge.b;
            const double exact = exact_directed_edge_flux(a, b, field);
            const double raw_flux = raw.target_fluxes[dof];
            const double conforming_flux = conforming.target_fluxes[dof];
            raw_cell_sum += raw_flux;
            conforming_cell_sum += conforming_flux;
            raw_diff_l2 += (conforming_flux - raw_flux) * (conforming_flux - raw_flux);
            conf_norm_l2 += conforming_flux * conforming_flux;
            raw_edge_num += (raw_flux - exact) * (raw_flux - exact);
            conf_edge_num += (conforming_flux - exact) * (conforming_flux - exact);
            edge_den += exact * exact;
            ++dof;
        }

        const double rhs = conforming.target_divergence_integrals[cell_index];
        metrics.max_raw_cell_residual = std::max(metrics.max_raw_cell_residual, std::abs(raw_cell_sum - rhs));
        metrics.max_conforming_cell_residual =
            std::max(metrics.max_conforming_cell_residual, std::abs(conforming_cell_sum - rhs));

        if (expect_linear_rhs) {
            const double exact_rhs = 2.0 * polygon_area(target.points);
            metrics.max_rhs_error = std::max(metrics.max_rhs_error, std::abs(rhs - exact_rhs));
        }
    }

    std::vector<std::vector<double>> raw_by_unique(conforming.unique_edge_fluxes.size());
    std::vector<std::vector<double>> conf_by_unique(conforming.unique_edge_fluxes.size());
    for (std::size_t i = 0; i < conforming.target_edge_to_unique.size(); ++i) {
        const std::size_t unique = conforming.target_edge_to_unique[i];
        raw_by_unique[unique].push_back(raw.target_fluxes[i]);
        conf_by_unique[unique].push_back(conforming.target_fluxes[i]);
    }
    for (std::size_t unique = 0; unique < raw_by_unique.size(); ++unique) {
        if (raw_by_unique[unique].size() == 2) {
            metrics.max_raw_edge_jump = std::max(metrics.max_raw_edge_jump,
                                                 std::abs(raw_by_unique[unique][0] + raw_by_unique[unique][1]));
            metrics.max_conforming_edge_jump = std::max(metrics.max_conforming_edge_jump,
                                                        std::abs(conf_by_unique[unique][0] + conf_by_unique[unique][1]));
        }
    }

    metrics.correction_l2_relative =
        (conf_norm_l2 > 1.0e-30) ? std::sqrt(raw_diff_l2 / conf_norm_l2) : std::sqrt(raw_diff_l2);
    metrics.raw_edge_l2_relative =
        (edge_den > 1.0e-30) ? std::sqrt(raw_edge_num / edge_den) : std::sqrt(raw_edge_num);
    metrics.conforming_edge_l2_relative =
        (edge_den > 1.0e-30) ? std::sqrt(conf_edge_num / edge_den) : std::sqrt(conf_edge_num);
    return metrics;
}

void print_metrics(const std::string& label, const ProjectionMetrics& m)
{
    std::cout << label << ":\n"
              << std::scientific << std::setprecision(6)
              << "  max_raw_cell_residual=" << m.max_raw_cell_residual
              << " max_conf_cell_residual=" << m.max_conforming_cell_residual << "\n"
              << "  max_raw_edge_jump=" << m.max_raw_edge_jump
              << " max_conf_edge_jump=" << m.max_conforming_edge_jump << "\n"
              << "  correction_l2_relative=" << m.correction_l2_relative
              << " raw_edge_l2_relative=" << m.raw_edge_l2_relative
              << " conf_edge_l2_relative=" << m.conforming_edge_l2_relative << "\n"
              << "  max_rhs_error=" << m.max_rhs_error << "\n";
}

}  // namespace

int main()
{
    try {
        std::cout << "--- Global H(div)-Conforming Target Projection Test ---\n\n";

        const ProjectionMetrics linear =
            run_projection_case(16, 25, 0, 200, linear_field, true);
        const ProjectionMetrics divfree_coarse =
            run_projection_case(16, 25, 0, 200, divergence_free_field, false);
        const ProjectionMetrics divfree_medium =
            run_projection_case(36, 49, 0, 400, divergence_free_field, false);
        const ProjectionMetrics divfree_fine =
            run_projection_case(64, 81, 0, 800, divergence_free_field, false);

        print_metrics("Exact linear field", linear);
        print_metrics("Divergence-free field coarse", divfree_coarse);
        print_metrics("Divergence-free field medium", divfree_medium);
        print_metrics("Divergence-free field fine", divfree_fine);

        bool ok = true;
        ok = mimetic::test::near(linear.max_raw_cell_residual, 0.0, mimetic::kConservationTolerance,
                                 "linear raw target-cell residual") && ok;
        ok = mimetic::test::near(linear.max_conforming_cell_residual, 0.0, mimetic::kConservationTolerance,
                                 "linear conforming target-cell residual") && ok;
        ok = mimetic::test::near(linear.correction_l2_relative, 0.0, mimetic::kConservationTolerance,
                                 "linear correction relative norm") && ok;
        ok = mimetic::test::near(linear.max_rhs_error, 0.0, mimetic::kConservationTolerance,
                                 "linear exact target divergence rhs") && ok;

        ok = mimetic::test::near(divfree_coarse.max_conforming_cell_residual, 0.0, mimetic::kConservationTolerance,
                                 "divfree coarse conforming residual") && ok;
        ok = mimetic::test::near(divfree_medium.max_conforming_cell_residual, 0.0, mimetic::kConservationTolerance,
                                 "divfree medium conforming residual") && ok;
        ok = mimetic::test::near(divfree_fine.max_conforming_cell_residual, 0.0, mimetic::kConservationTolerance,
                                 "divfree fine conforming residual") && ok;
        ok = mimetic::test::near(divfree_coarse.max_raw_edge_jump, 0.0, mimetic::kConservationTolerance,
                                 "divfree coarse raw edge jump") && ok;
        ok = mimetic::test::near(divfree_medium.max_raw_edge_jump, 0.0, mimetic::kConservationTolerance,
                                 "divfree medium raw edge jump") && ok;
        ok = mimetic::test::near(divfree_fine.max_raw_edge_jump, 0.0, mimetic::kConservationTolerance,
                                 "divfree fine raw edge jump") && ok;
        ok = mimetic::test::near(divfree_coarse.max_conforming_edge_jump, 0.0, mimetic::kConservationTolerance,
                                 "divfree coarse conforming edge jump") && ok;
        ok = mimetic::test::near(divfree_medium.max_conforming_edge_jump, 0.0, mimetic::kConservationTolerance,
                                 "divfree medium conforming edge jump") && ok;
        ok = mimetic::test::near(divfree_fine.max_conforming_edge_jump, 0.0, mimetic::kConservationTolerance,
                                 "divfree fine conforming edge jump") && ok;

        ok = (divfree_fine.correction_l2_relative < divfree_medium.correction_l2_relative) && ok;
        ok = (divfree_medium.correction_l2_relative < divfree_coarse.correction_l2_relative) && ok;
        ok = (divfree_fine.max_raw_cell_residual < divfree_medium.max_raw_cell_residual) && ok;
        ok = (divfree_medium.max_raw_cell_residual < divfree_coarse.max_raw_cell_residual) && ok;

        if (!ok) {
            std::cout << "\n[FAILED] Global H(div)-conforming target projection test failed.\n";
            return 1;
        }

        std::cout << "\n[SUCCESS] The target-edge postprocess is exactly conservative and the raw transfer converges toward it under refinement.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 2;
    }
}
