#include "mimetic/mimetic.hpp"
#include "test_utils.hpp"

#include <Eigen/Dense>
#include <moab/Core.hpp>

#include <array>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct CellInfo {
    moab::EntityHandle handle;
    std::vector<Eigen::Vector2d> points;
};

Eigen::Vector2d harmonic_exact_field(const Eigen::Vector2d& p)
{
    return Eigen::Vector2d(1.0 + 2.0 * p.x() + 2.0 * p.y(),
                           1.0 - 2.0 * p.y() + 2.0 * p.x());
}

Eigen::Vector2d constant_field(const Eigen::Vector2d&)
{
    return Eigen::Vector2d(1.0, 1.0);
}

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

std::vector<CellInfo> create_rect_mesh(moab::Core& mb,
                                       const std::vector<double>& xs,
                                       const std::vector<double>& ys)
{
    std::vector<CellInfo> cells;
    for (std::size_t j = 0; j + 1 < ys.size(); ++j) {
        for (std::size_t i = 0; i + 1 < xs.size(); ++i) {
            const std::vector<Eigen::Vector2d> points = {
                Eigen::Vector2d(xs[i], ys[j]),
                Eigen::Vector2d(xs[i + 1], ys[j]),
                Eigen::Vector2d(xs[i + 1], ys[j + 1]),
                Eigen::Vector2d(xs[i], ys[j + 1]),
            };
            cells.push_back(CellInfo{mimetic::create_polygon(mb, points), points});
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
        const std::vector<Eigen::Vector2d> points = voronoi_cell_polygon(seed, seeds);
        if (points.size() < 3 || std::abs(mimetic::signed_area(points)) < 1.0e-12) {
            continue;
        }
        cells.push_back(CellInfo{mimetic::create_polygon(mb, points), points});
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

using FieldFunc = Eigen::Vector2d(*)(const Eigen::Vector2d&);

Eigen::Vector3d exact_cell_average(const std::vector<Eigen::Vector2d>& points, FieldFunc field)
{
    const Eigen::Vector2d centroid = mimetic::polygon_centroid(points);
    const Eigen::Vector2d value = field(centroid);
    return Eigen::Vector3d(value.x(), value.y(), 0.0);
}

double polygon_area(const std::vector<Eigen::Vector2d>& points)
{
    return std::abs(mimetic::signed_area(points));
}

bool near_vec3(const Eigen::Vector3d& actual,
               const Eigen::Vector3d& expected,
               const double tolerance,
               const std::string& label)
{
    bool ok = true;
    ok = mimetic::test::near(actual.x(), expected.x(), tolerance, label + " [x]") && ok;
    ok = mimetic::test::near(actual.y(), expected.y(), tolerance, label + " [y]") && ok;
    ok = mimetic::test::near(actual.z(), expected.z(), tolerance, label + " [z]") && ok;
    return ok;
}

bool run_case(const std::string& label,
              moab::Core& mb,
              const std::vector<CellInfo>& source_cells,
              const std::vector<CellInfo>& target_cells,
              const int expected_target_ngons,
              FieldFunc field,
              const bool require_exact_cell_average,
              const bool require_reconstruction_match,
              const bool require_total_exact)
{
    mimetic::MimeticInterpolator interpolator(mb);

    const std::vector<moab::EntityHandle> source_handles = handles(source_cells);
    const std::vector<moab::EntityHandle> target_handles = handles(target_cells);

    for (const CellInfo& source : source_cells) {
        mimetic::test::set_source_fluxes_from_absolute_field(
            mb, interpolator, source.handle, field);
        interpolator.reconstruct_source_polygon(source.handle);
    }

    const mimetic::CellAverageTransferResult direct =
        interpolator.transfer_source_to_target_cell_averages(
            source_handles, target_handles, mimetic::CellAverageReductionMode::Harmonic);
    const mimetic::EdgeTransferResult edge_transfer =
        interpolator.transfer_source_to_target_edges(source_handles, target_handles);

    std::vector<double> coverage(target_cells.size(), 0.0);
    std::vector<Eigen::Vector3d> integral_from_contrib(target_cells.size(), Eigen::Vector3d::Zero());
    for (const mimetic::CellAverageContribution& contrib : direct.contributions) {
        coverage[contrib.target_cell_index] += contrib.overlap_area;
        integral_from_contrib[contrib.target_cell_index] += contrib.integral;
    }

    bool ok = true;
    int target_ngons = 0;
    Eigen::Vector3d total_direct_integral = Eigen::Vector3d::Zero();
    Eigen::Vector3d total_exact_integral = Eigen::Vector3d::Zero();

    std::size_t dof = 0;
    for (std::size_t cell_index = 0; cell_index < target_cells.size(); ++cell_index) {
        const CellInfo& target = target_cells[cell_index];
        const double area = polygon_area(target.points);
        if (target.points.size() > 4) {
            ++target_ngons;
        }

        const Eigen::Vector3d exact_average = exact_cell_average(target.points, field);
        const Eigen::Vector3d exact_integral = area * exact_average;
        total_direct_integral += direct.target_integrals[cell_index];
        total_exact_integral += exact_integral;

        ok = mimetic::test::near(direct.target_areas[cell_index], area, mimetic::kConservationTolerance,
                                 label + " target area " + std::to_string(cell_index)) &&
             ok;
        ok = mimetic::test::near(coverage[cell_index], area, mimetic::kConservationTolerance,
                                 label + " overlap coverage " + std::to_string(cell_index)) &&
             ok;
        ok = near_vec3(integral_from_contrib[cell_index], direct.target_integrals[cell_index],
                       mimetic::kConservationTolerance,
                       label + " contribution integral sum " + std::to_string(cell_index)) &&
             ok;
        if (require_exact_cell_average) {
            ok = near_vec3(direct.target_integrals[cell_index], exact_integral, 1.0e-12,
                           label + " direct integral " + std::to_string(cell_index)) &&
                 ok;
            ok = near_vec3(direct.target_averages[cell_index], exact_average, 1.0e-12,
                           label + " direct average " + std::to_string(cell_index)) &&
                 ok;
        }

        const mimetic::LocalPolygon target_poly = mimetic::local_polygon(mb, target.handle);
        const std::vector<mimetic::LocalEdge> target_edges = mimetic::local_edges(mb, target_poly);
        for (std::size_t edge_index = 0; edge_index < target_edges.size(); ++edge_index) {
            interpolator.set_source_edge_flux(target.handle, edge_index, edge_transfer.target_fluxes[dof++]);
        }
        interpolator.reconstruct_source_polygon(target.handle);
        const Eigen::Vector3d reconstructed_average = interpolator.cell_average(target.handle);
        if (require_reconstruction_match) {
            ok = near_vec3(reconstructed_average, direct.target_averages[cell_index], 1.0e-12,
                           label + " edge-to-cell average " + std::to_string(cell_index)) &&
                 ok;
        }
        if (require_exact_cell_average) {
            ok = near_vec3(reconstructed_average, exact_average, 1.0e-12,
                           label + " reconstructed constant average " + std::to_string(cell_index)) &&
                 ok;
            ok = near_vec3(direct.target_averages[cell_index], exact_average, 1.0e-12,
                           label + " direct constant average " + std::to_string(cell_index)) &&
                 ok;
        }
    }

    ok = mimetic::test::near(static_cast<double>(target_ngons), static_cast<double>(expected_target_ngons),
                             mimetic::kTolerance, label + " target n-gons") &&
         ok;
    if (require_total_exact) {
        ok = near_vec3(total_direct_integral, total_exact_integral, 1.0e-12, label + " total integral") && ok;
    }
    return ok;
}

}  // namespace

int main()
{
    try {
        std::cout << "--- Harmonic Edge-To-Cell-Average Transfer Test ---\n\n";

        bool ok = true;

        {
            moab::Core mb;
            const std::vector<CellInfo> source =
                create_rect_mesh(mb, {0.0, 0.5, 1.0}, {0.0, 0.5, 1.0});
            const std::vector<CellInfo> target =
                create_rect_mesh(mb, {0.0, 0.2, 0.6, 1.0}, {0.0, 0.35, 0.8, 1.0});
            ok = run_case("rectangular constant multi-source", mb, source, target, 0,
                          constant_field, true, false, true) && ok;
        }

        {
            moab::Core mb;
            std::vector<Eigen::Vector2d> source_seeds;
            std::vector<Eigen::Vector2d> target_seeds;
            for (int i = 1; i <= 9; ++i) {
                source_seeds.emplace_back(0.05 + 0.90 * halton(i, 2),
                                          0.05 + 0.90 * halton(i, 3));
            }
            for (int i = 10; i <= 23; ++i) {
                target_seeds.emplace_back(0.05 + 0.90 * halton(i, 2),
                                          0.05 + 0.90 * halton(i, 3));
            }
            const std::vector<CellInfo> source = create_voronoi_mesh(mb, source_seeds);
            const std::vector<CellInfo> target = create_voronoi_mesh(mb, target_seeds);
            ok = run_case("voronoi constant multi-source", mb, source, target, 10,
                          constant_field, true, false, true) && ok;
        }

        {
            moab::Core mb;
            const std::vector<CellInfo> source = {
                CellInfo{mimetic::create_polygon(mb, {
                    Eigen::Vector2d(0.0, 0.0),
                    Eigen::Vector2d(1.0, 0.0),
                    Eigen::Vector2d(1.0, 1.0),
                    Eigen::Vector2d(0.0, 1.0)}),
                         {
                             Eigen::Vector2d(0.0, 0.0),
                             Eigen::Vector2d(1.0, 0.0),
                             Eigen::Vector2d(1.0, 1.0),
                             Eigen::Vector2d(0.0, 1.0),
                         }}
            };
            const std::vector<CellInfo> target =
                create_rect_mesh(mb, {0.0, 0.2, 0.6, 1.0}, {0.0, 0.35, 0.8, 1.0});
            ok = run_case("rectangular harmonic single-source", mb, source, target, 0,
                          harmonic_exact_field, false, true, true) && ok;
        }

        {
            moab::Core mb;
            const std::vector<CellInfo> source = {
                CellInfo{mimetic::create_polygon(mb, {
                    Eigen::Vector2d(0.0, 0.0),
                    Eigen::Vector2d(1.0, 0.0),
                    Eigen::Vector2d(1.0, 1.0),
                    Eigen::Vector2d(0.0, 1.0)}),
                         {
                             Eigen::Vector2d(0.0, 0.0),
                             Eigen::Vector2d(1.0, 0.0),
                             Eigen::Vector2d(1.0, 1.0),
                             Eigen::Vector2d(0.0, 1.0),
                         }}
            };
            std::vector<Eigen::Vector2d> target_seeds;
            for (int i = 10; i <= 23; ++i) {
                target_seeds.emplace_back(0.05 + 0.90 * halton(i, 2),
                                          0.05 + 0.90 * halton(i, 3));
            }
            const std::vector<CellInfo> target = create_voronoi_mesh(mb, target_seeds);
            ok = run_case("voronoi harmonic single-source", mb, source, target, 10,
                          harmonic_exact_field, false, false, true) && ok;
        }

        if (!ok) {
            std::cout << "\n[FAILED] Harmonic edge-to-cell-average transfer test failed.\n";
            return 1;
        }

        std::cout << "\n[SUCCESS] Direct harmonic cell averages are exact and agree with edge-transferred target reconstructions.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 2;
    }
}
