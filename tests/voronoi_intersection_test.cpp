#include "mimetic/mimetic.hpp"
#include "test_utils.hpp"

#include <Eigen/Dense>
#include <moab/Core.hpp>

#include <algorithm>
#include <exception>
#include <iostream>
#include <map>
#include <vector>

namespace {

struct VoronoiCell {
    Eigen::Vector2d seed;
    std::vector<Eigen::Vector2d> points;
    moab::EntityHandle cell;
};

// Test-local area wrapper. Production code exposes signed_area so tests can
// verify convex clipping and source-target coverage independently.
double polygon_area(const std::vector<Eigen::Vector2d>& points)
{
    return std::abs(mimetic::signed_area(points));
}

// Half-plane clipping kernel for both Voronoi construction and convex polygon
// intersection. This is the Sutherland-Hodgman step used in report Algorithm 3.
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

// Clip a subject polygon by every target polygon edge. All Voronoi cells are
// convex, so repeated half-plane clipping is sufficient for exact intersections
// up to floating-point roundoff.
std::vector<Eigen::Vector2d> convex_intersection(std::vector<Eigen::Vector2d> subject,
                                                 const std::vector<Eigen::Vector2d>& clipper)
{
    if (mimetic::signed_area(subject) < 0.0) {
        std::reverse(subject.begin(), subject.end());
    }

    std::vector<Eigen::Vector2d> clip = clipper;
    if (mimetic::signed_area(clip) < 0.0) {
        std::reverse(clip.begin(), clip.end());
    }

    for (std::size_t i = 0; i < clip.size(); ++i) {
        const Eigen::Vector2d a = clip[i];
        const Eigen::Vector2d b = clip[(i + 1) % clip.size()];
        const Eigen::Vector2d edge = b - a;
        const Eigen::Vector2d inward_normal(-edge.y(), edge.x());
        subject = clip_by_halfplane(subject, -inward_normal, -inward_normal.dot(a));
        if (subject.size() < 3) {
            return {};
        }
    }
    return subject;
}

// Construct one bounded planar Voronoi cell by starting from the unit square and
// clipping by the perpendicular bisector to every other seed.
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

// Create all nondegenerate Voronoi cells as MOAB polygons. Cells with four sides
// are written as MBQUAD by create_polygon; all n>4 cells are MBPOLYGON.
std::vector<VoronoiCell> create_voronoi_mesh(moab::Core& mb, const std::vector<Eigen::Vector2d>& seeds)
{
    std::vector<VoronoiCell> cells;
    for (const Eigen::Vector2d& seed : seeds) {
        std::vector<Eigen::Vector2d> points = voronoi_cell_polygon(seed, seeds);
        if (points.size() < 3 || polygon_area(points) < 1.0e-12) {
            continue;
        }
        cells.push_back(VoronoiCell{seed, points, mimetic::create_polygon(mb, points)});
    }
    return cells;
}

std::vector<moab::EntityHandle> cell_handles(const std::vector<VoronoiCell>& cells)
{
    std::vector<moab::EntityHandle> handles;
    handles.reserve(cells.size());
    for (const VoronoiCell& cell : cells) {
        handles.push_back(cell.cell);
    }
    return handles;
}

std::size_t directed_edge_count(const std::vector<VoronoiCell>& cells)
{
    std::size_t count = 0;
    for (const VoronoiCell& cell : cells) {
        count += cell.points.size();
    }
    return count;
}

// Convert absolute overlap vertices into the source-cell local frame expected
// by MimeticInterpolator::polygon_boundary_flux.
std::vector<Eigen::Vector2d> points_in_source_frame(const std::vector<Eigen::Vector2d>& points,
                                                    const mimetic::LocalPolygon& source)
{
    std::vector<Eigen::Vector2d> shifted;
    shifted.reserve(points.size());
    for (const Eigen::Vector2d& p : points) {
        shifted.push_back(p - source.centroid);
    }
    return shifted;
}

std::map<std::size_t, int> side_histogram(const std::vector<VoronoiCell>& cells)
{
    std::map<std::size_t, int> histogram;
    for (const VoronoiCell& cell : cells) {
        ++histogram[cell.points.size()];
    }
    return histogram;
}

int count_ngons(const std::vector<VoronoiCell>& cells)
{
    int count = 0;
    for (const VoronoiCell& cell : cells) {
        if (cell.points.size() > 4) {
            ++count;
        }
    }
    return count;
}

void print_histogram(const std::string& label, const std::map<std::size_t, int>& histogram)
{
    std::cout << label << " side-count histogram:";
    for (const auto& item : histogram) {
        std::cout << " n=" << item.first << ":" << item.second;
    }
    std::cout << "\n";
}

}  // namespace

int main()
{
    try {
        std::cout << "--- Voronoi Conservative Interpolation Test ---\n\n";

        moab::Core mb;
        mimetic::MimeticInterpolator interpolator(mb);

        const std::vector<Eigen::Vector2d> source_seeds = {
            {0.12, 0.18}, {0.42, 0.12}, {0.78, 0.20}, {0.18, 0.52}, {0.52, 0.48},
            {0.86, 0.55}, {0.20, 0.84}, {0.55, 0.86}, {0.88, 0.86},
        };
        const std::vector<Eigen::Vector2d> target_seeds = {
            {0.10, 0.12}, {0.34, 0.18}, {0.62, 0.12}, {0.90, 0.18}, {0.18, 0.38},
            {0.48, 0.34}, {0.76, 0.42}, {0.10, 0.68}, {0.38, 0.62}, {0.66, 0.70},
            {0.92, 0.68}, {0.24, 0.92}, {0.58, 0.90}, {0.86, 0.92},
        };

        const std::vector<VoronoiCell> source_cells = create_voronoi_mesh(mb, source_seeds);
        const std::vector<VoronoiCell> target_cells = create_voronoi_mesh(mb, target_seeds);
        const std::vector<moab::EntityHandle> source_handles = cell_handles(source_cells);
        const std::vector<moab::EntityHandle> target_handles = cell_handles(target_cells);
        const int source_ngons = count_ngons(source_cells);
        const int target_ngons = count_ngons(target_cells);

        print_histogram("Source Voronoi", side_histogram(source_cells));
        print_histogram("Target Voronoi", side_histogram(target_cells));

        for (const VoronoiCell& source : source_cells) {
            mimetic::test::set_source_fluxes_from_absolute_field(
                mb, interpolator.source_flux_tag(), source.cell, mimetic::test::linear_absolute_field);
            interpolator.reconstruct_source_polygon(source.cell);
        }

        bool ok = true;
        int overlap_count = 0;
        double total_overlap_area = 0.0;
        double total_boundary_flux = 0.0;
        double total_expected_flux = 0.0;
        int ngon_overlap_count = 0;

        for (const VoronoiCell& target : target_cells) {
            double target_area_from_overlaps = 0.0;
            double target_flux_from_overlaps = 0.0;

            for (const VoronoiCell& source : source_cells) {
                const std::vector<Eigen::Vector2d> overlap = convex_intersection(source.points, target.points);
                if (overlap.size() < 3 || polygon_area(overlap) < 1.0e-12) {
                    continue;
                }

                const mimetic::LocalPolygon source_poly = mimetic::local_polygon(mb, source.cell);
                mimetic::ReconstructionCoeffs coeffs{};
                mimetic::check_moab(mb.tag_get_data(interpolator.coeffs_tag(), &source.cell, 1, &coeffs),
                                    "Failed to read source coefficients");

                const double area = polygon_area(overlap);
                const double boundary_flux =
                    interpolator.polygon_boundary_flux(coeffs, points_in_source_frame(overlap, source_poly));
                const double expected_flux = coeffs.d * area;
                if (source.points.size() > 4 && target.points.size() > 4) {
                    ++ngon_overlap_count;
                }

                ++overlap_count;
                target_area_from_overlaps += area;
                target_flux_from_overlaps += expected_flux;
                total_overlap_area += area;
                total_boundary_flux += boundary_flux;
                total_expected_flux += expected_flux;

                ok = mimetic::test::near(boundary_flux, expected_flux, 5.0e-11,
                                         "Voronoi overlap " + std::to_string(overlap_count)) &&
                     ok;
            }

            ok = mimetic::test::near(target_area_from_overlaps, polygon_area(target.points), 5.0e-11,
                                     "target cell covered by overlaps") &&
                 ok;
            ok = mimetic::test::near(target_flux_from_overlaps, 2.0 * polygon_area(target.points), 5.0e-11,
                                     "target conservative integral") &&
                 ok;
        }

        std::cout << "\nAggregate Voronoi checks:\n";
        ok = mimetic::test::near(source_ngons, 7.0, mimetic::kTolerance, "source cells with n > 4") && ok;
        ok = mimetic::test::near(target_ngons, 10.0, mimetic::kTolerance, "target cells with n > 4") && ok;
        ok = mimetic::test::near(ngon_overlap_count, 23.0, mimetic::kTolerance, "n-gon to n-gon overlaps") && ok;
        ok = mimetic::test::near(total_overlap_area, 1.0, 5.0e-11, "total overlap area") && ok;
        ok = mimetic::test::near(total_boundary_flux, total_expected_flux, 5.0e-10,
                                 "summed overlap boundary flux") &&
             ok;
        ok = mimetic::test::near(total_expected_flux, 2.0, 5.0e-10, "integral of div(x,y)") && ok;

        std::cout << "\nEdge-wise Voronoi source-to-target transfer checks:\n";
        const mimetic::EdgeTransferResult edge_transfer =
            interpolator.transfer_source_to_target_edges(source_handles, target_handles);
        ok = mimetic::test::near(static_cast<double>(edge_transfer.target_edges.size()),
                                 static_cast<double>(directed_edge_count(target_cells)), mimetic::kTolerance,
                                 "directed target Voronoi edge DOFs") &&
             ok;

        std::size_t target_dof = 0;
        for (const VoronoiCell& target : target_cells) {
            const mimetic::LocalPolygon target_poly = mimetic::local_polygon(mb, target.cell);
            const std::vector<mimetic::LocalEdge> target_edges = mimetic::local_edges(mb, target_poly);
            double target_cell_flux = 0.0;
            for (const mimetic::LocalEdge& edge : target_edges) {
                const Eigen::Vector2d a = target_poly.centroid + edge.a;
                const Eigen::Vector2d b = target_poly.centroid + edge.b;
                const double exact_flux =
                    mimetic::test::directed_edge_flux_from_absolute_field(a, b, mimetic::test::linear_absolute_field);
                target_cell_flux += edge_transfer.target_fluxes[target_dof];
                ok = mimetic::test::near(edge_transfer.target_fluxes[target_dof], exact_flux, 5.0e-11,
                                         "Voronoi target directed edge " + std::to_string(target_dof)) &&
                     ok;
                ++target_dof;
            }

            ok = mimetic::test::near(target_cell_flux, 2.0 * polygon_area(target.points), 5.0e-11,
                                     "Voronoi target cell edge-flux divergence") &&
                 ok;
        }

        if (!ok) {
            std::cout << "\n[FAILED] Voronoi conservative interpolation test failed.\n";
            return 1;
        }

        std::cout << "\n[SUCCESS] Conservative interpolation is exact on clipped Voronoi overlaps.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 2;
    }
}
