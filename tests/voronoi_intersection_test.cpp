#include "mimetic/mimetic.hpp"
#include "test_utils.hpp"

#include <Eigen/Dense>
#include <moab/Core.hpp>

#include <algorithm>
#include <exception>
#include <iostream>
#include <vector>

namespace {

struct VoronoiCell {
    Eigen::Vector2d seed;
    std::vector<Eigen::Vector2d> points;
    moab::EntityHandle cell;
};

double polygon_area(const std::vector<Eigen::Vector2d>& points)
{
    return std::abs(mimetic::signed_area(points));
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

        for (const VoronoiCell& source : source_cells) {
            mimetic::test::set_source_fluxes_from_field(
                mb, interpolator.source_flux_tag(), source.cell, mimetic::test::linear_source_field);
            interpolator.reconstruct_source_polygon(source.cell);
        }

        bool ok = true;
        int overlap_count = 0;
        double total_overlap_area = 0.0;
        double total_boundary_flux = 0.0;
        double total_expected_flux = 0.0;

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
        ok = mimetic::test::near(total_overlap_area, 1.0, 5.0e-11, "total overlap area") && ok;
        ok = mimetic::test::near(total_boundary_flux, total_expected_flux, 5.0e-10,
                                 "summed overlap boundary flux") &&
             ok;
        ok = mimetic::test::near(total_expected_flux, 2.0, 5.0e-10, "integral of div(x,y)") && ok;

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
