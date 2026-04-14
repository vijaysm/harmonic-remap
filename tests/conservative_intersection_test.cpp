#include "mimetic/mimetic.hpp"
#include "test_utils.hpp"

#include <Eigen/Dense>
#include <moab/Core.hpp>

#include <algorithm>
#include <array>
#include <exception>
#include <iostream>
#include <vector>

namespace {

struct Rect {
    double xmin;
    double xmax;
    double ymin;
    double ymax;
    moab::EntityHandle cell;
};

std::vector<Rect> create_rect_mesh(moab::Core& mb, const std::vector<double>& xs, const std::vector<double>& ys)
{
    std::vector<Rect> cells;
    for (std::size_t j = 0; j + 1 < ys.size(); ++j) {
        for (std::size_t i = 0; i + 1 < xs.size(); ++i) {
            const std::array<Eigen::Vector2d, 4> points = {{
                Eigen::Vector2d(xs[i], ys[j]),
                Eigen::Vector2d(xs[i + 1], ys[j]),
                Eigen::Vector2d(xs[i + 1], ys[j + 1]),
                Eigen::Vector2d(xs[i], ys[j + 1]),
            }};
            cells.push_back(Rect{xs[i], xs[i + 1], ys[j], ys[j + 1], mimetic::create_quad(mb, points)});
        }
    }
    return cells;
}

bool rect_intersection(const Rect& a, const Rect& b, Rect& intersection)
{
    intersection.xmin = std::max(a.xmin, b.xmin);
    intersection.xmax = std::min(a.xmax, b.xmax);
    intersection.ymin = std::max(a.ymin, b.ymin);
    intersection.ymax = std::min(a.ymax, b.ymax);
    intersection.cell = 0;
    return intersection.xmax > intersection.xmin + mimetic::kTolerance &&
           intersection.ymax > intersection.ymin + mimetic::kTolerance;
}

double rect_area(const Rect& rect)
{
    return (rect.xmax - rect.xmin) * (rect.ymax - rect.ymin);
}

std::vector<Eigen::Vector2d> rect_points_in_source_frame(const Rect& rect, const mimetic::LocalPolygon& source)
{
    return {
        Eigen::Vector2d(rect.xmin, rect.ymin) - source.centroid,
        Eigen::Vector2d(rect.xmax, rect.ymin) - source.centroid,
        Eigen::Vector2d(rect.xmax, rect.ymax) - source.centroid,
        Eigen::Vector2d(rect.xmin, rect.ymax) - source.centroid,
    };
}

}  // namespace

int main()
{
    try {
        std::cout << "--- Conservative Source/Target Intersection Test ---\n\n";

        moab::Core mb;
        mimetic::MimeticInterpolator interpolator(mb);

        const std::vector<Rect> source_cells = create_rect_mesh(mb, {0.0, 0.5, 1.0}, {0.0, 0.5, 1.0});
        const std::vector<Rect> target_cells = create_rect_mesh(mb, {0.0, 0.25, 0.65, 1.0}, {0.0, 0.40, 0.70, 1.0});

        for (const Rect& source : source_cells) {
            mimetic::test::set_source_fluxes_from_field(
                mb, interpolator.source_flux_tag(), source.cell, mimetic::test::linear_source_field);
            interpolator.reconstruct_source_polygon(source.cell);
        }

        bool ok = true;
        double total_overlap_area = 0.0;
        double total_boundary_flux = 0.0;
        double total_expected_flux = 0.0;
        int overlap_count = 0;

        for (const Rect& source_rect : source_cells) {
            const mimetic::LocalPolygon source_poly = mimetic::local_polygon(mb, source_rect.cell);
            mimetic::ReconstructionCoeffs coeffs{};
            mimetic::check_moab(mb.tag_get_data(interpolator.coeffs_tag(), &source_rect.cell, 1, &coeffs),
                                "Failed to read source coefficients");

            for (const Rect& target_rect : target_cells) {
                Rect overlap{};
                if (!rect_intersection(source_rect, target_rect, overlap)) {
                    continue;
                }

                const std::vector<Eigen::Vector2d> overlap_points = rect_points_in_source_frame(overlap, source_poly);
                const double boundary_flux = interpolator.polygon_boundary_flux(coeffs, overlap_points);
                const double expected_flux = coeffs.d * rect_area(overlap);

                ++overlap_count;
                total_overlap_area += rect_area(overlap);
                total_boundary_flux += boundary_flux;
                total_expected_flux += expected_flux;

                ok = mimetic::test::near(boundary_flux, expected_flux, 2.0e-12,
                                         "overlap " + std::to_string(overlap_count) + " boundary flux") &&
                     ok;
            }
        }

        std::cout << "\nAggregate overlap checks:\n";
        ok = mimetic::test::near(total_overlap_area, 1.0, mimetic::kTolerance, "source-target overlap area") && ok;
        ok = mimetic::test::near(total_boundary_flux, total_expected_flux, 5.0e-12,
                                 "summed overlap divergence theorem") &&
             ok;
        ok = mimetic::test::near(total_expected_flux, 2.0, 5.0e-12, "exact integral of div(x,y)") && ok;

        if (!ok) {
            std::cout << "\n[FAILED] Conservative intersection test failed.\n";
            return 1;
        }

        std::cout << "\n[SUCCESS] Every source-target intersection integrates the reconstructed field exactly.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 2;
    }
}
