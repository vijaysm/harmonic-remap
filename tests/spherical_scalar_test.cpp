#include "mimetic/mimetic.hpp"
#include "spherical_transfer_test_utils.hpp"

#include <Eigen/Dense>
#include <moab/Core.hpp>

#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

namespace {

double scalar_field(const Eigen::Vector3d& point)
{
    const Eigen::Vector3d p = point.normalized();
    return 0.5 * (3.0 * p.z() * p.z() - 1.0);
}

double polygon_signed_area(const std::vector<Eigen::Vector2d>& points)
{
    double area2 = 0.0;
    for (std::size_t i = 0; i < points.size(); ++i) {
        const Eigen::Vector2d& a = points[i];
        const Eigen::Vector2d& b = points[(i + 1) % points.size()];
        area2 += a.x() * b.y() - b.x() * a.y();
    }
    return 0.5 * area2;
}

double polygon_area(const std::vector<Eigen::Vector2d>& points)
{
    if (points.size() < 3) {
        return 0.0;
    }
    return std::abs(polygon_signed_area(points));
}

std::vector<Eigen::Vector2d> clip_polygon_by_halfplane(const std::vector<Eigen::Vector2d>& polygon,
                                                       const Eigen::Vector2d& a,
                                                       const Eigen::Vector2d& b,
                                                       const double tolerance)
{
    std::vector<Eigen::Vector2d> output;
    if (polygon.empty()) {
        return output;
    }

    const Eigen::Vector2d edge = b - a;
    auto inside = [&](const Eigen::Vector2d& p) {
        const double cross = edge.x() * (p.y() - a.y()) - edge.y() * (p.x() - a.x());
        return cross >= -tolerance;
    };

    auto intersect = [&](const Eigen::Vector2d& p0, const Eigen::Vector2d& p1) -> Eigen::Vector2d {
        const Eigen::Vector2d d = p1 - p0;
        const double numerator = edge.x() * (a.y() - p0.y()) - edge.y() * (a.x() - p0.x());
        const double denominator = edge.x() * d.y() - edge.y() * d.x();
        if (std::abs(denominator) <= tolerance) {
            return p0;
        }
        return p0 + (numerator / denominator) * d;
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

std::vector<Eigen::Vector2d> clip_convex_polygon(const std::vector<Eigen::Vector2d>& subject,
                                                 const std::vector<Eigen::Vector2d>& clip,
                                                 const double tolerance)
{
    std::vector<Eigen::Vector2d> output = subject;
    if (output.empty()) {
        return output;
    }

    std::vector<Eigen::Vector2d> clip_ccw = clip;
    if (polygon_area(clip_ccw) > 0.0) {
        if (polygon_signed_area(clip_ccw) < 0.0) {
            std::reverse(clip_ccw.begin(), clip_ccw.end());
        }
    }

    for (std::size_t i = 0; i < clip_ccw.size(); ++i) {
        output = clip_polygon_by_halfplane(output,
                                           clip_ccw[i],
                                           clip_ccw[(i + 1) % clip_ccw.size()],
                                           tolerance);
        if (output.size() < 3) {
            return {};
        }
    }
    return output;
}

double exact_cell_average(moab::Core& mb, const moab::EntityHandle cell)
{
    mimetic::GeometryOptions options;
    options.mode = mimetic::GeometryMode::SphericalGnomonic;
    const mimetic::SphericalPolygon poly = mimetic::spherical_polygon(mb, cell, options);
    const Eigen::Vector2d origin(0.0, 0.0);

    double integral = 0.0;
    for (std::size_t i = 0; i < poly.local_points.size(); ++i) {
        const Eigen::Vector2d& a = poly.local_points[i];
        const Eigen::Vector2d& b = poly.local_points[(i + 1) % poly.local_points.size()];
        integral += mimetic::integrate_triangle_scalar(origin, a, b, [&](const Eigen::Vector2d& local_point) {
            const Eigen::Vector2d xi = local_point + poly.projected_centroid;
            const Eigen::Vector3d surface_point = mimetic::inverse_gnomonic(xi, poly.frame);
            const double weight = mimetic::gnomonic_area_scale(xi, poly.frame);
            return scalar_field(surface_point) * weight;
        });
    }

    return integral / poly.spherical_area;
}

struct Metrics {
    int source_n;
    int target_n;
    double l2_relative_error;
    double max_error;
    double max_target_coverage_residual;
    double global_conservation_residual;
};

Metrics run_case(const int source_n, const int target_n)
{
    moab::Core mb;
    const std::vector<moab::EntityHandle> source_mesh = mimetic::test_sphere::generate_cubed_sphere(mb, source_n);
    const std::vector<moab::EntityHandle> target_mesh = mimetic::test_sphere::generate_cubed_sphere(mb, target_n);

    mimetic::GeometryOptions options;
    options.mode = mimetic::GeometryMode::SphericalGnomonic;

    std::vector<double> source_values(source_mesh.size(), 0.0);
    std::vector<mimetic::SphericalPolygon> source_polys;
    source_polys.reserve(source_mesh.size());
    for (std::size_t i = 0; i < source_mesh.size(); ++i) {
        source_values[i] = exact_cell_average(mb, source_mesh[i]);
        source_polys.push_back(mimetic::spherical_polygon(mb, source_mesh[i], options));
    }

    double l2_num = 0.0;
    double l2_den = 0.0;
    double max_error = 0.0;
    double max_target_coverage_residual = 0.0;
    double total_source_integral = 0.0;
    double total_target_integral = 0.0;

    for (std::size_t i = 0; i < source_mesh.size(); ++i) {
        total_source_integral += source_values[i] * source_polys[i].spherical_area;
    }

    for (const moab::EntityHandle target : target_mesh) {
        const mimetic::SphericalPolygon target_poly = mimetic::spherical_polygon(mb, target, options);
        double remapped_integral = 0.0;
        double covered_area = 0.0;

        for (std::size_t i = 0; i < source_polys.size(); ++i) {
            std::vector<Eigen::Vector2d> target_in_source;
            target_in_source.reserve(target_poly.points.size());
            bool valid = true;
            for (const Eigen::Vector3d& p : target_poly.points) {
                try {
                    target_in_source.push_back(mimetic::project_gnomonic(p, source_polys[i].frame));
                } catch (const std::runtime_error&) {
                    valid = false;
                    break;
                }
            }
            if (!valid) {
                continue;
            }

            const std::vector<Eigen::Vector2d> overlap =
                clip_convex_polygon(target_in_source, source_polys[i].projected_points, 1.0e-13);
            if (polygon_area(overlap) <= 1.0e-14) {
                continue;
            }

            const double overlap_spherical_area =
                mimetic::chart_polygon_surface_area(overlap, source_polys[i].frame);
            remapped_integral += source_values[i] * overlap_spherical_area;
            covered_area += overlap_spherical_area;
        }

        const double remapped_average = remapped_integral / target_poly.spherical_area;
        const double exact_average = exact_cell_average(mb, target);
        const double error = std::abs(remapped_average - exact_average);
        l2_num += target_poly.spherical_area * error * error;
        l2_den += target_poly.spherical_area * exact_average * exact_average;
        max_error = std::max(max_error, error);
        max_target_coverage_residual =
            std::max(max_target_coverage_residual, std::abs(covered_area - target_poly.spherical_area));
        total_target_integral += remapped_integral;
    }

    Metrics metrics{};
    metrics.source_n = source_n;
    metrics.target_n = target_n;
    metrics.l2_relative_error = l2_den > std::numeric_limits<double>::epsilon() ? std::sqrt(l2_num / l2_den) : std::sqrt(l2_num);
    metrics.max_error = max_error;
    metrics.max_target_coverage_residual = max_target_coverage_residual;
    metrics.global_conservation_residual = std::abs(total_target_integral - total_source_integral);
    return metrics;
}

void print_metrics(const Metrics& metrics)
{
    std::cout << "  source_n=" << metrics.source_n << " target_n=" << metrics.target_n << "\n"
              << std::scientific << std::setprecision(6)
              << "  scalar_l2_rel=" << metrics.l2_relative_error
              << " scalar_linf=" << metrics.max_error << "\n"
              << "  target_coverage_residual=" << metrics.max_target_coverage_residual
              << " global_conservation_residual=" << metrics.global_conservation_residual << "\n";
}

}

int main()
{
    try {
        std::cout << "--- Spherical Scalar Remap Control Test ---\n\n";

        const Metrics coarse = run_case(4, 6);
        const Metrics fine = run_case(6, 8);

        std::cout << "Coarse case:\n";
        print_metrics(coarse);
        std::cout << "Fine case:\n";
        print_metrics(fine);

        const bool ok = fine.l2_relative_error < coarse.l2_relative_error &&
                        fine.max_error < coarse.max_error &&
                        coarse.max_target_coverage_residual <= mimetic::kConservationTolerance &&
                        fine.max_target_coverage_residual <= mimetic::kConservationTolerance &&
                        coarse.global_conservation_residual <= mimetic::kConservationTolerance &&
                        fine.global_conservation_residual <= mimetic::kConservationTolerance;
        if (!ok) {
            std::cout << "\n[FAILED] Scalar control test did not converge under refinement.\n";
            return 1;
        }

        std::cout << "\n[SUCCESS] Scalar control remap converges under refinement.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 2;
    }
}
