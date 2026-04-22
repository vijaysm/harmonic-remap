#include "mimetic/mimetic.hpp"
#include "spherical_transfer_test_utils.hpp"
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

double clamp_unit(const double value)
{
    return std::max(-1.0, std::min(1.0, value));
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

std::vector<double> exact_surface_edge_moments(moab::Core& mb,
                                               const moab::EntityHandle cell,
                                               const std::size_t local_edge_index,
                                               const int order)
{
    mimetic::GeometryOptions spherical;
    spherical.mode = mimetic::GeometryMode::SphericalGnomonic;
    const mimetic::LocalPolygon poly = mimetic::local_polygon(mb, cell, spherical);
    const std::vector<mimetic::LocalEdge> edges = mimetic::local_edges(mb, poly);
    if (local_edge_index >= edges.size()) {
        throw std::runtime_error("Spherical edge index out of range");
    }

    const mimetic::LocalEdge& edge = edges[local_edge_index];
    const mimetic::GnomonicFrame frame{poly.n, poly.e_x, poly.e_y, 1.0};
    const Eigen::Vector3d a3 = poly.points_3d[local_edge_index].normalized();
    const Eigen::Vector3d b3 = poly.points_3d[(local_edge_index + 1) % poly.points_3d.size()].normalized();
    const double total_angle = std::acos(clamp_unit(a3.dot(b3)));

    std::vector<double> moments(static_cast<std::size_t>(order + 1), 0.0);
    for (int degree = 0; degree <= order; ++degree) {
        moments[degree] = mimetic::test_sphere::integrate_edge_gauss16(edge.a, edge.b, [&](const Eigen::Vector2d& p_local) {
            const Eigen::Vector2d xi = p_local + poly.centroid;
            const Eigen::Vector3d point = mimetic::inverse_gnomonic(xi, frame).normalized();
            const Eigen::Vector2d chart_vector =
                mimetic::pullback_contravariant_piola(mimetic::test_sphere::spherical_harmonic_gradient(point), xi, frame);
            double t = 0.0;
            if (total_angle > mimetic::kTolerance) {
                const double angle = std::acos(clamp_unit(a3.dot(point)));
                t = 2.0 * (angle / total_angle) - 1.0;
            }
            return chart_vector.dot(edge.outward_normal) * legendre(degree, t);
        });
    }
    return moments;
}

std::vector<Eigen::Vector2d> exact_surface_cell_vector_moments(moab::Core& mb,
                                                               const moab::EntityHandle cell,
                                                               const int order)
{
    mimetic::GeometryOptions spherical;
    spherical.mode = mimetic::GeometryMode::SphericalGnomonic;
    const mimetic::LocalPolygon poly = mimetic::local_polygon(mb, cell, spherical);
    const mimetic::GnomonicFrame frame{poly.n, poly.e_x, poly.e_y, 1.0};

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
                integral.x() += mimetic::integrate_triangle_scalar(a, b, c, [&](const Eigen::Vector2d& p_local) {
                    const Eigen::Vector2d xi = p_local + poly.centroid;
                    const Eigen::Vector3d point = mimetic::inverse_gnomonic(xi, frame).normalized();
                    const Eigen::Vector2d chart_vector =
                        mimetic::pullback_contravariant_piola(mimetic::test_sphere::spherical_harmonic_gradient(point), xi, frame);
                    return chart_vector.x() * std::pow(p_local.x(), a_pow) * std::pow(p_local.y(), b_pow);
                });
                integral.y() += mimetic::integrate_triangle_scalar(a, b, c, [&](const Eigen::Vector2d& p_local) {
                    const Eigen::Vector2d xi = p_local + poly.centroid;
                    const Eigen::Vector3d point = mimetic::inverse_gnomonic(xi, frame).normalized();
                    const Eigen::Vector2d chart_vector =
                        mimetic::pullback_contravariant_piola(mimetic::test_sphere::spherical_harmonic_gradient(point), xi, frame);
                    return chart_vector.y() * std::pow(p_local.x(), a_pow) * std::pow(p_local.y(), b_pow);
                });
            }
            moments.push_back(integral);
        }
    }
    return moments;
}

void set_source_moments(moab::Core& mb,
                        mimetic::PlanarMomentInterpolator& interpolator,
                        const std::vector<moab::EntityHandle>& source_cells,
                        const int order)
{
    for (const moab::EntityHandle cell : source_cells) {
        const mimetic::LocalPolygon poly = mimetic::local_polygon(mb, cell, true);
        for (std::size_t edge_index = 0; edge_index < poly.vertices.size(); ++edge_index) {
            interpolator.set_source_edge_moments(cell, edge_index, exact_surface_edge_moments(mb, cell, edge_index, order));
        }
        interpolator.set_source_cell_vector_moments(cell, exact_surface_cell_vector_moments(mb, cell, std::max(0, order - 1)));
    }
}

struct CaseMetrics {
    double l2_moment0_rel = 0.0;
    double l2_all_rel = 0.0;
    double max_conforming_divergence_residual = 0.0;
};

CaseMetrics run_case(const int source_n,
                     const int target_n,
                     const int order)
{
    moab::Core mb;
    const std::vector<moab::EntityHandle> source_cells = mimetic::test_sphere::generate_cubed_sphere(mb, source_n);
    const std::vector<moab::EntityHandle> target_cells = mimetic::test_sphere::generate_cubed_sphere(mb, target_n);

    mimetic::PlanarMomentInterpolator interpolator(mb);
    interpolator.set_spherical(true);

    mimetic::MomentMethodOptions options;
    options.edge_moment_order = order;
    options.cell_moment_order = std::max(0, order - 1);
    options.quadrature_points = 10;
    options.regularization = 1.0e-12;
    options.exact_constraints = false;

    set_source_moments(mb, interpolator, source_cells, order);
    for (const moab::EntityHandle cell : source_cells) {
        interpolator.reconstruct_source_polygon(cell, options);
    }

    const mimetic::EdgeMomentTransferResult raw =
        interpolator.transfer_source_to_target_edge_moments(source_cells, target_cells, order);
    const mimetic::ConformingEdgeMomentTransferResult conforming =
        interpolator.project_target_edge_moments_to_hdiv_conforming(source_cells, target_cells, raw);

    double flux_num = 0.0;
    double flux_den = 0.0;
    double all_num = 0.0;
    double all_den = 0.0;
    double max_conforming_divergence_residual = 0.0;
    std::size_t dof = 0;
    for (std::size_t cell_index = 0; cell_index < target_cells.size(); ++cell_index) {
        const moab::EntityHandle cell = target_cells[cell_index];
        const mimetic::LocalPolygon poly = mimetic::local_polygon(mb, cell, true);
        const std::vector<mimetic::LocalEdge> edges = mimetic::local_edges(mb, poly);
        double conf_div = 0.0;
        for (std::size_t edge_index = 0; edge_index < edges.size(); ++edge_index, ++dof) {
            const std::vector<double> exact = exact_surface_edge_moments(mb, cell, edge_index, order);
            const std::vector<double>& moments = raw.target_moments[dof];
            const std::vector<double>& conf_moments = conforming.target_moments[dof];
            conf_div += conf_moments[0];
            const double flux_error = moments[0] - exact[0];
            flux_num += flux_error * flux_error;
            flux_den += exact[0] * exact[0];
            for (int degree = 0; degree <= order; ++degree) {
                const double err = moments[degree] - exact[degree];
                all_num += err * err;
                all_den += exact[degree] * exact[degree];
            }
        }
        max_conforming_divergence_residual =
            std::max(max_conforming_divergence_residual,
                     std::abs(conf_div - conforming.target_divergence_integrals[cell_index]));
    }

    CaseMetrics metrics;
    metrics.l2_moment0_rel = (flux_den > 1.0e-30) ? std::sqrt(flux_num / flux_den) : std::sqrt(flux_num);
    metrics.l2_all_rel = (all_den > 1.0e-30) ? std::sqrt(all_num / all_den) : std::sqrt(all_num);
    metrics.max_conforming_divergence_residual = max_conforming_divergence_residual;
    return metrics;
}

}  // namespace

int main()
{
    try {
        std::cout << "--- Spherical High-Order Edge Moment Transfer Test ---\n";

        const CaseMetrics p2_coarse = run_case(4, 6, 2);
        const CaseMetrics p2_fine = run_case(6, 8, 2);
        const CaseMetrics p3_coarse = run_case(4, 6, 3);
        const CaseMetrics p3_fine = run_case(6, 8, 3);

        std::cout << std::scientific << std::setprecision(6)
                  << "p2 coarse l2_m0=" << p2_coarse.l2_moment0_rel
                  << " l2_all=" << p2_coarse.l2_all_rel
                  << " conf_div=" << p2_coarse.max_conforming_divergence_residual << "\n"
                  << "p2 fine   l2_m0=" << p2_fine.l2_moment0_rel
                  << " l2_all=" << p2_fine.l2_all_rel
                  << " conf_div=" << p2_fine.max_conforming_divergence_residual << "\n"
                  << "p3 coarse l2_m0=" << p3_coarse.l2_moment0_rel
                  << " l2_all=" << p3_coarse.l2_all_rel
                  << " conf_div=" << p3_coarse.max_conforming_divergence_residual << "\n"
                  << "p3 fine   l2_m0=" << p3_fine.l2_moment0_rel
                  << " l2_all=" << p3_fine.l2_all_rel
                  << " conf_div=" << p3_fine.max_conforming_divergence_residual << "\n";

        bool ok = true;
        ok = mimetic::test::near(p2_coarse.max_conforming_divergence_residual, 0.0, 5.0e-13,
                                 "p2 coarse spherical conforming divergence") && ok;
        ok = mimetic::test::near(p2_fine.max_conforming_divergence_residual, 0.0, 5.0e-13,
                                 "p2 fine spherical conforming divergence") && ok;
        ok = mimetic::test::near(p3_coarse.max_conforming_divergence_residual, 0.0, 5.0e-13,
                                 "p3 coarse spherical conforming divergence") && ok;
        ok = mimetic::test::near(p3_fine.max_conforming_divergence_residual, 0.0, 5.0e-13,
                                 "p3 fine spherical conforming divergence") && ok;

        if (!(p2_fine.l2_moment0_rel < p2_coarse.l2_moment0_rel &&
              p3_fine.l2_moment0_rel < p3_coarse.l2_moment0_rel &&
              p3_fine.l2_moment0_rel < p2_fine.l2_moment0_rel)) {
            throw std::runtime_error("Spherical high-order moment errors did not improve under refinement/order increase");
        }

        if (!ok) {
            throw std::runtime_error("Spherical high-order moment regression failed");
        }

        std::cout << "\n[SUCCESS] Spherical high-order edge moments transfer conservatively and improve under refinement.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }
}
