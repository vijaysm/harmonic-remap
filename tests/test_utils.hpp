#ifndef MIMETIC_TEST_UTILS_HPP
#define MIMETIC_TEST_UTILS_HPP

#include "mimetic/mimetic.hpp"

#include <Eigen/Dense>
#include <iomanip>
#include <iostream>
#include <string>

namespace mimetic {
namespace test {

inline bool near(const double actual, const double expected, const double tolerance, const std::string& label)
{
    const double error = std::abs(actual - expected);
    std::cout << "  " << std::left << std::setw(40) << label << " actual=" << std::right << std::setw(22)
              << std::setprecision(15) << actual << " expected=" << std::setw(22) << expected
              << " error=" << error << "\n";
    return error <= tolerance;
}

inline Eigen::Vector2d constant_field(const Eigen::Vector2d&)
{
    return Eigen::Vector2d(1.0, 1.0);
}

inline Eigen::Vector2d linear_source_field(const Eigen::Vector2d& p)
{
    return Eigen::Vector2d(p.x(), p.y());
}

inline Eigen::Vector2d linear_absolute_field(const Eigen::Vector2d& p)
{
    return Eigen::Vector2d(p.x(), p.y());
}

template <typename Field>
double directed_edge_flux_from_absolute_field(const Eigen::Vector2d& a, const Eigen::Vector2d& b, const Field& field)
{
    const Eigen::Vector2d delta = b - a;
    const Eigen::Vector2d normal(delta.y(), -delta.x());
    return integrate_edge_scalar(a, b, [&](const Eigen::Vector2d& p) { return field(p).dot(normal / delta.norm()); });
}

template <typename Field>
void set_source_fluxes_from_field(moab::Core& mb,
                                  mimetic::MimeticInterpolator& interpolator,
                                  const moab::EntityHandle polygon,
                                  const Field& field)
{
    const LocalPolygon poly = local_polygon(mb, polygon);
    const std::vector<LocalEdge> edges = local_edges(mb, poly);
    for (std::size_t edge_index = 0; edge_index < edges.size(); ++edge_index) {
        const LocalEdge& edge = edges[edge_index];
        const double flux =
            integrate_edge_scalar(edge.a, edge.b, [&](const Eigen::Vector2d& p) { return field(p).dot(edge.outward_normal); });
        interpolator.set_source_edge_flux(polygon, edge_index, flux);
    }
}

template <typename Field>
void set_source_fluxes_from_absolute_field(moab::Core& mb,
                                           mimetic::MimeticInterpolator& interpolator,
                                           const moab::EntityHandle polygon,
                                           const Field& field)
{
    const LocalPolygon poly = local_polygon(mb, polygon);
    const std::vector<LocalEdge> edges = local_edges(mb, poly);
    for (std::size_t edge_index = 0; edge_index < edges.size(); ++edge_index) {
        const LocalEdge& edge = edges[edge_index];
        const double flux = integrate_edge_scalar(edge.a, edge.b, [&](const Eigen::Vector2d& p) {
            return field(p + poly.centroid).dot(edge.outward_normal);
        });
        interpolator.set_source_edge_flux(polygon, edge_index, flux);
    }
}

}  // namespace test
}  // namespace mimetic

#endif
