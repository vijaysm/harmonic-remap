#ifndef MIMETIC_MIMETIC_HPP
#define MIMETIC_MIMETIC_HPP

#include <Eigen/Dense>
#include <moab/Core.hpp>

#include <array>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace mimetic {

constexpr double kTolerance = 1.0e-12;

void check_moab(moab::ErrorCode code, const std::string& message);

double p1(const Eigen::Vector2d& p);
double q1(const Eigen::Vector2d& p);
double p2(const Eigen::Vector2d& p);
double q2(const Eigen::Vector2d& p);

Eigen::Vector2d grad_p1(const Eigen::Vector2d& p);
Eigen::Vector2d grad_q1(const Eigen::Vector2d& p);
Eigen::Vector2d grad_p2(const Eigen::Vector2d& p);
Eigen::Vector2d grad_q2(const Eigen::Vector2d& p);

template <typename Func>
double integrate_edge_scalar(const Eigen::Vector2d& a, const Eigen::Vector2d& b, const Func& func)
{
    const double xi = 1.0 / std::sqrt(3.0);
    const double length = (b - a).norm();
    const Eigen::Vector2d midpoint = 0.5 * (a + b);
    const Eigen::Vector2d half_delta = 0.5 * (b - a);
    return 0.5 * length * (func(midpoint - xi * half_delta) + func(midpoint + xi * half_delta));
}

template <typename Func>
double integrate_triangle_scalar(const Eigen::Vector2d& a,
                                 const Eigen::Vector2d& b,
                                 const Eigen::Vector2d& c,
                                 const Func& func)
{
    const double signed_twice_area = (b - a).x() * (c - a).y() - (b - a).y() * (c - a).x();
    const double area = 0.5 * std::abs(signed_twice_area);
    const std::array<std::array<double, 3>, 3> bary = {{
        {{1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0}},
        {{1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0}},
        {{2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0}},
    }};

    double sum = 0.0;
    for (const auto& w : bary) {
        const Eigen::Vector2d p = w[0] * a + w[1] * b + w[2] * c;
        sum += func(p);
    }
    return area * sum / 3.0;
}

struct ReconstructionCoeffs {
    double c;
    double d;
    double a1;
    double b1;
    double a2;
    double b2;
};

struct LocalPolygon {
    std::vector<moab::EntityHandle> vertices;
    std::vector<Eigen::Vector2d> points;
    Eigen::Vector2d centroid;
    double area;
};

struct LocalEdge {
    moab::EntityHandle handle;
    Eigen::Vector2d a;
    Eigen::Vector2d b;
    Eigen::Vector2d outward_normal;
    double length;
};

double signed_area(const std::vector<Eigen::Vector2d>& points);
Eigen::Vector2d polygon_centroid(const std::vector<Eigen::Vector2d>& points);
moab::EntityHandle find_or_create_edge(moab::Core& mb, moab::EntityHandle v0, moab::EntityHandle v1);
LocalPolygon local_polygon(moab::Core& mb, moab::EntityHandle polygon);
std::vector<LocalEdge> local_edges(moab::Core& mb, const LocalPolygon& polygon);
moab::EntityHandle create_polygon(moab::Core& mb, const std::vector<Eigen::Vector2d>& points);
moab::EntityHandle create_quad(moab::Core& mb, const std::array<Eigen::Vector2d, 4>& points);

class MimeticInterpolator {
  public:
    explicit MimeticInterpolator(moab::Core& moab_instance);

    moab::Tag source_flux_tag() const;
    moab::Tag target_flux_tag() const;
    moab::Tag coeffs_tag() const;

    ReconstructionCoeffs reconstruct_source_polygon(moab::EntityHandle polygon);
    Eigen::Vector2d velocity(const ReconstructionCoeffs& coeffs, const Eigen::Vector2d& p) const;
    double line_integral(moab::EntityHandle source_polygon, const Eigen::Vector2d& a, const Eigen::Vector2d& b) const;
    double edge_flux(const ReconstructionCoeffs& coeffs, const Eigen::Vector2d& a, const Eigen::Vector2d& b) const;
    double polygon_boundary_flux(const ReconstructionCoeffs& coeffs, const std::vector<Eigen::Vector2d>& points) const;
    std::vector<double> transfer_to_target_polygon_edges(moab::EntityHandle source_polygon, moab::EntityHandle target_polygon);

  private:
    moab::Core& mb_;
    moab::Tag tag_source_flux_ = 0;
    moab::Tag tag_target_flux_ = 0;
    moab::Tag tag_coeffs_ = 0;
};

}  // namespace mimetic

#endif
