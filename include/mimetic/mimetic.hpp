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

/**
 * Absolute tolerance used in geometry degeneracy checks and regression tests.
 *
 * The tolerance is deliberately small because the current implementation and
 * tests are planar, deterministic, and use double precision arithmetic only.
 * Spherical geometry and large-scale mesh inputs should pass their own
 * problem-dependent tolerances into higher-level drivers.
 */
constexpr double kTolerance = 1.0e-12;

/**
 * Convert a MOAB return code into an exception with local context.
 *
 * This wrapper is used consistently so that test failures cannot silently
 * proceed after a failed mesh, adjacency, or tag operation.
 */
void check_moab(moab::ErrorCode code, const std::string& message);

/// Level-1 harmonic polynomial P_1(x,y)=x.
double p1(const Eigen::Vector2d& p);
/// Level-1 harmonic polynomial Q_1(x,y)=y.
double q1(const Eigen::Vector2d& p);
/// Level-2 harmonic polynomial P_2(x,y)=x^2-y^2.
double p2(const Eigen::Vector2d& p);
/// Level-2 harmonic polynomial Q_2(x,y)=2xy.
double q2(const Eigen::Vector2d& p);

/// Gradient of P_1 used in the reconstruction basis.
Eigen::Vector2d grad_p1(const Eigen::Vector2d& p);
/// Gradient of Q_1 used in the reconstruction basis.
Eigen::Vector2d grad_q1(const Eigen::Vector2d& p);
/// Gradient of P_2 used in the reconstruction basis.
Eigen::Vector2d grad_p2(const Eigen::Vector2d& p);
/// Gradient of Q_2 used in the reconstruction basis.
Eigen::Vector2d grad_q2(const Eigen::Vector2d& p);

/**
 * Integrate a scalar function on a straight edge with two-point Gauss-Legendre.
 *
 * This is exact for cubic polynomials on the edge. In the current level-2
 * planar method it exactly integrates all edge quantities used by the
 * manufactured tests.
 */
template <typename Func>
double integrate_edge_scalar(const Eigen::Vector2d& a, const Eigen::Vector2d& b, const Func& func)
{
    const double xi = 1.0 / std::sqrt(3.0);
    const double length = (b - a).norm();
    const Eigen::Vector2d midpoint = 0.5 * (a + b);
    const Eigen::Vector2d half_delta = 0.5 * (b - a);
    return 0.5 * length * (func(midpoint - xi * half_delta) + func(midpoint + xi * half_delta));
}

/**
 * Integrate a scalar function over a triangle using a symmetric three-point rule.
 *
 * The cell integrals in this prototype are computed by fan-triangulating a
 * centroid-relative polygon around the local origin. The rule is exact for
 * quadratic polynomials, which covers the level-2 gradient inner products and
 * divergence coupling terms used by Algorithm 1 in the report.
 */
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

/**
 * Coefficients of the level-2 mimetic reconstruction on one source polygon.
 *
 * The reconstructed field is
 *   u_h(x) = (d/2) x + a1 grad P1 + b1 grad Q1 + a2 grad P2 + b2 grad Q2.
 * The scalar c is retained for notation compatibility with the paper/report
 * but drops out of all gradient and line-integral evaluations in this prototype.
 */
struct ReconstructionCoeffs {
    double c;
    double d;
    double a1;
    double b1;
    double a2;
    double b2;
};

/**
 * Polygon geometry expressed in a centroid-relative local planar frame.
 *
 * `vertices` preserves the MOAB vertex ordering after enforcing positive
 * orientation. `points` are the corresponding coordinates shifted by
 * `centroid`, so the local origin is the source-cell centroid.
 */
struct LocalPolygon {
    std::vector<moab::EntityHandle> vertices;
    std::vector<Eigen::Vector2d> points;
    Eigen::Vector2d centroid;
    double area;
};

/**
 * Oriented polygon edge in a LocalPolygon frame.
 *
 * Edges follow counter-clockwise polygon order. The outward normal is therefore
 * (dy,-dx)/|edge| in the local planar frame.
 */
struct LocalEdge {
    moab::EntityHandle handle;
    Eigen::Vector2d a;
    Eigen::Vector2d b;
    Eigen::Vector2d outward_normal;
    double length;
};

/// Signed shoelace area; positive for counter-clockwise point order.
double signed_area(const std::vector<Eigen::Vector2d>& points);
/// Area-weighted polygon centroid in absolute planar coordinates.
Eigen::Vector2d polygon_centroid(const std::vector<Eigen::Vector2d>& points);
/// Return an existing MOAB edge between two vertices or create one if absent.
moab::EntityHandle find_or_create_edge(moab::Core& mb, moab::EntityHandle v0, moab::EntityHandle v1);
/// Extract a MOAB polygon/quad into a centroid-relative LocalPolygon.
LocalPolygon local_polygon(moab::Core& mb, moab::EntityHandle polygon);
/// Build ordered LocalEdge records from a LocalPolygon.
std::vector<LocalEdge> local_edges(moab::Core& mb, const LocalPolygon& polygon);
/// Create a MOAB polygon, using MBQUAD for 4-sided cells and MBPOLYGON otherwise.
moab::EntityHandle create_polygon(moab::Core& mb, const std::vector<Eigen::Vector2d>& points);
/// Convenience wrapper for creating a four-sided polygon.
moab::EntityHandle create_quad(moab::Core& mb, const std::array<Eigen::Vector2d, 4>& points);

/**
 * Level-2 mimetic interpolation kernel backed by MOAB tags.
 *
 * The class implements the source-cell reconstruction and reduction operations
 * described in Algorithms 1 and 2 of docs/mimetic_voronoi_report.tex. The class
 * owns no mesh data; it stores references to a MOAB Core instance and creates
 * tags for source edge fluxes, target edge fluxes, and per-cell coefficients.
 */
class MimeticInterpolator {
  public:
    /// Create SOURCE_FLUX, TARGET_FLUX, and COEFFS tags if needed.
    explicit MimeticInterpolator(moab::Core& moab_instance);

    /// MOAB tag storing signed integrated normal flux on source edges.
    moab::Tag source_flux_tag() const;
    /// MOAB tag storing signed integrated normal flux on target edges.
    moab::Tag target_flux_tag() const;
    /// MOAB tag storing ReconstructionCoeffs on source cells.
    moab::Tag coeffs_tag() const;

    /**
     * Reconstruct the level-2 mimetic field on one source polygon.
     *
     * Implements report Algorithm 1:
     * 1. read ordered source-edge fluxes from SOURCE_FLUX,
     * 2. compute constant divergence d=sum(U_f)/area,
     * 3. assemble the 4x4 harmonic Gram matrix V and moment RHS,
     * 4. solve for a1,b1,a2,b2 with Eigen LDLT,
     * 5. store coefficients on the source cell.
     */
    ReconstructionCoeffs reconstruct_source_polygon(moab::EntityHandle polygon);
    /// Evaluate u_h(x) in the source polygon's centroid-relative frame.
    Eigen::Vector2d velocity(const ReconstructionCoeffs& coeffs, const Eigen::Vector2d& p) const;
    /// Exact line integral of the gradient-form reconstruction from a to b.
    double line_integral(moab::EntityHandle source_polygon, const Eigen::Vector2d& a, const Eigen::Vector2d& b) const;
    /// Integrated outward-normal flux over one directed boundary edge.
    double edge_flux(const ReconstructionCoeffs& coeffs, const Eigen::Vector2d& a, const Eigen::Vector2d& b) const;
    /// Sum outward-normal flux over a clipped intersection polygon.
    double polygon_boundary_flux(const ReconstructionCoeffs& coeffs, const std::vector<Eigen::Vector2d>& points) const;
    /// Compute target edge fluxes for a target polygon contained in one source cell.
    std::vector<double> transfer_to_target_polygon_edges(moab::EntityHandle source_polygon, moab::EntityHandle target_polygon);

  private:
    moab::Core& mb_;
    moab::Tag tag_source_flux_ = 0;
    moab::Tag tag_target_flux_ = 0;
    moab::Tag tag_coeffs_ = 0;
};

}  // namespace mimetic

#endif
