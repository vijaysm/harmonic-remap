#ifndef MIMETIC_MIMETIC_HPP
#define MIMETIC_MIMETIC_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <moab/Core.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <map>
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
constexpr double kConservationTolerance = 5.0e-13;

enum class GeometryMode {
    Planar,
    SphericalGnomonic,
};

struct GeometryOptions {
    GeometryMode mode = GeometryMode::Planar;
    double radius = 1.0;
    double conservation_tolerance = kConservationTolerance;
    double geometry_tolerance = 1.0e-13;
    bool metric_weighted = false;
};

/**
 * Convert a MOAB return code into an exception with local context.
 *
 * This wrapper is used consistently so that test failures cannot silently
 * proceed after a failed mesh, adjacency, or tag operation.
 */
void check_moab(moab::ErrorCode code, const std::string& message);

/**
 * Evaluate the k-th harmonic basis functions P_k, Q_k and their gradients.
 * P_k(x,y) = Re((x+iy)^k), Q_k(x,y) = Im((x+iy)^k).
 */
void eval_harmonic_basis(int k, const Eigen::Vector2d& p,
                         double& P, double& Q,
                         Eigen::Vector2d& gradP, Eigen::Vector2d& gradQ);

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
    const double length = (b - a).norm();
    const Eigen::Vector2d mid = 0.5 * (a + b);
    const Eigen::Vector2d half_delta = 0.5 * (b - a);
    // 4-point Gauss-Legendre (degree 7)
    const double x1 = 0.3399810435848563, w1 = 0.6521451548625461;
    const double x2 = 0.8611363115940526, w2 = 0.3478548451374538;
    return 0.5 * length * (w1 * func(mid - x1 * half_delta) + w1 * func(mid + x1 * half_delta) +
                           w2 * func(mid - x2 * half_delta) + w2 * func(mid + x2 * half_delta));
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

    const double a1 = 0.4701420641051151, b1 = 1.0 - 2.0 * a1, w1 = 0.1323941527885062;
    const double a2 = 0.1012865073234563, b2 = 1.0 - 2.0 * a2, w2 = 0.1259391805448271;
    struct Pt { double w, u, v, w_bary; };
    const Pt pts[7] = {
        {0.225, 1.0/3.0, 1.0/3.0, 1.0/3.0},
        {w1, a1, a1, b1}, {w1, a1, b1, a1}, {w1, b1, a1, a1},
        {w2, a2, a2, b2}, {w2, a2, b2, a2}, {w2, b2, a2, a2}
    };
    double sum = 0.0;
    for (int i = 0; i < 7; ++i) {
        sum += pts[i].w * func(pts[i].u * a + pts[i].v * b + pts[i].w_bary * c);
    }
    return area * sum;
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
    double d;
    std::vector<double> harmonic;
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
    double spherical_area;

    // Spherical geometry data
    std::vector<Eigen::Vector3d> points_3d;
    Eigen::Vector3d centroid_3d;
    Eigen::Vector3d e_x;
    Eigen::Vector3d e_y;
    Eigen::Vector3d n;
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

struct GnomonicFrame {
    Eigen::Vector3d center;
    Eigen::Vector3d e_x;
    Eigen::Vector3d e_y;
    double radius = 1.0;
};

struct SphericalEdge {
    moab::EntityHandle handle;
    Eigen::Vector3d a;
    Eigen::Vector3d b;
    Eigen::Vector2d chart_a;
    Eigen::Vector2d chart_b;
    double arc_length;
};

struct SphericalPolygon {
    std::vector<moab::EntityHandle> vertices;
    std::vector<Eigen::Vector3d> points;
    std::vector<Eigen::Vector2d> projected_points;
    std::vector<Eigen::Vector2d> local_points;
    Eigen::Vector2d projected_centroid;
    double chart_area;
    double spherical_area;
    GnomonicFrame frame;
};

/**
 * Directed cell-edge degree of freedom.
 *
 * The current prototype treats source and target edge data as cell-local,
 * directed edge DOFs. This avoids silently losing orientation information on
 * shared MOAB edges. Production shared-edge meshes should add an explicit
 * orientation map before collapsing these directed DOFs to unique mesh edges.
 */
struct DirectedEdgeDof {
    moab::EntityHandle polygon;
    moab::EntityHandle edge;
    std::size_t local_edge_index;
};

/// One clipped source-cell contribution to one directed target edge.
struct EdgeTransferContribution {
    std::size_t target_dof_index;
    moab::EntityHandle source_polygon;
    Eigen::Vector2d segment_a;
    Eigen::Vector2d segment_b;
    double flux;
};

/// Result of applying the reconstructed source field to all directed target edges.
struct EdgeTransferResult {
    std::vector<DirectedEdgeDof> target_edges;
    std::vector<double> target_fluxes;
    std::vector<EdgeTransferContribution> contributions;
};

enum class CellAverageReductionMode {
    Harmonic,
};

struct CellAverageContribution {
    std::size_t target_cell_index;
    moab::EntityHandle source_polygon;
    double overlap_area;
    Eigen::Vector3d integral;
};

struct CellAverageTransferResult {
    std::vector<moab::EntityHandle> target_cells;
    std::vector<double> target_areas;
    std::vector<Eigen::Vector3d> target_integrals;
    std::vector<Eigen::Vector3d> target_averages;
    std::vector<CellAverageContribution> contributions;
};

/**
 * Sparse source-edge to target-edge projection.
 *
 * `matrix.rows()` equals `target_edges.size()` and `matrix.cols()` equals
 * `source_edges.size()`. The matrix maps directed source edge fluxes to directed
 * target edge fluxes: U_t = P U_s.
 */
struct SparseEdgeProjection {
    Eigen::SparseMatrix<double, Eigen::RowMajor> matrix;
    std::vector<DirectedEdgeDof> source_edges;
    std::vector<DirectedEdgeDof> target_edges;
};

/**
 * Globally conforming target-edge flux field derived from a raw directed transfer.
 *
 * `target_fluxes` stores one signed flux per directed target edge in the same
 * ordering convention as EdgeTransferResult.  `unique_edge_fluxes` stores one
 * global edge flux per geometric target edge after collapsing opposite cell-local
 * orientations.  The relation is
 *   target_fluxes[i] = target_edge_signs[i] * unique_edge_fluxes[target_edge_to_unique[i]].
 * `target_divergence_integrals` stores the exact per-target-cell divergence
 * constraints assembled from the source-cell reconstructions.
 */
struct ConformingEdgeTransferResult {
    std::vector<DirectedEdgeDof> target_edges;
    std::vector<double> target_fluxes;
    std::vector<moab::EntityHandle> target_cells;
    std::vector<double> target_divergence_integrals;
    std::vector<std::size_t> target_edge_to_unique;
    std::vector<int> target_edge_signs;
    std::vector<double> unique_edge_fluxes;
};

struct MomentMethodOptions {
    int edge_moment_order = 0;
    int harmonic_degree = -1;
    int cell_moment_order = -1;
    int quadrature_points = 8;
    double regularization = 1.0e-12;
    bool exact_constraints = true;
    double edge_weight = 1.0;
    double cell_weight = 1.0;
};

struct MomentReconstruction {
    MomentMethodOptions options;
    int vector_polynomial_degree = 0;
    int harmonic_degree = -1;
    double length_scale = 1.0;
    std::vector<double> coefficients;
};

struct EdgeMomentTransferResult {
    std::vector<DirectedEdgeDof> target_edges;
    std::vector<std::vector<double>> target_moments;
};

struct ConformingEdgeMomentTransferResult {
    std::vector<DirectedEdgeDof> target_edges;
    std::vector<std::vector<double>> target_moments;
    std::vector<moab::EntityHandle> target_cells;
    std::vector<double> target_divergence_integrals;
    std::vector<std::size_t> target_edge_to_unique;
    std::vector<int> target_edge_orientations;
    std::vector<std::vector<double>> unique_edge_moments;
};

/// Signed shoelace area; positive for counter-clockwise point order.
double signed_area(const std::vector<Eigen::Vector2d>& points);
/// Area-weighted polygon centroid in absolute planar coordinates.
Eigen::Vector2d polygon_centroid(const std::vector<Eigen::Vector2d>& points);
/// Return an existing MOAB edge between two vertices or create one if absent.
moab::EntityHandle find_or_create_edge(moab::Core& mb, moab::EntityHandle v0, moab::EntityHandle v1);
/// Extract a MOAB polygon/quad into a centroid-relative LocalPolygon.
LocalPolygon local_polygon(moab::Core& mb, moab::EntityHandle polygon, bool is_spherical = false);
/// Extract a MOAB polygon/quad using explicit geometry options.
LocalPolygon local_polygon(moab::Core& mb, moab::EntityHandle polygon, const GeometryOptions& options);
/// Extract a MOAB polygon/quad into a spherical gnomonic chart.
SphericalPolygon spherical_polygon(moab::Core& mb,
                                   moab::EntityHandle polygon,
                                   const GeometryOptions& options = GeometryOptions());
/// Build ordered spherical edge records from a SphericalPolygon.
std::vector<SphericalEdge> spherical_edges(moab::Core& mb, const SphericalPolygon& polygon);
/// Gnomonic projection from the unit sphere to a local tangent chart.
Eigen::Vector2d project_gnomonic(const Eigen::Vector3d& point, const GnomonicFrame& frame);
/// Inverse gnomonic projection from a tangent chart to the sphere.
Eigen::Vector3d inverse_gnomonic(const Eigen::Vector2d& xi, const GnomonicFrame& frame);
/// Differential of inverse gnomonic projection. Columns are dr/dxi and dr/deta.
Eigen::Matrix<double, 3, 2> gnomonic_jacobian(const Eigen::Vector2d& xi, const GnomonicFrame& frame);
/// Area scale |dr/dxi x dr/deta| for the inverse gnomonic chart.
double gnomonic_area_scale(const Eigen::Vector2d& xi, const GnomonicFrame& frame);
/// Contravariant Piola lift from chart vector components to a surface tangent vector.
Eigen::Vector3d lift_contravariant_piola(const Eigen::Vector2d& chart_vector,
                                         const Eigen::Vector2d& xi,
                                         const GnomonicFrame& frame);
/// Inverse Piola map from a surface tangent vector to chart vector components.
Eigen::Vector2d pullback_contravariant_piola(const Eigen::Vector3d& surface_vector,
                                             const Eigen::Vector2d& xi,
                                             const GnomonicFrame& frame);
/// Build ordered LocalEdge records from a LocalPolygon.
std::vector<LocalEdge> local_edges(moab::Core& mb, const LocalPolygon& polygon);
/// Create a MOAB polygon, using MBQUAD for 4-sided cells and MBPOLYGON otherwise.
moab::EntityHandle create_polygon(moab::Core& mb, const std::vector<Eigen::Vector2d>& points);
/// Convenience wrapper for creating a four-sided polygon.
moab::EntityHandle create_quad(moab::Core& mb, const std::array<Eigen::Vector2d, 4>& points);
/// Merge coincident vertices and higher-dimensional adjacencies on one planar mesh patch.
void merge_polygon_vertices(moab::Core& mb,
                            const std::vector<moab::EntityHandle>& polygons,
                            double merge_tolerance = 1.0e-12);
/// Clip a directed segment against a convex counter-clockwise polygon.
bool clip_segment_to_convex_polygon(const Eigen::Vector2d& segment_a,
                                    const Eigen::Vector2d& segment_b,
                                    const std::vector<Eigen::Vector2d>& polygon,
                                    Eigen::Vector2d& clipped_a,
                                    Eigen::Vector2d& clipped_b,
                                    double tolerance = kTolerance);
/// Write a sparse projection and its directed-edge maps in MatrixMarket/CSV form.
void write_matrix_market(const SparseEdgeProjection& projection,
                         const std::string& matrix_path,
                         const std::string& source_edges_path,
                         const std::string& target_edges_path);

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

    void set_geometry_options(const GeometryOptions& options);
    GeometryOptions geometry_options() const;
    void set_spherical(bool is_spherical);
    bool is_spherical() const;

    void set_source_edge_flux(moab::EntityHandle polygon, std::size_t local_edge_index, double flux);
    double source_edge_flux(moab::EntityHandle polygon, std::size_t local_edge_index, moab::EntityHandle edge) const;
    double target_edge_flux(moab::EntityHandle polygon, std::size_t local_edge_index, moab::EntityHandle edge) const;

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
    /// Area integral of the stored reconstruction on one polygon in ambient Cartesian coordinates.
    Eigen::Vector3d cell_integral(moab::EntityHandle polygon) const;
    /// Area average of the stored reconstruction on one polygon in ambient Cartesian coordinates.
    Eigen::Vector3d cell_average(moab::EntityHandle polygon) const;
    /// Compute target edge fluxes for a target polygon contained in one source cell.
    std::vector<double> transfer_to_target_polygon_edges(moab::EntityHandle source_polygon, moab::EntityHandle target_polygon);
    /// Compute edge-wise source-to-target transfer on nonmatching convex meshes.
    EdgeTransferResult transfer_source_to_target_edges(const std::vector<moab::EntityHandle>& source_polygons,
                                                       const std::vector<moab::EntityHandle>& target_polygons);
    /// Compute direct source-edge to target-cell-average transfer using the harmonic reconstruction.
    CellAverageTransferResult transfer_source_to_target_cell_averages(
        const std::vector<moab::EntityHandle>& source_polygons,
        const std::vector<moab::EntityHandle>& target_polygons,
        CellAverageReductionMode mode = CellAverageReductionMode::Harmonic);
    /// Assemble the linear sparse operator U_t = P U_s for directed edge DOFs.
    SparseEdgeProjection assemble_edge_projection_operator(const std::vector<moab::EntityHandle>& source_polygons,
                                                           const std::vector<moab::EntityHandle>& target_polygons);
    /**
     * Project raw directed target-edge fluxes onto a globally conforming target skeleton.
     *
     * The postprocess solves a constrained least-squares problem on unique
     * geometric target edges: it stays as close as possible to `raw_transfer`
     * in the directed edge-flux norm while enforcing one exact divergence
     * constraint per target cell.
     */
    ConformingEdgeTransferResult project_target_fluxes_to_hdiv_conforming(
        const std::vector<moab::EntityHandle>& source_polygons,
        const std::vector<moab::EntityHandle>& target_polygons,
        const EdgeTransferResult& raw_transfer);

  private:
    moab::Core& mb_;
    GeometryOptions options_;
    moab::Tag tag_source_flux_ = 0;
    moab::Tag tag_target_flux_ = 0;
    moab::Tag tag_coeffs_ = 0;
    std::map<std::pair<moab::EntityHandle, std::size_t>, double> directed_source_flux_;
    std::map<std::pair<moab::EntityHandle, std::size_t>, double> directed_target_flux_;
};

/**
 * Planar higher-order edge-moment reconstruction.
 *
 * The prototype reconstructs one planar vector polynomial per cell from
 * directed edge-normal moments and optional cell vector moments. The edge
 * moment order controls the polynomial degree of the reconstructed field.
 *
 * This is a verified planar research kernel for higher-order edge-to-edge
 * transfer. The `harmonic_degree` option is reserved for future divergence-free
 * enrichment but is not used by the current implementation.
 */
class PlanarMomentInterpolator {
  public:
    explicit PlanarMomentInterpolator(moab::Core& moab_instance);

    void set_geometry_options(const GeometryOptions& options);
    GeometryOptions geometry_options() const;
    void set_spherical(bool is_spherical);
    bool is_spherical() const;

    void set_source_edge_moments(moab::EntityHandle polygon,
                                 std::size_t local_edge_index,
                                 const std::vector<double>& moments);
    std::vector<double> source_edge_moments(moab::EntityHandle polygon,
                                            std::size_t local_edge_index) const;
    void set_source_cell_vector_moments(moab::EntityHandle polygon,
                                        const std::vector<Eigen::Vector2d>& moments);
    std::vector<Eigen::Vector2d> source_cell_vector_moments(moab::EntityHandle polygon) const;

    MomentReconstruction reconstruct_source_polygon(moab::EntityHandle polygon,
                                                    const MomentMethodOptions& options);
    Eigen::Vector2d velocity(const MomentReconstruction& reconstruction,
                             const Eigen::Vector2d& p) const;
    std::vector<double> edge_moments(const MomentReconstruction& reconstruction,
                                     const Eigen::Vector2d& a,
                                     const Eigen::Vector2d& b,
                                     int order) const;
    EdgeMomentTransferResult transfer_source_to_target_edge_moments(
        const std::vector<moab::EntityHandle>& source_polygons,
        const std::vector<moab::EntityHandle>& target_polygons,
        int target_moment_order) const;
    ConformingEdgeMomentTransferResult project_target_edge_moments_to_hdiv_conforming(
        const std::vector<moab::EntityHandle>& source_polygons,
        const std::vector<moab::EntityHandle>& target_polygons,
        const EdgeMomentTransferResult& raw_transfer) const;

  private:
    moab::Core& mb_;
    GeometryOptions options_;
    std::map<std::pair<moab::EntityHandle, std::size_t>, std::vector<double>> directed_source_moments_;
    std::map<moab::EntityHandle, std::vector<Eigen::Vector2d>> source_cell_vector_moments_;
    std::map<moab::EntityHandle, MomentReconstruction> reconstructions_;
};

}  // namespace mimetic

#endif
