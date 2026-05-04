# Code Documentation and Manuscript Traceability

This document is the implementation companion to
`docs/mimetic_voronoi_report.tex`.  The report is intended to read as a
technical manuscript: background, mathematical method, algorithmic analysis,
validation, discussion, and future work.  This file keeps the code-facing
details separate so that implementation choices remain auditable without
dominating the manuscript.

## Source Layout

| Area | Files | Role |
| --- | --- | --- |
| Public API and data types | `include/mimetic/mimetic.hpp` | Geometry options, local polygon data, reconstruction coefficients, sparse projection types, global target-edge conforming projection types, and `MimeticInterpolator` interface. |
| Numerical kernels | `src/mimetic.cpp` | Local geometry construction, harmonic basis evaluation, low-order KKT reconstruction, unified split-basis high-order reconstruction, VEM H(div) projection infrastructure, patch-based moment recovery, clipped edge transfer, sparse operator assembly, global target-edge constrained projection, and MatrixMarket output. |
| Shared spherical test utilities | `tests/spherical_transfer_test_utils.hpp` | Cubed-sphere generators, manufactured spherical fields, edge quadrature, conservative edge-flux assignment, and diagnostic helpers. |
| Planar validation | `tests/patch_test.cpp`, `tests/conservative_intersection_test.cpp`, `tests/voronoi_intersection_test.cpp`, `tests/convergence_validation_test.cpp`, `tests/hdiv_conforming_projection_test.cpp`, `tests/high_order_edge_moment_test.cpp`, `tests/high_order_hdiv_convergence_test.cpp`, `tests/vem_projection_test.cpp`, `tests/patch_recovery_test.cpp` | Constant patch, rectangular overlap, Voronoi n-gon, planar convergence, global target-edge conforming-projection checks, exact higher-order patch tests, `p=1,2,3` high-order convergence studies, VEM vs SplitBasis comparison with trace diagnostics, and patch recovery convergence on quad and Voronoi meshes. |
| Spherical validation | `tests/spherical_geometry_test.cpp`, `tests/spherical_quad_test.cpp`, `tests/spherical_voronoi_test.cpp`, `tests/spherical_scalar_test.cpp`, `tests/spherical_high_order_moment_test.cpp`, `tests/spherical_high_order_hdiv_convergence_test.cpp` | Spherical geometry primitives, structured cubed-sphere transfer, unstructured Voronoi transfer, exact scalar-overlap control, and unified higher-order structured plus Voronoi-patch convergence studies in gnomonic charts. |
| Study scripts | `scripts/convergence_study.sh`, `scripts/plot_convergence.py`, `scripts/run_high_order_hdiv_convergence.sh`, `scripts/plot_high_order_hdiv_convergence.py`, `scripts/run_spherical_high_order_hdiv_convergence.sh`, `scripts/plot_spherical_high_order_hdiv_convergence.py` | Structured low-order spherical convergence sweep, planar high-order `H(div)`-style study, and spherical structured plus Voronoi high-order study figure generation. |

## Manuscript-to-Code Map

| Manuscript concept | Primary code location | Notes |
| --- | --- | --- |
| Geometry mode and tolerances | `GeometryMode`, `GeometryOptions` in `include/mimetic/mimetic.hpp` | Selects planar or spherical gnomonic geometry and stores tolerances. |
| Local planar polygon data | `LocalPolygon` in `include/mimetic/mimetic.hpp`; `local_polygon(...)` in `src/mimetic.cpp` | Stores centroid-relative 2D points, chart area, spherical area, and spherical frame metadata. |
| Spherical chart construction | `SphericalPolygon`, `spherical_polygon(...)`, `make_gnomonic_frame(...)` | Normalizes vertices, constructs source-cell tangent frames, projects great-circle vertices to chart coordinates. |
| Gnomonic map and inverse | `project_gnomonic(...)`, `inverse_gnomonic(...)` | Implements the chart map used for spherical clipping. |
| Piola lift and pullback | `lift_contravariant_piola(...)`, `pullback_contravariant_piola(...)` | Preserves normal fluxes between chart and surface curves. |
| Harmonic basis | `eval_harmonic_basis(...)` and gradient helpers in `src/mimetic.cpp` | Evaluates \(P_k,Q_k\) and gradients used in the local reconstruction. |
| Source-cell reconstruction | `MimeticInterpolator::reconstruct_source_polygon(...)` | Builds local moment matrices, KKT constraints, and reconstruction coefficients from directed edge fluxes; in spherical mode the Gram matrix uses the Hodge/Piola tensor \(J^T J / |J|\). |
| Source reconstruction matrix | `source_reconstruction_matrix(...)` in `src/mimetic.cpp` | Builds the local linear map from source edge fluxes to reconstruction coefficients for sparse projection assembly, using the same option-controlled spherical metric as the direct reconstruction path. |
| Direct edge transfer | `MimeticInterpolator::transfer_source_to_target_edges(...)` | Clips each directed target edge against candidate source cells and evaluates the source reconstruction on clipped segments. |
| Sparse edge projection | `MimeticInterpolator::assemble_edge_projection_operator(...)` | Applies the same clipped-segment functional to local reconstruction matrices and assembles \(U_t = P U_s\). |
| Global target-edge conforming projection | `ConformingEdgeTransferResult`, `project_target_fluxes_to_hdiv_conforming(...)` | Collapses directed target edges to unique geometric edges, assembles exact target-cell divergence constraints, and solves the constrained least-squares correction described in manuscript Algorithm 6. |
| Unified higher-order moment reconstruction | `MomentMethodOptions`, `MomentReconstruction`, `PlanarMomentInterpolator` | Reconstructs a local field from polynomial-divergence, harmonic-gradient, and divergence-free completion modes, then transfers target edge moments with the same clipping machinery used by the low-order remap. |
| VEM H(div) projection | `ReconstructionMode::VemProjection`, `build_vem_decomposed_basis(...)`, `vem_mass_matrix(...)`, `vem_reconstruct_divergence(...)`, `vem_projection_rhs(...)`, `vem_decomposed_to_cartesian(...)` | Decomposes `[P_p]²` into `∇P_{p+1} ⊕ x⊥P_{p-1}` and computes the L²-optimal polynomial projection from VEM DOFs via integration-by-parts identities. Topology-independent, well-conditioned for all p. |
| Patch-based moment recovery | `ReconstructionMode::PatchRecoveryVem`, `patch_recover_moments_impl(...)`, `PlanarMomentInterpolator::recover_moments_from_patch(...)` | Bootstraps high-order edge moments and cell vector moments from single-flux-per-edge data via neighbor-cell least-squares polynomial recovery. Feeds recovered DOFs into the VEM projection pipeline. |
| Trace operator diagnostic | `diagnose_trace_operator(...)`, `TraceOperatorDiagnostic` | Computes singular values and condition number of the edge trace operator `A: [P_p]² → R^{N_e(p+1)}` for any cell, quantifying the ill-conditioning that motivates the VEM path. |
| Matrix output | `write_matrix_market(...)`, `write_edge_map_csv(...)` | Writes the sparse directed edge operator and row/column maps. |

## Directed Edge Convention

The library treats edge fluxes as cell-local directed degrees of freedom.  A
shared mesh edge can appear twice with opposite signs, once for each adjacent
cell orientation.  This convention is required because the transferred quantity
is

```text
integral over target directed edge of u dot target outward normal ds
```

not an orientation-free edge scalar.  Any downstream workflow that collapses
shared edges must maintain an explicit orientation map.

## Reconstruction Path

The reconstruction path implements the level-2 mimetic interpolation of
Perot and Subramanian [16] (see also the mimetic finite difference
frameworks [9,10,15]), extended to higher order via the split H(div) basis
hierarchy:

1. `local_polygon(...)` builds a cell-local coordinate frame.
2. `local_edges(...)` builds ordered directed edges and outward normals.
3. `source_edge_flux(...)` reads the local directed flux vector.
4. `reconstruct_source_polygon(...)` computes the constant divergence and
   harmonic coefficients by solving a constrained local KKT system [30].
5. `velocity(...)`, `line_integral(...)`, and `edge_flux(...)` evaluate the
   reconstructed chart field and its edge integrals.

For spherical geometry, the reconstructed field is a chart vector.  Surface
vectors used in diagnostics are obtained through `lift_contravariant_piola(...)`.

## Transfer Path

The direct source-to-target path extends the conservative remap framework of
Jones [23] and Ullrich et al. [24,25] from scalar cell-average quantities to
H(div) edge-flux fields:

1. Reconstruct every source cell.
2. For every directed target edge, project its endpoints into each source-cell
   chart via the gnomonic map [21,22].
3. Because gnomonic projection maps great circles to straight lines, clip the
   projected target segment against the projected source polygon using the
   Sutherland-Hodgman algorithm [26].
4. Evaluate the source reconstruction on each retained subsegment.
5. Sum all subsegment contributions into the directed target edge flux.

Coincident source-target edges on the sphere require deterministic ownership.
The current code keeps the source cell on the target-cell-interior side of the
directed target edge to avoid double counting.

## Higher-Order Planar Moment Path

The higher-order kernel is intentionally separate from
`MimeticInterpolator`, but it now uses one hierarchy for `p=1,2,3`.  The
DOF structure follows the Raviart-Thomas convention [17]: each edge carries
p+1 Legendre moments of the normal-trace flux, matching the RT_p DOF
count.  Cell vector moments (when provided) play the role of the internal
DOFs in BDM-type spaces [18].

1. `PlanarMomentInterpolator::set_source_edge_moments(...)` stores one vector of
   Legendre edge moments per directed source-cell edge.
2. `set_source_cell_vector_moments(...)` stores optional interior vector
   moments.
3. `reconstruct_source_polygon(...)` builds a local split basis consisting of
   polynomial-divergence modes, harmonic-gradient modes, and divergence-free
   completion modes, assembles edge and cell moment constraints, and solves for
   the local coefficients via a constrained least-squares (KKT) system [30].
4. `transfer_source_to_target_edge_moments(...)` clips every directed target
   edge against all source cells and accumulates the corresponding target edge
   moments.

The implementation is currently a polygonal `H(div)`-style research kernel
rather than a full polygonal VEM or generalized Whitney space.  It is exact for
the current `p=1` affine and `p=2` manufactured polynomial regressions because
those source fields lie in the reconstructed local space and the source moments
are integrated exactly.

Two details matter:

- The Duffy triangle map [29] used for cell moments is implemented in
  `integrate_triangle_duffy(...)`.  A wrong map here will silently corrupt the
  higher-order moment constraints.
- For fully or over-determined local systems the code solves the constraint
  matrix directly with QR.  The KKT minimum-energy solve [30] is only used
  when the local system is under-determined.
- In constrained least-squares mode
  (`MomentMethodOptions::exact_constraints = false`), the zeroth edge moment on
  every source edge is kept as a hard conservation constraint, while higher
  edge moments and optional cell moments are treated as weighted soft rows in a
  constrained least-squares solve.
- The local basis is evaluated in centroid-relative coordinates scaled by
  `local_length_scale(poly)`.  This keeps `p=1,2,3` conditioning under control
  on refined quads and irregular Voronoi polygons.

The planar convergence driver `tests/high_order_hdiv_convergence_test.cpp`
uses the harmonic-compatible lowest-order remap for `p=1` and the split-moment
enrichment for `p=2,3`.  The moment-0 edge flux remains an exact hard
constraint at every order.  On the current Voronoi-to-Voronoi sequence the
observed average relative edge-flux rates are approximately `1.89`, `2.77`,
and `3.34` for `p=1,2,3`.

The spherical convergence driver
`tests/spherical_high_order_hdiv_convergence_test.cpp` applies the same order
split on source-cell gnomonic charts and then projects the resulting target
edge moments onto a globally conforming target skeleton.  It reports both the
zeroth-moment edge error and the all-moment edge error on structured
cubed-sphere transfers and deterministic spherical Voronoi patches.

## Sparse Projection Path

The sparse operator path mirrors direct transfer and assembles the linear
map \(U_t = P U_s\) in the sense of Ullrich and Taylor [25].  For each
source cell, `source_reconstruction_matrix(...)` computes a local matrix
\(B_K\) such that the reconstruction coefficients are linear in the source
edge fluxes.  Each clipped target-edge segment then contributes a row
functional \(L_{\gamma,K} B_K\) to the sparse matrix.

The acceptance criterion is that applying the sparse matrix to the source
edge-flux vector reproduces direct transfer to the conservation tolerance.

## Global Target-Edge Conforming Projection

The global `H(div)` postprocess is deliberately separate from the raw transfer:

1. `transfer_source_to_target_edges(...)` computes the raw directed target-edge
   fluxes.
2. `build_directed_target_edges(...)` re-enumerates the target directed edges
   and records absolute endpoint coordinates.
3. `collapse_target_edges(...)` groups opposite cell-local orientations of the
   same geometric target edge into one unique flux unknown.
4. `target_divergence_rhs(...)` reuses the source-target overlap clipping to
   assemble one exact divergence integral per target cell,
   \(b_t=\sum_s d_s |K_t\cap K_s|\).
5. `project_target_fluxes_to_hdiv_conforming(...)` solves the constrained
   least-squares problem from manuscript Eq. (H(div) projection) through its
   Schur complement and writes the corrected directed target-edge fluxes back to
   the target-flux tag.

This produces a conforming target-edge skeleton field: one unique signed flux
per geometric target edge and exact target-cell divergence constraints.  It does
not yet build a single globally conforming cell-interior vector field on the
target mesh.

### High-order coupled multi-degree projection

`PlanarMomentInterpolator::project_target_edge_moments_to_hdiv_conforming`
generalizes the lowest-order skeleton projection to the full Legendre
moment vector `m = 0, ..., p` per directed target edge.  The block
formulation makes the structure of the conforming subspace explicit:

- Variables: `x in R^{N_unique * (p+1)}`, the orientation-corrected
  Legendre moment per geometric edge per degree.
- Per-directed-edge mapping: `U_i^{(m)} = f_m(O_i) * x_{u(i), m}` where
  `f_m` is the Legendre parity-orientation factor encoded in
  `target_edge_moment_orientation_factor`.
- Objective: edge-mass-weighted L2 deviation from the raw transferred
  moments, with weights `w_{e, m} = L_e * 2 / (2m+1)` (physical edge
  length times Legendre normalization on `[-1, 1]`).
- Constraint: the cell divergence theorem `A_0 * x_0 = b` couples the
  `m = 0` block across cells; for `m >= 1` no analogous cell-level
  constraint exists, so each higher block reduces to the unconstrained
  per-edge L2 minimizer.

For `m = 0` the Schur complement of the divergence constraint is
factored once via complete-orthogonal decomposition and reused for
every degree; the per-degree answer is byte-identical to the previous
degree-decoupled implementation, so all spherical and planar
high-order convergence tests remain unchanged.

Three diagnostic fields are added to `ConformingEdgeMomentTransferResult`:

- `unique_edge_lengths[u]` -- planar chord length, or great-circle arc
  length with the configured `GeometryOptions::radius`, computed once
  per unique geometric edge.
- `trace_jump_per_unique_edge[u][m]` -- a CORRECTNESS check on the raw
  transfer.  By construction of
  `transfer_source_to_target_edge_moments`, both directed views of any
  unique target edge integrate the same source field with opposite
  outward normals, so after the Legendre parity flip the orientation-
  corrected raw values agree to machine roundoff.  This field is
  therefore identically zero (modulo roundoff) on every case; a
  nonzero value indicates a transfer-side bug, not a refinement-faithful
  diagnostic.  Boundary edges (single directed view) report zero by
  definition.
- `source_skeleton_jump_l2[m]` -- the genuinely refining diagnostic.
  For each interior source-mesh edge shared by source cells `a`, `b`,
  the per-degree squared moment of `(v_a . n - v_b . n) L_m(t)` is
  accumulated; `source_skeleton_jump_l2[m] = sqrt(sum)`.  This refines
  as `O(h^{p+1})` for a smooth source field reconstructed at order `p`,
  independently of the source/target meshes chosen.  It is the natural
  asymptotic indicator of source-reconstruction continuity and is the
  quantity exercised by the regression test below.

A regression suite in `tests/coupled_conforming_projection_test.cpp`
verifies post-projection trace continuity to `1e-12`, divergence
balance to `5e-13`, that the per-target-edge raw trace jump is at the
roundoff floor (< `1e-12`), and that the source-skeleton jump rate is
`>= 1.5` between successive levels of a 4 -> 8 -> 16 quad-to-quad
refinement at `p = 2` (asymptotic rate `O(h^3) ~ 3`; conservative
threshold to absorb pre-asymptotic effects).

A future extension layered on the same block scaffold introduces
target cell vector moments as additional variables, coupled to the
edge moments through the VEM-Pi_p integration-by-parts identity; see
`docs/plans/2026-05-02-coupled-conforming-projection.md` for the full
plan.

## Diagnostics and VTK Output

`tests/spherical_quad_test.cpp` writes target-cell diagnostic tags:

- `TARGET_DIV_RECON`: sum of transferred directed edge fluxes around a target
  cell.
- `TARGET_FLUX_ERROR`: maximum directed edge-flux error on that target cell.
- `TARGET_FIELD_DIRECT`: target-cell average obtained directly from overlap
  reduction of source reconstructions.
- `TARGET_FIELD_RECON`: target-cell average obtained by reconstructing the
  transferred target edge fluxes.
- `TARGET_FIELD_EXACT`: exact manufactured target-cell average.
- `TARGET_FIELD_ERROR`: vector difference
  `TARGET_FIELD_RECON - TARGET_FIELD_EXACT`.
- `TARGET_FIELD_DIRECT_ERROR`: vector difference
  `TARGET_FIELD_DIRECT - TARGET_FIELD_EXACT`.
- `TARGET_FIELD_ERROR_NORM`: norm of `TARGET_FIELD_RECON - TARGET_FIELD_EXACT`.
- `TARGET_FIELD_DIRECT_ERROR_NORM`: norm of
  `TARGET_FIELD_DIRECT - TARGET_FIELD_EXACT`.

The edge-flux norms printed by the test are edge-integral norms, not vector
field norms.  Therefore a small `edge_flux_l2_rel` does not imply identical
global bounds for `TARGET_FIELD_RECON` and `TARGET_FIELD_EXACT`, even though
they are now evaluated as the same physical quantity.

## Planar Visualization Figures

The PNG figures generated by `tests/dump_visuals.cpp` and `scripts/visualize.py`
do not show the vector field itself. They plot the scalar cell-divergence
diagnostic

\[
\frac{1}{|K|}\sum_{e\subset\partial K} U_e,
\]

for source and target cells, where `U_e` is the directed edge flux. This is why
the divergence-free manufactured field in Figure 6 has exact source and exact
target panels that are zero to roundoff, while the reconstructed target panel
can still be nonzero on coarse or irregular meshes: that panel reflects local
target-cell flux-closure error, not the value of the exact vector solution.

## Spherical Accuracy Improvements

The following changes improved spherical higher-order convergence rates,
particularly for all-moment errors on cubed-sphere [21,22] and Voronoi
patches:

1. **Metric-weighted Gram matrix for the high-order path.**
   `PlanarMomentInterpolator::reconstruct_source_polygon()` now applies the
   gnomonic Hodge metric `J^T J / |J|` [22] in the `G_raw` inner product when
   `metric_weighted` is enabled, consistent with the low-order reconstruction.

2. **Duffy quadrature for metric-weighted integrals.**  Both
   `source_reconstruction_matrix()` and `reconstruct_source_polygon()` use a
   10-point Gauss-Legendre [27] Duffy [29] integration rule instead of the
   7-point symmetric Dunavant triangle rule [28] when the metric is active.
   The Hodge metric introduces rational (non-polynomial) integrands that
   require higher-order
   quadrature for accurate integration.

3. **Spherical arc-length parametrization for target edge moments.**  The
   Legendre parameter `t` in `transfer_source_to_target_edge_moments()` now
   uses the great-circle arc-angle fraction for spherical edges, matching the
   parametrization already used in `basis_edge_moments()` for source edges.
   The previous Euclidean chart-coordinate parametrization introduced a
   systematic O(h^2) bias in higher moments.

4. **Adaptive harmonic mode stabilization.**  The fixed diagonal penalty
   `1e2 * area` for k>=3 harmonic modes was replaced with a relative penalty
   `10 * V(i,i)`, which adapts to the actual magnitude of each mode rather
   than applying a uniform damping proportional to cell area.

These changes improved spherical all-moment convergence rates (before degree
elevation was also applied):

- cubed-sphere p=2 all: 2.72 → 3.79
- cubed-sphere p=3 all: 2.15 → 2.62
- Voronoi-patch p=2 all: 2.22 → 2.70
- Voronoi-patch p=3 all: 2.04 → 3.32

Conservation residuals and moment-0 rates were unaffected.  The degree
elevation described below provides the final improvement for p=3.

The convergence study now uses clean factor-of-2 refinement with
non-commensurate source and target resolutions to avoid aliasing artifacts.
Previous levels (`{4,6}, {6,8}, {8,10}, {10,12}`) shared resolutions between
consecutive source and target meshes, which introduced rate oscillations.
The current levels are:

- Cubed-sphere: `{4,7}, {8,14}, {16,28}` (h ~ 1/n, doubling n)
- Voronoi: `{16,25}, {64,100}, {256,400}` (h ~ 1/sqrt(N), quadrupling N)

## Pre-Asymptotic Regime on Cubed-Sphere p=3 (Resolved)

Before degree elevation, the cubed-sphere p=3 moment-0 convergence rate
(~1.77 average) was well below the expected O(h^4).  This was a
pre-asymptotic effect caused by the gnomonic chart distortion [22]: the
flat-space polynomial basis could not approximate the rational
Piola-pulled [20] surface field beyond O(h^3) edge-flux accuracy.  The
degree-elevated basis (described below) resolved this, raising the rate
to 4.57.

Evidence for the pre-asymptotic behavior (prior to degree elevation):

- The fine-pair rate (h=1/14 → 1/28) was ~2.55, trending upward toward the
  expected rate.
- The coarse-pair rate (h=1/7 → 1/14) was ~0.99, indicating the error had not
  yet entered the asymptotic regime.
- Voronoi patches, which cover a smaller solid angle and thus have less metric
  distortion, achieved p=3 rates of ~3.46 even without degree elevation.

### Metric-Corrected Basis Orthogonalization

The split moment basis is now orthonormalized against the metric-weighted
Gram matrix `G_raw` [22] rather than the Euclidean identity.  The function
`build_split_moment_basis()` accepts G_raw as input and uses a new helper
`metric_orthonormal_column_basis()` that performs eigendecomposition of the
block Gram `M^T G M` to produce mode vectors satisfying `Q^T G Q = I`.

Cross-block orthogonality is enforced sequentially: harmonic modes are
projected against divergence modes before orthonormalization, and bubble
modes are projected against both divergence and harmonic blocks.  The
per-row polynomial scaling (by `local_length_scale`) is applied before
orthogonalization so mode vectors and G_raw are in the same coordinate
system.

This change ensures the KKT and constrained least-squares solves operate
with the correct surface inner product.  However, experiments confirmed
that the spherical p=3 convergence rates are unchanged: the O(h^2) error
floor is caused by the **basis span** (flat-space polynomials), not by the
inner product used within that span.  The chart-coordinate polynomial
basis can only approximate the exact surface field (which is a rational
function of chart coordinates) to polynomial accuracy, and this
approximation has an irreducible O(h^{degree+1}) component that for p>=3
is dominated by the O(h^3) rational-to-polynomial approximation error on
cells subtending ~10-15 degrees.

### Root Cause: Basis Span Limitation

The exact surface vector field, when expressed in gnomonic chart
coordinates via the contravariant Piola pullback [20], is a rational
function of (x,y) because the Piola mapping involves `1/|ray|^3` (the
gnomonic Jacobian determinant [22]).  Polynomial vector fields in the chart
coordinates can approximate this rational field, but the approximation
error is controlled by the metric variation across each cell, which is
O(h^2) in the chart coordinates.

For an edge flux integral of this approximation error, the total error
scales as O(h^2) × O(h) = O(h^3).  This is:
- negligible for p=1 (asymptotic O(h^2) dominates)
- comparable for p=2 (O(h^3) matches asymptotic)
- dominant for p=3 (O(h^3) beats asymptotic O(h^4))

The Voronoi patches confirm this: they subtend a smaller solid angle
(~0.55 rad vs ~1.57 rad for a cubed-sphere face), reducing the metric
variation constant by ~(0.55/1.57)^2 ≈ 0.12, which pushes the crossover
to finer meshes and allows p=3 rates of ~3.46 to be observed.

### Degree-Elevated Basis for Spherical p >= 3

The basis span limitation was resolved by **degree elevation**: for p >= 3
in spherical mode, the vector polynomial degree is raised from p to p+2.
A degree-(p+2) polynomial in the chart coordinates (x,y) spans all
products `|ξ|^2 × (degree-p monomial)`, which captures the leading
rational correction from the Piola mapping [1,2,4].

The implementation is a conditional elevation in
`reconstruct_source_polygon()`:

```cpp
const int degree_elevation = (use_surface_metric && options.edge_moment_order >= 3) ? 2 : 0;
const int vector_degree = options.edge_moment_order + degree_elevation;
```

The edge moment constraints remain at order p (from the source data).
The system becomes under-determined (B > C), and the extra degrees of
freedom are resolved by the minimum-energy solve [30] with the metric-weighted
Gram matrix and Tikhonov regularization.  The constrained least-squares
solver (KKT with hard zeroth-moment and soft higher-moment constraints)
handles this naturally.

For p <= 2, no elevation is applied because the O(h^{p+1}) asymptotic
term is comparable to or larger than the O(h^3) Piola crime.

Results after degree elevation:

| Domain | p | Before | After |
|--------|---|--------|-------|
| Cubed-sphere | 1 | 1.75 | 1.75 |
| Cubed-sphere | 2 | 3.08 | 3.08 |
| Cubed-sphere | 3 | **1.77** | **4.57** |
| Voronoi-patch | 1 | 2.05 | 2.05 |
| Voronoi-patch | 2 | 2.52 | 2.52 |
| Voronoi-patch | 3 | 3.46 | **3.62** |

The cubed-sphere p=3 fine-pair rate is 5.52, indicating the degree-elevated
basis captures the Piola correction well enough that the actual convergence
approaches O(h^5).  Conservation residuals remain at machine precision.

## Current Numerical Caveats

The current implementation passes the conservation and direct-vs-sparse tests.
The main remaining numerical caveats are:

1. The global correction is a conforming target-edge skeleton solve.  It does
   not yet build one globally conforming target-cell interior field.
2. The spherical higher-order study currently covers structured cubed-sphere
   transfers and deterministic Voronoi chart patches.  It does not yet cover
   larger mixed-topology or production spherical meshes.

## Build and Validation Commands

```bash
cmake -S . -B build
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Structured spherical convergence sweep:

```bash
bash scripts/convergence_study.sh build > /tmp/mimetic_convergence.csv
conda run -n climate-vis python scripts/plot_convergence.py /tmp/mimetic_convergence.csv docs/figures/spherical_convergence.png
```

The convergence script currently invokes the VTK-writing driver path for each
case.  A future CLI flag should allow convergence sweeps without writing VTK
files.

Planar and spherical high-order studies:

```bash
./build/high_order_hdiv_convergence_test docs/high_order_hdiv_convergence.csv
conda run -n climate-vis python scripts/plot_high_order_hdiv_convergence.py \
  docs/high_order_hdiv_convergence.csv \
  docs/figures/high_order_hdiv_convergence.png

./build/spherical_high_order_hdiv_convergence_test docs/spherical_high_order_hdiv_convergence.csv
conda run -n climate-vis python scripts/plot_spherical_high_order_hdiv_convergence.py \
  docs/spherical_high_order_hdiv_convergence.csv \
  docs/figures/spherical_high_order_hdiv_convergence.png
```

## VEM H(div) Projection

The sections above describe the `SplitBasis` reconstruction path: a direct
least-squares or KKT fit of `[P_p]²` coefficients from edge moment
constraints.  That path is well-conditioned for p ≤ 1, but at p ≥ 2 it
suffers from Piola-frame ill-conditioning on non-affine or irregular cells
(κ(A) ~ 10⁴–10⁷; see "Current Numerical Caveats" and the condition number
tables below).  The Virtual Element Method (VEM) projection path, selected
via `ReconstructionMode::VemProjection`, provides a topology-independent
alternative that is well-conditioned for all p and all polygon shapes.

This section provides a self-contained mathematical description of the VEM
projection, its stability properties, convergence behavior, and the
assumptions under which it operates.

### Mathematical Framework

#### Polynomial space and decomposition

The local reconstruction space is `[P_p(K)]²`, the space of two-component
vector polynomials of total degree at most p on a cell K.  Its dimension is

    dim([P_p]²) = (p+1)(p+2).

This space admits the direct-sum decomposition

    [P_p]² = ∇P_{p+1}(K) ⊕ x⊥P_{p-1}(K)

where `x⊥ = (-y, x)` is the 90° rotation operator.  The two subspaces are:

- **Gradient modes** `∇P_{p+1}`: curl-free vector polynomials.  For each
  scalar potential `φ(x,y)` of total degree 1 ≤ |α| ≤ p+1 (excluding the
  constant), the gradient `∇φ = (∂φ/∂x, ∂φ/∂y)` is a degree-p vector field.
  Dimension: `dim(P_{p+1}) - 1 = (p+2)(p+3)/2 - 1`.

- **Rotational modes** `x⊥P_{p-1}`: divergence-free vector polynomials
  `(-y·m, x·m)` for each scalar `m(x,y) ∈ P_{p-1}`.  These have zero
  divergence identically: `div(x⊥m) = 0`.
  Dimension: `dim(P_{p-1}) = p(p+1)/2`.

The dimension count checks out:
`(p+2)(p+3)/2 - 1 + p(p+1)/2 = (p+1)(p+2)`.  This decomposition is
classical in finite element exterior calculus [19,20] (specifically, it is
the 2D Koszul complex splitting of polynomial 1-forms, [19] §4, Lemma 4.2)
and is used explicitly in the H(div)-VEM construction of Beirão da Veiga,
Brezzi, Marini, and Russo [6].

#### VEM degrees of freedom

The local VEM space `V_h(K)` is an infinite-dimensional function space
containing `[P_p]²` as a subspace.  Functions in `V_h` are not known
pointwise inside K, but are determined by two types of DOFs:

**Type 1 — Edge normal-trace moments.** For each edge e of K:

    μₘᵉ = ∫_e (v·n) Lₘ(t) ds,    m = 0, 1, ..., p

where `Lₘ` is the m-th Legendre polynomial on [-1,1] and t parametrizes the
edge.  This gives `(p+1) × N_e` scalar DOFs, where N_e is the number of
edges of K.

**Type 2 — Cell vector moments.** For each scalar monomial `x^a y^b` of
total degree `|α| = a + b ≤ p` and each coordinate direction `ê_i`:

    c_{ab}^i = ∫_K (v · ê_i) x^a y^b dA

This gives `2 × dim(P_p) = (p+1)(p+2)` scalar DOFs.

The cell moments are **primitive** — they cannot be derived from edge data
alone.  This is the fundamental difference from the `SplitBasis` path, which
attempts to determine all polynomial coefficients from edge constraints
only.  The rotational modes `x⊥P_{p-1}` have zero normal trace on every
edge (`(x⊥m)·n = 0` on ∂K does NOT hold in general, but the rotational
modes are interior-invisible to the boundary trace operator in a degenerate
sense that grows with p), so they require cell moment data for
observability.

#### Elliptic projection

The VEM elliptic projection `Π_p : V_h(K) → [P_p(K)]²` is the L²-optimal
polynomial approximation defined by

    ∫_K (Π_p v - v) · q dA = 0    for all q ∈ [P_p]²

which, in the decomposed basis `{φ_i}` of `[P_p]²`, yields the linear
system

    M c = b

where `M_{ij} = ∫_K φ_i · φ_j dA` (mass matrix) and
`b_i = ∫_K v · φ_i dA` (right-hand side).

The central VEM insight ([6] Prop. 4.1, [7] §3, [8] §2–3) is that both M
and b are **exactly computable** from the DOFs:

- **M** depends only on the cell geometry (monomial integrals), not on v.
  It is assembled from the precomputed integral table
  `∫_K x^a y^b dA` via the fan-triangulated Duffy [29] quadrature rule.

- **b** is computed via integration by parts, exploiting the decomposition:

  For gradient modes `φ_i = ∇ψ_i`:

      ∫_K v · ∇ψ_i dA = ∫_{∂K} (v·n) ψ_i ds − ∫_K (div v) ψ_i dA

  The boundary term uses Type 1 DOFs (edge moments).  The volume term
  requires `div(v)`, which is itself recovered from DOFs: applying the
  divergence theorem to `div(v) · q` for test functions `q ∈ P_{p-1}`
  produces

      ∫_K div(v) · q dA = ∫_{∂K} (v·n) q ds − ∫_K v · ∇q dA

  where the boundary integral uses edge moments and the volume integral
  uses cell vector moments (Type 2 DOFs).  This is the key identity from
  [6] §4, eq. (4.5).

  For rotational modes `φ_j = x⊥m_j`:

      ∫_K v · (x⊥m_j) dA = Type 2 cell vector moment DOFs directly.

  These are exactly the DOFs that make the rotational subspace observable.

The resulting system `M c = b` is symmetric positive definite and solved via
LDLT factorization with an SVD fallback: if the LDLT factorization reports
failure or a non-positive-definite matrix (e.g. due to a degenerate sliver
cell), the solve falls back to a Jacobi SVD pseudoinverse for robustness.
The same LDLT-with-SVD-fallback strategy is applied to the divergence
reconstruction Gram matrix `G d = g`.  The decomposed coefficients are then
converted back to the Cartesian `[P_p]²` monomial basis for field evaluation
and transfer.

### Stability and Conditioning

The VEM projection avoids the Piola ill-conditioning that plagues the
`SplitBasis` path.  The key difference is structural:

**SplitBasis path:** Inverts the edge trace operator
`A: [P_p]² → R^{N_e(p+1)}`, whose condition number depends on the cell
geometry.  On distorted quadrilaterals or irregular polygons:

| p | Trace κ_max (Voronoi) | Trace κ_max (quad) |
|---|-----------------------|--------------------|
| 1 | 2.10×10¹              | 6.00               |
| 2 | 2.41×10³              | ∞ (singular)       |
| 3 | 9.07×10⁷              | 1.51×10¹⁹          |

Arnold, Boffi, and Falk [1,2] (see also the surveys [3,5]) proved that
this ill-conditioning is *intrinsic* to the `[P_p]²` basis: the Piola
mapping from a reference element produces a trace operator whose condition
number grows as the cell departs from an affine (parallelogram) shape.
The condition `Ŝ ⊇ Q_r(K̂)` is necessary and sufficient for a
well-conditioned trace on general quadrilaterals, but `[P_p]²` does not
contain `Q_p` for p ≥ 2.

**VEM path:** The mass matrix `M = ∫_K φ_i · φ_j dA` in the decomposed
basis is always well-conditioned when the cell is scaled to unit diameter
(κ(M) ~ O(1) on centroid-centered coordinates).  The RHS is assembled via
integration-by-parts identities that use edge moments and cell moments as
*data*, not as constraints on an ill-conditioned system.  The decomposition
`∇P_{p+1} ⊕ x⊥P_{p-1}` naturally separates modes by their observability:
gradient modes from boundary data, rotational modes from cell moments.  No
single solve mixes well- and poorly-conditioned directions.

This is a direct consequence of the VEM design principle: the projection
operator is defined to be exactly computable from DOFs, by construction
([8] §2), regardless of cell topology.

### Assumptions

The VEM projection path operates under the following assumptions:

1. **Cell geometry.**  The cell K is a simple (non-self-intersecting) polygon
   with at least 3 edges.  No regularity or convexity assumption is required
   — the projection is well-defined on arbitrary simple polygons including
   concave shapes and cells with very short edges.  This is the key
   advantage of VEM over reference-element-based finite elements.

2. **Coordinate frame.**  All DOFs and basis functions are expressed in a
   centroid-relative local frame.  The cell centroid is at the origin, and
   no coordinate scaling is applied (unlike the `SplitBasis` path, which
   uses `local_length_scale`).  This means the monomial integrals
   `∫_K x^a y^b dA` have natural magnitude O(h^{a+b+2}) for a cell of
   diameter h, and the mass matrix inherits this scaling.  The LDLT solver
   handles the resulting dynamic range without difficulty for p ≤ 3.

3. **DOF completeness.**  The caller must provide:
   - Edge moments of order exactly p on every edge of K.
   - Cell vector moments up to total monomial degree p.

   If any DOFs are missing or inconsistent, the projection will be
   inaccurate or produce large residuals.  The code does not currently
   detect incomplete DOF sets beyond a dimension check on edge moments.

4. **Polynomial approximability.**  The exact field v must be well-
   approximated by `[P_p]²` on K for the convergence estimates to hold.
   In spherical gnomonic charts [21,22], the Piola pullback [20] introduces
   rational (non-polynomial) components at O(h²), which limits the achievable
   accuracy to O(h³) on cells subtending large solid angles (see
   "Root Cause: Basis Span Limitation" above).  For the VEM path in
   planar geometry, no such limitation exists.

5. **Edge moment orientation.**  Edge moments are directed (cell-local
   outward normal).  Moment signs must be consistent with the cell's
   counter-clockwise vertex ordering.  Reversing an edge normal flips the
   sign of all odd-degree Legendre moments.

### Convergence: VEM vs SplitBasis

The following table compares the VEM and SplitBasis convergence rates on
planar quad and Voronoi meshes, using exact polynomial DOFs (edge moments
and cell moments computed from a known analytical field).  The test field
is `v(x,y) = (sin(πx)cos(πy), -cos(πx)sin(πy))` on `[0,1]²`.  The L²
edge-flux error is measured as a relative RMS over all directed target
edges.  Rates are computed from Richardson extrapolation on a geometric
mesh refinement sequence (N = 4², 8², 16² for quads; N = 20, 80, 320
Voronoi cells).

**Quad meshes:**

| p | VEM L² errors | VEM rate | SplitBasis L² errors | SplitBasis rate |
|---|---------------|----------|----------------------|-----------------|
| 1 | 5.35e-2 → 1.35e-2 → 3.38e-3 | **2.00** | 5.40e-2 → 1.36e-2 → 3.41e-3 | **2.00** |
| 2 | 2.40e-3 → 3.01e-4 → 3.77e-5 | **3.00** | 2.40e-3 → 3.01e-4 → 3.77e-5 | **3.00** |
| 3 | 6.28e-5 → 3.93e-6 → 2.46e-7 | **4.00** | 6.28e-5 → 3.93e-6 → 2.51e-7 | **3.97** |

On structured quad meshes, VEM and SplitBasis produce nearly identical
results because quads are affine (parallelogram) cells where the Piola trace
operator is well-conditioned at all orders.  The VEM path achieves exactly
O(h^{p+1}) convergence, confirming correctness of the projection.

**Voronoi meshes:**

| p | VEM L² errors | VEM rate | SplitBasis L² errors | SplitBasis rate |
|---|---------------|----------|----------------------|-----------------|
| 1 | 5.52e-2 → 1.72e-2 → 4.87e-3 | **1.82** | 1.42e-1 → 2.57e-2 → 6.69e-1 | **diverges** |
| 2 | 3.72e-3 → 5.54e-4 → 9.38e-5 | **2.56** | 3.77e-3 → 7.60e-4 → 1.82e-4 | **2.06** |
| 3 | 2.92e-4 → 2.39e-5 → 2.78e-6 | **3.11** | 3.93e-4 → 5.07e-5 → 6.83e-6 | **2.89** |

The advantage of VEM becomes decisive on Voronoi meshes:

- **p=1:** SplitBasis *diverges* on irregular Voronoi cells (rate −4.70 in
  the finest pair), while VEM maintains stable O(h^{1.8}) convergence.  The
  SplitBasis failure is caused by a few highly irregular cells where the
  trace operator has condition number > 10⁴, amplifying small moment errors
  into catastrophic coefficient errors.

- **p=2:** VEM achieves rate 2.56 vs SplitBasis 2.06 — a factor of ~2×
  smaller error at the finest level.  The SplitBasis rate is eroded by the
  growing trace operator conditioning (κ_max ~ 2.4×10³ on Voronoi at p=2).

- **p=3:** VEM achieves rate 3.11 vs SplitBasis 2.89.  The gap is smaller
  here because the test meshes are relatively coarse and the SplitBasis
  conditioning problems manifest primarily at finer resolution.

These results confirm the theoretical prediction: VEM is topology-
independent and resolution-independent, while SplitBasis degrades on cells
that are far from affine.

### Single-Cell Exact Recovery

The VEM projection exactly recovers polynomials that lie within the
reconstruction space `[P_p]²`.  On a single irregular pentagon with unit-
scale coordinates, the following exact recovery errors were measured:

| Test | p | Max pointwise error |
|------|---|---------------------|
| Linear field on pentagon | 1 | 1.73×10⁻¹⁵ |
| Linear field on pentagon | 2 | 8.40×10⁻¹⁵ |
| Quadratic field on pentagon | 2 | 1.26×10⁻¹⁵ |

These confirm that the VEM projection, mass matrix assembly, and coordinate
conversion are implemented correctly to machine precision.

### Trace Operator Diagnostics

The `diagnose_trace_operator()` function computes the condition number of
the edge trace operator `A: [P_p]² → R^{N_e(p+1)}` for any cell.  This
quantifies the fundamental conditioning challenge that motivates the VEM
path.  On a 20-cell Voronoi mesh:

| p | κ_min | κ_avg | κ_max |
|---|-------|-------|-------|
| 1 | 6.20  | 1.12×10¹ | 2.10×10¹ |
| 2 | 1.36×10² | 5.06×10² | 2.41×10³ |
| 3 | 1.86×10³ | 8.29×10⁶ | 9.07×10⁷ |

And on a 4×4 structured quad mesh (where quads are exact parallelograms):

| p | κ_max |
|---|-------|
| 1 | 6.00 |
| 2 | ∞ (exactly determined, singular) |
| 3 | 1.51×10¹⁹ |

The p=2 quad case has κ = ∞ because the 12 DOFs exactly match
dim([P_2]²) = 12, leaving no overdetermination for stability.  The p=3
quad case has 16 DOFs < dim([P_3]²) = 20, producing 4 unobservable
interior modes.  These are precisely the regimes where VEM is essential.


## Patch-Based Moment Recovery

The VEM projection requires high-order edge moments and cell vector
moments as input (§VEM H(div) Projection, "DOF completeness").  In many
production workflows — particularly in atmosphere and ocean remap codes —
only a **single scalar flux value per edge** is available.  This is the
zeroth Legendre moment `μ₀ᵉ = ∫_e v·n ds`, which is the natural
conserved quantity in a finite-volume discretization.

The `ReconstructionMode::PatchRecoveryVem` mode addresses this data gap by
recovering the missing high-order DOFs from neighbor-cell flux data via
local least-squares polynomial fitting — a strategy rooted in the
superconvergent patch recovery (SPR) framework of Zienkiewicz and
Zhu [13,14] — then feeding the recovered DOFs into the VEM projection
pipeline.

### Mathematical Formulation

Given a target cell K with face-neighbor cells N₁, ..., Nₖ, define the
**patch** P = K ∪ N₁ ∪ ... ∪ Nₖ.  The patch contains all edges of all
cells in P, each with a known single flux value.  The goal is to find a
vector polynomial `v_h ∈ [P_q]²` (with fitting degree q = p, the desired
reconstruction order) that best fits the patch flux data in a least-squares
sense.

#### Fitting problem

Let `{e_i}_{i=1}^{N_patch}` be the edges of the patch with known fluxes
`{F_i}`.  We seek coefficients `c = (c_1, ..., c_B)` of the polynomial

    v_h(x) = Σ_j c_j φ_j(x)

where `{φ_j}` is the `[P_q]²` monomial basis with `B = (q+1)(q+2)` terms,
that minimize

    ‖Ac - f‖² → min

with the LS matrix

    A_{ij} = ∫_{e_i} (φ_j · n_i) ds

and the right-hand side `f_i = F_i`.  The integrals are evaluated by
Gauss-Legendre quadrature [27] with `max(8, 2q+2)` points per edge.

#### Coordinate scaling

All coordinates are expressed relative to the target cell's centroid and
scaled by the patch diameter `D = max_{v ∈ patch} |v - centroid|`.  In
scaled coordinates `x_s = x/D`, the monomials `x_s^a y_s^b` have magnitude
O(1), which keeps the LS matrix well-conditioned ([11] §3, [12]).  The
SVD solve is applied to the scaled system, and the coefficients are
un-scaled afterwards: `c_unscaled = c_scaled / D^{a+b}`.

#### SVD solver

The system is solved via the Jacobi SVD with thin U and V factors.  This
is numerically stable even when the system is mildly rank-deficient (which
can occur at p=3 when the patch has few edges) and provides automatic
regularization through the SVD truncation threshold.  A degenerate-system
guard rejects fits where the largest singular value is below `kTolerance`
(fully rank-deficient patch), in which case the recovery returns no
moments for that cell and the VEM path falls through to the LS/KKT
reconstruction.

### Recovery Pipeline

For each target cell K, `patch_recover_moments_impl()` executes:

1. **Collect patch geometry.**  Gather all edges from K and its face-
   neighbors.  Re-express neighbor-cell edge coordinates in K's local frame
   using the 3D centroid offset projected onto K's tangent plane basis
   vectors (e_x, e_y).

2. **Build and solve LS system.**  Assemble the `N_patch × B` matrix A
   and the flux vector f in scaled coordinates.  SVD-solve for the fitted
   polynomial coefficients.

3. **Evaluate edge moments.**  For each edge of K, evaluate the
   high-order edge moments

       μₘᵉ = ∫_e (v_h · n) Lₘ(t) ds,    m = 0, 1, ..., p

   via Gauss-Legendre [27] quadrature with `max(10, 2p+4)` points.

4. **Preserve conservation.**  Overwrite `μ₀ᵉ` with the exact source flux
   from the original data.  Only the higher-order moments (m ≥ 1) come
   from the LS recovery.

5. **Evaluate cell vector moments.**  Compute

       c_{ab}^i = ∫_K (v_h · ê_i) x^a y^b dA

   for monomials up to total degree p, using fan-triangulated Duffy [29]
   quadrature over K.

6. **Store recovered DOFs.**  The edge moments and cell moments are
   written into `PlanarMomentInterpolator`'s internal storage, ready for
   `reconstruct_source_polygon()` with the VEM projection path.

### DOF Counting and System Solvability

The LS system has `N_patch` rows (one per patch edge) and `B = (q+1)(q+2)`
columns.  System solvability depends on p and the mesh topology:

| p | B = dim([P_p]²) | Typical N_patch (Voronoi) | Typical N_patch (quad) | Overdetermination |
|---|-----------------|---------------------------|------------------------|-------------------|
| 1 | 6               | 15–20                     | 12–16                  | 2.5–3.3×          |
| 2 | 12              | 15–20                     | 12–16                  | 1.0–1.7×          |
| 3 | 20              | 15–20                     | 12–16                  | **< 1× (under)**  |

At p=1 the system is well-overdetermined on any reasonable mesh.  At p=2 it
is mildly overdetermined — enough for a stable LS fit on most cells, but
with reduced accuracy on cells where the patch geometry is nearly symmetric
(producing near-null modes in A).  At p=3 the system is typically
underdetermined, and the SVD provides a minimum-norm solution that fills in
the unobservable modes with zeros — this does not produce accurate high-
order moments.

For robust p ≥ 3 reconstruction, genuine high-order edge moments and cell
vector moments should be provided directly (via analytical evaluation or
upstream high-order discretization), using `ReconstructionMode::VemProjection`.

### Graceful Degradation

When `PatchRecoveryVem` mode fails to produce full-order edge moments for a
cell (e.g. boundary cells with insufficient face-neighbors, or patch systems
rejected by the SVD degenerate-system guard), the VEM path falls through to
the LS/KKT reconstruction using whatever edge moments are available.  This
ensures the pipeline does not abort on problematic cells — those cells receive
a lower-order reconstruction while interior cells with sufficient neighbors
get the full VEM projection.

Similarly, if cell vector moments are unavailable or undersized, the VEM
projection pads with zeros and proceeds.  This degrades only the rotational
(divergence-free) component of the projection; the gradient modes remain fully
determined by edge boundary data.

### Convergence Results

The full pipeline — single flux per edge → patch recovery → VEM projection
→ edge-to-edge transfer — was tested on the same mesh sequence and field
as the VEM-only convergence study above.

**Single flux → PatchRecoveryVem convergence:**

| Domain  | p | L² errors | Rate |
|---------|---|-----------|------|
| Quad    | 1 | 5.79e-2 → 1.45e-2 → 3.54e-3 | **2.03** |
| Quad    | 2 | 1.50e-2 → 1.75e-3 → 6.32e-4 | **1.47** |
| Voronoi | 1 | 5.57e-2 → 1.76e-2 → 5.08e-3 | **1.80** |
| Voronoi | 2 | 4.89e-2 → 1.77e-2 → 7.63e-3 | **1.21** |

**Comparison with exact-DOF VEM (same meshes):**

| Domain  | p | Exact-DOF VEM rate | PatchRecoveryVem rate | Rate loss |
|---------|---|--------------------|-----------------------|-----------|
| Quad    | 1 | 2.00               | 2.03                  | none      |
| Quad    | 2 | 3.00               | 1.47                  | 1.53      |
| Voronoi | 1 | 1.82               | 1.80                  | 0.02      |
| Voronoi | 2 | 2.56               | 1.21                  | 1.35      |

At p=1 the patch recovery introduces essentially no degradation: the
single-flux data is sufficient to determine the leading polynomial modes
(dim([P_1]²) = 6 unknowns from ~15 flux constraints).  The rate matches
the exact-DOF VEM within mesh-dependent variations.

At p=2 there is a significant rate loss of ~1.3–1.5 orders.  The initial
convergence (coarsest pair) shows rates of 3.1 on quads and 1.47 on
Voronoi, but the rate decays at finer meshes.  This is consistent with the
information-theoretic analysis below.

### Limitations and Information-Theoretic Bounds

The convergence degradation at p ≥ 2 is not a bug in the implementation —
it is a fundamental consequence of the data available.  Three independent
lines of analysis confirm this:

**1. k-Exact reconstruction theory ([11], [12]).**  A k-exact least-squares
reconstruction from integral data achieves at most O(h^{k+1}) accuracy for
the leading polynomial modes.  However, this accuracy applies to the
*cell-average* or *edge-integral* of the polynomial, not to its higher
Legendre moments.  The m-th Legendre moment of the recovered polynomial
has accuracy O(h^{k+1-m}) because each additional moment extracts finer-
scale information from the polynomial that is less well-constrained by
integral data.  For p=2 with q=2 (fitting degree = reconstruction order):

    μ₀: O(h³) — well-constrained (overwritten with exact data)
    μ₁: O(h²) — first correction term; moderately constrained
    μ₂: O(h¹) — second correction; poorly constrained by integrals

The effective convergence rate of the transfer is limited by the worst
moment accuracy that enters the VEM projection.

**2. Piola degradation (Arnold-Boffi-Falk [1,2]).**  Even with exact
high-order DOFs, the `[P_p]²` basis loses optimal convergence on non-affine
cells because it does not contain the full tensor-product space `Q_p` needed
for a well-conditioned Piola trace map.  The VEM projection avoids this
ill-conditioning for the *projection step*, but it cannot compensate for
*inaccurate input DOFs*.  When the DOFs come from a patch recovery that
itself uses `[P_p]²` as the fitting space, the Piola limitation re-enters
through the recovery step.

**3. Dimension counting.**  The single-flux functional `F_e = ∫_e v·n ds`
is a single scalar per edge — it projects the full `[P_p]²` polynomial
onto a one-dimensional subspace of the edge's normal-trace space.  The
`p` higher Legendre moments `μ₁, ..., μ_p` capture the intra-edge
polynomial variation that is invisible to the integral.  No amount of
neighbor-cell information can fully recover this information because the
field is reconstructed from the same type of data (single fluxes) on
all edges.

**Practical implication:**  For production remap at p ≤ 1, the
`PatchRecoveryVem` pipeline delivers full-order convergence with single-
flux-per-edge data.  For p ≥ 2, it provides a useful accuracy improvement
over the raw SplitBasis path (which diverges on irregular cells), but the
convergence rate will be below O(h^{p+1}).  To achieve full high-order
convergence at p ≥ 2, the source discretization must provide genuine
high-order edge moments and cell moments.

### Conservation Properties

The patch recovery preserves the zeroth edge moment (net flux) exactly on
every edge of the target cell by overwriting `μ₀ᵉ` with the original
source flux after the LS recovery.  This means:

- **Local conservation** is preserved: `Σ_e μ₀ᵉ = Σ_e F_e` (sum of
  directed fluxes around K), which equals the cell divergence integral.

- **Global conservation** is preserved through the downstream conforming
  projection (§Global Target-Edge Conforming Projection), which enforces
  exact divergence constraints.

The higher-order moments (m ≥ 1) are *not* conservative observables — they
are model-dependent extrapolations from the LS fit.  This is inherent to
any recovery procedure that bootstraps high-order information from low-
order data.


## Combined Pipeline: PatchRecoveryVem Workflow

The `PatchRecoveryVem` mode implements the complete edge-to-edge remap
workflow for the common case of single-flux-per-edge source data:

```text
Source mesh                  Target mesh
   │                            │
   │  single flux/edge          │
   ▼                            │
┌──────────────────┐            │
│ Patch recovery   │            │
│ (per source cell)│            │
│                  │            │
│ 1. Collect       │            │
│    neighbor      │            │
│    fluxes        │            │
│                  │            │
│ 2. LS fit        │            │
│    v_h ∈ [P_p]²  │            │
│                  │            │
│ 3. Evaluate      │            │
│    edge moments  │            │
│    + cell moments│            │
└────────┬─────────┘            │
         │                      │
         │  high-order DOFs     │
         ▼                      │
┌──────────────────┐            │
│ VEM projection   │            │
│ (per source cell)│            │
│                  │            │
│ Π_p : V_h → [P_p]²          │
│                  │            │
│ Well-conditioned │            │
│ for all p, all   │            │
│ topologies       │            │
└────────┬─────────┘            │
         │                      │
         │  polynomial coeffs   │
         ▼                      │
┌──────────────────┐            │
│ Edge transfer    │◄───────────┘
│                  │  target edges
│ Clip target      │  clipped against
│ edges against    │  source cells
│ source cells;    │
│ evaluate v_h     │
│ on segments      │
└────────┬─────────┘
         │
         │  raw target edge moments
         ▼
┌──────────────────┐
│ Conforming       │
│ projection       │
│                  │
│ LS + divergence  │
│ constraints      │
└────────┬─────────┘
         │
         ▼
   Target edge moments
   (H(div)-conforming)
```

### API Usage

```cpp
PlanarMomentInterpolator interp(mb);
interp.set_geometry_options(geo_opts);

// Step 1: Load single-flux data
for (auto cell : source_cells) {
    for (int ei = 0; ei < n_edges(cell); ++ei) {
        interp.set_source_edge_moments(cell, ei, {flux_value});
    }
}

// Step 2: Patch recovery (per cell)
for (auto cell : source_cells) {
    auto neighbors = get_face_neighbors(cell);
    interp.recover_moments_from_patch(cell, p, neighbors);
}

// Step 3: VEM reconstruction + transfer
MomentMethodOptions opts;
opts.edge_moment_order = p;
opts.reconstruction_mode = ReconstructionMode::PatchRecoveryVem;

for (auto cell : source_cells) {
    interp.reconstruct_source_polygon(cell, opts);
}

auto result = interp.transfer_source_to_target_edge_moments(
    source_cells, target_cells, p);
```

### Regimes of Validity

The following table summarizes the recommended reconstruction mode for
different data availability and accuracy requirements:

| Source data | Target accuracy | Recommended mode | Expected rate |
|-------------|-----------------|------------------|---------------|
| Single flux/edge | O(h²) | `PatchRecoveryVem` (p=1) | 1.8–2.0 |
| Single flux/edge | O(h^{1.5}) | `PatchRecoveryVem` (p=2) | 1.2–1.5 |
| Single flux/edge | ≥ O(h²) | `PatchRecoveryVem` (p=1) | 1.8–2.0 |
| Full moments (p) + cell moments | O(h^{p+1}) | `VemProjection` | 2.0–4.0 |
| Full moments (p), no cell moments | O(h^{p+1}) | `SplitBasis` (quads only) | 2.0–4.0 |
| Full moments (p), no cell moments | any | `VemProjection` with zero cell moments | reduced |

**When to use `PatchRecoveryVem`:**

- Production atmosphere/ocean remap with single-flux-per-edge FV data.
- Desired accuracy is O(h²) or better (p=1 is the sweet spot).
- Source mesh is arbitrary polygonal (Voronoi, cubed-sphere, etc.).
- Cell moments are not available from the source discretization.

**When to use `VemProjection` directly:**

- Source discretization provides high-order edge moments (e.g., high-order
  DG with moment DOFs on edges).
- Cell vector moments are available (e.g., from a mixed method).
- p ≥ 2 with full-order convergence required.
- Testing and validation with manufactured solutions.

**When `SplitBasis` is adequate:**

- Structured quad meshes where cells are affine (parallelograms).
- p ≤ 1 on any mesh (well-conditioned at low order).
- Legacy workflows that provide cell moments but not the VEM infrastructure.

### Mesh Topology Independence

A key motivation for the VEM path is **topology independence** [6,8]: the
same code and algorithm handles quadrilaterals, triangles, pentagons,
hexagons, and arbitrary n-gons without any topology-specific branches.
The SplitBasis path also handles arbitrary polygons in principle, but its
conditioning depends on the polygon shape through the trace operator [1,2],
creating *topology-dependent failure modes* at p ≥ 2 (see the Voronoi
divergence in the convergence table above).

The VEM path eliminates this dependence because:

1. The decomposed basis `∇P_{p+1} ⊕ x⊥P_{p-1}` is defined purely in terms
   of polynomial spaces, not reference-element mappings.

2. The mass matrix `∫_K φ_i · φ_j dA` depends on cell geometry only through
   the monomial integrals, which are computed by fan-triangulated quadrature
   that works on any simple polygon.

3. The DOF-to-coefficient map uses integration-by-parts identities that
   involve boundary integrals (summed over edges) and cell integrals, both
   of which are well-defined on any polygon.

This makes the VEM path particularly well-suited for climate model remap,
where the source mesh may contain a mix of quads (cubed-sphere), pentagons
and hexagons (icosahedral), and arbitrary Voronoi cells.

**Verified:** `test_vem_topology_independence()` in `vem_projection_test.cpp`
confirms exact polynomial recovery (error ~10⁻¹⁵) on triangles (3 edges),
pentagons (5 edges), hexagons (6 edges), and irregular 7-gons — all using
the same code path with no topology-specific branches.

### Spherical Lift (Hodge-Weighted VEM)

When `GeometryOptions::metric_weighted = true` and `GeometryOptions::mode
= SphericalGnomonic` are both set, the VEM branch of
`reconstruct_source_polygon` switches to a Hodge-area-weighted assembly
that respects the surface L²(S²) inner product:

- **Mass matrix.**  `vem_mass_matrix_weighted` integrates the basis with
  the chart Hodge metric `h(ξ) = JᵀJ / |J|`:
  `M_{ij} = ∫_K̂ φᵢᵀ h(ξ) φⱼ dξ dη`.  The metric is factored through three
  precomputed monomial integral tables `I_xx`, `I_xy`, `I_yy` built once
  per cell via `polygon_monomial_integral_table_weighted`.

- **Divergence projection.**  `vem_reconstruct_divergence_weighted`
  projects `div_S(u)` onto `P_{p-1}` in surface L²: the Gram matrix uses
  the Hodge-area weight `|J|` (table `I_J`), and the RHS reduces by the
  Piola identity `div_S(u)·|J| = div_flat(v̂)` to the chart-IBP
  expression that the planar VEM already computes.

- **Projection RHS.**  `vem_projection_rhs_weighted` evaluates
  `(v_h, φᵢ)_S` for each basis mode.  Gradient modes use the surface IBP
  to the same chart-IBP form as planar (with `I_J` for the divergence
  volume term).  Rotational modes carry an `O(h²)` centroid-Hodge
  approximation: `h(ξ)` is evaluated at the cell centroid and factored
  out so the chart-area cell vector moments suffice.

The verified scope today is the **gradient subspace** of the spherical
VEM:

- p = 1 cubed-sphere edge-moment-0 error matches the SplitBasis
  baseline (rate ≈ 2.0 between `h = 1/4` and `h = 1/8` on the
  manufactured spherical-harmonic gradient field).
- Conservation: cell-divergence residuals at machine roundoff
  (≤ 5 × 10⁻¹³) at every tested resolution.

The **rotational subspace** is currently capped at `O(h²)` by the
centroid-Hodge approximation, which dominates the asymptotic error for
`p ≥ 2`.  For `p ≥ 2` on spherical meshes, the SplitBasis path remains
the recommended choice today.  Lifting the rotational rate to the full
`O(h^{p+1})` requires either (a) introducing Hodge-component cell-moment
DOFs `c^{(α)}_{xx} = ∫_K̂ v̂ₓ x^a y^b h_xx dξ dη` (and similar for
xy, yy) or (b) adopting the SplitBasis path's degree-elevation rule for
the spherical VEM polynomial basis.  This is the single open task in the
spherical-vem-lift plan (`docs/plans/2026-05-02-spherical-vem-lift.md`).

Test: `tests/spherical_vem_test.cpp` exercises the path at p = 1 with
strict refinement and conservation assertions, and at p = 2 with
smoke + conservation only (rate test deferred per the limitation above).

### Other Limitations of the Current Implementation

2. **No adaptive stabilization.**  The mass matrix solve uses LDLT with SVD
   fallback but without Tikhonov regularization or mode truncation.  On
   very thin or degenerate cells, the mass matrix may have small eigenvalues
   that amplify noise in the DOFs.  The SVD fallback provides robustness
   against factorization failure, but does not apply the SplitBasis path's
   regularization parameter (`MomentMethodOptions::regularization`).

3. **No incremental update.**  Each call to `reconstruct_source_polygon()`
   recomputes the monomial integral table, decomposed basis, and mass matrix
   from scratch.  For production use on large meshes, caching the per-cell
   VEM infrastructure (which depends only on cell geometry, not on DOF
   values) would improve performance.

4. **Patch recovery at p ≥ 3.**  The face-neighbor stencil provides too few
   equations for a well-determined LS fit at p=3.  Extended stencils
   (vertex-neighbors or two-ring neighbors) would increase the row count but
   also increase the patch diameter, potentially degrading the locality of
   the fit.  This is an open research direction.


## References

This work sits at the intersection of several areas: classical H(div) finite
element spaces and their Piola degradation on non-affine meshes, the Virtual
Element Method (VEM) for H(div) on polygonal meshes, k-exact and
superconvergent recovery from integral data, conservative remap on the
sphere, and numerical methods for quadrature and polygon intersection.  The
references below are organized by topic and annotated to explain their
relevance to the specific algorithms and design decisions in this codebase.

The codebase implements a level-2 mimetic reconstruction (§Reconstruction
Path) on polygonal meshes using harmonic basis functions [15,16], extended to
arbitrary polynomial order via a split H(div)-style basis hierarchy
(§Higher-Order Planar Moment Path).  The classical Raviart-Thomas [17] and
Brezzi-Douglas-Marini [18] spaces provide the finite element context for the
edge DOF structure and trace operators used in the SplitBasis path.  The
Piola degradation theory [1–5] explains why the SplitBasis path loses
conditioning at p ≥ 2 on non-affine cells, motivating the VEM projection
[6–8] which operates in the decomposed basis ∇P_{p+1} ⊕ x⊥P_{p-1} from
finite element exterior calculus [19,20].  The k-exact [11,12] and
superconvergent patch recovery [13,14] theory bounds the accuracy of the
patch-based moment recovery when only single-flux-per-edge data is
available.  The spherical remap infrastructure uses gnomonic chart
construction [21,22], conservative remapping algorithms [23–25], and polygon
clipping [26].  The numerical infrastructure relies on Gauss-Legendre
quadrature [27], Dunavant triangle rules [28], the Duffy singularity
transform [29], and standard constrained least-squares methods [30].

### Piola transformation and convergence loss on non-affine elements

[1] D.N. Arnold, D. Boffi, R.S. Falk.
    "Quadrilateral H(div) Finite Elements."
    SIAM J. Numer. Anal., 42(6):2429–2451, 2005.
    DOI: 10.1137/S0036142903431924

    Proves that standard RT and BDM elements lose optimal convergence
    order for ∇·u when mapped to non-affine quadrilaterals via the Piola
    transform.  Provides necessary and sufficient conditions on the
    reference shape function space for optimal L²-approximation of both
    the vector field and its divergence.  The condition `Ŝ ⊇ Q_r(K̂)` is
    required; `[P_p]²` fails this for p ≥ 2.  This is the primary
    theoretical justification for the VEM path.

[2] D.N. Arnold, D. Boffi, R.S. Falk.
    "Approximation by Quadrilateral Finite Elements."
    Math. Comput., 71(239):909–922, 2002.

    Establishes the fundamental theory: on quadrilateral meshes with
    bilinear maps, P_r-based finite elements can fail to achieve optimal
    approximation order.  The condition `Ŝ ⊇ Q_r(K̂)` is necessary and
    sufficient for optimal order on general quadrilaterals, while
    `Ŝ ⊇ P_r(K̂)` suffices only on asymptotically affine meshes.

[3] D. Boffi.
    "On the Finite Element Method on Quadrilateral Meshes."
    Appl. Numer. Math., 2006.

    Survey of approximation degradation phenomena for H(div) and H(curl)
    elements on non-affine quadrilaterals.

[4] D.N. Arnold, D. Boffi, F. Bonizzoni.
    "Finite Element Differential Forms on Curvilinear Cubic Meshes and
    Their Approximation Properties."
    Numer. Math., 129:1–20, 2015.

    Extends the Piola degradation analysis to 3D.  Shows that curvilinear
    meshes lose one order in 2D and two orders in 3D versus parallelotope
    meshes, with the effect more severe for higher-order differential
    forms (H(div) worse than H¹).

[5] M.R. Correa, T. Arbogast.
    "Two Families of H(div) Mixed Finite Elements on Quadrilaterals of
    Minimal Dimension."
    SIAM J. Sci. Comput., 38(6):A3388–A3411, 2016.
    DOI: 10.1137/15M1013705

    Constructs minimal-dimension quadrilateral H(div) elements that avoid
    Piola degradation by defining basis functions directly on physical
    quadrilaterals rather than mapping from reference elements.

### Virtual Element Method for H(div) on polygons

[6] L. Beirão da Veiga, F. Brezzi, L.D. Marini, A. Russo.
    "H(div) and H(curl)-Conforming Virtual Element Methods."
    Numer. Math., 133:303–332, 2016.
    DOI: 10.1007/s00211-015-0746-1

    Constructs H(div)-conforming VEM spaces on general polygonal and
    polyhedral elements.  §3 defines the local space; §4 defines the DOFs
    (edge moments + cell moments) and proves that the polynomial projection
    Π_p is exactly computable (Prop. 4.1).  This is the foundational
    reference for the VEM projection implemented in this codebase.

[7] F. Brezzi, R.S. Falk, L.D. Marini.
    "Basic Principles of Mixed Virtual Element Methods."
    ESAIM: M2AN, 48(4):1227–1240, 2014.

    Establishes the mixed VEM framework on polygonal meshes, showing how
    to construct polynomial projections from VEM DOFs without explicitly
    computing non-polynomial basis functions.

[8] L. Beirão da Veiga, F. Brezzi, A. Cangiani, G. Manzini,
    L.D. Marini, A. Russo.
    "Basic Principles of Virtual Element Methods."
    Math. Models Methods Appl. Sci., 23(1):199–214, 2013.

    The foundational VEM paper.  §2 introduces the key design principle:
    DOFs are chosen so that the stiffness matrix (and projection operator)
    is computable without knowing the non-polynomial basis functions
    explicitly.  §3 proves optimal convergence estimates.

### Mimetic finite differences on polygonal meshes

[9] F. Brezzi, K. Lipnikov, M. Shashkov.
    "Convergence of the Mimetic Finite Difference Method for Diffusion
    Problems on Polyhedral Meshes."
    SIAM J. Numer. Anal., 43(5):1872–1896, 2005.

    Proves first-order convergence for flux and second-order for pressure
    on general polyhedral meshes using the MFD method.  The MFD framework
    is a precursor to VEM and operates with a single flux DOF per face —
    this is the theoretical floor for single-flux accuracy.

[10] Y. Kuznetsov, K. Lipnikov, M. Shashkov.
     "The Mimetic Finite Difference Method on Polygonal Meshes for
     Diffusion-Type Problems."
     Comput. Geosci., 8:301–317, 2004.

     Derives mimetic discretizations on polygonal meshes with one flux DOF
     per edge, demonstrating first-order accuracy for velocity.  Confirms
     the inherent limitation of single-flux data for higher-order recovery.

### k-Exact reconstruction from integral data

[11] T.J. Barth, P.O. Frederickson.
     "Higher Order Solution of the Euler Equations on Unstructured Grids
     Using Quadratic Reconstruction."
     AIAA Paper 90-0013, 28th Aerospace Sciences Meeting, 1990.

     The foundational paper on k-exact reconstruction: fitting a degree-k
     polynomial over a neighbor-cell stencil from cell-average data.  A
     k-exact reconstruction achieves at most (k+1)-order accuracy.  §2–3
     provide the LS formulation and stencil analysis.  Our patch-based
     edge-flux recovery is a direct adaptation to edge-normal-flux data.

[12] T.J. Barth.
     "Recent Developments in High Order k-Exact Reconstruction on
     Unstructured Meshes."
     AIAA Paper 93-0668, 1993.  Also: NASA TM-108975.

     Extends k-exact theory with improved stencil selection and conditioning
     analysis.  Shows that reconstruction accuracy depends on both stencil
     geometry and the conditioning of the moment matrix — directly relevant
     to our patch diameter scaling.

### Superconvergent patch recovery

[13] O.C. Zienkiewicz, J.Z. Zhu.
     "The Superconvergent Patch Recovery and A Posteriori Error Estimates.
     Part 1: The Recovery Technique."
     Int. J. Numer. Methods Eng., 33:1331–1364, 1992.

     The SPR technique: recover nodal gradients by least-squares polynomial
     fitting over a patch of elements.  Under mesh regularity conditions,
     the recovered gradient converges one order faster than the raw FE
     gradient.  Our patch recovery adapts the same idea to edge-flux
     integral data rather than nodal point values.

[14] O.C. Zienkiewicz, J.Z. Zhu.
     "The Superconvergent Patch Recovery and A Posteriori Error Estimates.
     Part 2: Error Estimates and Adaptivity."
     Int. J. Numer. Methods Eng., 33:1365–1382, 1992.

     Proves that the SPR-based error estimator is asymptotically exact
     under the superconvergence conditions of Part 1.

### Mimetic reconstruction on polygons

[15] A. Palha, M. Gerritsma.
     "A Mimetic Method for Polygons."
     J. Comput. Phys., 425:109891, 2021.

     Describes harmonic-function-based mimetic interpolation on polygonal
     meshes, shown to be a direct extension of lowest-order Raviart-Thomas
     to polygons.  Demonstrates that the naive harmonic expansion is not
     stable and proposes truncated expansions.  Closely related to the
     low-order harmonic reconstruction path in this codebase.

[16] B. Perot, R. Subramanian.
     "Discrete Calculus Methods for Diffusion."
     J. Comput. Phys., 224(1):59–81, 2007.

     Introduces the level-2 mimetic reconstruction from edge fluxes using
     a constant divergence term plus harmonic corrections.  This is the
     mathematical foundation for the low-order `MimeticInterpolator`
     reconstruction path in this codebase (§Reconstruction Path, step 4):
     the field is expressed as u_h(x) = (d/2)x + Σ(aₖ∇Pₖ + bₖ∇Qₖ) where
     d is the cell divergence and Pₖ, Qₖ are harmonic basis functions.
     The extension to higher-order via a split basis hierarchy (§Higher-
     Order Planar Moment Path) generalizes this to polynomial-divergence,
     harmonic-gradient, and divergence-free completion modes.

### Classical H(div) finite element spaces

[17] P.-A. Raviart, J.-M. Thomas.
     "A Mixed Finite Element Method for 2nd Order Elliptic Problems."
     In: Mathematical Aspects of Finite Element Methods, Lecture Notes in
     Math. 606, Springer, 1977, pp. 292–315.

     Defines the Raviart-Thomas (RT) finite element spaces on simplices.
     RT_p has one normal-trace DOF of degree p per edge, matching the DOF
     structure used in this codebase's edge moment hierarchy (§Higher-Order
     Planar Moment Path).  The extension to polygons requires either VEM
     [6–8] or mimetic [9,10,15,16] formulations because RT spaces are
     defined only on simplicial or reference-element-based meshes.

[18] F. Brezzi, J. Douglas Jr., L.D. Marini.
     "Two Families of Mixed Finite Elements for Second Order Elliptic
     Problems."
     Numer. Math., 47(2):217–235, 1985.

     Defines the BDM finite element spaces on simplices and quadrilaterals.
     BDM_p has full polynomial normal traces of degree p on each edge,
     unlike RT_p which has specific internal enrichments.  The `SplitBasis`
     path in this codebase uses a `[P_p]²` reconstruction space that
     contains BDM_p as a subspace on triangles, but on general polygons the
     BDM structure is not directly available — this motivates the VEM
     approach.

### Finite element exterior calculus

[19] D.N. Arnold, R.S. Falk, R. Winther.
     "Finite Element Exterior Calculus, Homological Techniques, and
     Applications."
     Acta Numer., 15:1–155, 2006.

     Comprehensive treatment of the de Rham complex structure underlying
     H(div) finite element spaces.  The decomposition `[P_p]² = ∇P_{p+1}
     ⊕ x⊥P_{p-1}` used in the VEM projection (§VEM H(div) Projection,
     "Polynomial space and decomposition") is the 2D instance of the
     Koszul complex splitting of polynomial differential forms described
     in §4.  The orthogonality of gradient and rotational subspaces under
     the L² inner product is proved in Lemma 4.2.

[20] D.N. Arnold, R.S. Falk, R. Winther.
     "Finite Element Exterior Calculus: From Hodge Theory to Numerical
     Stability."
     Bull. Amer. Math. Soc., 47(2):281–354, 2010.

     Survey and extension of [19].  §5 discusses the Piola transformation
     in the FEEC framework: the contravariant Piola map preserves normal
     traces (flux continuity) but distorts polynomial structure, which is
     the root cause of the conditioning issues analyzed in §Stability and
     Conditioning.  The VEM approach bypasses this by working directly in
     physical coordinates without reference-element mappings.

### Spherical geometry and cubed-sphere construction

[21] M. Rancic, R.J. Purser, F. Mesinger.
     "A Global Shallow-Water Model Using an Expanded Spherical Cube:
     Gnomonic versus Conformal Coordinates."
     Quart. J. Roy. Meteor. Soc., 122(532):959–982, 1996.

     Introduces the gnomonic cubed-sphere grid for atmospheric modeling.
     The gnomonic projection used throughout this codebase (§Gnomonic map
     and inverse, `project_gnomonic()`, `inverse_gnomonic()`) maps each
     cube face onto the sphere via central projection.  The key property
     exploited in §Transfer Path is that gnomonic projection maps great
     circles to straight lines, making polygon clipping exact in chart
     coordinates.

[22] C. Ronchi, R. Iacono, P.S. Paolucci.
     "The 'Cubed Sphere': A New Method for the Solution of Partial
     Differential Equations in Spherical Geometry."
     J. Comput. Phys., 124(1):93–114, 1996.

     Independent development of the cubed-sphere discretization with
     analysis of the gnomonic metric tensor.  The Hodge metric
     `J^T J / |J|` used in the spherical Gram matrix computation
     (§Spherical Accuracy Improvements, item 1) derives from the Jacobian
     of the inverse gnomonic map described in this paper.  The metric
     variation across a cubed-sphere face (~O(h²) per cell) is the root
     cause of the basis span limitation analyzed in §Root Cause: Basis
     Span Limitation.

### Conservative remap on the sphere

[23] P.W. Jones.
     "First- and Second-Order Conservative Remapping Schemes for Grids in
     Spherical Coordinates."
     Mon. Wea. Rev., 127(9):2204–2210, 1999.

     Describes the SCRIP (Spherical Coordinate Remapping and Interpolation
     Package) algorithm for first- and second-order conservative scalar
     remap between spherical meshes.  The edge-to-edge transfer in this
     codebase (§Transfer Path) extends SCRIP's cell-overlap approach to
     vector (H(div)) quantities by computing overlap integrals of the
     reconstructed source field against target edge functionals, rather
     than simple area-weighted cell averages.

[24] P.A. Ullrich, P.H. Lauritzen, C. Jablonowski.
     "Geometrically Exact Conservative Remapping (GECoRe): Regular
     Latitude-Longitude and Cubed-Sphere Grids."
     Mon. Wea. Rev., 137(6):1721–1741, 2009.

     Introduces geometrically exact overlap computation for conservative
     remap on cubed-sphere grids.  The line-integral formulation of overlap
     areas used here is related to the edge-flux transfer mechanism in
     this codebase, where source reconstruction coefficients are integrated
     along clipped target-edge segments (§Sparse Projection Path).

[25] P.A. Ullrich, M.A. Taylor.
     "Arbitrary-Order Conservative and Consistent Remapping and a Theory
     of Linear Maps: Part I."
     Mon. Wea. Rev., 143(6):2419–2440, 2015.

     Develops the TempestRemap framework for arbitrary-order conservative
     remap.  The theory of linear remap maps (U_t = P U_s) is directly
     relevant to the sparse projection operator assembled in §Sparse
     Projection Path.  However, TempestRemap operates on scalar fields
     (cell averages), while this codebase extends the approach to H(div)
     edge-flux fields with an additional conforming projection step
     (§Global Target-Edge Conforming Projection).

### Polygon clipping

[26] I.E. Sutherland, G.W. Hodgman.
     "Reentrant Polygon Clipping."
     Commun. ACM, 17(1):32–42, 1974.

     The Sutherland-Hodgman algorithm clips a subject polygon against a
     convex clip polygon by iterating over clip edges.  This algorithm is
     used in the overlap computation between source and target cells
     (§Transfer Path, step 3; see `voronoi_intersection_test.cpp` which
     explicitly references this as "the Sutherland-Hodgman step used in
     report Algorithm 3").  In this codebase, the algorithm operates in
     gnomonic chart coordinates where great-circle edges become straight
     lines, making the planar clipping exact on the sphere.

### Quadrature rules

[27] M. Abramowitz, I.A. Stegun (eds.).
     Handbook of Mathematical Functions with Formulas, Graphs, and
     Mathematical Tables.
     National Bureau of Standards, Applied Mathematics Series 55, 1964.
     §25.4: Gauss-Legendre quadrature nodes and weights.

     The Gauss-Legendre quadrature rules used for edge moment computation
     (§Recovery Pipeline, step 3) and the 10-point rule referenced in
     `convergence_validation_test.cpp` and `dump_visuals.cpp` are taken
     from Table 25.4.  Gauss-Legendre quadrature with n points integrates
     polynomials of degree 2n-1 exactly; this guarantees exact edge moment
     evaluation for polynomial fields of degree ≤ p when using ≥ (p+1)/2
     quadrature points.

[28] D.A. Dunavant.
     "High Degree Efficient Symmetrical Gaussian Quadrature Rules for the
     Triangle."
     Int. J. Numer. Methods Eng., 21:1129–1148, 1985.

     Provides symmetric quadrature rules on the reference triangle that
     are exact for polynomials up to specified degrees.  The 13-point
     degree-7 and 25-point degree-10 rules used in
     `integrate_triangle_highorder()` (see `include/mimetic/mimetic.hpp`)
     are adapted from this source.  These rules are used for cell-moment
     computation and Gram matrix assembly on fan-triangulated polygons when
     the integrand is polynomial (non-metric-weighted case).

[29] M.G. Duffy.
     "Quadrature Over a Pyramid or Cube of Integrands with a Singularity
     at a Vertex."
     SIAM J. Numer. Anal., 19(6):1260–1262, 1982.

     Introduces the Duffy transformation: a change of variables that
     removes vertex singularities from integrands over simplices by mapping
     the reference triangle to a unit square.  In this codebase, the Duffy
     transform is used in `integrate_triangle_duffy()` for two purposes:
     (1) fan-triangulated cell integrals where thin triangles near the
     centroid would cause quadrature inaccuracy with symmetric rules, and
     (2) metric-weighted integrals in spherical mode where the gnomonic
     Hodge metric introduces rational (non-polynomial) integrands that
     require higher-order quadrature.  The Duffy rule with n×n
     Gauss-Legendre points integrates polynomials of degree 2n-1 exactly
     after the variable substitution.

### Constrained least-squares and KKT systems

[30] Å. Björck.
     Numerical Methods for Least Squares Problems.
     SIAM, Philadelphia, 1996.

     Comprehensive treatment of constrained least-squares problems and
     their solution via KKT (Karush-Kuhn-Tucker) systems.  The local
     reconstruction in the `SplitBasis` path (§Reconstruction Path, step 4;
     §Higher-Order Planar Moment Path, step 3) solves a KKT system where
     the zeroth edge moment on each edge is an exact constraint and higher-
     order moments are weighted soft rows.  The SVD solver used in the
     patch recovery (§Patch-Based Moment Recovery, "SVD solver") provides
     a minimum-norm solution when the system is underdetermined, following
     the analysis in Chapter 3.  The LDLT factorization used in the VEM
     mass matrix solve (§VEM H(div) Projection, "Elliptic projection") is
     appropriate because the mass matrix is symmetric positive definite.
