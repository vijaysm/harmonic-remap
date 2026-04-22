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
| Numerical kernels | `src/mimetic.cpp` | Local geometry construction, harmonic basis evaluation, low-order KKT reconstruction, unified split-basis high-order reconstruction, clipped edge transfer, sparse operator assembly, global target-edge constrained projection, and MatrixMarket output. |
| Shared spherical test utilities | `tests/spherical_transfer_test_utils.hpp` | Cubed-sphere generators, manufactured spherical fields, edge quadrature, conservative edge-flux assignment, and diagnostic helpers. |
| Planar validation | `tests/patch_test.cpp`, `tests/conservative_intersection_test.cpp`, `tests/voronoi_intersection_test.cpp`, `tests/convergence_validation_test.cpp`, `tests/hdiv_conforming_projection_test.cpp`, `tests/high_order_edge_moment_test.cpp`, `tests/high_order_hdiv_convergence_test.cpp` | Constant patch, rectangular overlap, Voronoi n-gon, planar convergence, global target-edge conforming-projection checks, exact higher-order patch tests, and `p=1,2,3` high-order convergence studies. |
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

The reconstruction path is:

1. `local_polygon(...)` builds a cell-local coordinate frame.
2. `local_edges(...)` builds ordered directed edges and outward normals.
3. `source_edge_flux(...)` reads the local directed flux vector.
4. `reconstruct_source_polygon(...)` computes the constant divergence and
   harmonic coefficients by solving a constrained local KKT system.
5. `velocity(...)`, `line_integral(...)`, and `edge_flux(...)` evaluate the
   reconstructed chart field and its edge integrals.

For spherical geometry, the reconstructed field is a chart vector.  Surface
vectors used in diagnostics are obtained through `lift_contravariant_piola(...)`.

## Transfer Path

The direct source-to-target path is:

1. Reconstruct every source cell.
2. For every directed target edge, project its endpoints into each source-cell
   chart.
3. Because gnomonic projection maps great circles to straight lines, clip the
   projected target segment against the projected source polygon.
4. Evaluate the source reconstruction on each retained subsegment.
5. Sum all subsegment contributions into the directed target edge flux.

Coincident source-target edges on the sphere require deterministic ownership.
The current code keeps the source cell on the target-cell-interior side of the
directed target edge to avoid double counting.

## Higher-Order Planar Moment Path

The higher-order kernel is intentionally separate from
`MimeticInterpolator`, but it now uses one hierarchy for `p=1,2,3`.

1. `PlanarMomentInterpolator::set_source_edge_moments(...)` stores one vector of
   Legendre edge moments per directed source-cell edge.
2. `set_source_cell_vector_moments(...)` stores optional interior vector
   moments.
3. `reconstruct_source_polygon(...)` builds a local split basis consisting of
   polynomial-divergence modes, harmonic-gradient modes, and divergence-free
   completion modes, assembles edge and cell moment constraints, and solves for
   the local coefficients.
4. `transfer_source_to_target_edge_moments(...)` clips every directed target
   edge against all source cells and accumulates the corresponding target edge
   moments.

The implementation is currently a polygonal `H(div)`-style research kernel
rather than a full polygonal VEM or generalized Whitney space.  It is exact for
the current `p=1` affine and `p=2` manufactured polynomial regressions because
those source fields lie in the reconstructed local space and the source moments
are integrated exactly.

Two details matter:

- The Duffy triangle map used for cell moments is implemented in
  `integrate_triangle_duffy(...)`.  A wrong map here will silently corrupt the
  higher-order moment constraints.
- For fully or over-determined local systems the code solves the constraint
  matrix directly with QR.  The KKT minimum-energy solve is only used when the
  local system is under-determined.
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
observed average relative edge-flux rates are approximately `1.90`, `2.77`,
and `3.34` for `p=1,2,3`.

The spherical convergence driver
`tests/spherical_high_order_hdiv_convergence_test.cpp` applies the same order
split on source-cell gnomonic charts and then projects the resulting target
edge moments onto a globally conforming target skeleton.  It reports both the
zeroth-moment edge error and the all-moment edge error on structured
cubed-sphere transfers and deterministic spherical Voronoi patches.

## Sparse Projection Path

The sparse operator path mirrors direct transfer.  For each source cell,
`source_reconstruction_matrix(...)` computes a local matrix \(B_K\) such that
the reconstruction coefficients are linear in the source edge fluxes.  Each
clipped target-edge segment then contributes a row functional
\(L_{\gamma,K} B_K\) to the sparse matrix.

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
