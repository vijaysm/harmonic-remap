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
| Public API and data types | `include/mimetic/mimetic.hpp` | Geometry options, local polygon data, reconstruction coefficients, sparse projection types, and `MimeticInterpolator` interface. |
| Numerical kernels | `src/mimetic.cpp` | Local geometry construction, harmonic basis evaluation, KKT reconstruction, clipped edge transfer, sparse operator assembly, and MatrixMarket output. |
| Shared spherical test utilities | `tests/spherical_transfer_test_utils.hpp` | Cubed-sphere generators, manufactured spherical fields, edge quadrature, conservative edge-flux assignment, and diagnostic helpers. |
| Planar validation | `tests/patch_test.cpp`, `tests/conservative_intersection_test.cpp`, `tests/voronoi_intersection_test.cpp`, `tests/convergence_validation_test.cpp` | Constant patch, rectangular overlap, Voronoi n-gon, and planar convergence checks. |
| Spherical validation | `tests/spherical_geometry_test.cpp`, `tests/spherical_quad_test.cpp`, `tests/spherical_voronoi_test.cpp`, `tests/spherical_scalar_test.cpp` | Spherical geometry primitives, structured cubed-sphere transfer, unstructured Voronoi transfer, and scalar control. |
| Study scripts | `scripts/convergence_study.sh`, `scripts/plot_convergence.py` | Structured spherical convergence sweep and rate reporting. |

## Manuscript-to-Code Map

| Manuscript concept | Primary code location | Notes |
| --- | --- | --- |
| Geometry mode and tolerances | `GeometryMode`, `GeometryOptions` in `include/mimetic/mimetic.hpp` | Selects planar or spherical gnomonic geometry and stores tolerances. |
| Local planar polygon data | `LocalPolygon` in `include/mimetic/mimetic.hpp`; `local_polygon(...)` in `src/mimetic.cpp` | Stores centroid-relative 2D points, chart area, spherical area, and spherical frame metadata. |
| Spherical chart construction | `SphericalPolygon`, `spherical_polygon(...)`, `make_gnomonic_frame(...)` | Normalizes vertices, constructs source-cell tangent frames, projects great-circle vertices to chart coordinates. |
| Gnomonic map and inverse | `project_gnomonic(...)`, `inverse_gnomonic(...)` | Implements the chart map used for spherical clipping. |
| Piola lift and pullback | `lift_contravariant_piola(...)`, `pullback_contravariant_piola(...)` | Preserves normal fluxes between chart and surface curves. |
| Harmonic basis | `eval_harmonic_basis(...)` and gradient helpers in `src/mimetic.cpp` | Evaluates \(P_k,Q_k\) and gradients used in the local reconstruction. |
| Source-cell reconstruction | `MimeticInterpolator::reconstruct_source_polygon(...)` | Builds local moment matrices, KKT constraints, and reconstruction coefficients from directed edge fluxes. |
| Source reconstruction matrix | `source_reconstruction_matrix(...)` in `src/mimetic.cpp` | Builds the local linear map from source edge fluxes to reconstruction coefficients for sparse projection assembly. |
| Direct edge transfer | `MimeticInterpolator::transfer_source_to_target_edges(...)` | Clips each directed target edge against candidate source cells and evaluates the source reconstruction on clipped segments. |
| Sparse edge projection | `MimeticInterpolator::assemble_edge_projection_operator(...)` | Applies the same clipped-segment functional to local reconstruction matrices and assembles \(U_t = P U_s\). |
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

## Sparse Projection Path

The sparse operator path mirrors direct transfer.  For each source cell,
`source_reconstruction_matrix(...)` computes a local matrix \(B_K\) such that
the reconstruction coefficients are linear in the source edge fluxes.  Each
clipped target-edge segment then contributes a row functional
\(L_{\gamma,K} B_K\) to the sparse matrix.

The acceptance criterion is that applying the sparse matrix to the source
edge-flux vector reproduces direct transfer to the conservation tolerance.

## Diagnostics and VTK Output

`tests/spherical_quad_test.cpp` writes target-cell diagnostic tags:

- `TARGET_DIV_RECON`: sum of transferred directed edge fluxes around a target
  cell.
- `TARGET_FLUX_ERROR`: maximum directed edge-flux error on that target cell.
- `TARGET_FIELD_RECON`: centroid-sampled reconstructed surface vector.
- `TARGET_FIELD_EXACT`: centroid-sampled manufactured surface vector.
- `TARGET_FIELD_ERROR_NORM`: norm of the difference between the two sampled
  vectors.

The edge-flux norms printed by the test are edge-integral norms, not vector
field norms.  Therefore a small `edge_flux_l2_rel` does not imply identical
global bounds for `TARGET_FIELD_RECON` and `TARGET_FIELD_EXACT`.

Current diagnostic caveat: `TARGET_FIELD_RECON` is sampled at the projected
chart centroid, while `TARGET_FIELD_EXACT` is sampled at the gnomonic frame
center.  Those should be made the same physical point before interpreting the
VTK vector fields quantitatively.  A future diagnostic should also write
`TARGET_FIELD_ERROR` as a 3-vector and `TARGET_FIELD_SAMPLE_POINT`.

## Current Numerical Caveats

The current implementation passes the conservation and direct-vs-sparse tests,
but the following issues should be resolved before treating the spherical
accuracy diagnostics as final:

1. The metric-weighted reconstruction currently multiplies some interior
   integrands by the gnomonic area scale but still divides averages by chart
   area.  If the intended norm is physical area weighted, the normalization and
   vector Hodge/Piola metric should be made consistent.
2. `source_reconstruction_matrix(...)` currently infers metric weighting from
   geometry, while direct reconstruction uses `GeometryOptions::metric_weighted`.
   These should share one option-controlled reconstruction operator.
3. The scalar control test estimates spherical overlap area with a centroid
   area-scale approximation.  For a strict conservative scalar baseline, the
   overlap area should be integrated over the clipped polygon or computed as a
   spherical polygon area, and global scalar conservation should be asserted.

## Build and Validation Commands

```bash
cmake -S . -B build
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Structured spherical convergence sweep:

```bash
bash scripts/convergence_study.sh build > /tmp/mimetic_convergence.csv
python3 scripts/plot_convergence.py /tmp/mimetic_convergence.csv
```

The convergence script currently invokes the VTK-writing driver path for each
case.  A future CLI flag should allow convergence sweeps without writing VTK
files.
