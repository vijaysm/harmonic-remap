# Mimetic-SphPoly

Prototype C++14 implementation of a conservative mimetic polygon interpolation
kernel using MOAB for mesh entities/tags and Eigen3 for dense and sparse linear
algebra.

The repository now contains two geometry backends:

- `GeometryMode::Planar`, the original validated polygon implementation.
- `GeometryMode::SphericalGnomonic`, a unit-sphere great-circle polygon
  implementation using source-cell gnomonic charts and contravariant Piola
  flux mapping.

Both backends use directed cell-local edge degrees of freedom and are tested
against a `5.0e-13` absolute conservation tolerance.

## What The Code Does

The source data are signed integrated normal fluxes on directed source-cell
edges,

```text
U_f = integral_f u . n_f ds.
```

For each source polygon, `MimeticInterpolator::reconstruct_source_polygon`
builds a level-2 Perot-Chartrand-style reconstruction

```text
u_h(x) = (d/2) x + a1 grad P1 + b1 grad Q1 + a2 grad P2 + b2 grad Q2,
```

where `d` is the constant cell divergence and the harmonic coefficients are
computed from source edge flux moments.

There are two distinct transfer/verification paths:

1. **Overlap conservation diagnostic.**  
   For each source-target cell overlap `O = Ks intersect Kt`, the code checks
   the local divergence theorem,

   ```text
   integral_boundary(O) u_h^s . n ds = d_s area(O).
   ```

   This verifies the conservative reconstruction and polygon clipping, but it is
   not by itself a final target-edge remap.

2. **Edge-wise source-to-target transfer.**  
   For each directed target edge, the code clips that edge against all source
   polygons it crosses, evaluates the reconstructed source field on each clipped
   subsegment, and sums the contributions:

   ```text
   U_t(e) = sum_s integral_{e intersect Ks} u_h^s . n_t(e) ds.
   ```

   This is the actual directed source-edge to directed target-edge transfer used
   by the current nonmatching rectangular and Voronoi tests.

## Geometry Backends

### Planar

Planar polygons are stored in centroid-relative coordinates.  Edge normals are
the right normals of counter-clockwise cell edges, and the KKT reconstruction is
assembled directly in the physical plane.

### Spherical Gnomonic

Spherical cells are assumed to be convex great-circle polygons contained in the
gnomonic hemisphere of their source-cell chart.  `spherical_polygon()` stores:

- normalized 3D vertices,
- a tangent frame,
- gnomonic projected vertices,
- chart and spherical areas,
- oriented great-circle edge metadata.

For a source-cell tangent frame, `project_gnomonic()` maps unit-sphere points to
chart coordinates and `inverse_gnomonic()` maps chart points back to the sphere.
`gnomonic_jacobian()`, `pullback_contravariant_piola()`, and
`lift_contravariant_piola()` implement the chart/surface vector mapping.  Source
edge physical fluxes are treated as chart fluxes under this contravariant Piola
mapping, so the same dynamic KKT flux-matching reconstruction can be reused in
the source chart.

The spherical direct transfer projects each target great-circle edge into every
candidate source chart.  Gnomonic projection maps great circles to straight
chart segments, so the existing convex segment clipping and line-functional
evaluation remain valid.  Coincident source/target boundary segments are
disambiguated by retaining the source cell on the target-cell interior side of
the directed edge.

## Sparse Projection Operator

The edge-wise transfer is linear in the source edge fluxes.  The implementation
can assemble this map as

```text
U_t = P U_s,
```

where rows are directed target-cell edges and columns are directed source-cell
edges.

The relevant API is:

```cpp
mimetic::EdgeTransferResult transfer =
    interpolator.transfer_source_to_target_edges(source_cells, target_cells);

mimetic::SparseEdgeProjection projection =
    interpolator.assemble_edge_projection_operator(source_cells, target_cells);

mimetic::write_matrix_market(
    projection,
    "/tmp/edge_projection.mtx",
    "/tmp/source_edges.csv",
    "/tmp/target_edges.csv");
```

`edge_projection.mtx` is MatrixMarket coordinate format.  The two CSV files map
matrix row and column numbers back to `(polygon_handle, edge_handle,
local_edge_index)`.

The planar rectangular test exercises this on a nonmatching mesh.  The assembled
projection has 36 directed target-edge rows, 16 directed source-edge columns,
and 80 nonzeros.  The spherical structured and unstructured tests also assemble
`P` and require `P * U_source` to match direct edge transfer within `5.0e-13`.

## Spherical Structured And Unstructured Drivers

The files

```text
tests/spherical_quad_test.cpp
tests/spherical_voronoi_test.cpp
```

build spherical source and target meshes on the unit sphere, set
`GeometryMode::SphericalGnomonic`, and exercise conservative edge transfer in
source-cell charts.

The analytical field in that driver is the surface gradient of the zonal
spherical harmonic

```text
f(x,y,z) = 0.5 * (3 z^2 - 1),
```

with tangent vector field

```text
u = grad_S f = grad_3D f - (grad_3D f . r) r.
```

Source and diagnostic target edge fluxes are evaluated with 16-point
Gauss-Legendre quadrature in the local gnomonic chart using the Piola pullback.
The tests first enforce conservative antisymmetry on shared manufactured source
edges, then verify direct transfer, target-cell reconstruction, and sparse
matrix application.

### Running The Cubed-Sphere Driver

```bash
./build/spherical_quad_test
./build/spherical_quad_test source_n target_n output_prefix
```

Examples:

```bash
./build/spherical_quad_test
./build/spherical_quad_test 4 6 run_4_6
./build/spherical_quad_test 8 12 run_8_12
```

With no arguments, the test runs four acceptance cases: `4 -> 4` identity,
`4 -> 6` refinement, `6 -> 4` coarsening, and `6 -> 8` refinement convergence.
With arguments, `source_n` and `target_n` are the number of elements per
cubed-sphere face side for the source and target meshes.

The unstructured spherical test is run through CTest:

```bash
ctest --test-dir build -R spherical_voronoi_test --output-on-failure
```

It covers deterministic n-sided gnomonic Voronoi patch transfers,
Voronoi-to-structured patch transfer, and structured-to-Voronoi patch transfer.

### Spherical VTK Outputs

The spherical driver writes:

```text
<prefix>_source.vtk
<prefix>_target.vtk
```

The source VTK file contains cell data:

- `SOURCE_DIV_EXACT`

The target VTK file contains cell data:

- `TARGET_DIV_RECON`
- `TARGET_FLUX_ERROR`
- `TARGET_FIELD_RECON`
- `TARGET_FIELD_EXACT`
- `TARGET_FIELD_ERROR_NORM`

### Spherical Error Diagnostics

The spherical driver reports two distinct diagnostic families so transfer and
reconstruction quality can be separated.

1. **Edge-flux analytical errors on directed target edges**
   - `L2_rel`: relative norm of transferred integrated flux errors
   - `Linf`: maximum transferred integrated flux error

2. **Cell-centered target vector errors**
   - `L1`: area-weighted sum of centroid-sampled vector error magnitudes
   - `Linf`: maximum centroid-sampled vector error magnitude

These diagnostics are intentionally separate from conservation checks.  The
spherical tests fail if global conservation, direct-vs-sparse agreement, or KKT
edge re-integration exceeds `5.0e-13`.  Smooth-field edge errors are expected to
decrease under refinement.

## Directed Edge Convention

The prototype treats edge DOFs as **directed cell-local edge DOFs**.  This is
intentional: normal flux orientation belongs to a cell edge, not to an
orientation-free MOAB edge handle.  The library stores directed source and
target values internally by `(polygon, local_edge_index)` and mirrors values to
MOAB tags for diagnostics.  Production workflows that need unique shared-edge
DOFs should add an explicit orientation/sign map before collapsing the directed
operator.

## Build And Test

```bash
cmake -S . -B /tmp/mimetic-sphpoly-build
cmake --build /tmp/mimetic-sphpoly-build --parallel
ctest --test-dir /tmp/mimetic-sphpoly-build --output-on-failure
```

Current tests:

- `patch_test`: recovers a constant conservative field and target edge fluxes.
- `conservative_intersection_test`: checks rectangular overlap conservation,
  directed target edge transfer, target-cell divergence from edge fluxes, sparse
  matrix application, and MatrixMarket output.
- `voronoi_intersection_test`: checks n-sided Voronoi overlap conservation and
  directed target edge transfer on 71 target edge DOFs.
- `convergence_validation_test`: checks h-refinement behavior and conservation on
  the planar manufactured solution.
- `spherical_geometry_test`: checks spherical area, great-circle edge length,
  gnomonic round trip, straight-line projection of great circles, and KKT source
  edge flux re-integration.
- `spherical_quad_test`: checks cubed-sphere identity, refinement, coarsening,
  sparse projection agreement, and edge-error convergence.
- `spherical_voronoi_test`: checks deterministic n-sided spherical Voronoi patch
  transfers, Voronoi-to-structured patch transfer, structured-to-Voronoi patch
  transfer, sparse projection agreement, and edge-error convergence.

Representative spherical CTest output:

```text
spherical_quad_test:
  max global residual        1.01e-14
  max direct-sparse delta    2.78e-16
  4->6 L2_rel edge error     6.53e-02
  6->8 L2_rel edge error     3.06e-02

spherical_voronoi_test:
  max global residual        2.66e-15
  max direct-sparse delta    3.05e-13
  coarse Voronoi L2_rel      1.11e-02
  fine Voronoi L2_rel        5.07e-03
```

## Manuscript And Figures

The manuscript source is:

```text
docs/mimetic_voronoi_report.tex
```

Figures are generated by:

```bash
env MPLCONFIGDIR=/tmp/mimetic-sphpoly-mpl python3 docs/make_report_figures.py
```

Then compile the manuscript with:

```bash
cd docs
pdflatex mimetic_voronoi_report.tex
bibtex mimetic_voronoi_report
pdflatex mimetic_voronoi_report.tex
pdflatex mimetic_voronoi_report.tex
```

## Current Limitations

- Spherical v1 assumes unit-sphere, convex, great-circle cells that fit in each
  source-cell gnomonic chart hemisphere.
- Candidate search is deterministic all-pairs.  Production code should use MOAB
  spatial acceleration or an advancing-front overlap traversal without changing
  candidate ordering semantics.
- The sparse matrix stores directed cell-edge DOFs, not globally collapsed unique
  edge DOFs.
- No MPI, distributed ownership, ghosting, or parallel I/O is implemented in
  this phase.
