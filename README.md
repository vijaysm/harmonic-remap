# Mimetic-SphPoly

Prototype C++14 implementation of a conservative mimetic polygon interpolation
kernel using MOAB for mesh entities/tags and Eigen3 for dense and sparse linear
algebra.

The repository now contains both a verified planar interpolation kernel and an
experimental spherical cubed-sphere driver.  The planar path is the validated
reference implementation; the spherical path currently exercises local gnomonic
polygon projection, conservative edge transfer, and reconstruction diagnostics on
unit-sphere quad meshes.

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

There are now two distinct transfer/verification paths:

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

The current planar tests exercise this on the rectangular nonmatching mesh.  The
assembled projection has 36 directed target-edge rows, 16 directed source-edge
columns, and 80 nonzeros.

## Experimental Spherical Cubed-Sphere Driver

The file

```text
tests/spherical_quad_test.cpp
```

builds a source and target cubed-sphere mesh on the unit sphere, sets spherical
mode on `MimeticInterpolator`, and uses local gnomonic tangent-plane projections
per polygon so the existing 2D clipping and reconstruction logic can be reused.

The analytical field in that driver is the surface gradient of the zonal
spherical harmonic

```text
f(x,y,z) = 0.5 * (3 z^2 - 1),
```

with tangent vector field

```text
u = grad_S f = grad_3D f - (grad_3D f . r) r.
```

Source edge fluxes are integrated on great-circle arcs with 7-point
Gauss-Legendre quadrature.  Target edge fluxes are then transferred
conservatively from the source mesh and used to reconstruct a target field.

### Running The Spherical Driver

```bash
./build/spherical_quad_test [source_n] [target_n] [output_prefix]
```

Examples:

```bash
./build/spherical_quad_test
./build/spherical_quad_test 4 6 run_4_6
./build/spherical_quad_test 8 12 run_8_12
```

Here `source_n` and `target_n` are the number of elements per cubed-sphere face
side for the source and target meshes.

### Spherical VTK Outputs

The spherical driver writes:

```text
<prefix>_source.vtk
<prefix>_target.vtk
```

The source VTK file contains cell data:

- `SOURCE_DIV_EXACT`
- `SOURCE_FLUX_L1_EXACT`
- `SOURCE_FIELD_EXACT`

The target VTK file contains cell data:

- `TARGET_DIV_RECON`
- `TARGET_FLUX_L1_RECON`
- `TARGET_DIV_ERROR`
- `TARGET_FIELD_RECON`
- `TARGET_FIELD_EXACT`
- `TARGET_FIELD_ERROR`
- `TARGET_FIELD_ERROR_NORM`

### Spherical Error Diagnostics

The spherical driver reports two distinct diagnostic families so transfer and
reconstruction quality can be separated.

1. **Edge-based analytical errors on target great-circle arcs**
   - `L1`: integral of pointwise normal-flux-density error along all target edges
   - `L2`: square-root of the integral of squared pointwise normal-flux-density
     error along all target edges
   - `Linf`: maximum sampled pointwise normal-flux-density error on quadrature
     points along target edges

2. **Cell-centered target vector errors**
   - `L1`: area-weighted sum of centroid-sampled vector error magnitudes
   - `Linf`: maximum centroid-sampled vector error magnitude

These diagnostics are intentionally separate from the global conservation check.
It is therefore possible for the spherical path to remain globally conservative
while still exhibiting large edge or centroid reconstruction errors.

## Directed Edge Convention

The prototype treats edge DOFs as **directed cell-local edge DOFs**.  This is
intentional: normal flux orientation belongs to a cell edge, not to an
orientation-free MOAB edge handle.

The current test meshes create independent polygon edges per cell, so the MOAB
tag storage is unambiguous.  For production meshes with shared edges, add an
explicit orientation/sign map before collapsing directed cell-edge DOFs onto
unique mesh edges.

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
- `spherical_quad_test`: exercises the experimental spherical cubed-sphere path,
  conservative edge transfer, VTK export, and analytical edge/cell error
  diagnostics.

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

- The planar path is the validated reference implementation; the spherical path
  is still experimental.
- Spherical transfer currently conserves global flux to machine precision on the
  cubed-sphere test, but the analytical edge and centroid error norms remain
  large, so spherical reconstruction accuracy is not yet acceptable.
- The spherical implementation currently relies on local gnomonic projections on
  relatively small cubed-sphere quads; more general spherical polygon cases have
  not yet been validated.
- Spatial search is currently all-pairs in the tests; production code should use
  MOAB acceleration or an advancing-front overlap traversal.
- Shared mesh-edge orientation handling is not implemented yet.
- The sparse matrix currently stores directed cell-edge DOFs, not globally
  collapsed unique edge DOFs.
