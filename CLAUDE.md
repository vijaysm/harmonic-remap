# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Conservative mimetic polygon interpolation kernel for nonmatching mesh remapping. Reconstructs vector fields from directed edge fluxes using a harmonic basis (Perot-Chartrand level-2 method) and transfers them edge-to-edge between overlapping polygonal meshes. Supports both planar and spherical (gnomonic chart) geometry on quads and Voronoi cells, with higher-order edge-moment transfer at p=1,2,3.

**Dependencies:** C++14, MOAB (mesh entities/tags), Eigen3 (dense/sparse linear algebra)

## Build and Test

```bash
# Configure and build (MOAB searched in /opt/moab/ hierarchy)
cmake -S . -B build
cmake --build build --parallel

# Run all tests (13 CTest targets)
ctest --test-dir build --output-on-failure

# Run a single test
ctest --test-dir build -R <test_name> --output-on-failure

# Example: run only the high-order convergence study
ctest --test-dir build -R high_order_hdiv_convergence_test --output-on-failure
```

### Convergence Studies

```bash
# Planar high-order (writes CSV, then plot)
./build/high_order_hdiv_convergence_test docs/high_order_hdiv_convergence.csv
conda run -n climate-vis python scripts/plot_high_order_hdiv_convergence.py \
  docs/high_order_hdiv_convergence.csv docs/figures/high_order_hdiv_convergence.png

# Spherical high-order
./build/spherical_high_order_hdiv_convergence_test docs/spherical_high_order_hdiv_convergence.csv
conda run -n climate-vis python scripts/plot_spherical_high_order_hdiv_convergence.py \
  docs/spherical_high_order_hdiv_convergence.csv docs/figures/spherical_high_order_hdiv_convergence.png

# Low-order spherical convergence sweep
bash scripts/convergence_study.sh build > /tmp/convergence.csv
conda run -n climate-vis python scripts/plot_convergence.py /tmp/convergence.csv output.png
```

### Spherical Driver with VTK Output

```bash
./build/spherical_quad_test                     # 4 acceptance cases
./build/spherical_quad_test 4 6 run_4_6         # custom source_n, target_n, output prefix
# Produces: run_4_6_source.vtk, run_4_6_target.vtk
```

### Manuscript

```bash
env MPLCONFIGDIR=/tmp/mimetic-sphpoly-mpl python3 docs/make_report_figures.py
cd docs && pdflatex mimetic_voronoi_report.tex && bibtex mimetic_voronoi_report \
  && pdflatex mimetic_voronoi_report.tex && pdflatex mimetic_voronoi_report.tex
```

## Architecture

### Core Library (single translation unit)

- **`include/mimetic/mimetic.hpp`** -- Public API: geometry options, data types, `MimeticInterpolator` (low-order), `PlanarMomentInterpolator` (high-order p=1,2,3)
- **`src/mimetic.cpp`** -- All numerical kernels in one file (~2900 lines)

### Algorithm Pipeline

**Low-order (level-2 mimetic):**
1. `local_polygon()` / `spherical_polygon()` -- build centroid-relative chart geometry
2. `local_edges()` -- ordered directed edges with outward normals
3. `reconstruct_source_polygon()` -- KKT saddle-point solve: constant divergence d + harmonic coefficients [a_k, b_k] from edge fluxes
4. `transfer_source_to_target_edges()` -- clip each target edge against source cells, evaluate reconstruction on subsegments
5. `project_target_fluxes_to_hdiv_conforming()` -- collapse directed edges to unique edges, constrained least-squares with divergence constraints

**High-order (split-basis moment enrichment):**
1. `PlanarMomentInterpolator::set_source_edge_moments()` -- Legendre edge moments m=0..p
2. `reconstruct_source_polygon()` -- unified split basis (polynomial-divergence + harmonic-gradient + bubble modes), constrained LS with m=0 hard constraint
3. `transfer_source_to_target_edge_moments()` -- clipping-based moment transfer
4. `project_target_edge_moments_to_hdiv_conforming()` -- global skeleton solve on moments

**Spherical geometry path:**
- Gnomonic projection maps great circles to straight lines -> reuse planar clipping
- Piola contravariant mapping preserves normal fluxes between chart and surface
- Source-cell charts with `GnomonicFrame` (center, tangent basis e_x/e_y)
- Optional Hodge metric weighting in Gram matrix (`metric_weighted` flag)

### Key Design Decisions

- **Directed edge DOFs:** Edge fluxes are cell-local signed quantities (polygon, local_edge_index), not global unique edge scalars. Shared mesh edges appear twice with opposite signs.
- **Conservation tolerance:** `5.0e-13` absolute. All tests enforce this for flux balance, direct-vs-sparse agreement, and KKT re-integration.
- **Basis conditioning:** Coordinates scaled by `local_length_scale(poly)` -- controls conditioning on refined/irregular cells at all polynomial orders.
- **Solver selection:** Under-determined -> KKT minimum-energy; fully/over-determined -> QR; constrained LS mode -> exact m=0, weighted soft higher moments.

### Test Organization

Tests in `tests/` are standalone C++ executables (no test framework, use assertions and tolerance checks):

| Category | Tests |
|----------|-------|
| Planar low-order | `patch_test`, `conservative_intersection_test`, `voronoi_intersection_test`, `convergence_validation_test` |
| Planar high-order | `high_order_edge_moment_test`, `high_order_hdiv_convergence_test`, `hdiv_conforming_projection_test` |
| Spherical low-order | `spherical_geometry_test`, `spherical_quad_test`, `spherical_voronoi_test`, `spherical_scalar_test` |
| Spherical high-order | `spherical_high_order_moment_test`, `spherical_high_order_hdiv_convergence_test` |
| Cell average | `cell_average_transfer_test` |

Shared utilities: `test_utils.hpp` (manufactured fields, flux integration), `spherical_transfer_test_utils.hpp` (cubed-sphere generators, Y_2^0 harmonic field, 16-point Gauss quadrature).

### Manufactured Test Fields

| Field | Expression | Divergence | Used in |
|-------|-----------|------------|---------|
| A (harmonic exact) | (1+2x+2y, 1-2y+2x) | 0 | convergence, high-order |
| B (sincos divfree) | (sin(pi*x)cos(pi*y), -cos(pi*x)sin(pi*y)) | 0 | convergence, high-order |
| C (variable div) | (x^2, -y^2) | 2x-2y | convergence, high-order |
| D (exponential) | (exp(x)sin(y), exp(x)cos(y)) | 0 | convergence, high-order |
| Spherical | grad_S(Y_2^0) = grad_S(0.5*(3z^2-1)) | computed | spherical tests |

## Observed Convergence Rates

| Domain | p=1 | p=2 | p=3 |
|--------|-----|-----|-----|
| Planar quad-to-quad | 2.26 | 3.39 | 4.46 |
| Planar Voronoi-to-Voronoi | 1.90 | 2.77 | 3.34 |
| Spherical cubed-sphere (moment-0) | 2.49 | 3.80 | 2.42 |
| Spherical Voronoi-patch (moment-0) | 2.08 | 2.66 | 3.31 |

## Technical Documentation

- **Manuscript:** `docs/mimetic_voronoi_report.tex` (full background, method, algorithms, validation)
- **Code traceability:** `docs/code_documentation.md` (manuscript-to-code map, directed edge convention, reconstruction/transfer/conforming paths)
- **Convergence data:** `docs/high_order_hdiv_convergence.csv`, `docs/spherical_high_order_hdiv_convergence.csv`

## Working Conventions

- All geometry changes must preserve conservation residuals below `5.0e-13`
- Direct transfer and sparse projection operator must agree within `5.0e-13`
- New test fields should be added to `test_utils.hpp` or `spherical_transfer_test_utils.hpp`
- Adding a new test: add executable and `add_test()` in `CMakeLists.txt`, link against `mimetic`, `${MOAB_LIBRARIES}`, `Eigen3::Eigen`
- VTK diagnostic arrays on target cells follow the `TARGET_*` naming convention
