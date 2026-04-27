# Mimetic-SphPoly Repository - Comprehensive Exploration

## 1. PROJECT OVERVIEW

### What It Does
**Mimetic-SphPoly** is a C++14 prototype implementation of a conservative mimetic polygon interpolation kernel for nonmatching mesh remapping. It reconstructs vector fields from directed edge fluxes and transfers them between overlapping polygonal meshes.

**Key Purpose:** Conservative edge-to-edge transfer of vector fields on non-conforming meshes (planar quads/polygons and spherical cubed-sphere/Voronoi cells).

### Scientific Domain
- **Field:** Scientific computing / computational fluid dynamics
- **Mathematics:** Mimetic finite elements, H(div)-conforming discretizations, harmonic basis methods
- **Applications:** Climate modeling, atmospheric simulations (MOAB is used in climate codes)

### Core Algorithm
Uses a **Perot-Chartrand level-2 harmonic reconstruction** that builds a field from:
```
u_h(x) = (d/2) x + a1 grad P1 + b1 grad Q1 + a2 grad P2 + b2 grad Q2
```
Where:
- `d` = constant cell divergence (from conservation)
- `P_k, Q_k` = harmonic basis functions (real/imaginary parts of (x+iy)^k)
- `[a_k, b_k]` = harmonic coefficients computed from edge flux moments

---

## 2. DIRECTORY STRUCTURE & ORGANIZATION

```
Mimetic-SphPoly/
├── CMakeLists.txt                    # CMake build configuration
├── README.md                         # Project documentation
├── CLAUDE.md                         # Claude Code guidelines
├── include/
│   └── mimetic/
│       └── mimetic.hpp               # PUBLIC API (634 lines)
├── src/
│   └── mimetic.cpp                   # Core numerics (3680 lines)
├── tests/
│   ├── patch_test.cpp                # Basic constant field patch test
│   ├── conservative_intersection_test.cpp   # Rectangular overlap conservation
│   ├── voronoi_intersection_test.cpp       # N-sided Voronoi test
│   ├── convergence_validation_test.cpp     # Planar h-refinement study
│   ├── spherical_geometry_test.cpp         # Unit-sphere geometry primitives
│   ├── spherical_quad_test.cpp            # Cubed-sphere transfer driver (VTK output)
│   ├── spherical_voronoi_test.cpp         # Spherical Voronoi transfer
│   ├── spherical_scalar_test.cpp          # Scalar overlap conservation
│   ├── cell_average_transfer_test.cpp     # Cell-average transfer mode
│   ├── hdiv_conforming_projection_test.cpp # Global edge conforming solver
│   ├── high_order_edge_moment_test.cpp    # Planar higher-order patch tests (p=1,2,3)
│   ├── high_order_hdiv_convergence_test.cpp # Planar p=1,2,3 convergence study
│   ├── spherical_high_order_moment_test.cpp # Spherical high-order regression
│   ├── spherical_high_order_hdiv_convergence_test.cpp # Spherical p=1,2,3 study
│   ├── dump_visuals.cpp              # Visualization helper
│   ├── roundtrip_analysis.cpp        # Round-trip diagnostics
│   ├── p3_roundtrip_study.cpp        # P=3 round-trip convergence
│   ├── test_utils.hpp                # Shared planar test utilities
│   └── spherical_transfer_test_utils.hpp # Shared spherical utilities
├── scripts/
│   ├── convergence_study.sh          # Low-order spherical sweep
│   ├── plot_convergence.py           # Plot convergence data
│   ├── run_high_order_hdiv_convergence.sh
│   ├── plot_high_order_hdiv_convergence.py
│   ├── run_spherical_high_order_hdiv_convergence.sh
│   ├── plot_spherical_high_order_hdiv_convergence.py
│   ├── check_openmp_acceptance.sh
│   ├── run_mercator_roundtrip_parallel.sh
│   └── visualize.py
├── docs/
│   ├── mimetic_voronoi_report.tex    # Full technical manuscript
│   ├── mimetic_voronoi_report.pdf    # Compiled report
│   ├── code_documentation.md         # Code-facing traceability notes
│   ├── references.bib                # BibTeX bibliography
│   ├── make_report_figures.py        # Figure generation script
│   ├── high_order_hdiv_convergence.csv     # Planar convergence data
│   ├── spherical_high_order_hdiv_convergence.csv # Spherical data
│   └── figures/                      # Generated plots
└── build/*, build-omp/*, build-omp-opt/  # Build artifacts
```

**Key: Single-file library design**
- **Header-only public API:** `include/mimetic/mimetic.hpp`
- **All implementation:** `src/mimetic.cpp` (monolithic ~3700 lines)
- **Modular tests:** Each test is standalone executable, no test framework

---

## 3. KEY SOURCE FILES & THEIR ROLES

### Header: `include/mimetic/mimetic.hpp` (634 lines)

**Enums & Options:**
- `GeometryMode::Planar` / `GeometryMode::SphericalGnomonic` – geometry backend selector
- `GeometryOptions` – tolerances, radius, metric weighting, conservation threshold
- `MomentMethodOptions` – high-order parameters (edge order p, quadrature points, regularization)

**Data Structures:**

| Structure | Purpose |
|-----------|---------|
| `LocalPolygon` | Centroid-relative planar cell geometry + 3D coordinates + spherical frame |
| `LocalEdge` | Directed edge in local frame (start, end, outward_normal, length) |
| `SphericalPolygon` | 3D vertices, gnomonic chart projection, chart area, spherical area |
| `SphericalEdge` | 3D edge with gnomonic chart coordinates |
| `GnomonicFrame` | Tangent-space frame for spherical cells (center, e_x, e_y) |
| `ReconstructionCoeffs` | Reconstructed field coefficients (d + [a1,b1,a2,b2]) |
| `DirectedEdgeDof` | Cell-local directed edge identifier (polygon, edge, local_index) |
| `EdgeTransferResult` | Direct transfer output (target edges, fluxes, per-segment contributions) |
| `SparseEdgeProjection` | Sparse matrix U_t = P U_s mapping source to target edge fluxes |
| `ConformingEdgeTransferResult` | Global target-edge conforming projection result |
| `MomentReconstruction` | High-order reconstruction data (coeffs, basis modes, length scale) |
| `EdgeMomentTransferResult` | Higher-order moment transfer output |

**Key Functions (Geometry):**
- `local_polygon()` – Extract planar cell from MOAB mesh → centroid-relative coordinates
- `spherical_polygon()` – Extract 3D cell, normalize vertices, project to gnomonic chart
- `project_gnomonic()` / `inverse_gnomonic()` – Gnomonic tangent plane maps
- `gnomonic_jacobian()` / `gnomonic_area_scale()` – Differential geometry of gnomonic chart
- `lift_contravariant_piola()` / `pullback_contravariant_piola()` – Surface↔chart vector maps
- `local_edges()` – Build directed edge list from polygon vertices

**Key Functions (Quadrature):**
- `integrate_edge_scalar()` – 4-point Gauss-Legendre on edges (degree-7 exact)
- `integrate_triangle_scalar()` – 7-point rule on triangles (degree-4 exact)
- `integrate_triangle_highorder()` – 13-point rule (degree-7 exact, Dunavant)
- `integrate_triangle_adaptive()` – Recursive subdivision for thin triangles
- `integrate_polygon_adaptive()` – Fan-triangulate from centroid, adaptive quad

**Key Functions (Basis & Reconstruction):**
- `eval_harmonic_basis()` – Compute P_k, Q_k and gradients
- `clip_segment_to_convex_polygon()` – Sutherland-Hodgman-style clipping
- `signed_area()` / `polygon_centroid()` – Planar geometry utilities

**Main Classes:**

```cpp
class MimeticInterpolator {
  // Low-order (level-2) mimetic kernel
  void set_source_edge_flux(polygon, local_edge_index, flux);
  ReconstructionCoeffs reconstruct_source_polygon(polygon);
  Eigen::Vector2d velocity(coeffs, point);
  double line_integral(source_polygon, a, b);
  double edge_flux(coeffs, a, b);
  EdgeTransferResult transfer_source_to_target_edges(sources, targets);
  SparseEdgeProjection assemble_edge_projection_operator(sources, targets);
  ConformingEdgeTransferResult project_target_fluxes_to_hdiv_conforming(...);
};

class PlanarMomentInterpolator {
  // Higher-order (p=1,2,3) edge-moment kernel
  void set_source_edge_moments(polygon, local_index, moments_vector);
  void set_source_cell_vector_moments(polygon, moment_vector);
  MomentReconstruction reconstruct_source_polygon(polygon, options);
  EdgeMomentTransferResult transfer_source_to_target_edge_moments(sources, targets, order);
  ConformingEdgeMomentTransferResult project_target_edge_moments_to_hdiv_conforming(...);
};
```

---

### Implementation: `src/mimetic.cpp` (3680 lines)

**Structure:**
1. **Includes** – MOAB, Eigen, nanoflann, STL
2. **Anonymous namespace** – Internal helpers
3. **Geometry construction** – `local_polygon()`, `spherical_polygon()`, coordinate transforms
4. **Harmonic basis** – `eval_harmonic_basis()` and gradient kernels
5. **Quadrature rules** – All integration routines (edges, triangles, adaptive)
6. **Clipping** – Sutherland-Hodgman segment-to-polygon clipping
7. **KKT reconstruction** – Harmonic coefficient solve (low-order)
8. **Spatial indexing** – nanoflann k-d tree for candidate acceleration
9. **Direct transfer** – Edge clipping, source reconstruction evaluation
10. **Sparse operator** – Local reconstruction matrices, assembly
11. **Conforming projection** – Constrained least-squares edge collapse
12. **High-order moments** – Split basis modes, Legendre moment handling
13. **Spherical specifics** – Gnomonic charts, Piola mapping, great-circle handling
14. **MatrixMarket I/O** – Sparse matrix export

**Key Internal Functions:**
- `make_gnomonic_frame()` – Build tangent space for spherical cell
- `source_reconstruction_matrix()` – Linear map from edge fluxes → coefficients
- `local_length_scale()` – Basis scaling for numerical stability
- `split_basis_modes()` – Polynomial divergence + harmonic + bubble modes
- `evaluate_split_basis()` – Evaluate high-order basis at point
- `clipped_edge_transfer_contribution()` – Single source-target edge pair

---

## 4. BUILD SYSTEM (CMake)

**CMakeLists.txt Structure:**
```cmake
cmake_minimum_required(VERSION 3.10)
project(MimeticPatchTest LANGUAGES CXX)

# C++14 required
set(CMAKE_CXX_STANDARD 14)

# Find dependencies
find_package(Eigen3 3.3 REQUIRED)
find_library(MOAB_LIBRARY MOAB HINTS /opt/moab/...)  # MOAB from /opt/moab hierarchy

# Core library (SINGLE translation unit)
add_library(mimetic STATIC src/mimetic.cpp)
target_link_libraries(mimetic PUBLIC Eigen3::Eigen MOAB)
target_include_directories(mimetic PUBLIC include ${MOAB_INCLUDE_DIR} ${NANOFLANN_INCLUDE_DIR})

# Optional OpenMP (macOS-specific workaround for Clang)
if(MIMETIC_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
  target_link_libraries(mimetic PUBLIC OpenMP::OpenMP_CXX)
endif()

# 13 test executables
add_executable(patch_test tests/patch_test.cpp)
add_executable(conservative_intersection_test tests/conservative_intersection_test.cpp)
# ... etc
target_link_libraries(patch_test PRIVATE mimetic)

# Enable CTest
enable_testing()
add_test(NAME patch_test COMMAND patch_test)
```

**Build Invocation:**
```bash
cmake -S . -B build
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

**Dependencies:**
- **Eigen3** – Dense and sparse linear algebra (header-only)
- **MOAB** – Mesh entities, tags, adjacencies (library)
- **nanoflann** – k-d tree for spatial indexing (header-only)
- **C++ Standard Library** – Standard algorithms, complex numbers

---

## 5. TEST FILES & VALIDATION

### Test Matrix (13 CTest targets)

**Planar Low-Order (4 tests):**
1. **`patch_test`** – Constant field on unit square; verifies reconstruction recovers constant interpolant
2. **`conservative_intersection_test`** – 4×4 source quad overlapping 6×6 target grid; checks edge conservation, divergence from fluxes, sparse matrix agreement
3. **`voronoi_intersection_test`** – n-sided Voronoi polygon overlaps; 71 target edge DOFs, conservation verification
4. **`convergence_validation_test`** – h-refinement study on planar meshes; 4 manufactured test fields

**Planar High-Order (3 tests):**
5. **`high_order_edge_moment_test`** – Exact recovery of p=1,2 fields on nonmatching patch tests
6. **`high_order_hdiv_convergence_test`** – p=1,2,3 convergence on quad-to-quad and Voronoi-to-Voronoi; writes CSV for plotting
7. **`hdiv_conforming_projection_test`** – Global target-edge conforming skeleton solve

**Spherical Low-Order (4 tests):**
8. **`spherical_geometry_test`** – Gnomonic projection round-trip, great-circle length, area computation, KKT re-integration
9. **`spherical_quad_test`** – Cubed-sphere identity/refinement/coarsening; 4 acceptance cases; writes VTK output
10. **`spherical_voronoi_test`** – Spherical Voronoi-to-Voronoi transfer; sparse matrix agreement
11. **`spherical_scalar_test`** – Scalar overlap coverage, global conservation (geometry-only control remap)

**Spherical High-Order (2 tests):**
12. **`spherical_high_order_moment_test`** – Regression coverage for p=1,2,3 on structured/unstructured
13. **`spherical_high_order_hdiv_convergence_test`** – p=1,2,3 convergence study on cubed-sphere + Voronoi patches

**Other Tests:**
- `cell_average_transfer_test` – Alternative reduction mode (cell averages instead of edges)
- `dump_visuals`, `roundtrip_analysis`, `p3_roundtrip_study` – Diagnostic/research tools

### Conservation Tolerance
**All tests enforce:** `5.0e-13` absolute tolerance on:
- Global flux conservation (sum of target fluxes = source divergence × overlap area)
- Direct vs. sparse projection agreement
- KKT reconstruction re-integration (verify edge fluxes from reconstructed field match inputs)

### Manufactured Test Fields (in `test_utils.hpp`)

| Field | Expression | Divergence | Notes |
|-------|-----------|------------|-------|
| A (harmonic) | (1+2x+2y, 1-2y+2x) | 0 | Divergence-free |
| B (sincos) | (sin(πx)cos(πy), -cos(πx)sin(πy)) | 0 | Div-free |
| C (variable) | (x², -y²) | 2x - 2y | Nonzero div |
| D (exponential) | (eˣsin(y), eˣcos(y)) | 0 | Div-free |
| Spherical (Y₂⁰) | grad_S(½(3z² - 1)) | computed | Spherical harmonic |

---

## 6. DOCUMENTATION & RESOURCES

### Manuscript & Technical Notes
- **`docs/mimetic_voronoi_report.tex`** – Full technical manuscript (source, methods, validation, future work)
- **`docs/mimetic_voronoi_report.pdf`** – Compiled PDF with figures
- **`docs/code_documentation.md`** – Code-to-manuscript traceability, directed edge convention, algorithm paths, solver details
- **`docs/references.bib`** – Scientific references

### Data & Figures
- **`docs/high_order_hdiv_convergence.csv`** – Planar p=1,2,3 convergence data (quad and Voronoi meshes)
- **`docs/spherical_high_order_hdiv_convergence.csv`** – Spherical p=1,2,3 data (structured + Voronoi)
- **`docs/figures/`** – Generated plots (convergence curves, error comparisons)

### Generation Scripts
```bash
# Planar high-order study
./build/high_order_hdiv_convergence_test docs/high_order_hdiv_convergence.csv
python3 scripts/plot_high_order_hdiv_convergence.py docs/high_order_hdiv_convergence.csv output.png

# Spherical high-order study
./build/spherical_high_order_hdiv_convergence_test docs/spherical_high_order_hdiv_convergence.csv
python3 scripts/plot_spherical_high_order_hdiv_convergence.py docs/spherical_high_order_hdiv_convergence.csv output.png

# Low-order convergence sweep
bash scripts/convergence_study.sh build > convergence.csv
python3 scripts/plot_convergence.py convergence.csv output.png

# Manuscript figures
python3 docs/make_report_figures.py
```

---

## 7. ALGORITHM PIPELINE

### Low-Order (Level-2) Path

**Input:** Directed edge flux DOFs `U_f = ∫_f u · n_f ds` on source cells

**Step 1: Reconstruct source cell**
```
local_polygon() → centroid-relative geometry
local_edges()   → directed edges with outward normals
reconstruct_source_polygon() → solve KKT system:
  - constant divergence d = Σ U_f / area
  - harmonic coefficients [a₁,b₁,a₂,b₂] from Gram matrix
  Result: u_h(x) = (d/2)x + a₁ grad P₁ + b₁ grad Q₁ + a₂ grad P₂ + b₂ grad Q₂
```

**Step 2: Transfer to target edges**
```
For each directed target edge e_t:
  For each source polygon Ks (candidate search):
    1. Project edge endpoints into source chart
    2. Clip projected edge against source polygon boundary
    3. Integrate reconstructed flux over clipped segment
    4. Accumulate contribution
Result: target edge fluxes U_t
```

**Step 3: Global conforming projection (optional)**
```
project_target_fluxes_to_hdiv_conforming():
  - Collapse directed edges to unique geometric edges
  - Assemble exact divergence constraints (one per target cell)
  - Solve constrained least-squares:
    minimize ‖unique_fluxes - raw_directed_fluxes‖²
    subject to: divergence constraints
  Result: ConformingEdgeTransferResult with unique edges
```

### High-Order (p=1,2,3) Path

**Input:** Legendre edge moments (0 through p) + optional cell vector moments

**Reconstruction:**
```
Split basis consisting of:
  1. Polynomial divergence modes (p+1 modes for order p)
  2. Harmonic gradient modes (divergence-free: ∇P_k, ∇Q_k)
  3. Divergence-free completion ("bubble") modes (interior)

Solve constrained least-squares:
  - Moment-0 edge flux: EXACT constraint (conservation)
  - Higher moments: soft constraints in weighted least-squares
  - Optional cell moments: soft interior constraints
  Basis scaled by local_length_scale() for stability on refined/irregular cells
```

**Transfer:**
```
For each directed target edge:
  Clip against each source polygon
  Evaluate Legendre moments of source reconstruction on clipped segment
  Accumulate into target edge moment vector
```

### Spherical Gnomonic Path

**Special handling:** All computations in source-cell tangent-plane charts

```
spherical_polygon() → normalize 3D vertices, construct GnomonicFrame
project_gnomonic() → flatten great-circle polygon to chart (straight edges!)
transfer_source_to_target_edges() → project target edge to each source chart
                                  → clip in chart
                                  → evaluate reconstruction in chart
lift_contravariant_piola() → surface diagnostic vectors from chart components
```

**Key insight:** Great circles → straight lines under gnomonic projection, so planar clipping still works!

---

## 8. TECHNICAL DESIGN DECISIONS

### Directed Cell-Local Edge DOFs
- Edge fluxes are signed quantities belonging to cell-local orientations
- Shared mesh edges appear **twice** with opposite signs (one per adjacent cell)
- **Reason:** Normal flux belongs to oriented edge, not orientation-free edge handle
- **Consequence:** Production code must explicitly collapse directed DOFs to unique edges
- **Implemented in:** `ConformingEdgeTransferResult` projection

### Coordinate Frames & Local Scaling
- All planar reconstruction: centroid-relative coordinates
- High-order basis: scaled by `local_length_scale()` (cell characteristic size)
- **Why:** Keeps polynomial degree p=1,2,3 well-conditioned on refined/irregular meshes

### Spatial Acceleration
- Uses nanoflann k-d tree for candidate source-cell search
- Falls back to all-pairs for meshes < 50 cells
- **Note:** "Deterministic all-pairs" mentioned as future optimization opportunity

### Solver Selection
- **Under-determined systems:** KKT minimum-energy fallback (mimetic philosophy)
- **Fully/over-determined:** QR solve
- **Constrained least-squares:** Moment-0 hard constraint, higher moments soft

### Conservation & Tolerance
- **Conservation tolerance:** `5.0e-13` (absolute) – enforced globally
- **Geometry tolerance:** `1.0e-13` – for degeneracy checks
- **Rationale:** Double precision, deterministic planar code, single-machine validation

---

## 9. CONVERGENCE PERFORMANCE

### Observed Rates

**Planar (quad-to-quad):**
- p=1: 2.26 (affine order ~2)
- p=2: 3.39 (quadratic order ~3)
- p=3: 4.46 (cubic order ~4)

**Planar (Voronoi-to-Voronoi):**
- p=1: 1.89 (affine order)
- p=2: 2.77 (quadratic order)
- p=3: 3.34 (cubic order)

**Spherical (cubed-sphere, moment-0 edge):**
- p=1: 1.75
- p=2: 3.08
- p=3: 4.57 (uses degree-elevated basis for Piola metric)

**Spherical (Voronoi-patch, moment-0):**
- p=1: 2.05
- p=2: 2.52
- p=3: 3.62

**Conservation residuals:** Roundoff (1e-15 to 1e-16)

---

## 10. KNOWN LIMITATIONS

1. **Spherical:** Unit-sphere assumption, convex great-circle cells, gnomonic hemisphere constraint
2. **Spatial search:** Deterministic all-pairs (slow for large meshes); nanoflann is fallback acceleration
3. **Directed DOFs:** Not collapsed to unique global edges (production code must add orientation map)
4. **Parallelism:** Optional OpenMP pragma on one loop; no MPI, ghosting, or distributed I/O
5. **Candidate search:** No advancing-front or spatial structure exploitation
6. **High-order:** Not a full polygonal VEM or generalized Whitney space (research kernel)

---

## 11. WORKING WITH THE CODE

### Building
```bash
cmake -S . -B build -DMIMETIC_ENABLE_OPENMP=ON  # Optional OpenMP
cmake --build build --parallel
```

### Running Tests
```bash
ctest --test-dir build --output-on-failure
ctest --test-dir build -R spherical_quad_test --verbose
./build/spherical_quad_test 4 6 my_run  # Custom args for VTK output
```

### Running Studies
```bash
./build/high_order_hdiv_convergence_test output.csv
python3 scripts/plot_high_order_hdiv_convergence.py output.csv figure.png
```

### Key Conventions
- **Edge DOFs:** (polygon_handle, local_edge_index)
- **Fluxes:** Signed normal flux (outward positive)
- **Coordinate frames:** Centroid-relative planar, gnomonic for spherical
- **Tolerance:** 5.0e-13 absolute (conservation, direct-vs-sparse agreement, KKT re-integration)

---

## 12. CODE STATISTICS

| File | Lines | Purpose |
|------|-------|---------|
| `mimetic.hpp` | 634 | Public API, data types, class declarations |
| `mimetic.cpp` | 3680 | All numerical kernels (monolithic) |
| `test_utils.hpp` | 82 | Planar test utilities |
| `spherical_transfer_test_utils.hpp` | ~400 | Spherical test utilities |
| 13 test files | 1000–2000 each | Individual test executables |
| **Total** | ~4300 core | (excluding tests, docs, build artifacts) |

---

## 13. KEY DEPENDENCIES & THEIR ROLES

| Library | Version | Role |
|---------|---------|------|
| **Eigen** | 3.3+ | Dense matrix ops, sparse matrix storage, linear solvers (LDLT, QR) |
| **MOAB** | from /opt/moab | Mesh topology, entity handles, tag storage/retrieval |
| **nanoflann** | header-only | k-d tree spatial indexing for candidate acceleration |
| **C++ Standard** | C++14 | Algorithms, complex numbers, STL containers |

---

## CONCLUSION

**Mimetic-SphPoly** is a well-engineered research prototype for conservative edge-to-edge vector field transfer on overlapping polygonal meshes. It demonstrates:

✅ **Mathematical rigor:** Perot-Chartrand level-2 reconstruction, exact harmonic basis, convergence studies  
✅ **Practical engineering:** MOAB integration, spatial indexing, sparse matrix assembly  
✅ **Comprehensive validation:** 13 test suites, manufactured fields, convergence rates, conservation checks  
✅ **Clear documentation:** Technical manuscript, code traceability, convergence data  
✅ **Extensibility:** Modular API, planar + spherical backends, low-order + high-order kernels  

**Primary use case:** Conservative remapping for climate/atmospheric modeling on cubed-sphere and Voronoi unstructured grids.

