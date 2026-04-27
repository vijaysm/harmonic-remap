# Mimetic-SphPoly Quick Reference

## What is this project?

**Conservative mimetic polygon interpolation kernel** for nonmatching mesh remapping. Reconstructs vector fields from directed edge normal fluxes and transfers them between overlapping polygonal meshes (planar quads/Voronoi or spherical cubed-sphere/Voronoi).

**Math:** Perot-Chartrand level-2 harmonic basis reconstruction + H(div)-conforming higher-order moments (p=1,2,3)

**Purpose:** Climate/atmospheric modeling remapping (MOAB-based mesh framework)

---

## Quick Start

```bash
# Build
cmake -S . -B build
cmake --build build --parallel

# Run all tests (13 CTest targets)
ctest --test-dir build --output-on-failure

# Run specific test with output
ctest --test-dir build -R spherical_quad_test --verbose

# Generate VTK output for cubed-sphere transfer
./build/spherical_quad_test 4 6 my_run
# Produces: my_run_source.vtk, my_run_target.vtk

# High-order convergence study
./build/high_order_hdiv_convergence_test output.csv
python3 scripts/plot_high_order_hdiv_convergence.py output.csv figure.png
```

---

## Project Structure

| Path | Purpose |
|------|---------|
| `include/mimetic/mimetic.hpp` | **Public API** (634 lines): classes, data types, function declarations |
| `src/mimetic.cpp` | **All numerics** (3680 lines): geometry, quadrature, reconstruction, transfer, sparse operators |
| `tests/` | **13 test executables** (planar low/high-order, spherical low/high-order, utilities) |
| `scripts/` | Convergence study runners and plotters |
| `docs/` | Manuscript (LaTeX), convergence data (CSV), code traceability notes |

---

## Core Components

### Low-Order Reconstruction (`MimeticInterpolator`)

Builds a level-2 field on each source cell:
```
u_h(x) = (d/2)x + a₁ grad P₁ + b₁ grad Q₁ + a₂ grad P₂ + b₂ grad Q₂
```

**Key methods:**
- `reconstruct_source_polygon()` – KKT solve for [d, a₁, b₁, a₂, b₂]
- `transfer_source_to_target_edges()` – Clip target edges, accumulate flux
- `assemble_edge_projection_operator()` – Sparse matrix U_t = P U_s
- `project_target_fluxes_to_hdiv_conforming()` – Global edge collapse + constrained LS

### High-Order Moments (`PlanarMomentInterpolator`)

Reconstructs from Legendre edge moments and optional cell moments:
- **Basis:** polynomial divergence + harmonic gradients + divergence-free bubbles
- **Constraints:** m₀ edge flux exact, higher moments soft-weighted
- **p=1,2,3:** Convergence rates 1.9–4.5 (quad) and 1.8–3.6 (Voronoi)

**Key methods:**
- `set_source_edge_moments()` – Store per-edge Legendre moments
- `reconstruct_source_polygon()` – Split basis LS solve
- `transfer_source_to_target_edge_moments()` – Clipping-based moment transfer

### Spherical Gnomonic Path

All reconstruction/transfer done in source-cell tangent-plane charts:
- `spherical_polygon()` – Normalize 3D vertices, construct tangent frame
- `project_gnomonic()` / `inverse_gnomonic()` – Tangent plane maps
- **Key insight:** Great circles → straight lines under gnomonic projection = planar clipping works!
- Piola contravariant flux mapping preserves conservation

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Directed edge DOFs** | Flux belongs to oriented edge, not orientation-free handle. Shared edges appear twice with opposite signs. |
| **Single translation unit** | All numerics in one ~3700-line file for clarity and optimization. |
| **Centroid-relative coordinates** | Simplifies geometric algorithms; extends naturally to spherical via gnomonic charts. |
| **Basis scaling by length scale** | Keeps polynomial degree p=1,2,3 well-conditioned on refined/irregular meshes. |
| **KKT minimum-energy solve** | Under-determined systems → mimetic philosophy of minimal energy. |
| **5.0e-13 conservation tolerance** | Enforced globally; achievable with double precision on deterministic planar code. |

---

## Convergence Rates

### Planar (edge flux error)
- **Quad-to-quad:** p=1: 2.26, p=2: 3.39, p=3: 4.46
- **Voronoi-to-Voronoi:** p=1: 1.89, p=2: 2.77, p=3: 3.34

### Spherical (moment-0 edge)
- **Cubed-sphere:** p=1: 1.75, p=2: 3.08, p=3: 4.57
- **Voronoi-patch:** p=1: 2.05, p=2: 2.52, p=3: 3.62

**Conservation residuals:** Roundoff (1e-15 to 1e-16) ✅

---

## Test Categories

| Category | Tests | Purpose |
|----------|-------|---------|
| **Planar Low-Order** | patch, conservative_intersection, voronoi_intersection, convergence_validation | Constant field, rectangular overlap, n-gons, h-refinement |
| **Planar High-Order** | high_order_edge_moment, high_order_hdiv_convergence, hdiv_conforming_projection | Exact recovery, p=1,2,3 convergence, global edge solve |
| **Spherical Low-Order** | spherical_geometry, spherical_quad, spherical_voronoi, spherical_scalar | Primitives, cubed-sphere, Voronoi, scalar conservation |
| **Spherical High-Order** | spherical_high_order_moment, spherical_high_order_hdiv_convergence | Regression, p=1,2,3 structured + Voronoi |

**All tests enforce:** `5.0e-13` absolute tolerance on conservation, direct-vs-sparse agreement, KKT re-integration.

---

## Algorithm Pipeline (Low-Order Example)

```
Input: Source cells with directed edge fluxes U_f

1. Geometry
   local_polygon(mesh, cell) → [LocalPolygon: centroid, points, area, frame]
   local_edges(polygon) → [LocalEdge: directed edges, outward normals]

2. Reconstruction (Per Source Cell)
   d = Σ U_f / area
   Assemble Gram matrix V for harmonics P_k, Q_k
   Solve KKT: [a₁, b₁, a₂, b₂] = V⁻¹ × moments
   Store [d, a₁, b₁, a₂, b₂]

3. Direct Transfer (Per Target Edge)
   For each target edge e_t:
     For each source cell K_s:
       project(e_t) → e'_t in source chart
       clip(e'_t) ∩ polygon(K_s) → segments
       ∫ u_h^s · n_t ds on each segment
       accumulate into U_t[e_t]

4. Global Conforming Projection (Optional)
   Collapse directed → unique edges
   Assemble exact divergence constraints
   Solve constrained LS:
     min ‖unique - directed‖²
     s.t. div = exact per cell
   Output: globally valid fluxes

Output: Target edge fluxes [U_t] with conservation ✅
```

---

## Dependencies

| Library | Role |
|---------|------|
| **Eigen3** (3.3+) | Dense/sparse matrices, solvers (LDLT, QR) |
| **MOAB** | Mesh topology, entity handles, tag I/O |
| **nanoflann** | k-d tree for candidate acceleration |
| **C++14** | Standard library algorithms, complex numbers |

---

## Important Files

### Must Read
- **README.md** – Full documentation with mathematical background
- **CLAUDE.md** – Guidance for working with code
- **EXPLORATION.md** – Comprehensive 13-section walkthrough (this repo)
- **ARCHITECTURE.md** – Detailed architecture diagrams and data flows

### Core Implementation
- **include/mimetic/mimetic.hpp** – Public API
- **src/mimetic.cpp** – All algorithms
- **tests/test_utils.hpp**, **spherical_transfer_test_utils.hpp** – Test helpers

### Manuscript & Data
- **docs/mimetic_voronoi_report.tex/pdf** – Full technical document
- **docs/code_documentation.md** – Code-to-manuscript traceability
- **docs/high_order_hdiv_convergence.csv** – Planar convergence data
- **docs/spherical_high_order_hdiv_convergence.csv** – Spherical data

---

## Common Tasks

### Run a Single Test
```bash
ctest --test-dir build -R patch_test --verbose
```

### Debug a Test
```bash
gdb ./build/patch_test
# or
./build/patch_test 2>&1 | head -50
```

### Generate Convergence Plots
```bash
./build/high_order_hdiv_convergence_test output.csv
python3 scripts/plot_high_order_hdiv_convergence.py output.csv output.png
```

### Add a New Test
1. Create `tests/my_test.cpp`
2. Add to `CMakeLists.txt`:
   ```cmake
   add_executable(my_test tests/my_test.cpp)
   target_link_libraries(my_test PRIVATE mimetic)
   add_test(NAME my_test COMMAND my_test)
   ```
3. Rebuild: `cmake --build build`

### Check Code Spans
- **Header:** `wc -l include/mimetic/mimetic.hpp` → 634 lines
- **Implementation:** `wc -l src/mimetic.cpp` → 3680 lines
- **Total core:** ~4300 lines (excluding tests/docs)

---

## Known Limitations

1. **Spherical:** Unit-sphere, convex great-circle cells, gnomonic hemisphere fit
2. **Spatial search:** All-pairs for small meshes; nanoflann for acceleration
3. **Directed DOFs:** Not collapsed to global edges (production code must handle)
4. **Parallelism:** Optional OpenMP on one loop only; no MPI/distributed I/O
5. **High-order:** Research kernel, not full polygonal VEM or Whitney space

---

## Directory Tree

```
Mimetic-SphPoly/
├── README.md (full docs)
├── CLAUDE.md (working conventions)
├── EXPLORATION.md ⭐ (comprehensive guide)
├── ARCHITECTURE.md ⭐ (detailed diagrams)
├── QUICK_REFERENCE.md ⭐ (this file)
├── CMakeLists.txt
├── include/mimetic/mimetic.hpp (634 lines)
├── src/mimetic.cpp (3680 lines)
├── tests/
│   ├── *_test.cpp (13 tests)
│   ├── test_utils.hpp
│   └── spherical_transfer_test_utils.hpp
├── scripts/ (convergence, plotting, helpers)
├── docs/
│   ├── mimetic_voronoi_report.tex/pdf
│   ├── code_documentation.md
│   ├── *.csv (convergence data)
│   └── figures/
└── build/ (artifacts)
```

---

## Performance Characteristics

- **Small meshes (<50 cells):** All-pairs candidate search
- **Medium meshes:** nanoflann k-d tree acceleration
- **Large meshes:** Still deterministic all-pairs (noted as future work)
- **Solver:** LDLT for 4×4 Gram matrix (low-order), QR/LS for high-order
- **Throughput:** Dominated by edge clipping and quadrature
- **Memory:** Sparse matrix stored in row-major CSR format

---

## Recent Commits

```
6accc20 Skip cell moment constraints when cell_weight=0 in reconstruction
34246b1 Study 12: complete resolution confirmed — Voronoi p=3 achieves expected convergence rate
5c81dbe Study 11 + critical bug fix: degree elevation on backward leg causes Voronoi p=3 failure
fb512b7 Add Study 10: round-trip convergence rates confirm fix enables correct behavior
aaa4a0d Fix CS p=3 round-trip: optimal CS resolution with 3:1 source/intermediate ratio
```

Latest focus: **p=3 round-trip convergence studies** and **cell moment constraint handling**

---

## Getting Help

1. **README.md** → Full mathematical background and algorithm overview
2. **code_documentation.md** → Algorithm paths, solver details, directed edge convention
3. **ARCHITECTURE.md** → Function signatures, data structures, module organization
4. **EXPLORATION.md** → Comprehensive section-by-section walkthrough
5. **Git log** → Recent changes and bug fixes

---

Generated: 2024 | Comprehensive exploration completed by AI assistant
