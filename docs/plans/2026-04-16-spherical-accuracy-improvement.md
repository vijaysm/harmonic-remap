# Spherical Accuracy Improvement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Establish convergence baselines, fix diagnostic weighting, implement metric-weighted reconstruction, add scalar remap comparison, and update the technical report.

**Architecture:** Five phases executed sequentially: (1) multi-resolution convergence study to measure h-convergence rate, (2) fix L1 diagnostic to use spherical area, (3) add scalar remap comparison test, (4) implement metric-weighted Gram matrix and moment integrals in the reconstruction, (5) update the LaTeX technical report with all findings.

**Tech Stack:** C++14, MOAB, Eigen3, CMake, pdflatex/bibtex

---

## Baseline Numbers (Before Any Changes)

| Case | source_n | target_n | edge_flux_l2_rel | edge_flux_linf | cell_field_l1 | cell_field_linf |
|------|----------|----------|------------------|----------------|---------------|----------------|
| Identity | 4 | 4 | 2.42e-16 | 3.33e-16 | 1.42 | 0.144 |
| Refine | 4 | 6 | 6.53e-02 | 3.02e-02 | 1.35 | 0.175 |
| Coarsen | 6 | 4 | 3.37e-02 | 2.36e-02 | 1.74 | 0.181 |
| Fine | 6 | 8 | 3.06e-02 | 1.31e-02 | 0.695 | 0.095 |

All conservation residuals are < 1e-14. All direct-sparse deltas are < 5e-13.

---

## Phase 1: Multi-Resolution Convergence Study

### Task 1.1: Add a convergence study driver script

**Files:**
- Create: `scripts/convergence_study.sh`

**Step 1: Write the convergence study script**

```bash
#!/bin/bash
set -euo pipefail

BUILD_DIR="${1:-/tmp/mimetic-sphpoly-build}"
BIN="$BUILD_DIR/spherical_quad_test"

if [ ! -x "$BIN" ]; then
    echo "ERROR: $BIN not found. Build first." >&2
    exit 1
fi

echo "source_n,target_n,edge_flux_l2_rel,edge_flux_linf,cell_field_l1,cell_field_linf"

for src_n in 3 4 6 8 12 16; do
    tgt_n=$((src_n + 2))
    output=$("$BIN" "$src_n" "$tgt_n" "/tmp/conv_${src_n}_${tgt_n}" 2>&1)
    l2_rel=$(echo "$output" | grep edge_flux_l2_rel | sed 's/.*edge_flux_l2_rel=//;s/ .*//')
    linf=$(echo "$output" | grep edge_flux_linf | sed 's/.*edge_flux_linf=//;s/ .*//')
    l1=$(echo "$output" | grep cell_field_l1 | sed 's/.*cell_field_l1=//;s/ .*//')
    clinf=$(echo "$output" | grep cell_field_linf | sed 's/.*cell_field_linf=//;s/ .*//')
    echo "$src_n,$tgt_n,$l2_rel,$linf,$l1,$clinf"
done
```

**Step 2: Make it executable and run**

Run: `chmod +x scripts/convergence_study.sh && bash scripts/convergence_study.sh /tmp/mimetic-sphpoly-build`
Expected: CSV table with 6 rows of convergence data.

**Step 3: Commit**

```bash
git add scripts/convergence_study.sh
git commit -m "feat: add multi-resolution convergence study script"
```

### Task 1.2: Add a Python convergence plotting script

**Files:**
- Create: `scripts/plot_convergence.py`

**Step 1: Write the plotting script**

```python
#!/usr/bin/env python3
"""Plot h-convergence of spherical edge transfer errors.

Usage:
    bash scripts/convergence_study.sh > /tmp/convergence.csv
    python3 scripts/plot_convergence.py /tmp/convergence.csv
"""
import sys
import csv
import math

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plot_convergence.py <csv_file>", file=sys.stderr)
        sys.exit(1)

    rows = []
    with open(sys.argv[1]) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"{'source_n':>10} {'h (rad)':>10} {'L2_rel':>12} {'Linf':>12} {'L1_cell':>12} {'Linf_cell':>12}")
    print("-" * 70)

    prev_h = None
    prev_l2 = None
    for row in rows:
        src_n = int(row["source_n"])
        h = math.pi / (2.0 * src_n)  # approximate cell angular width
        l2 = float(row["edge_flux_l2_rel"])
        linf = float(row["edge_flux_linf"])
        l1 = float(row["cell_field_l1"])
        clinf = float(row["cell_field_linf"])

        rate_str = ""
        if prev_h is not None and prev_l2 is not None and l2 > 0 and prev_l2 > 0:
            rate = math.log(l2 / prev_l2) / math.log(h / prev_h)
            rate_str = f"  rate={rate:.2f}"

        print(f"{src_n:>10} {h:>10.4f} {l2:>12.4e} {linf:>12.4e} {l1:>12.4e} {clinf:>12.4e}{rate_str}")
        prev_h = h
        prev_l2 = l2

if __name__ == "__main__":
    main()
```

**Step 2: Run the convergence study and plot**

Run:
```bash
bash scripts/convergence_study.sh /tmp/mimetic-sphpoly-build > /tmp/convergence.csv
python3 scripts/plot_convergence.py /tmp/convergence.csv
```

Expected: Table showing convergence rates. We expect O(h^1) for edge flux L2_rel.

**Step 3: Commit**

```bash
git add scripts/plot_convergence.py
git commit -m "feat: add convergence rate plotting script"
```

---

## Phase 2: Fix L1 Diagnostic to Use Spherical Area

### Task 2.1: Add spherical_area to LocalPolygon and use it for diagnostic weighting

**Files:**
- Modify: `include/mimetic/mimetic.hpp` (LocalPolygon struct)
- Modify: `src/mimetic.cpp` (local_polygon spherical path)
- Modify: `tests/spherical_quad_test.cpp` (L1 error computation)

**Step 1: Add `spherical_area` field to `LocalPolygon`**

In `include/mimetic/mimetic.hpp`, add after `double area;` (line 133):
```cpp
    double spherical_area = 0.0;  // physical area on the sphere (only set for SphericalGnomonic)
```

**Step 2: Populate `spherical_area` in the spherical path of `local_polygon()`**

In `src/mimetic.cpp`, in the `SphericalGnomonic` branch of `local_polygon()` (around line 445), the `LocalPolygon` initializer already uses `sph.chart_area` for `area`. Add the spherical area to the struct. The `LocalPolygon` constructor call should set `spherical_area` from `sph.spherical_area`.

Update the `LocalPolygon` construction at L445-456 to also store `sph.spherical_area`.

**Step 3: Use `spherical_area` for L1 diagnostic weighting in `spherical_quad_test.cpp`**

At line 170, change:
```cpp
cell_field_l1_error += poly.area * field_error_norm;
```
to:
```cpp
const double weight = (poly.spherical_area > 0.0) ? poly.spherical_area : poly.area;
cell_field_l1_error += weight * field_error_norm;
```

**Step 4: Build and run tests**

Run: `cmake --build /tmp/mimetic-sphpoly-build --parallel && ctest --test-dir /tmp/mimetic-sphpoly-build --output-on-failure`
Expected: All 7 tests pass. L1 cell error values may change slightly.

**Step 5: Commit**

```bash
git add include/mimetic/mimetic.hpp src/mimetic.cpp tests/spherical_quad_test.cpp
git commit -m "fix: use spherical area for L1 diagnostic weighting on the sphere"
```

---

## Phase 3: Add Scalar Remap Comparison Test

### Task 3.1: Create a scalar spherical harmonic remap test

**Files:**
- Create: `tests/spherical_scalar_test.cpp`
- Modify: `CMakeLists.txt` (add test target)

**Step 1: Write the scalar remap test**

This test remaps the scalar field `f(x,y,z) = 0.5*(3z^2-1)` (Y_2^0) between cubed-sphere meshes using cell-area-weighted reconstruction, computing L2 and Linf errors against the exact cell averages. The purpose is to isolate whether the gnomonic projection + clipping machinery introduces errors on its own (separate from the vector reconstruction issues).

The scalar test computes:
- For each source cell: exact cell-average of f via quadrature over the gnomonic chart
- For each target cell: remap via overlap-area-weighted average of source cell values
- Compare against exact target cell-averages

Create `tests/spherical_scalar_test.cpp` implementing a first-order conservative scalar remap using the existing polygon clipping machinery.

**Step 2: Add to CMakeLists.txt**

Add a new test target:
```cmake
add_executable(spherical_scalar_test tests/spherical_scalar_test.cpp src/mimetic.cpp)
target_link_libraries(spherical_scalar_test ${MOAB_LIBRARIES} Eigen3::Eigen)
target_include_directories(spherical_scalar_test PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/tests ${MOAB_INCLUDE_DIRS})
add_test(NAME spherical_scalar_test COMMAND spherical_scalar_test)
```

**Step 3: Build and run**

Run: `cmake --build /tmp/mimetic-sphpoly-build --parallel && /tmp/mimetic-sphpoly-build/spherical_scalar_test`
Expected: Scalar L2/Linf errors should be significantly smaller than vector errors at the same resolution, confirming the metric-distortion hypothesis.

**Step 4: Commit**

```bash
git add tests/spherical_scalar_test.cpp CMakeLists.txt
git commit -m "feat: add scalar Y_2^0 remap test to isolate vector-specific errors"
```

---

## Phase 4: Implement Metric-Weighted Reconstruction

### Task 4.1: Add metric-weighted integration to reconstruct_source_polygon()

**Files:**
- Modify: `include/mimetic/mimetic.hpp` (add GeometryOptions field)
- Modify: `src/mimetic.cpp` (reconstruct_source_polygon)

**Step 1: Add a `metric_weighted` option to GeometryOptions**

In `include/mimetic/mimetic.hpp`, add to `GeometryOptions`:
```cpp
    bool metric_weighted = false;  // use gnomonic area scale in reconstruction integrals
```

**Step 2: Implement metric-weighted integrals in `reconstruct_source_polygon()`**

In `src/mimetic.cpp`, the Gram matrix `V` and moment vector `M` are computed via triangle-fan integrals (L766-806). When `options_.metric_weighted && is_spherical()`, multiply the integrand by the gnomonic area scale factor `gnomonic_area_scale(p + poly.centroid, frame)` where `frame` is built from `poly.n, poly.e_x, poly.e_y`.

The key changes:
1. Before the V/M loop, construct a `GnomonicFrame` from the polygon's spherical data
2. In each `integrate_triangle_scalar` lambda for V (L767), multiply `gi.dot(gj)` by the area scale
3. In each `integrate_triangle_scalar` lambda for M (L783, L788), multiply the integrand by the area scale
4. For the divergence `d = sum(flux) / area`, keep using `chart_area` (this is correct for chart fluxes)
5. For the KKT constraint rows C (L815), the edge integrals should NOT be metric-weighted (they are exact edge constraints)

The metric weighting only affects the L2 minimization (V matrix and M vector), not the conservation constraints (C matrix and F vector).

**Step 3: Enable metric weighting in the spherical tests**

In `tests/spherical_quad_test.cpp`, set `options.metric_weighted = true` (around line 86).
In `tests/spherical_voronoi_test.cpp`, set `options.metric_weighted = true` similarly.

**Step 4: Build and run tests**

Run: `cmake --build /tmp/mimetic-sphpoly-build --parallel && ctest --test-dir /tmp/mimetic-sphpoly-build --output-on-failure`
Expected: All tests pass. Conservation residuals remain < 5e-13. Edge flux errors should decrease, especially on coarse meshes.

**Step 5: Run convergence study with metric weighting**

Run: `bash scripts/convergence_study.sh /tmp/mimetic-sphpoly-build > /tmp/convergence_metric.csv && python3 scripts/plot_convergence.py /tmp/convergence_metric.csv`
Expected: Improved convergence rates or reduced error constants compared to unweighted baseline.

**Step 6: Commit**

```bash
git add include/mimetic/mimetic.hpp src/mimetic.cpp tests/spherical_quad_test.cpp tests/spherical_voronoi_test.cpp
git commit -m "feat: add metric-weighted reconstruction for spherical gnomonic charts"
```

---

## Phase 5: Update Technical Report

### Task 5.1: Add spherical extension section to the LaTeX report

**Files:**
- Modify: `docs/mimetic_voronoi_report.tex`

**Step 1: Add a new section covering:**

1. **Spherical gnomonic chart construction** -- frame, projection, inverse, Jacobian
2. **Contravariant Piola mapping** -- pullback/lift for chart-to-surface vector correspondence
3. **Conservation analysis** -- why chart-area divergence + Piola-pulled fluxes yield exact conservation
4. **Accuracy analysis** -- the variational crime of Euclidean basis on non-Euclidean chart
5. **Metric-weighted reconstruction** -- modification to the Gram matrix V and moment vector M
6. **Convergence results** -- table and/or figure showing h-convergence rates before and after metric weighting
7. **Literature comparison** -- relationship to GECoRe, TempestRemap, VEM H(div), and the gap in vector remap literature
8. **Future directions** -- metric-augmented basis, VEM H(div) on spherical polygons, higher-order reconstruction

**Step 2: Add bibliography entries for the key references**

Add entries for:
- Perot & Chartrand 2021 (mimetic method for polygons)
- Ullrich et al. 2009 (GECoRe)
- Ullrich & Taylor 2015 (TempestRemap)
- Beirao da Veiga et al. 2016 (VEM H(div))
- Brezzi, Falk, Marini 2014 (mixed VEM)
- Jones 1999 (SCRIP)
- Lauritzen & Nair 2008 (CaRS)
- Putman & Lin 2007 (cubed-sphere transport)

**Step 3: Compile and verify**

Run:
```bash
cd docs && pdflatex mimetic_voronoi_report.tex && bibtex mimetic_voronoi_report && pdflatex mimetic_voronoi_report.tex && pdflatex mimetic_voronoi_report.tex
```
Expected: PDF compiles without errors.

**Step 4: Commit**

```bash
git add docs/mimetic_voronoi_report.tex docs/mimetic_voronoi_report.bib
git commit -m "docs: add spherical extension, convergence analysis, and literature survey to report"
```

---

## Verification Checklist

After all phases:

- [ ] All 8 tests pass (7 existing + 1 new scalar test)
- [ ] Conservation residuals remain < 5e-13 on all tests
- [ ] Convergence study CSV and plot script produce correct output
- [ ] Metric-weighted reconstruction shows improved or equal accuracy vs baseline
- [ ] Scalar remap errors are significantly smaller than vector remap errors
- [ ] Technical report compiles cleanly
- [ ] README.md updated if new tests/scripts are added
