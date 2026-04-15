# Spherical Quad CLI + VTK Export Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `spherical_quad_test` accept source and target cubed-sphere face resolutions as command-line arguments and write source/target VTK outputs that include scalar diagnostics and centroid-sampled vector data for visualization.

**Architecture:** Keep the implementation local to `tests/spherical_quad_test.cpp` so experimental controls and visualization logic stay in the test harness rather than expanding the core library API. Parse `source_n`, `target_n`, and optional `output_prefix` from `argv`, generate meshes with those values, compute exact source diagnostics and reconstructed target diagnostics, attach them as MOAB tags, and write each mesh to its own VTK file through meshsets.

**Tech Stack:** C++14, MOAB tags/meshsets/VTK writer, Eigen, existing mimetic reconstruction/interpolation code.

---

### Task 1: Add runtime argument parsing

**Files:**
- Modify: `tests/spherical_quad_test.cpp`
- Test: `build/spherical_quad_test`

**Step 1: Add small parsing helpers**

Add helper logic near the top of the file to:
- parse positive integers from `argv`
- choose a default output prefix when not supplied
- print a usage message on invalid input

**Step 2: Update `main` signature**

Change:
```cpp
int main()
```
To:
```cpp
int main(int argc, char** argv)
```

**Step 3: Read source and target resolutions**

Use defaults compatible with the current behavior:
```cpp
int source_n = 4;
int target_n = 6;
std::string output_prefix = "cubed_sphere";
```
Override them from `argv` when supplied.

**Step 4: Validate arguments**

Reject values smaller than 1 and print a clear message like:
```text
Usage: spherical_quad_test [source_n] [target_n] [output_prefix]
```

**Step 5: Use parsed values in mesh generation**

Replace hardcoded `N_coarse` / `N_fine` usage with the parsed variables.

**Step 6: Run test binary manually**

Run:
```bash
./build/spherical_quad_test
./build/spherical_quad_test 3 5 demo
```
Expected:
- both runs succeed
- element counts differ between runs
- output filenames follow the prefix

### Task 2: Add source-cell visualization fields

**Files:**
- Modify: `tests/spherical_quad_test.cpp`
- Test: `build/spherical_quad_test`

**Step 1: Create MOAB tags for source diagnostics**

Add dense tags for source cells:
- `SOURCE_DIV_EXACT` (scalar)
- `SOURCE_FLUX_L1_EXACT` (scalar)
- `SOURCE_FIELD_EXACT` (3-vector)

**Step 2: Compute centroid-sampled exact source vector**

For each source polygon:
- compute a 3D centroid direction from `poly.centroid_3d.normalized()`
- evaluate `spherical_harmonic_gradient(...)`

**Step 3: Compute scalar diagnostics per source cell**

Store:
- sum of exact edge fluxes for the cell
- L1 sum of absolute edge flux magnitudes

**Step 4: Write source tags to MOAB**

Use `tag_set_data` on the cell handle for all three source tags.

**Step 5: Verify no regression in source reconstruction**

Run:
```bash
./build/spherical_quad_test 4 6 source_check
```
Expected:
- reconstruction still succeeds
- source diagnostics are written before VTK export

### Task 3: Add target-cell reconstructed and exact visualization fields

**Files:**
- Modify: `tests/spherical_quad_test.cpp`
- Test: `build/spherical_quad_test`

**Step 1: Create MOAB tags for target diagnostics**

Add dense tags for target cells:
- `TARGET_DIV_RECON`
- `TARGET_FLUX_L1_RECON`
- `TARGET_DIV_ERROR`
- `TARGET_FIELD_RECON` (3-vector)
- `TARGET_FIELD_EXACT` (3-vector)
- `TARGET_FIELD_ERROR` (3-vector)

**Step 2: Reconstruct target polygons from transferred fluxes**

For each target cell:
- write transferred edge fluxes into the existing target/source flux tag path needed for reconstruction, or create a local reconstruction path using the same algorithm already used for source polygons
- obtain a target-cell reconstructed coefficient set

**Step 3: Evaluate reconstructed vector at target centroid**

Use the target polygon local frame to evaluate the reconstructed planar field at the local origin and map it back into 3D tangent components with `e_x` and `e_y`.

**Step 4: Evaluate exact vector at target centroid**

Use the normalized 3D centroid and call `spherical_harmonic_gradient(...)`.

**Step 5: Store scalar diagnostics per target cell**

Store:
- reconstructed cell divergence from transferred flux sum
- L1 sum of transferred edge flux magnitudes
- divergence error relative to exact/discrete target divergence

**Step 6: Store vector diagnostics per target cell**

Store:
- reconstructed vector
- exact vector
- difference vector

### Task 4: Write configurable VTK outputs

**Files:**
- Modify: `tests/spherical_quad_test.cpp`
- Test: `build/spherical_quad_test`

**Step 1: Build output filenames from prefix**

Use:
```cpp
const std::string source_path = output_prefix + "_source.vtk";
const std::string target_path = output_prefix + "_target.vtk";
```

**Step 2: Keep meshset-based export**

Continue writing source and target meshsets separately so ParaView loads each as a distinct mesh with its own cell fields.

**Step 3: Print filenames to stdout**

Emit exact paths so the user can find the files quickly.

**Step 4: Verify output creation**

Run:
```bash
./build/spherical_quad_test 3 5 demo
ls demo_source.vtk demo_target.vtk
```
Expected:
- both files exist
- no MOAB “Nothing to write!” error

### Task 5: Verification

**Files:**
- Modify: `tests/spherical_quad_test.cpp`
- Verify: build + tests

**Step 1: Build project**

Run:
```bash
cmake --build build --parallel
```
Expected: success

**Step 2: Run targeted spherical test**

Run:
```bash
ctest --test-dir build -R spherical_quad_test --output-on-failure
```
Expected: pass

**Step 3: Run full test suite**

Run:
```bash
ctest --test-dir build --output-on-failure
```
Expected: all tests pass

**Step 4: Manual CLI verification**

Run:
```bash
./build/spherical_quad_test 4 6
./build/spherical_quad_test 6 4 swapped
```
Expected:
- both succeed
- conservation still holds
- VTK outputs are produced with the requested prefixes
