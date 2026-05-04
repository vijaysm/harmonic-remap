# Conforming-Projection Geometry Cache + Phase-1 Serial Cleanup

**Date:** 2026-05-02
**Branch:** `feature/conforming-vem`
**Predecessor plan:** `docs/plans/2026-04-24-high-order-performance-openmp.md`,
Phase 0–1, Phase 6
**Reviewed gap:** OpenMP plan §A (serial redundancy), §B (geometry overhead)

## Objective

Eliminate the redundant geometry pass in the conforming projection and
the per-call basis rebuilds in the high-order transfer hot loops, without
changing any numerical answer to roundoff. The single largest test
runtime today is `spherical_high_order_hdiv_convergence_test` at
`~400 s`; the second pass through `target_divergence_rhs` and the
per-evaluation basis rebuilds are the leading contributors.

This is a pure performance refactor. It is also a prerequisite for safe
OpenMP parallelization, because the parallel loops must operate on
immutable index-addressable caches and never call MOAB inside the
parallel region.

## Non-negotiable requirements

- Numerical answers byte-identical to the current code (or differing
  only by floating-point reduction order, in which case test tolerances
  must explicitly accommodate the deviation, with the deviation
  measured and recorded).
- Conservation tolerance `5e-13` preserved.
- All 13 ctest targets continue to pass at the same tolerances.
- No new MOAB calls inside any loop that can be parallelized.

## Performance baseline (must measure before any change)

Capture `time` for each of:

```bash
ctest --test-dir build -R high_order_edge_moment_test --output-on-failure
ctest --test-dir build -R spherical_high_order_moment_test --output-on-failure
ctest --test-dir build -R high_order_hdiv_convergence_test --output-on-failure
ctest --test-dir build -R spherical_high_order_hdiv_convergence_test --output-on-failure
```

Save to `docs/plans/2026-05-02-perf-baseline.txt`.

## Phase 1 — Conforming projection geometry cache

### Diagnosis

`target_divergence_rhs` (two overloads, `src/mimetic.cpp:1436` and
`src/mimetic.cpp:1561`) builds full `SourceCache` and `TargetCache`
arrays on every call. The conforming projection calls one of them
*after* `transfer_source_to_target_edge_moments` has already done a full
geometry pass. The caches are byte-identical between the two passes.

### Implementation

Files: `include/mimetic/mimetic.hpp`, `src/mimetic.cpp`.

1. Promote the local `SourceCache` and `TargetCache` types in
   `target_divergence_rhs` and the equivalent structures in
   `transfer_source_to_target_edge_moments` to a shared file-local
   `MomentTransferGeometryCache` struct.
2. Add an opt-in cache parameter to both routines:

   ```cpp
   struct MomentTransferGeometryCache {
       std::vector<SourceGeometry> sources;
       std::vector<TargetGeometry> targets;
       SpatialIndex spatial_index;
       bool valid = false;
   };
   ```

3. Build the cache once in
   `project_target_edge_moments_to_hdiv_conforming` (or in a new
   `MomentTransferContext` helper used by both raw and conforming
   transfers when the caller wants to amortize), and pass it down.
4. The raw transfer must accept an optional pointer; if `nullptr`, it
   constructs and discards as today.
5. Add timers (compile-time gated by `MIMETIC_PROFILE`) around each
   geometry build to verify the caches save the second pass.

### Acceptance

- The two `target_divergence_rhs` overloads no longer rebuild geometry
  when called from the conforming projection.
- `spherical_high_order_hdiv_convergence_test` wall time decreases by
  ≥ 30 % relative to the baseline.
- Numerical outputs match baseline to `1e-14` on every test.

## Phase 2 — Compiled basis evaluator

### Diagnosis

`moment_velocity_value(reconstruction, p)` rebuilds the vector
polynomial basis on every call. In the spherical convergence test it is
called O(`N_target_cells × N_overlap × N_quadrature × p`) times.

### Implementation

1. Add a `CompiledMomentBasis` struct holding the precomputed monomial
   exponent table, basis-mode indices, and per-mode coefficient vectors.
2. Cache one `CompiledMomentBasis` per `MomentReconstruction`,
   constructed lazily on first evaluation and stored on the
   `MomentReconstruction` itself or in a side-cache keyed on its
   identity.
3. Replace the per-call basis assembly in `moment_velocity_value` with
   a `compiled.evaluate(point, coeffs)` call.

### Acceptance

- `moment_velocity_value` allocates zero heap memory in steady state.
- `high_order_hdiv_convergence_test` wall time decreases by ≥ 15 %
  vs the Phase 1 baseline.
- Numerical outputs match Phase 1 to `1e-14`.

## Phase 3 — Fused multi-degree edge-moment quadrature

### Diagnosis

`transfer_source_to_target_edge_moments` loops over moment degree
*outside* the per-segment quadrature, repeating the velocity evaluation
and the chart projection at each degree.

### Implementation

1. For each clipped target-edge subsegment, evaluate `v · n` once at
   each quadrature point.
2. Multiply by all `p+1` Legendre polynomial values at the per-edge
   parametrization `t ∈ [-1, +1]` and accumulate into a length-`(p+1)`
   moment vector.
3. Cache the Legendre polynomial table for the chosen quadrature rule
   once per call.

### Acceptance

- `transfer_source_to_target_edge_moments` wall time per moment vector
  is roughly independent of `p` (the cost dominated by quadrature once,
  not per-degree).
- `spherical_high_order_hdiv_convergence_test` wall time decreases by
  ≥ 25 % vs the Phase 2 baseline.

## Phase 4 — Tightened spherical candidate radii

### Diagnosis

The spherical KD-tree query uses a global `max_radius`. Per-cell radii
would prune false positives substantially.

### Implementation

1. Use the per-cell `PolygonSearchGeometry::radius` already computed in
   `target_divergence_rhs` for each query.
2. Verify candidate counts decrease in profiling.

### Acceptance

- Average candidate count per target cell decreases by ≥ 30 %.
- No false negatives: the conforming residual remains ≤ `5e-13`.

## Phase 5 — Verification gate

```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/high_order_hdiv_convergence_test /tmp/high_order_perf.csv
./build/spherical_high_order_hdiv_convergence_test /tmp/spherical_high_order_perf.csv
diff -u docs/high_order_hdiv_convergence.csv /tmp/high_order_perf.csv
diff -u docs/spherical_high_order_hdiv_convergence.csv /tmp/spherical_high_order_perf.csv
```

Acceptance:

- All tests pass.
- Both CSV diffs empty (or differ only at the `1e-14` floor).
- Total wall-time savings on `spherical_high_order_hdiv_convergence_test`
  ≥ 50 % vs the recorded Phase 0 baseline.

## Risks

- **FP non-associativity** in fused quadrature can perturb the last
  digit. Mitigation: keep the original accumulation order; only cache
  intermediate values, do not rearrange.
- **Cache invalidation** between meshes. Mitigation: tag the cache with
  the source/target handle vectors and invalidate on mismatch.
- **Compiled-basis lifetime** vs `MomentReconstruction` mutation.
  Mitigation: make `CompiledMomentBasis` an owned member that is rebuilt
  if the reconstruction is reassigned.

## Out of scope

- Adding OpenMP `pragma`s. This plan only prepares the substrate; the
  pragmas are the next, separate, plan.
- The dense Schur solve in the conforming projection itself.

## Findings (during execution)

The original 400 s motivating runtime referenced in the plan is no
longer reproducible.  As of the recorded baseline
`docs/plans/baselines/2026-05-02-perf-baseline.txt`, the dominant
test `spherical_high_order_hdiv_convergence_test` runs in ~72 s
single-threaded.  The OpenMP work that landed earlier already
addressed the largest costs.  The per-phase outcomes were:

- **Phase 1 (geometry cache)** — implemented as planned.  The
  `MomentTransferSourceCache`/`MomentTransferTargetCache` types and
  `build_moment_transfer_*` helpers were extracted from
  `target_divergence_rhs`.  In `project_target_edge_moments_to_hdiv_
  conforming`, the cache is now built once and reused for both the
  divergence-RHS computation and the source-skeleton trace-jump
  diagnostic.  Numerics are byte-identical; wall-time delta is at
  the measurement noise floor (~72 s before and after).  The work is
  still valuable as a prerequisite for safe future OpenMP scaling
  and as documentation of the cache layout.
- **Phase 2 (compiled basis evaluator)** — already in place.
  `moment_velocity_value` (src/mimetic.cpp around line 1184) uses
  thread-local `MonomialScratch` so it allocates zero heap memory
  in steady state.
- **Phase 3 (fused multi-degree edge-moment quadrature)** — already
  in place.  `accumulate_edge_moment_bundle` (src/mimetic.cpp around
  line 1082) evaluates the velocity sample once per quadrature point
  and accumulates into all `p+1` Legendre moment slots in a single
  pass.
- **Phase 4 (tightened spherical candidate radii)** — already in
  place.  `find_overlap_candidates` (src/mimetic.cpp around line 123)
  uses per-cell radii in the kd-tree filter; only the broad-phase
  pruning uses a global `max_radius`, which is correct for any
  spatial-index query.
- **Phase 5 (verification gate)** — passes.  All 18 ctest targets
  remain green; convergence CSVs differ only in the
  `conforming_divergence_residual` column at the roundoff floor (the
  l2_moment0 and l2_all rate-determining columns are byte-identical).

The overall conclusion is that the geometry-cache plan ships as a
cleanup/restructuring patch and as a prerequisite for further
parallel work; the originally projected ~50 % wall-time gain is no
longer available because the underlying inefficiencies it targeted
have been addressed by separate work in the meantime.
