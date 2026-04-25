# High-Order Performance And OpenMP Rollout Plan

## Purpose

This plan integrates the current high-order performance review findings with an
OpenMP implementation strategy. It complements
`docs/plans/2026-04-22-high-order-unified-hdiv.md` by focusing on wall-clock
cost rather than numerical formulation.

The immediate objective is to reduce the runtime of the expensive high-order
conservative transfer paths, especially spherical high-order edge and
cell-moment transfer, without weakening any of the existing conservation or
verification requirements.

## Current Performance Facts

From the current targeted test timings:

- `high_order_edge_moment_test`: about `0.22 s`
- `spherical_high_order_moment_test`: about `8.85 s`
- `high_order_hdiv_convergence_test`: about `16.35 s`
- `spherical_high_order_hdiv_convergence_test`: about `400.72 s`

These timings imply:

1. the basic high-order algebra is not the dominant cost by itself;
2. the expensive path is the spherical high-order transfer plus the backward
   cell-moment and conforming machinery;
3. the KD-tree reduces candidate enumeration, but the remaining bottleneck is
   still the amount of repeated quadrature, repeated basis evaluation, and
   repeated geometry/chart work inside each candidate interaction.

## Non-Negotiable Constraints

- Conservation tolerance remains `5.0e-13`
- Planar and spherical numerical answers must remain unchanged to roundoff, or
  any deviation must be explicitly measured and justified
- OpenMP must be optional and build-time configurable
- No parallel loop may mutate MOAB topology or rely on MOAB thread safety
- Any parallel accumulation must use deterministic merge order if we want
  stable regression baselines

## High-Priority Bottlenecks To Resolve

### A. Serial Redundancy In Hot Loops

1. `moment_velocity_value(...)` rebuilds the vector polynomial basis on every
   point evaluation.
2. `reconstruct_source_polygon(...)` recomputes full basis cell moments for
   every scalar moment row.
3. `transfer_source_to_target_edge_moments(...)` loops over moment degree and
   repeats the same quadrature sweep for each degree.
4. `transfer_source_to_target_cell_moments(...)` is a deeply nested hotspot:
   target cell -> candidate source -> overlap polygon -> monomial -> fan
   triangle -> adaptive quadrature, with repeated Piola and chart transforms.
5. `project_target_edge_moments_to_hdiv_conforming(...)` recomputes an
   expensive divergence right-hand side after the raw transfer is already done.

### B. Geometry / Materialization Overhead

1. `local_polygon(...)` rebuilds charts and rereads MOAB geometry in hot paths.
2. `local_edges(...)` calls `find_or_create_edge(...)`, which performs adjacency
   queries repeatedly.
3. Candidate search is still bypassed entirely for planar mode.
4. The spherical KD-tree uses a loose global `max_radius` query expansion that
   can still leave many false positives.

### C. Parallelization Blockers

The current code cannot safely add OpenMP directly around the main loops
because:

1. hot loops still call MOAB geometry extractors and edge lookup/creation;
2. result containers are mostly `std::map` or push-back-driven vectors rather
   than index-addressable output arrays;
3. some loops write into class-owned state (`reconstructions_`,
   `directed_target_flux_`, MOAB tags) during traversal.

Therefore, the first OpenMP milestone is **not** adding pragmas. It is to
separate hot compute kernels from MOAB/materialization and from mutable shared
state.

## OpenMP Build Configuration Plan

### Build Surface

Add a build option:

```cmake
option(MIMETIC_ENABLE_OPENMP "Enable OpenMP parallel acceleration" OFF)
```

Then:

```cmake
if(MIMETIC_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
  target_link_libraries(mimetic PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(mimetic PUBLIC MIMETIC_ENABLE_OPENMP=1)
endif()
```

For this machine, the intended configure path is:

```bash
cmake -S . -B build-omp \
  -DMIMETIC_ENABLE_OPENMP=ON \
  -DOpenMP_ROOT=/opt/homebrew/Cellar/libomp/21.1.3
```

### Runtime Surface

Primary controls:

- `OMP_NUM_THREADS`
- `OMP_PROC_BIND=close`
- `OMP_PLACES=cores`
- `OMP_SCHEDULE=static`

Default guidance for deterministic runs:

```bash
OMP_NUM_THREADS=1
OMP_PROC_BIND=close
OMP_PLACES=cores
OMP_SCHEDULE=static
```

Performance runs can then compare `1, 2, 4, 8`.

## Threading Strategy

### Rule 1: Precompute Immutable Geometry Caches Serially

Before any OpenMP loop, build immutable arrays for:

- `LocalPolygon`
- `LocalEdge`
- absolute vertex coordinates
- spherical frames
- precomputed target edge metadata
- source reconstruction metadata
- source/target polygon index maps

These caches must be index-addressable and detached from MOAB calls inside the
parallel region.

### Rule 2: Use Thread-Private Scratch, Not Shared Temporary Containers

Introduce a thread scratch structure for the hot transfer kernels:

- projected target vertices
- overlap polygon buffers
- quadrature work arrays
- Legendre values per quadrature point
- temporary pointwise velocity / local-coordinate arrays
- per-thread reduction buffers for moments and divergence contributions

Implementation preference:

- `thread_local` scratch for helper workspaces where lifetime is naturally
  per-thread, or
- explicit local scratch objects inside `#pragma omp parallel` regions

Avoid global mutable scratch and avoid allocating STL containers repeatedly in
the innermost loops.

### Rule 3: No MOAB Mutation Inside Parallel Regions

Do **not** call these inside OpenMP loops:

- `find_or_create_edge(...)`
- `create_element(...)`
- `tag_set_data(...)`
- mesh merge / adjacency mutation

Any MOAB writes should be staged into dense arrays and committed afterward in a
serial pass.

### Rule 4: Deterministic Reductions

For conservative quantities and regression-sensitive outputs:

- accumulate into thread-local arrays;
- merge in fixed thread order after the parallel region;
- do not rely on unordered floating-point reductions if we want stable
  regression baselines.

## Implementation Sequence

### Phase 0: Instrumentation And Baselines

Add lightweight timers around:

- `reconstruct_source_polygon(...)`
- `transfer_source_to_target_edge_moments(...)`
- `transfer_source_to_target_cell_moments(...)`
- `project_target_edge_moments_to_hdiv_conforming(...)`
- `target_divergence_rhs(...)`

Acceptance:

- per-kernel timing printed only in opt-in profiling mode;
- baseline timings recorded for:
  - `high_order_hdiv_convergence_test`
  - `spherical_high_order_moment_test`
  - `spherical_high_order_hdiv_convergence_test`

### Phase 1: Serial Kernel Cleanup Before OpenMP

#### Task 1.1
Cache basis terms in `MomentReconstruction` or a compiled evaluator object so
`moment_velocity_value(...)` does not rebuild basis descriptors per call.

#### Task 1.2
Precompute `basis_cell_vector_moments(...)` once per basis column during source
reconstruction.

#### Task 1.3
Fuse all target edge moment degrees into one quadrature sweep per clipped
segment.

#### Task 1.4
Refactor backward cell-moment transfer to reuse pointwise chart transforms and
field evaluations across monomials/components.

#### Task 1.5
Exploit Gram-matrix symmetry and reduce repeated dense setup in
`reconstruct_source_polygon(...)`.

Acceptance:

- no numerical changes;
- measurable reduction in single-thread runtime before any OpenMP changes.

### Phase 2: Geometry Cache Refactor

#### Task 2.1
Create precomputed source and target cache objects for high-order transfer
paths.

#### Task 2.2
Remove `find_or_create_edge(...)` and `local_polygon(...)`/`local_edges(...)`
from hot loops by building edge/polygon metadata once.

#### Task 2.3
Extend spatial indexing to planar mode and tighten spherical candidate
pruning using per-cell radii, not just `max_radius`.

Acceptance:

- hot loops become pure compute over immutable caches;
- candidate counts drop in both planar and spherical paths.

### Phase 3: OpenMP Parallel Reconstruction

Parallelize over independent source cells for high-order reconstruction using
the immutable geometry/source-moment caches.

Important:

- reconstructions must be written by source-cell index into a pre-sized vector;
- class-owned maps should be populated in a serial commit pass afterward, or
  replaced with index-addressable storage.

Acceptance:

- `OMP_NUM_THREADS=1` reproduces serial results;
- `OMP_NUM_THREADS>1` preserves conservation and regression metrics;
- source reconstruction wall time decreases with thread count.

### Phase 4: OpenMP Parallel Edge Transfer

Parallelize over target cells, then over their local edges, in
`transfer_source_to_target_edge_moments(...)`.

Requirements:

- thread-private overlap and quadrature scratch;
- no shared push-back into contributions vectors inside the parallel loop;
- write moment results into pre-sized arrays by deterministic DOF index.

Acceptance:

- exact target edge moments match serial within tolerance;
- measurable speedup on spherical high-order moment tests.

### Phase 5: OpenMP Parallel Cell-Moment Transfer

Parallelize `transfer_source_to_target_cell_moments(...)` over target cells.

This is the highest-payoff OpenMP target after the serial cleanup.

Requirements:

- each thread owns its local `moments` array;
- overlap buffers and projected coordinates are thread-private;
- no MOAB calls inside the inner quadrature loops.

Acceptance:

- round-trip results remain unchanged to tolerance;
- spherical high-order round-trip and convergence studies show the largest
  wall-time reduction here.

### Phase 6: Conforming Projection Optimization

Do not parallelize the dense Schur solve first. Instead:

1. reduce duplicate geometry work by reusing raw-transfer overlap metadata or
   cached target-cell RHS contributions;
2. replace dense assembly where possible with sparse/indexed assembly;
3. parallelize only the embarrassingly parallel preprocessing pieces.

Acceptance:

- conforming projection no longer performs a second fully redundant geometry
  pass for the same source-target relation;
- preprocessing cost drops before the solve itself becomes the next target.

## Verification Plan

For every phase:

1. serial build and test:
   ```bash
   cmake --build build --parallel
   ctest --test-dir build --output-on-failure
   ```
2. OpenMP build:
   ```bash
   cmake -S . -B build-omp \
     -DMIMETIC_ENABLE_OPENMP=ON \
     -DOpenMP_ROOT=/opt/homebrew/Cellar/libomp/21.1.3
   cmake --build build-omp --parallel
   ```
3. thread-count checks:
   ```bash
   OMP_NUM_THREADS=1 ctest --test-dir build-omp -R "high_order|spherical_high_order" --output-on-failure
   OMP_NUM_THREADS=2 ctest --test-dir build-omp -R "high_order|spherical_high_order" --output-on-failure
   OMP_NUM_THREADS=4 ctest --test-dir build-omp -R "high_order|spherical_high_order" --output-on-failure
   ```
4. timing checks:
   - compare `OMP_NUM_THREADS=1/2/4/8` for:
     - `high_order_hdiv_convergence_test`
     - `spherical_high_order_moment_test`
     - `spherical_high_order_hdiv_convergence_test`

## Acceptance Criteria

### Numerical

- conservation checks remain within `5.0e-13`
- conforming target divergence residuals remain within existing tolerances
- round-trip Mercator results do not regress
- no test is weakened to hide OpenMP-induced drift

### Performance

Minimum expected gains after all phases:

- noticeable single-thread improvement from serial cleanup
- planar high-order transfers faster due to planar spatial indexing
- spherical high-order moment transfer substantially faster than current
  baseline
- spherical high-order convergence test reduced materially from the current
  `~400 s` scale

## Recommended Commit Sequence

1. `plan/perf`: add timers and baseline reporting
2. `perf/cache-basis`: cached evaluator and precomputed basis moments
3. `perf/geometry-cache`: immutable geometry/materialization refactor
4. `perf/kdtree-planar`: planar spatial index + tighter spherical pruning
5. `build/openmp`: CMake option and OpenMP wiring
6. `perf/omp-reconstruct`: parallel source reconstruction
7. `perf/omp-edge-transfer`: parallel edge moment transfer
8. `perf/omp-cell-moments`: parallel backward cell-moment transfer
9. `perf/conforming`: reduce duplicate conforming-projection geometry work

## Out Of Scope For This Phase

- MPI
- GPU offload
- changing the numerical method itself
- weakening conservation or regression tolerances for speed
