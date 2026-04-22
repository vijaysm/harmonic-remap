# High-Order Unified H(div) Implementation Plan

## Goal

Replace the current split high-order prototype with a single higher-order
polygonal `H(div)`-style hierarchy that:

1. uses one local family for `p=1,2,3`,
2. supports high-order target-edge moment transfer,
3. enforces globally conforming target-edge moment data,
4. lifts the same high-order method to spherical gnomonic charts, and
5. replaces the raw `[P_p]^2` basis with a structured
   divergence/harmonic/bubble split.

This phase keeps the existing low-order harmonic mimetic code as a reference
path until the unified high-order hierarchy is verified.  After verification,
the report should describe the new unified hierarchy as the primary method and
stop presenting the older split formulation as the main high-order approach.

## Non-Negotiable Numerical Requirements

- Absolute conservation tolerance: `5.0e-13`
- Deterministic serial verification only
- No duplicate vertices/edges/elements in generated test meshes:
  always merge coincident vertices with tolerance `1e-12`
- Direct transfer and global conforming projection must agree with their
  defining constraints to roundoff

## Implementation Order

### Task 1: High-Order Target-Edge Moment Conformity

Add a higher-order target-edge postprocess for the current high-order moment
transfer.

Implementation:
- Introduce a new result type for conforming target-edge moment vectors.
- Collapse directed target edges to one geometric target edge as is already done
  for the lowest-order flux solve.
- For moment `m=0`, solve the constrained least-squares projection that enforces
  exact target-cell divergence balance.
- For higher moments `m>=1`, collapse to one geometric target-edge moment vector
  by least-squares averaging with the same orientation map.

Acceptance:
- shared target edges carry one unique signed moment vector,
- `m=0` target-cell residuals are `<= 5e-13`,
- raw-to-conforming correction shrinks under refinement,
- new regression test covers quad and Voronoi cases.

### Task 2: High-Order Spherical Lift

Lift the higher-order moment transfer to `GeometryMode::SphericalGnomonic`.

Implementation:
- Generalize the current planar moment interpolator into a geometry-aware
  moment interpolator with `GeometryOptions`.
- In spherical mode, define source edge moments in the source chart using the
  Piola pullback.
- Project target great-circle edges into each source chart, clip the projected
  straight segment, and accumulate target edge moments in that source chart.
- Add structured cubed-sphere and spherical Voronoi higher-order tests.

Acceptance:
- direct spherical high-order transfer conserves `m=0` to `5e-13`,
- structured `p=2,3` edge errors decrease under refinement,
- unstructured spherical n-gon tests pass direct-vs-conforming checks.

### Task 3: Basis Improvement From Raw `[P_p]^2` To A Divergence/Harmonic/Bubble Split

Replace the current raw vector-polynomial basis with a structured local basis:

- divergence-carrying particular modes,
- divergence-free harmonic-gradient modes,
- interior bubble modes that vanish on the cell boundary.

Implementation:
- Add a basis-family option and make the structured basis the default.
- Use a polygon bubble built from normalized inward edge distances.
- Ensure the basis dimension still matches the target local coefficient count
  needed for the projected solve.

Acceptance:
- exact polynomial patch tests remain passing,
- conditioning is at least as good as the current scaled raw basis on the
  existing refinement studies,
- the report can describe the local space in structured terms rather than as a
  raw monomial fit.

### Task 4: Unify The High-Order Family So `p=1,2,3` Come From One Hierarchy

Implementation:
- stop using the low-order harmonic `MimeticInterpolator` as the `p=1`
  convergence baseline in the high-order study,
- make the same moment interpolator and structured basis support `p=1,2,3`,
- keep exact hard enforcement of the zeroth edge moment at every order.

Acceptance:
- the convergence study reports `p=1,2,3` from one code path,
- `p=1` remains conservative and convergent on quad and Voronoi studies,
- the manuscript no longer needs to describe a split baseline for the
  high-order hierarchy.

### Task 5: Extend The Global Target-Edge Conforming Solve To Higher Moments

Implementation:
- generalize the target-edge conforming projection from scalar edge fluxes to
  edge moment vectors,
- enforce exact target-cell balance through the `m=0` moment,
- enforce one globally single-valued target-edge moment vector on each
  geometric edge,
- apply the same extension to the spherical high-order path.

Acceptance:
- one unique target-edge moment vector per geometric edge,
- exact target-cell balance via `m=0`,
- raw and conforming high-order transfers can both be compared in new tests and
  plots.

## Verification Sequence

After each task:

1. build the targeted test executable(s),
2. run the relevant regression(s),
3. commit only the files for that task.

Final gate:

```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/high_order_hdiv_convergence_test docs/high_order_hdiv_convergence.csv
conda run -n climate-vis python scripts/plot_high_order_hdiv_convergence.py \
  docs/high_order_hdiv_convergence.csv \
  docs/figures/high_order_hdiv_convergence.png
```

Then rebuild the manuscript and refresh all reported tables/figures.

## Report Rewrite Requirements

Once all five tasks pass:

- rewrite the methodology to present the unified high-order hierarchy as the
  primary high-order method,
- remove language that treats the raw `[P_p]^2` prototype or split `p=1` line
  as the active high-order formulation,
- document the higher-order conforming target-edge moment solve,
- document the spherical high-order lift,
- update results for planar and spherical `p=1,2,3`.
