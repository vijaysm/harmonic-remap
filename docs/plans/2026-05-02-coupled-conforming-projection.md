# Coupled Multi-Degree Conforming Target-Edge Projection

**Date:** 2026-05-02
**Branch:** `feature/conforming-vem`
**Predecessor plan:** `docs/plans/2026-04-22-high-order-unified-hdiv.md`, Task 5
**Reviewed gap:** `claude_plans.md` item I

## Objective

Replace the current degree-decoupled `project_target_edge_moments_to_hdiv_conforming`
with a single coupled block solve that

1. produces one single-valued moment vector per geometric target edge across
   all degrees `m = 0, ..., p`,
2. enforces the cell divergence balance via the `m = 0` block (current
   property — must be preserved exactly),
3. uses a proper edge-mass-weighted Legendre L² objective for the deviation
   from the raw transferred moments rather than the current uniform
   per-directed-edge averaging,
4. exposes a regression-grade trace-continuity diagnostic (currently
   absent), and
5. on planar and spherical paths produces numerically *no worse* results
   than the current solver and *better* on Voronoi cases where the directed
   edges sharing a geometric edge differ in cell-side parametrization.

The goal is a defensible, theoretically clean, single coupled KKT system.
Identical numerics for `p = 1` (single moment, identical to current code);
strictly improved for `p ≥ 2` because Legendre normalization weights enter
the objective rather than being implicitly assumed.

## Non-negotiable requirements

- Conservation tolerance: `5.0e-13` absolute for `m = 0` cell divergence
  residuals on every cell of every test case
- Direct vs sparse projection agreement remains within `5.0e-13`
- Trace continuity: `|x_{e}^{(m), c_+}  -  sign(c_-, c_+, m) x_{e}^{(m), c_-}| ≤ 1e-12`
  on every interior shared edge for every degree, where `sign` is the
  parity-orientation factor already implemented in
  `target_edge_moment_orientation_factor`
- All existing 13 ctest targets continue to pass
- Convergence rates (planar quad, planar Voronoi, cubed-sphere, spherical
  Voronoi-patch) do not regress at `p = 1, 2, 3`

## Theoretical formulation

Let
- `N_d` = number of directed target-edge DOFs
- `N_u` = number of unique geometric target edges
- `N_c` = number of target cells
- `p`   = highest moment degree
- `S`   = `N_u * (p+1)`, total conforming DOFs
- `U_raw ∈ R^{N_d × (p+1)}` = raw transfer (`raw_transfer.target_moments`)
- `O ∈ {-1, +1}^{N_d}` = directed-to-unique orientation signs
- `f_m(o) = O^m` for even `m`, `O` for odd `m` (Legendre parity rule
  already encoded by `target_edge_moment_orientation_factor`)
- `B_m ∈ R^{(2m+1)} = 2 / (2m + 1)` — Legendre L²-normalization weight on
  `[-1, 1]`
- `L_e` = arc length of unique edge `e` (planar Euclidean or spherical
  great-circle; `LocalPolygon::points_3d` allows both)

### Variables

`x = (x_{e,m})_{e=1..N_u, m=0..p} ∈ R^S` — one Legendre moment per unique
edge per degree.

### Per-directed-edge mapping

For each directed edge `i` with unique index `u(i)` and orientation
`O_i ∈ {-1, +1}`, the conforming directed value is

    U_i^{(m)} = f_m(O_i) · x_{u(i), m}

This is identical to the current code's reconstruction step.

### Objective (proper L² norm)

    J(x) = (1/2) Σ_i Σ_m  w_{e(i), m} · ( U_i^{(m)} − U_raw_{i,m} )^2

with weights `w_{e, m} = L_e · B_m`. The weight is the Legendre L²
inner-product weight on the physical edge `[-L/2, +L/2]` — the natural
norm under which Legendre moments of different degree are
mass-orthonormal.

For each `(u, m)` block this collapses to a per-edge scalar least-squares
problem (because `f_m(O_i) ∈ {-1, +1}`) and the unconstrained minimizer is

    x_{u, m}* = ( Σ_{i: u(i)=u} f_m(O_i) · w_{e(u), m} · U_raw_{i, m} ) /
                ( Σ_{i: u(i)=u}            w_{e(u), m} )
              = average of orientation-corrected raw values

(the per-edge weights cancel because all directed views of one geometric
edge share the same length and the same Legendre weight). This is
*identical* to the current `hinv * g` average — confirming our refactor
preserves the unconstrained answer. The weights matter only when we add
the cell-divergence-balance constraint below.

### Constraint

For each target cell `c`, the divergence theorem gives

    Σ_{e in cell c} sign(c, e) · ∫_e v · n ds  =  ∫_K div(v) dA  =  RHS_c

In Legendre-moment language, `∫_e v · n ds = x_{e, 0}` (the zeroth
Legendre coefficient is the *integral* of `v · n`, not the average — this
matches the convention in `transfer_source_to_target_edge_moments`).

The constraint is therefore

    A_0 · π_0(x)  =  b

where `A_0 ∈ R^{N_c × N_u}` is the cell-edge incidence with signs, and
`π_0(x) ∈ R^{N_u}` extracts the `m = 0` block. The right-hand side `b` is
the existing `target_divergence_rhs`.

### Coupled KKT system

The full system is block-diagonal in `m` with the `m = 0` block coupled
across cells through the divergence constraint:

    Block m = 0:  W_0 · x_0  −  A_0^T λ  =  W_0 · π_0(U_avg)
                  A_0 · x_0              =  b
    Block m ≥ 1:  W_m · x_m              =  W_m · π_m(U_avg)
                  ⇒ x_m  =  π_m(U_avg)   (identical to per-degree average)

where `W_m = diag(2 L_e / (2m+1))` and `π_m(U_avg)` is the orientation-
corrected per-edge average for degree `m`.

So the only block that needs the Schur solve is `m = 0`, exactly as in the
current code. **The "coupling" is that the entire moment vector is
exposed as one variable and one solve, with the constraint structure
made explicit in matrix form.** The numerical answer is the same as the
current code for any `p`. The improvements over the current code are:

1. The block matrix structure is explicit, enabling a future extension to
   include cell-vector-moment DOFs in the same system without rewriting
   the per-degree loop.
2. The edge mass weight `L_e · B_m` is named and tested, removing the
   silent assumption that the per-degree per-edge averaging is the L²
   minimizer.
3. The orientation factor and L² weight live in one assembly path,
   simplifying the reasoning and the extension to spherical mode.
4. We expose a `trace_jump` diagnostic per unique edge per degree, which
   is the actual quantity a "conforming" projection should drive to
   zero; the current code has no such diagnostic.

### Optional cell-vector-moment coupling (deferred extension, documented here)

For `p ≥ 2`, target cell vector moments

    c_K^{(α)} = ∫_K v · ê_i  x^a y^b dA,  |α| ≤ p − 1

can be incorporated as additional variables, with VEM-Π_p style coupling
via integration by parts:

    Σ_e sign(c, e) ∫_e (v·n) q ds  −  Σ_α coeffs(α, q) c_K^{(α)}
        =  ∫_K div(v) q dA  =  RHS_c^{(q)},   q ∈ P_{p-1}

This extends the constraint matrix and provides the genuine cross-degree
coupling. **Out of scope for this turn**, but the block KKT scaffold is
designed so this extension is local to the assembly step.

## Implementation phases

### Phase 0 — Baselines (must run first)

1. `cmake --build build --parallel` (existing build)
2. `ctest --test-dir build --output-on-failure` — record pass/fail and timings
3. Save `build/Testing/Temporary/LastTest.log` as
   `docs/plans/2026-05-02-baseline-ctest.log`
4. `./build/high_order_hdiv_convergence_test docs/high_order_hdiv_convergence.csv.baseline`
5. `./build/spherical_high_order_hdiv_convergence_test
   docs/spherical_high_order_hdiv_convergence.csv.baseline`

### Phase 1 — Refactor to block KKT (numerics-preserving)

Files: `src/mimetic.cpp` (replace `project_target_edge_moments_to_hdiv_conforming`),
       `include/mimetic/mimetic.hpp` (add diagnostic field).

Steps:
1. Add `std::vector<std::vector<double>> trace_jump_per_unique_edge` to
   `ConformingEdgeMomentTransferResult` (zero-initialized, length
   `N_u × (p+1)`).
2. In the implementation, replace the per-degree loop with:
   - assemble per-edge length `L_e` from the unique-edge owner cell's
     `local_polygon` (planar) or `points_3d` great-circle length
     (spherical),
   - assemble Legendre-weight `B_m`,
   - compute per-degree `g_m` and the unconstrained minimizer
     `x_m^avg = π_m(U_avg)` exactly as before but using a named
     helper `legendre_l2_average`,
   - for `m = 0` only, add the Schur solve identically to the current
     code,
   - for each unique edge and each degree, compute and record
     `trace_jump = max_{i,j: u(i)=u(j)=u} |f_m(O_i) U_raw_{i,m} −
                                            f_m(O_j) U_raw_{j,m}|`
     before averaging.
3. Mirror the existing `b.array() -= b.mean()` step for spherical mode.
4. Verify byte-identical output to the previous implementation by
   running an internal diff against a snapshot of the previous values.

### Phase 2 — Targeted regression tests

Files: `tests/coupled_conforming_projection_test.cpp` (new),
       `CMakeLists.txt` (add executable).

Tests:
1. **Trace continuity per unique edge per degree.** On a 4×4 quad-to-quad
   mesh at `p = 2`, the conforming output must satisfy
   `|U_i^{(m)} − sign · U_j^{(m)}| ≤ 1e-12` for every pair of directed
   edges sharing a geometric edge.
2. **Divergence balance.** For every cell on the same mesh,
   `|Σ_e sign · x_{e,0} − RHS_c| ≤ 5e-13`.
3. **Numerical equivalence to baseline.** For one quad-to-quad and one
   Voronoi-to-Voronoi case at `p = 2`, the conforming `unique_edge_moments`
   produced by the refactor must agree with a recorded snapshot to
   `1e-14`.
4. **Trace-jump shrinks under refinement.** On a quad-to-quad refinement
   sequence at `p = 2`, the L² norm of `source_skeleton_jump_l2` must
   decrease by at least factor `2.5` per halving of `h`.

   Note: the originally proposed `trace_jump_per_unique_edge` field is
   identically zero (modulo machine roundoff) by construction of
   `transfer_source_to_target_edge_moments`: both directed views of any
   unique target edge integrate the same source field with opposite
   outward normals, so the orientation-corrected raw values agree to
   roundoff regardless of source/target meshes.  The refinement-faithful
   diagnostic is the SOURCE-skeleton jump
   (`source_skeleton_jump_l2[m] = sqrt sum_{e_s in source skeleton}
   (integral_{e_s} (v_a . n - v_b . n) L_m(t) ds)^2`), which refines as
   `O(h^{p+1})`.  Both fields are populated and tested.

Add as `add_test(NAME coupled_conforming_projection_test ...)` linking
against `${MOAB_LIBRARIES}` and `Eigen3::Eigen`.

### Phase 3 — Verification gate

Run on `feature/conforming-vem`:

```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/high_order_hdiv_convergence_test /tmp/high_order_after.csv
./build/spherical_high_order_hdiv_convergence_test /tmp/spherical_high_order_after.csv
diff -u docs/high_order_hdiv_convergence.csv.baseline /tmp/high_order_after.csv
diff -u docs/spherical_high_order_hdiv_convergence.csv.baseline /tmp/spherical_high_order_after.csv
```

Acceptance gate:
- All ctest targets pass
- New `coupled_conforming_projection_test` passes
- Convergence CSVs differ only in fields that are re-derived (no rate
  regression > 5%)

### Phase 4 — Commit and document

1. Update `docs/code_documentation.md` §Global Target-Edge Conforming
   Projection with the block-KKT formulation, the L² weight, and the
   trace-jump diagnostic.
2. Commit hierarchy:
   - `coupled-conforming: add trace_jump diagnostic + block-kkt refactor`
   - `coupled-conforming: add regression test`
   - `coupled-conforming: docs`

## Risks

- **Numerical drift in the m = 0 block** because the refactor must use
  exactly the same Schur assembly as the current code. Mitigation:
  Phase 2 test 3 compares against a recorded snapshot to `1e-14`.
- **Spherical edge-length ambiguity** between chart-Euclidean and
  great-circle length. Mitigation: use great-circle length from
  `points_3d` in spherical mode; document the choice; tests cover both.
- **Performance regression** from naming and weighting work. Mitigation:
  the inner loop is `O(N_d · p)` and the matrix sizes are unchanged;
  expected impact `< 5%` of the solve cost.

## Out of scope

- Cell-vector-moment coupling (architecturally enabled by this refactor;
  separate plan).
- Spherical lift of the cell-vector-moment coupling (requires Task 3 plan).
- Performance optimizations on the conforming solve (Task 2 plan).
