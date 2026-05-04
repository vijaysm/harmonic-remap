# Spherical Lift of the VEM H(div) Path

**Date:** 2026-05-02
**Branch:** `feature/conforming-vem`
**Predecessor docs:** `docs/code_documentation.md` §VEM H(div) Projection
(planar derivation), §Degree-Elevated Basis (rational-correction
treatment for SplitBasis on the sphere)
**Reviewed gap:** `docs/code_documentation.md:1132` "Limitations of the
Current Implementation": "Planar only."

## Objective

Extend the existing `ReconstructionMode::VemProjection` path from planar
to spherical (gnomonic chart) geometry, with the same correctness and
conditioning properties as the planar VEM. After this task:

- Spherical VEM produces full convergence rates `O(h^{p+1})` on cubed-
  sphere and Voronoi-patch refinement studies for `p ∈ {1, 2, 3}`.
- The Piola pullback's rational `1/|ξ|^3` factor is captured by the
  same degree-elevation mechanism that the SplitBasis path already uses
  for `p ≥ 3` on cubed-sphere cells.
- Conditioning is at least as good as the planar VEM on equivalent
  Voronoi cells.
- All existing spherical tests continue to pass.

## Theoretical formulation

### Setting

- Source cell `K_s` lives on the unit sphere; we work in its gnomonic
  chart `(ξ, η)` (already implemented as `GnomonicFrame` and
  `project_gnomonic`).
- Surface vector field `u : S^2 → T S^2`.
- Chart vector field `û = J^{-1} (J^T J)^{1/2} u` is the contravariant
  Piola pullback (already implemented as
  `pullback_contravariant_piola`).
- The Hodge metric is `h(ξ) = J^T J / |J|`, equal to
  `gnomonic_hodge_metric(ξ, frame)` in code.
- The metric area scale is `dA_S = |J(ξ)| dξ dη`,
  `gnomonic_area_scale(ξ, frame)`.

The Piola transform makes flux moments invariant:
`∫_{e_S} u · n_S dℓ_S = ∫_{e_chart} û · n_chart dℓ_chart`.
This is the exact property the SplitBasis path already exploits.

### VEM-Π_p in the chart with Hodge weight

The local VEM space is spanned by `[P_p(ξ, η)]^2`. The mass matrix is

    M_{ij} = ∫_{K̂_s} φ_i · h(ξ) · φ_j  dξ dη

i.e. the inner product is taken with respect to the Hodge metric so the
Piola-pulled chart objective is the L²(S²) objective on the sphere.

The right-hand side uses the same VEM identities as the planar path,
except every `∫_K v · φ dA` is interpreted as
`∫_{K̂} v̂ · h(ξ) · φ dξ dη` and every `∫_∂K (v·n) ψ ds` is taken on
the chart edge (great-circle arc-length parametrized for moments — this
already matches `transfer_source_to_target_edge_moments`).

The integration-by-parts identities
([BdV14] §4, eq. (4.5)) become

    ∫_{K̂} v̂ · h ∇ψ dξ dη = ∫_{∂K̂} (v̂ · n_chart) ψ dℓ_chart
                              − ∫_{K̂} divergence_h(v̂) ψ |J| dξ dη

where `divergence_h` is the sphere-divergence expressed in chart
coordinates. By the Piola identity, `divergence_S(u) |J| =
divergence_flat(û)`, so the volume term reduces to the flat divergence
times the moment, weighted by the chart area. This means the existing
divergence-recovery infrastructure (`vem_reconstruct_divergence`) lifts
unchanged provided the cell-vector-moment DOFs are interpreted as
`∫_K̂ û · ê x^a y^b |J| dξ dη` — i.e., physical-area-weighted chart
moments rather than chart-area moments.

### Degree elevation for `p ≥ 3`

The exact surface field `û` is rational in `(ξ, η)` because `|J| ∼
(1 + ξ² + η²)^{−3/2}`. A polynomial basis of degree `p` cannot capture
this rational factor exactly, leaving an `O(h^3)` residual that
dominates `O(h^{p+1})` for `p ≥ 3`. The SplitBasis path already
resolves this by elevating the polynomial degree to `p + 2` for
`p ≥ 3` in spherical mode; the same elevation must be applied to the
spherical VEM path.

The elevated VEM space `[P_{p+2}]^2` is determined by the same
`(p+1)`-degree edge moments and the cell vector moments
`∫_K̂ û · ê x^a y^b |J| dξ dη` for `|α| ≤ p − 1`. The extra dimensions
are resolved by the L²-min objective (mass matrix is SPD and the
elevated basis has `O(h^{degree+1})` approximation), exactly as in the
SplitBasis path.

## Non-negotiable requirements

- Conservation tolerance `5e-13` preserved.
- Spherical SplitBasis tests continue to pass at current rates.
- Spherical VEM `p = 1`: rate ≥ `1.7` (cubed-sphere), `≥ 2.0`
  (Voronoi-patch).
- Spherical VEM `p = 2`: rate ≥ `3.0` (cubed-sphere), `≥ 2.5`
  (Voronoi-patch).
- Spherical VEM `p = 3`: rate ≥ `4.0` (cubed-sphere with degree
  elevation), `≥ 3.5` (Voronoi-patch).
- Conditioning: `κ(M) ≤ 10^4` on the worst cell at `p = 3`.

## Implementation phases

### Phase 0 — Baselines

1. Run all spherical tests; record timings and outputs.
2. Save current `ReconstructionMode::SplitBasis` rates from
   `docs/spherical_high_order_hdiv_convergence.csv` for comparison.

### Phase 1 — Hodge-weighted polygon monomial integrals

Files: `src/mimetic.cpp` (extend `polygon_monomial_integral`).

1. Add an overload `polygon_monomial_integral_weighted(vertices, a, b,
   weight_fn)` that takes a `std::function<double(Eigen::Vector2d)>`
   integrand multiplier.
2. Use Duffy fan triangulation with sufficient quadrature degree
   (10-point Gauss-Legendre, matching the metric-weighted SplitBasis
   path).
3. Default weight is `1`, recovering the existing behavior.
4. Add a test fixture comparing weighted vs unweighted integrals on a
   known polygon and a known weight to `1e-12`.

### Phase 2 — Hodge-weighted VEM mass matrix

Files: `src/mimetic.cpp` (extend `vem_mass_matrix`).

1. Add a `metric_weighted` flag to the VEM mass-matrix builder
   following the pattern in `reconstruct_source_polygon`.
2. When the flag is set and `options.mode == SphericalGnomonic`,
   build `M_{ij}` using the Hodge weight `h(ξ + centroid)`.
3. Verify the matrix is still SPD (LDLT succeeds).
4. Test: in planar mode the weighted and unweighted matrices coincide.

### Phase 3 — Hodge-weighted VEM RHS

Files: `src/mimetic.cpp` (extend `vem_projection_rhs`).

1. Replace the chart-area `∫_K̂ x^a y^b dA` table with the Hodge-area
   `∫_K̂ x^a y^b |J| dξ dη` table when `metric_weighted` is on.
2. Replace `vem_reconstruct_divergence` accordingly: divergence on the
   sphere is `div_h(v̂) = (1/|J|) div_flat(v̂ |J|)`, so its L²(S²)
   projection onto `P_{p−1}` uses the Hodge area as weight.
3. Test: on a planar mesh the result is unchanged.
4. Test: on a single spherical cell, the projection of a known surface
   gradient is recovered to `1e-12`.

### Phase 4 — Spherical reconstruction wiring

Files: `src/mimetic.cpp` (extend
`PlanarMomentInterpolator::reconstruct_source_polygon`'s VEM branch).

1. When `is_spherical()` and `mode == VemProjection`, set
   `metric_weighted = true` for the VEM assembly.
2. Apply the existing degree-elevation rule (`p → p + 2` for `p ≥ 3`)
   to the VEM polynomial space.
3. Cell-vector-moment input must be interpreted as Hodge-weighted
   chart moments. Update `set_source_cell_vector_moments` documentation
   and provide a helper `vem_cell_moment_evaluator` that integrates a
   user-supplied chart field against the Hodge weight.

### Phase 5 — Spherical VEM regression tests

Files: `tests/spherical_vem_test.cpp` (new), `CMakeLists.txt`.

Tests:
1. **Single-cell exact polynomial recovery** on a spherical Voronoi
   patch at `p = 1, 2, 3`, error `≤ 1e-10`.
2. **Conservation on cubed-sphere `p = 1` transfer**, residual `≤ 5e-13`.
3. **Convergence study** on cubed-sphere refinement levels
   `{4, 8, 16}` at `p = 1, 2, 3`. Rates must meet the requirements
   above.
4. **Conditioning check**: `κ(M) ≤ 1e4` for every source cell on the
   `level-2` cubed-sphere mesh at `p = 3`.

### Phase 6 — Verification gate

```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure
ctest --test-dir build -R spherical_vem_test --output-on-failure
ctest --test-dir build -R spherical_high_order --output-on-failure
```

Acceptance:

- All ctest pass.
- Spherical VEM rates meet or exceed the Non-negotiable Requirements.
- SplitBasis spherical numbers unchanged.

### Phase 7 — Documentation

1. Add a "Spherical Lift" subsection to
   `docs/code_documentation.md` §VEM H(div) Projection.
2. Document the Hodge-weighted mass matrix, the cell-moment
   interpretation, the degree elevation rule, and the regression
   results.
3. Add bibliography entries for the Piola/spherical metric references
   if not present.

## Risks

- **Hodge weight non-polynomial integrand.** Mitigation: 10-point
  Gauss-Legendre Duffy already used by the SplitBasis spherical path
  — adopt identical quadrature.
- **Cell-vector-moment input convention break.** Mitigation: keep the
  planar convention unchanged; introduce a clearly labeled spherical
  helper, with the convention documented at the call site.
- **Conditioning growth at `p = 3`.** Mitigation: degree elevation
  raises basis size, so guard with the SVD fallback already present in
  the planar VEM solve. Tests verify `κ(M) ≤ 10^4` worst-case.
- **Cubed-sphere face-corner cells** subtend the largest solid angle
  and stress the degree-elevation. Mitigation: include the corner
  cells explicitly in the regression set.

## Out of scope

- Spherical patch-recovery → VEM. The patch-recovery least-squares
  step would itself need metric-aware weighting; deferred.
- Performance profiling of the spherical VEM path. Initial
  implementation may be slower than SplitBasis; optimization deferred.
- Coupling the spherical VEM cell-vector-moment output back into the
  conforming target-edge projection (covered by the conforming-projection
  plan's Phase 4 follow-up).
