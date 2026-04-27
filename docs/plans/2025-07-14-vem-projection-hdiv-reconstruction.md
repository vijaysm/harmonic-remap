# Plan: H(div)-VEM Polynomial Projection for Topology-Independent High-Order Reconstruction

**Date**: 2025-07-14  
**Branch**: `feature/vem-hdiv-projection`  
**Worktree**: `~/worktrees/Mimetic-SphPoly/vem-hdiv-projection`  
**Status**: Approved for implementation  

---

## 1. Problem Statement

The current `PlanarMomentInterpolator` achieves near-optimal convergence rates on
structured quad meshes but systematically underperforms on irregular Voronoi polygons
for polynomial order p ≥ 2:

| Domain | p=1 | p=2 | p=3 | Expected |
|--------|-----|-----|-----|----------|
| Quad→Quad | 2.26 | 3.39 | 4.46 | 2, 3, 4 |
| Voronoi→Voronoi | **1.89** | **2.77** | **3.34** | 2, 3, 4 |
| Cubed-sphere (m0) | 1.75 | 3.08 | 4.57 | 2, 3, 4 |
| Spherical Voronoi (m0) | 2.05 | **2.52** | **3.62** | 2, 3, 4 |

The goal is to achieve O(h^{p+1}) convergence rates **independent of polygon topology
and mesh irregularity**, under standard star-shapedness and mesh-regularity assumptions.

## 2. Root Cause Analysis

### Root Cause 1: Non-uniform conditioning of the boundary trace operator

The map T_K from basis coefficients to edge-normal Legendre moments has singular
values that depend on polygon geometry. On Voronoi cells with short edges, clustered
orientations, and varying edge counts, higher-order edge moments (m ≥ 1) become
weakly observable for certain basis directions — particularly divergence-free bubble
modes. The constrained LS/KKT solve amplifies these weakly-observed directions.

**Evidence**: The `resolved_cell_moment_order()` function (line 1156 of `mimetic.cpp`)
heuristically adds cell vector moments when edge constraints alone are insufficient,
but this does not guarantee uniform conditioning.

### Root Cause 2: Algebraic (not geometric) basis decomposition

The split basis decomposes [P_p]² into divergence-particular, harmonic-gradient, and
bubble modes algebraically, then orthogonalizes against the bulk Gram matrix G_raw.
This ensures interior linear independence but NOT uniform observability through edge
moments. On irregular polygons, the decomposition can become nearly rank-deficient
when viewed through the boundary trace operator.

### Root Cause 3: Reconstruction vs. projection

The method explicitly solves for ALL coefficients of a pointwise-evaluable field from
boundary data. Proven polygonal methods (VEM, MFD) instead compute a polynomial
*projector* from DOFs using integration-by-parts identities, and add stabilization
only on the unresolvable kernel. This ensures topology-independent stability by
construction.

## 3. Proposed Solution: H(div)-VEM Polynomial Projection

Replace the constrained LS/KKT coefficient solve in `reconstruct_source_polygon()`
with a VEM-style polynomial projection + stabilization.

### 3.1 Mathematical Framework

**Local VEM space** (Beirão da Veiga, Brezzi, Marini, Russo 2016):

For a polygon K with edges e_1, ..., e_N, define:

```
V_{p}(K) = { v ∈ H(div; K) ∩ H(rot; K) :
              v · n|_e ∈ P_p(e)  ∀ edge e,
              div v ∈ P_{p-1}(K) }
```

**DOFs** (already implemented in the current code):
- D1: Edge normal moments  ∫_e (v · n_e) L_m(s) ds,  m = 0, ..., p
- D2: Cell vector moments   ∫_K v · q dA,  q ∈ [P_{p-2}]²  (when k ≥ 2)

These are exactly the DOFs the current `PlanarMomentInterpolator` already stores.

**Polynomial projection** Π_K : V_p(K) → [P_p(K)]²:

For any q ∈ [P_p(K)]², the projection satisfies:

```
∫_K (Π_K v) · q dA = ∫_K v · q dA
```

This is computed from DOFs using integration by parts (Green's formula):

```
∫_K v · q dA = ∫_∂K (v · n)(q · x_⊥) ds - ∫_K (div v)(some potential of q) dA
```

where the boundary integral uses edge normal moments (D1) and the volume integral
uses the known polynomial divergence (computable from D1 via Gauss' theorem for
the zeroth moment and from D2 for higher-order divergence content).

**Stabilization** S_K on ker(Π_K):

```
S_K(u, v) = h_K^{-2} Σ_i dof_i(u - Π_K u) · dof_i(v - Π_K v)
```

This provides spectral equivalence on the kernel, with constants independent of
polygon topology under star-shapedness.

### 3.2 Key Property: Commuting Diagram

The VEM projection commutes with divergence:

```
div(Π_K v) = Π_{p-1}(div v)
```

where Π_{p-1} is the L² projection onto P_{p-1}(K). This is the property that
guarantees optimal H(div) convergence independently of polygon shape.

### 3.3 What Changes in the Code

**ONLY the reconstruction core changes.** Everything else stays:

| Component | Change? | Details |
|-----------|---------|---------|
| DOF storage (directed_source_moments_) | NO | Already stores edge Legendre moments |
| Cell vector moments | NO | Already stores cell moments |
| Edge clipping and transfer | NO | Will evaluate Π_K v (a polynomial) on clipped segments |
| Conforming projection | NO | Target skeleton solve unchanged |
| Conservation (m=0 hard constraint) | NO | VEM div-commuting property preserves this |
| Spherical gnomonic/Piola | MINOR | VEM projection in chart coordinates, same Piola mapping |
| `reconstruct_source_polygon()` | YES | Replace LS/KKT with VEM projection formula |
| `MomentReconstruction` struct | MINOR | May store projection coefficients instead of raw coefficients |
| `velocity()` evaluation | MINOR | Evaluates Π_K v polynomial instead of raw basis expansion |
| `build_split_moment_basis()` | MAYBE REMOVE | No longer needed if using VEM projection |

### 3.4 Isolation Strategy for Parallel Development

The implementation is on a **separate branch** (`feature/vem-hdiv-projection`) in
a **separate worktree** (`~/worktrees/Mimetic-SphPoly/vem-hdiv-projection`).

**Files that will be modified:**
- `src/mimetic.cpp`: New VEM projection functions (added alongside existing code)
- `include/mimetic/mimetic.hpp`: Possible new reconstruction mode enum or options
- `tests/`: New VEM-specific convergence test

**Conflict avoidance strategy:**
- The VEM reconstruction will be implemented as a NEW code path selectable via
  `MomentMethodOptions`, not as a replacement of the existing LS path
- The existing `reconstruct_source_polygon()` remains untouched
- A new method or option flag selects VEM vs. LS reconstruction
- Tests compare VEM and LS rates side-by-side
- Merge will be straightforward: additive changes, no destructive edits

## 4. Implementation Phases

### Phase 1: Diagnostic — Singular Value Analysis (Day 1)

Add a diagnostic function that computes and reports the singular values of the
boundary moment operator T_K for each source cell. This quantifies exactly where
conditioning degrades on Voronoi cells and confirms the root cause.

```cpp
// New diagnostic in mimetic.cpp
struct TraceOperatorDiagnostic {
    int num_edges;
    int basis_dim;
    int constraint_rows;
    double condition_number;
    std::vector<double> singular_values;
};

TraceOperatorDiagnostic diagnose_trace_operator(
    moab::Core& mb, moab::EntityHandle polygon,
    const MomentMethodOptions& options,
    const GeometryOptions& geom_options);
```

**Acceptance**: Condition numbers should be systematically worse on Voronoi cells
than quads for p ≥ 2.

### Phase 2: VEM Projection Core — Planar p=1 (Day 1-2)

Implement the VEM polynomial projection Π_K for p=1 on planar polygons.

```cpp
// New function in mimetic.cpp
MomentReconstruction vem_reconstruct_source_polygon(
    moab::Core& mb,
    moab::EntityHandle polygon,
    const MomentMethodOptions& options,
    const GeometryOptions& geom_options,
    const std::map<std::pair<moab::EntityHandle, std::size_t>,
                   std::vector<double>>& edge_moments,
    const std::map<moab::EntityHandle,
                   std::vector<Eigen::Vector2d>>& cell_moments);
```

Key steps:
1. Compute the polynomial projection Π_K v from edge moments using Green's formula
2. The result is stored as coefficients of a polynomial in [P_p]²
3. `velocity()` evaluates this polynomial exactly (same as current monomial evaluation)

**Acceptance**: p=1 rates on Voronoi meshes should match or exceed current rates.

### Phase 3: VEM Projection — Planar p=2,3 (Day 2-3)

Extend to p=2 and p=3. The integration-by-parts formula scales naturally with degree.
Cell moments (D2 DOFs) become essential for k ≥ 2.

**Acceptance**: 
- p=2 Voronoi rates ≥ 3.0 (vs current 2.77)
- p=3 Voronoi rates ≥ 4.0 (vs current 3.34)

### Phase 4: VEM Stabilization (Day 3)

Add the stabilization term S_K for under-determined systems (many-edged polygons
where DOFs exceed projectable polynomial content).

### Phase 5: Spherical Extension (Day 4)

Extend VEM projection to gnomonic chart coordinates with Piola metric.
The VEM projection is computed in the chart; the Piola mapping is applied
during transfer, same as now.

### Phase 6: Regression and Comparison (Day 4-5)

Run full convergence study comparing VEM and LS paths side by side.
Write comparison data to CSV.

## 5. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| VEM integration-by-parts formula needs careful implementation | MEDIUM | Test on known polynomial fields first |
| Stabilization parameter sensitivity | LOW | Standard VEM theory gives mesh-uniform bounds |
| Conservation regression | LOW | Div-commuting property ensures conservation |
| Spherical Piola interaction | MEDIUM | Validate planar first; gnomonic chart is local |
| Conflict with parallel development on main branch | LOW | Additive changes only; new code path selectable by option |
| Performance regression | LOW | VEM projection is one matrix solve per cell, similar cost to current KKT |

## 6. Key Literature References

1. Beirão da Veiga, Brezzi, Marini, Russo (2016). H(div) and H(curl)-conforming VEM. Numer. Math. 133.
2. Beirão da Veiga, Brezzi, Cangiani, Manzini, Marini, Russo (2013). Basic principles of VEM. M3AS 23(1).
3. Gyrya, Lipnikov (2008). High-order MFD on polygonal meshes. JCP 227.
4. Beirão da Veiga, Lipnikov, Manzini (2009). Convergence of high-order MFD. Numer. Math. 113.
5. Beirão da Veiga, Lipnikov, Manzini (2014). Arbitrary order mixed MFD. SIAM JSC.
6. Abgrall, Le Mélédo, Öffner (2021). General polytopal H(div)-conformal elements. ESAIM: M2AN 55.
7. Chen, Wang (2017). Minimal degree H(div) elements on polytopal meshes. Math. Comp. 86.
8. Arnold, Boffi, Falk (2005). Quadrilateral H(div) finite elements. SIAM JNA 42.
9. Palha (2021). A mimetic method for polygons. JCP 424.
10. Kreeft, Palha, Gerritsma (2011). Mimetic framework on curvilinear quadrilaterals. arXiv:1111.4304.
11. Lipnikov, Manzini (2016). High-order mimetic method on unstructured polyhedral meshes. JCP 327.

## 7. Files Modified (Conflict Surface)

The following files will be touched. Any parallel changes to these files should
be coordinated:

| File | Nature of change |
|------|-----------------|
| `src/mimetic.cpp` | ADD new VEM functions (~300-500 lines). Existing functions NOT modified. |
| `include/mimetic/mimetic.hpp` | ADD new option/enum for VEM mode. Existing API NOT modified. |
| `tests/vem_hdiv_convergence_test.cpp` | NEW file. No conflict possible. |
| `CMakeLists.txt` | ADD new test target. Minimal conflict surface. |

**Guarantee**: The existing `reconstruct_source_polygon()` LS/KKT path will NOT
be modified. The VEM path is a new alternative activated by an option flag.
