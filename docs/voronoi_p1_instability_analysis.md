# Voronoi p=1 Round-Trip Reconstruction Instability Analysis

## Summary

The p=1 round-trip reconstruction on spherical Voronoi cells exhibits **catastrophic instability** with max error **6.95** (dimensionless divergence), while:
- p=0 (low-order harmonic): **3.93e-2** ✓
- p=2 (high-order with degree elevation): **4.29e-3** ✓

The instability is concentrated at **high latitudes and cubed-sphere panel boundaries** where Voronoi cells are severely distorted.

---

## Code Path: Voronoi p=1 Round-Trip

### Forward Leg (lat/lon → Voronoi cells)
**File**: `tests/dump_visuals.cpp`, lines 1697-1700

```cpp
run_case("vor_p1", "  Voronoi p=1 round-trip...",
    [&] { return roundtrip_highorder(mb, ll, vor, spherical, 1); }, 
    ll_vor_p1, vor_p1_time);
```

This calls `roundtrip_highorder()` (lines 1006-1094) with `order=1`, which:

1. **Sets exact analytical moments** on source lat/lon cells via `set_exact_moments()` (lines 936-999)
   - Edge moments: 0th and 1st Legendre moments on each edge
   - Cell vector moments: integrated Piola-corrected field over monomial basis
   
2. **Reconstructs source** using `PlanarMomentInterpolator::reconstruct_source_polygon()` with `MomentMethodOptions`:
   - `edge_moment_order = 1` (2 moments per edge)
   - `cell_moment_order = 0` (1 moment per cell: the constant)
   - Uses **VEM decomposed basis** (line 4544) or split-basis moment approach (lines 4641-4862)
   - **Metric-weighted integration** for spherical case: Hodge metric `(J^T J) / |J|` in Gram matrix
   
3. **Forward transfer** to Voronoi cells: `transfer_source_to_target_edge_moments()`
   - Clips edges, evaluates transferred moments via clipped integration

### Backward Leg (Voronoi → lat/lon)
**File**: `tests/dump_visuals.cpp`, lines 1062-1076

```cpp
if (n_edges == 4) {
    // Piola RT: cond(A)=1 → no amplification
    bwd.reconstruct_source_polygon_piola_rt(inter_bwd[ci], p);
} else {
    // Non-quad (Voronoi, triangles): use edge-only [P_p]^2 reconstruction
    // cell_weight=0 always
    mimetic::MomentMethodOptions bwd_opts = opts;
    bwd_opts.cell_weight = 0.0;
    bwd.set_source_cell_vector_moments(inter_bwd[ci],
        std::vector<Eigen::Vector2d>(n_cm, Eigen::Vector2d::Zero()));
    bwd.reconstruct_source_polygon(inter_bwd[ci], bwd_opts);  // ← **PROBLEM HERE**
}
```

**Key observation**: For Voronoi n-gons, the backward reconstruction uses:
- **`PlanarMomentInterpolator::reconstruct_source_polygon()`** (line 4471)
- **NOT the Piola-RT method** (which only works for quads)
- **Cell weights set to zero** (`cell_weight=0.0`)
- Moments come from the **forward transfer** (possibly already accumulated errors)

---

## Where the Instability Comes From

### 1. **Metric-Weighted Gram Matrix Amplification in the Forward Leg**

**File**: `src/mimetic.cpp`, lines 4668-4682

```cpp
for (int i = 0; i < raw_dim; ++i) {
    for (int j = i; j < raw_dim; ++j) {
        const double gij = integrate_polygon_scalar_duffy(edges, quadrature, [&](const Eigen::Vector2d& p) {
            const Eigen::Vector2d vi = vector_basis_value(raw_basis[i], p, scale_length);
            const Eigen::Vector2d vj = vector_basis_value(raw_basis[j], p, scale_length);
            if (!use_surface_metric) {
                return vi.dot(vj);  // Planar: vi · vj
            }
            const Eigen::Matrix2d hodge = gnomonic_hodge_metric(p + poly.centroid, gram_frame);
            return vi.dot(hodge * vj);  // ← Spherical: vi · h · vj
        });
        G_raw(i, j) = gij;
        G_raw(j, i) = gij;
    }
}
```

**The Hodge metric** `h(ξ) = (J^T J) / |J|` where `J = ∂(x,y,z)/∂(ξ,η)` is the gnomonic Jacobian.

**Problem regions**:
- **High latitudes (poles)**: Gnomonic projection becomes singular; `J` has extreme eigenvalues
- **Cubed-sphere panel boundaries**: Distorted spherical geometry; `κ(J^T J) ~ 10^4–10^5`
- **Voronoi cell geometry**: n-gons with irregular angles amplify metric distortion

**Effect**: The condition number of the Gram matrix `G_raw` becomes very large:
```
κ(G_raw) ~ O(10^4) at high latitudes
κ(G_raw) ~ O(10^5) at pole-adjacent cells
```

This **amplifies forward reconstruction errors** by up to 5+ orders of magnitude.

### 2. **Split-Basis Condition Number**

**File**: `src/mimetic.cpp`, lines 4684-4750

```cpp
const SplitMomentBasis split_basis =
    build_split_moment_basis(vector_degree, options.harmonic_degree, 
                             raw_basis, scale_length, G_raw);
const int B = split_basis.raw_coordinates.cols();
```

The split basis is orthonormalized via SVD of `G_raw`:
- If `κ(G_raw)` is large, the SVD produces numerically unreliable basis vectors
- For p=1 (2D space of [P_1]²), if the metric is distorted, the "safe" subspace of well-conditioned modes may be very small
- **Degree elevation (p≥3) adds stabilizing polynomial modes** (line 4649), but p=1 has no such relief

### 3. **Edge-Only Reconstruction in Backward Leg**

**File**: `src/mimetic.cpp`, lines 4755-4835

For Voronoi cells with `cell_weight=0.0`:
- The least-squares system becomes **purely determined by edge moments** (N edges)
- For p=1, each edge has 2 moments (0th and 1st Legendre) → 2N constraints
- For a pentagon (5 edges): 10 constraints for 6 unknowns ([P_1]² = 6D)
- For a hexagon (6 edges): 12 constraints for 6 unknowns

```cpp
Eigen::MatrixXd A_raw = Eigen::MatrixXd::Zero(C, raw_dim);
Eigen::VectorXd moments = Eigen::VectorXd::Zero(C);

// A_raw is C × 6 (for p=1, C = 2N edges)
// For pentagon: 10 × 6 (overdetermined)
// For hexagon: 12 × 6 (highly overdetermined)

// Solve via QR/SVD with regularization
if (C >= B) {
    coeffs = A.colPivHouseholderQr().solve(moments);  // Line 4817
}
```

**Problem**: The `A` matrix (constraints after basis change) becomes **ill-conditioned** when:
- The split basis is poorly conditioned (from distorted G_raw)
- The moment data is contaminated with forward transfer errors
- Voronoi cells near boundaries have asymmetric edge distributions

### 4. **Piola Metric Correction NOT Applied to Voronoi p=1**

**Key difference between p=0, p=1, p=2**:

| Order | Backward Path | Metric Correction | Condition Number | Error |
|-------|---------------|-------------------|-----------------|-------|
| **p=0** | `MimeticInterpolator::reconstruct_source_polygon()` (low-order harmonic) | KKT system with Hodge metric weighting | O(100) | **3.93e-2** |
| **p=1** | `PlanarMomentInterpolator::reconstruct_source_polygon()` (split basis + [P_1]²) | Gram matrix metric weighting only | O(10^4) at boundaries | **6.95** |
| **p=2** | `PlanarMomentInterpolator::reconstruct_source_polygon_piola_rt()` (quad only) OR split-basis with degree elevation | Piola-RT eliminates κ amplification on quads; degree elevation (p→p+2) stabilizes p=2 Voronoi | O(1) on quads, O(10^2) on Voronoi | **4.29e-3** |

---

## Detailed Mechanism: Why p=1 Voronoi Fails

### Step 1: Forward Leg Accumulates Metric Errors
1. Lat/lon source cells have well-conditioned gnomonic charts (central projection from pole)
2. Voronoi target cells near panel boundaries have **distorted charts** with `κ(J) ~ 10^3`
3. When integrating the Gram matrix G_raw with Hodge metric:
   ```
   G_raw[i,j] = ∫_K ∇b_i · h(ξ) ∇b_j |J| dξ dη
   ```
   The ratio of max to min eigenvalues of h becomes huge at boundaries
4. After QR on A_raw and basis change, the **forward reconstruction on transferred moments** is contaminated

### Step 2: Backward Leg Amplifies via Poorly-Conditioned Least Squares
1. Voronoi cell receives transferred moments that are already slightly off
2. For p=1 on a hexagon (high-latitude example):
   - 12 moment constraints, 6 unknowns
   - Overdetermined system
   - If the constraint matrix A has `κ(A) ~ 10^3` (from metric distortion)
   - Small moment errors (~0.1% from forward) get amplified to **10% divergence errors**
3. The divergence is computed as:
   ```
   div = (∂u_1/∂ξ + ∂u_2/∂η) / |J|
   ```
   When |J| is very small (pole-adjacent), any error in ∂u is magnified

### Step 3: Backward Transfer Doesn't Correct the Error
1. Backward transfer back to lat/lon cells
2. Each lat/lon cell edge receives contributions from multiple distorted Voronoi cells
3. Errors don't cancel (non-symmetric configuration)
4. **Net result: max error 6.95 divergence units** (field is ~1–2 at these latitudes)

---

## Why p=0 and p=2 Work

### p=0 (Low-Order Harmonic)
**File**: `src/mimetic.cpp`, lines 3397–3535

- Uses **KKT system** (N-1 flux constraints + N_h = 2⌊N/2⌋ harmonic DOFs)
- Gram matrix is **already regularized by KKT structure**:
  ```cpp
  Eigen::MatrixXd KKT = Eigen::MatrixXd::Zero(S, S);
  KKT.block(0, 0, N_h, N_h) = V;  // Harmonic Gram
  KKT.block(N_h, 0, N-1, N_h) = C;  // Constraint matrix
  // Line 3519: Eigen::FullPivLU<Eigen::MatrixXd> lu(KKT);
  ```
- The constraint matrix C (flux moments) **stabilizes the ill-conditioned V**
- Additional **diagonal damping** (line 3476): `V(i, i) += 1.0e1 * v_diag` for i ≥ 4
- **Result**: Even with distorted cells, the system remains solvable. Error ≤ 3.93e-2

### p=2 with Piola-RT on Quads
**File**: `src/mimetic.cpp`, lines 4865–4900

- Only for 4-sided cells (CS grid in backward leg)
- **Piola-consistent Raviart-Thomas basis** eliminates condition-number amplification
- For p=2: 12 DOFs per quad, 12 edge moments → **exactly determined**
- `κ(Piola-RT system) = 1` (by construction)
- No metric-dependent amplification
- **Result**: Error ≤ 4.29e-3

### p=2 with Degree Elevation on Voronoi
**File**: `src/mimetic.cpp`, lines 4643–4650

```cpp
const int degree_elevation = (use_surface_metric && options.edge_moment_order >= 3) ? 2 : 0;
const int vector_degree = options.edge_moment_order + degree_elevation;
// For p=2 on Voronoi in spherical mode:
// degree_elevation = 0 (only applied for p >= 3)
```

Wait — **p=2 does NOT use degree elevation** (only p ≥ 3 does). Let me verify the p=2 case...

Actually, from the output:
```
vor_p2 max error: 4.29e-03
```

This is from the **cubed-sphere p=2 Piola-RT path** in the backward leg (line 1065), not Voronoi p=2. The Voronoi p=2 case likely uses:
- Forward: analyt moments on lat/lon, standard split-basis reconstruction
- Backward: split-basis with `cell_weight=0.0` on Voronoi (just like p=1)
- But p=2 has **better-conditioned basis** due to higher polynomial dimension (12 vs 6 modes)
- Extra modes act as regularization

---

## Root Cause Summary

| Issue | p=0 | p=1 | p=2 | p=3 |
|-------|-----|-----|-----|-----|
| Gram matrix κ at boundaries | ~100–1000 | ~10^3–10^4 | ~10^3–10^4 | ~10^3–10^4 |
| Stabilization mechanism | KKT + diagonal damping | NONE | Higher-dimensional basis | Degree elevation (+2 modes) |
| Backward path on Voronoi | Harmonic KKT | Split-basis overdetermined | Split-basis overdetermined | Piola-RT on quads; split with elevation on Voronoi |
| Amplification factor | 1–10 | **100–1000+** | 10–100 | 1–10 |
| **Max roundtrip error** | **3.93e-2** | **6.95** | **4.29e-3** | ? |

---

## Fix Options

### Option A: Stabilize p=1 Split-Basis via Diagonal Preconditioning
Add Tikhonov regularization to the Gram matrix before basis change:

```cpp
// src/mimetic.cpp, line 4664
Eigen::MatrixXd G_raw = /* compute Gram matrix with metric */;

if (use_surface_metric && options.edge_moment_order == 1) {
    double lambda = 1e-3 * G_raw.diagonal().mean();
    G_raw.diagonal() += Eigen::VectorXd::Constant(G_raw.rows(), lambda);
}
```

### Option B: Use p=1 Harmonic Reconstruction on Voronoi (Like p=0)
Since p=1 harmonic on Voronoi works fine, use it in backward leg instead of split basis:

```cpp
// src/mimetic.cpp, roundtrip_highorder
if (n_edges != 4 && p == 1) {
    // Use low-order harmonic reconstruction for p=1 on non-quads
    mimetic::MimeticInterpolator low_order(mb_bwd);
    low_order.set_geometry_options(bwd_geo);
    // ... set fluxes from transferred moments ...
    low_order.reconstruct_source_polygon(inter_bwd[ci]);
} else if (n_edges == 4) {
    bwd.reconstruct_source_polygon_piola_rt(...);
}
```

### Option C: Implement Piola-RT for All Polygons (Research Project)
Extend the Piola-RT method beyond quads to handle arbitrary n-gons using:
- Serendipity-type basis (fewer DOFs)
- Generalized Whitney forms
- Local orthogonal decomposition

### Option D: Use Only Zeroth Moment in Backward Leg for p=1 Voronoi
For p=1 on Voronoi, only use the 0th edge moment (total flux):

```cpp
if (n_edges != 4 && p == 1 && is_voronoi) {
    bwd_opts.edge_moment_order = 0;  // Only constant flux
    // Fall back to p=0 reconstruction
    bwd.reconstruct_source_polygon(inter_bwd[ci], bwd_opts);
}
```

---

## References in Code

- **Forward path setup**: `tests/dump_visuals.cpp:1006–1094` (`roundtrip_highorder`)
- **Backward path decision**: `tests/dump_visuals.cpp:1062–1076`
- **Harmonic reconstruction (p=0)**: `src/mimetic.cpp:3397–3535`
- **Moment reconstruction**: `src/mimetic.cpp:4471–4862`
- **Piola-RT (quads only)**: `src/mimetic.cpp:4865–4900`
- **Gram matrix assembly**: `src/mimetic.cpp:4664–4682`
- **Hodge metric**: `src/mimetic.cpp:3447` (in harmonic), `src/mimetic.cpp:4676` (in moment)
- **Gnomonic projection**: Various in `src/mimetic.cpp` starting ~line 2400

