Thorough Algorithmic Review

I've read every line of the implementation (mimetic.hpp, mimetic.cpp), all documentation (code_documentation.md, README.md, the spherical accuracy improvement plan), and the
convergence data. Here is a structured review of the method, algorithm, and implementation, with specific areas where convergence and results can be improved.

---
1. Summary of Current State

The code implements a conservative mimetic edge-to-edge interpolation kernel on polygonal meshes with two backends (planar and spherical gnomonic). The core idea:
- Reconstruct a vector field on each source cell from directed edge fluxes using a constrained harmonic-basis solve (KKT saddle-point)
- Transfer to target edges by clipping and evaluating the reconstruction on subsegments
- Optionally project onto a globally H(div)-conforming target skeleton

The higher-order path (p=1,2,3) uses a split basis: polynomial-divergence + harmonic-gradient + divergence-free bubbles, with Legendre edge moments.

---
2. Identified Improvement Areas

A. Quadrature Accuracy Mismatch (High Impact)

Problem: The triangle integration rule used for the Gram matrix V and moment vector M in the low-order reconstruction (integrate_triangle_scalar, lines 86-108 of header) is a 7-point
rule exact for degree-5 polynomials. However, the integrands involve products of harmonic basis gradients like ∇P_k · ∇P_j, which for k=2, j=2 are degree-2 polynomials in each
gradient component times degree-2, giving a degree-4 integrand. This is fine for k_max=2. But for polygons with many edges (N≥6), k_max=N/2≥3, and the Gram matrix integrands ∇P_3 ·
∇P_3 are degree-4 polynomials -- still within the degree-5 exactness.

However, the metric-weighted spherical path multiplies by the Hodge metric gnomonic_hodge_metric(p + poly.centroid, frame) (line 1840). This Hodge metric is a rational function of the
chart coordinates, not polynomial. The 7-point triangle rule has no polynomial exactness guarantee for these rational integrands. On large cells (coarse cubed-sphere), the metric
varies significantly across the cell, and a 7-point rule may introduce O(h^2) quadrature error in the Gram matrix itself.

Recommendation: Use higher-order quadrature for metric-weighted integrals. Either:
- Switch to a Duffy-style rule with the gauss10_rule for the metric-weighted path (already available)
- Or implement adaptive subdivision of the triangle fan when metric_weighted=true

This could explain the spherical p=3 convergence rate degradation (2.42 vs expected ~4).

B. Spherical p=3 Convergence Rate Degradation (High Impact)

Observation: Cubed-sphere p=3 moment-0 rate is 2.42, far below the expected ~4. Meanwhile Voronoi-patch p=3 gets 3.31. This suggests the cubed-sphere structured case has a specific
pathology.

Root cause hypothesis: The cubed-sphere test uses relatively large cells with significant gnomonic distortion. The harmonic basis P_k, Q_k are Euclidean harmonic functions (solutions
of the flat Laplacian), not spherical harmonic functions. On the chart, the natural inner product involves the Hodge metric J^T J / |J|, but the basis functions themselves are
flat-space harmonics. At p=3, the divergence-particular modes ∇(x^{a+1}y^b/(a+1)) have degree-3 monomials that interact with the chart metric more strongly.

Recommendations:
1. Use surface-adapted basis functions. Replace P_k(x,y) = Re((x+iy)^k) with spherical harmonics projected onto the gnomonic chart. At minimum, orthogonalize the basis against the
metric-weighted Gram matrix at each cell (already partially done by QR in the split basis, but the raw monomial basis inherits flat-space structure).
2. Apply the Duffy high-order quadrature from gauss_legendre_rule(10) to the metric-weighted Gram matrix computation. Currently the 7-point symmetric triangle rule is used even in the
metric-weighted path.
3. Consider a chart-centered coordinate rescaling that accounts for the gnomonic Jacobian determinant variation across the cell, reducing the metric variation that the basis must
capture.

C. Edge Moment Transfer Quadrature on the Sphere (Medium Impact)

Problem: In transfer_source_to_target_edge_moments() (line 2769), the Legendre parameter t for the target edge is computed using the whole target edge parametrization projected into
the source chart:
const double t = 2.0 * (global_p - whole_a).dot(whole_delta) / whole_denom - 1.0;
This is a Euclidean arc-length parametrization in chart coordinates. But on the sphere, the natural parametrization is the great-circle arc-length. The mismatch means the Legendre
moments on the target edge are not exactly the spherical-arc Legendre moments -- they are the chart-projected Legendre moments. For p≥2, this introduces a systematic O(h^2) bias.

Recommendation: In the spherical case, parametrize the target edge by great-circle arc fraction (as already done for source edges in basis_edge_moments() at line 894-900). The
source-edge moment code already does this correctly -- the target-edge transfer should be made consistent.

D. Missing Metric Weighting in Higher-Order Gram Matrix (Medium Impact)

Problem: The higher-order PlanarMomentInterpolator::reconstruct_source_polygon() (line 2500) computes the raw Gram matrix G_raw as:
G_raw(i, j) = ∫_K v_i · v_j dA
using integrate_polygon_scalar_duffy with the flat Euclidean inner product. In spherical mode, this should use the Hodge metric weighting to account for the gnomonic chart distortion,
just as the low-order reconstruct_source_polygon() does (line 1837). Without this, the minimum-energy objective in the KKT/constrained-LS solve minimizes the wrong functional on the
sphere.

Recommendation: Add the same gnomonic_hodge_metric weighting to the high-order Gram matrix computation when options_.mode == GeometryMode::SphericalGnomonic.

E. Stabilization of Higher Harmonic Modes (Low-Medium Impact)

Problem: In the low-order reconstruction (line 1862):
if (i >= 4) {
    V(i, i) += 1.0e2 * stabilization_area;
}
This adds a large diagonal penalty to harmonic modes k≥3. The penalty magnitude 1.0e2 is a hand-tuned constant. For Voronoi cells with many edges (N>>4), this aggressively damps the
higher harmonics. While this prevents ill-conditioning, it also limits the accuracy benefit of having more harmonic modes available.

Recommendation:
1. Make the stabilization proportional to the expected magnitude of the harmonic mode, not a flat constant. A natural choice is to scale by V(i,i) itself (relative stabilization)
rather than by stabilization_area.
2. Consider using a truncated SVD or Tikhonov regularization with a problem-dependent parameter rather than a fixed diagonal penalty.
3. Alternatively, limit K_max adaptively: for N-sided polygons, only use harmonics up to the degree where the Gram matrix condition number stays below a threshold.

F. Divergence Computation Uses Chart Area, Not Spherical Area (Low-Medium Impact)

Problem: In reconstruct_source_polygon() (line 1815):
const double divergence = source_flux.sum() / poly.area;
where poly.area is the chart area. Under the Piola mapping, the edge fluxes are chart fluxes, so d = sum(fluxes) / chart_area gives the chart-space divergence. This is mathematically
consistent. However, the reported divergence diagnostic and the target-divergence RHS use this same chart divergence multiplied by chart overlap area, which gives a chart-level
divergence theorem. The physical divergence on the sphere is d_physical = sum(fluxes) / spherical_area.

Current status: This is actually correct as implemented -- the chart-level divergence theorem is self-consistent under the Piola mapping. But the cell-field diagnostics compare
against exact physical fields, creating an accuracy gap at coarse resolutions where chart_area ≠ spherical_area significantly.

Recommendation: No change needed for conservation (which is exact). But for accuracy improvement, consider tracking both chart and physical divergence and using the physical
divergence for error diagnostics.

G. All-Pairs Candidate Search (Performance, Not Accuracy)

Problem: The transfer loops (lines 2067, 2261, 2738) iterate over all source cells for every target edge. This is O(N_source × N_target_edges), which is fine for the current small
test cases but will not scale.

Recommendation: Implement a spatial index (R-tree, k-d tree, or MOAB's built-in spatial search) for candidate filtering. This doesn't affect accuracy but is needed for production use.

H. Redundant LocalPolygon Construction (Performance)

Problem: local_polygon() is called multiple times for the same polygon in different code paths (once in source cache construction, again in target processing, etc.). Each call does
MOAB queries, coordinate extraction, and orientation enforcement.

Recommendation: Cache LocalPolygon objects by entity handle within a transfer call. The SourceCache struct already does this for source cells, but target cells are reconstructed
per-source-cell iteration in some paths.

I. Higher-Order Conforming Projection Decouples Moments (Correctness Concern)

Problem: In project_target_edge_moments_to_hdiv_conforming() (line 2856-2883), the conforming solve decouples the moment degrees:
- Degree 0: full Schur complement solve with divergence constraints
- Degree ≥ 1: simple weighted average unique_moments = hinv * g (no constraints)

This means higher moments are averaged but not constrained by any physical requirement. For a truly conforming H(div) projection, the higher moments should also satisfy consistency
conditions (e.g., the tangential derivative of the flux should be continuous across shared edges).

Recommendation: For p≥2, add inter-cell moment coupling constraints. The current decoupled approach works at p=1 but may lose accuracy at p=2,3. A coupled global solve over all moment
degrees simultaneously would be more rigorous.

J. Basis Conditioning for Voronoi Cells (Robustness)

Problem: The local_length_scale() function (lines 811-818) takes the max of sqrt(area) and vertex distances. For highly elongated Voronoi cells, this may over-scale coordinates in one
direction while under-scaling in the other.

Recommendation: Consider using an anisotropic scaling based on the cell's principal axes (from the inertia tensor) rather than a single isotropic scale. This would improve
conditioning for irregular cells at higher polynomial orders.

K. Triangle Fan Integration Origin (Subtle Accuracy Issue)

Problem: The Gram matrix and moment integrals use a triangle fan centered at the origin (centroid-relative coordinates). For highly non-convex or elongated cells, some triangles in
the fan can be very thin or nearly degenerate, reducing quadrature accuracy.

Recommendation: Use a Delaunay triangulation of the polygon interior instead of a centroid fan for the integration. Alternatively, detect thin triangles and subdivide them.

---
3. Priority-Ordered Improvement Plan

┌──────────┬────────────────────────────────────────────────────────────────────────┬───────────────────────────────────┬────────┐
│ Priority │                                  Area                                  │          Expected Impact          │ Effort │
├──────────┼────────────────────────────────────────────────────────────────────────┼───────────────────────────────────┼────────┤
│ 1        │ Metric-weighted Gram for high-order path (D)                           │ Fix spherical p=2,3 accuracy      │ Low    │
├──────────┼────────────────────────────────────────────────────────────────────────┼───────────────────────────────────┼────────┤
│ 2        │ Higher-order quadrature for metric-weighted integrals (A)              │ Fix spherical p=3 stalling        │ Medium │
├──────────┼────────────────────────────────────────────────────────────────────────┼───────────────────────────────────┼────────┤
│ 3        │ Consistent spherical arc-length parametrization for target moments (C) │ Fix spherical p≥2 bias            │ Low    │
├──────────┼────────────────────────────────────────────────────────────────────────┼───────────────────────────────────┼────────┤
│ 4        │ Coupled multi-degree conforming projection (I)                         │ Improve p=2,3 conforming accuracy │ Medium │
├──────────┼────────────────────────────────────────────────────────────────────────┼───────────────────────────────────┼────────┤
│ 5        │ Adaptive harmonic mode stabilization (E)                               │ Improve Voronoi accuracy          │ Low    │
├──────────┼────────────────────────────────────────────────────────────────────────┼───────────────────────────────────┼────────┤
│ 6        │ Anisotropic basis scaling (J)                                          │ Robustness on irregular cells     │ Medium │
├──────────┼────────────────────────────────────────────────────────────────────────┼───────────────────────────────────┼────────┤
│ 7        │ Surface-adapted basis functions (B)                                    │ Optimal spherical convergence     │ High   │
├──────────┼────────────────────────────────────────────────────────────────────────┼───────────────────────────────────┼────────┤
│ 8        │ Spatial acceleration (G)                                               │ Performance at scale              │ Medium │
└──────────┴────────────────────────────────────────────────────────────────────────┴───────────────────────────────────┴────────┘

The top three items (D, A, C) are the most likely explanations for the observed spherical convergence rate shortfalls and are relatively straightforward to implement. Items D and C
are essentially consistency fixes -- making the high-order spherical path use the same metric-aware techniques already present in the low-order path.

