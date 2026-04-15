# Diagnostic Summary: Mass Conservation Failure in Voronoi Remap

## 1. Root Cause: Weakly-Enforced Flux Reconstruction
- The Level-2 Perot-Chartrand vector reconstruction solves a local least-squares system (`v.ldlt().solve(rhs)`) that minimizes the difference between reconstructed and input edge-normal fluxes **in a weighted L² (Galerkin) sense**.
- This is a **weak enforcement**: the reconstructed field satisfies the flux constraints only on average across all edges simultaneously, not pointwise on each edge. Concretely, for an individual edge $e$, there is no guarantee that $\int_e u_h \cdot n ds = U_e$. The residuals on individual edges are nonzero and form a pattern of interior normal-flux discontinuities.

## 2. Why Quads Passed (False Conservation)
- On **uniform Cartesian grids**, the geometric regularity imposes a high degree of symmetry on the least-squares residuals. The flux-jump errors at every interior edge are equal and opposite to those on the parallel edge across the same cell.
- When these residuals are summed over the entire mesh, their contributions cancel to **machine precision** ($O(10^{-15})$). This is an artifact of symmetry, not a property of the algorithm.
- The quad tests therefore pass not because conservation is correctly enforced, but because the error distribution is perfectly antisymmetric and self-cancelling — creating a false signal of exact conservation.

## 3. Why Voronoi Failed (Symmetry Breaking Exposes the Bug)
- On **unstructured Voronoi meshes**, cells have irregular shapes, varying edge lengths, and non-uniform connectivity. The least-squares residuals on interior edges no longer exhibit any cancellation symmetry.
- During the remap step (`transfer_source_to_target_edges`), target edges are integrated against the reconstructed source field. When a target edge overlaps a source interior edge carrying an unbalanced flux jump, it **accumulates that spurious jump** directly into the remapped quantity.
- These unbalanced interior discontinuities add coherently across the mesh rather than cancelling, producing a **macroscopic spurious source/sink** of mass — manifesting as the `1.32e-02` conservation error observed in the failing Voronoi tests.
- The error scales with mesh irregularity, not mesh resolution, which is why it does not converge away under refinement.

## 4. Proposed Fix: Strongly Flux-Matching Reconstruction
To recover **exact discrete conservation on arbitrary polygonal meshes**, the reconstruction must be upgraded to be *strongly flux-matching*.

**Constrained Least-Squares (CLS) with Lagrange Multipliers**
Reformulate the local normal-equation system to include one Lagrange multiplier $\lambda_e$ per edge, strictly enforcing $\int_e u_h \cdot n ds = U_e$ as a hard equality constraint. This replaces the current unconstrained `v.ldlt().solve(rhs)` with a saddle-point KKT solve. By expanding the harmonic basis ($P_n, Q_n$) dynamically to ensure the number of degrees of freedom is $\ge$ the number of edges, the flux-matching conditions are satisfied to machine precision by construction, regardless of cell geometry.
