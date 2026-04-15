# Spherical Quad Error Norms Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add analytical edge-based and cell-centered error norms to `spherical_quad_test` so spherical transfer accuracy and reconstruction accuracy can be diagnosed separately.

**Architecture:** Keep all new logic in `tests/spherical_quad_test.cpp`. Compute edge L1/L2 norms as quadrature integrals of flux-density error along great-circle target edges, compute edge Linf as the max sampled pointwise error on those quadrature points, and compute cell-centered vector norms from centroid-sampled reconstructed versus exact vectors.

**Tech Stack:** C++14, MOAB, Eigen, existing mimetic reconstruction and spherical test utilities.

---

### Task 1: Add exact-versus-reconstructed edge error evaluators

**Files:**
- Modify: `tests/spherical_quad_test.cpp`

**Step 1:** Add helper(s) to evaluate exact normal flux density along a target great-circle edge.

**Step 2:** Add helper(s) to evaluate reconstructed normal flux density along the same edge by mapping quadrature points into the local target polygon frame and calling `interpolator.velocity(...)`.

**Step 3:** Accumulate per-edge L1, L2, and Linf errors using the existing 7-point Gauss rule.

### Task 2: Add global cell-centered vector norms

**Files:**
- Modify: `tests/spherical_quad_test.cpp`

**Step 1:** For each target cell, compute centroid-sampled exact and reconstructed vectors.

**Step 2:** Accumulate global cell L1 and Linf norms from the vector error magnitude.

**Step 3:** Store a scalar `TARGET_FIELD_ERROR_NORM` tag for VTK inspection.

### Task 3: Report diagnostics and verify

**Files:**
- Modify: `tests/spherical_quad_test.cpp`

**Step 1:** Print the new edge and cell norms to stdout.

**Step 2:** Build and run `spherical_quad_test` manually with a non-default `(source_n, target_n)` pair.

**Step 3:** Run the full CTest suite to ensure no regressions.
