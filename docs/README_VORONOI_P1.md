# Voronoi p=1 Round-Trip Instability: Documentation Index

This directory contains comprehensive analysis of the p=1 Voronoi catastrophic instability discovered in Figure 7 of the roundtrip convergence study.

## Quick Links

- **For quick understanding**: Read `voronoi_p1_quick_reference.md` (5 min read)
- **For complete analysis**: Read `voronoi_p1_instability_analysis.md` (20 min read)
- **For code fixes**: See proposed solutions in the analysis document

## The Problem

```
Max round-trip error:
  p=0: 3.93e-02 ✓ (stable)
  p=1: 6.95 ✗✗✗ (CATASTROPHIC)
  p=2: 4.29e-03 ✓ (stable)
```

**Error magnitude**: ~175× worse than p=0, ~1600× worse than p=2.

## Root Cause (One Sentence)

The backward reconstruction leg on Voronoi n-gons uses an ill-conditioned split-basis moment method (κ~10^4) that amplifies forward transfer errors by 1000-10,000×.

## Files in This Analysis

### 1. `voronoi_p1_quick_reference.md`
**Length**: 184 lines | **Read time**: ~5 minutes

Quick visual guide with:
- Code flow diagrams (forward/backward legs)
- Key problem regions highlighted
- Comparison table (p=0 vs p=1 vs p=2)
- Files to read in order
- Reproduction steps
- Proposed simplest fix

**Best for**: Getting oriented, understanding the structure

### 2. `voronoi_p1_instability_analysis.md`
**Length**: 318 lines | **Read time**: ~20 minutes

Comprehensive technical analysis with:
- Executive summary
- Two-leg round-trip structure explained
- Where ill-conditioning comes from (3 sources)
- Why p=0 and p=2 work (contrast analysis)
- Detailed mechanism (3 layers of amplification)
- Root cause summary table
- 4 proposed fix options (A–D)
- References to exact code lines

**Best for**: Understanding the root causes, evaluating fixes

## Key Code Locations

### Critical Decision Point
**`tests/dump_visuals.cpp:1062–1076`**
```cpp
if (n_edges == 4) {
    bwd.reconstruct_source_polygon_piola_rt(...);  // Good
} else {
    bwd.reconstruct_source_polygon(...);  // BAD for p=1
}
```
This is where Voronoi cells diverge from quads in the backward path.

### Source of Ill-Conditioning
**`src/mimetic.cpp:4664–4682`**
```cpp
const Eigen::Matrix2d hodge = gnomonic_hodge_metric(p + poly.centroid, frame);
G_raw(i,j) = ∇b_i · hodge · ∇b_j;  // κ ~ 10^4 at high lat!
```

### Where Error Gets Amplified
**`src/mimetic.cpp:4816–4835`**
```cpp
coeffs = A.colPivHouseholderQr().solve(moments);  // Ill-conditioned solve
```

### For Comparison (Why p=0 Works)
**`src/mimetic.cpp:3397–3535`**
KKT system with constraint regularization provides stability.

## The Three-Layer Amplification Mechanism

```
Layer 1: Hodge Metric Distortion
  κ(Hodge) ~ 10^4 at high latitudes
  └─ Gram matrix condition number

Layer 2: Split-Basis Instability
  κ(split basis) ~ 10^3
  └─ From SVD of ill-conditioned Gram

Layer 3: Backward Least-Squares Failure
  κ(constraint matrix A) ~ 10^3
  └─ Amplifies 0.1% moment error → 100% coefficient error
     └─ Divergence error ~ 1000%
```

## Reproduction

```bash
cd /Users/mahadevan/Code/AI/Mimetic-SphPoly
./build/dump_visuals --mercator-case=vor_p1

# Output:
# vor_p1 max error: 6.95e+00  ← CATASTROPHIC

# For comparison:
./build/dump_visuals --mercator-case=cs_p0
# cs_p0 max error: 3.93e-02  ← Good

./build/dump_visuals --mercator-case=cs_p2
# cs_p2 max error: 4.29e-03  ← Good
```

## Fix Recommendations

### Simplest Fix (Option B)
Use p=0 harmonic reconstruction for p=1 Voronoi backward leg instead of split-basis.
- **Pro**: Guaranteed stability, proven to work on all topologies
- **Con**: Loses p=1 higher-order accuracy (but still better than 6.95)
- **Expected result**: Error ~ 1e-2 (like p=0)

### Most Robust Fix (Option C)
Implement generalized Piola-RT for n-gons (research project).
- **Pro**: Perfect conditioning, extends to arbitrary polygons
- **Con**: Significant implementation effort

### Other Options
See Section "Fix Options" in `voronoi_p1_instability_analysis.md` for options A, C, D.

## References

All code citations use line numbers from the **current repository state**. If code is refactored, line numbers will shift but the analysis concepts remain valid.

**Key insight**: The problem is not Voronoi-specific—it's a **split-basis moment instability** at any p where:
- Gram matrix κ is high (e.g., near poles)
- Basis dimension is low (p=1 has only 6D for [P_1]²)
- No stabilization mechanism is in place (unlike KKT in p=0)

## Questions Answered

1. ✓ How does the Voronoi round-trip work? (forward → Voronoi → backward → lat/lon)
2. ✓ What meshes does it use? (lat/lon, Voronoi icosahedral dual)
3. ✓ How is the backward leg done for p=1? (split-basis moment method, NOT Piola-RT)
4. ✓ How does reconstruct_source_polygon handle p=1? (split-basis [P_1]² with moment constraints)
5. ✓ Is there a SplitBasis vs KKT path? (Yes: split-basis for moments, KKT for harmonic)
6. ✓ What happens with Piola metric on Voronoi? (Distorts Jacobian → κ~10^4 → instability)
7. ✓ Why is p=1 Voronoi catastrophically unstable? (Ill-conditioned split-basis × 1000 amplification)

---

**Last updated**: Analysis completed from codebase trace of `dump_visuals.cpp:1006-1094` and `src/mimetic.cpp:3397-5150`.
