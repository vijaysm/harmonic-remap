# First-Order Mimetic Voronoi-to-Voronoi Edge-Flux Remap Algorithm

A self-contained pure-Python reference implementation of a lowest-order
(`p=0`, level-2 mimetic) conservative edge-flux projection between two
Voronoi tessellations of the unit square `[0,1]²`, with an optional
H(div)-conforming post-projection. Outputs the remap weights in SCRIP
NetCDF format using an "edges-as-grid" layout.

## What it does

Given two nonmatching Voronoi meshes covering the unit square, transfer
edge-normal flux degrees of freedom from the source mesh to the target
mesh through a sparse remap operator that is

- **exactly conservative** at the global integral level (machine precision),
- **exactly conservative** on every source-target cell overlap (machine precision),
- **exactly conservative** per source cell (by construction),
- **exactly consistent** with source-determined per-target-cell discrete
  divergence (after H(div) projection, machine precision),
- **first-order convergent** in the L² norm of edge fluxes for smooth
  analytical fields (rates 1.08 / 1.32 on the test fields).

The output is a single signed scalar per unique target edge, with the sign
convention derived from the cell's outward normal and the edge's intrinsic
direction (`v_low → v_high`, lexicographic).

## Dependencies

```
numpy
scipy        # Voronoi, cKDTree, sparse linear algebra
netCDF4      # SCRIP file I/O
matplotlib   # convergence plot
```

A working `climate-vis` conda environment is already configured in this
repository and contains all dependencies.

## Quick start

```bash
# Run the embedded test suite + convergence study + write SCRIP files.
conda run -n climate-vis python scripts/voronoi_to_voronoi_p0.py \
    --output-dir /tmp/voronoi_p0

# Custom refinement levels:
conda run -n climate-vis python scripts/voronoi_to_voronoi_p0.py \
    --output-dir /tmp/voronoi_p0 --levels 32,64,128,256

# Adjust the convergence-rate failure threshold (default 0.9):
conda run -n climate-vis python scripts/voronoi_to_voronoi_p0.py \
    --output-dir /tmp/voronoi_p0 --rate-threshold 1.0
```

Exit code 0 on success (all convergence rates ≥ threshold), 1 on failure.

## Outputs

In `--output-dir`:

| File | Content |
|------|---------|
| `voronoi_p0_weights_div_free_N{Ns}_to_N{Nt}.nc` | SCRIP weight file, divergence-free field |
| `voronoi_p0_weights_smooth_div_N{Ns}_to_N{Nt}.nc` | SCRIP weight file, smooth-divergence field |
| `voronoi_p0_convergence.png` | Log-log convergence plot, both fields |

To stdout: mesh sanity, constant-field exactness, per-overlap conservation
(50 sampled overlaps), and a convergence table for each field.

## Algorithm

Two-stage pipeline:

**Stage 1 — Clipped-overlap assembly.** For each source cell, a level-2
mimetic reconstruction yields a piecewise-polynomial velocity field with
constant divergence equal to the cell's discrete divergence. For each
target edge, Sutherland-Hodgman half-plane clipping intersects the edge
with each candidate source cell; the source reconstruction is integrated
along each sub-segment using 4-point Gauss-Legendre and accumulated into a
sparse remap matrix `W`. By construction `W` satisfies the constant-
divergence overlap identity to machine precision.

**Stage 2 — H(div)-conforming projection.** The Stage-1 fluxes are
projected onto the H(div)-conforming subspace by minimising
`‖Φ - W·src‖²` subject to the per-target-cell constraint
`Σ σ_K(e)·Φ_e = Σ_{K_s} d_s · |K_s ∩ K_t|`. The closed-form
Lagrange-multiplier solution is `Φ = W·src - Aᵀλ`, where `λ` solves a
small sparse symmetric system `A Aᵀ λ = A·(W·src) - b`.

For full mathematical detail, derivations, references, and convergence
results see [`docs/voronoi_p0_report.tex`](docs/voronoi_p0_report.pdf).

## Robustness items

Two implementation details are required for stable operation at refined
Voronoi resolutions (N_s ≥ 1024) and were not optional in our testing:

1. **Length-scale rescaling of the harmonic basis.** The discrete Gram
   matrix's diagonal scales as `|K|^k` for the k-th harmonic mode; on
   refined meshes (`h ~ 1e-2`, `K_max = 4`) this drives the condition
   number above `1e10`. The basis is evaluated in coordinates rescaled
   by the cell length scale `s = max(√|K|, max_v |v - centroid|)`.

2. **Perpendicular tie-breaking probe.** Halton-seed Voronoi meshes
   occasionally place target edges *exactly* on shared source-cell
   boundaries; without a perpendicular probe both adjacent source cells
   claim the segment and the contribution is double-counted. The probe
   nudges the segment midpoint by `ε ≈ 1e-7 · |segment|` perpendicular
   to the segment and falls back to the opposite direction only when the
   primary probe leaves the unit square (boundary-aligned target edges).

## Embedded tests

Run from `__main__`, all gating:

| Test | Tolerance | Result |
|------|-----------|--------|
| Mesh sanity (areas sum to 1, CCW, every edge has 1 or 2 cells) | 1e-10 | exact |
| Constant field `u = (1,1)` exact reproduction | 1e-9 | ~1e-15 |
| Per-overlap conservation residual (50 samples) | 1e-12 | ~1e-17 |
| L² edge-flux rate (div-free) | ≥ 0.9 | 1.08 |
| L² edge-flux rate (smooth-div) | ≥ 0.9 | 1.32 |

## SCRIP file layout

The remap weights are written using a non-standard but self-describing
*edges-as-grid* reading of SCRIP \[Jones 1999\]: each unique edge is a
"grid cell" with center = midpoint, two corners = endpoints, area = edge
length. The weight matrix is stored as 1-based `src_address` /
`dst_address` / `remap_matrix` arrays. Global attribute
`map_method = "mimetic_p0_edge_flux"` flags the layout for downstream
SCRIP-aware tools.

## Files

```
scripts/voronoi_to_voronoi_p0.py            main implementation (~900 lines)
docs/voronoi_p0_report.{tex,pdf}            4-page methodology + results
docs/make_voronoi_p0_report_figures.py      regenerable report figures
docs/figures/                               convergence + projection + divergence figures
```

## Known limitations / future work

- **Planar only.** Spherical extension via local gnomonic projection
  is planned. Convergence on MPAS-style spherical Voronoi meshes is the
  natural next step.
- **Lowest order.** Higher-order edge-moment transfer (`p ≥ 1`)
  will be future work; this script targets the simplest `p=0` case 
  for clarity.
- **Dense projection block.** The H(div) projection's `A Aᵀ` matrix is
  factored once per mesh pair via sparse LU; for meshes with `>10⁵`
  target cells consider iterative solves (CG / MINRES) instead.

## References

See `docs/references.bib` and the report for full citations:

- Perot & Chartrand (2021), level-2 mimetic method
- Raviart & Thomas (1977), Brezzi & Fortin (1991), mixed FEM / RT₀
- Bochev & Hyman (2006), Lipnikov-Manzini-Shashkov (2014), mimetic methods
- Brezzi-Falk-Marini (2014), Beirão-da-Veiga (2016), mixed VEM
- Arnold-Falk-Winther (2006), H(div)-conforming FEEC
- Sutherland & Hodgman (1974), polygon clipping
- Jones (1999), SCRIP conservative remap format
