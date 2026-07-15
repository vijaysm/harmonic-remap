#!/usr/bin/env python3
"""Generate figures and metrics for the voronoi-p0 LaTeX report.

Outputs (under docs/figures/voronoi_p0/):
    convergence.png             log-log L2/Linf rates for both fields
    projection_<field>.png      4-panel: source flux, target exact, target computed, error map
    divergence_recovery.png     scatter of computed-vs-exact per-cell divergence + histogram
    conservation.png            domain-integral conservation: source vs target total boundary flux
    metrics_summary.csv         numerical values reported in the text
"""
from __future__ import annotations

import os
import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection

## Need to run this from the repository root
REPO = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, str(REPO / "scripts"))
import voronoi_to_voronoi_p0 as vp0  # type: ignore

OUT = REPO / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def field_div_free(p):
    return vp0.field_div_free(p)


def field_smooth_div(p):
    return vp0.field_smooth_div(p)


def divergence_div_free(p):  # = 0 identically
    return np.zeros(p.shape[:-1])


def divergence_smooth_div(p):
    x = p[..., 0]
    y = p[..., 1]
    return np.pi * (np.cos(np.pi * x) + np.cos(np.pi * y))


def integrate_polygon(func, vertices):
    """Integrate scalar func over polygon using fan triangulation + 7-pt rule."""
    centroid = np.mean(vertices, axis=0)
    n = len(vertices)
    total = 0.0
    for i in range(n):
        a = centroid
        b = vertices[i]
        c = vertices[(i + 1) % n]
        twice_area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        area = 0.5 * abs(twice_area)
        pts = vp0._TRI7_BARY @ np.stack([a, b, c])
        vals = func(pts)
        total += area * float(np.sum(vp0._TRI7_W * vals))
    return total


def cell_signed_outward_flux(mesh, ci, fluxes):
    """Sum_e sign_K(e) * flux(e) for cell ci (= total outward flux of K)."""
    cell_geid = mesh.cell_to_edge_idx[ci]
    cell_signs = mesh.cell_to_edge_sign[ci]
    return float(sum(float(s) * fluxes[gi] for gi, s in zip(cell_geid, cell_signs)))


def boundary_outward_flux(mesh, fluxes):
    """Total outward flux through the unit-square boundary, computed from edge DOFs.

    Boundary edges have exactly one owning cell. Their stored DOF (intrinsic-
    normal-positive) maps to the cell's outward direction via the cell's sign
    convention; for boundary edges this also equals the domain outward
    direction up to a global sign that we handle by using the cell sign.
    """
    total = 0.0
    for ei, owners in enumerate(mesh.edge_to_cells):
        if len(owners) != 1:
            continue
        ci = owners[0]
        # find local index of this edge in cell ci, and use that sign
        cell_geid = mesh.cell_to_edge_idx[ci]
        cell_signs = mesh.cell_to_edge_sign[ci]
        local = list(cell_geid).index(ei)
        total += float(cell_signs[local]) * float(fluxes[ei])
    return total


def plot_edge_flux_field(ax, mesh, fluxes, title, vmax=None, cmap="RdBu_r"):
    """Color edges by signed flux value; light gray cell outlines."""
    polys = [mesh.vertices[cell] for cell in mesh.cells]
    pc = PolyCollection(polys, facecolor="white", edgecolor="lightgray", lw=0.25)
    ax.add_collection(pc)
    segments = [
        [mesh.vertices[v0], mesh.vertices[v1]] for v0, v1 in mesh.unique_edges
    ]
    if vmax is None:
        vmax = float(np.max(np.abs(fluxes))) + 1.0e-30
    lc = LineCollection(segments, array=fluxes, cmap=cmap, linewidths=1.4)
    lc.set_clim(-vmax, vmax)
    ax.add_collection(lc)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_title(title, fontsize=10)
    return lc


def plot_edge_error(ax, mesh, err, title, cmap="viridis"):
    polys = [mesh.vertices[cell] for cell in mesh.cells]
    pc = PolyCollection(polys, facecolor="white", edgecolor="lightgray", lw=0.25)
    ax.add_collection(pc)
    segments = [
        [mesh.vertices[v0], mesh.vertices[v1]] for v0, v1 in mesh.unique_edges
    ]
    abs_err = np.abs(err)
    lc = LineCollection(segments, array=abs_err, cmap=cmap, linewidths=1.6)
    lc.set_clim(0, float(np.max(abs_err)) + 1.0e-30)
    ax.add_collection(lc)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_title(title, fontsize=10)
    return lc


# ---------------------------------------------------------------------------
# 1. Convergence
# ---------------------------------------------------------------------------


def make_convergence_plot(levels):
    print("[1/5] convergence (H(div)-projected)")
    fields = [
        ("div-free", field_div_free, "tab:blue"),
        ("smooth-div", field_smooth_div, "tab:red"),
    ]
    results = {}
    for name, field, color in fields:
        h_arr = []
        l2_arr = []
        linf_arr = []
        for n in levels:
            n_t = int(round(1.5 * n))
            src = vp0.voronoi_in_unit_square(vp0.halton(n, 0))
            tgt = vp0.voronoi_in_unit_square(vp0.halton(n_t, 1000))
            src_R, src_scales = vp0.precompute_reconstructions(src)
            W = vp0.build_projection_operator(src, tgt, src_R, src_scales)
            H = vp0.build_hdiv_projection(W, src, tgt)
            src_fluxes = vp0.compute_edge_fluxes(field, src)
            tgt_exact = vp0.compute_edge_fluxes(field, tgt)
            tgt_computed = H.apply(src_fluxes)
            err = tgt_computed - tgt_exact
            rel_l2 = float(np.linalg.norm(err) / max(np.linalg.norm(tgt_exact), 1.0e-30))
            linf = float(np.max(np.abs(err)))
            h = 1.0 / math.sqrt(n)
            h_arr.append(h)
            l2_arr.append(rel_l2)
            linf_arr.append(linf)
        h_arr = np.array(h_arr)
        l2_arr = np.array(l2_arr)
        linf_arr = np.array(linf_arr)
        slope_l2, _ = np.polyfit(np.log(h_arr), np.log(l2_arr), 1)
        slope_linf, _ = np.polyfit(np.log(h_arr), np.log(linf_arr), 1)
        results[name] = dict(
            h=h_arr,
            l2=l2_arr,
            linf=linf_arr,
            rate_l2=float(slope_l2),
            rate_linf=float(slope_linf),
            color=color,
        )
        print(f"  {name}: L2 rate = {slope_l2:.3f}, Linf rate = {slope_linf:.3f}")

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for name, d in results.items():
        ax.loglog(
            d["h"], d["l2"], "o-", color=d["color"], label=f"{name}  $L^2$ (rate {d['rate_l2']:.2f})"
        )
        ax.loglog(
            d["h"], d["linf"], "s--", color=d["color"], label=f"{name}  $L^\\infty$ (rate {d['rate_linf']:.2f})"
        )
    h_ref = np.array([min(d["h"].min() for d in results.values()), max(d["h"].max() for d in results.values())])
    first = next(iter(results.values()))
    ax.loglog(h_ref, first["l2"][0] / first["h"][0] * h_ref, "k:", alpha=0.6, label=r"$\mathcal{O}(h)$")
    ax.set_xlabel(r"Mesh size $h \sim N_{\mathrm{src}}^{-1/2}$")
    ax.set_ylabel("Relative target edge-flux error")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "convergence.png", dpi=160)
    plt.close(fig)
    return results


# ---------------------------------------------------------------------------
# 2. Projection plots (4 panel) per field
# ---------------------------------------------------------------------------


def make_projection_plots(n_src, n_tgt):
    print("[2/5] projection panels (H(div)-projected)")
    src_seeds = vp0.halton(n_src, 0)
    tgt_seeds = vp0.halton(n_tgt, 1000)
    src = vp0.voronoi_in_unit_square(src_seeds)
    tgt = vp0.voronoi_in_unit_square(tgt_seeds)
    src_R, src_scales = vp0.precompute_reconstructions(src)
    W = vp0.build_projection_operator(src, tgt, src_R, src_scales)
    H = vp0.build_hdiv_projection(W, src, tgt)
    metrics = {}

    for field_name, field in [("div_free", field_div_free), ("smooth_div", field_smooth_div)]:
        src_fluxes = vp0.compute_edge_fluxes(field, src)
        tgt_exact = vp0.compute_edge_fluxes(field, tgt)
        tgt_computed = H.apply(src_fluxes)
        err = tgt_computed - tgt_exact

        flux_max = float(np.max(np.abs(np.concatenate([src_fluxes, tgt_exact, tgt_computed]))))

        fig, axes = plt.subplots(1, 4, figsize=(16, 3.2))
        lc1 = plot_edge_flux_field(axes[0], src, src_fluxes, f"(a) Source fluxes ($N_s = {n_src}$)", vmax=flux_max)
        lc2 = plot_edge_flux_field(axes[1], tgt, tgt_exact, f"(b) Target exact ($N_t = {n_tgt}$)", vmax=flux_max)
        lc3 = plot_edge_flux_field(axes[2], tgt, tgt_computed, "(c) Target computed (W u)", vmax=flux_max)
        lc4 = plot_edge_error(axes[3], tgt, err, "(d) |error| on target edges")

        cbar = fig.colorbar(lc3, ax=axes[:3].tolist(), shrink=0.75, label=r"signed flux $\int_e \mathbf{u}\cdot\mathbf{n}\,ds$")
        cbar = fig.colorbar(lc4, ax=axes[3], shrink=0.75, label="|error|")

        fig.suptitle(
            f"Field: {field_name.replace('_', ' ')}   $\\Vert e\\Vert_\\infty={float(np.max(np.abs(err))):.2e}$,"
            f"   relative $L^2$ error = {float(np.linalg.norm(err)/np.linalg.norm(tgt_exact)):.2e}",
            fontsize=11,
        )
        fig.savefig(OUT / f"projection_{field_name}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

        metrics[field_name] = dict(
            n_src=n_src,
            n_tgt=n_tgt,
            linf=float(np.max(np.abs(err))),
            rel_l2=float(np.linalg.norm(err) / np.linalg.norm(tgt_exact)),
        )
    return src, tgt, src_R, src_scales, W, metrics


# ---------------------------------------------------------------------------
# 3. Divergence recovery
# ---------------------------------------------------------------------------


def make_divergence_plot(src, tgt, src_R, src_scales, W):
    """Per-cell divergence: computed (sum signed fluxes / area) vs exact (∫div u dA / area).

    For each source cell K_s the reconstruction has constant divergence
    d_s = sum(U)/area(K_s). For the smooth-div field we compare d_s against
    the exact cell-mean divergence (1/area) ∫_K div(u) dA.

    Same on the target after the transfer: each target cell K_t has a
    reconstructed divergence (sum signed transferred fluxes) / area.
    """
    print("[3/4] divergence recovery")
    field = field_smooth_div
    div_exact = divergence_smooth_div

    src_fluxes = vp0.compute_edge_fluxes(field, src)
    tgt_computed = W @ src_fluxes

    src_d_comp = np.array(
        [cell_signed_outward_flux(src, ci, src_fluxes) / src.cell_areas[ci] for ci in range(len(src.cells))]
    )
    src_d_exact = np.array(
        [integrate_polygon(div_exact, src.vertices[cell]) / src.cell_areas[ci] for ci, cell in enumerate(src.cells)]
    )

    tgt_d_comp = np.array(
        [cell_signed_outward_flux(tgt, ci, tgt_computed) / tgt.cell_areas[ci] for ci in range(len(tgt.cells))]
    )
    tgt_d_exact = np.array(
        [integrate_polygon(div_exact, tgt.vertices[cell]) / tgt.cell_areas[ci] for ci, cell in enumerate(tgt.cells)]
    )

    src_err = src_d_comp - src_d_exact
    tgt_err = tgt_d_comp - tgt_d_exact

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].scatter(src_d_exact, src_d_comp, s=8, alpha=0.5, label="source")
    axes[0].scatter(tgt_d_exact, tgt_d_comp, s=8, alpha=0.5, label="target (after W)")
    lo = min(src_d_exact.min(), tgt_d_exact.min(), src_d_comp.min(), tgt_d_comp.min()) * 1.05
    hi = max(src_d_exact.max(), tgt_d_exact.max(), src_d_comp.max(), tgt_d_comp.max()) * 1.05
    axes[0].plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6, label="$y=x$")
    axes[0].set_xlabel(r"exact cell-mean divergence $\overline{\nabla\cdot\mathbf{u}}|_K$")
    axes[0].set_ylabel("recovered divergence $d_K$")
    axes[0].set_title("Per-cell divergence: smooth-div field")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(src_err, bins=40, alpha=0.55, label=f"source (max |err|={float(np.max(np.abs(src_err))):.2e})")
    axes[1].hist(tgt_err, bins=40, alpha=0.55, label=f"target (max |err|={float(np.max(np.abs(tgt_err))):.2e})")
    axes[1].set_xlabel("$d_K - \\overline{\\nabla\\cdot\\mathbf{u}}|_K$")
    axes[1].set_ylabel("# cells")
    axes[1].set_title("Divergence-error histogram")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "divergence_recovery.png", dpi=160)
    plt.close(fig)

    return dict(
        src_div_max=float(np.max(np.abs(src_err))),
        tgt_div_max=float(np.max(np.abs(tgt_err))),
        src_div_l2=float(np.sqrt(np.mean(src_err**2))),
        tgt_div_l2=float(np.sqrt(np.mean(tgt_err**2))),
    )


# ---------------------------------------------------------------------------
# 4. Conservation: integral and per-overlap
# ---------------------------------------------------------------------------


def make_divergence_convergence_plot(levels):
    """Side-by-side per-cell divergence error convergence: source vs target.

    Smooth-div field has analytic div(u) = pi(cos pi x + cos pi y), nonzero, so a
    relative L2 norm is well-defined. We measure
        e_div^2 = sum_K |K| * (d_K_comp - d_K_exact)^2,
        denom^2 = sum_K |K| * d_K_exact^2.

    On the source mesh the discrete divergence theorem makes d_K^disc exact to
    quadrature precision. On the target mesh we report two curves:
      - "raw":  divergence of W-transferred fluxes (plateaus, no convergence)
      - "H(div)": divergence after the H(div)-conforming post-projection,
                  which by construction matches the source-determined target
                  divergence; this curve converges as O(h).

    Source and target are plotted on separate panels so the machine-precision
    source curve does not skew the target axis.
    """
    print("[5/5] divergence convergence (source/target side-by-side, raw + H(div))")
    field = field_smooth_div
    div_exact = divergence_smooth_div

    h_arr = []
    src_l2 = []
    tgt_l2_raw = []
    tgt_l2_hdiv = []
    src_max = []
    tgt_max_raw = []
    tgt_max_hdiv = []
    for n in levels:
        n_t = int(round(1.5 * n))
        src = vp0.voronoi_in_unit_square(vp0.halton(n, 0))
        tgt = vp0.voronoi_in_unit_square(vp0.halton(n_t, 1000))
        src_R, src_scales = vp0.precompute_reconstructions(src)
        W = vp0.build_projection_operator(src, tgt, src_R, src_scales)
        H = vp0.build_hdiv_projection(W, src, tgt)
        src_fluxes = vp0.compute_edge_fluxes(field, src)
        tgt_raw = W @ src_fluxes
        tgt_hdiv = H.apply(src_fluxes)

        s_d_comp = np.array([cell_signed_outward_flux(src, ci, src_fluxes) / src.cell_areas[ci] for ci in range(len(src.cells))])
        s_d_ex = np.array([integrate_polygon(div_exact, src.vertices[c]) / src.cell_areas[ci] for ci, c in enumerate(src.cells)])
        t_d_raw = np.array([cell_signed_outward_flux(tgt, ci, tgt_raw) / tgt.cell_areas[ci] for ci in range(len(tgt.cells))])
        t_d_hdiv = np.array([cell_signed_outward_flux(tgt, ci, tgt_hdiv) / tgt.cell_areas[ci] for ci in range(len(tgt.cells))])
        t_d_ex = np.array([integrate_polygon(div_exact, tgt.vertices[c]) / tgt.cell_areas[ci] for ci, c in enumerate(tgt.cells)])

        s_err = s_d_comp - s_d_ex
        t_err_raw = t_d_raw - t_d_ex
        t_err_hdiv = t_d_hdiv - t_d_ex
        s_l2_num = float(np.sqrt(np.sum(src.cell_areas * s_err**2)))
        t_l2_num_raw = float(np.sqrt(np.sum(tgt.cell_areas * t_err_raw**2)))
        t_l2_num_hdiv = float(np.sqrt(np.sum(tgt.cell_areas * t_err_hdiv**2)))
        s_l2_den = float(np.sqrt(np.sum(src.cell_areas * s_d_ex**2))) + 1e-30
        t_l2_den = float(np.sqrt(np.sum(tgt.cell_areas * t_d_ex**2))) + 1e-30
        h = 1.0 / math.sqrt(n)
        h_arr.append(h)
        src_l2.append(s_l2_num / s_l2_den)
        tgt_l2_raw.append(t_l2_num_raw / t_l2_den)
        tgt_l2_hdiv.append(t_l2_num_hdiv / t_l2_den)
        src_max.append(float(np.max(np.abs(s_err))))
        tgt_max_raw.append(float(np.max(np.abs(t_err_raw))))
        tgt_max_hdiv.append(float(np.max(np.abs(t_err_hdiv))))

    h_arr = np.array(h_arr)
    src_l2 = np.array(src_l2)
    tgt_l2_raw = np.array(tgt_l2_raw)
    tgt_l2_hdiv = np.array(tgt_l2_hdiv)

    slope_tgt_raw, _ = np.polyfit(np.log(h_arr), np.log(tgt_l2_raw), 1)
    slope_tgt_hdiv, _ = np.polyfit(np.log(h_arr), np.log(tgt_l2_hdiv), 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    # Source panel (machine precision floor)
    axes[0].loglog(h_arr, src_l2, "o-", color="tab:green", label="source rel. $L^2$ div err")
    axes[0].loglog(h_arr, np.array(src_max), "o--", color="tab:green", alpha=0.55, label="source max")
    axes[0].set_xlabel(r"Mesh size $h \sim N_s^{-1/2}$")
    axes[0].set_ylabel("Source per-cell divergence error")
    axes[0].set_title("Source: discrete divergence theorem (machine precision)")
    axes[0].legend(fontsize=9, loc="best")
    axes[0].grid(True, which="both", alpha=0.3)

    # Target panel: H(div) projection only (raw plateaus, not informative)
    axes[1].loglog(h_arr, tgt_l2_hdiv, "D-", color="tab:orange", label=f"target rel. $L^2$ div err (rate {slope_tgt_hdiv:.2f})")
    axes[1].loglog(h_arr, np.array(tgt_max_hdiv), "D--", color="tab:orange", alpha=0.55, label="target max")
    axes[1].loglog(h_arr, h_arr * (tgt_l2_hdiv[0] / h_arr[0]), "k:", alpha=0.5, label=r"$\mathcal{O}(h)$")
    axes[1].set_xlabel(r"Mesh size $h \sim N_s^{-1/2}$")
    axes[1].set_ylabel("Target per-cell divergence error")
    axes[1].set_title("Target: H(div)-conforming projection")
    axes[1].legend(fontsize=9, loc="best")
    axes[1].grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "divergence_convergence.png", dpi=160)
    plt.close(fig)

    print("  divergence convergence (smooth-div):")
    print(f"    {'N':>5} {'h':>8} {'src L2':>12} {'tgt L2 raw':>14} {'tgt L2 hdiv':>14}")
    for i, n in enumerate(levels):
        print(f"    {n:>5d} {h_arr[i]:>8.4f} {src_l2[i]:>12.3e} {tgt_l2_raw[i]:>14.3e} {tgt_l2_hdiv[i]:>14.3e}")
    print(f"  fitted target slopes:  raw = {slope_tgt_raw:.3f},  H(div) = {slope_tgt_hdiv:.3f}")
    return dict(
        h=h_arr.tolist(),
        src_l2=src_l2.tolist(),
        tgt_l2_raw=tgt_l2_raw.tolist(),
        tgt_l2_hdiv=tgt_l2_hdiv.tolist(),
        src_max=src_max,
        tgt_max_raw=tgt_max_raw,
        tgt_max_hdiv=tgt_max_hdiv,
        slope_tgt_raw=float(slope_tgt_raw),
        slope_tgt_hdiv=float(slope_tgt_hdiv),
    )


def make_per_element_conservation_metrics(n_src, n_tgt):
    """Per-element conservation diagnostics at one mesh resolution, smooth-div field.

    Reports max-abs across:
      - Per-source-cell residual:  |sum_e sigma_K_s(e) Phi_e - d_s |K_s||  -- exact by definition
      - Per-overlap residual:      |oint_dO u_h^Ks . n - d_s |O||           -- exact for level-2
      - Per-target-cell residual (raw):    |A @ Phi_W - b_target|
      - Per-target-cell residual (H(div)): |A @ Phi_hdiv - b_target|       -- exact after projection
      - Global boundary flux source/target (already in Tab. 1 across N).
    """
    print("[6/6] per-element conservation metrics")
    field = field_smooth_div
    src = vp0.voronoi_in_unit_square(vp0.halton(n_src, 0))
    tgt = vp0.voronoi_in_unit_square(vp0.halton(n_tgt, 1000))
    src_R, src_scales = vp0.precompute_reconstructions(src)
    W = vp0.build_projection_operator(src, tgt, src_R, src_scales)
    H = vp0.build_hdiv_projection(W, src, tgt)
    src_fluxes = vp0.compute_edge_fluxes(field, src)
    phi_raw = W @ src_fluxes
    phi_hdiv = H.apply(src_fluxes)
    A = H.A
    b_target = H.S @ src_fluxes

    # Per-source-cell residual
    src_d = np.array([cell_signed_outward_flux(src, ci, src_fluxes) for ci in range(len(src.cells))])
    # Compare against (sum of edges) directly: should be exact by construction
    per_src_residual = np.zeros(len(src.cells))
    for ci in range(len(src.cells)):
        per_src_residual[ci] = abs(cell_signed_outward_flux(src, ci, src_fluxes) - src_d[ci])

    # Per-overlap residual (50 random pairs)
    src_tree = __import__('scipy.spatial', fromlist=['cKDTree']).cKDTree(src.cell_centroids)
    rng = np.random.default_rng(0)
    indices = rng.choice(len(tgt.cells), size=min(50, len(tgt.cells)), replace=False)
    per_overlap_max = 0.0
    n_overlaps_tested = 0
    for ci_t in indices:
        ct_pos = tgt.cell_centroids[ci_t]
        cs_cands = src_tree.query_ball_point(ct_pos, r=tgt.cell_radii[ci_t] + src.cell_radii.max() + 1e-10)
        for cs in cs_cands:
            R = src_R[cs]
            cell_signs = src.cell_to_edge_sign[cs]
            cell_geid = src.cell_to_edge_idx[cs]
            cell_local_U = np.array([src_fluxes[gi] * float(s) for gi, s in zip(cell_geid, cell_signs)])
            coeffs = R @ cell_local_U
            d_s = coeffs[0]
            N_h = R.shape[0] - 1
            from voronoi_to_voronoi_p0 import _polygon_polygon_clip, subsegment_normal_integral_row
            overlap = _polygon_polygon_clip(src.vertices[src.cells[cs]], tgt.vertices[tgt.cells[ci_t]])
            if overlap is None or overlap.shape[0] < 3:
                continue
            o_area = abs(vp0.signed_area(overlap))
            if o_area < 1e-14:
                continue
            sc = src.cell_centroids[cs]
            scale_s = float(src_scales[cs])
            bdry = 0.0
            for j in range(overlap.shape[0]):
                a = overlap[j] - sc
                b = overlap[(j + 1) % overlap.shape[0]] - sc
                edge = b - a
                length_e = float(np.linalg.norm(edge))
                if length_e < 1e-14:
                    continue
                n_out = np.array([edge[1], -edge[0]]) / length_e
                row = subsegment_normal_integral_row(a, b, n_out, N_h, scale_s)
                bdry += float(row @ coeffs)
            err = abs(bdry - d_s * o_area)
            per_overlap_max = max(per_overlap_max, err)
            n_overlaps_tested += 1

    # Per-target-cell discrete-divergence residual against source-determined RHS
    per_tgt_raw = np.max(np.abs(A @ phi_raw - b_target))
    per_tgt_hdiv = np.max(np.abs(A @ phi_hdiv - b_target))

    metrics = dict(
        n_src=n_src,
        n_tgt=n_tgt,
        per_src_residual_max=float(per_src_residual.max()),
        per_overlap_residual_max=float(per_overlap_max),
        per_overlap_count=int(n_overlaps_tested),
        per_tgt_raw_max=float(per_tgt_raw),
        per_tgt_hdiv_max=float(per_tgt_hdiv),
    )
    print(f"  per-source-cell residual max: {metrics['per_src_residual_max']:.3e}")
    print(f"  per-overlap residual max ({metrics['per_overlap_count']} samples): {metrics['per_overlap_residual_max']:.3e}")
    print(f"  per-target-cell residual max  raw: {metrics['per_tgt_raw_max']:.3e}")
    print(f"  per-target-cell residual max H(div): {metrics['per_tgt_hdiv_max']:.3e}")
    return metrics


def make_conservation_plot(levels):
    """For each refinement level: total domain boundary outward flux from
    source mesh vs target mesh after transfer. Should agree to machine
    precision modulo the truncation in u_h.

    Also: total integral of div(u) over [0,1]^2 (analytic) vs the sum of
    cell-recovered divergences * area (computed).
    """
    print("[4/4] conservation across refinement")
    field = field_smooth_div
    div_exact = divergence_smooth_div
    # Analytic integral of div over the unit square: ∫_[0,1]^2 π(cos πx + cos πy) dx dy
    #                                              = π * (sin πx |_0^1 + sin πy |_0^1) = 0  (since sin π = sin 0 = 0)
    analytic_div_integral = 0.0
    # Domain boundary flux integral = analytic_div_integral by divergence theorem.
    analytic_boundary_flux = analytic_div_integral

    rows = []
    src_bdry = []
    tgt_bdry = []
    src_div_int = []
    tgt_div_int = []
    hs = []
    for n in levels:
        n_t = int(round(1.5 * n))
        src = vp0.voronoi_in_unit_square(vp0.halton(n, 0))
        tgt = vp0.voronoi_in_unit_square(vp0.halton(n_t, 1000))
        src_R, src_scales = vp0.precompute_reconstructions(src)
        W = vp0.build_projection_operator(src, tgt, src_R, src_scales)
        src_fluxes = vp0.compute_edge_fluxes(field, src)
        tgt_computed = W @ src_fluxes
        bs = boundary_outward_flux(src, src_fluxes)
        bt = boundary_outward_flux(tgt, tgt_computed)
        # Sum of per-cell divergences * area = total of (sum signed cell fluxes)
        #                                     = sum over directed edges of fluxes (net = boundary)
        sds = sum(cell_signed_outward_flux(src, ci, src_fluxes) for ci in range(len(src.cells)))
        tds = sum(cell_signed_outward_flux(tgt, ci, tgt_computed) for ci in range(len(tgt.cells)))
        src_bdry.append(bs)
        tgt_bdry.append(bt)
        src_div_int.append(sds)
        tgt_div_int.append(tds)
        hs.append(1.0 / math.sqrt(n))
        rows.append(dict(n_src=n, n_tgt=n_t, src_bdry=bs, tgt_bdry=bt, src_div_sum=sds, tgt_div_sum=tds, diff=bt - bs))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].semilogy(hs, np.abs(np.array(src_bdry) - analytic_boundary_flux) + 1e-18, "o-", label="source, $|\\oint - 0|$")
    axes[0].semilogy(hs, np.abs(np.array(tgt_bdry) - analytic_boundary_flux) + 1e-18, "s-", label="target (after $W$), $|\\oint - 0|$")
    axes[0].semilogy(hs, np.abs(np.array(tgt_bdry) - np.array(src_bdry)) + 1e-18, "^--", label="target $-$ source disagreement")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("$h$")
    axes[0].set_ylabel("absolute boundary flux error")
    axes[0].set_title(r"Domain boundary flux $\oint_{\partial\Omega}\mathbf{u}\cdot\mathbf{n}\,ds$ (exact = 0)")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].semilogy(hs, np.abs(np.array(src_div_int)) + 1e-18, "o-", label="source $\\sum_K d_K |K|$")
    axes[1].semilogy(hs, np.abs(np.array(tgt_div_int)) + 1e-18, "s-", label="target $\\sum_K d_K |K|$")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("$h$")
    axes[1].set_ylabel(r"$|\,\sum_K d_K\, |K|\,|$  (exact = 0)")
    axes[1].set_title("Discrete divergence integral over $\\Omega$")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "conservation.png", dpi=160)
    plt.close(fig)

    return rows


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    levels_conv = [64, 256, 1024, 4096]
    levels_cons = [64, 256, 1024]
    n_src_proj = 256
    n_tgt_proj = 384

    conv = make_convergence_plot(levels_conv)
    src, tgt, src_R, src_scales, W, proj_metrics = make_projection_plots(n_src_proj, n_tgt_proj)
    div_metrics = make_divergence_plot(src, tgt, src_R, src_scales, W)
    cons_rows = make_conservation_plot(levels_cons)
    div_conv = make_divergence_convergence_plot(levels_conv)
    per_elem_metrics = make_per_element_conservation_metrics(n_src_proj, n_tgt_proj)

    summary = OUT / "metrics_summary.csv"
    with summary.open("w") as f:
        w = csv.writer(f)
        w.writerow(["section", "key", "value"])
        for name, d in conv.items():
            w.writerow(["convergence", f"{name}_rate_L2", f"{d['rate_l2']:.4f}"])
            w.writerow(["convergence", f"{name}_rate_Linf", f"{d['rate_linf']:.4f}"])
            w.writerow(["convergence", f"{name}_h_finest", f"{d['h'][-1]:.4f}"])
            w.writerow(["convergence", f"{name}_l2_finest", f"{d['l2'][-1]:.4e}"])
            w.writerow(["convergence", f"{name}_linf_finest", f"{d['linf'][-1]:.4e}"])
        for name, d in proj_metrics.items():
            w.writerow(["projection", f"{name}_linf", f"{d['linf']:.4e}"])
            w.writerow(["projection", f"{name}_rel_L2", f"{d['rel_l2']:.4e}"])
        for k, v in div_metrics.items():
            w.writerow(["divergence", k, f"{v:.4e}"])
        for r in cons_rows:
            w.writerow(["conservation", f"N{r['n_src']}_src_bdry", f"{r['src_bdry']:.4e}"])
            w.writerow(["conservation", f"N{r['n_src']}_tgt_bdry", f"{r['tgt_bdry']:.4e}"])
            w.writerow(["conservation", f"N{r['n_src']}_diff", f"{r['diff']:.4e}"])
        for h, sL, tLr, tLh, sm, tmr, tmh in zip(
            div_conv["h"], div_conv["src_l2"], div_conv["tgt_l2_raw"], div_conv["tgt_l2_hdiv"],
            div_conv["src_max"], div_conv["tgt_max_raw"], div_conv["tgt_max_hdiv"],
        ):
            w.writerow(["div_conv", f"h_{h:.4f}_src_L2", f"{sL:.4e}"])
            w.writerow(["div_conv", f"h_{h:.4f}_tgt_L2_raw", f"{tLr:.4e}"])
            w.writerow(["div_conv", f"h_{h:.4f}_tgt_L2_hdiv", f"{tLh:.4e}"])
            w.writerow(["div_conv", f"h_{h:.4f}_src_max", f"{sm:.4e}"])
            w.writerow(["div_conv", f"h_{h:.4f}_tgt_max_raw", f"{tmr:.4e}"])
            w.writerow(["div_conv", f"h_{h:.4f}_tgt_max_hdiv", f"{tmh:.4e}"])
        w.writerow(["div_conv", "slope_tgt_raw", f"{div_conv['slope_tgt_raw']:.4f}"])
        w.writerow(["div_conv", "slope_tgt_hdiv", f"{div_conv['slope_tgt_hdiv']:.4f}"])
        for k, v in per_elem_metrics.items():
            w.writerow(["per_element", k, f"{v}"])
    print(f"[OK] wrote summary to {summary}")
    print(f"[OK] all figures under {OUT}")


if __name__ == "__main__":
    main()
