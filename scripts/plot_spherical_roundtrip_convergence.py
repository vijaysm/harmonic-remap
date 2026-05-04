"""
Plot the spherical round-trip convergence study.

Two side-by-side panels share the x-axis (cell-size h in degrees):

  Left panel  -- Round-trip RLL -> CS -> RLL convergence.
                 p=0 baseline (Level-2 mimetic) plus the practical
                 patch-recovery + VEM round-trip at p=1 and p=2.

  Right panel -- Forward-only RLL -> CS convergence.  Stops after the
                 forward leg and measures cell-average divergence
                 error on the CS grid.  Practical patch-recovery
                 path: only mu_0 seeded, higher moments synthesized
                 via face-neighbor least-squares fit.  Isolates
                 source-side reconstruction quality from backward-leg
                 amplification.

Usage:
    python scripts/plot_spherical_roundtrip_convergence.py \\
        docs/spherical_roundtrip_convergence.csv \\
        docs/figures/spherical_roundtrip_convergence.png
"""

import sys
import csv
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(path):
    data = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            n = int(row["n_cs"])
            for k, v in row.items():
                if k == "n_cs":
                    continue
                if k not in data:
                    data[k] = {}
                try:
                    data[k][n] = float(v)
                except ValueError:
                    data[k][n] = np.nan
    ns = sorted(set(n for k in data for n in data[k]))
    result = {"n_cs": np.array(ns)}
    for k, d in data.items():
        result[k] = np.array([d.get(n, np.nan) for n in ns])
    return result


def plot_curve(ax, h, y, color, ls, marker, label):
    mask = np.isfinite(y) & (y > 0)
    if mask.sum() >= 1:
        ax.semilogy(h[mask], y[mask], color=color, ls=ls, marker=marker,
                    ms=6, label=label)


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else \
        "docs/spherical_roundtrip_convergence.csv"
    out_path = sys.argv[2] if len(sys.argv) > 2 else \
        "docs/figures/spherical_roundtrip_convergence.png"

    d = read_csv(csv_path)
    ns = d["n_cs"]
    # Approximate equatorial cell size in degrees (CS great-circle side).
    h = 90.0 / ns

    fig, (ax_rt, ax_fwd) = plt.subplots(1, 2, figsize=(13, 5),
                                         sharey=True)

    color_p0 = "#9467bd"
    color_p1 = "#1f77b4"
    color_p2 = "#ff7f0e"

    # ---- Left panel: round-trip ----
    plot_curve(ax_rt, h, d["std_p0"], color_p0, "-", "D",
               r"$p{=}0$ Level-2 mimetic")
    if "patch_p1" in d:
        plot_curve(ax_rt, h, d["patch_p1"], color_p1, "-", "o",
                   r"$p{=}1$ patch recovery + VEM round-trip")
    if "patch_p2" in d:
        plot_curve(ax_rt, h, d["patch_p2"], color_p2, "-", "o",
                   r"$p{=}2$ patch recovery + VEM round-trip")

    # Reference O(h^1) slope
    ref_y = None
    for cand in ("rt_p1", "std_p0"):
        if cand in d:
            v = d[cand]
            mask = np.isfinite(v) & (v > 0)
            if mask.sum() >= 1:
                ref_y = float(v[mask][0]) * 1.5
                break
    if ref_y is not None:
        h_ref = np.array([h[0], h[-1]])
        ax_rt.semilogy(h_ref, ref_y * (h_ref / h_ref[0]) ** 1,
                       "k:", lw=1.0, label=r"$O(h^1)$ reference")

    ax_rt.set_xlabel("Approximate cell size $h$ (degrees)")
    ax_rt.set_ylabel(r"Max cell-divergence error ($L_\infty$)")
    ax_rt.set_title("Round-trip RLL$\\to$CS$\\to$RLL")
    ax_rt.set_xscale("log")
    ax_rt.invert_xaxis()
    ax_rt.grid(True, which="both", alpha=0.3)
    ax_rt.legend(fontsize=8, loc="lower right")

    # ---- Right panel: forward-only ----
    if "fwd_patch_p1" in d:
        plot_curve(ax_fwd, h, d["fwd_patch_p1"], color_p1, "-", "o",
                   r"$p{=}1$ patch recovery (forward)")
    if "fwd_patch_p2" in d:
        plot_curve(ax_fwd, h, d["fwd_patch_p2"], color_p2, "-", "o",
                   r"$p{=}2$ patch recovery (forward)")

    if ref_y is not None:
        for slope, lbl in [(1, r"$O(h^1)$"), (2, r"$O(h^2)$"),
                            (3, r"$O(h^3)$")]:
            ax_fwd.semilogy(h_ref, ref_y * (h_ref / h_ref[0]) ** slope,
                            "k:", lw=0.8, alpha=0.5)
            ax_fwd.annotate(lbl, xy=(h_ref[-1],
                                     ref_y * (h_ref[-1] / h_ref[0]) ** slope),
                            fontsize=7, color="gray", ha="left", va="center")

    ax_fwd.set_xlabel("Approximate cell size $h$ (degrees)")
    ax_fwd.set_title("Forward-only RLL$\\to$CS")
    ax_fwd.set_xscale("log")
    ax_fwd.invert_xaxis()
    ax_fwd.grid(True, which="both", alpha=0.3)
    ax_fwd.legend(fontsize=8, loc="lower right")

    fig.suptitle("Spherical round-trip and forward-only convergence "
                 "($\\nabla_S Y_2^0$ field)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
