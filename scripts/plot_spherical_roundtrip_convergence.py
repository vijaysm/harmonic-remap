"""
Plot spherical round-trip convergence study: standard [P_p]^2 vs Piola RT.

Usage:
    python scripts/plot_spherical_roundtrip_convergence.py \
        docs/spherical_roundtrip_convergence.csv \
        docs/figures/spherical_roundtrip_convergence.png
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv

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
    # Convert to sorted arrays
    ns = sorted(set(n for k in data for n in data[k]))
    result = {"n_cs": np.array(ns)}
    for k, d in data.items():
        result[k] = np.array([d.get(n, np.nan) for n in ns])
    return result

def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "docs/spherical_roundtrip_convergence.csv"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "docs/figures/spherical_roundtrip_convergence.png"

    d = read_csv(csv_path)
    ns = d["n_cs"]
    h  = 90.0 / ns  # approximate cell size in degrees

    fig, ax = plt.subplots(figsize=(7, 5))

    colors = {"p0": "#9467bd", "p1": "#1f77b4", "p2": "#ff7f0e"}
    ls_std  = "--"
    ls_rt   = "-"

    # p=0: MimeticInterpolator (same std and RT — only plot once)
    y0 = d["std_p0"]
    mask0 = np.isfinite(y0) & (y0 > 0)
    if mask0.sum() >= 1:
        ax.semilogy(h[mask0], y0[mask0],
                    color=colors["p0"], ls="-", marker="D", ms=6,
                    label="p=0 (Level-2 mimetic)")

    # Standard (dashed) for p=1,2
    for p in [1, 2]:
        key = f"std_p{p}"
        y = d[key]
        mask = np.isfinite(y) & (y > 0)
        if mask.sum() >= 1:
            ax.semilogy(h[mask], y[mask],
                        color=colors[f"p{p}"], ls=ls_std, marker="s", ms=6,
                        label=f"Standard p={p}")

    # Piola RT (solid) for p=1,2
    for p in [1, 2]:
        key = f"rt_p{p}"
        y = d[key]
        mask = np.isfinite(y) & (y > 0)
        if mask.sum() >= 1:
            ax.semilogy(h[mask], y[mask],
                        color=colors[f"p{p}"], ls=ls_rt, marker="o", ms=6,
                        label=f"Piola RT p={p}")

    # Reference slope O(h^1)
    h_ref = np.array([h[0], h[-1]])
    e0 = d["rt_p1"][np.isfinite(d["rt_p1"])][0] * 2.0
    ax.semilogy(h_ref, e0 * (h_ref / h_ref[0])**1, "k:", lw=1.2, label=r"$O(h^1)$")

    ax.set_xlabel("Approximate cell size h (degrees)", fontsize=12)
    ax.set_ylabel("Max cell-div error (L∞)", fontsize=12)
    ax.set_title("Round-trip RLL→CS→RLL: Standard vs Piola RT", fontsize=12)
    ax.legend(ncol=2, fontsize=9, loc="upper left")
    ax.invert_xaxis()
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)

    # Annotation: highlight divergence
    std_p2 = d["std_p2"]
    valid = np.isfinite(std_p2) & (std_p2 > 0)
    if valid.sum() >= 2:
        ax.annotate("Standard p=1,2\ndivergence", xy=(h[valid][-1], std_p2[valid][-1]),
                    xytext=(h[valid][-1]*1.8, std_p2[valid][-1]*2),
                    arrowprops=dict(arrowstyle="->", color="gray"),
                    fontsize=8, color="gray")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
