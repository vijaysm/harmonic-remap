#!/usr/bin/env python3
"""Plot h-convergence of spherical edge transfer errors.

Usage:
    bash scripts/convergence_study.sh build > /tmp/convergence.csv
    python3 scripts/plot_convergence.py /tmp/convergence.csv [output_png]
"""

import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def load_rows(csv_path: Path):
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "source_n": int(row["source_n"]),
                    "target_n": int(row["target_n"]),
                    "h": math.pi / (2.0 * int(row["source_n"])),
                    "edge_flux_l2_rel": float(row["edge_flux_l2_rel"]),
                    "edge_flux_linf": float(row["edge_flux_linf"]),
                    "direct_cell_avg_l1": float(row["direct_cell_avg_l1"]),
                    "direct_cell_avg_linf": float(row["direct_cell_avg_linf"]),
                    "recon_cell_avg_l1": float(row["recon_cell_avg_l1"]),
                    "recon_cell_avg_linf": float(row["recon_cell_avg_linf"]),
                }
            )
    return rows


def print_table(rows):
    print(
        f"{'source_n':>10} {'h (rad)':>10} {'edge_L2':>12} {'edge_Linf':>12} "
        f"{'direct_L1':>12} {'recon_L1':>12}"
    )
    print("-" * 78)

    prev_h = None
    prev_l2 = None
    for row in rows:
        rate_str = ""
        if prev_h is not None and prev_l2 is not None:
            l2 = row["edge_flux_l2_rel"]
            if l2 > 0.0 and prev_l2 > 0.0:
                rate = math.log(l2 / prev_l2) / math.log(row["h"] / prev_h)
                rate_str = f"  rate={rate:.2f}"

        print(
            f"{row['source_n']:>10} {row['h']:>10.4f} "
            f"{row['edge_flux_l2_rel']:>12.4e} {row['edge_flux_linf']:>12.4e} "
            f"{row['direct_cell_avg_l1']:>12.4e} {row['recon_cell_avg_l1']:>12.4e}{rate_str}"
        )
        prev_h = row["h"]
        prev_l2 = row["edge_flux_l2_rel"]


def add_reference_slope(ax, h_values, anchor_error, order, label):
    if not h_values or anchor_error <= 0.0:
        return
    h0 = h_values[-1]
    ref_x = [h0, h_values[0]]
    ref_y = [anchor_error, anchor_error * (h_values[0] / h0) ** order]
    ax.loglog(ref_x, ref_y, "--", linewidth=1.0, label=label)


def save_plot(rows, output_path: Path):
    h = [row["h"] for row in rows]
    edge_l2 = [row["edge_flux_l2_rel"] for row in rows]
    edge_linf = [row["edge_flux_linf"] for row in rows]
    direct_l1 = [row["direct_cell_avg_l1"] for row in rows]
    recon_l1 = [row["recon_cell_avg_l1"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].loglog(h, edge_l2, "o-", linewidth=2.0, label="edge L2")
    axes[0].loglog(h, edge_linf, "s-", linewidth=2.0, label="edge Linf")
    add_reference_slope(axes[0], h, edge_l2[-1], 1.0, "O(h)")
    add_reference_slope(axes[0], h, edge_l2[-1], 2.0, "O(h^2)")
    axes[0].set_xlabel("h (radians)")
    axes[0].set_ylabel("edge-flux error")
    axes[0].set_title("Structured Spherical Edge Errors")
    axes[0].grid(True, which="both", linestyle=":", linewidth=0.6)
    axes[0].legend()

    axes[1].loglog(h, direct_l1, "o-", linewidth=2.0, label="direct cell-average L1")
    axes[1].loglog(h, recon_l1, "s-", linewidth=2.0, label="reconstructed cell-average L1")
    add_reference_slope(axes[1], h, direct_l1[-1], 1.0, "O(h)")
    add_reference_slope(axes[1], h, direct_l1[-1], 2.0, "O(h^2)")
    axes[1].set_xlabel("h (radians)")
    axes[1].set_ylabel("cell-average error")
    axes[1].set_title("Structured Spherical Cell-Average Errors")
    axes[1].grid(True, which="both", linestyle=":", linewidth=0.6)
    axes[1].legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print("Usage: python3 plot_convergence.py <csv_file> [output_png]", file=sys.stderr)
        return 1

    csv_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) == 3 else None
    rows = load_rows(csv_path)
    if not rows:
        print("No rows found in convergence CSV", file=sys.stderr)
        return 1

    print_table(rows)
    if output_path is not None:
        save_plot(rows, output_path)
        print(f"Wrote {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
