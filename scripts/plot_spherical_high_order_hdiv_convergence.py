#!/usr/bin/env python3
import csv
import math
import sys
from collections import defaultdict

import matplotlib.pyplot as plt


def load_rows(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "domain": row["domain"],
                    "order": int(row["order"]),
                    "h": float(row["h"]),
                    "l2_moment0_rel": float(row["l2_moment0_rel"]),
                    "l2_all_rel": float(row["l2_all_rel"]),
                    "conforming_divergence_residual": float(row["conforming_divergence_residual"]),
                }
            )
    return rows


def convergence_rate(points):
    if len(points) < 2:
        return float("nan")
    rates = []
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        if x0 <= 0.0 or x1 <= 0.0 or y0 <= 0.0 or y1 <= 0.0 or math.isclose(x0, x1):
            continue
        rates.append(math.log(y0 / y1) / math.log(x0 / x1))
    if not rates:
        return float("nan")
    return sum(rates) / len(rates)


def main():
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: plot_spherical_high_order_hdiv_convergence.py <csv> <output_png>"
        )

    rows = load_rows(sys.argv[1])
    output_path = sys.argv[2]

    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["domain"]][row["order"]].append(row)

    domain_titles = {
        "spherical_cubed_sphere": "Cubed Sphere",
        "spherical_voronoi_patch": "Voronoi Patch",
    }
    metric_titles = {
        "l2_moment0_rel": "Relative L2 moment-0 edge error",
        "l2_all_rel": "Relative L2 all-moment edge error",
    }
    colors = {1: "#1b9e77", 2: "#d95f02", 3: "#7570b3"}

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for row_index, domain in enumerate(["spherical_cubed_sphere", "spherical_voronoi_patch"]):
        for col_index, metric_key in enumerate(["l2_moment0_rel", "l2_all_rel"]):
            ax = axes[row_index][col_index]
            for order in sorted(grouped[domain]):
                entries = sorted(grouped[domain][order], key=lambda row: row["h"], reverse=True)
                hs = [entry["h"] for entry in entries]
                errs = [entry[metric_key] for entry in entries]
                rate = convergence_rate(list(zip(hs, errs)))
                label = f"p={order}"
                if not math.isnan(rate):
                    label += f" (rate {rate:.2f})"
                ax.loglog(hs, errs, marker="o", linewidth=2, color=colors[order], label=label)

            ax.set_title(f"{domain_titles[domain]}: {metric_titles[metric_key]}")
            ax.set_xlabel("h")
            ax.set_ylabel(metric_titles[metric_key])
            ax.grid(True, which="both", linestyle=":", linewidth=0.6)
            ax.legend(frameon=False)

    fig.suptitle("Spherical High-Order Split-Moment Edge Transfer Convergence")
    fig.savefig(output_path, dpi=200)


if __name__ == "__main__":
    main()
