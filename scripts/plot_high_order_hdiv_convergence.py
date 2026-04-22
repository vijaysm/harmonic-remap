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
                    "l2_flux_rel": float(row["l2_flux_rel"]),
                }
            )
    return rows


def convergence_rate(points):
    if len(points) < 2:
        return float("nan")
    x0, y0 = points[0]
    x1, y1 = points[-1]
    if y0 <= 0.0 or y1 <= 0.0:
        return float("nan")
    return math.log(y0 / y1) / math.log(x0 / x1)


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: plot_high_order_hdiv_convergence.py <csv> <output_png>")

    csv_path = sys.argv[1]
    output_path = sys.argv[2]
    rows = load_rows(csv_path)

    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["domain"]][row["order"]].append((row["h"], row["l2_flux_rel"]))

    domain_titles = {
        "quad_to_quad": "Quad to Quad",
        "voronoi_to_voronoi": "Voronoi to Voronoi",
    }
    colors = {1: "#1b9e77", 2: "#d95f02", 3: "#7570b3"}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, domain in zip(axes, ["quad_to_quad", "voronoi_to_voronoi"]):
        for order in sorted(grouped[domain]):
            points = sorted(grouped[domain][order], reverse=True)
            hs = [p[0] for p in points]
            errs = [p[1] for p in points]
            rate = convergence_rate(points)
            label = f"p={order}"
            if not math.isnan(rate):
                label += f" (rate {rate:.2f})"
            ax.loglog(hs, errs, marker="o", linewidth=2, color=colors[order], label=label)

        ax.set_title(domain_titles[domain])
        ax.set_xlabel("h")
        ax.set_ylabel("Relative L2 edge-flux error")
        ax.grid(True, which="both", linestyle=":", linewidth=0.6)
        ax.legend(frameon=False)

    fig.suptitle("High-Order Polygonal H(div)-Style Edge Transfer Convergence")
    fig.savefig(output_path, dpi=200)


if __name__ == "__main__":
    main()
