#!/usr/bin/env python3
"""Plot h-convergence of spherical edge transfer errors.

Usage:
    bash scripts/convergence_study.sh > /tmp/convergence.csv
    python3 scripts/plot_convergence.py /tmp/convergence.csv
"""

import csv
import math
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python3 plot_convergence.py <csv_file>", file=sys.stderr)
        return 1

    rows = []
    with open(sys.argv[1], newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)

    print(
        f"{'source_n':>10} {'h (rad)':>10} {'L2_rel':>12} {'Linf':>12} {'L1_cell':>12} {'Linf_cell':>12}"
    )
    print("-" * 70)

    prev_h = None
    prev_l2 = None
    for row in rows:
        src_n = int(row["source_n"])
        h = math.pi / (2.0 * src_n)
        l2 = float(row["edge_flux_l2_rel"])
        linf = float(row["edge_flux_linf"])
        l1 = float(row["cell_field_l1"])
        clinf = float(row["cell_field_linf"])

        rate_str = ""
        if prev_h is not None and prev_l2 is not None and l2 > 0.0 and prev_l2 > 0.0:
            rate = math.log(l2 / prev_l2) / math.log(h / prev_h)
            rate_str = f"  rate={rate:.2f}"

        print(
            f"{src_n:>10} {h:>10.4f} {l2:>12.4e} {linf:>12.4e} {l1:>12.4e} {clinf:>12.4e}{rate_str}"
        )
        prev_h = h
        prev_l2 = l2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
