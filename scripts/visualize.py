#!/usr/bin/env python3

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np


def read_mesh(filename: Path):
    polys = []
    vals = []
    with filename.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if not parts:
                continue
            value = float(parts[2])
            nverts = int(parts[3])
            verts = []
            idx = 4
            for _ in range(nverts):
                verts.append([float(parts[idx]), float(parts[idx + 1])])
                idx += 2
            polys.append(verts)
            vals.append(value)
    return polys, np.array(vals)


def render_triplet(source_file: Path, output_dir: Path):
    base_name = source_file.name.replace("_source.txt", "")
    target_file = source_file.with_name(base_name + "_target.txt")
    target_exact_file = source_file.with_name(base_name + "_target_exact.txt")
    if not target_file.exists() or not target_exact_file.exists():
        return

    src_polys, src_vals = read_mesh(source_file)
    tgt_polys, tgt_vals = read_mesh(target_file)
    _, tgt_exact_vals = read_mesh(target_exact_file)
    tgt_error = tgt_vals - tgt_exact_vals

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    vmin = min(src_vals.min(), tgt_vals.min(), tgt_exact_vals.min())
    vmax = max(src_vals.max(), tgt_vals.max(), tgt_exact_vals.max())
    err_max = max(np.abs(tgt_error).max(), 1.0e-16)

    panels = [
        (axes[0], src_polys, src_vals, "viridis", (vmin, vmax), "Source Analytical Projection"),
        (axes[1], tgt_polys, tgt_exact_vals, "viridis", (vmin, vmax), "Target Exact Projection"),
        (axes[2], tgt_polys, tgt_vals, "viridis", (vmin, vmax), "Target Harmonic Projection"),
        (axes[3], tgt_polys, tgt_error, "RdBu_r", (-err_max, err_max), f"Target Error Profile\n(Max: {err_max:.2e})"),
    ]

    value_mappables = []
    error_mappable = None
    for ax, polys, values, cmap, clim, title in panels:
        coll = PolyCollection(polys, array=values, cmap=cmap, edgecolors="k", linewidths=0.5)
        coll.set_clim(*clim)
        ax.add_collection(coll)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal")
        ax.set_title(title)
        if cmap == "viridis":
            value_mappables.append(coll)
        else:
            error_mappable = coll

    if value_mappables:
        fig.colorbar(value_mappables[-1], ax=axes[:3], orientation="horizontal", fraction=0.06, pad=0.10)
    if error_mappable is not None:
        fig.colorbar(error_mappable, ax=axes[3], orientation="horizontal", fraction=0.06, pad=0.10)

    display_name = base_name.replace("vis_", "").replace("_", " ")
    output_name = base_name.replace(": ", "__").replace(" ", "_")
    fig.suptitle(display_name, fontsize=14)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{output_name}.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


def main():
    input_dir = Path(sys.argv[1]) if len(sys.argv) >= 2 else Path.cwd()
    output_dir = Path(sys.argv[2]) if len(sys.argv) >= 3 else input_dir
    for source_file in sorted(input_dir.glob("vis_*_source.txt")):
        render_triplet(source_file, output_dir)


if __name__ == "__main__":
    main()
