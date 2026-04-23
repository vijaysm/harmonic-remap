#!/usr/bin/env python3

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.collections import LineCollection
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


def read_edges(filename: Path):
    segments = []
    vals = []
    counts = []
    with filename.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if not parts:
                continue
            x0, y0, x1, y1 = map(float, parts[:4])
            value = float(parts[4])
            count = int(parts[5])
            segments.append([[x0, y0], [x1, y1]])
            vals.append(value)
            counts.append(count)
    return segments, np.array(vals), np.array(counts)


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
        (axes[0], src_polys, src_vals, "viridis", (vmin, vmax), "Source Exact Cell Divergence"),
        (axes[1], tgt_polys, tgt_exact_vals, "viridis", (vmin, vmax), "Target Exact Cell Divergence"),
        (axes[2], tgt_polys, tgt_vals, "viridis", (vmin, vmax), "Target Reconstructed Cell Divergence"),
        (axes[3], tgt_polys, tgt_error, "RdBu_r", (-err_max, err_max), f"Target Cell-Divergence Error\n(Max: {err_max:.2e})"),
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
    fig.suptitle(f"{display_name} cell-divergence diagnostic", fontsize=14)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{output_name}.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


def render_hdiv_comparison(source_file: Path, output_dir: Path):
    base_name = source_file.name.replace("_source.txt", "")
    target_file = source_file.with_name(base_name + "_target.txt")
    target_exact_file = source_file.with_name(base_name + "_target_exact.txt")
    target_conforming_file = source_file.with_name(base_name + "_target_conforming.txt")
    if not target_file.exists() or not target_exact_file.exists() or not target_conforming_file.exists():
        return

    src_polys, src_vals = read_mesh(source_file)
    tgt_polys, tgt_vals = read_mesh(target_file)
    _, tgt_exact_vals = read_mesh(target_exact_file)
    _, tgt_conforming_vals = read_mesh(target_conforming_file)

    conforming_error = tgt_conforming_vals - tgt_exact_vals

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    vmin = min(src_vals.min(), tgt_vals.min(), tgt_exact_vals.min(), tgt_conforming_vals.min())
    vmax = max(src_vals.max(), tgt_vals.max(), tgt_exact_vals.max(), tgt_conforming_vals.max())
    err_max = max(np.abs(conforming_error).max(), 1.0e-16)

    panels = [
        (axes[0], src_polys, src_vals, "viridis", (vmin, vmax), "Source Exact Cell Divergence"),
        (axes[1], tgt_polys, tgt_exact_vals, "viridis", (vmin, vmax), "Target Exact Cell Divergence"),
        (axes[2], tgt_polys, tgt_conforming_vals, "viridis", (vmin, vmax), "Global Constrained Cell Divergence"),
        (axes[3], tgt_polys, conforming_error, "RdBu_r", (-err_max, err_max), f"Global Constrained Error\n(Max: {np.abs(conforming_error).max():.2e})"),
    ]

    value_mappables = []
    error_mappables = []
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
            error_mappables.append(coll)

    if value_mappables:
        fig.colorbar(value_mappables[-1], ax=axes[:3],
                     orientation="horizontal", fraction=0.06, pad=0.10)
    if error_mappables:
        fig.colorbar(error_mappables[-1], ax=axes[3],
                     orientation="horizontal", fraction=0.06, pad=0.10)

    display_name = base_name.replace("vis_", "").replace("_", " ")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = base_name.replace(": ", "__").replace(" ", "_")
    output_path = output_dir / f"{output_name}_hdiv_compare.png"
    fig.suptitle(f"{display_name} global constrained cell-divergence diagnostic", fontsize=14)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


def render_edge_jump(source_file: Path, output_dir: Path):
    base_name = source_file.name.replace("_source.txt", "")
    raw_edge_file = source_file.with_name(base_name + "_raw_edge_jump.txt")
    conf_edge_file = source_file.with_name(base_name + "_conforming_edge_jump.txt")
    if not raw_edge_file.exists() or not conf_edge_file.exists():
        return

    raw_segments, raw_vals, raw_counts = read_edges(raw_edge_file)
    conf_segments, conf_vals, conf_counts = read_edges(conf_edge_file)

    if len(raw_segments) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    jump_max = max(np.abs(raw_vals[raw_counts == 2]).max(initial=0.0),
                   np.abs(conf_vals[conf_counts == 2]).max(initial=0.0),
                   1.0e-16)

    panels = [
        (axes[0], raw_segments, raw_vals, raw_counts, f"Raw Edge Jump\n(Max: {np.abs(raw_vals[raw_counts == 2]).max(initial=0.0):.2e})"),
        (axes[1], conf_segments, conf_vals, conf_counts, f"Global Constrained Edge Jump\n(Max: {np.abs(conf_vals[conf_counts == 2]).max(initial=0.0):.2e})"),
    ]

    mappable = None
    for ax, segments, values, counts, title in panels:
        background = LineCollection(segments, colors="#999999", linewidths=1.0, alpha=0.6)
        ax.add_collection(background)
        interior_segments = [seg for seg, count in zip(segments, counts) if count == 2]
        interior_values = np.array([val for val, count in zip(values, counts) if count == 2])
        if len(interior_segments) > 0:
            coll = LineCollection(interior_segments, array=interior_values, cmap="RdBu_r", linewidths=3.0)
            coll.set_clim(-jump_max, jump_max)
            ax.add_collection(coll)
            mappable = coll
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal")
        ax.set_title(title)

    if mappable is not None:
        fig.colorbar(mappable, ax=axes, orientation="horizontal", fraction=0.06, pad=0.12)

    display_name = base_name.replace("vis_", "").replace("_", " ")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = base_name.replace(": ", "__").replace(" ", "_")
    output_path = output_dir / f"{output_name}_edge_jump.png"
    fig.suptitle(f"{display_name} target-edge jump diagnostic", fontsize=14)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


def render_order_comparison(source_file: Path, output_dir: Path):
    """Render 2x3 comparison: source exact / p=1 recon / p=1 error,
       target exact / p=3 recon / p=3 error."""
    base_name = source_file.name.replace("_source.txt", "")
    target_p1_file = source_file.with_name(base_name + "_target_p1.txt")
    target_p3_file = source_file.with_name(base_name + "_target_p3.txt")
    target_exact_file = source_file.with_name(base_name + "_target_exact.txt")
    if not all(f.exists() for f in [target_p1_file, target_p3_file, target_exact_file]):
        return

    src_polys, src_vals = read_mesh(source_file)
    tgt_p1_polys, tgt_p1_vals = read_mesh(target_p1_file)
    tgt_p3_polys, tgt_p3_vals = read_mesh(target_p3_file)
    _, tgt_exact_vals = read_mesh(target_exact_file)
    p1_error = tgt_p1_vals - tgt_exact_vals
    p3_error = tgt_p3_vals - tgt_exact_vals

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    vmin = min(src_vals.min(), tgt_p1_vals.min(), tgt_p3_vals.min(), tgt_exact_vals.min())
    vmax = max(src_vals.max(), tgt_p1_vals.max(), tgt_p3_vals.max(), tgt_exact_vals.max())
    err_max = max(np.abs(p1_error).max(), np.abs(p3_error).max(), 1.0e-16)

    # Auto-detect coordinate bounds from all polygon vertices
    all_polys = src_polys + tgt_p1_polys + tgt_p3_polys
    all_coords = np.concatenate([np.array(p) for p in all_polys if len(p) >= 3])
    xlo, ylo = all_coords.min(axis=0) - 0.02
    xhi, yhi = all_coords.max(axis=0) + 0.02

    field_panels = [
        (axes[0, 0], src_polys, src_vals, "Source Exact Divergence"),
        (axes[0, 1], tgt_p1_polys, tgt_p1_vals, r"Target Recon. ($p=1$)"),
        (axes[1, 0], tgt_p1_polys, tgt_exact_vals, "Target Exact Divergence"),
        (axes[1, 1], tgt_p3_polys, tgt_p3_vals, r"Target Recon. ($p=3$)"),
    ]
    error_panels = [
        (axes[0, 2], tgt_p1_polys, p1_error,
         f"$p=1$ Error (max: {np.abs(p1_error).max():.2e})"),
        (axes[1, 2], tgt_p3_polys, p3_error,
         f"$p=3$ Error (max: {np.abs(p3_error).max():.2e})"),
    ]

    value_mappable = None
    error_mappable = None
    for ax, polys, values, title in field_panels:
        coll = PolyCollection(polys, array=values, cmap="viridis", edgecolors="k", linewidths=0.3)
        coll.set_clim(vmin, vmax)
        ax.add_collection(coll)
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11)
        value_mappable = coll

    for ax, polys, values, title in error_panels:
        coll = PolyCollection(polys, array=values, cmap="RdBu_r", edgecolors="k", linewidths=0.3)
        coll.set_clim(-err_max, err_max)
        ax.add_collection(coll)
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11)
        error_mappable = coll

    # Colorbars below the figure, outside panel bounding boxes
    if value_mappable is not None:
        fig.colorbar(value_mappable, ax=axes[:, :2].ravel().tolist(),
                     orientation="horizontal", fraction=0.05, pad=0.10,
                     label="Cell-averaged divergence")
    if error_mappable is not None:
        fig.colorbar(error_mappable, ax=axes[:, 2].ravel().tolist(),
                     orientation="horizontal", fraction=0.05, pad=0.10,
                     label="Divergence error")

    display_name = base_name.replace("vis_order_compare_", "").replace("_", " ")
    output_name = base_name.replace(": ", "__").replace(" ", "_")
    fig.suptitle(f"Order comparison: {display_name}", fontsize=14)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{output_name}_order_compare.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


def render_roundtrip(source_file: Path, output_dir: Path):
    """Render 3-panel round-trip: source exact, round-tripped, error."""
    base_name = source_file.name.replace("_source.txt", "")
    target_p1_file = source_file.with_name(base_name + "_target_p1.txt")
    target_exact_file = source_file.with_name(base_name + "_target_exact.txt")
    if not target_p1_file.exists() or not target_exact_file.exists():
        return

    src_polys, src_vals = read_mesh(source_file)
    tgt_polys, tgt_vals = read_mesh(target_p1_file)
    _, tgt_exact_vals = read_mesh(target_exact_file)
    error = tgt_vals - tgt_exact_vals

    all_coords = np.concatenate([np.array(p) for p in src_polys + tgt_polys if len(p) >= 3])
    xlo, ylo = all_coords.min(axis=0) - 0.05
    xhi, yhi = all_coords.max(axis=0) + 0.05

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    vmin = min(src_vals.min(), tgt_vals.min(), tgt_exact_vals.min())
    vmax = max(src_vals.max(), tgt_vals.max(), tgt_exact_vals.max())
    err_max = max(np.abs(error).max(), 1.0e-16)

    panels = [
        (axes[0], src_polys, src_vals, "viridis", (vmin, vmax), "Source Exact Divergence"),
        (axes[1], tgt_polys, tgt_vals, "viridis", (vmin, vmax), "Round-Tripped Divergence"),
        (axes[2], tgt_polys, error, "RdBu_r", (-err_max, err_max),
         f"Round-Trip Error (max: {np.abs(error).max():.2e})"),
    ]

    value_mappable = None
    error_mappable = None
    for ax, polys, values, cmap, clim, title in panels:
        coll = PolyCollection(polys, array=values, cmap=cmap, edgecolors="k", linewidths=0.3)
        coll.set_clim(*clim)
        ax.add_collection(coll)
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11)
        if cmap == "viridis":
            value_mappable = coll
        else:
            error_mappable = coll

    if value_mappable is not None:
        fig.colorbar(value_mappable, ax=axes[:2].tolist(),
                     orientation="horizontal", fraction=0.06, pad=0.12,
                     label="Cell-averaged divergence")
    if error_mappable is not None:
        fig.colorbar(error_mappable, ax=[axes[2]],
                     orientation="horizontal", fraction=0.06, pad=0.12,
                     label="Divergence error")

    display_name = base_name.replace("vis_roundtrip_", "").replace("_", " ")
    output_name = base_name.replace(": ", "__").replace(" ", "_")
    fig.suptitle(f"Round-trip: {display_name} (stereographic projection)", fontsize=14)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{output_name}_roundtrip.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


def main():
    input_dir = Path(sys.argv[1]) if len(sys.argv) >= 2 else Path.cwd()
    output_dir = Path(sys.argv[2]) if len(sys.argv) >= 3 else input_dir
    for source_file in sorted(input_dir.glob("vis_*_source.txt")):
        if "order_compare" in source_file.name:
            render_order_comparison(source_file, output_dir)
        elif "roundtrip" in source_file.name:
            render_roundtrip(source_file, output_dir)
        else:
            render_triplet(source_file, output_dir)
            render_hdiv_comparison(source_file, output_dir)
            render_edge_jump(source_file, output_dir)


if __name__ == "__main__":
    main()
