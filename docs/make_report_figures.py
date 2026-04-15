#!/usr/bin/env python3
"""Generate deterministic figures used by mimetic_voronoi_report.tex.

The geometries mirror the C++ regression tests:
  * tests/conservative_intersection_test.cpp
  * tests/voronoi_intersection_test.cpp

The script intentionally has no dependency on MOAB.  It reproduces only the
planar polygon coordinates and clipping operations needed for report figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PolygonPatch


BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
GRAY = "#555555"
LIGHT_GRAY = "#E8E8E8"


def signed_area(points):
    area2 = 0.0
    for i, (x0, y0) in enumerate(points):
        x1, y1 = points[(i + 1) % len(points)]
        area2 += x0 * y1 - x1 * y0
    return 0.5 * area2


def polygon_area(points):
    return abs(signed_area(points))


def ensure_ccw(points):
    return list(points) if signed_area(points) >= 0.0 else list(reversed(points))


def centroid(points):
    area2 = 0.0
    cx = 0.0
    cy = 0.0
    for i, (x0, y0) in enumerate(points):
        x1, y1 = points[(i + 1) % len(points)]
        cross = x0 * y1 - x1 * y0
        area2 += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    return (cx / (3.0 * area2), cy / (3.0 * area2))


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def scale(s, a):
    return (s * a[0], s * a[1])


def clip_by_halfplane(polygon, normal, offset, tol=1.0e-13):
    if not polygon:
        return []

    def inside(p):
        return dot(normal, p) <= offset + tol

    def intersect(a, b):
        da = dot(normal, a) - offset
        db = dot(normal, b) - offset
        return add(a, scale(da / (da - db), sub(b, a)))

    output = []
    previous = polygon[-1]
    previous_inside = inside(previous)
    for current in polygon:
        current_inside = inside(current)
        if current_inside:
            if not previous_inside:
                output.append(intersect(previous, current))
            output.append(current)
        elif previous_inside:
            output.append(intersect(previous, current))
        previous = current
        previous_inside = current_inside
    return output


def convex_intersection(subject, clipper):
    subject = ensure_ccw(subject)
    clipper = ensure_ccw(clipper)
    for i, a in enumerate(clipper):
        b = clipper[(i + 1) % len(clipper)]
        edge = sub(b, a)
        inward = (-edge[1], edge[0])
        subject = clip_by_halfplane(subject, (-inward[0], -inward[1]), -dot(inward, a))
        if len(subject) < 3:
            return []
    return subject


def voronoi_cell(seed, seeds):
    cell = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    for other in seeds:
        if (other[0] - seed[0]) ** 2 + (other[1] - seed[1]) ** 2 < 1.0e-24:
            continue
        normal = (2.0 * (other[0] - seed[0]), 2.0 * (other[1] - seed[1]))
        offset = other[0] ** 2 + other[1] ** 2 - seed[0] ** 2 - seed[1] ** 2
        cell = clip_by_halfplane(cell, normal, offset)
        if len(cell) < 3:
            return []
    return ensure_ccw(cell)


def add_polygon(ax, points, edgecolor, facecolor="none", alpha=1.0, lw=1.5, ls="-", label=None, zorder=1):
    patch = PolygonPatch(
        points,
        closed=True,
        facecolor=facecolor,
        edgecolor=edgecolor,
        alpha=alpha,
        linewidth=lw,
        linestyle=ls,
        label=label,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def finish_unit_square(ax):
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.04)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.grid(True, alpha=0.25, linewidth=0.5)


def rect_poly(xmin, xmax, ymin, ymax):
    return [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]


def make_rectangular_overlap(figdir):
    source = [rect_poly(0.0, 0.5, 0.0, 0.5), rect_poly(0.5, 1.0, 0.0, 0.5),
              rect_poly(0.0, 0.5, 0.5, 1.0), rect_poly(0.5, 1.0, 0.5, 1.0)]
    target_x = [0.0, 0.25, 0.65, 1.0]
    target_y = [0.0, 0.40, 0.70, 1.0]
    target = [
        rect_poly(target_x[i], target_x[i + 1], target_y[j], target_y[j + 1])
        for j in range(len(target_y) - 1)
        for i in range(len(target_x) - 1)
    ]

    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    for poly in source:
        add_polygon(ax, poly, BLUE, facecolor="#DCEBFA", alpha=0.45, lw=2.0, label=None, zorder=1)
    for poly in target:
        add_polygon(ax, poly, ORANGE, facecolor="none", alpha=1.0, lw=1.4, ls="--", zorder=3)

    highlighted = rect_poly(0.25, 0.50, 0.00, 0.40)
    add_polygon(ax, highlighted, GREEN, facecolor="#CFEFE3", alpha=0.75, lw=2.2, zorder=4)
    ax.plot([], [], color=BLUE, lw=2.0, label="source cells")
    ax.plot([], [], color=ORANGE, lw=1.4, ls="--", label="target cells")
    ax.plot([], [], color=GREEN, lw=2.2, label="one clipped overlap")
    ax.text(0.03, 0.97, "16 nonempty intersections", ha="left", va="top", fontsize=10,
            bbox={"facecolor": "white", "edgecolor": LIGHT_GRAY, "alpha": 0.9, "pad": 3})
    finish_unit_square(ax)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(figdir / "rectangular_overlap.pdf")
    plt.close(fig)


def source_and_target_voronoi():
    source_seeds = [
        (0.12, 0.18), (0.42, 0.12), (0.78, 0.20), (0.18, 0.52), (0.52, 0.48),
        (0.86, 0.55), (0.20, 0.84), (0.55, 0.86), (0.88, 0.86),
    ]
    target_seeds = [
        (0.10, 0.12), (0.34, 0.18), (0.62, 0.12), (0.90, 0.18), (0.18, 0.38),
        (0.48, 0.34), (0.76, 0.42), (0.10, 0.68), (0.38, 0.62), (0.66, 0.70),
        (0.92, 0.68), (0.24, 0.92), (0.58, 0.90), (0.86, 0.92),
    ]
    source_cells = [voronoi_cell(seed, source_seeds) for seed in source_seeds]
    target_cells = [voronoi_cell(seed, target_seeds) for seed in target_seeds]
    return source_seeds, source_cells, target_seeds, target_cells


def make_voronoi_patch(figdir):
    source_seeds, source_cells, target_seeds, target_cells = source_and_target_voronoi()
    fig, ax = plt.subplots(figsize=(5.3, 4.7))
    for poly in source_cells:
        add_polygon(ax, poly, BLUE, facecolor="#DCEBFA", alpha=0.28, lw=1.8, zorder=1)
        if len(poly) > 4:
            x, y = centroid(poly)
            ax.text(x, y, str(len(poly)), color=BLUE, fontsize=8, ha="center", va="center", zorder=5)
    for poly in target_cells:
        add_polygon(ax, poly, ORANGE, facecolor="none", alpha=1.0, lw=1.2, ls="--", zorder=3)
    ax.scatter([p[0] for p in source_seeds], [p[1] for p in source_seeds],
               s=18, color=BLUE, edgecolor="white", linewidth=0.4, zorder=6, label="source seeds")
    ax.scatter([p[0] for p in target_seeds], [p[1] for p in target_seeds],
               s=22, color=ORANGE, marker="x", linewidth=1.0, zorder=6, label="target seeds")
    ax.plot([], [], color=BLUE, lw=1.8, label="source Voronoi")
    ax.plot([], [], color=ORANGE, lw=1.2, ls="--", label="target Voronoi")
    ax.text(0.03, 0.97, "numbers mark source n-gons", ha="left", va="top", fontsize=9,
            bbox={"facecolor": "white", "edgecolor": LIGHT_GRAY, "alpha": 0.9, "pad": 3})
    finish_unit_square(ax)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "voronoi_patch.pdf")
    plt.close(fig)


def make_voronoi_overlap_detail(figdir):
    _, source_cells, _, target_cells = source_and_target_voronoi()
    
    # We want a few different elements in different parts of the domain.
    # Let's collect all valid overlaps of n-gon to n-gon.
    valid_overlaps = []
    for si, source in enumerate(source_cells):
        for ti, target in enumerate(target_cells):
            overlap = convex_intersection(source, target)
            if len(overlap) < 3:
                continue
            area = polygon_area(overlap)
            if len(source) > 4 and len(target) > 4 and area > 1.0e-12:
                valid_overlaps.append((area, si, ti, source, target, overlap))
    
    if not valid_overlaps:
        raise RuntimeError("No n-gon to n-gon overlap found")
        
    # Sort by area to get a few good ones
    valid_overlaps.sort(key=lambda x: x[0], reverse=True)
    
    # Pick top 3 that are somewhat spatially distinct if possible, 
    # but just picking top 3 or specific ones will do. 
    # Let's pick 3 distinct overlaps (e.g., top 1, and two others)
    selected = [valid_overlaps[0]]
    for cand in valid_overlaps[1:]:
        # Ensure it's not the exact same source or target to get variety
        if cand[1] not in [s[1] for s in selected] and cand[2] not in [s[2] for s in selected]:
            selected.append(cand)
            if len(selected) == 3:
                break
                
    if len(selected) < 3:
        # If we couldn't find 3 distinct ones, just take the top 3
        selected = valid_overlaps[:3]
        
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    
    # Draw all background cells faintly
    for poly in source_cells:
        add_polygon(ax, poly, "#B0D0ED", facecolor="none", alpha=0.3, lw=1.0, zorder=0)
    for poly in target_cells:
        add_polygon(ax, poly, "#E8A87C", facecolor="none", alpha=0.3, lw=1.0, ls="--", zorder=0)

    # Highlight selected ones
    for area, si, ti, source, target, overlap in selected:
        add_polygon(ax, source, BLUE, facecolor="#DCEBFA", alpha=0.55, lw=2.0, zorder=1)
        add_polygon(ax, target, ORANGE, facecolor="#F6D6C8", alpha=0.38, lw=2.0, ls="--", zorder=2)
        add_polygon(ax, overlap, GREEN, facecolor="#BDEBD9", alpha=0.85, lw=2.2, zorder=4)
        
        # We don't want to plot dots for all points, it gets messy. Just the overlap maybe.
        # ax.scatter([p[0] for p in overlap], [p[1] for p in overlap], s=18, color=GREEN, zorder=5)
        x, y = centroid(overlap)
        ax.text(x, y, "$O$", fontsize=12, color=GRAY, ha="center", va="center", zorder=6)
        
    # Add dummy lines for legend
    ax.plot([], [], color=BLUE, lw=2.0, label="source n-gon")
    ax.plot([], [], color=ORANGE, lw=2.0, ls="--", label="target n-gon")
    ax.plot([], [], color=GREEN, lw=2.2, label="intersection")
    
    finish_unit_square(ax)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "voronoi_overlap_detail.pdf")
    plt.close(fig)


def make_results_summary(figdir):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8.0, 3.6))

    labels = ["source\nn>4", "target\nn>4", "overlaps", "n-gon\noverlaps"]
    counts = [7, 10, 36, 23]
    colors = [BLUE, ORANGE, GRAY, GREEN]
    ax0.bar(labels, counts, color=colors, alpha=0.82)
    ax0.set_ylabel("count")
    ax0.set_title("Voronoi patch coverage")
    ax0.grid(True, axis="y", alpha=0.25)
    for i, value in enumerate(counts):
        ax0.text(i, value + 0.8, str(value), ha="center", va="bottom", fontsize=9)
    ax0.set_ylim(0, 40)

    error_labels = ["patch", "rect.\noverlap", "rect.\nedge", "Vor.\noverlap", "Vor.\nedge"]
    errors = [1.0e-16, 4.44e-16, 5.55e-17, 4.44e-16, 2.78e-16]
    ax1.bar(error_labels, errors, color=[BLUE, PURPLE, GRAY, GREEN, ORANGE], alpha=0.82)
    ax1.axhline(5.0e-10, color=ORANGE, linestyle="--", linewidth=1.4, label="loose test tolerance")
    ax1.set_yscale("log")
    ax1.set_ylim(1.0e-17, 1.0e-8)
    ax1.set_ylabel("maximum absolute error")
    ax1.set_title("Conservation residuals")
    ax1.grid(True, axis="y", which="both", alpha=0.25)
    ax1.legend(frameon=True, framealpha=0.95, fontsize=8)

    fig.tight_layout()
    fig.savefig(figdir / "results_summary.pdf")
    plt.close(fig)


def main():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    figdir = Path(__file__).resolve().parent / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    make_rectangular_overlap(figdir)
    make_voronoi_patch(figdir)
    make_voronoi_overlap_detail(figdir)
    make_results_summary(figdir)


if __name__ == "__main__":
    main()
