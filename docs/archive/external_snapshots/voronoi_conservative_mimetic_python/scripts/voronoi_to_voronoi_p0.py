#!/usr/bin/env python3
"""
voronoi_to_voronoi_p0.py
========================
Python implementation of the lowest-order (p=0, level-2 mimetic)
conservative edge-flux projection from a source Voronoi mesh to a target
Voronoi mesh on the unit square [0, 1]^2.

This is a pure-Python re-implementation of the planar low-order
``MimeticInterpolator`` path in this repository's C++ library, restricted to
what is needed to demonstrate first-order convergence of the edge-flux
transfer and to write the resulting projection weights in SCRIP format.

Design reference: docs/superpowers/specs/2026-05-04-voronoi-p0-python-script-design.md

Algorithm summary (per source cell K with N edges and signed edge fluxes U):

    Reconstruct  u_h(x) = (d/2) x + sum_{k=1..K_max} [a_k grad P_k + b_k grad Q_k]
    where P_k(x,y)+iQ_k(x,y) = (x+iy)^k in centroid-relative coords,
          K_max = N // 2, N_h = 2 K_max,
          d = (sum U_e) / area(K) (discrete divergence theorem),
          (a_k, b_k) come from the KKT saddle-point system

        [V   C^T] [a]   [M]
        [C    0 ] [l] = [F]

    with V = Gram of basis gradients, C the edge-normal constraint operator,
    M the moment RHS, F the divergence-corrected edge-flux RHS.

For each target unique edge e_t with intrinsic normal n_t, we clip e_t against
each candidate source cell polygon (Sutherland-Hodgman), integrate u_h^{K_s}.n_t
on the subsegment via 4-point Gauss-Legendre, and scatter the resulting
linear-in-source-fluxes coefficients into a global sparse remap matrix W.

Conservation (verified at runtime): for any (K_s, K_t) with overlap O,
    sum over target edges of int_{O cap edge} u_h^{K_s} . n ds = d_{K_s} * area(O).

Run:
    python voronoi_to_voronoi_p0.py [--output-dir <dir>] [--levels N1,N2,N3,N4]

Outputs:
    voronoi_p0_weights_<field>_N<src>_to_N<tgt>.nc   SCRIP weight files
    voronoi_p0_convergence.png                       log-log convergence plot
    stdout                                           pass/fail and convergence tables
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import scipy.sparse as sp
from scipy.spatial import Voronoi, cKDTree

try:
    import netCDF4
except ImportError as exc:  # pragma: no cover
    raise SystemExit("netCDF4 is required: pip install netCDF4") from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit("matplotlib is required: pip install matplotlib") from exc


# ---------------------------------------------------------------------------
# Constants and quadrature rules
# ---------------------------------------------------------------------------

TOL_GEOM = 1.0e-12
TOL_CONSERVATION = 1.0e-12
TOL_EXACTNESS = 1.0e-9
TOL_CLIP = 1.0e-13

# 4-point Gauss-Legendre nodes/weights on [-1, 1] (degree-7 exact).
# Matches include/mimetic/mimetic.hpp:integrate_edge_scalar.
_GL4_X = np.array(
    [-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526]
)
_GL4_W = np.array(
    [0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538]
)

# 7-point symmetric triangle rule (degree-5 exact) on barycentric coords.
# Matches include/mimetic/mimetic.hpp:integrate_triangle_scalar.
_TRI7_A1 = 0.4701420641051151
_TRI7_A2 = 0.1012865073234563
_TRI7_BARY = np.array(
    [
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        [_TRI7_A1, _TRI7_A1, 1.0 - 2.0 * _TRI7_A1],
        [_TRI7_A1, 1.0 - 2.0 * _TRI7_A1, _TRI7_A1],
        [1.0 - 2.0 * _TRI7_A1, _TRI7_A1, _TRI7_A1],
        [_TRI7_A2, _TRI7_A2, 1.0 - 2.0 * _TRI7_A2],
        [_TRI7_A2, 1.0 - 2.0 * _TRI7_A2, _TRI7_A2],
        [1.0 - 2.0 * _TRI7_A2, _TRI7_A2, _TRI7_A2],
    ]
)
_TRI7_W = np.array(
    [
        0.225,
        0.1323941527885062,
        0.1323941527885062,
        0.1323941527885062,
        0.1259391805448271,
        0.1259391805448271,
        0.1259391805448271,
    ]
)

# 16-point GL for "exact" reference flux integration on edges.
_GL16_X, _GL16_W = np.polynomial.legendre.leggauss(16)


# ---------------------------------------------------------------------------
# Mesh data structure and Voronoi generation
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Mesh:
    """Half-edge-ish polygon mesh with unique edge DOFs (single signed scalar per edge).

    Sign convention per cell K visiting unique edge e = (v_low, v_high):
      forward (v_low -> v_high): cell_to_edge_sign = +1, cell outward normal
        equals edge intrinsic normal.
      reverse (v_high -> v_low): cell_to_edge_sign = -1, cell outward normal
        equals minus edge intrinsic normal.
    The single stored DOF for edge e is flux(e) = int_e u . n_intrinsic ds.
    """

    vertices: np.ndarray  # (Nv, 2)
    cells: list[np.ndarray]  # CCW vertex indices per cell
    cell_centroids: np.ndarray  # (Nc, 2)
    cell_areas: np.ndarray  # (Nc,)
    cell_radii: np.ndarray  # (Nc,) max distance from centroid to vertex
    unique_edges: np.ndarray  # (Ne, 2) [v_low, v_high]
    edge_lengths: np.ndarray  # (Ne,)
    edge_midpoints: np.ndarray  # (Ne, 2)
    edge_intrinsic_normals: np.ndarray  # (Ne, 2) right-rotation of (v_high - v_low)
    cell_to_edge_idx: list[np.ndarray]  # per cell, global edge indices in cell-traversal order
    cell_to_edge_sign: list[np.ndarray]  # per cell, +/-1 in cell-traversal order
    edge_to_cells: list[list[int]]  # per edge, owning cell indices (1 for boundary, 2 for interior)


def halton(n: int, seed_offset: int = 0) -> np.ndarray:
    """n 2D Halton points in (0,1)^2 with bases (2, 3). Deterministic."""

    def vdc(idx: int, base: int) -> float:
        result = 0.0
        f = 1.0 / base
        i = idx
        while i > 0:
            result += f * (i % base)
            i //= base
            f /= base
        return result

    pts = np.zeros((n, 2))
    for i in range(n):
        pts[i, 0] = vdc(i + seed_offset + 1, 2)
        pts[i, 1] = vdc(i + seed_offset + 1, 3)
    return pts


def signed_area(pts: np.ndarray) -> float:
    """Shoelace, signed. Positive for CCW."""
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def polygon_area_centroid(pts: np.ndarray) -> tuple[float, np.ndarray]:
    """Shoelace area and area-weighted centroid for a simple polygon."""
    n = pts.shape[0]
    a = 0.0
    cx = 0.0
    cy = 0.0
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        a += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    a *= 0.5
    if abs(a) < 1.0e-30:
        return 0.0, np.mean(pts, axis=0)
    return abs(a), np.array([cx / (6.0 * a), cy / (6.0 * a)])


def voronoi_in_unit_square(seeds: np.ndarray) -> Mesh:
    """Voronoi tessellation of (0,1)^2 with the 4-mirror trick for clean boundaries.

    For each seed (x,y) we add four mirror copies (-x,y), (2-x,y), (x,-y),
    (x,2-y). This forces each interior seed's Voronoi cell to terminate exactly
    on the corresponding side of the unit square (the perpendicular bisector of
    a mirror pair lies on the side itself). The result is a clean polygonal
    mesh of [0,1]^2.
    """
    if np.any(seeds <= 0.0) or np.any(seeds >= 1.0):
        raise ValueError("All seeds must lie strictly inside (0,1)^2")

    mirrors_left = np.column_stack([-seeds[:, 0], seeds[:, 1]])
    mirrors_right = np.column_stack([2.0 - seeds[:, 0], seeds[:, 1]])
    mirrors_bottom = np.column_stack([seeds[:, 0], -seeds[:, 1]])
    mirrors_top = np.column_stack([seeds[:, 0], 2.0 - seeds[:, 1]])
    all_seeds = np.vstack([seeds, mirrors_left, mirrors_right, mirrors_bottom, mirrors_top])

    vor = Voronoi(all_seeds)
    n_orig = seeds.shape[0]
    cell_vertex_lists: list[list[int]] = []
    for i in range(n_orig):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        if -1 in region or len(region) < 3:
            raise RuntimeError(f"Cell {i} has unbounded Voronoi region (mirror trick failed)")
        cell_vertex_lists.append(list(region))

    # Compact vertex set (drop unused exterior vertices)
    used_ids: set[int] = set()
    for cell in cell_vertex_lists:
        used_ids.update(cell)
    used_sorted = sorted(used_ids)
    old_to_new = {old: new for new, old in enumerate(used_sorted)}
    new_vertices = np.array([vor.vertices[v] for v in used_sorted])
    new_cells = [
        np.array([old_to_new[v] for v in cell], dtype=np.int64) for cell in cell_vertex_lists
    ]

    # Snap vertices to [0, 1]^2 (mirror-trick can leave tiny numerical creep on the boundary).
    new_vertices = np.clip(new_vertices, 0.0, 1.0)

    # Force CCW orientation for every cell.
    for i, cell in enumerate(new_cells):
        if signed_area(new_vertices[cell]) < 0.0:
            new_cells[i] = cell[::-1]

    return _build_mesh_topology(new_vertices, new_cells)


def _build_mesh_topology(vertices: np.ndarray, cells: list[np.ndarray]) -> Mesh:
    n_cells = len(cells)
    cell_centroids = np.zeros((n_cells, 2))
    cell_areas = np.zeros(n_cells)
    cell_radii = np.zeros(n_cells)
    for i, cell in enumerate(cells):
        pts = vertices[cell]
        area, centroid = polygon_area_centroid(pts)
        cell_areas[i] = area
        cell_centroids[i] = centroid
        cell_radii[i] = float(np.max(np.linalg.norm(pts - centroid, axis=1)))

    edge_dict: dict[tuple[int, int], int] = {}
    cell_to_edge_idx = [np.zeros(len(cell), dtype=np.int64) for cell in cells]
    cell_to_edge_sign = [np.zeros(len(cell), dtype=np.int8) for cell in cells]
    edge_to_cells_dict: dict[int, list[int]] = {}
    for ci, cell in enumerate(cells):
        n = len(cell)
        for li in range(n):
            v0 = int(cell[li])
            v1 = int(cell[(li + 1) % n])
            if v0 == v1:
                raise RuntimeError(f"Degenerate edge in cell {ci}")
            if v0 < v1:
                key = (v0, v1)
                sign = 1
            else:
                key = (v1, v0)
                sign = -1
            ei = edge_dict.setdefault(key, len(edge_dict))
            cell_to_edge_idx[ci][li] = ei
            cell_to_edge_sign[ci][li] = sign
            edge_to_cells_dict.setdefault(ei, []).append(ci)

    n_edges = len(edge_dict)
    unique_edges = np.zeros((n_edges, 2), dtype=np.int64)
    edge_lengths = np.zeros(n_edges)
    edge_midpoints = np.zeros((n_edges, 2))
    edge_intrinsic_normals = np.zeros((n_edges, 2))
    edge_to_cells: list[list[int]] = [[] for _ in range(n_edges)]
    for (v_lo, v_hi), ei in edge_dict.items():
        unique_edges[ei] = (v_lo, v_hi)
        a = vertices[v_lo]
        b = vertices[v_hi]
        delta = b - a
        length = float(np.linalg.norm(delta))
        if length < TOL_GEOM:
            raise RuntimeError(f"Edge {ei} between vertices {v_lo},{v_hi} has zero length")
        edge_lengths[ei] = length
        edge_midpoints[ei] = 0.5 * (a + b)
        t = delta / length
        edge_intrinsic_normals[ei] = np.array([t[1], -t[0]])
        edge_to_cells[ei] = edge_to_cells_dict[ei]

    return Mesh(
        vertices=vertices,
        cells=cells,
        cell_centroids=cell_centroids,
        cell_areas=cell_areas,
        cell_radii=cell_radii,
        unique_edges=unique_edges,
        edge_lengths=edge_lengths,
        edge_midpoints=edge_midpoints,
        edge_intrinsic_normals=edge_intrinsic_normals,
        cell_to_edge_idx=cell_to_edge_idx,
        cell_to_edge_sign=cell_to_edge_sign,
        edge_to_cells=edge_to_cells,
    )


# ---------------------------------------------------------------------------
# Harmonic basis and per-cell quadrature primitives
# ---------------------------------------------------------------------------


def harmonic_eval(k: int, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate P_k = Re(z^k), Q_k = Im(z^k) and gradients at a batch of points.

    Matches src/mimetic.cpp:eval_harmonic_basis.

    Args:
        k: harmonic order (>= 1)
        points: (M, 2)

    Returns:
        P (M,), Q (M,), gradP (M, 2), gradQ (M, 2)
    """
    z = points[:, 0] + 1j * points[:, 1]
    zk = z**k
    if k == 1:
        zk1 = np.ones_like(z)
    else:
        zk1 = z ** (k - 1)
    P = zk.real
    Q = zk.imag
    gradP = np.column_stack([k * zk1.real, -k * zk1.imag])
    gradQ = np.column_stack([k * zk1.imag, k * zk1.real])
    return P, Q, gradP, gradQ


def harmonic_eval_scaled(k: int, points: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the length-rescaled harmonic basis P~_k(p) = P_k(p/s), Q~_k(p) = Q_k(p/s)
    and physical-frame gradients (1/s)*(grad P_k)(p/s), (1/s)*(grad Q_k)(p/s).

    Without rescaling, V_{ii} ~ s^{2k} for cell length scale s, which becomes
    extremely small for refined meshes (s ~ 0.01) and produces ill-conditioned
    KKT systems. Rescaling makes V_{ii} ~ s^0 = O(1) regardless of cell size.
    """
    P, Q, gradP, gradQ = harmonic_eval(k, points / scale)
    return P, Q, gradP / scale, gradQ / scale


def _basis_from_index(i: int) -> tuple[int, bool]:
    """Map flat basis index i -> (k, is_Q). i=0,1 -> (k=1, P/Q); i=2,3 -> (k=2, P/Q); ..."""
    return (i // 2) + 1, (i % 2 == 1)


def integrate_triangle_func(a: np.ndarray, b: np.ndarray, c: np.ndarray, func: Callable[[np.ndarray], np.ndarray]) -> float:
    """7-point degree-5 rule. func: (M,2) -> (M,)."""
    twice_area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    area = 0.5 * abs(twice_area)
    pts = _TRI7_BARY @ np.stack([a, b, c])
    vals = func(pts)
    return area * float(np.sum(_TRI7_W * vals))


def integrate_edge_func(a: np.ndarray, b: np.ndarray, func: Callable[[np.ndarray], np.ndarray]) -> float:
    """4-point Gauss-Legendre on segment a->b. func: (M,2) -> (M,)."""
    length = float(np.linalg.norm(b - a))
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)
    pts = mid[None, :] + _GL4_X[:, None] * half[None, :]
    vals = func(pts)
    return 0.5 * length * float(np.sum(_GL4_W * vals))


# ---------------------------------------------------------------------------
# Per-cell reconstruction map
# ---------------------------------------------------------------------------


def per_cell_reconstruction_map(local_points: np.ndarray, area: float) -> tuple[np.ndarray, float]:
    """Compute the linear map R: edge_fluxes (N,) -> coeffs (1 + N_h,) and the basis scale s.

    coeffs = [d, a~_1, b~_1, a~_2, b~_2, ...] where d is the reconstruction
    divergence and (a~_k, b~_k) are coefficients in the LENGTH-RESCALED harmonic
    basis: psi~_k(p) = psi_k(p/s), with s the cell length scale (max distance
    from centroid to vertex). The same s is needed when integrating u_h on a
    subsegment via subsegment_normal_integral_row.

    edge_fluxes are CELL-LOCAL signed (positive = outward from this cell).

    Mirrors src/mimetic.cpp:reconstruct_source_polygon, with d removed
    analytically (d = sum(U)/area from the discrete divergence theorem) and
    with length-rescaling added for robustness on refined Voronoi meshes.
    """
    N = local_points.shape[0]
    K_max = N // 2
    N_h = 2 * K_max

    edge_a = local_points
    edge_b = np.roll(local_points, -1, axis=0)
    deltas = edge_b - edge_a
    edge_lengths_local = np.linalg.norm(deltas, axis=1)
    edge_normals_local = np.column_stack([deltas[:, 1], -deltas[:, 0]]) / edge_lengths_local[:, None]

    # Length scale: max distance from centroid (origin in local frame) to any
    # vertex, floored by sqrt(area). Matches src/mimetic.cpp:local_length_scale.
    scale = max(float(np.max(np.linalg.norm(local_points, axis=1))), math.sqrt(max(area, TOL_GEOM)))
    scale = max(scale, 1.0e-12)

    if N_h == 0:
        # Cell with N=2 edges (degenerate). Only constant divergence.
        return (1.0 / area) * np.ones((1, N)), scale

    origin = np.zeros(2)

    # ---- V (Gram matrix of basis gradients) and pieces of M.
    V = np.zeros((N_h, N_h))
    cell_basis_int = np.zeros(N_h)
    div_int = np.zeros(N_h)

    # Pre-tabulate basis at 7 quadrature points of every fan triangle.
    # Layout: tri_pts shape (N, 7, 2), tri_weights shape (N,) (area of each triangle).
    tri_pts_list = []
    tri_areas = np.zeros(N)
    for e in range(N):
        twice_area = (edge_a[e, 0] - origin[0]) * (edge_b[e, 1] - origin[1]) - (
            edge_a[e, 1] - origin[1]
        ) * (edge_b[e, 0] - origin[0])
        tri_areas[e] = 0.5 * abs(twice_area)
        pts = _TRI7_BARY @ np.stack([origin, edge_a[e], edge_b[e]])  # (7, 2)
        tri_pts_list.append(pts)
    tri_pts = np.stack(tri_pts_list)  # (N, 7, 2)
    tri_pts_flat = tri_pts.reshape(-1, 2)  # (7N, 2)

    # Evaluate every (length-rescaled) basis function at every quadrature point.
    # basis_P[k_idx], basis_Q[k_idx] of shape (7N,)
    # gradP[k_idx], gradQ[k_idx] of shape (7N, 2)
    basis_P = []
    basis_Q = []
    gradP_all = []
    gradQ_all = []
    for k_idx in range(K_max):
        P, Q, gP, gQ = harmonic_eval_scaled(k_idx + 1, tri_pts_flat, scale)
        basis_P.append(P)
        basis_Q.append(Q)
        gradP_all.append(gP)
        gradQ_all.append(gQ)

    # Build "Psi" (function value, shape (N_h, 7N)) and "gradPsi" (shape (N_h, 7N, 2)).
    Psi = np.zeros((N_h, tri_pts_flat.shape[0]))
    gradPsi = np.zeros((N_h, tri_pts_flat.shape[0], 2))
    for i in range(N_h):
        k_idx = i // 2
        is_Q = i % 2 == 1
        if is_Q:
            Psi[i] = basis_Q[k_idx]
            gradPsi[i] = gradQ_all[k_idx]
        else:
            Psi[i] = basis_P[k_idx]
            gradPsi[i] = gradP_all[k_idx]

    # Integration weights per quadrature point: tri_areas (N,) broadcast with _TRI7_W (7,).
    qweights = (tri_areas[:, None] * _TRI7_W[None, :]).reshape(-1)  # (7N,)

    # V[i,j] = sum_p qweights[p] * (gradPsi[i,p] . gradPsi[j,p])
    V = np.einsum("p,ipd,jpd->ij", qweights, gradPsi, gradPsi)

    # cell_basis_int[i] = sum_p qweights[p] * Psi[i, p]
    cell_basis_int = np.einsum("p,ip->i", qweights, Psi)

    # div_int[i] = sum_p qweights[p] * (p_q . gradPsi[i, p])
    p_dot_grad = np.einsum("pd,ipd->ip", tri_pts_flat, gradPsi)
    div_int = np.einsum("p,ip->i", qweights, p_dot_grad)

    # Diagonal stabilization for higher modes (i >= 4 means k >= 3).
    # Matches src/mimetic.cpp:3480-3483.
    for i in range(4, N_h):
        v_diag = max(V[i, i], TOL_GEOM * area)
        V[i, i] += 10.0 * v_diag

    cell_basis_avg = cell_basis_int / area  # (N_h,)
    div_term = 0.5 * div_int / area  # (N_h,)

    # ---- Edge integrals: edge_avg_Psi[i, e] = mean of Psi_i along edge e.
    # 4-point GL on each edge.
    edge_pts = np.zeros((N, 4, 2))
    for e in range(N):
        mid = 0.5 * (edge_a[e] + edge_b[e])
        half = 0.5 * (edge_b[e] - edge_a[e])
        edge_pts[e] = mid[None, :] + _GL4_X[:, None] * half[None, :]
    edge_pts_flat = edge_pts.reshape(-1, 2)  # (4N, 2)

    Psi_edge = np.zeros((N_h, edge_pts_flat.shape[0]))
    gradPsi_edge = np.zeros((N_h, edge_pts_flat.shape[0], 2))
    for k_idx in range(K_max):
        P, Q, gP, gQ = harmonic_eval_scaled(k_idx + 1, edge_pts_flat, scale)
        for is_Q, Pi, gPi in [(False, P, gP), (True, Q, gQ)]:
            i = 2 * k_idx + (1 if is_Q else 0)
            Psi_edge[i] = Pi
            gradPsi_edge[i] = gPi

    # edge_avg_Psi[i, e] = (1 / edge_length[e]) * sum_q (length[e]/2) * w_q * Psi[q]
    #                   = 0.5 * sum_q w_q * Psi[q]   (independent of length)
    Psi_edge_3d = Psi_edge.reshape(N_h, N, 4)  # (N_h, N, 4)
    edge_avg_Psi = 0.5 * np.einsum("q,ieq->ie", _GL4_W, Psi_edge_3d)

    # M_coef[i, e] = edge_avg_Psi[i, e] - cell_basis_avg[i] - div_term[i]
    M_coef = edge_avg_Psi - cell_basis_avg[:, None] - div_term[:, None]  # (N_h, N)

    # ---- Constraint matrix C (N-1, N_h) and divergence-correction E_vec (N-1).
    C = np.zeros((N - 1, N_h))
    E_vec = np.zeros(N - 1)
    for e in range(N - 1):
        normal_e = edge_normals_local[e]
        length_e = edge_lengths_local[e]
        # gradPsi_edge for edge e are slots 4*e..4*e+4 in flat indexing.
        sl = slice(4 * e, 4 * e + 4)
        # C[e, i] = int_{edge e} gradPsi_i . n_e ds = 0.5 * length_e * sum_q w_q * (gradPsi . n_e)
        gn = np.einsum("iqd,d->iq", gradPsi_edge[:, sl, :], normal_e)  # (N_h, 4)
        C[e, :] = 0.5 * length_e * np.einsum("q,iq->i", _GL4_W, gn)
        # E_vec[e] = int_{edge e} p . n_e ds
        pn = edge_pts[e] @ normal_e  # (4,)
        E_vec[e] = 0.5 * length_e * float(np.sum(_GL4_W * pn))

    # ---- F_coef as a function of U: F = U[:N-1] - 0.5 * d * E_vec
    # Where d = (1 / area) * sum(U).
    F_coef = np.zeros((N - 1, N))
    for e in range(N - 1):
        F_coef[e, e] = 1.0
    F_coef -= (0.5 / area) * E_vec[:, None] * np.ones((1, N))

    # ---- KKT system, solved against the N-column identity RHS.
    KKT = np.zeros((N_h + N - 1, N_h + N - 1))
    KKT[:N_h, :N_h] = V
    KKT[N_h:, :N_h] = C
    KKT[:N_h, N_h:] = C.T
    RHS = np.vstack([M_coef, F_coef])  # (N_h + N - 1, N)

    sol = np.linalg.solve(KKT, RHS)
    X = sol[:N_h, :]  # (N_h, N)

    d_coef = (1.0 / area) * np.ones((1, N))
    R = np.vstack([d_coef, X])  # (1 + N_h, N)
    return R, scale


# ---------------------------------------------------------------------------
# Subsegment integral coefficients (linear in source coeffs)
# ---------------------------------------------------------------------------


def subsegment_normal_integral_row(
    a_local: np.ndarray, b_local: np.ndarray, n_t: np.ndarray, N_h: int, scale: float
) -> np.ndarray:
    """Row r in R^{1+N_h} such that

        int_{a->b in source local frame} u_h(p; coeffs) . n_t ds = r . coeffs

    where u_h(p) = (d/2) p + sum_k [a~_k gradP~_k(p) + b~_k gradQ~_k(p)],
    psi~_k(p) = psi_k(p/s), and coeffs = [d, a~_1, b~_1, ...].

    `scale` must equal the scale returned by per_cell_reconstruction_map for
    the source cell whose coeffs are passed in (otherwise the basis evaluation
    here disagrees with the basis used to derive the coefficients).

    n_t is the target edge intrinsic normal (a 2-vector, unit length); since
    the source local frame differs from the absolute frame only by translation,
    n_t can be passed in unchanged.
    """
    K_max = N_h // 2
    length = float(np.linalg.norm(b_local - a_local))
    if length < TOL_GEOM:
        return np.zeros(1 + N_h)
    mid = 0.5 * (a_local + b_local)
    half = 0.5 * (b_local - a_local)
    pts = mid[None, :] + _GL4_X[:, None] * half[None, :]  # (4, 2)
    weights = 0.5 * length * _GL4_W  # (4,)

    r = np.zeros(1 + N_h)
    # Coefficient of d: 0.5 * sum_q w_q * (p_q . n_t)
    p_dot_n = pts @ n_t  # (4,)
    r[0] = 0.5 * float(np.sum(weights * p_dot_n))

    if K_max > 0:
        for k_idx in range(K_max):
            k = k_idx + 1
            _, _, gradP, gradQ = harmonic_eval_scaled(k, pts, scale)
            r[1 + 2 * k_idx] = float(np.sum(weights * (gradP @ n_t)))
            r[1 + 2 * k_idx + 1] = float(np.sum(weights * (gradQ @ n_t)))
    return r


# ---------------------------------------------------------------------------
# Sutherland-Hodgman segment-vs-convex-polygon clipping
# ---------------------------------------------------------------------------


def clip_segment_to_convex_polygon(
    a: np.ndarray, b: np.ndarray, polygon: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Clip segment [a, b] against convex CCW polygon. Returns (a', b') or None."""
    pa = a.astype(np.float64).copy()
    pb = b.astype(np.float64).copy()
    n = polygon.shape[0]
    for i in range(n):
        p0 = polygon[i]
        p1 = polygon[(i + 1) % n]
        edge = p1 - p0
        # Inward normal for CCW = left rotation of edge.
        n_in = np.array([-edge[1], edge[0]])
        s_a = float(np.dot(pa - p0, n_in))
        s_b = float(np.dot(pb - p0, n_in))
        if s_a < -TOL_CLIP and s_b < -TOL_CLIP:
            return None
        if s_a >= -TOL_CLIP and s_b >= -TOL_CLIP:
            continue
        if s_a >= -TOL_CLIP:
            t = s_a / (s_a - s_b)
            pb = pa + t * (pb - pa)
        else:
            t = s_a / (s_a - s_b)
            pa = pa + t * (pb - pa)
        if float(np.linalg.norm(pb - pa)) < TOL_CLIP:
            return None
    return pa, pb


def point_in_convex_polygon(p: np.ndarray, polygon: np.ndarray, tol: float = 1.0e-12) -> bool:
    n = polygon.shape[0]
    for i in range(n):
        p0 = polygon[i]
        p1 = polygon[(i + 1) % n]
        edge = p1 - p0
        n_in = np.array([-edge[1], edge[0]])
        if float(np.dot(p - p0, n_in)) < -tol:
            return False
    return True


def subsegment_interior_probe_inside(
    a: np.ndarray, b: np.ndarray, polygon: np.ndarray, domain_xmin: float = 0.0, domain_xmax: float = 1.0, domain_ymin: float = 0.0, domain_ymax: float = 1.0
) -> bool:
    """Tie-breaking interior probe: accept this source polygon iff a small
    nudge perpendicular to the segment (right-rotation of tangent) lands
    strictly inside the polygon -- OR the right-perp nudge has left the unit
    square (in which case the polygon must be the unique source cell touching
    the segment from the OTHER side, and we accept on the left-perp probe).

    Without tie-breaking, target edges that coincide with shared source-cell
    boundaries (which happens routinely with Halton seeds at high resolution
    because both meshes share base-2 fraction structure) get double-counted.

    The fallback to left-perp keeps boundary-aligned target edges working: on
    e.g. the bottom of the unit square, right-perp goes south outside [0,1]^2
    so we fall through to the left-perp probe which lands in the boundary
    source cell to the north.
    """
    delta = b - a
    length = float(np.linalg.norm(delta))
    if length < TOL_CLIP:
        return False
    n_perp = np.array([delta[1], -delta[0]]) / length
    eps = 1.0e-7 * length
    mid = 0.5 * (a + b)
    probe_right = mid + eps * n_perp
    if point_in_convex_polygon(probe_right, polygon, tol=0.0):
        return True
    out_of_domain = (
        probe_right[0] < domain_xmin - 1.0e-12
        or probe_right[0] > domain_xmax + 1.0e-12
        or probe_right[1] < domain_ymin - 1.0e-12
        or probe_right[1] > domain_ymax + 1.0e-12
    )
    if out_of_domain:
        probe_left = mid - eps * n_perp
        if point_in_convex_polygon(probe_left, polygon, tol=0.0):
            return True
    return False


# ---------------------------------------------------------------------------
# Sparse projection-operator assembly
# ---------------------------------------------------------------------------


def precompute_reconstructions(mesh: Mesh) -> tuple[list[np.ndarray], np.ndarray]:
    """Per-cell reconstruction maps R and per-cell length scales.

    Returns:
        R_list: per-cell map from cell-local edge fluxes to scaled-basis coeffs.
        scales: per-cell length scale s such that the basis is psi_k(p/s).
    """
    R_list = []
    scales = np.zeros(len(mesh.cells))
    for ci in range(len(mesh.cells)):
        local_pts = mesh.vertices[mesh.cells[ci]] - mesh.cell_centroids[ci]
        R, s = per_cell_reconstruction_map(local_pts, float(mesh.cell_areas[ci]))
        R_list.append(R)
        scales[ci] = s
    return R_list, scales


def build_projection_operator(
    src: Mesh, tgt: Mesh, src_R_list: list[np.ndarray], src_scales: np.ndarray
) -> sp.csr_matrix:
    """Sparse W mapping source unique-edge fluxes to target unique-edge fluxes."""
    n_tgt = tgt.unique_edges.shape[0]
    n_src = src.unique_edges.shape[0]

    src_tree = cKDTree(src.cell_centroids)
    max_src_radius = float(src.cell_radii.max())

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for et in range(n_tgt):
        v0, v1 = tgt.unique_edges[et]
        a_t = tgt.vertices[v0]
        b_t = tgt.vertices[v1]
        n_t = tgt.edge_intrinsic_normals[et]
        edge_len = tgt.edge_lengths[et]
        mid_t = tgt.edge_midpoints[et]

        search_r = 0.5 * edge_len + max_src_radius + 1.0e-10
        candidates = src_tree.query_ball_point(mid_t, r=search_r)

        for cs in candidates:
            cell_v = src.cells[cs]
            src_poly = src.vertices[cell_v]
            clip = clip_segment_to_convex_polygon(a_t, b_t, src_poly)
            if clip is None:
                continue
            a_clip, b_clip = clip
            sub_len = float(np.linalg.norm(b_clip - a_clip))
            if sub_len < TOL_CLIP:
                continue
            # Interior probe: perpendicular nudge breaks ties on shared boundaries.
            if not subsegment_interior_probe_inside(a_clip, b_clip, src_poly):
                continue

            src_centroid = src.cell_centroids[cs]
            a_local = a_clip - src_centroid
            b_local = b_clip - src_centroid

            R = src_R_list[cs]  # (1 + N_h, Ns)
            N_h = R.shape[0] - 1
            r_row = subsegment_normal_integral_row(a_local, b_local, n_t, N_h, float(src_scales[cs]))
            w = r_row @ R  # (Ns,)

            cell_signs = src.cell_to_edge_sign[cs]
            cell_geid = src.cell_to_edge_idx[cs]
            for local_e in range(len(cell_geid)):
                contribution = float(w[local_e]) * float(cell_signs[local_e])
                if abs(contribution) > 1.0e-18:
                    rows.append(et)
                    cols.append(int(cell_geid[local_e]))
                    data.append(contribution)

    W = sp.coo_matrix((data, (rows, cols)), shape=(n_tgt, n_src))
    W = W.tocsr()
    W.sum_duplicates()
    W.eliminate_zeros()
    return W


def build_target_divergence_rhs_operator(src: Mesh, tgt: Mesh) -> sp.csr_matrix:
    """Build sparse operator S of shape (n_t_cells, n_s_edges) such that

        (S @ src_fluxes)[K_t]  =  sum over source cells K_s of d_s * |K_s cap K_t|

    where d_s = (sum_e sigma_K_s(e) * src_flux[e_global]) / |K_s| is the source
    cell discrete divergence. This RHS is the source-determined target-cell
    discrete-divergence integral that the H(div)-conforming projection enforces:

        sum_{e in partial K_t} sigma_K_t(e) * Phi'_e  =  (S @ src_fluxes)[K_t].
    """
    n_t_cells = len(tgt.cells)
    n_s_edges = src.unique_edges.shape[0]
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    src_tree = cKDTree(src.cell_centroids)
    max_src_radius = float(src.cell_radii.max())

    for ct in range(n_t_cells):
        tgt_poly = tgt.vertices[tgt.cells[ct]]
        ct_center = tgt.cell_centroids[ct]
        ct_radius = float(tgt.cell_radii[ct])
        candidates = src_tree.query_ball_point(ct_center, r=ct_radius + max_src_radius + 1.0e-10)
        for cs in candidates:
            src_poly = src.vertices[src.cells[cs]]
            overlap = _polygon_polygon_clip(src_poly, tgt_poly)
            if overlap is None or overlap.shape[0] < 3:
                continue
            overlap_area = abs(signed_area(overlap))
            if overlap_area < 1.0e-14:
                continue
            # d_s from source cell K_s = (sum_e sigma_K_s(e) * src_flux[e]) / |K_s|
            # Contribution to (S @ src_fluxes)[ct] is overlap_area * d_s.
            # = (overlap_area / |K_s|) * sum_e sigma_K_s(e) * src_flux[e]
            scale = overlap_area / float(src.cell_areas[cs])
            cell_geid = src.cell_to_edge_idx[cs]
            cell_signs = src.cell_to_edge_sign[cs]
            for local_e in range(len(cell_geid)):
                rows.append(ct)
                cols.append(int(cell_geid[local_e]))
                data.append(scale * float(cell_signs[local_e]))

    S = sp.coo_matrix((data, (rows, cols)), shape=(n_t_cells, n_s_edges))
    S = S.tocsr()
    S.sum_duplicates()
    return S


def build_target_cell_edge_signed_incidence(tgt: Mesh) -> sp.csr_matrix:
    """Sparse signed cell-edge incidence A on the target mesh:

        A[K_t, e]  =  sigma_K_t(e)  if e is in partial K_t  else  0.

    Then (A @ Phi)[K_t] = sum_{e in partial K_t} sigma_K_t(e) * Phi_e is the
    discrete cell divergence integral (without the |K_t| factor).
    """
    n_t_cells = len(tgt.cells)
    n_t_edges = tgt.unique_edges.shape[0]
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for ct in range(n_t_cells):
        cell_geid = tgt.cell_to_edge_idx[ct]
        cell_signs = tgt.cell_to_edge_sign[ct]
        for local_e in range(len(cell_geid)):
            rows.append(ct)
            cols.append(int(cell_geid[local_e]))
            data.append(float(cell_signs[local_e]))
    A = sp.coo_matrix((data, (rows, cols)), shape=(n_t_cells, n_t_edges))
    A = A.tocsr()
    A.sum_duplicates()
    return A


class HdivProjector:
    """Apply the H(div)-conforming post-projection to raw W-transferred fluxes.

    The raw remap W is per-overlap conservative: for any overlap O = K_s cap K_t,
    the boundary flux of u_h^{K_s} through partial O equals d_s |O| exactly. The
    H(div)-conforming projection enforces the *per-target-cell* identity

        sum_{e in partial K_t} sigma_K_t(e) Phi'_e  =  sum_{K_s} d_s |K_s cap K_t|

    by minimising ||Phi' - Phi_W||^2 subject to that linear constraint. The
    resulting Phi' is a single signed scalar per unique target edge whose sign
    convention is the intrinsic-normal one already used everywhere in this code.
    """

    def __init__(self, src: Mesh, tgt: Mesh, W: sp.csr_matrix, regularize: float = 1.0e-14) -> None:
        self.A = build_target_cell_edge_signed_incidence(tgt)
        self.S = build_target_divergence_rhs_operator(src, tgt)
        self.W = W
        # KKT-style normal-equation matrix M = A A^T (sparse, sym PSD).
        # For closed-domain meshes M is rank-deficient by one (the global
        # divergence-theorem constraint sums to zero), so we add a tiny diagonal
        # regulariser to make the solver unambiguous.
        n_c = self.A.shape[0]
        M = (self.A @ self.A.T).tocsc() + regularize * sp.eye(n_c, format="csc")
        from scipy.sparse.linalg import splu
        self._solver = splu(M)

    def apply(self, src_fluxes: np.ndarray) -> np.ndarray:
        """Project the raw transfer (W @ src_fluxes) onto the H(div)-conforming subspace."""
        phi_raw = self.W @ src_fluxes
        b_target = self.S @ src_fluxes
        residual = self.A @ phi_raw - b_target
        lam = self._solver.solve(residual)
        return phi_raw - self.A.T @ lam

    def build_matrix(self, threshold: float = 0.0) -> sp.csr_matrix:
        """Return W_hdiv = W - A^T (A A^T + eps I)^{-1} (A W - S) as a sparse matrix.

        The pseudo-inverse densifies the operator, so each row of W_hdiv
        typically has many more nonzeros than the corresponding row of W.
        Pass `threshold > 0` to drop weights with absolute value below the
        threshold (useful for SCRIP storage).
        """
        from scipy.sparse.linalg import splu  # noqa: F401  (already imported by __init__)
        AW = self.A @ self.W
        K = (AW - self.S).toarray()  # (n_t_cells, n_s_edges) -- dense intermediate
        Lambda = self._solver.solve(K)  # (n_t_cells, n_s_edges)
        correction = self.A.T @ Lambda  # dense
        W_dense = self.W.toarray() - correction
        if threshold > 0.0:
            W_dense[np.abs(W_dense) < threshold] = 0.0
        W_hdiv = sp.csr_matrix(W_dense)
        W_hdiv.eliminate_zeros()
        return W_hdiv


def build_hdiv_projection(W_raw: sp.csr_matrix, src: Mesh, tgt: Mesh) -> HdivProjector:
    """Construct an HdivProjector wrapping the per-overlap-conservative remap W_raw.

    Use ``HdivProjector.apply`` to obtain H(div)-conforming target fluxes for any
    given source-flux vector, or ``HdivProjector.build_matrix`` to obtain the
    composed sparse operator.
    """
    return HdivProjector(src, tgt, W_raw)


# ---------------------------------------------------------------------------
# Analytical fields
# ---------------------------------------------------------------------------


def field_div_free(p: np.ndarray) -> np.ndarray:
    """u = (sin pi x cos pi y, -cos pi x sin pi y). div u = 0."""
    x = p[..., 0]
    y = p[..., 1]
    return np.stack(
        [np.sin(np.pi * x) * np.cos(np.pi * y), -np.cos(np.pi * x) * np.sin(np.pi * y)],
        axis=-1,
    )


def field_smooth_div(p: np.ndarray) -> np.ndarray:
    """u = (sin pi x, sin pi y). div u = pi (cos pi x + cos pi y)."""
    x = p[..., 0]
    y = p[..., 1]
    return np.stack([np.sin(np.pi * x), np.sin(np.pi * y)], axis=-1)


def field_harmonic_linear(p: np.ndarray) -> np.ndarray:
    """u = (1 + 2x + 2y, 1 + 2x - 2y). Linear, harmonic, div = 0, in level-2 K=2 space.

    Exactly recoverable on cells with N=5 edges (uniquely determined system) or
    on regular polygons of any N (recovered by symmetry of the constraint
    system). On irregular polygons with N=4 or N>=6 the level-2 saddle-point
    system is rank-deficient or under-/over-determined and the energy-minimizing
    coefficients differ from the field's coefficients; the resulting transfer
    error still converges as O(h) with mesh refinement.
    """
    x = p[..., 0]
    y = p[..., 1]
    return np.stack([1.0 + 2.0 * x + 2.0 * y, 1.0 + 2.0 * x - 2.0 * y], axis=-1)


def field_constant(p: np.ndarray) -> np.ndarray:
    """u = (1, 1). Pure constant: a_1 = b_1 = 1, all other coefficients zero.

    Exactly recoverable on every cell (the constant modes are always in the
    constraint system's range), used as the strict exactness sanity check.
    """
    out = np.empty_like(p)
    out[..., 0] = 1.0
    out[..., 1] = 1.0
    return out


def exact_edge_flux(field: Callable[[np.ndarray], np.ndarray], a: np.ndarray, b: np.ndarray) -> float:
    """int_a^b field . n_intrinsic ds with 16-point Gauss-Legendre."""
    delta = b - a
    length = float(np.linalg.norm(delta))
    n = np.array([delta[1], -delta[0]]) / length  # right rotation = intrinsic normal
    mid = 0.5 * (a + b)
    half = 0.5 * delta
    pts = mid[None, :] + _GL16_X[:, None] * half[None, :]
    vals = field(pts)
    integrand = vals @ n
    return 0.5 * length * float(np.sum(_GL16_W * integrand))


def compute_edge_fluxes(field: Callable[[np.ndarray], np.ndarray], mesh: Mesh) -> np.ndarray:
    """Stored edge fluxes (intrinsic-normal direction) for every unique edge."""
    n_edges = mesh.unique_edges.shape[0]
    out = np.zeros(n_edges)
    for i in range(n_edges):
        v0, v1 = mesh.unique_edges[i]
        out[i] = exact_edge_flux(field, mesh.vertices[v0], mesh.vertices[v1])
    return out


# ---------------------------------------------------------------------------
# SCRIP file writer (edges-as-grid layout)
# ---------------------------------------------------------------------------


def write_scrip_edges_as_grid(
    path: Path,
    src_mesh: Mesh,
    tgt_mesh: Mesh,
    W: sp.csr_matrix,
    title: str,
) -> None:
    """Write SCRIP NetCDF file with each unique edge serving as a 'grid cell'.

    Center = edge midpoint. Corners = edge endpoints (2 corners). Area = edge length.
    Weights are scalar per (src_edge, dst_edge) link, intrinsic-normal flux convention.
    """
    W_coo = W.tocoo()
    n_links = int(W_coo.nnz)

    n_src = src_mesh.unique_edges.shape[0]
    n_dst = tgt_mesh.unique_edges.shape[0]

    with netCDF4.Dataset(path, "w", format="NETCDF3_CLASSIC") as nc:
        nc.createDimension("src_grid_size", n_src)
        nc.createDimension("dst_grid_size", n_dst)
        nc.createDimension("src_grid_corners", 2)
        nc.createDimension("dst_grid_corners", 2)
        nc.createDimension("src_grid_rank", 1)
        nc.createDimension("dst_grid_rank", 1)
        nc.createDimension("num_links", n_links)
        nc.createDimension("num_wgts", 1)

        v = nc.createVariable("src_grid_dims", "i4", ("src_grid_rank",))
        v[:] = np.array([n_src], dtype=np.int32)
        v = nc.createVariable("dst_grid_dims", "i4", ("dst_grid_rank",))
        v[:] = np.array([n_dst], dtype=np.int32)

        # Sources.
        src_v0 = src_mesh.unique_edges[:, 0]
        src_v1 = src_mesh.unique_edges[:, 1]
        v = nc.createVariable("src_grid_center_lat", "f8", ("src_grid_size",))
        v.units = "none"
        v[:] = src_mesh.edge_midpoints[:, 1]
        v = nc.createVariable("src_grid_center_lon", "f8", ("src_grid_size",))
        v.units = "none"
        v[:] = src_mesh.edge_midpoints[:, 0]
        v = nc.createVariable("src_grid_corner_lat", "f8", ("src_grid_size", "src_grid_corners"))
        v.units = "none"
        v[:, 0] = src_mesh.vertices[src_v0, 1]
        v[:, 1] = src_mesh.vertices[src_v1, 1]
        v = nc.createVariable("src_grid_corner_lon", "f8", ("src_grid_size", "src_grid_corners"))
        v.units = "none"
        v[:, 0] = src_mesh.vertices[src_v0, 0]
        v[:, 1] = src_mesh.vertices[src_v1, 0]
        v = nc.createVariable("src_grid_imask", "i4", ("src_grid_size",))
        v[:] = np.ones(n_src, dtype=np.int32)
        v = nc.createVariable("src_grid_area", "f8", ("src_grid_size",))
        v.units = "none"
        v[:] = src_mesh.edge_lengths

        # Destinations.
        dst_v0 = tgt_mesh.unique_edges[:, 0]
        dst_v1 = tgt_mesh.unique_edges[:, 1]
        v = nc.createVariable("dst_grid_center_lat", "f8", ("dst_grid_size",))
        v.units = "none"
        v[:] = tgt_mesh.edge_midpoints[:, 1]
        v = nc.createVariable("dst_grid_center_lon", "f8", ("dst_grid_size",))
        v.units = "none"
        v[:] = tgt_mesh.edge_midpoints[:, 0]
        v = nc.createVariable("dst_grid_corner_lat", "f8", ("dst_grid_size", "dst_grid_corners"))
        v.units = "none"
        v[:, 0] = tgt_mesh.vertices[dst_v0, 1]
        v[:, 1] = tgt_mesh.vertices[dst_v1, 1]
        v = nc.createVariable("dst_grid_corner_lon", "f8", ("dst_grid_size", "dst_grid_corners"))
        v.units = "none"
        v[:, 0] = tgt_mesh.vertices[dst_v0, 0]
        v[:, 1] = tgt_mesh.vertices[dst_v1, 0]
        v = nc.createVariable("dst_grid_imask", "i4", ("dst_grid_size",))
        v[:] = np.ones(n_dst, dtype=np.int32)
        v = nc.createVariable("dst_grid_area", "f8", ("dst_grid_size",))
        v.units = "none"
        v[:] = tgt_mesh.edge_lengths

        # Mapping (1-based as per SCRIP convention).
        v = nc.createVariable("src_address", "i4", ("num_links",))
        v[:] = (W_coo.col + 1).astype(np.int32)
        v = nc.createVariable("dst_address", "i4", ("num_links",))
        v[:] = (W_coo.row + 1).astype(np.int32)
        v = nc.createVariable("remap_matrix", "f8", ("num_links", "num_wgts"))
        v[:, 0] = W_coo.data

        # Global attributes.
        nc.title = title
        nc.conventions = "SCRIP"
        nc.map_method = "mimetic_p0_edge_flux"
        nc.normalization = "none"
        nc.source_grid = "voronoi_unit_square"
        nc.dest_grid = "voronoi_unit_square"
        nc.note = (
            "Edges-as-grid layout: each grid 'cell' is a unique mesh edge; corners are "
            "edge endpoints; area is edge length. Sign convention: weight maps source "
            "edge flux (in source-edge intrinsic normal direction, defined as the "
            "right-rotation of the v_low->v_high tangent) to target edge flux (target-"
            "edge intrinsic normal direction, same convention)."
        )


# ---------------------------------------------------------------------------
# Tests and convergence driver
# ---------------------------------------------------------------------------


def check_mesh_sanity(mesh: Mesh, name: str) -> None:
    """Basic mesh invariants: areas sum to 1, every edge is interior or boundary, all CCW."""
    total_area = float(np.sum(mesh.cell_areas))
    if not math.isclose(total_area, 1.0, rel_tol=1.0e-10, abs_tol=1.0e-10):
        raise RuntimeError(f"[{name}] cell areas sum to {total_area} != 1")
    for ei, owners in enumerate(mesh.edge_to_cells):
        if len(owners) not in (1, 2):
            raise RuntimeError(f"[{name}] edge {ei} has {len(owners)} owning cells")
    for ci, cell in enumerate(mesh.cells):
        if signed_area(mesh.vertices[cell]) <= 0.0:
            raise RuntimeError(f"[{name}] cell {ci} is not CCW")
    print(
        f"[OK] [{name}] {len(mesh.cells)} cells, {mesh.unique_edges.shape[0]} unique edges, total area = {total_area:.16f}"
    )


def check_constant_exactness(src: Mesh, src_R_list: list[np.ndarray], tgt: Mesh, W: sp.csr_matrix) -> None:
    """Round-trip exact reproduction of u = (1, 1) through W. Strict exactness check.

    This is the strongest exactness statement that holds for every Voronoi mesh:
    constants are in the basis (k=1 modes) and recoverable on every cell with
    nonzero area, regardless of edge count or symmetry.
    """
    src_fluxes = compute_edge_fluxes(field_constant, src)
    tgt_fluxes_exact = compute_edge_fluxes(field_constant, tgt)
    tgt_fluxes_computed = W @ src_fluxes
    err = tgt_fluxes_computed - tgt_fluxes_exact
    linf = float(np.max(np.abs(err)))
    if linf > TOL_EXACTNESS:
        raise RuntimeError(
            f"[FAIL] constant exactness: Linf error = {linf:.3e} > {TOL_EXACTNESS:.0e}"
        )
    print(f"[OK] constant field u=(1,1) reproduces exactly through W: Linf error = {linf:.3e}")


def check_harmonic_recovery_per_cell(src: Mesh, src_R_list: list[np.ndarray], src_scales: np.ndarray) -> None:
    """Diagnostic: per-cell reconstruction error for the linear harmonic field.

    The level-2 method recovers this field exactly only on cells where the edge-
    flux constraint system uniquely determines all four (a_1, b_1, a_2, b_2)
    modes - in practice cells with N=5 edges, regular hexagons/octagons, or
    other special-symmetry shapes. We report the worst-cell error in
    DIMENSIONAL terms (max over a_k, b_k of |a_k - a_k_true|), with the scaled-
    basis coeffs converted back to physical-basis units via a_k = a~_k / s^k.
    """
    src_fluxes = compute_edge_fluxes(field_harmonic_linear, src)
    worst_err = 0.0
    worst_n = 0
    for ci in range(len(src.cells)):
        cell_geid = src.cell_to_edge_idx[ci]
        cell_signs = src.cell_to_edge_sign[ci]
        cell_local_U = np.array(
            [src_fluxes[gi] * float(s) for gi, s in zip(cell_geid, cell_signs)]
        )
        coeffs_scaled = src_R_list[ci] @ cell_local_U
        s = float(src_scales[ci])
        # Convert scaled coefficients back to physical basis: a_k = a~_k / s^k.
        coeffs_phys = coeffs_scaled.copy()
        n_h = len(coeffs_scaled) - 1
        for i in range(n_h):
            k = (i // 2) + 1
            coeffs_phys[1 + i] = coeffs_scaled[1 + i] / (s**k)
        cx, cy = src.cell_centroids[ci]
        expected = np.zeros_like(coeffs_phys)
        expected[1] = 1.0 + 2.0 * cx + 2.0 * cy
        expected[2] = 1.0 + 2.0 * cx - 2.0 * cy
        if len(expected) > 3:
            expected[3] = 1.0
        if len(expected) > 4:
            expected[4] = 1.0
        err = float(np.max(np.abs(coeffs_phys - expected)))
        if err > worst_err:
            worst_err = err
            worst_n = len(cell_geid)
    print(
        f"[INFO] harmonic field per-cell coeff recovery (physical basis): worst max-coeff err = {worst_err:.3e} "
        f"(on a cell with N={worst_n} edges)"
    )


def check_per_overlap_conservation(
    src: Mesh,
    src_R_list: list[np.ndarray],
    src_scales: np.ndarray,
    tgt: Mesh,
    src_fluxes: np.ndarray,
    n_samples: int = 50,
    rng_seed: int = 0,
) -> None:
    """For random (K_s, K_t) overlapping pairs, verify divergence theorem on the overlap."""
    rng = np.random.default_rng(rng_seed)
    src_tree = cKDTree(src.cell_centroids)
    tested = 0
    max_err = 0.0
    indices = rng.choice(len(tgt.cells), size=min(n_samples, len(tgt.cells)), replace=False)
    for ci_t in indices:
        ct = tgt.cell_centroids[ci_t]
        rt = float(tgt.cell_radii[ci_t])
        rs_max = float(src.cell_radii.max())
        cs_candidates = src_tree.query_ball_point(ct, r=rt + rs_max + 1.0e-10)
        for cs in cs_candidates:
            R = src_R_list[cs]
            cell_signs = src.cell_to_edge_sign[cs]
            cell_geid = src.cell_to_edge_idx[cs]
            cell_local_U = np.array(
                [src_fluxes[gi] * float(s) for gi, s in zip(cell_geid, cell_signs)]
            )
            coeffs = R @ cell_local_U  # (1 + N_h,)
            d_s = coeffs[0]
            N_h = R.shape[0] - 1

            # Compute the polygon overlap K_s ∩ K_t by Sutherland-Hodgman polygon-polygon clip.
            overlap = _polygon_polygon_clip(src.vertices[src.cells[cs]], tgt.vertices[tgt.cells[ci_t]])
            if overlap is None or overlap.shape[0] < 3:
                continue
            overlap_area = abs(signed_area(overlap))
            if overlap_area < 1.0e-14:
                continue

            # Walk overlap boundary, integrate u_h^s . n_outward at each edge.
            # In source local frame.
            src_centroid = src.cell_centroids[cs]
            scale_s = float(src_scales[cs])
            boundary_flux = 0.0
            n_pts = overlap.shape[0]
            for j in range(n_pts):
                a = overlap[j] - src_centroid
                b = overlap[(j + 1) % n_pts] - src_centroid
                # Boundary outward normal for CCW polygon is right-rotation of edge.
                edge = b - a
                length_e = float(np.linalg.norm(edge))
                if length_e < TOL_CLIP:
                    continue
                n_out = np.array([edge[1], -edge[0]]) / length_e
                row = subsegment_normal_integral_row(a, b, n_out, N_h, scale_s)
                boundary_flux += float(row @ coeffs)

            expected = d_s * overlap_area
            err = abs(boundary_flux - expected)
            max_err = max(max_err, err)
            tested += 1

    if tested == 0:
        raise RuntimeError("[FAIL] per-overlap conservation: no overlaps tested")
    if max_err > TOL_CONSERVATION:
        raise RuntimeError(
            f"[FAIL] per-overlap conservation max error {max_err:.3e} > {TOL_CONSERVATION:.0e} "
            f"over {tested} sampled overlaps"
        )
    print(
        f"[OK] per-overlap conservation: max |bdry_flux - d * area| = {max_err:.3e} over {tested} overlaps"
    )


def _polygon_polygon_clip(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray | None:
    """Sutherland-Hodgman polygon clip of subject (CCW) against clipper (CCW)."""
    output = subject.tolist()
    nclip = clipper.shape[0]
    for i in range(nclip):
        if not output:
            return None
        c0 = clipper[i]
        c1 = clipper[(i + 1) % nclip]
        edge = c1 - c0
        n_in = np.array([-edge[1], edge[0]])
        new_output: list[np.ndarray] = []
        m = len(output)
        for j in range(m):
            curr = np.asarray(output[j])
            prev = np.asarray(output[(j - 1) % m])
            s_curr = float(np.dot(curr - c0, n_in))
            s_prev = float(np.dot(prev - c0, n_in))
            if s_curr >= -TOL_CLIP:
                if s_prev < -TOL_CLIP:
                    t = s_prev / (s_prev - s_curr)
                    new_output.append(prev + t * (curr - prev))
                new_output.append(curr)
            elif s_prev >= -TOL_CLIP:
                t = s_prev / (s_prev - s_curr)
                new_output.append(prev + t * (curr - prev))
        output = new_output
    if len(output) < 3:
        return None
    return np.array(output)


def run_one_pair(n_src_seeds: int, n_tgt_seeds: int, field: Callable[[np.ndarray], np.ndarray], halton_offset_tgt: int = 1000) -> dict:
    """Build one source/target pair, transfer field, return diagnostics."""
    src_seeds = halton(n_src_seeds, seed_offset=0)
    tgt_seeds = halton(n_tgt_seeds, seed_offset=halton_offset_tgt)
    src = voronoi_in_unit_square(src_seeds)
    tgt = voronoi_in_unit_square(tgt_seeds)
    src_R, src_scales = precompute_reconstructions(src)
    W = build_projection_operator(src, tgt, src_R, src_scales)

    src_fluxes = compute_edge_fluxes(field, src)
    tgt_exact = compute_edge_fluxes(field, tgt)
    tgt_computed = W @ src_fluxes
    err = tgt_computed - tgt_exact
    rel_l2 = float(np.linalg.norm(err) / max(np.linalg.norm(tgt_exact), 1.0e-30))
    linf = float(np.max(np.abs(err)))
    h = 1.0 / math.sqrt(n_src_seeds)
    return dict(
        n_src=n_src_seeds,
        n_tgt=n_tgt_seeds,
        src=src,
        tgt=tgt,
        src_R=src_R,
        src_scales=src_scales,
        W=W,
        src_fluxes=src_fluxes,
        tgt_exact=tgt_exact,
        tgt_computed=tgt_computed,
        rel_l2=rel_l2,
        linf=linf,
        h=h,
    )


def convergence_study(field: Callable[[np.ndarray], np.ndarray], levels: list[int], field_name: str) -> dict:
    """Run convergence at given source-seed counts. Returns h, l2, linf, rate."""
    print(f"\n=== Convergence study: {field_name} ===")
    print(f"{'N_src':>8} {'N_tgt':>8} {'h':>10} {'rel L2':>14} {'Linf':>14}")
    h_arr = []
    l2_arr = []
    linf_arr = []
    for n in levels:
        n_tgt = int(round(1.5 * n))
        result = run_one_pair(n, n_tgt, field)
        h_arr.append(result["h"])
        l2_arr.append(result["rel_l2"])
        linf_arr.append(result["linf"])
        print(f"{n:>8d} {n_tgt:>8d} {result['h']:>10.5f} {result['rel_l2']:>14.4e} {result['linf']:>14.4e}")

    h_arr = np.array(h_arr)
    l2_arr = np.array(l2_arr)
    linf_arr = np.array(linf_arr)
    log_h = np.log(h_arr)
    log_l2 = np.log(l2_arr)
    log_linf = np.log(linf_arr)
    slope_l2, _ = np.polyfit(log_h, log_l2, 1)
    slope_linf, _ = np.polyfit(log_h, log_linf, 1)
    print(f"  fitted slopes: L2 rate = {slope_l2:.3f}, Linf rate = {slope_linf:.3f}")
    return dict(h=h_arr, l2=l2_arr, linf=linf_arr, rate_l2=float(slope_l2), rate_linf=float(slope_linf))


def plot_convergence(results: dict[str, dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"div_free": "tab:blue", "smooth_div": "tab:red"}
    markers = {"l2": "o", "linf": "s"}
    for field_name, data in results.items():
        h = data["h"]
        c = colors.get(field_name, "k")
        ax.loglog(h, data["l2"], f"{markers['l2']}-", color=c, label=f"{field_name} L2 (rate {data['rate_l2']:.2f})")
        ax.loglog(h, data["linf"], f"{markers['linf']}--", color=c, label=f"{field_name} Linf (rate {data['rate_linf']:.2f})")

    h_ref = np.array([min(d["h"].min() for d in results.values()), max(d["h"].max() for d in results.values())])
    # O(h) reference, scaled to pass near the first point of the first dataset.
    first = next(iter(results.values()))
    scale = first["l2"][0] / first["h"][0]
    ax.loglog(h_ref, scale * h_ref, "k:", alpha=0.6, label="O(h) reference")

    ax.set_xlabel(r"Mesh size $h \sim N_{\mathrm{src}}^{-1/2}$")
    ax.set_ylabel("Relative edge-flux error")
    ax.set_title("p=0 mimetic Voronoi-to-Voronoi edge-flux transfer")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote convergence plot to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="p=0 mimetic Voronoi-to-Voronoi edge-flux transfer")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd(), help="Directory for output files")
    parser.add_argument(
        "--levels",
        type=str,
        default="64,256,1024,4096",
        help="Comma-separated source seed counts for convergence study",
    )
    parser.add_argument(
        "--rate-threshold",
        type=float,
        default=0.9,
        help="Minimum acceptable convergence slope (asserted at end)",
    )
    args = parser.parse_args(argv)

    levels = [int(x) for x in args.levels.split(",")]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Single-pair sanity / harmonic exactness / per-overlap conservation.
    # Use the smallest level to keep this fast.
    # ------------------------------------------------------------------
    n_check = max(64, levels[0])
    n_check_tgt = int(round(1.5 * n_check))
    print(f"\n=== Sanity checks on N_src={n_check}, N_tgt={n_check_tgt} ===")
    src_seeds = halton(n_check, 0)
    tgt_seeds = halton(n_check_tgt, 1000)
    src = voronoi_in_unit_square(src_seeds)
    tgt = voronoi_in_unit_square(tgt_seeds)
    check_mesh_sanity(src, "source")
    check_mesh_sanity(tgt, "target")
    src_R, src_scales = precompute_reconstructions(src)
    W_check = build_projection_operator(src, tgt, src_R, src_scales)
    print(f"[INFO] W shape {W_check.shape}, nnz = {W_check.nnz}")

    src_fluxes_div_free = compute_edge_fluxes(field_div_free, src)
    check_constant_exactness(src, src_R, tgt, W_check)
    check_harmonic_recovery_per_cell(src, src_R, src_scales)
    check_per_overlap_conservation(src, src_R, src_scales, tgt, src_fluxes_div_free, n_samples=50)

    # ------------------------------------------------------------------
    # Convergence study on both fields.
    # ------------------------------------------------------------------
    results = {}
    results["div_free"] = convergence_study(field_div_free, levels, "div_free  u=(sinpx cospy, -cospx sinpy)")
    results["smooth_div"] = convergence_study(field_smooth_div, levels, "smooth_div u=(sinpx, sinpy)")

    plot_convergence(results, args.output_dir / "voronoi_p0_convergence.png")

    # ------------------------------------------------------------------
    # SCRIP files at the finest level, one per field (the mesh and W are the
    # same — only src_fluxes differ — so writing one SCRIP per (src, tgt) pair
    # really suffices, but per-field naming keeps provenance explicit).
    # ------------------------------------------------------------------
    n_finest = levels[-1]
    n_finest_tgt = int(round(1.5 * n_finest))
    print(f"\n=== SCRIP output: finest pair N_src={n_finest}, N_tgt={n_finest_tgt} ===")
    src_finest_seeds = halton(n_finest, 0)
    tgt_finest_seeds = halton(n_finest_tgt, 1000)
    src_finest = voronoi_in_unit_square(src_finest_seeds)
    tgt_finest = voronoi_in_unit_square(tgt_finest_seeds)
    src_R_finest, src_scales_finest = precompute_reconstructions(src_finest)
    W_finest = build_projection_operator(src_finest, tgt_finest, src_R_finest, src_scales_finest)
    for field_name in ("div_free", "smooth_div"):
        path = args.output_dir / f"voronoi_p0_weights_{field_name}_N{n_finest}_to_N{n_finest_tgt}.nc"
        write_scrip_edges_as_grid(
            path,
            src_finest,
            tgt_finest,
            W_finest,
            title=f"Mimetic p=0 Voronoi-to-Voronoi edge-flux remap ({field_name})",
        )
        print(f"[OK] wrote SCRIP file {path}")

    # ------------------------------------------------------------------
    # Final pass/fail.
    # ------------------------------------------------------------------
    print("\n=== Summary ===")
    all_ok = True
    for field_name, data in results.items():
        ok = data["rate_l2"] >= args.rate_threshold
        all_ok = all_ok and ok
        flag = "PASS" if ok else "FAIL"
        print(f"[{flag}] {field_name}: L2 rate {data['rate_l2']:.3f} (threshold {args.rate_threshold})")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
