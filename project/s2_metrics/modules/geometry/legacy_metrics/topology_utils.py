from __future__ import annotations

import heapq
from functools import lru_cache
from typing import Iterable

import numpy as np


def valid_triangles_for_vertices(triangles: np.ndarray, vertex_count: int) -> np.ndarray:
    tri = np.asarray(triangles, dtype=np.int64)
    if tri.ndim != 2 or tri.shape[1] != 3 or vertex_count <= 0:
        return np.zeros((0, 3), dtype=np.int64)
    mask = np.all((tri >= 0) & (tri < vertex_count), axis=1)
    return tri[mask]


def recompute_vertex_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray | None:
    """Per-vertex normals from mesh triangulation (area-weighted face normals)."""
    v = np.asarray(vertices, dtype=np.float64)
    tri = valid_triangles_for_vertices(triangles, len(v))
    if len(v) == 0 or len(tri) == 0:
        return None
    normals = np.zeros_like(v)
    counts = np.zeros(len(v), dtype=np.float64)
    for a, b, c in tri:
        va, vb, vc = v[a], v[b], v[c]
        fn = np.cross(vb - va, vc - va)
        nlen = float(np.linalg.norm(fn))
        if nlen < 1e-12:
            continue
        fn = fn / nlen
        for idx in (a, b, c):
            normals[idx] += fn
            counts[idx] += 1.0
    mask = counts > 0
    normals[mask] /= counts[mask, None]
    nlen = np.linalg.norm(normals, axis=1, keepdims=True)
    nlen = np.maximum(nlen, 1e-8)
    normals = normals / nlen
    return normals.astype(np.float64)


def region_surface_area(vertices: np.ndarray, triangles: np.ndarray, region_indices: Iterable[int], *, mode: str = "inside") -> float | None:
    """Surface area for a vertex region.

    mode="inside" uses triangles whose 3 vertices are inside region.
    If there are too few fully-inside triangles, falls back to triangles touching
    the region with fractional contribution (#inside/3).
    """
    v = np.asarray(vertices, dtype=float)
    tri = valid_triangles_for_vertices(triangles, len(v))
    idx = np.asarray(list(region_indices), dtype=np.int64)
    idx = idx[(idx >= 0) & (idx < len(v))]
    if len(v) == 0 or len(tri) == 0 or len(idx) < 3:
        return None
    mask_vertices = np.zeros(len(v), dtype=bool)
    mask_vertices[idx] = True
    inside_count = np.sum(mask_vertices[tri], axis=1)
    if mode == "inside":
        selected = tri[inside_count == 3]
        weights = np.ones(len(selected), dtype=float)
        if len(selected) < 2:
            selected = tri[inside_count > 0]
            weights = inside_count[inside_count > 0].astype(float) / 3.0
    else:
        selected = tri[inside_count > 0]
        weights = inside_count[inside_count > 0].astype(float) / 3.0
    if len(selected) == 0:
        return None
    a, b, c = v[selected[:, 0]], v[selected[:, 1]], v[selected[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    return float(np.sum(areas * weights))


def build_adjacency(vertices: np.ndarray, triangles: np.ndarray, allowed_indices: Iterable[int] | None = None) -> dict[int, list[tuple[int, float]]]:
    v = np.asarray(vertices, dtype=float)
    tri = valid_triangles_for_vertices(triangles, len(v))
    if allowed_indices is not None:
        allowed = set(int(i) for i in allowed_indices if 0 <= int(i) < len(v))
    else:
        allowed = set(range(len(v)))
    adj: dict[int, dict[int, float]] = {i: {} for i in allowed}
    for t in tri:
        a, b, c = map(int, t)
        for u, w in ((a, b), (b, c), (c, a)):
            if u not in allowed or w not in allowed:
                continue
            d = float(np.linalg.norm(v[u] - v[w]))
            old = adj[u].get(w)
            if old is None or d < old:
                adj[u][w] = d
                adj[w][u] = d
    return {k: list(vv.items()) for k, vv in adj.items()}


def dijkstra_distance(adj: dict[int, list[tuple[int, float]]], start: int, goal: int) -> float | None:
    if start not in adj or goal not in adj:
        return None
    pq: list[tuple[float, int]] = [(0.0, start)]
    seen: set[int] = set()
    dist = {start: 0.0}
    while pq:
        d, u = heapq.heappop(pq)
        if u == goal:
            return float(d)
        if u in seen:
            continue
        seen.add(u)
        for w, weight in adj.get(u, []):
            nd = d + weight
            if nd < dist.get(w, float("inf")):
                dist[w] = nd
                heapq.heappush(pq, (nd, w))
    return None


def region_geodesic_extrema(vertices: np.ndarray, triangles: np.ndarray, region_indices: Iterable[int], *, axis: int = 0) -> float | None:
    v = np.asarray(vertices, dtype=float)
    idx = np.asarray(list(region_indices), dtype=np.int64)
    idx = idx[(idx >= 0) & (idx < len(v))]
    if idx.size < 2:
        return None
    start = int(idx[np.argmin(v[idx, axis])])
    goal = int(idx[np.argmax(v[idx, axis])])
    if start == goal:
        return 0.0
    adj = build_adjacency(v, triangles, allowed_indices=idx)
    return dijkstra_distance(adj, start, goal)


def laplacian_curvature_proxy(vertices: np.ndarray, triangles: np.ndarray, region_indices: Iterable[int]) -> float | None:
    """Mean umbrella-Laplacian magnitude inside region; proxy for local bending."""
    v = np.asarray(vertices, dtype=float)
    idx = np.asarray(list(region_indices), dtype=np.int64)
    idx = idx[(idx >= 0) & (idx < len(v))]
    if idx.size < 3:
        return None
    adj = build_adjacency(v, triangles, allowed_indices=None)
    mags: list[float] = []
    region_set = set(int(i) for i in idx)
    for i in idx:
        neigh = [j for j, _ in adj.get(int(i), [])]
        if not neigh:
            continue
        # Use all mesh neighbours; region boundary remains informative.
        mean_n = np.mean(v[neigh], axis=0)
        mags.append(float(np.linalg.norm(v[int(i)] - mean_n)))
    if not mags:
        return None
    return float(np.mean(mags))


def discrete_curvature_stats(vertices: np.ndarray, triangles: np.ndarray, region_indices: Iterable[int]) -> dict[str, float]:
    """Discrete curvature estimates for a region.

    Uses standard lightweight triangular-mesh estimators:
    - Gaussian curvature: angle deficit / mixed area (barycentric area here).
    - Mean curvature proxy: umbrella Laplacian magnitude / local edge scale.
    This is not CT-grade differential geometry, but it is topology-based and
    materially stronger than pure z-span proxies.
    """
    v = np.asarray(vertices, dtype=float)
    tri = valid_triangles_for_vertices(triangles, len(v))
    idx = np.asarray(list(region_indices), dtype=np.int64)
    idx = idx[(idx >= 0) & (idx < len(v))]
    if len(v) == 0 or len(tri) == 0 or idx.size < 3:
        return {}
    region = set(int(i) for i in idx)
    area = {int(i): 0.0 for i in idx}
    angle_sum = {int(i): 0.0 for i in idx}
    edge_lengths: dict[int, list[float]] = {int(i): [] for i in idx}

    def _angle(a, b, c) -> float:
        ba = a - b
        bc = c - b
        den = np.linalg.norm(ba) * np.linalg.norm(bc)
        if den <= 1e-12:
            return 0.0
        return float(np.arccos(np.clip(np.dot(ba, bc) / den, -1.0, 1.0)))

    for t in tri:
        t = [int(x) for x in t]
        if not any(x in region for x in t):
            continue
        p = [v[x] for x in t]
        tri_area = 0.5 * float(np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0])))
        if tri_area <= 1e-12:
            continue
        angles = [_angle(p[1], p[0], p[2]), _angle(p[0], p[1], p[2]), _angle(p[0], p[2], p[1])]
        for local_i, vi in enumerate(t):
            if vi in region:
                area[vi] += tri_area / 3.0
                angle_sum[vi] += angles[local_i]
                for vj in t:
                    if vj != vi:
                        edge_lengths[vi].append(float(np.linalg.norm(v[vi] - v[vj])))

    adj = build_adjacency(v, tri, allowed_indices=None)
    H_vals = []
    K_vals = []
    k1_vals = []
    k2_vals = []
    for vi in idx:
        vi = int(vi)
        if area.get(vi, 0.0) <= 1e-12:
            continue
        K = (2.0 * np.pi - angle_sum[vi]) / (area[vi] + 1e-12)
        neigh = [j for j, _ in adj.get(vi, [])]
        if neigh:
            lap = v[vi] - np.mean(v[neigh], axis=0)
            local_scale = float(np.median(edge_lengths.get(vi) or [1.0]))
            H = float(np.linalg.norm(lap) / (local_scale * local_scale + 1e-12))
        else:
            H = 0.0
        disc = max(H * H - K, 0.0)
        root = float(np.sqrt(disc))
        k1 = H + root
        k2 = H - root
        if np.isfinite(H) and np.isfinite(K):
            H_vals.append(H)
            K_vals.append(float(K))
            k1_vals.append(float(k1))
            k2_vals.append(float(k2))
    if not H_vals:
        return {}
    H_arr = np.asarray(H_vals, dtype=float)
    K_arr = np.asarray(K_vals, dtype=float)
    k1_arr = np.asarray(k1_vals, dtype=float)
    k2_arr = np.asarray(k2_vals, dtype=float)
    return {
        "mean_curvature_H_mean": float(np.mean(np.abs(H_arr))),
        "mean_curvature_H_p95": float(np.percentile(np.abs(H_arr), 95)),
        "gaussian_curvature_K_mean": float(np.mean(K_arr)),
        "gaussian_curvature_K_p95": float(np.percentile(np.abs(K_arr), 95)),
        "principal_k1_mean": float(np.mean(k1_arr)),
        "principal_k2_mean": float(np.mean(k2_arr)),
    }


def region_signed_volume_to_plane(vertices: np.ndarray, triangles: np.ndarray, region_indices: Iterable[int], plane_point: np.ndarray, plane_normal: np.ndarray) -> float | None:
    """Signed prism-volume proxy from region surface to a reference plane."""
    v = np.asarray(vertices, dtype=float)
    tri = valid_triangles_for_vertices(triangles, len(v))
    idx = np.asarray(list(region_indices), dtype=np.int64)
    idx = idx[(idx >= 0) & (idx < len(v))]
    if idx.size < 3 or len(tri) == 0:
        return None
    mask = np.zeros(len(v), dtype=bool)
    mask[idx] = True
    inside_count = np.sum(mask[tri], axis=1)
    selected = tri[inside_count > 0]
    weights = inside_count[inside_count > 0].astype(float) / 3.0
    if len(selected) == 0:
        return None
    n = np.asarray(plane_normal, dtype=float)
    n = n / (np.linalg.norm(n) + 1e-12)
    p0 = np.asarray(plane_point, dtype=float)
    a, b, c = v[selected[:, 0]], v[selected[:, 1]], v[selected[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    centroids = (a + b + c) / 3.0
    signed_h = (centroids - p0) @ n
    return float(np.sum(areas * signed_h * weights))
