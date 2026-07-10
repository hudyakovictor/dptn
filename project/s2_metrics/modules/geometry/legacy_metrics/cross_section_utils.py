from __future__ import annotations

import numpy as np

from .topology_utils import valid_triangles_for_vertices


def plane_mesh_intersections(
    vertices: np.ndarray,
    triangles: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    *,
    eps: float = 1e-8,
) -> np.ndarray:
    """Return points where mesh triangle edges intersect a plane.

    This is a lightweight deterministic slicer; it returns unordered points from
    triangle-edge intersections. Downstream metrics should sort/project them.
    """
    v = np.asarray(vertices, dtype=float)
    tri = valid_triangles_for_vertices(triangles, len(v))
    if len(v) == 0 or len(tri) == 0:
        return np.zeros((0, 3), dtype=float)
    p0 = np.asarray(plane_point, dtype=float).reshape(3)
    n = np.asarray(plane_normal, dtype=float).reshape(3)
    nn = np.linalg.norm(n)
    if nn <= eps:
        return np.zeros((0, 3), dtype=float)
    n = n / nn
    pts: list[np.ndarray] = []
    for t in tri:
        pv = v[t]
        d = (pv - p0) @ n
        # all same side and not touching
        if np.all(d > eps) or np.all(d < -eps):
            continue
        for i, j in ((0, 1), (1, 2), (2, 0)):
            di, dj = float(d[i]), float(d[j])
            vi, vj = pv[i], pv[j]
            if abs(di) <= eps:
                pts.append(vi)
            if di * dj < -eps:
                alpha = di / (di - dj)
                pts.append(vi + alpha * (vj - vi))
            elif abs(dj) <= eps:
                pts.append(vj)
    if not pts:
        return np.zeros((0, 3), dtype=float)
    arr = np.asarray(pts, dtype=float)
    # Deduplicate by quantization to avoid repeated triangle-edge points.
    q = np.round(arr / max(eps * 1000, 1e-6)).astype(np.int64)
    _, uniq = np.unique(q, axis=0, return_index=True)
    return arr[np.sort(uniq)]


def section_metrics(points: np.ndarray, *, axes: tuple[int, int] = (1, 2)) -> dict[str, float]:
    """Basic metrics for unordered section points projected onto two axes."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return {}
    a0, a1 = axes
    p = pts[:, [a0, a1]]
    span0 = float(np.ptp(p[:, 0]))
    span1 = float(np.ptp(p[:, 1]))
    # Ordered polyline proxy: sort by first axis.
    ps = p[np.argsort(p[:, 0])]
    curve = float(np.sum(np.linalg.norm(np.diff(ps, axis=0), axis=1))) if len(ps) > 1 else 0.0
    area_bbox = span0 * span1
    # Concavity/convexity proxy from residual around linear fit.
    if len(ps) > 3 and np.ptp(ps[:, 0]) > 1e-8:
        coef = np.polyfit(ps[:, 0], ps[:, 1], 1)
        resid = ps[:, 1] - np.polyval(coef, ps[:, 0])
        residual_std = float(np.std(resid))
        residual_p95 = float(np.percentile(np.abs(resid), 95))
    else:
        residual_std = 0.0
        residual_p95 = 0.0
    return {
        "width": span0,
        "depth": span1,
        "bbox_area": area_bbox,
        "curve_length": curve,
        "residual_std": residual_std,
        "residual_p95": residual_p95,
        "point_count": float(len(pts)),
    }
