from __future__ import annotations

import numpy as np


BILATERAL_ZONE_PAIRS = (
    ("orbit_L", "orbit_R"),
    ("brow_ridge_L", "brow_ridge_R"),
    ("cheekbone_L", "cheekbone_R"),
    ("temporal_L", "temporal_R"),
    ("jaw_L", "jaw_R"),
)


def _centroid(vertices: np.ndarray, indices) -> np.ndarray | None:
    if not indices:
        return None
    idx = np.asarray(list(indices), dtype=np.int64)
    idx = idx[(idx >= 0) & (idx < len(vertices))]
    if idx.size == 0:
        return None
    c = np.mean(vertices[idx], axis=0)
    return c if np.isfinite(c).all() else None


def estimate_mid_sagittal_plane(vertices: np.ndarray, macro_indices: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """Estimate mid-sagittal plane from bilateral zone centroids.

    Returns (point_on_plane, unit_normal, confidence). In canonical 3DDFA space the
    plane is usually close to x=const, but this estimator follows reconstructed
    bilateral anchors and is more stable for small roll/yaw residuals.
    """
    v = np.asarray(vertices, dtype=float)
    diffs: list[np.ndarray] = []
    mids: list[np.ndarray] = []
    for left, right in BILATERAL_ZONE_PAIRS:
        cl = _centroid(v, macro_indices.get(left, []))
        cr = _centroid(v, macro_indices.get(right, []))
        if cl is None or cr is None:
            continue
        d = cr - cl
        if np.linalg.norm(d) <= 1e-8:
            continue
        diffs.append(d)
        mids.append((cl + cr) / 2.0)
    if not diffs:
        point = np.mean(v, axis=0) if len(v) else np.zeros(3)
        return point, np.array([1.0, 0.0, 0.0]), 0.0
    normal = np.mean([d / (np.linalg.norm(d) + 1e-8) for d in diffs], axis=0)
    nn = np.linalg.norm(normal)
    if nn <= 1e-8:
        normal = np.array([1.0, 0.0, 0.0])
    else:
        normal = normal / nn
    # Keep deterministic orientation: positive x component where possible.
    if normal[0] < 0:
        normal = -normal
    point = np.mean(mids, axis=0)
    # Confidence: number of pairs and agreement of normals.
    unit_diffs = np.asarray([d / (np.linalg.norm(d) + 1e-8) for d in diffs])
    agreement = float(np.linalg.norm(np.mean(unit_diffs, axis=0)))
    confidence = float(np.clip((len(diffs) / len(BILATERAL_ZONE_PAIRS)) * agreement, 0.0, 1.0))
    return point, normal, confidence


def reflect_points_across_plane(points: np.ndarray, plane_point: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    n = np.asarray(plane_normal, dtype=float)
    n = n / (np.linalg.norm(n) + 1e-8)
    p0 = np.asarray(plane_point, dtype=float)
    signed = (pts - p0) @ n
    return pts - 2.0 * signed[:, None] * n[None, :]
