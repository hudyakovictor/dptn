from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .types import MetricSpec, MetricValue

ALL_BUCKETS = (
    "frontal",
    "left_threequarter_light",
    "left_threequarter_mid",
    "left_threequarter_deep",
    "left_profile",
    "right_threequarter_light",
    "right_threequarter_mid",
    "right_threequarter_deep",
    "right_profile",
    "unclassified",
)

MIN_VISIBLE_RATIO = 0.50


def finite_float(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def metric(
    *,
    name: str,
    family: str,
    group: str,
    zone: str,
    side: str = "NA",
    source_spaces=("canon_bucket",),
    unit="ratio",
    normalization="none",
    implementation="",
    buckets=ALL_BUCKETS,
    scope="single",
    expression_sensitive=False,
    pose_sensitive="medium",
    tags=(),
) -> MetricSpec:
    return MetricSpec(
        name=name,
        family=family,
        group=group,
        zone=zone,
        side=side,  # type: ignore[arg-type]
        buckets=tuple(buckets),
        source_spaces=tuple(source_spaces),  # type: ignore[arg-type]
        scope=scope,  # type: ignore[arg-type]
        unit=unit,
        normalization=normalization,
        implementation=implementation,
        expression_sensitive=expression_sensitive,
        pose_sensitive=pose_sensitive,
        tags=tuple(tags),
    )


def emit(spec: MetricSpec, value: Any, *, confidence: float = 1.0, visibility: float | None = None, notes: str = "", source_space: str | None = None) -> MetricValue | None:
    v = finite_float(value)
    if v is None:
        return None
    return MetricValue(spec=spec, value=v, confidence=float(np.clip(confidence, 0.0, 1.0)), visibility=visibility, notes=notes, source_space=source_space)


def zone_points(ctx, zone_name: str, *, space: str = "canon_bucket", visible_only: bool = False) -> np.ndarray:
    verts = {
        "raw": ctx.vertices_raw,
        "canon_bucket": ctx.vertices_canon,
        "shape_neutral": ctx.vertices_shape_neutral,
    }.get(space)
    if verts is None:
        return np.zeros((0, 3), dtype=float)
    raw = ctx.macro_indices.get(zone_name, [])
    if raw is None or len(raw) == 0:
        return np.zeros((0, 3), dtype=float)
    idx = np.asarray(list(raw), dtype=np.int64)
    idx = idx[(idx >= 0) & (idx < len(verts))]
    if visible_only:
        vis = ctx.visibility_canon if space == "canon_bucket" else ctx.visibility_raw
        vis_arr = vis.binary_mask if hasattr(vis, "binary_mask") else vis
        if vis_arr is not None and len(vis_arr) == len(verts):
            vis_mask = np.asarray(vis_arr, dtype=bool)
            total = int(idx.size)
            idx = idx[vis_mask[idx]]
            if total > 0 and float(idx.size) / float(total) < MIN_VISIBLE_RATIO:
                return np.zeros((0, 3), dtype=float)
    if idx.size == 0:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(verts[idx], dtype=float)


def centroid(ctx, zone_name: str, *, space: str = "canon_bucket") -> np.ndarray | None:
    pts = zone_points(ctx, zone_name, space=space)
    if pts.size == 0:
        return None
    c = np.mean(pts, axis=0)
    return c if np.isfinite(c).all() else None


def distance(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    return finite_float(np.linalg.norm(a - b))


def face_scale(ctx, *, space: str = "canon_bucket") -> float:
    l = centroid(ctx, "cheekbone_L", space=space)
    r = centroid(ctx, "cheekbone_R", space=space)
    d = distance(l, r)
    if d and d > 1e-8:
        return d
    verts = ctx.vertices_canon if space == "canon_bucket" else ctx.vertices_raw
    span = float(np.ptp(verts[:, 0])) if verts is not None and len(verts) else 1.0
    return max(span, 1e-6)


def angle_deg(a: np.ndarray | None, b: np.ndarray | None, c: np.ndarray | None) -> float | None:
    if a is None or b is None or c is None:
        return None
    v1, v2 = a - b, c - b
    den = np.linalg.norm(v1) * np.linalg.norm(v2)
    if den <= 1e-8:
        return None
    return finite_float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / den, -1.0, 1.0))))


def triangle_area(a: np.ndarray | None, b: np.ndarray | None, c: np.ndarray | None) -> float | None:
    if a is None or b is None or c is None:
        return None
    return finite_float(0.5 * np.linalg.norm(np.cross(b - a, c - a)))
