from __future__ import annotations

import numpy as np

from .catalog_specs import specs_for_module
from .common import centroid, emit, face_scale, zone_points

IMPLEMENTATION = "zone_relations.py"

RELATION_PAIRS = (
    ("temporal_L", "orbit_L"),
    ("temporal_R", "orbit_R"),
    ("brow_ridge_L", "jaw_angle_L"),
    ("brow_ridge_R", "jaw_angle_R"),
    ("cheekbone_L", "temporal_L"),
    ("cheekbone_R", "temporal_R"),
    ("cheekbone_L", "orbit_L"),
    ("cheekbone_R", "orbit_R"),
    ("nose_wing_L", "nose_bridge_tip"),
    ("nose_wing_R", "nose_bridge_tip"),
    # Jaw-to-orbit relations — canonical analogue for mirror_jaw_centroid_asymmetry
    ("jaw_L", "orbit_L"),
    ("jaw_R", "orbit_R"),
    ("jaw_L", "brow_ridge_L"),
    ("jaw_R", "brow_ridge_R"),
)


def specs():
    return specs_for_module(IMPLEMENTATION, families={"F_zone_rel"})


def _normal_mean(ctx, region: str, space: str) -> np.ndarray | None:
    normals = (
        ctx.normals_shape_neutral
        if space == "shape_neutral"
        else ctx.normals_canon
        if space == "canon_bucket"
        else ctx.normals_raw
    )
    raw = ctx.macro_indices.get(region, [])
    if normals is None or not raw:
        return None
    idx = np.asarray(list(raw), dtype=int)
    idx = idx[(idx >= 0) & (idx < len(normals))]
    if len(idx) < 2:
        return None
    ns = np.asarray(normals[idx], dtype=float)
    m = np.mean(ns, axis=0)
    nlen = float(np.linalg.norm(m))
    return m / nlen if nlen > 1e-8 else None


def compute(ctx, specs_):
    out = []
    spec_by = {s.name: s for s in specs_}
    for space in ("canon_bucket", "shape_neutral"):
        if space == "shape_neutral" and ctx.vertices_shape_neutral is None:
            continue
        scale = face_scale(ctx, space=space)
        for a, b in RELATION_PAIRS:
            ca, cb = centroid(ctx, a, space=space), centroid(ctx, b, space=space)
            if ca is None or cb is None:
                continue
            dist_name = f"zone_rel_{a}_to_{b}_distance_ratio"
            if dist_name in spec_by:
                val = float(np.linalg.norm(ca - cb) / (scale + 1e-8))
                if mv := emit(spec_by[dist_name], val, confidence=0.68, source_space=space):
                    out.append(mv)
            ang_name = f"zone_rel_{a}_to_{b}_normal_angle_deg"
            if ang_name in spec_by:
                na, nb = _normal_mean(ctx, a, space), _normal_mean(ctx, b, space)
                if na is not None and nb is not None:
                    dot = float(np.clip(np.dot(na, nb), -1.0, 1.0))
                    ang = float(np.degrees(np.arccos(dot)))
                    if mv := emit(spec_by[ang_name], ang, confidence=0.68, source_space=space):
                        out.append(mv)
    return out
