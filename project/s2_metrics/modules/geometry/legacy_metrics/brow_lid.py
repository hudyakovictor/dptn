from __future__ import annotations

import numpy as np

from .catalog_specs import specs_for_module
from .common import emit, face_scale, zone_points
from .primitives import eye_points

IMPLEMENTATION = "brow_lid.py"


def specs():
    return specs_for_module(IMPLEMENTATION, families={"F_brow"})


def _compute_side(ctx, side: str, space: str) -> dict[str, float]:
    scale = face_scale(ctx, space=space)
    bp = zone_points(ctx, f"brow_ridge_{side}", space=space)
    ep = eye_points(ctx, side, space=space)
    if len(bp) == 0 or len(ep) == 0:
        return {}
    brow_c, eye_c = np.mean(bp, axis=0), np.mean(ep, axis=0)
    arc = float(np.sum(np.linalg.norm(np.diff(bp[np.argsort(bp[:, 0])], axis=0), axis=1))) if len(bp) > 1 else 0.0
    return {
        f"{side}_brow_lid_depth_gap_ratio": float((brow_c[2] - eye_c[2]) / scale),
        f"{side}_brow_lid_vertical_gap_ratio": float((brow_c[1] - eye_c[1]) / scale),
        f"{side}_brow_overhang_proxy": float((np.percentile(bp[:, 2], 75) - np.percentile(ep[:, 2], 50)) / scale),
        f"{side}_brow_lid_overlap_area": float((np.ptp(bp[:, 0]) * max(0.0, np.ptp(ep[:, 1]))) / (scale * scale + 1e-8)),
        f"{side}_supraorbital_shelf_projection": float((np.max(bp[:, 2]) - np.mean(ep[:, 2])) / scale),
        f"{side}_brow_arc_length": arc / scale,
        f"{side}_brow_arc_curvature": float(np.std(bp[:, 2]) / (np.ptp(bp[:, 0]) + 1e-8)),
        f"{side}_brow_lid_occlusion_index": float(max(0.0, np.percentile(bp[:, 2], 75) - np.percentile(ep[:, 2], 75)) / scale),
    }


def compute(ctx, specs_):
    out = []
    cache = {}
    for spec in specs_:
        side = "L" if spec.name.startswith("L_") else "R" if spec.name.startswith("R_") else None
        if side is None:
            continue
        for space in spec.source_spaces:
            if space == "shape_neutral" and ctx.vertices_shape_neutral is None:
                continue
            if space not in {"canon_bucket", "shape_neutral", "raw"}:
                continue
            key = (side, space)
            if key not in cache:
                cache[key] = _compute_side(ctx, side, space)
            val = cache[key].get(spec.name)
            if (mv := emit(spec, val, confidence=0.72, source_space=space)):
                out.append(mv)
    return out
