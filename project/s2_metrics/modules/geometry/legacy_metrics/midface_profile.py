from __future__ import annotations

import numpy as np

from .catalog_specs import specs_for_module
from .common import emit, face_scale, zone_points

IMPLEMENTATION = "midface_profile.py"


def specs():
    return specs_for_module(IMPLEMENTATION, families={"F_midface"})


def _best_cheek_points(ctx, space: str) -> np.ndarray:
    left = zone_points(ctx, "cheekbone_L", space=space, visible_only=True)
    right = zone_points(ctx, "cheekbone_R", space=space, visible_only=True)
    if len(left) >= len(right):
        return left
    return right


def compute(ctx, specs_):
    out = []
    spec_by = {s.name: s for s in specs_}
    space = "shape_neutral" if ctx.vertices_shape_neutral is not None else "canon_bucket"
    pts = _best_cheek_points(ctx, space)
    if len(pts) < 3:
        return out
    scale = face_scale(ctx, space=space)
    width = float(np.ptp(pts[:, 0])) / (scale + 1e-8)
    if "midface_width_profile_ratio" in spec_by and (mv := emit(spec_by["midface_width_profile_ratio"], width, confidence=0.60)):
        out.append(mv)
    return out

