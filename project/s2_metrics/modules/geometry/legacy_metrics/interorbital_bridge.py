from __future__ import annotations

import numpy as np

from .catalog_specs import specs_for_module
from .common import emit, face_scale
from .primitives import point, distance

IMPLEMENTATION = "interorbital_bridge.py"


def specs():
    return specs_for_module(IMPLEMENTATION, families={"F_bridge"})


def compute(ctx, specs_):
    out = []
    spec_by = {s.name: s for s in specs_}

    # В отличие от legacy interorbital_ratio, здесь цель — мост переносицы между медиальными кантусами.
    l = point(ctx, "inner_canthus_L", space="shape_neutral") or point(ctx, "inner_canthus_L")
    r = point(ctx, "inner_canthus_R", space="shape_neutral") or point(ctx, "inner_canthus_R")
    scale = face_scale(ctx, space="shape_neutral") if ctx.vertices_shape_neutral is not None else face_scale(ctx)
    if l is None or r is None:
        return out

    w = (distance(l, r) or 0.0) / (scale + 1e-8)
    if "interorbital_bridge_width_ratio" in spec_by and (mv := emit(spec_by["interorbital_bridge_width_ratio"], float(w), confidence=0.70)):
        out.append(mv)

    return out

