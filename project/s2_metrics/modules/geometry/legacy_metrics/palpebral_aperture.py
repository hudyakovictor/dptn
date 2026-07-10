from __future__ import annotations

import numpy as np

from .catalog_specs import specs_for_module
from .common import emit, face_scale
from .primitives import eye_points

IMPLEMENTATION = "palpebral_aperture.py"


def specs():
    return specs_for_module(IMPLEMENTATION, families={"F_palpebral"})


def _aperture_stats(ctx, side: str, scale: float) -> dict[str, float] | None:
    pts = eye_points(ctx, side)
    if len(pts) < 2:
        return None
    width = float(np.ptp(pts[:, 0])) / (scale + 1e-8)
    height = float(np.ptp(pts[:, 1])) / (scale + 1e-8)
    return {
        "palpebral_aperture_width_ratio": width,
        "palpebral_aperture_height_ratio": height,
        "palpebral_aperture_aspect_ratio": height / (width + 1e-8),
    }


def compute(ctx, specs_):
    out = []
    spec_by = {s.name: s for s in specs_}
    scale = face_scale(ctx, space="shape_neutral") if ctx.vertices_shape_neutral is not None else face_scale(ctx)

    stats_l = _aperture_stats(ctx, "L", scale)
    stats_r = _aperture_stats(ctx, "R", scale)
    if stats_l is None and stats_r is None:
        return out

    for side, stats in (("L", stats_l), ("R", stats_r)):
        if not stats:
            continue
        for key, val in stats.items():
            name = f"{key}_{side}"
            if name in spec_by and (mv := emit(spec_by[name], float(val), confidence=0.68)):
                out.append(mv)

    if stats_l and stats_r:
        # Симметрия — диагностическая сводка, пригодна для discrimination, но не ядро.
        name = "palpebral_aperture_height_asymmetry_ratio"
        if name in spec_by:
            asym = abs(float(stats_l["palpebral_aperture_height_ratio"]) - float(stats_r["palpebral_aperture_height_ratio"]))
            if mv := emit(spec_by[name], float(asym), confidence=0.65):
                out.append(mv)

    return out

