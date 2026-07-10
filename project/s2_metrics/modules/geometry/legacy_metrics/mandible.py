from __future__ import annotations

import numpy as np

from .catalog_specs import specs_for_module
from .common import centroid, distance, emit, face_scale, zone_points
from .primitives import angle, point

IMPLEMENTATION = "mandible.py"


def specs():
    return specs_for_module(IMPLEMENTATION, families={"F_mandible"})


def _arc(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return 0.0
    p = pts[np.argsort(pts[:, 1])]
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


def _compute_space(ctx, space: str) -> dict[str, float | None]:
    vals: dict[str, float | None] = {}
    scale = face_scale(ctx, space=space)
    chin_pts = zone_points(ctx, "chin", space=space)
    chin_c = centroid(ctx, "chin", space=space)
    for side, region in (("L", "jaw_L"), ("R", "jaw_R")):
        pts = zone_points(ctx, region, space=space)
        if len(pts) > 1:
            vals.update({
                f"{side}_jawline_arc_length_ratio": _arc(pts) / scale,
                f"{side}_ramus_height_proxy_ratio": float(np.ptp(pts[:, 1]) / scale),
                f"{side}_jaw_depth_span_ratio": float(np.ptp(pts[:, 2]) / scale),
                f"{side}_gonion_to_chin_curve_sharpness": float(np.std(pts[:, 2]) / (np.ptp(pts[:, 1]) + 1e-8)),
                f"{side}_jaw_UV_shape_index": float(np.ptp(pts[:, 0]) / (np.ptp(pts[:, 1]) + 1e-8)),
                f"{side}_gonial_flare_index": float(np.ptp(pts[:, 0]) / (np.ptp(pts[:, 2]) + 1e-8)),
            })
        gon = point(ctx, f"gonion_{side}", space=space)
        sub = point(ctx, "subnasale", space=space)
        if gon is not None and chin_c is not None and sub is not None:
            vals[f"{side}_mandibular_plane_angle"] = angle(gon, chin_c, sub)
    if len(chin_pts) > 0:
        vals.update({
            "chin_width_ratio": float(np.ptp(chin_pts[:, 0]) / scale),
            "chin_height_ratio": float(np.ptp(chin_pts[:, 1]) / scale),
            "chin_depth_ratio": float(np.ptp(chin_pts[:, 2]) / scale),
        })
    jl, jr = centroid(ctx, "jaw_L", space=space), centroid(ctx, "jaw_R", space=space)
    sub = point(ctx, "subnasale", space=space)
    if jl is not None and jr is not None and chin_c is not None:
        big = distance(jl, jr) or 0.0
        mid = ((distance(jl, chin_c) or 0.0) + (distance(jr, chin_c) or 0.0)) / 2.0
        vals["jaw_taper_index"] = big / (mid + 1e-8)
        vals["bigonial_to_bizygomatic_ratio"] = big / scale
    if sub is not None and chin_c is not None:
        vals["mental_angle"] = angle(sub, chin_c, jl) if jl is not None else None
        vals["symphysis_projection"] = float((chin_c[2] - sub[2]) / scale)
    return vals


def compute(ctx, specs_):
    out = []
    cache = {}
    for spec in specs_:
        for space in spec.source_spaces:
            if space == "shape_neutral" and ctx.vertices_shape_neutral is None:
                continue
            if space not in {"canon_bucket", "shape_neutral", "raw"}:
                continue
            if space not in cache:
                cache[space] = _compute_space(ctx, space)
            val = cache[space].get(spec.name)
            if (mv := emit(spec, val, confidence=0.72, source_space=space)):
                out.append(mv)
    return out
