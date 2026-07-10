from __future__ import annotations

import numpy as np

from .catalog_specs import specs_for_module
from .common import emit, face_scale, zone_points
from .primitives import distance, point

IMPLEMENTATION = "nose_bridge.py"


def specs():
    return specs_for_module(IMPLEMENTATION, families={"F_nose"})


def _level_metrics(pts: np.ndarray, level: int, scale: float) -> dict[str, float]:
    qs = np.quantile(pts[:, 1], [0.0, 0.25, 0.5, 0.75, 1.0])
    lo, hi = qs[min(level, 3)], qs[min(level + 1, 4)]
    slab = pts[(pts[:, 1] >= lo) & (pts[:, 1] <= hi)]
    if len(slab) < 2:
        slab = pts
    return {
        f"upper_bridge_level_{level}_width_ratio": float(np.ptp(slab[:, 0])) / scale,
        f"upper_bridge_level_{level}_depth_ratio": float(np.mean(slab[:, 2])) / scale,
        f"upper_bridge_level_{level}_cross_section_area_ratio": float(np.ptp(slab[:, 0]) * np.ptp(slab[:, 2])) / (scale * scale + 1e-8),
    }


def _compute_space(ctx, space: str) -> dict[str, float | None]:
    scale = face_scale(ctx, space=space)
    pts = zone_points(ctx, "nose_bridge_tip", space=space)
    vals: dict[str, float | None] = {}
    if len(pts) >= 2:
        vals.update({
            "radix_depth_ratio": float(np.mean(pts[:, 2])) / scale,
            "radix_width_ratio": float(np.ptp(pts[:, 0])) / scale,
            "radix_height_ratio": float(np.ptp(pts[:, 1])) / scale,
            "upper_bridge_width_ratio": float(np.percentile(pts[:, 0], 90) - np.percentile(pts[:, 0], 10)) / scale,
            "upper_bridge_depth_span_ratio": float(np.ptp(pts[:, 2])) / scale,
            "upper_bridge_taper_index": float((np.ptp(pts[:, 0]) + 1e-8) / (np.ptp(pts[:, 1]) + 1e-8)),
            "nasal_dorsum_curve_proxy": float(np.std(pts[:, 2]) / (np.ptp(pts[:, 1]) + 1e-8)),
            "nasal_dorsum_convexity": float(np.mean(pts[:, 2]) - np.median(pts[:, 2])) / scale,
            "nose_bridge_torsion_proxy": float(np.corrcoef(pts[:, 0], pts[:, 2])[0, 1]) if len(pts) > 3 and np.std(pts[:, 0]) > 0 and np.std(pts[:, 2]) > 0 else 0.0,
        })
        for level in range(4):
            vals.update(_level_metrics(pts, level, scale))
    nasion, glabella = point(ctx, "nasion", space=space), point(ctx, "glabella", space=space)
    if nasion is not None and glabella is not None:
        vals["nasion_to_glabella_depth_drop"] = float((glabella[2] - nasion[2]) / scale)
    for side in ("L", "R"):
        inner = point(ctx, f"inner_canthus_{side}", space=space)
        if nasion is not None and inner is not None:
            vals[f"nasion_to_inner_canthus_depth_{side}"] = float((inner[2] - nasion[2]) / scale)
    pron, sub = point(ctx, "pronasale", space=space), point(ctx, "subnasale", space=space)
    if pron is not None and nasion is not None:
        vals["nasal_tip_projection"] = float((pron[2] - nasion[2]) / scale)
        d = distance(nasion, pron)
        vals["nose_root_to_tip_spline_length"] = None if d is None else d / scale
    if pron is not None and sub is not None:
        vals["nasal_tip_rotation"] = float((pron[1] - sub[1]) / scale)
    l, r = point(ctx, "inner_canthus_L", space=space), point(ctx, "inner_canthus_R", space=space)
    al, ar = point(ctx, "alar_L", space=space), point(ctx, "alar_R", space=space)
    if al is not None and ar is not None:
        vals["alar_base_width"] = (distance(al, ar) or 0.0) / scale
        vals["nostril_floor_width_proxy"] = vals["alar_base_width"]
    elif l is not None and r is not None:
        vals["alar_base_width"] = (distance(l, r) or 0.0) / scale
        vals["nostril_floor_width_proxy"] = vals["alar_base_width"]
    if l is not None and r is not None:
        vals["bridge_trapezoid_angle"] = float(np.degrees(np.arctan2(abs(l[1] - r[1]), abs(l[0] - r[0]) + 1e-8)))
    vals["nose_asymmetry_score"] = abs(float(vals.get("nasion_to_inner_canthus_depth_L") or 0.0) - float(vals.get("nasion_to_inner_canthus_depth_R") or 0.0))
    return vals


def compute(ctx, specs_):
    out = []
    spec_by = {s.name: s for s in specs_}
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
