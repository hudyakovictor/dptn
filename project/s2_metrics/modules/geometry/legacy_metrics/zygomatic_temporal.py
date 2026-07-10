from __future__ import annotations

import numpy as np

from .catalog_specs import specs_for_module
from .common import centroid, distance, emit, face_scale, zone_points
from .topology_utils import region_surface_area

IMPLEMENTATION = "zygomatic_temporal.py"


def specs():
    return specs_for_module(IMPLEMENTATION, families={"F_zyg_temp"})


def _compute_side(ctx, side: str, space: str) -> dict[str, float | None]:
    zyg, temp, orb, jaw = f"cheekbone_{side}", f"temporal_{side}", f"orbit_{side}", f"jaw_{side}"
    scale = face_scale(ctx, space=space)
    zp, tp = zone_points(ctx, zyg, space=space), zone_points(ctx, temp, space=space)
    vals: dict[str, float | None] = {}
    zc, tc, oc, jc = centroid(ctx, zyg, space=space), centroid(ctx, temp, space=space), centroid(ctx, orb, space=space), centroid(ctx, jaw, space=space)
    verts = ctx.vertices_shape_neutral if space == "shape_neutral" else ctx.vertices_raw if space == "raw" else ctx.vertices_canon
    if len(zp):
        vals.update({
            f"{side}_malar_peak_depth_ratio": float(np.max(zp[:, 2]) / scale),
            f"{side}_malar_peak_height_ratio": float(np.ptp(zp[:, 1]) / scale),
            f"{side}_submalar_hollow_proxy": float(np.std(zp[:, 2]) / scale),
        })
        if zc is not None and oc is not None:
            vals[f"{side}_malar_peak_relative_to_pupil"] = (distance(zc, oc) or 0.0) / scale
        nose = centroid(ctx, "nose_bridge_tip", space=space)
        if zc is not None and nose is not None:
            vals[f"{side}_malar_peak_relative_to_nose_base"] = (distance(zc, nose) or 0.0) / scale
        if zc is not None and jc is not None:
            vals[f"{side}_malar_peak_relative_to_gonion"] = (distance(zc, jc) or 0.0) / scale
    if len(tp):
        vals[f"{side}_temporal_concavity_proxy"] = float(np.std(tp[:, 2]) / scale)
        surf = region_surface_area(verts, ctx.triangles, ctx.macro_indices.get(temp, [])) if verts is not None else None
        bbox_area = float(np.ptp(tp[:,0])*np.ptp(tp[:,1]) + np.ptp(tp[:,0])*np.ptp(tp[:,2]) + np.ptp(tp[:,1])*np.ptp(tp[:,2]))
        vals[f"{side}_temporal_concavity_area"] = float((bbox_area if surf is None else surf) / (scale * scale + 1e-8))
        vals[f"{side}_temporal_concavity_volume"] = float(np.ptp(tp[:, 0]) * np.ptp(tp[:, 1]) * np.ptp(tp[:, 2]) / (scale ** 3 + 1e-8))
        if zc is not None and tc is not None:
            vals[f"{side}_temporal_to_zygoma_step_ratio"] = float(abs(tc[2] - zc[2]) / scale)
        if oc is not None and tc is not None:
            vals[f"{side}_temporal_to_orbit_depth_gradient"] = float(abs(tc[2] - oc[2]) / ((distance(tc, oc) or 0.0) + 1e-8))
        vals[f"{side}_temporal_edge_cliff_index"] = float(np.ptp(tp[:, 2]) / (np.ptp(tp[:, 0]) + np.ptp(tp[:, 1]) + 1e-8))
    return vals


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
