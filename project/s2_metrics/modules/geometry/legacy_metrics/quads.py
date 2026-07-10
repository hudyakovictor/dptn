from __future__ import annotations

from .catalog_specs import specs_for_module
from .common import emit, face_scale
from .primitives import point, quad_metrics

IMPLEMENTATION = "quads.py"

QUAD_POINTS = {
    "brow_orbit_quad": ("brow_ridge_L", "brow_ridge_R", "orbit_R", "orbit_L"),
    "orbit_depth_width_quad_L": ("inner_canthus_L", "outer_canthus_L", "brow_ridge_L", "cheekbone_L"),
    "orbit_depth_width_quad_R": ("inner_canthus_R", "outer_canthus_R", "brow_ridge_R", "cheekbone_R"),
    "fissure_jaw_quad_L": ("inner_canthus_L", "outer_canthus_L", "gonion_L", "chin"),
    "fissure_jaw_quad_R": ("inner_canthus_R", "outer_canthus_R", "gonion_R", "chin"),
    "cranial_face_orbit_quad": ("forehead", "chin", "orbit_R", "orbit_L"),
    "brow_lid_rectangle_L": ("brow_ridge_L", "outer_canthus_L", "lid_apex_L", "inner_canthus_L"),
    "brow_lid_rectangle_R": ("brow_ridge_R", "outer_canthus_R", "lid_apex_R", "inner_canthus_R"),
    "temporal_zygoma_orbit_quad_L": ("temporal_L", "cheekbone_L", "orbit_L", "brow_ridge_L"),
    "temporal_zygoma_orbit_quad_R": ("temporal_R", "cheekbone_R", "orbit_R", "brow_ridge_R"),
    "bridge_pupil_quad": ("inner_canthus_L", "inner_canthus_R", "pupil_R", "pupil_L"),
    "upper_bridge_trapezoid": ("inner_canthus_L", "inner_canthus_R", "brow_ridge_R", "brow_ridge_L"),
}

SUFFIXES = ["area_ratio", "diagonal_ratio", "aspect_ratio", "twist_angle_deg", "nonplanarity_ratio", "normal_x", "normal_y", "normal_z", "signed_depth_imbalance_ratio"]


def specs():
    return specs_for_module(IMPLEMENTATION, families={"F4"})


def _parse(name: str):
    for q in sorted(QUAD_POINTS, key=len, reverse=True):
        pref = q + "_"
        if name.startswith(pref):
            return q, name[len(pref):]
    return None, None


def compute(ctx, specs_):
    out = []
    cache = {}
    for spec in specs_:
        q, suffix = _parse(spec.name)
        if q is None or suffix not in SUFFIXES:
            continue
        for space in spec.source_spaces:
            if space == "shape_neutral" and ctx.vertices_shape_neutral is None:
                continue
            if space not in {"canon_bucket", "shape_neutral", "raw"}:
                continue
            key = (q, space)
            if key not in cache:
                scale = face_scale(ctx, space=space)
                pts = [point(ctx, p, space=space) for p in QUAD_POINTS[q]]
                cache[key] = quad_metrics(pts, scale)
            val = cache[key].get(suffix)
            mv = emit(spec, val, confidence=0.7, source_space=space)
            if mv:
                out.append(mv)
    return out
