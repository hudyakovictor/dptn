from __future__ import annotations

from .catalog_specs import specs_for_module
from .common import emit, face_scale
from .primitives import point, tri_metrics

IMPLEMENTATION = "triangles.py"

TRI_POINTS = {
    "midface_triangle": ("zygoma_L", "zygoma_R", "subnasale"),
    "orbital_bridge_triangle": ("orbit_L", "orbit_R", "nasion"),
    "jaw_triangle": ("gonion_L", "gonion_R", "chin"),
    "brow_nose_triangle": ("brow_ridge_L", "brow_ridge_R", "nasion"),
    "zygoma_orbit_nasion_triangle_L": ("zygoma_L", "orbit_L", "nasion"),
    "zygoma_orbit_nasion_triangle_R": ("zygoma_R", "orbit_R", "nasion"),
    "pupil_inner_nasion_triangle_L": ("pupil_L", "inner_canthus_L", "nasion"),
    "pupil_inner_nasion_triangle_R": ("pupil_R", "inner_canthus_R", "nasion"),
    "inner_outer_apex_triangle_L": ("inner_canthus_L", "outer_canthus_L", "lid_apex_L"),
    "inner_outer_apex_triangle_R": ("inner_canthus_R", "outer_canthus_R", "lid_apex_R"),
    "gonion_pupil_outer_canthus_triangle_L": ("gonion_L", "pupil_L", "outer_canthus_L"),
    "gonion_pupil_outer_canthus_triangle_R": ("gonion_R", "pupil_R", "outer_canthus_R"),
    "glabella_nasion_pupil_triangle_L": ("glabella", "nasion", "pupil_L"),
    "glabella_nasion_pupil_triangle_R": ("glabella", "nasion", "pupil_R"),
    "zygoma_nasion_gonion_triangle_L": ("zygoma_L", "nasion", "gonion_L"),
    "zygoma_nasion_gonion_triangle_R": ("zygoma_R", "nasion", "gonion_R"),
    "nose_triangle": ("inner_canthus_L", "inner_canthus_R", "pronasale"),
    "chin_nasion_subnasale_triangle": ("chin", "nasion", "subnasale"),
}

METRIC_SUFFIXES = [
    "area_ratio", "perimeter_ratio", "edge_ab_ratio", "edge_bc_ratio", "edge_ca_ratio",
    "apex_angle_deg", "base_to_height_ratio", "plane_normal_x", "plane_normal_y", "plane_normal_z",
    "normal_to_face_plane_angle_deg", "signed_apex_depth_ratio",
]


def specs():
    return specs_for_module(IMPLEMENTATION, families={"F3"})


def _parse(name: str):
    for tri in sorted(TRI_POINTS, key=len, reverse=True):
        pref = tri + "_"
        if name.startswith(pref):
            return tri, name[len(pref):]
    return None, None


def compute(ctx, specs_):
    out = []
    cache = {}
    for spec in specs_:
        tri, suffix = _parse(spec.name)
        if tri is None or suffix not in METRIC_SUFFIXES:
            continue
        for space in spec.source_spaces:
            if space == "shape_neutral" and ctx.vertices_shape_neutral is None:
                continue
            if space not in {"canon_bucket", "shape_neutral", "raw"}:
                continue
            key = (tri, space)
            if key not in cache:
                scale = face_scale(ctx, space=space)
                pts = TRI_POINTS[tri]
                cache[key] = tri_metrics(point(ctx, pts[0], space=space), point(ctx, pts[1], space=space), point(ctx, pts[2], space=space), scale)
            val = cache[key].get(suffix)
            mv = emit(spec, val, confidence=0.75, source_space=space)
            if mv:
                out.append(mv)
    return out
