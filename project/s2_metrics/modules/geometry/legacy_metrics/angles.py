from __future__ import annotations

from .catalog_specs import specs_for_module
from .common import emit
from .primitives import angle, point

IMPLEMENTATION = "angles.py"

def _resolve_angle_point(name: str, spec_name: str, pose_bucket: str) -> str:
    pb = pose_bucket or "frontal"
    if name == "gonion_L" and "right" in pb:
        return "gonion_R"
    if name == "gonion_R" and pb.startswith("left"):
        return "gonion_L"
    return name


ANGLE_POINTS = {
    "forehead_nasion_pronasale_angle": ("forehead", "nasion", "pronasale"),
    "subnasale_labrale_pogonion_angle": ("subnasale", "labrale", "pogonion"),
    "nasofrontal_angle": ("glabella", "nasion", "pronasale"),
    "nasolabial_angle": ("pronasale", "subnasale", "labrale"),
    "radix_angle": ("glabella", "nasion", "nose_bridge_tip"),
    "nasal_tip_rotation_angle": ("nasion", "pronasale", "subnasale"),
    "mandibular_plane_angle_L": ("gonion_L", "chin", "subnasale"),
    "mandibular_plane_angle_R": ("gonion_R", "chin", "subnasale"),
    "brow_apex_lateral_canthus_angle_L": ("brow_ridge_L", "lid_apex_L", "outer_canthus_L"),
    "brow_apex_lateral_canthus_angle_R": ("brow_ridge_R", "lid_apex_R", "outer_canthus_R"),
    "orbital_tilt_3d_L": ("inner_canthus_L", "orbit_L", "outer_canthus_L"),
    "orbital_tilt_3d_R": ("inner_canthus_R", "orbit_R", "outer_canthus_R"),
    "orbital_floor_slope_L": ("inner_canthus_L", "orbit_L", "cheekbone_L"),
    "orbital_floor_slope_R": ("inner_canthus_R", "orbit_R", "cheekbone_R"),
    "orbital_roof_slope_L": ("inner_canthus_L", "brow_ridge_L", "outer_canthus_L"),
    "orbital_roof_slope_R": ("inner_canthus_R", "brow_ridge_R", "outer_canthus_R"),
    "zygoma_to_orbit_transition_angle_L": ("zygoma_L", "orbit_L", "nasion"),
    "zygoma_to_orbit_transition_angle_R": ("zygoma_R", "orbit_R", "nasion"),
    "nasal_axis_vs_face_axis": ("forehead", "nasion", "chin"),
    "mandible_axis_vs_face_axis": ("jaw_L", "chin", "jaw_R"),
}


def specs():
    return specs_for_module(IMPLEMENTATION, families={"F2"})


def compute(ctx, specs_):
    out = []
    for spec in specs_:
        pts = ANGLE_POINTS.get(spec.name)
        if not pts:
            continue
        for space in spec.source_spaces:
            if space == "shape_neutral" and ctx.vertices_shape_neutral is None:
                continue
            if space not in {"canon_bucket", "shape_neutral", "raw"}:
                continue
            p0 = _resolve_angle_point(pts[0], spec.name, ctx.pose_bucket)
            p1 = _resolve_angle_point(pts[1], spec.name, ctx.pose_bucket)
            p2 = _resolve_angle_point(pts[2], spec.name, ctx.pose_bucket)
            val = angle(point(ctx, p0, space=space), point(ctx, p1, space=space), point(ctx, p2, space=space))
            mv = emit(spec, val, confidence=0.75, source_space=space)
            if mv:
                out.append(mv)
    return out
