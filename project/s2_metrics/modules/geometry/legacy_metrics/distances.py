from __future__ import annotations

from .catalog_specs import specs_for_module
from .common import emit, face_scale
from .primitives import distance, point

IMPLEMENTATION = "distances.py"


def specs():
    return specs_for_module(IMPLEMENTATION, families={"F1"})


def _split_name(name: str):
    for suffix in ("_signed_depth_delta_ratio", "_vertical_gap_ratio", "_lateral_gap_ratio", "_distance_ratio"):
        if name.endswith(suffix):
            core = name[: -len(suffix)]
            # Greedy split by known separators: catalog names are A_B_suffix and A/B may contain underscores.
            candidates = [
                "inner_canthus_L", "inner_canthus_R", "outer_canthus_L", "outer_canthus_R",
                "nose_bridge_tip", "brow_ridge_L", "brow_ridge_R", "cheekbone_L", "cheekbone_R",
                "temporal_L", "temporal_R", "gonion_L", "gonion_R", "zygoma_L", "zygoma_R",
                "orbit_L", "orbit_R", "jaw_L", "jaw_R", "pupil_L", "pupil_R", "forehead",
                "nasion", "glabella", "chin", "pronasale", "subnasale",
            ]
            for a in sorted(candidates, key=len, reverse=True):
                pref = a + "_"
                if core.startswith(pref):
                    return a, core[len(pref):], suffix[1:]
    return None, None, None


def compute(ctx, specs_):
    out = []
    scale = face_scale(ctx)
    for spec in specs_:
        a, b, kind = _split_name(spec.name)
        if not a or not b:
            continue
        for space in spec.source_spaces:
            if space == "shape_neutral" and ctx.vertices_shape_neutral is None:
                continue
            if space not in {"canon_bucket", "shape_neutral", "raw"}:
                continue
            scale_s = face_scale(ctx, space=space)
            pa, pb = point(ctx, a, space=space), point(ctx, b, space=space)
            if pa is None or pb is None:
                continue
            if kind == "distance_ratio":
                val = distance(pa, pb)
                val = None if val is None else val / scale_s
            elif kind == "signed_depth_delta_ratio":
                val = (pb[2] - pa[2]) / scale_s
            elif kind == "vertical_gap_ratio":
                val = abs(pb[1] - pa[1]) / scale_s
            elif kind == "lateral_gap_ratio":
                val = abs(pb[0] - pa[0]) / scale_s
            else:
                val = None
            mv = emit(spec, val, confidence=0.8, source_space=space)
            if mv:
                out.append(mv)
    return out
