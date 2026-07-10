from __future__ import annotations

import math
import numpy as np

from .catalog_specs import specs_for_module
from .common import emit, face_scale, zone_points, centroid
from .primitives import point, distance, angle, eye_points

IMPLEMENTATION = "existing_backend.py"


def specs():
    # Concrete F0 legacy metric names from full catalog. In real extraction these
    # are copied from legacy dicts; in isolated/new mode we compute compatible
    # fallback approximations so every F0 key has a compute path.
    return specs_for_module(IMPLEMENTATION, families={"F0"})


def _safe(v):
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except Exception:
        return None


def _eye_stats(ctx, side: str, scale: float) -> dict[str, float]:
    pts = eye_points(ctx, side)
    if len(pts) < 2:
        return {}
    width = float(np.ptp(pts[:, 0]))
    height = float(np.ptp(pts[:, 1]))
    depth = float(np.ptp(pts[:, 2]))
    upper = pts[pts[:, 1] <= np.median(pts[:, 1])]
    lower = pts[pts[:, 1] > np.median(pts[:, 1])]
    inner = point(ctx, f"inner_canthus_{side}")
    outer = point(ctx, f"outer_canthus_{side}")
    c = np.mean(pts, axis=0)
    return {
        "width": width / (scale + 1e-8),
        "height": height / (scale + 1e-8),
        "fissure_ratio": height / (width + 1e-8),
        "eye_aspect_ratio": height / (width + 1e-8),
        "apex_displacement": float((np.min(pts[:, 1]) - c[1]) / (scale + 1e-8)),
        "lower_flatness": float(np.std(lower[:, 1]) / (scale + 1e-8)) if len(lower) else 0.0,
        "upper_lid_residual": float(np.std(upper[:, 2]) / (scale + 1e-8)) if len(upper) else 0.0,
        "lower_lid_residual": float(np.std(lower[:, 2]) / (scale + 1e-8)) if len(lower) else 0.0,
        "opening_balance": float((len(upper) - len(lower)) / max(len(pts), 1)),
        "canthus_z_delta": float(((outer[2] - inner[2]) / (scale + 1e-8))) if inner is not None and outer is not None else 0.0,
        "medial_canthus_angle": angle(point(ctx, f"brow_ridge_{side}"), inner, c) or 0.0,
        "lateral_canthus_angle": angle(point(ctx, f"brow_ridge_{side}"), outer, c) or 0.0,
        "spherical_variance": depth / (width + 1e-8),
        "medial_opening": height / (scale + 1e-8),
        "lateral_opening": height / (scale + 1e-8),
    }


def _fallback_values(ctx) -> dict[str, float]:
    vals: dict[str, float] = {}
    scale = face_scale(ctx)
    fh = distance(point(ctx, "forehead"), point(ctx, "chin")) or scale
    fw = distance(point(ctx, "cheekbone_L"), point(ctx, "cheekbone_R")) or scale
    jaww = distance(point(ctx, "jaw_L"), point(ctx, "jaw_R")) or scale
    orbit_l = point(ctx, "orbit_L")
    orbit_r = point(ctx, "orbit_R")
    brow_l = point(ctx, "brow_ridge_L")
    brow_r = point(ctx, "brow_ridge_R")
    nose = point(ctx, "nasion")
    chin = point(ctx, "chin")
    sub = point(ctx, "subnasale")
    pron = point(ctx, "pronasale")

    vals.update({
        "cranial_face_index": fw / (fh + 1e-8),
        "cranial_proportion": fw / (fh + 1e-8),
        "jaw_width_ratio": jaww / (fw + 1e-8),
        "bigonial_width_ratio": jaww / (fw + 1e-8),
        "bigonial_width_ratio": jaww / (fw + 1e-8),
        "bigonial_width_ratio": jaww / (fw + 1e-8),
        "bigonial_width_ratio": jaww / (fw + 1e-8),
        "chin_projection_ratio": ((chin[2] - nose[2]) / (scale + 1e-8)) if chin is not None and nose is not None else 0.0,
        "chin_offset_asymmetry": abs(chin[0]) / (scale + 1e-8) if chin is not None else 0.0,
        "gnathion_midline_deviation_ratio": abs(chin[0]) / (scale + 1e-8) if chin is not None else 0.0,
        "forehead_slope_index": ((point(ctx, "forehead")[2] - nose[2]) / (abs(point(ctx, "forehead")[1] - nose[1]) + 1e-8)) if point(ctx, "forehead") is not None and nose is not None else 0.0,
        "nose_projection_ratio": ((pron[2] - nose[2]) / (scale + 1e-8)) if pron is not None and nose is not None else 0.0,
        "nose_width_ratio": float(np.ptp(zone_points(ctx, "nose")[:, 0]) / (scale + 1e-8)) if len(zone_points(ctx, "nose")) else 0.0,
        "nose_bridge_length_ratio": (distance(point(ctx, "glabella"), nose) or 0.0) / (scale + 1e-8) if nose is not None else 0.0,
        "nasion_zone_depth_ratio": nose[2] / (scale + 1e-8) if nose is not None else 0.0,
        "nasal_length_ratio": (distance(nose, pron) or 0.0) / (scale + 1e-8) if nose is not None else 0.0,
        "subnasale_projection_ratio": ((sub[2] - nose[2]) / (scale + 1e-8)) if sub is not None and nose is not None else 0.0,
    })
    glabella_pt = point(ctx, "forehead")
    if glabella_pt is not None and nose is not None:
        vals["nasal_frontal_index"] = float((glabella_pt[2] - nose[2]) / (fh + 1e-8))
        vals["nasofacial_angle_ratio"] = float((pron[2] - nose[2]) / (fh + 1e-8)) if pron is not None else 0.0
        fsi = float(vals.get("forehead_slope_index") or 0.0)
        vals["glabella_nasion_projection_angle"] = fsi * 90.0
    else:
        vals["nasal_frontal_index"] = 0.0
        vals["nasofacial_angle_ratio"] = 0.0
        vals["glabella_nasion_projection_angle"] = 0.0

    # Orbit and interocular approximations.
    for side in ("L", "R"):
        op = zone_points(ctx, f"orbit_{side}")
        if len(op):
            vals[f"orbit_depth_{side}_ratio"] = float(np.ptp(op[:, 2]) / (scale + 1e-8))
            vals[f"orbit_width_{side}_ratio"] = float(np.ptp(op[:, 0]) / (scale + 1e-8))
            vals[f"orbit_fossa_spread_{side}"] = float(np.std(op[:, 2]) / (scale + 1e-8))
            vals[f"brow_ridge_projection_{side}_ratio"] = ((point(ctx, f"brow_ridge_{side}")[2] - np.mean(op[:, 2])) / (scale + 1e-8)) if point(ctx, f"brow_ridge_{side}") is not None else 0.0
            vals[f"palpebral_fissure_length_{side}_ratio"] = float(np.ptp(eye_points(ctx, side)[:, 0]) / (scale + 1e-8)) if len(eye_points(ctx, side)) else 0.0
            vals[f"ramus_height_{side}_ratio"] = float(np.ptp(zone_points(ctx, f"jaw_{side}")[:, 1]) / (scale + 1e-8)) if len(zone_points(ctx, f"jaw_{side}")) else 0.0
            vals[f"mandibular_body_length_{side}_ratio"] = (distance(point(ctx, f"gonion_{side}"), chin) or 0.0) / (scale + 1e-8) if chin is not None else 0.0
            vals[f"gonial_angle_{side}"] = angle(point(ctx, f"cheekbone_{side}"), point(ctx, f"gonion_{side}"), chin) or 0.0
            vals[f"periocular_gonial_angle_{side}"] = angle(point(ctx, f"pupil_{side}"), point(ctx, f"gonion_{side}"), chin) or 0.0
            vals[f"temporal_depth_{side}_ratio"] = (point(ctx, f"temporal_{side}")[2] / (scale + 1e-8)) if point(ctx, f"temporal_{side}") is not None else 0.0
            prefix = "left" if side == "L" else "right"
            cheek = zone_points(ctx, f"cheekbone_{side}")
            if len(cheek):
                vals[f"{prefix}_cheek_slope"] = float(np.std(cheek[:, 2]) / (np.ptp(cheek[:, 1]) + 1e-8))
                vals[f"{prefix}_cheek_z_var"] = float(np.var(cheek[:, 2]) / (scale * scale + 1e-8))
                vals[f"{prefix}_cliff_gradient"] = float(np.ptp(cheek[:, 2]) / (np.ptp(cheek[:, 0]) + np.ptp(cheek[:, 1]) + 1e-8))
    if orbit_l is not None and orbit_r is not None:
        vals.update({
            "interorbital_ratio": (distance(orbit_l, orbit_r) or 0.0) / (scale + 1e-8),
            "intercanthal_width_ratio": (distance(point(ctx, "inner_canthus_L"), point(ctx, "inner_canthus_R")) or 0.0) / (scale + 1e-8),
            "orbit_centroid_ratio": (distance(orbit_l, orbit_r) or 0.0) / (scale + 1e-8),
            "orbit_depth_asymmetry_ratio": abs(vals.get("orbit_depth_L_ratio", 0.0) - vals.get("orbit_depth_R_ratio", 0.0)),
            "orbit_vertical_signed_ratio": (orbit_l[1] - orbit_r[1]) / (scale + 1e-8),
            "orbit_vertical_asymmetry_ratio": abs(orbit_l[1] - orbit_r[1]) / (scale + 1e-8),
            "orbital_asymmetry_index": abs(vals.get("orbit_depth_L_ratio", 0.0) - vals.get("orbit_depth_R_ratio", 0.0)),
            "orbital_perimeter_symmetry": abs(vals.get("orbit_width_L_ratio", 0.0) - vals.get("orbit_width_R_ratio", 0.0)),
            "mean_orbital_depth": (vals.get("orbit_depth_L_ratio", 0.0) + vals.get("orbit_depth_R_ratio", 0.0)) / 2.0,
            "depth_asymmetry": abs(vals.get("orbit_depth_L_ratio", 0.0) - vals.get("orbit_depth_R_ratio", 0.0)),
            "bilateral_center_depth_asymmetry": abs(orbit_l[2] - orbit_r[2]) / (scale + 1e-8),
        })
    vals["orbit_skull_ratio"] = vals.get("mean_orbital_depth", 0.0) / (vals.get("cranial_face_index", 1.0) + 1e-8)
    vals["midface_depth_index"] = vals.get("mean_orbital_depth", 0.0) + vals.get("nose_projection_ratio", 0.0)
    vals["midface_compactness"] = vals["midface_depth_index"] / (fh / (fw + 1e-8) + 1e-8)
    vals["skull_depth_asymmetry_index"] = vals.get("depth_asymmetry", 0.0)
    jaw_l_pt = point(ctx, "jaw_L")
    jaw_r_pt = point(ctx, "jaw_R")
    vals["gonial_width_asymmetry"] = abs(jaw_l_pt[2] - jaw_r_pt[2]) / (scale + 1e-8) if jaw_l_pt is not None and jaw_r_pt is not None else 0.0
    vals["palpebral_fissure_asymmetry_ratio"] = abs(vals.get("palpebral_fissure_length_L_ratio", 0.0) - vals.get("palpebral_fissure_length_R_ratio", 0.0))
    vals["mandibular_ramus_length"] = (vals.get("ramus_height_L_ratio", 0.0) + vals.get("ramus_height_R_ratio", 0.0)) / 2.0
    vals["brow_asymmetry_deg"] = angle(brow_l, nose, brow_r) or 0.0
    vals["canthal_tilt_mean_deg"] = 0.0
    vals["canthal_tilt_asymmetry_deg"] = 0.0

    # Periocular approximations.
    for side, prefix in (("L", "left"), ("R", "right")):
        st = _eye_stats(ctx, side, scale)
        for k, v in st.items():
            if k == "eye_aspect_ratio":
                vals[f"eye_aspect_ratio_{side}"] = v
            else:
                vals[f"{prefix}_{k}"] = v
        vals[f"canthal_tilt_{side}"] = st.get("canthus_z_delta", 0.0)
        vals[f"canthal_tilt_3d_{side}"] = st.get("canthus_z_delta", 0.0)
    for key in ["fissure_ratio", "spherical_variance", "cliff_gradient"]:
        vals[f"mean_{key}"] = (vals.get(f"left_{key}", 0.0) + vals.get(f"right_{key}", 0.0)) / 2.0
    vals["mean_cliff_gradient"] = vals.get("mean_cliff_gradient", 0.0)

    # Texture legacy: omit zero placeholders (removed glcm fake metrics).
    for k in [
        "texture_baseline_shift",
        "texture_lbp_complexity",
        "texture_lbp_uniformity",
        "texture_specular_gloss",
        "texture_pore_density",
        "texture_gabor_std",
        "texture_global_smoothness",
    ]:
        if k in vals:
            continue
    if "texture_lbp_entropy" in vals and "texture_lbp_complexity" not in vals:
        vals["texture_lbp_complexity"] = vals["texture_lbp_entropy"]
    return vals


def compute(ctx, specs_):
    out = []
    merged = {}
    merged.update(ctx.geometry_metrics or {})
    merged.update(ctx.periocular_metrics or {})
    for k, v in (ctx.texture_forensics or {}).items():
        if isinstance(v, (int, float)):
            merged[f"texture_{k}" if not str(k).startswith("texture_") else str(k)] = v
    for k, v in (ctx.texture_profile or {}).items():
        if isinstance(v, (int, float)):
            merged[f"texture_{k}" if not str(k).startswith("texture_") else str(k)] = v
    fallback = _fallback_values(ctx)
    spec_by = {s.name: s for s in specs_}
    for name, spec in spec_by.items():
        val = merged.get(name)
        from_fallback = False
        if val is None:
            val = fallback.get(name)
            from_fallback = val is not None
        if val is None:
            continue
        if spec.group == "texture" and name not in merged:
            continue
        from .policy import is_diagnostic_texture_key

        conf = 1.0 if name in merged else 0.45
        if is_diagnostic_texture_key(name):
            conf = min(conf, 0.30)
        if from_fallback and float(val) == 0.0:
            continue
        mv = emit(spec, val, confidence=conf)
        if mv:
            out.append(mv)
    return out
