from __future__ import annotations

import numpy as np

from .catalog_specs import specs_for_module
from .common import emit, face_scale, zone_points
from .cross_section_utils import plane_mesh_intersections, section_metrics
from .mid_sagittal import estimate_mid_sagittal_plane
from .primitives import angle, point, distance

IMPLEMENTATION = "cross_sections.py"


def specs():
    return specs_for_module(IMPLEMENTATION, families={"F5"})


def _verts(ctx, space: str):
    if space == "shape_neutral":
        return ctx.vertices_shape_neutral
    if space == "raw":
        return ctx.vertices_raw
    return ctx.vertices_canon


def _midline_fit_slope(mid_pts: np.ndarray, y_lo: float, y_hi: float) -> float | None:
    if mid_pts is None or mid_pts.shape[0] < 4:
        return None
    y0, y1 = (min(y_lo, y_hi), max(y_lo, y_hi))
    seg = mid_pts[(mid_pts[:, 1] >= y0) & (mid_pts[:, 1] <= y1)]
    if seg.shape[0] < 3:
        return None
    y = seg[:, 1]
    z = seg[:, 2]
    a, b = np.polyfit(y, z, 1)
    return float(a)


def _horizontal_section(vertices: np.ndarray, triangles: np.ndarray, y: float, scale: float):
    pts = plane_mesh_intersections(vertices, triangles, np.array([0.0, y, 0.0]), np.array([0.0, 1.0, 0.0]))
    sm = section_metrics(pts, axes=(0, 2))
    if not sm:
        return {}
    return {
        "width_ratio": sm.get("width", 0.0) / scale,
        "depth_ratio": sm.get("depth", 0.0) / scale,
        "area_ratio": sm.get("bbox_area", 0.0) / (scale * scale + 1e-8),
        "curve_ratio": sm.get("curve_length", 0.0) / scale,
        "curvature_ratio": sm.get("residual_p95", 0.0) / scale,
    }


def _compute_space(ctx, space: str) -> dict[str, float | None]:
    vertices = _verts(ctx, space)
    if vertices is None or len(vertices) == 0:
        return {}
    scale = face_scale(ctx, space=space)
    vals: dict[str, float | None] = {}
    plane_point, plane_normal, plane_conf = estimate_mid_sagittal_plane(vertices, ctx.macro_indices)
    vals.update({
        "mid_sagittal_plane_confidence": plane_conf,
        "mid_sagittal_plane_normal_x": float(plane_normal[0]),
        "mid_sagittal_plane_normal_y": float(plane_normal[1]),
        "mid_sagittal_plane_normal_z": float(plane_normal[2]),
        "mid_sagittal_plane_offset_ratio": float(np.dot(plane_point - np.mean(vertices, axis=0), plane_normal) / scale),
    })

    mid_pts = plane_mesh_intersections(vertices, ctx.triangles, plane_point, plane_normal)
    sm = section_metrics(mid_pts, axes=(1, 2))
    if sm:
        vals.update({
            "real_midline_section_depth_ratio": sm.get("depth", 0.0) / scale,
            "real_midline_section_curve_ratio": sm.get("curve_length", 0.0) / scale,
            "real_midline_section_residual_p95_ratio": sm.get("residual_p95", 0.0) / scale,
            "real_midline_section_convexity_ratio": sm.get("residual_std", 0.0) / scale,
            "real_midline_section_concavity_ratio": sm.get("residual_p95", 0.0) / scale,
        })

    forehead = point(ctx, "forehead", space=space)
    nasion = point(ctx, "nasion", space=space)
    pron = point(ctx, "pronasale", space=space)
    sub = point(ctx, "subnasale", space=space)
    chin = point(ctx, "chin", space=space)
    if forehead is not None and nasion is not None:
        vals["real_midline_section_forehead_slope"] = float((forehead[2] - nasion[2]) / (abs(forehead[1] - nasion[1]) + 1e-8))
    if mid_pts is not None and len(mid_pts) >= 4 and nasion is not None and pron is not None:
        fit_slope = _midline_fit_slope(mid_pts, float(nasion[1]), float(pron[1]))
        if fit_slope is not None:
            vals["real_midline_section_nasal_dorsum_slope"] = fit_slope
        else:
            vals["real_midline_section_nasal_dorsum_slope"] = float((pron[2] - nasion[2]) / (abs(pron[1] - nasion[1]) + 1e-8))
    elif nasion is not None and pron is not None:
        vals["real_midline_section_nasal_dorsum_slope"] = float((pron[2] - nasion[2]) / (abs(pron[1] - nasion[1]) + 1e-8))
    if chin is not None and nasion is not None:
        vals["real_midline_section_chin_projection_ratio"] = float((chin[2] - nasion[2]) / scale)
    if chin is not None and sub is not None:
        vals["real_midline_section_mentolabial_depth_ratio"] = float((chin[2] - sub[2]) / scale)
    if forehead is not None and sub is not None and chin is not None:
        vals["real_midline_section_facial_convexity_angle"] = angle(forehead, sub, chin)
        if mid_pts is not None and len(mid_pts) >= 6:
            y_f, y_s, y_c = float(forehead[1]), float(sub[1]), float(chin[1])
            def _nearest_y(y_tgt: float) -> np.ndarray | None:
                idx = int(np.argmin(np.abs(mid_pts[:, 1] - y_tgt)))
                return mid_pts[idx]
            fp, sp, cp = _nearest_y(y_f), _nearest_y(y_s), _nearest_y(y_c)
            if fp is not None and sp is not None and cp is not None:
                ang = angle(fp, sp, cp)
                if ang is not None:
                    vals["real_midline_section_facial_convexity_angle"] = ang

    orbit_l = point(ctx, "orbit_L", space=space)
    orbit_r = point(ctx, "orbit_R", space=space)
    y_orbit = float(np.mean([p[1] for p in (orbit_l, orbit_r) if p is not None])) if (orbit_l is not None or orbit_r is not None) else float(np.median(vertices[:, 1]))
    hs = _horizontal_section(vertices, ctx.triangles, y_orbit, scale)
    vals.update({f"orbit_level_section_{k}": val for k, val in hs.items() if k in {"width_ratio", "depth_ratio", "area_ratio", "curve_ratio"}})
    vals["orbit_level_section_asymmetry_ratio"] = abs(orbit_l[2] - orbit_r[2]) / scale if orbit_l is not None and orbit_r is not None else None

    brow_pts = [p for p in [point(ctx, "brow_ridge_L", space=space), point(ctx, "brow_ridge_R", space=space)] if p is not None]
    if brow_pts:
        hs = _horizontal_section(vertices, ctx.triangles, float(np.mean([p[1] for p in brow_pts])), scale)
        vals.update({f"brow_level_section_{k}": val for k, val in hs.items() if k in {"width_ratio", "depth_ratio", "curvature_ratio"}})

    if nasion is not None:
        hs = _horizontal_section(vertices, ctx.triangles, float(nasion[1]), scale)
        vals.update({
            "nasion_level_section_bridge_width_ratio": hs.get("width_ratio"),
            "nasion_level_section_bridge_depth_ratio": hs.get("depth_ratio"),
            "nasion_level_section_bridge_area_ratio": hs.get("area_ratio"),
        })
    if sub is not None:
        hs = _horizontal_section(vertices, ctx.triangles, float(sub[1]), scale)
        vals.update({
            "subnasale_level_section_width_ratio": hs.get("width_ratio"),
            "subnasale_level_section_depth_ratio": hs.get("depth_ratio"),
            "subnasale_level_section_area_ratio": hs.get("area_ratio"),
        })

    for side in ("L", "R"):
        jaw = zone_points(ctx, f"jaw_{side}", space=space)
        if len(jaw) > 1:
            p = jaw[np.argsort(jaw[:, 1])]
            vals[f"jawline_section_{side}_arc_ratio"] = float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)) / scale)
            vals[f"jawline_section_{side}_depth_ratio"] = float(np.ptp(jaw[:, 2]) / scale)
            vals[f"jawline_section_{side}_curvature_ratio"] = float(np.std(jaw[:, 2]) / (np.ptp(jaw[:, 1]) + 1e-8))
        zyg = point(ctx, f"cheekbone_{side}", space=space)
        temp = point(ctx, f"temporal_{side}", space=space)
        orb = point(ctx, f"orbit_{side}", space=space)
        if zyg is not None and temp is not None:
            vals[f"zygoma_oblique_section_{side}_step_ratio"] = abs(temp[2] - zyg[2]) / scale
            vals[f"zygoma_oblique_section_{side}_convexity_ratio"] = abs(zyg[2] - (temp[2] + (orb[2] if orb is not None else zyg[2])) / 2.0) / scale
            vals[f"zygoma_oblique_section_{side}_cliff_index"] = abs(temp[2] - zyg[2]) / ((distance(temp, zyg) or 0.0) + 1e-8)
        op = zone_points(ctx, f"orbit_{side}", space=space)
        if len(op) > 2:
            vals[f"orbital_bowl_depth_section_{side}_mean_depth"] = float(np.mean(op[:, 2]) / scale)
            vals[f"orbital_bowl_depth_section_{side}_p95_depth"] = float(np.percentile(op[:, 2], 95) / scale)
            vals[f"orbital_bowl_depth_section_{side}_grid_variance"] = float(np.var(op[:, 2]) / (scale * scale + 1e-8))

    bridge = zone_points(ctx, "nose_bridge_tip", space=space)
    if len(bridge) > 2:
        qs = np.quantile(bridge[:, 1], [0, .25, .5, .75, 1])
        for level in range(4):
            slab = bridge[(bridge[:, 1] >= qs[level]) & (bridge[:, 1] <= qs[level+1])]
            if len(slab) < 2:
                slab = bridge
            vals[f"bridge_level_{level}_section_width_ratio"] = float(np.ptp(slab[:, 0]) / scale)
            vals[f"bridge_level_{level}_section_depth_ratio"] = float(np.ptp(slab[:, 2]) / scale)
            vals[f"bridge_level_{level}_section_area_ratio"] = float(np.ptp(slab[:, 0]) * np.ptp(slab[:, 2]) / (scale * scale + 1e-8))
    return vals


def compute(ctx, specs_):
    out = []
    cache: dict[str, dict[str, float | None]] = {}
    for spec in specs_:
        for space in spec.source_spaces:
            if space == "shape_neutral" and ctx.vertices_shape_neutral is None:
                continue
            if space not in {"canon_bucket", "shape_neutral", "raw"}:
                continue
            if space not in cache:
                cache[space] = _compute_space(ctx, space)
            val = cache[space].get(spec.name)
            if (mv := emit(spec, val, confidence=0.70, source_space=space)):
                out.append(mv)
    return out
