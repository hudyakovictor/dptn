from __future__ import annotations

import numpy as np

from .catalog_specs import specs_for_module
from .common import emit, face_scale
from .primitives import eye_points, point, distance

IMPLEMENTATION = "periocular_lid.py"


def specs():
    return specs_for_module(IMPLEMENTATION, families={"F_periocular"})


def _arc_length_sorted_x(pts: np.ndarray) -> float:
    if pts.shape[0] < 2:
        return 0.0
    p = pts[np.argsort(pts[:, 0])]
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


def _compute_side(ctx, side: str, space: str) -> dict[str, float | None]:
    scale = face_scale(ctx, space=space)
    pts = eye_points(ctx, side, space=space)
    if len(pts) < 3:
        return {}
    width, height, depth = float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])), float(np.ptp(pts[:, 2]))
    y_med = float(np.median(pts[:, 1]))
    upper = pts[pts[:, 1] <= y_med]
    lower = pts[pts[:, 1] > y_med]
    upper_arc = _arc_length_sorted_x(upper)
    lower_arc = _arc_length_sorted_x(lower)
    inner, outer = point(ctx, f"inner_canthus_{side}", space=space), point(ctx, f"outer_canthus_{side}", space=space)
    orbit, brow, cheek = point(ctx, f"orbit_{side}", space=space), point(ctx, f"brow_ridge_{side}", space=space), point(ctx, f"cheekbone_{side}", space=space)
    vals: dict[str, float | None] = {
        f"{side}_upper_lid_arc_length": upper_arc / (2 * scale),
        f"{side}_lower_lid_arc_length": lower_arc / (2 * scale),
        f"{side}_lid_surface_curvature": float(np.std(pts[:, 2]) / (width + 1e-8)),
        f"{side}_upper_lid_depth_profile": float(np.mean(upper[:, 2]) / scale) if len(upper) else None,
        f"{side}_lower_lid_depth_profile": float(np.mean(lower[:, 2]) / scale) if len(lower) else None,
        f"{side}_lid_fold_depth_proxy": float((np.mean(upper[:, 2]) - np.mean(lower[:, 2])) / scale) if len(upper) and len(lower) else None,
        f"{side}_canthus_depth": float(depth / scale),
        f"{side}_medial_canthus_depth": float((inner[2] - np.mean(pts[:, 2])) / scale) if inner is not None else None,
        f"{side}_lateral_canthus_depth": float((outer[2] - np.mean(pts[:, 2])) / scale) if outer is not None else None,
        f"{side}_eye_socket_bowl_depth": depth / scale,
        f"{side}_visible_opening_fraction": float(height / (width + 1e-8)),
    }
    if brow is not None and orbit is not None:
        vals[f"{side}_upper_lid_to_orbit_rim_distance"] = (distance(brow, orbit) or 0.0) / scale
    if cheek is not None and orbit is not None:
        vals[f"{side}_lower_lid_to_orbit_rim_distance"] = (distance(cheek, orbit) or 0.0) / scale
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
            if (mv := emit(spec, val, confidence=0.70, source_space=space)):
                out.append(mv)
    return out
