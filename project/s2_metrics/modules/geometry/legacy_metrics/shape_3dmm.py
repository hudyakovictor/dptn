from __future__ import annotations

import numpy as np

from .catalog_specs import specs_for_module
from .common import emit, face_scale, zone_points
from .topology_utils import region_surface_area, laplacian_curvature_proxy

IMPLEMENTATION = "shape_3dmm.py"


def specs():
    return specs_for_module(IMPLEMENTATION, families={"F11"})


def _id_confidence(bucket: str) -> float:
    b = str(bucket or "frontal").lower()
    if "profile" in b or "deep" in b:
        return 0.55
    if b == "frontal":
        return 0.85
    return 0.70


def _region_from_name(name: str):
    if not name.startswith("neutral_"):
        return None, None
    core = name[len("neutral_"):]
    regions = ["nose_bridge_tip", "brow_ridge_L", "brow_ridge_R", "cheekbone_L", "cheekbone_R", "temporal_L", "temporal_R", "orbit_L", "orbit_R", "jaw_L", "jaw_R", "chin", "forehead"]
    for r in sorted(regions, key=len, reverse=True):
        pref = r + "_"
        if core.startswith(pref):
            return r, core[len(pref):]
    return None, None


def compute(ctx, specs_):
    out = []
    spec_by = {s.name: s for s in specs_}
    id_conf = _id_confidence(ctx.pose_bucket)
    if ctx.id_params is not None:
        idv = np.asarray(ctx.id_params, dtype=float).reshape(-1)
        vals = {
            "id_params_norm": float(np.linalg.norm(idv)),
            "id_params_mean": float(np.mean(idv)),
            "id_params_std": float(np.std(idv)),
            "id_params_pca_distance": float(np.linalg.norm(idv)),
        }
        for i, coeff in enumerate(idv):
            vals[f"id_param_{i}"] = float(coeff)
        for name, val in vals.items():
            if name in spec_by and (mv := emit(spec_by[name], val, confidence=id_conf)):
                out.append(mv)
    if ctx.exp_params is not None and "exp_params_norm" in spec_by:
        if mv := emit(spec_by["exp_params_norm"], float(np.linalg.norm(ctx.exp_params)), confidence=0.75):
            out.append(mv)
    if ctx.vertices_shape_neutral is None:
        return out
    scale = face_scale(ctx, space="shape_neutral")
    for spec in specs_:
        region, metric = _region_from_name(spec.name)
        if region is None:
            continue
        pts = zone_points(ctx, region, space="shape_neutral", visible_only=True)
        if len(pts) < 2:
            continue
        span = np.ptp(pts, axis=0)
        val = None
        if metric == "centroid_depth":
            val = float(np.mean(pts[:, 2]) / scale)
        elif metric == "depth_span":
            val = float(np.ptp(pts[:, 2]) / scale)
        elif metric == "surface_area":
            surf = region_surface_area(ctx.vertices_shape_neutral, ctx.triangles, ctx.macro_indices.get(region, []))
            bbox_area = span[0] * span[1] + span[0] * span[2] + span[1] * span[2]
            val = (bbox_area if surf is None else surf) / (scale * scale + 1e-8)
        elif metric == "curvature":
            val = float(np.std(pts[:, 2]) / (np.ptp(pts[:, 0]) + np.ptp(pts[:, 1]) + 1e-8))
        elif metric == "volume":
            val = float(np.prod(span) / (scale ** 3 + 1e-8))
        elif metric == "pca_distance":
            val = float(np.linalg.norm(span) / scale)
        elif metric in {"id_coeff_sensitivity_mean", "id_coeff_sensitivity_p95"}:
            basis = ctx.shape_basis
            val = None
            if basis is not None:
                raw = ctx.macro_indices.get(region, [])
                idx = np.asarray(list(raw), dtype=int)
                idx = idx[(idx >= 0) & (idx < basis.shape[0])]
                if len(idx):
                    sens = np.linalg.norm(basis[idx], axis=1)
                    per_vertex = np.mean(np.abs(sens), axis=1)
                    val = float(
                        (np.mean(per_vertex) if metric.endswith("mean") else np.percentile(per_vertex, 95))
                        / (scale + 1e-8)
                    )
            if val is None:
                val = float(
                    (np.mean(np.abs(pts - np.mean(pts, axis=0))) if metric.endswith("mean") else np.percentile(np.linalg.norm(pts - np.mean(pts, axis=0), axis=1), 95))
                    / scale
                )
        if (mv := emit(spec, val, confidence=id_conf)):
            out.append(mv)
    return out
