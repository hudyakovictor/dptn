from __future__ import annotations

import numpy as np

from .catalog_specs import specs_for_module
from .common import emit, face_scale, zone_points

IMPLEMENTATION = "zone_morphology.py"


def specs():
    return specs_for_module(IMPLEMENTATION, families={"F_zone"})


def _fit_plane_residuals(pts: np.ndarray) -> np.ndarray:
    if len(pts) < 3:
        return np.zeros(len(pts), dtype=float)
    c = np.mean(pts, axis=0)
    X = pts - c
    try:
        _, _, vh = np.linalg.svd(X, full_matrices=False)
        n = vh[-1]
    except Exception:
        n = np.array([0.0, 0.0, 1.0])
    n = n / (np.linalg.norm(n) + 1e-8)
    return X @ n


def _region_from_metric_name(name: str) -> tuple[str | None, str | None]:
    if not name.startswith("zone_"):
        return None, None
    core = name[len("zone_"):]
    # Longest known suffixes first to avoid splitting inside zone names.
    suffixes = [
        "span_lateral_ratio", "span_vertical_ratio", "span_depth_ratio",
        "bbox_area_ratio", "bbox_volume_ratio",
        "depth_std_ratio", "depth_p25_ratio", "depth_p50_ratio", "depth_p75_ratio", "depth_p95_ratio",
        "radial_dispersion_ratio", "plane_residual_std_ratio", "plane_residual_p95_ratio",
        "normal_mean_x", "normal_mean_y", "normal_mean_z", "normal_variance", "vertex_count",
    ]
    for suf in sorted(suffixes, key=len, reverse=True):
        tail = "_" + suf
        if core.endswith(tail):
            return core[: -len(tail)], suf
    return None, None


def compute(ctx, specs_):
    out = []
    scale_default = face_scale(ctx)
    cache: dict[tuple[str, str], dict[str, float]] = {}
    for spec in specs_:
        region, metric = _region_from_metric_name(spec.name)
        if not region or not metric:
            continue
        for space in spec.source_spaces:
            if space == "shape_neutral" and ctx.vertices_shape_neutral is None:
                continue
            if space not in {"canon_bucket", "shape_neutral", "raw"}:
                continue
            key = (region, space)
            if key not in cache:
                scale = face_scale(ctx, space=space)
                pts = zone_points(ctx, region, space=space)
                vals: dict[str, float] = {}
                if len(pts) > 0:
                    c = np.mean(pts, axis=0)
                    vals.update({
                        "vertex_count": float(len(pts)),
                    })
                    if len(pts) >= 2:
                        span = np.ptp(pts, axis=0)
                        radial = np.linalg.norm(pts - c, axis=1)
                        plane_res = _fit_plane_residuals(pts)
                        vals.update({
                            "span_lateral_ratio": float(span[0] / (scale + 1e-8)),
                            "span_vertical_ratio": float(span[1] / (scale + 1e-8)),
                            "span_depth_ratio": float(span[2] / (scale + 1e-8)),
                            "depth_std_ratio": float(np.std(pts[:, 2]) / (scale + 1e-8)),
                        })
                        z = pts[:, 2]
                        if len(pts) >= 10:
                            vals.update({
                                "depth_p25_ratio": float((np.percentile(z, 25) - np.median(z)) / (scale + 1e-8)),
                                "depth_p50_ratio": float(np.median(z) / (scale + 1e-8)),
                                "depth_p75_ratio": float((np.percentile(z, 75) - np.median(z)) / (scale + 1e-8)),
                                "depth_p95_ratio": float((np.percentile(z, 95) - np.median(z)) / (scale + 1e-8)),
                            })
                    if len(pts) >= 3:
                        span = np.ptp(pts, axis=0)
                        radial = np.linalg.norm(pts - c, axis=1)
                        plane_res = _fit_plane_residuals(pts)
                        vals.update({
                            "bbox_area_ratio": float((span[0]*span[1] + span[0]*span[2] + span[1]*span[2]) / (scale*scale + 1e-8)),
                            "bbox_volume_ratio": float(np.prod(span) / (scale**3 + 1e-8)),
                            "radial_dispersion_ratio": float(np.std(radial) / (scale + 1e-8)),
                            "plane_residual_std_ratio": float(np.std(plane_res) / (scale + 1e-8)),
                            "plane_residual_p95_ratio": float(np.percentile(np.abs(plane_res), 95) / (scale + 1e-8)),
                        })
                    raw = ctx.macro_indices.get(region, [])
                    normals = (
                        ctx.normals_shape_neutral
                        if space == "shape_neutral"
                        else ctx.normals_canon
                        if space == "canon_bucket"
                        else ctx.normals_raw
                        if space == "raw"
                        else None
                    )
                    if normals is not None and raw is not None and len(raw) > 0:
                        idx = np.asarray(list(raw), dtype=int)
                        idx = idx[(idx >= 0) & (idx < len(normals))]
                        if len(idx):
                            ns = np.asarray(normals[idx], dtype=float)
                            vals.update({
                                "normal_mean_x": float(np.mean(ns[:,0])),
                                "normal_mean_y": float(np.mean(ns[:,1])),
                                "normal_mean_z": float(np.mean(ns[:,2])),
                                "normal_variance": float(np.mean(np.var(ns, axis=0))),
                            })
                cache[key] = vals
            val = cache[key].get(metric)
            if (mv := emit(spec, val, confidence=0.60, source_space=space)):
                out.append(mv)
    return out
