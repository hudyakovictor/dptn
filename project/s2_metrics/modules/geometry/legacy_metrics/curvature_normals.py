from __future__ import annotations

import numpy as np

from .catalog_specs import specs_for_module
from .common import emit, zone_points
from .topology_utils import laplacian_curvature_proxy, discrete_curvature_stats

IMPLEMENTATION = "curvature_normals.py"


def specs():
    return specs_for_module(IMPLEMENTATION, families={"F6"})


def _region_from_name(ctx, name: str):
    for cand in sorted(ctx.macro_indices.keys(), key=len, reverse=True):
        pref = cand + "_"
        if name.startswith(pref):
            return cand, name[len(pref):]
    return None, None


def _normal_stats(ctx, region: str, space: str):
    normals = (
        ctx.normals_shape_neutral
        if space == "shape_neutral"
        else ctx.normals_canon
        if space == "canon_bucket"
        else ctx.normals_raw
        if space == "raw"
        else None
    )
    raw = ctx.macro_indices.get(region, [])
    if normals is None or raw is None or len(raw) == 0:
        return {}
    idx = np.asarray(list(raw), dtype=int)
    idx = idx[(idx >= 0) & (idx < len(normals))]
    vis = ctx.visibility_canon if space == "canon_bucket" else ctx.visibility_raw
    if vis is not None and len(vis) == len(normals):
        vmask = np.asarray(vis, dtype=bool)
        idx = idx[vmask[idx]]
    if len(idx) < 2:
        return {}
    ns = np.asarray(normals[idx], dtype=float)
    dev = np.linalg.norm(ns - np.mean(ns, axis=0), axis=1)
    return {
        "normal_variance": float(np.mean(np.var(ns, axis=0))),
        "normal_gradient_proxy": float(np.mean(dev)),
        "normal_discontinuity_p95": float(np.percentile(dev, 95)),
        "edge_cliff_index": float(np.percentile(dev, 95) - np.median(dev)),
    }


def _verts(ctx, space: str):
    if space == "shape_neutral": return ctx.vertices_shape_neutral
    if space == "raw": return ctx.vertices_raw
    return ctx.vertices_canon


def compute(ctx, specs_):
    out = []
    cache = {}
    for spec in specs_:
        region, metric = _region_from_name(ctx, spec.name)
        if region is None:
            continue
        for space in spec.source_spaces:
            if space == "shape_neutral" and ctx.vertices_shape_neutral is None:
                continue
            if space not in {"canon_bucket", "shape_neutral", "raw"}:
                continue
            key = (region, space)
            if key not in cache:
                verts = _verts(ctx, space)
                pts = zone_points(ctx, region, space=space, visible_only=True)
                stats = _normal_stats(ctx, region, space)
                if verts is not None and len(pts) >= 2:
                    depth_curv = float(np.std(pts[:, 2]) / (np.ptp(pts[:, 0]) + np.ptp(pts[:, 1]) + 1e-8))
                    lap = laplacian_curvature_proxy(verts, ctx.triangles, ctx.macro_indices.get(region, []))
                    zdev = np.abs(pts[:, 2] - np.mean(pts[:, 2]))
                    if lap is None:
                        lap = depth_curv
                    stats.update({"depth_curvature_proxy": depth_curv, "laplacian_curvature_proxy": lap})
                    topo_curv = discrete_curvature_stats(verts, ctx.triangles, ctx.macro_indices.get(region, [])) if len(pts) > 2 else {}
                    if topo_curv:
                        stats.update(topo_curv)
                    else:
                        stats.update({
                            "mean_curvature_H_mean": lap,
                            "mean_curvature_H_p95": float(np.percentile(zdev, 95)),
                            "gaussian_curvature_K_mean": float(np.var(pts[:, 2]) / ((np.ptp(pts[:, 0]) * np.ptp(pts[:, 1])) + 1e-8)),
                            "gaussian_curvature_K_p95": float(np.percentile(zdev ** 2, 95)),
                            "principal_k1_mean": float(np.std(pts[:, 2]) / (np.ptp(pts[:, 0]) + 1e-8)),
                            "principal_k2_mean": float(np.std(pts[:, 2]) / (np.ptp(pts[:, 1]) + 1e-8)),
                        })
                cache[key] = stats
            val = cache[key].get(metric)
            if (mv := emit(spec, val, confidence=0.62, source_space=space)):
                out.append(mv)
    return out
