from __future__ import annotations

import numpy as np

from .catalog_specs import specs_for_module
from .common import centroid, emit, face_scale, zone_points
from .mid_sagittal import estimate_mid_sagittal_plane, reflect_points_across_plane

IMPLEMENTATION = "mirror_asymmetry.py"

REGION_PAIRS = {
    "orbit": ("orbit_L", "orbit_R"),
    "brow": ("brow_ridge_L", "brow_ridge_R"),
    "eye_lid": ("left_eye", "right_eye"),
    "zygoma": ("cheekbone_L", "cheekbone_R"),
    "temporal": ("temporal_L", "temporal_R"),
    "jaw": ("jaw_L", "jaw_R"),
}


def specs():
    return specs_for_module(IMPLEMENTATION, families={"F10"})


def _verts(ctx, space: str):
    if ctx.vertices_shape_neutral is not None:
        return ctx.vertices_shape_neutral
    if space == "raw":
        return ctx.vertices_raw
    return ctx.vertices_canon


def _pts(ctx, zone, space: str):
    verts=_verts(ctx,space)
    if verts is None: return np.zeros((0,3),dtype=float)
    if zone == "left_eye" and len(ctx.annotation_groups) > 1:
        idx = np.asarray(ctx.annotation_groups[1], dtype=int); idx = idx[(idx>=0)&(idx<len(verts))]
        return verts[idx]
    if zone == "right_eye" and len(ctx.annotation_groups) > 0:
        idx = np.asarray(ctx.annotation_groups[0], dtype=int); idx = idx[(idx>=0)&(idx<len(verts))]
        return verts[idx]
    return zone_points(ctx, zone, space=space)


def _centroid(ctx, zone: str, space: str):
    pts=_pts(ctx,zone,space)
    return np.mean(pts,axis=0) if len(pts) else None


def _compute_space(ctx, space: str):
    verts=_verts(ctx,space)
    if verts is None or len(verts)==0: return {}, 0.0
    scale=face_scale(ctx,space=space)
    plane_point, plane_normal, plane_conf = estimate_mid_sagittal_plane(verts, ctx.macro_indices)
    vals={}
    reflected=reflect_points_across_plane(verts,plane_point,plane_normal)
    dglob=np.linalg.norm(verts-reflected,axis=1)
    vals.update({
        "mirror_global_centroid_asymmetry": float(abs(np.dot(np.mean(verts,axis=0)-plane_point, plane_normal))/scale),
        "mirror_global_vertex_mean_asymmetry": float(np.mean(dglob)/scale),
        "mirror_global_vertex_p95_asymmetry": float(np.percentile(dglob,95)/scale),
        "mirror_global_heatmap_p95": float(np.percentile(dglob,95)/scale),
        "mirror_global_midline_deviation": float(abs(np.dot(np.mean(verts,axis=0)-plane_point, plane_normal))/scale),
    })
    for name,(zl,zr) in REGION_PAIRS.items():
        pl,pr=_pts(ctx,zl,space),_pts(ctx,zr,space)
        if len(pl)==0 or len(pr)==0: continue
        cl,cr=np.mean(pl,axis=0),np.mean(pr,axis=0)
        cr_ref=reflect_points_across_plane(np.asarray([cr]),plane_point,plane_normal)[0]
        vals[f"mirror_{name}_centroid_asymmetry"]=float(np.linalg.norm(cl-cr_ref)/scale)
        pr_ref=reflect_points_across_plane(pr,plane_point,plane_normal)
        dl=np.linalg.norm(pl-cl,axis=1); dr=np.linalg.norm(pr_ref-cr_ref,axis=1)
        m=min(len(dl),len(dr))
        if m:
            dd=np.abs(np.sort(dl)[:m]-np.sort(dr)[:m])
            vals[f"mirror_{name}_vertex_mean_asymmetry"]=float(np.mean(dd)/scale)
            vals[f"mirror_{name}_vertex_p95_asymmetry"]=float(np.percentile(dd,95)/scale)
            vals[f"mirror_{name}_heatmap_p95"]=vals[f"mirror_{name}_vertex_p95_asymmetry"]
        vals[f"mirror_{name}_midline_deviation"]=float(abs(np.dot((cl+cr_ref)/2-plane_point,plane_normal))/scale)
    for name,z in (("nose","nose_bridge_tip"),("chin","chin")):
        p=_centroid(ctx,z,space)
        if p is not None:
            dev=float(abs(np.dot(p-plane_point,plane_normal))/scale)
            for suffix in ("centroid_asymmetry","vertex_mean_asymmetry","vertex_p95_asymmetry","heatmap_p95","midline_deviation"):
                vals[f"mirror_{name}_{suffix}"]=dev
    return vals, plane_conf


def compute(ctx, specs_):
    out=[]; cache={}
    for spec in specs_:
        for space in spec.source_spaces:
            if space == "shape_neutral" and ctx.vertices_shape_neutral is None: continue
            if space not in {"canon_bucket","shape_neutral","raw"}: continue
            if space not in cache: cache[space]=_compute_space(ctx,space)
            vals,conf=cache[space]
            val=vals.get(spec.name)
            if (mv:=emit(spec,val,confidence=min(0.8,0.4+conf),source_space=space)): out.append(mv)
    return out
