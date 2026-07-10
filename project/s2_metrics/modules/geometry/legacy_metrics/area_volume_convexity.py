from __future__ import annotations

import numpy as np

from .catalog_specs import specs_for_module
from .common import emit, face_scale, zone_points
from .topology_utils import region_surface_area, region_signed_volume_to_plane
from .mid_sagittal import estimate_mid_sagittal_plane

IMPLEMENTATION = "area_volume_convexity.py"


def specs():
    return specs_for_module(IMPLEMENTATION, families={"F7"})


def _verts(ctx, space: str):
    if space == "shape_neutral": return ctx.vertices_shape_neutral
    if space == "raw": return ctx.vertices_raw
    return ctx.vertices_canon


def compute(ctx, specs_):
    out = []
    cache = {}
    for spec in specs_:
        region = None; metric = None
        for cand in sorted(ctx.macro_indices.keys(), key=len, reverse=True):
            pref = cand + "_"
            if spec.name.startswith(pref):
                region = cand; metric = spec.name[len(pref):]; break
        if region is None:
            continue
        for space in spec.source_spaces:
            if space == "shape_neutral" and ctx.vertices_shape_neutral is None:
                continue
            if space not in {"canon_bucket", "shape_neutral", "raw"}:
                continue
            key=(region, space)
            if key not in cache:
                verts=_verts(ctx, space)
                vals={}
                if verts is not None:
                    scale=face_scale(ctx, space=space)
                    pts=zone_points(ctx, region, space=space)
                    vals["vertex_count"] = float(len(pts))
                    if len(pts)>=3:
                        span=np.ptp(pts,axis=0)
                        plane_point, plane_normal, _ = estimate_mid_sagittal_plane(verts, ctx.macro_indices)
                        surf=region_surface_area(verts, ctx.triangles, ctx.macro_indices.get(region, []))
                        bbox_area = span[0]*span[1] + span[0]*span[2] + span[1]*span[2]
                        vals["surface_area_ratio"] = (bbox_area if surf is None else surf)/(scale*scale+1e-8)
                        vals["bbox_area_ratio"] = float(bbox_area/(scale*scale+1e-8))
                        vals["bbox_volume_ratio"] = float(np.prod(span)/(scale**3+1e-8))
                        vals["convex_hull_volume_ratio"] = vals["bbox_volume_ratio"]
                        vol=region_signed_volume_to_plane(verts, ctx.triangles, ctx.macro_indices.get(region, []), plane_point, plane_normal)
                        if vol is None:
                            signed=(pts-plane_point)@plane_normal; area=(surf if surf is not None else span[0]*span[1]); vol=float(np.mean(signed)*area)
                        vals["signed_volume_to_plane_ratio"] = float(vol/(scale**3+1e-8))
                        vals["convexity_index"] = float(np.std(pts[:,2])/(scale+1e-8))
                        z=pts[:,2]
                        vals["concavity_index"] = float((np.percentile(z,75)-np.median(z))/(scale+1e-8))
                        vals["local_area_stretch"] = float((bbox_area if surf is None else surf)/(span[0]*span[1]+1e-8))
                cache[key]=vals
            val=cache[key].get(metric)
            if val is not None and abs(val) < 1e-6 and metric != "vertex_count":
                continue
            if (mv:=emit(spec,val,confidence=0.68,source_space=space)): out.append(mv)
    return out
