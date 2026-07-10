from __future__ import annotations

import numpy as np

from .catalog_specs import specs_for_module
from .common import emit, face_scale
from .primitives import point
from .topology_utils import build_adjacency, dijkstra_distance, region_geodesic_extrema

IMPLEMENTATION = "geodesics.py"


def specs():
    return specs_for_module(IMPLEMENTATION, families={"F8"})

POINT_PATHS = {
    "geodesic_glabella_to_nasion_ratio": ("glabella", "nasion"),
    "geodesic_nasion_to_pronasale_ratio": ("nasion", "pronasale"),
    "geodesic_nasal_dorsum_length_ratio": ("nasion", "pronasale"),
    "geodesic_zygoma_to_gonion_L_ratio": ("zygoma_L", "gonion_L"),
    "geodesic_zygoma_to_gonion_R_ratio": ("zygoma_R", "gonion_R"),
}
REGION_PATHS = {
    "geodesic_orbit_rim_length_L_ratio": "orbit_L",
    "geodesic_orbit_rim_length_R_ratio": "orbit_R",
    "geodesic_brow_arc_length_L_ratio": "brow_ridge_L",
    "geodesic_brow_arc_length_R_ratio": "brow_ridge_R",
    "geodesic_jawline_length_L_ratio": "jaw_L",
    "geodesic_jawline_length_R_ratio": "jaw_R",
}


def _verts(ctx, space: str):
    if space == "shape_neutral": return ctx.vertices_shape_neutral
    if space == "raw": return ctx.vertices_raw
    return ctx.vertices_canon


def _nearest_idx(vertices: np.ndarray, p):
    if p is None or vertices is None or len(vertices) == 0:
        return None
    d = np.linalg.norm(vertices - p, axis=1)
    return int(np.argmin(d))


def compute(ctx, specs_):
    out=[]
    adj_cache={}
    for spec in specs_:
        for space in spec.source_spaces:
            if space == "shape_neutral" and ctx.vertices_shape_neutral is None:
                continue
            if space not in {"canon_bucket", "shape_neutral", "raw"}:
                continue
            vertices=_verts(ctx, space)
            if vertices is None: continue
            scale=face_scale(ctx, space=space)
            val=None
            if spec.name in REGION_PATHS:
                region=REGION_PATHS[spec.name]
                gd=region_geodesic_extrema(vertices, ctx.triangles, ctx.macro_indices.get(region, []), axis=0)
                if gd is None:
                    pts_idx = np.asarray(list(ctx.macro_indices.get(region, [])), dtype=int) if ctx.macro_indices.get(region, []) else np.array([], dtype=int)
                    pts_idx = pts_idx[(pts_idx>=0)&(pts_idx<len(vertices))]
                    if len(pts_idx) >= 2:
                        pts = vertices[pts_idx]
                        gd = float(np.linalg.norm(pts[np.argmax(pts[:,0])] - pts[np.argmin(pts[:,0])]))
                val=None if gd is None else gd/scale
            elif spec.name in POINT_PATHS:
                a,b=POINT_PATHS[spec.name]
                ia,ib=_nearest_idx(vertices, point(ctx,a,space=space)), _nearest_idx(vertices, point(ctx,b,space=space))
                if ia is not None and ib is not None:
                    if space not in adj_cache: adj_cache[space]=build_adjacency(vertices, ctx.triangles)
                    gd=dijkstra_distance(adj_cache[space], ia, ib)
                    if gd is None:
                        gd = float(np.linalg.norm(vertices[ia] - vertices[ib]))
                    val=None if gd is None else gd/scale
            if (mv:=emit(spec,val,confidence=0.65,source_space=space)): out.append(mv)
    return out
