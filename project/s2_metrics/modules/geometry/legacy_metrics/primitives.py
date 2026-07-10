from __future__ import annotations

import re
import numpy as np

from .common import centroid, face_scale, zone_points


def _valid(p):
    return p is not None and np.asarray(p).shape == (3,) and np.isfinite(p).all()


def eye_points(ctx, side: str, *, space: str = "canon_bucket") -> np.ndarray:
    verts = ctx.vertices_canon if space == "canon_bucket" else ctx.vertices_raw if space == "raw" else ctx.vertices_shape_neutral
    if verts is None:
        return np.zeros((0, 3), dtype=float)
    idx_i = 1 if side == "L" else 0
    if len(ctx.annotation_groups) <= idx_i:
        return np.zeros((0, 3), dtype=float)
    idx = np.asarray(ctx.annotation_groups[idx_i], dtype=int)
    idx = idx[(idx >= 0) & (idx < len(verts))]
    return np.asarray(verts[idx], dtype=float) if len(idx) else np.zeros((0, 3), dtype=float)




def _zone_percentile_mean(ctx, zone: str, *, space: str, axis: int, high_pct: float = 90.0) -> np.ndarray | None:
    pts = zone_points(ctx, zone, space=space)
    if len(pts) == 0:
        return None
    q = float(np.percentile(pts[:, axis], high_pct))
    subset = pts[pts[:, axis] >= q]
    if len(subset) == 0:
        subset = pts
    m = np.mean(subset, axis=0)
    return m if np.isfinite(m).all() else None

def _zone_extreme(ctx, zone: str, *, space: str, axis: int, mode: str) -> np.ndarray | None:
    pts = zone_points(ctx, zone, space=space)
    if len(pts) == 0:
        return None
    idx = np.argmax(pts[:, axis]) if mode == "max" else np.argmin(pts[:, axis])
    p = pts[int(idx)]
    return p if np.isfinite(p).all() else None


def _midpoint(a: np.ndarray | None, b: np.ndarray | None) -> np.ndarray | None:
    if a is None or b is None:
        return None
    return (a + b) / 2.0

def point(ctx, name: str, *, space: str = "canon_bucket") -> np.ndarray | None:
    n = name.strip()

    # Semantic anatomical proxies from available 3DDFA zones. These are deliberately
    # before aliases so e.g. pronasale/subnasale do not collapse to nose centroid.
    if n in {"pronasale", "nose_tip"}:
        val = _zone_percentile_mean(ctx, "nose", space=space, axis=2, high_pct=90.0)
        if val is not None:
            return val
        val = _zone_percentile_mean(ctx, "nose_bridge_tip", space=space, axis=2, high_pct=90.0)
        if val is not None:
            return val
        return _zone_extreme(ctx, "nose", space=space, axis=2, mode="max")
    if n in {"subnasale", "nose_base"}:
        val = _zone_percentile_mean(ctx, "nose", space=space, axis=1, high_pct=90.0)
        if val is not None:
            return val
        val = _zone_percentile_mean(ctx, "nose_bridge_tip", space=space, axis=1, high_pct=90.0)
        if val is not None:
            return val
        return _zone_extreme(ctx, "nose", space=space, axis=1, mode="max")
    if n == "glabella":
        bl = centroid(ctx, "brow_ridge_L", space=space)
        br = centroid(ctx, "brow_ridge_R", space=space)
        mid = _midpoint(bl, br)
        return mid if mid is not None else centroid(ctx, "forehead", space=space)
    if n == "nasion":
        bridge = zone_points(ctx, "nose_bridge_tip", space=space)
        if len(bridge):
            # Upper/central bridge point close to glabella.
            yq = np.percentile(bridge[:, 1], 25)
            upper = bridge[bridge[:, 1] <= yq]
            if len(upper):
                return np.mean(upper, axis=0)
        return centroid(ctx, "nose_bridge_tip", space=space)
    if n == "labrale":
        return centroid(ctx, "upper_lip", space=space)

    aliases = {
        "zygoma_L": "cheekbone_L", "zygoma_R": "cheekbone_R",
        "gonion_L": "jaw_L", "gonion_R": "jaw_R",
        "pogonion": "chin", "chin": "chin", "glabella": "forehead",
        "nasion": "nose_bridge_tip", "bridge": "nose_bridge_tip",
        "nose_bridge": "nose_bridge_tip", "nose_bridge_tip": "nose_bridge_tip",
        "alar_L": "nose_wing_L", "alar_R": "nose_wing_R", "nose_wing_L": "nose_wing_L", "nose_wing_R": "nose_wing_R",
        "forehead": "forehead", "jaw_L": "jaw_L", "jaw_R": "jaw_R",
        "brow_ridge_L": "brow_ridge_L", "brow_ridge_R": "brow_ridge_R",
        "orbit_L": "orbit_L", "orbit_R": "orbit_R",
        "temporal_L": "temporal_L", "temporal_R": "temporal_R",
        "cheekbone_L": "cheekbone_L", "cheekbone_R": "cheekbone_R",
        "inner_canthus_L": "orbit_L", "inner_canthus_R": "orbit_R",
        "outer_canthus_L": "orbit_L", "outer_canthus_R": "orbit_R",
    }
    if n in {"pupil_L", "eye_center_L"}:
        pts = eye_points(ctx, "L", space=space); return np.mean(pts, axis=0) if len(pts) else centroid(ctx, "orbit_L", space=space)
    if n in {"pupil_R", "eye_center_R"}:
        pts = eye_points(ctx, "R", space=space); return np.mean(pts, axis=0) if len(pts) else centroid(ctx, "orbit_R", space=space)
    if n in {"lid_apex_L", "apex_L"}:
        pts = eye_points(ctx, "L", space=space); return pts[np.argmin(pts[:,1])] if len(pts) else None
    if n in {"lid_apex_R", "apex_R"}:
        pts = eye_points(ctx, "R", space=space); return pts[np.argmin(pts[:,1])] if len(pts) else None
    if n.startswith("inner_canthus_") or n.startswith("outer_canthus_"):
        side = n[-1]
        pts = zone_points(ctx, f"orbit_{side}", space=space)
        if len(pts) == 0: return None
        # Approximate medial/lateral by x extrema. Convention may vary but stable as feature candidate.
        verts_ref = ctx.vertices_canon if space == "canon_bucket" else ctx.vertices_raw if space == "raw" else ctx.vertices_shape_neutral
        mid_x = float(np.median(verts_ref[:, 0])) if verts_ref is not None and len(verts_ref) else float(np.median(pts[:, 0]))
        if n.startswith("inner"):
            return pts[np.argmin(np.abs(pts[:,0] - mid_x))]
        return pts[np.argmax(np.abs(pts[:,0] - mid_x))]
    z = aliases.get(n, n)
    c = centroid(ctx, z, space=space)
    if c is not None:
        return c
    # Direct vertex fallbacks for known points.
    verts = ctx.vertices_canon if space == "canon_bucket" else ctx.vertices_raw if space == "raw" else ctx.vertices_shape_neutral
    if verts is not None and n == "subnasale" and len(verts) > 245:
        return np.asarray(verts[245], dtype=float)
    return None


def distance(a, b) -> float | None:
    if not _valid(a) or not _valid(b): return None
    return float(np.linalg.norm(np.asarray(a)-np.asarray(b)))


def angle(a, b, c) -> float | None:
    if not _valid(a) or not _valid(b) or not _valid(c): return None
    v1=np.asarray(a)-np.asarray(b); v2=np.asarray(c)-np.asarray(b)
    den=np.linalg.norm(v1)*np.linalg.norm(v2)
    if den<=1e-8: return None
    return float(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/den,-1,1))))


def tri_metrics(a, b, c, scale: float) -> dict[str, float]:
    if not _valid(a) or not _valid(b) or not _valid(c): return {}
    a=np.asarray(a); b=np.asarray(b); c=np.asarray(c)
    ab=np.linalg.norm(a-b); bc=np.linalg.norm(b-c); ca=np.linalg.norm(c-a)
    area=0.5*np.linalg.norm(np.cross(b-a,c-a))
    base=ab; height=2*area/(base+1e-8)
    normal=np.cross(b-a,c-a); nn=np.linalg.norm(normal); normal=normal/(nn+1e-8)
    # signed apex depth: c to line midpoint depth axis z proxy
    return {
        "area_ratio": float(area/(scale*scale+1e-8)),
        "perimeter_ratio": float((ab+bc+ca)/(scale+1e-8)),
        "edge_ab_ratio": float(ab/(scale+1e-8)),
        "edge_bc_ratio": float(bc/(scale+1e-8)),
        "edge_ca_ratio": float(ca/(scale+1e-8)),
        "apex_angle_deg": angle(a,c,b) or 0.0,
        "base_to_height_ratio": float(base/(height+1e-8)),
        "plane_normal_x": float(normal[0]), "plane_normal_y": float(normal[1]), "plane_normal_z": float(normal[2]),
        "normal_to_face_plane_angle_deg": float(np.degrees(np.arccos(np.clip(abs(normal[2]),-1,1)))),
        "signed_apex_depth_ratio": float((c[2]-((a[2]+b[2])/2))/(scale+1e-8)),
    }


def quad_metrics(points: list[np.ndarray | None], scale: float) -> dict[str,float]:
    if len(points)!=4 or any(not _valid(p) for p in points): return {}
    p=[np.asarray(x) for x in points]
    area=0.5*np.linalg.norm(np.cross(p[1]-p[0],p[2]-p[0]))+0.5*np.linalg.norm(np.cross(p[2]-p[0],p[3]-p[0]))
    d1=np.linalg.norm(p[0]-p[2]); d2=np.linalg.norm(p[1]-p[3])
    e=[np.linalg.norm(p[(i+1)%4]-p[i]) for i in range(4)]
    n1=np.cross(p[1]-p[0],p[2]-p[0]); n2=np.cross(p[2]-p[0],p[3]-p[0])
    n1=n1/(np.linalg.norm(n1)+1e-8); n2=n2/(np.linalg.norm(n2)+1e-8)
    normal=(n1+n2); normal=normal/(np.linalg.norm(normal)+1e-8)
    twist=angle(n1, np.zeros(3), n2) or 0.0
    plane_d=[np.dot(x-p[0], normal) for x in p]
    return {
        "area_ratio": float(area/(scale*scale+1e-8)),
        "diagonal_ratio": float(d1/(d2+1e-8)),
        "aspect_ratio": float(max(e)/(min(e)+1e-8)),
        "twist_angle_deg": float(twist),
        "nonplanarity_ratio": float(np.std(plane_d)/(scale+1e-8)),
        "normal_x": float(normal[0]), "normal_y": float(normal[1]), "normal_z": float(normal[2]),
        "signed_depth_imbalance_ratio": float(((p[0][2]+p[1][2])-(p[2][2]+p[3][2]))/(2*scale+1e-8)),
    }
