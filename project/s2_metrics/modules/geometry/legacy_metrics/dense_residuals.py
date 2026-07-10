from __future__ import annotations

import numpy as np

from .catalog_specs import specs_for_module
from .common import emit
from .topology_utils import valid_triangles_for_vertices

IMPLEMENTATION = "dense_residuals.py"

REGIONS = ("global", "orbit_L", "orbit_R", "brow_ridge_L", "brow_ridge_R", "nose_bridge_tip", "cheekbone_L", "cheekbone_R", "temporal_L", "temporal_R", "jaw_L", "jaw_R", "chin")


def specs():
    return specs_for_module(IMPLEMENTATION, families={"F9"}, scope="pair")


def _region_indices(ctx, region: str) -> np.ndarray:
    if region == "global":
        return np.asarray(ctx.shared_idx, dtype=np.int64)
    raw = ctx.macro_indices.get(region, [])
    if raw is None or len(raw) == 0:
        return np.array([], dtype=np.int64)
    shared = set(np.asarray(ctx.shared_idx, dtype=np.int64).tolist())
    return np.asarray([int(i) for i in raw if int(i) in shared and 0 <= int(i) < len(ctx.vertices_a_unit)], dtype=np.int64)


def _edge_strain_energy(ctx, idx: np.ndarray) -> float | None:
    """ARAP-like local edge strain energy in aligned unit space."""
    if ctx.triangles is None or idx.size < 3:
        return None
    idx_set = set(int(i) for i in idx)
    tri = valid_triangles_for_vertices(ctx.triangles, len(ctx.vertices_a_unit))
    strains = []
    seen = set()
    for t in tri:
        a, b, c = map(int, t)
        for u, v in ((a, b), (b, c), (c, a)):
            if u not in idx_set or v not in idx_set:
                continue
            e = (min(u, v), max(u, v))
            if e in seen:
                continue
            seen.add(e)
            la = float(np.linalg.norm(ctx.vertices_a_unit[u] - ctx.vertices_a_unit[v]))
            lb = float(np.linalg.norm(ctx.vertices_b_unit_aligned[u] - ctx.vertices_b_unit_aligned[v]))
            if la > 1e-8:
                strains.append(((lb - la) / la) ** 2)
    if not strains:
        return None
    return float(np.mean(strains))


def _smooth_bending_energy(ctx, idx: np.ndarray, residual: np.ndarray) -> float | None:
    """TPS-like bending proxy: variance of residual differences along local edges."""
    if ctx.triangles is None or idx.size < 3:
        return None
    idx_set = set(int(i) for i in idx)
    tri = valid_triangles_for_vertices(ctx.triangles, len(ctx.vertices_a_unit))
    # map residual by global vertex id
    res_map = {int(i): residual[k] for k, i in enumerate(idx)}
    vals = []
    seen = set()
    for t in tri:
        a, b, c = map(int, t)
        for u, v in ((a, b), (b, c), (c, a)):
            if u not in idx_set or v not in idx_set:
                continue
            e = (min(u, v), max(u, v))
            if e in seen:
                continue
            seen.add(e)
            vals.append(float((res_map[u] - res_map[v]) ** 2))
    if not vals:
        return None
    return float(np.mean(vals))

def compute_pair(ctx, specs_):
    out = []
    spec_by = {s.name: s for s in specs_}
    if ctx.shared_idx.size == 0:
        return out
    for region in REGIONS:
        idx = _region_indices(ctx, region)
        if idx.size < 1:
            continue
        disp = ctx.vertices_b_unit_aligned[idx] - ctx.vertices_a_unit[idx]
        d = np.linalg.norm(disp, axis=1)
        vals = {
            f"{region}_regional_mean_vertex_distance": float(np.mean(d)),
            f"{region}_regional_median_vertex_distance": float(np.median(d)),
            f"{region}_regional_p95_vertex_distance": float(np.percentile(d, 95)),
            f"{region}_stable_anchor_unstable_residual": float(np.percentile(d, 90) - np.median(d)),
        }
        bend = _smooth_bending_energy(ctx, idx, d)
        vals[f"{region}_tps_bending_energy_proxy"] = float(np.var(d) if bend is None else bend)
        arap = _edge_strain_energy(ctx, idx)
        vals[f"{region}_arap_deformation_energy"] = float(np.mean(np.square(d)) if arap is None else arap)
        if ctx.normals_a is not None and len(ctx.normals_a) > int(np.max(idx)):
            n = np.asarray(ctx.normals_a[idx], dtype=float)
            signed = np.sum(disp * n, axis=1)
            vals.update({
                f"{region}_regional_signed_normal_displacement_mean": float(np.mean(signed)),
                f"{region}_regional_signed_normal_displacement_median": float(np.median(signed)),
                f"{region}_regional_signed_normal_displacement_p95": float(np.percentile(signed, 95)),
            })
        for name, val in vals.items():
            spec = spec_by.get(name)
            if spec and (mv := emit(spec, val, confidence=0.82)):
                out.append(mv)
    return out
