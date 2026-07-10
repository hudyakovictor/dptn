from __future__ import annotations

import numpy as np

from .catalog_specs import specs_for_module
from .common import emit
from .topology_utils import valid_triangles_for_vertices

IMPLEMENTATION = "pair_zone_residuals.py"


def specs():
    # Pair-residual family is diagnostic-only by catalog policy.
    # Do not route to identity scoring until canonical extract analogs exist.
    return specs_for_module(IMPLEMENTATION, families={"F_pair_zone"}, scope="pair")


def _parse_name(name: str):
    if not name.startswith("pair_zone_"):
        return None, None
    core = name[len("pair_zone_"):]
    suffixes = [
        "mean_vertex_distance", "median_vertex_distance", "p95_vertex_distance",
        "signed_normal_mean", "signed_normal_median", "signed_normal_p95",
        "edge_strain_energy", "bending_energy", "residual_tail_p90_minus_p50",
    ]
    for suf in sorted(suffixes, key=len, reverse=True):
        tail = "_" + suf
        if core.endswith(tail):
            return core[:-len(tail)], suf
    return None, None


def _region_indices(ctx, region: str) -> np.ndarray:
    raw = ctx.macro_indices.get(region, [])
    if raw is None or len(raw) == 0:
        return np.array([], dtype=np.int64)
    shared = set(np.asarray(ctx.shared_idx, dtype=np.int64).tolist())
    return np.asarray([int(i) for i in raw if int(i) in shared and 0 <= int(i) < len(ctx.vertices_a_unit)], dtype=np.int64)


def _edge_pairs(ctx, idx: np.ndarray) -> list[tuple[int, int]]:
    if ctx.triangles is None or idx.size < 3:
        return []
    idx_set = set(int(i) for i in idx)
    tri = valid_triangles_for_vertices(ctx.triangles, len(ctx.vertices_a_unit))
    seen = set()
    out = []
    for t in tri:
        a, b, c = map(int, t)
        for u, v in ((a, b), (b, c), (c, a)):
            if u in idx_set and v in idx_set:
                e = (min(u, v), max(u, v))
                if e not in seen:
                    seen.add(e); out.append(e)
    return out


def _compute_region(ctx, region: str) -> dict[str, float]:
    idx = _region_indices(ctx, region)
    if idx.size < 1:
        return {}
    a = ctx.vertices_a_unit[idx]
    b = ctx.vertices_b_unit_aligned[idx]
    disp = b - a
    d = np.linalg.norm(disp, axis=1)
    vals = {
        "mean_vertex_distance": float(np.mean(d)),
        "median_vertex_distance": float(np.median(d)),
        "p95_vertex_distance": float(np.percentile(d, 95)),
        "residual_tail_p90_minus_p50": float(np.percentile(d, 90) - np.median(d)),
    }
    if ctx.normals_a is not None and len(ctx.normals_a) > int(np.max(idx)):
        n = np.asarray(ctx.normals_a[idx], dtype=float)
        signed = np.sum(disp * n, axis=1)
        vals.update({
            "signed_normal_mean": float(np.mean(signed)),
            "signed_normal_median": float(np.median(signed)),
            "signed_normal_p95": float(np.percentile(signed, 95)),
        })
    edges = _edge_pairs(ctx, idx)
    strains = []
    bends = []
    res_by_idx = {int(i): d[k] for k, i in enumerate(idx)}
    for u, v in edges:
        la = float(np.linalg.norm(ctx.vertices_a_unit[u] - ctx.vertices_a_unit[v]))
        lb = float(np.linalg.norm(ctx.vertices_b_unit_aligned[u] - ctx.vertices_b_unit_aligned[v]))
        if la > 1e-8:
            strains.append(((lb - la) / la) ** 2)
        if u in res_by_idx and v in res_by_idx:
            bends.append((res_by_idx[u] - res_by_idx[v]) ** 2)
    vals["edge_strain_energy"] = float(np.mean(strains)) if strains else float(np.mean(np.square(d)))
    vals["bending_energy"] = float(np.mean(bends)) if bends else float(np.var(d))
    return vals


def compute_pair(ctx, specs_):
    out = []
    cache: dict[str, dict[str, float]] = {}
    for spec in specs_:
        region, metric = _parse_name(spec.name)
        if not region or not metric:
            continue
        if region not in cache:
            cache[region] = _compute_region(ctx, region)
        val = cache[region].get(metric)
        if (mv := emit(spec, val, confidence=0.80)):
            out.append(mv)
    return out
