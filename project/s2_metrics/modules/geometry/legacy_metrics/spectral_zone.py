from __future__ import annotations

import numpy as np

from .catalog_specs import specs_for_module
from .common import emit, zone_points
from .topology_utils import valid_triangles_for_vertices

IMPLEMENTATION = "spectral_zone.py"

SPECTRAL_ZONES = (
    "temporal_L",
    "temporal_R",
    "orbit_L",
    "orbit_R",
    "jaw_angle_L",
    "jaw_angle_R",
)
N_EIGEN = 8


def specs():
    return specs_for_module(IMPLEMENTATION, families={"F_spectral"})


def _cotangent_laplacian(vertices: np.ndarray, triangles: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    v = np.asarray(vertices, dtype=np.float64)
    tri = valid_triangles_for_vertices(triangles, len(v))
    if len(idx) < 10:
        return None
    idx_set = set(int(i) for i in idx)
    rows, cols, data = [], [], []
    diag = np.zeros(len(idx), dtype=np.float64)
    imap = {int(g): i for i, g in enumerate(idx)}
    for a, b, c in tri:
        for u, v0, v1 in ((a, b, c), (b, c, a), (c, a, b)):
            if u not in idx_set or v0 not in idx_set or v1 not in idx_set:
                continue
            iu, iv0, iv1 = imap[int(u)], imap[int(v0)], imap[int(v1)]
            pu, p0, p1 = v[int(u)], v[int(v0)], v[int(v1)]
            cot = float(np.dot(p0 - pu, p1 - pu) / (np.linalg.norm(np.cross(p0 - pu, p1 - pu)) + 1e-12))
            w = max(cot, 0.0)
            rows.extend([iu, iv0])
            cols.extend([iv0, iu])
            data.extend([w, w])
            diag[iu] += w
            diag[iv0] += w
    n = len(idx)
    L = np.zeros((n, n), dtype=np.float64)
    for r, c, d in zip(rows, cols, data):
        L[r, c] += d
    for i in range(n):
        L[i, i] = -diag[i]
    return L, idx


def _eigenvalues(L: np.ndarray, k: int) -> list[float]:
    try:
        from scipy.sparse.linalg import eigsh

        w, _ = eigsh(L.astype(np.float64), k=min(k, max(1, L.shape[0] - 2)), which="SM")
        return sorted(float(x) for x in w)
    except Exception:
        w = np.linalg.eigvalsh(L)
        return sorted(float(x) for x in w[:k])


def compute(ctx, specs_):
    out = []
    spec_by = {s.name: s for s in specs_}
    verts = ctx.vertices_canon
    if verts is None:
        return out
    for zone in SPECTRAL_ZONES:
        pts = zone_points(ctx, zone, space="canon_bucket", visible_only=True)
        if len(pts) < 20:
            continue
        raw = ctx.macro_indices.get(zone, [])
        idx = np.asarray(list(raw), dtype=int)
        idx = idx[(idx >= 0) & (idx < len(verts))]
        lap = _cotangent_laplacian(verts, ctx.triangles, idx)
        if lap is None:
            continue
        L, _ = lap
        evals = _eigenvalues(L, N_EIGEN)
        for i, ev in enumerate(evals[:N_EIGEN]):
            name = f"spectral_{zone}_eigen_{i}"
            if name in spec_by and (mv := emit(spec_by[name], ev, confidence=0.55)):
                out.append(mv)
    return out
