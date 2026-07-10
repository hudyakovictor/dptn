from __future__ import annotations

import numpy as np

from .catalog_specs import specs_for_module
from .common import emit, face_scale, zone_points
from .primitives import point

IMPLEMENTATION = "orbit_special.py"


def specs():
    return specs_for_module(IMPLEMENTATION, families={"F_orbit"})


def _ellipse_metrics(pts: np.ndarray, scale: float) -> dict[str, float]:
    xy = pts[:, :2] - np.mean(pts[:, :2], axis=0)
    cov = np.cov(xy.T) if len(xy) > 1 else np.eye(2) * 1e-8
    eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
    major = float(np.sqrt(max(eig[0], 0.0)) * 2.0)
    minor = float(np.sqrt(max(eig[-1], 0.0)) * 2.0)
    ecc = float(np.sqrt(max(0.0, 1.0 - (minor * minor / (major * major + 1e-8))))) if major > 0 else 0.0
    rad = np.sqrt(np.sum(xy * xy, axis=1)) if len(xy) else np.array([0.0])
    rim_flat = float(np.std(rad) / (np.mean(rad) + 1e-8)) if len(rad) > 2 else 0.0
    roundness = float(minor / (major + 1e-8)) if major > 0 else 0.0
    super_proxy = float(np.clip(2.0 / (rim_flat + 0.25), 1.0, 8.0))
    fit_exp, fit_err = 2.0, 0.0
    if major > 1e-8 and minor > 1e-8 and len(xy) > 5:
        vals_e, vecs = np.linalg.eigh(cov)
        vecs = vecs[:, np.argsort(vals_e)[::-1]]
        uv = xy @ vecs
        # Robust axis estimates from high percentiles instead of covariance only.
        a = max(float(np.percentile(np.abs(uv[:, 0]), 95)), major / 2.0, 1e-8)
        b = max(float(np.percentile(np.abs(uv[:, 1]), 95)), minor / 2.0, 1e-8)
        x = np.clip(np.abs(uv[:, 0]) / a, 0.0, 3.0)
        y = np.clip(np.abs(uv[:, 1]) / b, 0.0, 3.0)

        def err_for(n_exp: float) -> float:
            return float(np.mean((np.power(x, n_exp) + np.power(y, n_exp) - 1.0) ** 2))

        best = (float("inf"), 2.0)
        for n_exp in np.linspace(0.75, 10.0, 38):
            err = err_for(float(n_exp))
            if err < best[0]:
                best = (err, float(n_exp))
        # Optional continuous refinement if scipy is available.
        try:
            from scipy.optimize import minimize_scalar
            res = minimize_scalar(err_for, bounds=(0.75, 10.0), method="bounded", options={"xatol": 1e-3})
            if res.success and float(res.fun) < best[0]:
                best = (float(res.fun), float(res.x))
        except Exception:
            pass
        fit_err, fit_exp = best
    return {
        "ellipse_major_ratio": major / scale,
        "ellipse_minor_ratio": minor / scale,
        "ellipse_eccentricity": ecc,
        "ellipse_area_ratio": float(np.pi * major * minor / (scale * scale + 1e-8)),
        "ellipse_aspect_ratio": float(major / (minor + 1e-8)),
        "ellipse_roundness": roundness,
        "rim_flatness_proxy": rim_flat,
        "superellipse_exponent_proxy": super_proxy,
        "superellipse_fit_exponent": fit_exp,
        "superellipse_fit_error": fit_err,
    }


def _fit_plane_residuals(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = np.mean(pts, axis=0)
    X = pts - c
    try:
        _, _, vh = np.linalg.svd(X, full_matrices=False)
        normal, u, v = vh[-1], vh[0], vh[1]
    except Exception:
        normal, u, v = np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    normal = normal / (np.linalg.norm(normal) + 1e-8)
    if normal[2] < 0:
        normal = -normal
    return X @ normal, u, v


def _grid_depth_stats(pts: np.ndarray, residuals: np.ndarray, u: np.ndarray, v: np.ndarray) -> dict[str, float]:
    c = np.mean(pts, axis=0)
    uv = np.column_stack(((pts - c) @ u, (pts - c) @ v))
    if len(uv) < 6:
        return {"grid_mean": float(np.mean(residuals)), "grid_p95": float(np.percentile(residuals, 95)), "grid_var": float(np.var(residuals))}
    xs = np.linspace(np.min(uv[:, 0]), np.max(uv[:, 0]) + 1e-8, 5)
    ys = np.linspace(np.min(uv[:, 1]), np.max(uv[:, 1]) + 1e-8, 5)
    bins = []
    for i in range(4):
        for j in range(4):
            mask = (uv[:, 0] >= xs[i]) & (uv[:, 0] < xs[i + 1]) & (uv[:, 1] >= ys[j]) & (uv[:, 1] < ys[j + 1])
            if np.any(mask):
                bins.append(float(np.median(residuals[mask])))
    arr = np.asarray(bins if bins else list(map(float, residuals)), dtype=float)
    return {"grid_mean": float(np.mean(arr)), "grid_p95": float(np.percentile(arr, 95)), "grid_var": float(np.var(arr))}


def _compute_side(ctx, side: str, space: str) -> dict[str, float | None]:
    region = f"orbit_{side}"
    pts = zone_points(ctx, region, space=space)
    if len(pts) < 4:
        return {}
    scale = face_scale(ctx, space=space)
    prefix = f"orbit_{side}"
    center = np.mean(pts, axis=0)
    vals = {f"{prefix}_{k}": v for k, v in _ellipse_metrics(pts, scale).items()}
    brow = point(ctx, f"brow_ridge_{side}", space=space)
    cheek = point(ctx, f"cheekbone_{side}", space=space)
    inner = point(ctx, f"inner_canthus_{side}", space=space)
    outer = point(ctx, f"outer_canthus_{side}", space=space)
    residuals, basis_u, basis_v = _fit_plane_residuals(pts)
    grid = _grid_depth_stats(pts, residuals, basis_u, basis_v)
    vals.update({
        f"{prefix}_orbital_bowl_depth_ratio": float(np.ptp(residuals) / scale),
        f"{prefix}_orbital_bowl_volume_proxy": float(np.ptp(pts[:, 0]) * np.ptp(pts[:, 1]) * np.ptp(residuals) / (scale ** 3 + 1e-8)),
        f"{prefix}_orbital_centroid_depth": float(center[2] / scale),
        f"{prefix}_orbital_rim_prominence": float((np.percentile(residuals, 95) - np.median(residuals)) / scale),
        f"{prefix}_medial_orbital_depth": float((inner[2] - center[2]) / scale) if inner is not None else None,
        f"{prefix}_lateral_orbital_depth": float((outer[2] - center[2]) / scale) if outer is not None else None,
        f"{prefix}_orbital_bowl_curvature": float(np.std(residuals) / (np.ptp(pts[:, 0]) + np.ptp(pts[:, 1]) + 1e-8)),
        f"{prefix}_orbital_bowl_depth_map_mean": float(grid["grid_mean"] / scale),
        f"{prefix}_orbital_bowl_depth_map_p95": float(grid["grid_p95"] / scale),
    })
    if brow is not None:
        vals[f"{prefix}_supraorbital_rim_projection"] = float((brow[2] - center[2]) / scale)
        vals[f"{prefix}_orbital_roof_slope"] = float((brow[1] - center[1]) / (abs(brow[2] - center[2]) + 1e-8))
    if cheek is not None:
        vals[f"{prefix}_infraorbital_rim_projection"] = float((cheek[2] - center[2]) / scale)
        vals[f"{prefix}_orbital_floor_slope"] = float((cheek[1] - center[1]) / (abs(cheek[2] - center[2]) + 1e-8))
    return vals


def compute(ctx, specs_):
    out = []
    spec_by = {s.name: s for s in specs_}
    cache = {}
    for spec in specs_:
        side = "L" if spec.name.startswith("orbit_L_") else "R" if spec.name.startswith("orbit_R_") else None
        if side is None:
            continue
        for space in spec.source_spaces:
            if space == "shape_neutral" and ctx.vertices_shape_neutral is None:
                continue
            if space not in {"canon_bucket", "shape_neutral", "raw"}:
                continue
            key = (side, space)
            if key not in cache:
                cache[key] = _compute_side(ctx, side, space)
            val = cache[key].get(spec.name)
            if (mv := emit(spec, val, confidence=0.74, source_space=space)):
                out.append(mv)
    return out
