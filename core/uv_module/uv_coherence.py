"""
Глобальные метрики «согласованности» UV-текстуры с мешом — без детальной сегментации зон.

Идея: при плохом соответствии фото и приорной геометрии (маска, экстремальный ракурс)
на bake часто появляются **доминирующие вертикальные полосы**, рост хаотичности направлений
градиента и всплеск высокочастотного шума; центр лица может давать **асимметрию** градиента
в узкой полосе (прокси кривого кончика носа в UV).

Использование: перед тяжёлым анализом ROI вызвать с ``analytic_mask`` из
``build_uv_analysis_bundle``; пороги подбираются по калибровочной выборке.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import cv2
import numpy as np

__all__ = ["compute_uv_global_coherence"]


def compute_uv_global_coherence(
    uv_texture: np.ndarray,
    mask: Optional[np.ndarray] = None,
    *,
    min_valid_pixels: int = 2048,
) -> Dict[str, Any]:
    """
    Args:
        uv_texture: H×W×3 BGR uint8 (как после HDUV).
        mask: H×W bool — предпочтительно analytic_mask; иначе видимость bake; None → gray>0.

    Returns:
        Словарь с числовыми признаками и ``uv_deformation_risk`` ∈ [0, 1] (выше = подозрительнее).
    """
    tex = np.asarray(uv_texture)
    if tex.ndim != 3 or tex.shape[2] < 3:
        raise ValueError("uv_texture must be HxWx3 BGR")

    gray = cv2.cvtColor(tex, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    h, w = gray.shape[:2]

    if mask is None:
        valid = gray > (1.0 / 255.0)
    else:
        valid = np.asarray(mask, dtype=bool)
        if valid.shape != (h, w):
            raise ValueError(f"mask shape {valid.shape} != texture {(h, w)}")

    n_valid = int(np.count_nonzero(valid))
    if n_valid < int(min_valid_pixels):
        return {
            "valid_pixels": float(n_valid),
            "usable": False,
            "vertical_gradient_bias": None,
            "gradient_direction_entropy_norm": None,
            "laplacian_p95_norm": None,
            "nose_strip_grad_asymmetry": None,
            "uv_deformation_risk": None,
        }

    g = gray.copy()
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)

    abs_gx = np.abs(gx[valid])
    abs_gy = np.abs(gy[valid])
    mean_gx = float(np.mean(abs_gx) + 1e-8)
    mean_gy = float(np.mean(abs_gy) + 1e-8)
    vertical_bias = float(mean_gy / mean_gx)

    ang = np.arctan2(gy[valid], gx[valid])
    hist, _ = np.histogram(ang, bins=36, range=(-math.pi, math.pi), density=True)
    hist = hist.astype(np.float64)
    hist = hist / (hist.sum() + 1e-12)
    ent = float(-np.sum(hist * np.log(hist + 1e-12)))
    ent_max = math.log(36)
    entropy_norm = float(np.clip(ent / ent_max, 0.0, 1.0))

    lap_v = np.abs(lap[valid])
    p50 = float(np.percentile(lap_v, 50))
    p95 = float(np.percentile(lap_v, 95))
    lap_p95_norm = float(np.clip((p95 - p50) / (p50 + 1e-4), 0.0, 15.0))

    nose_asym = _nose_strip_grad_asymmetry(gx, h, w)

    # --- эвристический риск [0,1] (калибруй на своих данных) ---
    vb_excess = max(0.0, vertical_bias - 1.15) / 1.25
    risk = float(
        np.clip(
            0.38 * min(vb_excess, 1.0)
            + 0.32 * entropy_norm
            + 0.20 * min(lap_p95_norm / 6.0, 1.0)
            + 0.10 * min(nose_asym / 0.5, 1.0),
            0.0,
            1.0,
        )
    )

    return {
        "valid_pixels": float(n_valid),
        "usable": True,
        "vertical_gradient_bias": vertical_bias,
        "gradient_direction_entropy_norm": entropy_norm,
        "laplacian_p95_norm": lap_p95_norm,
        "nose_strip_grad_asymmetry": nose_asym,
        "uv_deformation_risk": risk,
    }


def _nose_strip_grad_asymmetry(gx: np.ndarray, h: int, w: int) -> float:
    """Полоса ~центр UV: разница средних |∂/∂x| слева/справа от вертикали (прокси шва на носу)."""
    y0, y1 = int(0.40 * h), int(0.62 * h)
    x0, x1 = int(0.38 * w), int(0.62 * w)
    if y1 <= y0 or x1 <= x0:
        return 0.0
    px = gx[y0:y1, x0:x1]
    mid = px.shape[1] // 2
    if mid < 2 or px.shape[1] - mid < 2:
        return 0.0
    l = float(np.mean(np.abs(px[:, :mid])))
    r = float(np.mean(np.abs(px[:, mid:])))
    return float(abs(l - r) / (l + r + 1e-6))
