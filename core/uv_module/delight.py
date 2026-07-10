"""
Опциональный de-lighting (ТЗ 5.6): SH-based shading estimation, albedo = texture / shading.
Константы совместимы с 3DDFA_V3 (см. core/3ddfa_v3/model/recon.py, compute_texture()).
"""
from __future__ import annotations

import logging

import numpy as np


logger = logging.getLogger(__name__)

# Константы SH как в core/3ddfa_v3/model/recon.py
SH_A = np.array(
    [np.pi, 2 * np.pi / np.sqrt(3.0), 2 * np.pi / np.sqrt(8.0)],
    dtype=np.float32,
)
SH_C = np.array(
    [
        1.0 / np.sqrt(4 * np.pi),
        np.sqrt(3.0) / np.sqrt(4 * np.pi),
        3.0 * np.sqrt(5.0) / np.sqrt(12 * np.pi),
    ],
    dtype=np.float32,
)
INIT_LIT = np.array([0.8, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)


def compute_shading_uv(
    normals_uv: np.ndarray,
    alpha_sh: np.ndarray,
) -> np.ndarray:
    """
    Считает shading в UV по нормалям и SH-коэффициентам (3DDFA_V3-совместимая формула).

    Args:
        normals_uv: (H, W, 3) нормали в camera space, единичные
        alpha_sh: (27,) или (3, 9) — SH-параметры (init_lit добавляется внутри).

    Returns:
        shading (H, W, 3), float32, >= 0.
    """
    alpha_sh = np.asarray(alpha_sh, dtype=np.float32).reshape(-1)
    if alpha_sh.size == 27:
        alpha_sh = alpha_sh.reshape(3, 9)
    elif alpha_sh.size == 9:
        alpha_sh = alpha_sh.reshape(1, 9).repeat(3, axis=0)
    else:
        raise ValueError(f"alpha_sh must have 9 or 27 elements, got {alpha_sh.size}")

    alpha_sh = alpha_sh + INIT_LIT.reshape(1, 9)

    n = np.asarray(normals_uv, dtype=np.float32)
    if n.ndim == 2:
        n = n[:, np.newaxis, :]
    one = np.ones((*n.shape[:2], 1), dtype=np.float32)
    nx = n[..., 0:1]
    ny = n[..., 1:2]
    nz = n[..., 2:3]

    Y = np.concatenate(
        [
            SH_A[0] * SH_C[0] * one,
            -SH_A[1] * SH_C[1] * ny,
            SH_A[1] * SH_C[1] * nz,
            -SH_A[1] * SH_C[1] * nx,
            SH_A[2] * SH_C[2] * (nx * ny),
            -SH_A[2] * SH_C[2] * (ny * nz),
            0.5 * SH_A[2] * SH_C[2] / np.sqrt(3.0) * (3.0 * nz * nz - 1.0),
            -SH_A[2] * SH_C[2] * (nx * nz),
            0.5 * SH_A[2] * SH_C[2] * (nx * nx - ny * ny),
        ],
        axis=-1,
    )

    shading = np.dot(Y, alpha_sh.T).astype(np.float32)
    shading = np.maximum(shading, 0.0)

    logger.debug(
        "[DELIGHT] Shading stats: min=%.3f max=%.3f mean=%.3f",
        float(shading.min()),
        float(shading.max()),
        float(shading.mean()),
    )

    return shading


def albedo_from_texture(
    uv_texture: np.ndarray,
    shading_uv: np.ndarray,
    eps: float = 0.1,
    clamp_max: float = 255.0,
    shadow_threshold: float = 0.3,
) -> np.ndarray:
    """
    uv_texture и shading_uv: (H, W, 3), в одном масштабе (0..255 или 0..1).
    В хорошо освещённых областях считаем albedo = tex / shading; в глубоких тенях
    мягко блендим к исходной текстуре.
    """
    tex = np.asarray(uv_texture, dtype=np.float32)
    sh = np.asarray(shading_uv, dtype=np.float32)

    well_lit = sh[sh > eps]
    if well_lit.size > 0:
        sh_median = float(np.median(well_lit))
        if sh_median > 1e-2:
            sh = sh / sh_median

    sh_safe = np.maximum(sh, eps)
    albedo_raw = tex / sh_safe

    sh_mean = sh.mean(axis=-1, keepdims=True)
    confidence = np.clip(sh_mean / shadow_threshold, 0.0, 1.0)
    albedo = confidence * albedo_raw + (1.0 - confidence) * tex

    logger.debug(
        "[DELIGHT] Albedo confidence mean=%.3f range=[%.1f, %.1f]",
        float(confidence.mean()),
        float(albedo.min()),
        float(albedo.max()),
    )

    return np.clip(albedo, 0.0, clamp_max).astype(np.float32)
