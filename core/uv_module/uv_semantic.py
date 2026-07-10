"""
Семантические маски в UV-пространстве для локального анализа текстуры (морщины, НГС, периорбит).

Использует те же индексы вершин BFM, что и ``app.pipeline.zones.MACRO_BONE_INDICES``:
растеризация треугольников, у которых все три вершины входят в заданное множество.

Итоговая маска для метрик: ``region_uv & analytic_mask`` (видимость ∧ confidence ∧ original из bake).
"""
from __future__ import annotations

import logging
from typing import Dict, Iterable, Mapping, Optional, Set

import cv2
import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "prepare_uv_px",
    "rasterize_vertex_set_uv_mask",
    "build_semantic_uv_masks",
    "nasolabial_proxy_from_masks",
    "crow_feet_dilated_from_orbits",
    "analytic_times_region",
]


def prepare_uv_px(uv_coords: np.ndarray, size: int) -> np.ndarray:
    """UV [0,1] → пиксели карты (как в ``UVBaker``)."""
    uv = np.asarray(uv_coords, dtype=np.float32)
    if uv.ndim != 2 or uv.shape[1] < 2:
        raise ValueError("uv_coords must be (N, 2+)")
    uv = uv[:, :2].copy()
    uv_max = float(uv.max(initial=0.0))
    if uv_max > 1.5:
        uv = uv / max(uv_max, 1e-6)
    uv_px = np.empty((uv.shape[0], 2), dtype=np.float32)
    uv_px[:, 0] = uv[:, 0] * (size - 1)
    uv_px[:, 1] = (1.0 - uv[:, 1]) * (size - 1)
    return uv_px


def rasterize_vertex_set_uv_mask(
    uv_coords: np.ndarray,
    triangles: np.ndarray,
    vertex_indices: Set[int] | frozenset[int],
    uv_size: int,
) -> np.ndarray:
    """
    Бинарная маска ``(uv_size, uv_size)``: True, если центр треугольника покрыт и все три вершины ∈ ``vertex_indices``.
    """
    uv_px = prepare_uv_px(uv_coords, uv_size)
    tris = np.asarray(triangles, dtype=np.int64)
    allowed = frozenset(int(v) for v in vertex_indices)
    mask = np.zeros((uv_size, uv_size), dtype=np.uint8)

    for i in range(tris.shape[0]):
        a, b, c = int(tris[i, 0]), int(tris[i, 1]), int(tris[i, 2])
        if a not in allowed or b not in allowed or c not in allowed:
            continue
        pts = uv_px[tris[i]].astype(np.int32).reshape(-1, 1, 2)
        cv2.fillConvexPoly(mask, pts, 255)

    return mask > 0


def _load_macro_indices() -> dict[str, frozenset[int]]:
    """Ленивый импорт тяжёлого ``zones.py``."""
    from app.pipeline.zones import MACRO_BONE_INDICES

    return dict(MACRO_BONE_INDICES)


# Зоны по умолчанию для текстурного forensic (можно расширять)
DEFAULT_TEXTURE_ZONE_NAMES: tuple[str, ...] = (
    "forehead",
    "brow_ridge_L",
    "brow_ridge_R",
    "orbit_L",
    "orbit_R",
    "nose_bridge_tip",
    "cheekbone_L",
    "cheekbone_R",
    "chin",
    "jaw_L",
    "jaw_R",
)


def build_semantic_uv_masks(
    uv_coords: np.ndarray,
    triangles: np.ndarray,
    uv_size: int,
    *,
    zone_names: Optional[Iterable[str]] = None,
    macro_indices: Optional[Mapping[str, frozenset[int]]] = None,
) -> Dict[str, np.ndarray]:
    """
    Словарь ``имя_зоны → bool (H, W)`` в разрешении UV.

    Если ``macro_indices`` не передан, подгружается ``MACRO_BONE_INDICES`` из пайплайна.
    """
    names = tuple(zone_names) if zone_names is not None else DEFAULT_TEXTURE_ZONE_NAMES
    macro = macro_indices if macro_indices is not None else _load_macro_indices()

    out: Dict[str, np.ndarray] = {}
    for name in names:
        verts = macro.get(name)
        if not verts:
            logger.warning("Unknown or empty zone %r — skip", name)
            continue
        out[name] = rasterize_vertex_set_uv_mask(uv_coords, triangles, verts, uv_size)
    return out


def nasolabial_proxy_from_masks(
    cheekbone_L: np.ndarray,
    cheekbone_R: np.ndarray,
    nose_bridge: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Прокси носогубки: часть скуловой маски **ниже** медианной строки по Y маски переносицы в UV.

    На сильном профиле и масках качество ниже, чем у полигона по лендамаркам — зато без новых данных.
    """
    cheekbone_L = np.asarray(cheekbone_L, dtype=bool)
    cheekbone_R = np.asarray(cheekbone_R, dtype=bool)
    nose_bridge = np.asarray(nose_bridge, dtype=bool)
    if not np.any(nose_bridge):
        z = np.zeros_like(cheekbone_L, dtype=bool)
        return z, z
    ys = np.where(nose_bridge)[0]
    y0 = int(np.clip(np.median(ys), 0, cheekbone_L.shape[0] - 1))
    below = np.zeros_like(cheekbone_L, dtype=bool)
    below[y0:, :] = True
    return cheekbone_L & below, cheekbone_R & below


def crow_feet_dilated_from_orbits(
    orbit_L: np.ndarray,
    orbit_R: np.ndarray,
    *,
    dilate_px: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Расширенная периорбитальная зона с латерального края — для энергии мелких морщин («гусиные лапки»).

    Простая эвристика: морфологическое расширение орбитальных масок (без вычитания века — см. метрики).
    """
    k = max(1, int(dilate_px))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k | 1, k | 1))

    def dil(m: np.ndarray) -> np.ndarray:
        u8 = (np.asarray(m, dtype=bool).astype(np.uint8) * 255)
        d = cv2.dilate(u8, ker, iterations=1)
        return d > 0

    return dil(orbit_L), dil(orbit_R)


def analytic_times_region(
    analytic_mask: np.ndarray,
    region_uv: np.ndarray,
) -> np.ndarray:
    """Пересечение с аналитической маской (видимость ∧ confidence ∧ original)."""
    a = np.asarray(analytic_mask, dtype=bool)
    r = np.asarray(region_uv, dtype=bool)
    if a.shape != r.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {r.shape}")
    return a & r
