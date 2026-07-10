"""ROI лба на UV для анализа морщин (обрезка текстуры с фото + маска)."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# Эталон — разметка под UV 1024×1024: ширина/высота прямоугольника, гориз. центр холста.
_REF_UV = 1024
_ROI_W_REF = 650
_ROI_H_REF = 210
# Ужатие финального прямоугольника (в пикселях при ref_uv_size): с каждой стороны по ширине, снизу по высоте.
_INSET_LR_REF = 20
_INSET_BOTTOM_REF = 10


def _apply_edge_insets(
    x: int,
    y: int,
    rw: int,
    rh: int,
    canvas_h: int,
    canvas_w: int,
    *,
    ref_uv_size: int,
    inset_lr_ref: int,
    inset_bottom_ref: int,
) -> Tuple[int, int, int, int]:
    ilr = max(0, int(round(inset_lr_ref * canvas_w / ref_uv_size)))
    ib = max(0, int(round(inset_bottom_ref * canvas_h / ref_uv_size)))
    x2 = x + ilr
    rw2 = rw - 2 * ilr
    rh2 = rh - ib
    rw2 = max(1, rw2)
    rh2 = max(1, rh2)
    x2 = max(0, min(x2, canvas_w - rw2))
    y2 = max(0, min(y, canvas_h - rh2))
    return x2, y2, rw2, rh2


def compute_forehead_wrinkle_roi(
    uv_mask_visible: np.ndarray,
    *,
    ref_uv_size: int = _REF_UV,
    roi_w_ref: int = _ROI_W_REF,
    roi_h_ref: int = _ROI_H_REF,
    inset_lr_ref: int = _INSET_LR_REF,
    inset_bottom_ref: int = _INSET_BOTTOM_REF,
) -> Tuple[int, int, int, int]:
    """
    Прямоугольник ``(x, y, w, h)`` в пикселях UV: по ширине центрирован на холсте,
    по вертикали — от верхней границы развёртки лица (лоб), высота ``roi_h_ref`` в масштабе.
    Затем внутреннее ужатие: ``inset_lr_ref`` с каждого бока, ``inset_bottom_ref`` снизу (в координатах эталона ``ref_uv_size``).
    """
    m = np.asarray(uv_mask_visible, dtype=bool)
    h, w = m.shape[:2]
    rw = max(1, int(round(roi_w_ref * w / ref_uv_size)))
    rh = max(1, int(round(roi_h_ref * h / ref_uv_size)))
    x0 = max(0, (w - rw) // 2)
    ys = np.where(m)[0]
    if ys.size == 0:
        y0 = max(0, (h - rh) // 2)
    else:
        y_top = int(ys.min())
        band = int(ys.max() - ys.min() + 1)
        y0 = min(y_top + max(1, int(band * 0.03)), h - rh)
    y0 = max(0, min(y0, h - rh))
    return _apply_edge_insets(
        x0,
        y0,
        rw,
        rh,
        h,
        w,
        ref_uv_size=ref_uv_size,
        inset_lr_ref=inset_lr_ref,
        inset_bottom_ref=inset_bottom_ref,
    )


def forehead_wrinkle_crop_and_mask(
    uv_texture_bgr: np.ndarray,
    uv_mask_visible: np.ndarray,
    analytic_mask: Optional[np.ndarray] = None,
    *,
    ref_uv_size: int = _REF_UV,
    roi_w_ref: int = _ROI_W_REF,
    roi_h_ref: int = _ROI_H_REF,
    inset_lr_ref: int = _INSET_LR_REF,
    inset_bottom_ref: int = _INSET_BOTTOM_REF,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """
    Returns:
        crop_bgr — ``HxWx3`` вырезка UV-текстуры (цвет с фото в зоне бейка);
        mask_u8 — ``HxW`` uint8 0/255, зона анализа = ROI ∧ bake ∧ (analytic если передан);
        roi_xywh — итоговый прямоугольник после внутренних отступов (см. ``inset_*_ref``).
    """
    tex = np.asarray(uv_texture_bgr)
    if tex.ndim != 3 or tex.shape[2] != 3:
        raise ValueError("uv_texture_bgr must be HxWx3 BGR")
    vis = np.asarray(uv_mask_visible, dtype=bool)
    if tex.shape[:2] != vis.shape[:2]:
        raise ValueError("texture and uv_mask_visible shape mismatch")
    x, y, rw, rh = compute_forehead_wrinkle_roi(
        vis,
        ref_uv_size=ref_uv_size,
        roi_w_ref=roi_w_ref,
        roi_h_ref=roi_h_ref,
        inset_lr_ref=inset_lr_ref,
        inset_bottom_ref=inset_bottom_ref,
    )
    crop = tex[y : y + rh, x : x + rw].copy()
    roi_vis = vis[y : y + rh, x : x + rw]
    if analytic_mask is not None:
        am = np.asarray(analytic_mask, dtype=bool)
        if am.shape != vis.shape:
            raise ValueError("analytic_mask shape mismatch")
        roi_ok = roi_vis & am[y : y + rh, x : x + rw]
    else:
        roi_ok = roi_vis
    mask = (roi_ok.astype(np.uint8)) * 255
    return crop, mask, (x, y, rw, rh)
