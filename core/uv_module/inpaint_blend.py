from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np


logger = logging.getLogger(__name__)


class UVBeautyPostprocessor:
    """
    Постобработка для текстуры рендера (beauty):
      - симметрийное заполнение слабых зон с локальной коррекцией цвета
      - inpaint для оставшихся дыр
      - сглаживание швов (Laplacian smoothing)

    ВНИМАНИЕ: используется только для uv_tex_beauty.
    uv_tex_analysis и карты оригинальности менять нельзя.
    """

    def __init__(
        self,
        sym_low_threshold: float = 0.2,
        sym_high_threshold: float = 0.6,
        inpaint_radius_ratio: float = 0.005,
        seam_iterations: int = 30,
        color_correct_kernel: int = 31,
    ) -> None:
        self.sym_low = float(sym_low_threshold)
        self.sym_high = float(sym_high_threshold)
        self.inpaint_radius_ratio = float(inpaint_radius_ratio)
        self.seam_iterations = 60 # Increased from 30 for smoother seams
        self.cc_kernel = int(color_correct_kernel)

    def process(
        self,
        uv_tex_analysis: np.ndarray,
        uv_mask_visible: np.ndarray,
        uv_confidence: np.ndarray,
        bfm_symmetry: Optional[np.ndarray] = None,
        uv_coords: Optional[np.ndarray] = None,
        enable_symmetry: bool = True,
    ) -> np.ndarray:
        tex = np.clip(uv_tex_analysis, 0, 255).astype(np.uint8)
        conf = np.asarray(uv_confidence, dtype=np.float32)

        if enable_symmetry:
            filled_tex, fill_mask = self._symmetry_fill(
                tex, conf, bfm_symmetry=bfm_symmetry, uv_coords=uv_coords
            )
        else:
            filled_tex = tex.copy()
            fill_mask = np.zeros(tex.shape[:2], dtype=bool)

        # Создаем "сплошную" маску лица (solid), чтобы inpaint работал только внутри дыр (глаза, рот), а не размазывал лицо на весь фон
        mask_u8 = uv_mask_visible.astype(np.uint8) * 255
        kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        solid_mask = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel_fill)

        holes = ((solid_mask > 0) & ~uv_mask_visible & ~fill_mask).astype(np.uint8) * 255
        if np.any(holes > 0):
            inpaint_radius = max(3, int(tex.shape[0] * self.inpaint_radius_ratio))
            filled_tex = cv2.inpaint(
                filled_tex,
                holes,
                inpaintRadius=inpaint_radius,
                flags=cv2.INPAINT_TELEA,
            )
            logger.debug(
                "[BEAUTY] Inpaint %d texels (radius=%d)",
                int((holes > 0).sum()),
                inpaint_radius,
            )

        seam_zone = cv2.dilate(
            fill_mask.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=2,
        ).astype(bool)

        if seam_zone.any() and self.seam_iterations > 0:
            filled_tex = self._laplacian_smooth_seam(
                filled_tex, seam_zone, iterations=self.seam_iterations
            )

        return filled_tex

    def _symmetry_fill(
        self,
        tex: np.ndarray,
        conf: np.ndarray,
        bfm_symmetry: Optional[np.ndarray] = None,
        uv_coords: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        flipped_tex = np.asarray(cv2.flip(tex, 1), dtype=np.uint8)
        flipped_conf = np.asarray(cv2.flip(conf, 1), dtype=np.float32)

        low_conf = conf < self.sym_low
        high_conf_sym = flipped_conf > self.sym_high
        fill_mask = low_conf & high_conf_sym

        if not fill_mask.any():
            return tex.copy(), fill_mask

        corrected = self._local_color_correct(flipped_tex, tex, fill_mask)
        result = tex.copy()
        result[fill_mask] = corrected[fill_mask]

        logger.debug(
            "[BEAUTY] Symmetry fill %d texels (%.1f%%)",
            int(fill_mask.sum()),
            100.0 * float(fill_mask.sum()) / max(fill_mask.size, 1),
        )

        return result, fill_mask

    def _local_color_correct(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        corrected = src.astype(np.float32)
        dst_f = dst.astype(np.float32)
        mask_f = mask.astype(np.float32)
        kernel = (self.cc_kernel, self.cc_kernel)

        count = cv2.blur(mask_f, kernel).clip(1e-6)

        for c in range(3):
            src_local = cv2.blur(corrected[:, :, c] * mask_f, kernel) / count
            dst_local = cv2.blur(dst_f[:, :, c] * mask_f, kernel) / count
            corrected[:, :, c] += dst_local - src_local

        return np.clip(corrected, 0, 255).astype(np.uint8)

    @staticmethod
    def _laplacian_smooth_seam(
        img: np.ndarray,
        seam_mask: np.ndarray,
        iterations: int = 30,
    ) -> np.ndarray:
        result = img.astype(np.float32)
        seam = seam_mask.astype(bool)
        kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32) / 4.0

        for _ in range(iterations):
            for c in range(3):
                smoothed = cv2.filter2D(result[:, :, c], -1, kernel)
                result[:, :, c] = np.where(seam, smoothed, result[:, :, c])

        return np.clip(result, 0, 255).astype(np.uint8)

