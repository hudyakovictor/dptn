import logging
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


logger = logging.getLogger(__name__)


class UVDetailEnhancer:
    """
    Base/detail разложение и усиление деталей для аналитической текстуры.
    Работаем в LAB-пространстве, усиливаем только L-канал, избегая цветовых артефактов.
    """

    def __init__(
        self,
        detail_strength: float = 1.2,
        unsharp_amount: float = 0.4,
        base_sigma_s_ratio: float = 0.02,
        base_sigma_r: float = 0.15,
        semantic_region_boosts: Optional[Dict[str, float]] = None,
    ):
        self.detail_strength = float(detail_strength)
        self.unsharp_amount = float(unsharp_amount)
        self.base_sigma_s_ratio = float(base_sigma_s_ratio)
        self.base_sigma_r = float(base_sigma_r)
        self.semantic_region_boosts = semantic_region_boosts or {
            "eyes": 1.3,
            "eyebrows": 1.2,
            "lips": 1.2,
            "nose": 1.15,
        }

    def enhance(
        self,
        uv_texture_raw: np.ndarray,
        uv_mask_visible: np.ndarray,
        uv_confidence: np.ndarray,
        semantic_masks: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Возвращает:
            uv_tex_analysis (float32) — текстура для анализа (кускам без покрытия применяется исходник).
            detail_strength_map (float32) — карта усиления деталей (0 вне видимых зон).
        """
        tex = uv_texture_raw.astype(np.float32)
        tex_u8 = np.clip(tex, 0, 255).astype(np.uint8)
        uv_size = tex.shape[0]

        # --- LAB-представление ---
        lab = cv2.cvtColor(tex_u8, cv2.COLOR_BGR2LAB).astype(np.float32)

        # --- Edge-preserving base с масштабируемым sigma_s ---
        sigma_s = max(10, int(uv_size * self.base_sigma_s_ratio))
        base_u8 = cv2.edgePreservingFilter(
            tex_u8,
            flags=cv2.RECURS_FILTER,
            sigma_s=sigma_s,
            sigma_r=self.base_sigma_r,
        )
        base_lab = cv2.cvtColor(base_u8, cv2.COLOR_BGR2LAB).astype(np.float32)

        # --- Детали только по каналу L ---
        detail_L = lab[:, :, 0] - base_lab[:, :, 0]

        gain_map = 1.0 + (self.detail_strength - 1.0) * uv_confidence.astype(np.float32)

        if semantic_masks:
            for region, boost in self.semantic_region_boosts.items():
                if region not in semantic_masks:
                    continue
                mask = semantic_masks[region].astype(np.float32)
                if mask.shape != gain_map.shape:
                    continue
                gain_map *= 1.0 + (boost - 1.0) * mask

        enhanced_L = base_lab[:, :, 0] + detail_L * gain_map

        result_lab = lab.copy()
        result_lab[:, :, 0] = np.clip(enhanced_L, 0.0, 255.0)
        result_bgr = cv2.cvtColor(
            np.clip(result_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR
        ).astype(np.float32)

        result = tex.copy()
        result[uv_mask_visible] = result_bgr[uv_mask_visible]

        # --- Unsharp mask по L-каналу ---
        if self.unsharp_amount > 0 and uv_mask_visible.any():
            result_lab_full = cv2.cvtColor(
                np.clip(result, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB
            ).astype(np.float32)
            unsharp_sigma = max(1.0, uv_size / 1024.0)
            blurred_L = np.asarray(
                cv2.GaussianBlur(result_lab_full[:, :, 0], (0, 0), unsharp_sigma),
                dtype=np.float32,
            )
            sharp_L = result_lab_full[:, :, 0] + self.unsharp_amount * (
                result_lab_full[:, :, 0] - blurred_L
            )
            result_lab_full[:, :, 0] = np.clip(sharp_L, 0.0, 255.0)
            sharpened = cv2.cvtColor(
                np.clip(result_lab_full, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR
            ).astype(np.float32)
            result[uv_mask_visible] = sharpened[uv_mask_visible]

        result = np.clip(result, 0, 255).astype(np.float32)

        detail_strength_map = (gain_map - 1.0).astype(np.float32)
        detail_strength_map[~uv_mask_visible] = 0.0

        visible_ratio = 100.0 * float(uv_mask_visible.sum()) / max(uv_mask_visible.size, 1)
        logger.debug(
            "[DETAIL] strength=%.2f unsharp=%.2f sigma_s=%d visible=%.1f%%",
            self.detail_strength,
            self.unsharp_amount,
            sigma_s,
            visible_ratio,
        )

        return result, detail_strength_map

