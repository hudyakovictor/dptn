from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Union

from .constants import BLUR_THRESHOLD_DEFAULT, NOISE_THRESHOLD_DEFAULT
try:
    from core.contracts import MeasurementResult, ResultStatus, AdmissibilityDecision
except ImportError:
    # Заглушки для совместимости
    from dataclasses import dataclass
    from enum import Enum
    
    class ResultStatus(Enum):
        SUCCESS = "success"
        REJECTED = "rejected"
    
    @dataclass
    class MeasurementResult:
        status: ResultStatus
        value: float = 0.0
        reason: str = ""
    
    @dataclass
    class AdmissibilityDecision:
        is_rejected: bool = False
        reason: str = ""
        blocking_issues: list = None
        
        def __post_init__(self):
            if self.blocking_issues is None:
                self.blocking_issues = []

MIN_FACE_TEXTURE_PX = 120


def _reject_quality_dict(decision: AdmissibilityDecision) -> dict[str, object]:
    """QG-02: callers expect dict with .get(), not AdmissibilityDecision."""
    return {
        "success": False,
        "is_rejected": True,
        "overall_score": 0.0,
        "overall_quality": 0.0,
        "blur_value": 0.0,
        "sharpness_variance": 0.0,
        "noise_level": 0.0,
        "quality_scope": "rejected",
        "is_motion_blurred": False,
        "is_jpeg_blocky": False,
        "is_over_smoothed": False,
        "jpeg_blockiness": 1.0,
        "admissibility_reason": decision.reason,
        "blocking_issues": list(decision.blocking_issues),
    }


def _jpeg_block_boundary_slices(grid_offset: int) -> tuple[slice, slice, slice, slice]:
    """Column slices aligned to JPEG 8x8 grid given crop origin offset in full image."""
    ox = int(grid_offset) % 8
    boundary_a = (7 - ox) % 8
    boundary_b = boundary_a + 1
    inside_a = (3 - ox) % 8
    inside_b = inside_a + 1
    return (
        slice(boundary_a, None, 8),
        slice(boundary_b, None, 8),
        slice(inside_a, None, 8),
        slice(inside_b, None, 8),
    )


def _jpeg_blockiness_score(gray: np.ndarray, grid_offset_x: int) -> float:
    h_g, w_g = gray.shape[:2]
    if h_g <= 16 or w_g <= 16:
        return 1.0
    b_a, b_b, i_a, i_b = _jpeg_block_boundary_slices(grid_offset_x)
    boundary_a = gray[:, b_a]
    boundary_b = gray[:, b_b]
    n_blocks = min(boundary_a.shape[1], boundary_b.shape[1])
    if n_blocks <= 0:
        return 1.0
    diff_grid_x = float(np.mean(np.abs(boundary_a[:, :n_blocks] - boundary_b[:, :n_blocks])))
    inside_a = gray[:, i_a]
    inside_b = gray[:, i_b]
    n_inside = min(inside_a.shape[1], inside_b.shape[1])
    if n_inside <= 0:
        return 1.0
    diff_inside_x = float(np.mean(np.abs(inside_a[:, :n_inside] - inside_b[:, :n_inside])))
    return diff_grid_x / (diff_inside_x + 1e-5)


def _laplacian_sharpness_denominator(min_face_dim: int) -> float:
    """Scale Laplacian variance by face crop size (archive ~200px vs modern 4K crops)."""
    dim = max(int(min_face_dim), 64)
    return 400.0 * float(np.clip(dim / 224.0, 0.35, 2.5))


class QualityGate:
    """
    [QUAL-01] Technical Quality Gate.
    Ensures input imagery meets forensic standards for sharpness and SNR.
    """
    def __init__(
        self, 
        blur_threshold: float = BLUR_THRESHOLD_DEFAULT, 
        noise_threshold: float = NOISE_THRESHOLD_DEFAULT
    ):
        self.blur_threshold = float(blur_threshold)
        self.noise_threshold = float(noise_threshold)

    def _estimate_noise(self, gray: np.ndarray) -> float:
        median = cv2.medianBlur(gray, 3)
        return float(np.mean(np.abs(gray.astype(np.float32) - median.astype(np.float32))))

    def evaluate(self, image_path: Union[str, Path], bbox: dict | None = None) -> Dict[str, Union[float, Dict[str, bool], str, bool]]:
        """
        Оценивает качество фото. Защищает пайплайн от падений и отсеивает 
        нерелевантные (слишком мелкие или шумные) лица.
        """
        img = cv2.imread(str(image_path))
        if img is None:
            return _reject_quality_dict(
                AdmissibilityDecision(
                    admissible=False,
                    reason="INSUFFICIENT_DATA_UNREADABLE",
                    blocking_issues=["INSUFFICIENT_DATA_UNREADABLE"],
                )
            )

        h, w = img.shape[:2]

        if bbox is not None:
            face_h = bbox.get("h") or h
            face_w = bbox.get("w") or w
            if face_h < MIN_FACE_TEXTURE_PX or face_w < MIN_FACE_TEXTURE_PX:
                return _reject_quality_dict(
                    AdmissibilityDecision(
                        admissible=False,
                        reason=f"FACE_TOO_SMALL_{int(min(face_h, face_w))}px",
                        blocking_issues=[f"FACE_TOO_SMALL_{int(min(face_h, face_w))}px"],
                    )
                )

        quality_scope = "full_image"
        jpeg_grid_offset_x = 0
        if bbox is not None:
            x = int(bbox.get("x", bbox.get("x_min", 0)) or 0)
            y = int(bbox.get("y", bbox.get("y_min", 0)) or 0)
            bw = int(bbox.get("w", bbox.get("width", 0)) or 0)
            bh = int(bbox.get("h", bbox.get("height", 0)) or 0)
            if bw <= 0 and bbox.get("x_max") is not None:
                bw = int(bbox.get("x_max") or 0) - x
            if bh <= 0 and bbox.get("y_max") is not None:
                bh = int(bbox.get("y_max") or 0) - y
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(w, x0 + max(0, bw)), min(h, y0 + max(0, bh))
            if x1 > x0 and y1 > y0:
                img = img[y0:y1, x0:x1]
                quality_scope = "face_crop"
                jpeg_grid_offset_x = x0 % 8

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_min_dim = min(gray.shape[:2])
        
        # Оценка размытия (Лапласиан)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        
        # [FIX-Error-15] Directional motion blur detection using Sobel X/Y variance ratios
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        var_x = max(float(np.var(sobel_x)), 1e-5)
        var_y = max(float(np.var(sobel_y)), 1e-5)
        motion_ratio = max(var_x / var_y, var_y / var_x)
        is_motion_blurred = bool(motion_ratio > 3.0 and min(var_x, var_y) < 100.0)
        
        blockiness = _jpeg_blockiness_score(gray, jpeg_grid_offset_x)
        is_jpeg_blocky = bool(blockiness > 1.35)

        # [FIX QG-02]: Порог резкости должен быть динамическим или более реалистичным.
        # Для соцсетей Laplacian variance > 65 уже считается приемлемым.
        sharpness_score = float(
            np.clip(blur_score / _laplacian_sharpness_denominator(face_min_dim), 0.0, 1.0)
        )
        
        # Penalize sharpness for motion blur or JPEG blockiness
        if is_motion_blurred:
            sharpness_score *= 0.5
        if is_jpeg_blocky:
            sharpness_score *= 0.7

        noise_score = self._estimate_noise(gray)
        noise_quality = float(np.clip(1.0 - (noise_score / 25.0), 0.0, 1.0))

        # QG-01: waxy/over-smoothed skin (silicone/deepfake) — sharp + low blockiness, not motion blur.
        is_over_smoothed = bool(
            sharpness_score > 0.88
            and noise_quality > 0.82
            and blockiness < 1.08
            and not is_motion_blurred
        )
        overall_penalty = 0.75 if is_over_smoothed else 1.0
        if is_over_smoothed:
            sharpness_score *= 0.72

        overall_score = float(((sharpness_score * 0.7) + (noise_quality * 0.3)) * overall_penalty)
        
        return {
            "success": True,
            "is_rejected": is_motion_blurred, # [FORENSIC PATCH] Гладкие лица (силикон) теперь пропускаются дальше в пайплайн
            "blur_value": blur_score,
            "sharpness_variance": blur_score,
            "noise_level": noise_score,
            "overall_score": overall_score,
            "overall_quality": overall_score,
            "quality_scope": quality_scope,
            "is_motion_blurred": is_motion_blurred,
            "is_jpeg_blocky": is_jpeg_blocky,
            "is_over_smoothed": is_over_smoothed,
            "jpeg_blockiness": blockiness,
        }

    def evaluate_face_quality(self, img_full: np.ndarray, face_bbox: dict, skin_mask: np.ndarray) -> dict:
        """
        Оценивает качество СТРОГО внутри Bounding Box и маски кожи.
        Исправляет баг оценки качества по заднему фону (TX-07).
        """
        x, y, w, h = face_bbox['x'], face_bbox['y'], face_bbox['w'], face_bbox['h']
        
        # 1. Защита от микро-лиц (Баг: слишком маленькие лица проходили проверку)
        if w < MIN_FACE_TEXTURE_PX or h < MIN_FACE_TEXTURE_PX:
            return _reject_quality_dict(
                AdmissibilityDecision(
                    admissible=False,
                    reason="FACE_TOO_SMALL",
                    blocking_issues=["FACE_TOO_SMALL"],
                )
            )
            
        # 2. Вырезаем только лицо
        face_crop = img_full[y:y+h, x:x+w]
        
        mask_crop = None
        # Если маска кожи передана, применяем ее, чтобы исключить волосы и очки
        if skin_mask is not None:
            # Убедимся, что маска совпадает по размеру с кропом
            mask_crop = skin_mask[y:y+h, x:x+w]
            gray_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            
            # 3. Измерение резкости (Variance of Laplacian) только по коже
            # Пиксели вне маски не должны влиять на дисперсию
            laplacian = cv2.Laplacian(gray_crop, cv2.CV_64F)
            valid_laplacian = laplacian[mask_crop > 0]
            
            if len(valid_laplacian) < 100:
                return _reject_quality_dict(
                    AdmissibilityDecision(
                        admissible=False,
                        reason="INSUFFICIENT_SKIN",
                        blocking_issues=["INSUFFICIENT_SKIN"],
                    )
                )
                
            sharpness = np.var(valid_laplacian)
        else:
            gray_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray_crop, cv2.CV_64F).var()

        # 4. Оценка шума (Median Blur разница)
        median_blurred = cv2.medianBlur(gray_crop, 3)
        noise_diff = np.abs(gray_crop.astype(np.int16) - median_blurred.astype(np.int16))
        noise_level = np.mean(noise_diff[mask_crop > 0]) if skin_mask is not None else np.mean(noise_diff)

        success = True # [FORENSIC PATCH] Форензика не должна слепнуть от идеально гладких масок
        
        return {
            "success": success,
            "sharpness": float(sharpness),
            "noise_level": float(noise_level),
            "overall_score": float(
                np.clip(sharpness / _laplacian_sharpness_denominator(min(w, h)), 0, 1.0)
            ),
        }


def assess_facial_occlusion(
    *,
    periocular_metrics: dict[str, float] | None,
    specular_gloss: float | None,
    seg_visible_ratio: float | None = None,
) -> str:
    """Heuristic occlusion class: none | glasses | shadow."""
    peri = periocular_metrics or {}
    orbit_cov = float(peri.get("orbit_visible_ratio") or peri.get("periocular_visible_ratio") or 1.0)
    spec = float(specular_gloss or 0.0)
    if seg_visible_ratio is not None and seg_visible_ratio < 0.55:
        return "shadow"
    if orbit_cov < 0.45 and spec > 0.35:
        return "glasses"
    if orbit_cov < 0.35:
        return "shadow"
    return "none"

