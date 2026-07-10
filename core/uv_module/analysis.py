from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import cv2
import numpy as np


@dataclass
class UVMaskedTextureMetrics:
    gloss_score: float
    pore_variance: float
    homogeneity: float
    specular_ratio: float
    mean_intensity: float
    std_intensity: float
    edge_density: float
    valid_pixels: int
    coverage_ratio: float
    mean_confidence: float
    original_ratio: float
    usable: bool

    def to_dict(self) -> Dict[str, float | int | bool]:
        return {
            "gloss_score": float(self.gloss_score),
            "pore_variance": float(self.pore_variance),
            "homogeneity": float(self.homogeneity),
            "specular_ratio": float(self.specular_ratio),
            "mean_intensity": float(self.mean_intensity),
            "std_intensity": float(self.std_intensity),
            "edge_density": float(self.edge_density),
            "valid_pixels": int(self.valid_pixels),
            "coverage_ratio": float(self.coverage_ratio),
            "mean_confidence": float(self.mean_confidence),
            "original_ratio": float(self.original_ratio),
            "usable": bool(self.usable),
        }


@dataclass
class UVAnalysisBundle:
    uv_texture: np.ndarray
    visible_mask: np.ndarray
    analytic_mask: np.ndarray
    confidence_map: np.ndarray
    is_original_mask: np.ndarray
    coverage_ratio: float
    mean_confidence: float
    original_ratio: float
    usable_for_analysis: bool
    aux: Dict[str, Any]


def _as_bool_mask(mask: Optional[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.zeros(shape, dtype=bool)
    arr = np.asarray(mask)
    if arr.shape != shape:
        raise ValueError(f"mask shape {arr.shape} != expected {shape}")
    if arr.dtype == bool:
        return arr.copy()
    return arr > 0


def _as_float_map(confidence: Optional[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    if confidence is None:
        return np.zeros(shape, dtype=np.float32)
    arr = np.asarray(confidence, dtype=np.float32)
    if arr.shape != shape:
        raise ValueError(f"confidence shape {arr.shape} != expected {shape}")
    return np.clip(arr, 0.0, None)


def enhance_confidence_contrast(
    uv_confidence: np.ndarray,
    uv_mask_visible: np.ndarray,
    *,
    low_percentile: float = 5.0,
    high_percentile: float = 95.0,
    power: float = 1.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Растяжка ``uv_confidence`` по перцентилям **только внутри** ``uv_mask_visible``,
    затем опционально степень ``power > 1`` — сильнее отделяет «уверенные» тексели от слабых
    (удобно для порога в ``build_analytic_uv_mask``).

    Вне маски — нули. Исходная карта не мутируется.
    """
    visible = _as_bool_mask(uv_mask_visible, np.asarray(uv_mask_visible).shape)
    conf = np.clip(np.asarray(uv_confidence, dtype=np.float32), 0.0, 1.0)
    if conf.shape != visible.shape:
        raise ValueError("uv_confidence and uv_mask_visible shape mismatch")
    out = np.zeros_like(conf, dtype=np.float32)
    if not np.any(visible):
        return out
    vals = conf[visible]
    lo = float(np.percentile(vals, float(low_percentile)))
    hi = float(np.percentile(vals, float(high_percentile)))
    hi = max(hi, lo + eps)
    stretched = (conf - lo) / (hi - lo)
    stretched = np.clip(stretched, 0.0, 1.0)
    p = float(power)
    if p != 1.0 and p > 0.0:
        stretched = np.power(stretched, p).astype(np.float32)
    out[visible] = stretched[visible]
    return out


def build_analytic_uv_mask(
    uv_mask_visible: np.ndarray,
    uv_confidence: Optional[np.ndarray] = None,
    uv_is_original: Optional[np.ndarray] = None,
    *,
    min_confidence: float = 0.15,
    require_original: bool = True,
    confidence_contrast: Literal["none", "percentile"] = "none",
    contrast_low_percentile: float = 5.0,
    contrast_high_percentile: float = 95.0,
    contrast_power: float = 1.25,
) -> np.ndarray:
    visible = _as_bool_mask(uv_mask_visible, np.asarray(uv_mask_visible).shape)
    shape = visible.shape
    conf = _as_float_map(uv_confidence, shape)

    if uv_confidence is None:
        conf_ok = visible.copy()
    else:
        if confidence_contrast == "percentile":
            conf_gate = enhance_confidence_contrast(
                conf,
                visible,
                low_percentile=contrast_low_percentile,
                high_percentile=contrast_high_percentile,
                power=contrast_power,
            )
            conf_ok = conf_gate >= float(min_confidence)
        else:
            conf_ok = conf >= float(min_confidence)

    if require_original:
        orig = _as_bool_mask(uv_is_original, shape) if uv_is_original is not None else np.zeros(shape, dtype=bool)
        return visible & conf_ok & orig

    return visible & conf_ok


def build_uv_analysis_bundle(
    uv_texture: np.ndarray,
    uv_mask_visible: np.ndarray,
    uv_confidence: Optional[np.ndarray] = None,
    aux: Optional[Dict[str, Any]] = None,
    *,
    min_confidence: float = 0.15,
    require_original: bool = True,
    min_coverage_ratio: float = 0.03,
    confidence_contrast: Literal["none", "percentile"] = "none",
    contrast_low_percentile: float = 5.0,
    contrast_high_percentile: float = 95.0,
    contrast_power: float = 1.25,
) -> UVAnalysisBundle:
    tex = np.asarray(uv_texture)
    if tex.ndim < 2:
        raise ValueError("uv_texture must have shape HxW or HxWxC")

    shape = tex.shape[:2]
    visible = _as_bool_mask(uv_mask_visible, shape)
    conf = _as_float_map(uv_confidence, shape)
    aux = dict(aux or {})

    if aux.get("uv_is_original") is not None:
        is_original = _as_bool_mask(aux["uv_is_original"], shape)
    else:
        is_original = visible.copy()

    if confidence_contrast == "percentile" and uv_confidence is not None:
        aux["confidence_stretched"] = enhance_confidence_contrast(
            conf,
            visible,
            low_percentile=contrast_low_percentile,
            high_percentile=contrast_high_percentile,
            power=contrast_power,
        )

    analytic = build_analytic_uv_mask(
        visible,
        conf,
        is_original,
        min_confidence=min_confidence,
        require_original=require_original,
        confidence_contrast=confidence_contrast,
        contrast_low_percentile=contrast_low_percentile,
        contrast_high_percentile=contrast_high_percentile,
        contrast_power=contrast_power,
    )

    total_pixels = max(int(shape[0] * shape[1]), 1)
    valid_pixels = int(np.count_nonzero(analytic))
    coverage_ratio = float(valid_pixels / total_pixels)
    mean_conf = float(conf[analytic].mean()) if valid_pixels > 0 else 0.0
    original_ratio = float(is_original[analytic].mean()) if valid_pixels > 0 else 0.0
    usable = bool(valid_pixels > 0 and coverage_ratio >= float(min_coverage_ratio))

    return UVAnalysisBundle(
        uv_texture=tex,
        visible_mask=visible,
        analytic_mask=analytic,
        confidence_map=conf,
        is_original_mask=is_original,
        coverage_ratio=coverage_ratio,
        mean_confidence=mean_conf,
        original_ratio=original_ratio,
        usable_for_analysis=usable,
        aux=aux,
    )


def compute_masked_texture_metrics(
    texture: np.ndarray,
    mask: np.ndarray,
    *,
    confidence_map: Optional[np.ndarray] = None,
    is_original_mask: Optional[np.ndarray] = None,
    min_valid_pixels: int = 256,
) -> UVMaskedTextureMetrics:
    arr = np.asarray(texture)
    if arr.ndim == 3:
        a = arr
        if a.dtype in (np.float32, np.float64) and float(a.max(initial=0.0)) <= 1.0:
            a_u8 = np.clip(a * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            a_u8 = np.clip(a, 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(a_u8, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray = arr.astype(np.float32)

    valid = _as_bool_mask(mask, gray.shape)
    conf = _as_float_map(confidence_map, gray.shape)
    orig = _as_bool_mask(is_original_mask, gray.shape) if is_original_mask is not None else valid.copy()

    valid_pixels = int(np.count_nonzero(valid))
    total_pixels = max(gray.size, 1)
    coverage_ratio = float(valid_pixels / total_pixels)

    if valid_pixels == 0:
        return UVMaskedTextureMetrics(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0,
            coverage_ratio, 0.0, 0.0, False
        )

    vals = gray[valid]
    if vals.max(initial=0.0) > 1.0:
        vals = vals / 255.0

    mean_val = float(vals.mean())
    std_val = float(vals.std())
    p90 = float(np.percentile(vals, 90))
    p99 = float(np.percentile(vals, 99))

    gray_norm = gray / (255.0 if gray.max(initial=0.0) > 1.0 else 1.0)
    gy, gx = np.gradient(gray_norm)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    grad_vals = grad_mag[valid]

    pore_variance = float(np.var(grad_vals)) if grad_vals.size > 0 else 0.0
    edge_density = float(np.mean(grad_vals > np.percentile(grad_vals, 75))) if grad_vals.size > 0 else 0.0
    homogeneity = float(max(0.0, 1.0 - std_val))
    gloss_score = float(min(1.0, max(0.0, p99 - mean_val)))
    specular_ratio = float(np.mean(vals >= max(0.85, p90)))
    mean_conf = float(conf[valid].mean()) if confidence_map is not None else 1.0
    original_ratio = float(orig[valid].mean()) if is_original_mask is not None else 1.0
    usable = bool(valid_pixels >= int(min_valid_pixels))

    return UVMaskedTextureMetrics(
        gloss_score=gloss_score,
        pore_variance=pore_variance,
        homogeneity=homogeneity,
        specular_ratio=specular_ratio,
        mean_intensity=mean_val,
        std_intensity=std_val,
        edge_density=edge_density,
        valid_pixels=valid_pixels,
        coverage_ratio=coverage_ratio,
        mean_confidence=mean_conf,
        original_ratio=original_ratio,
        usable=usable,
    )
