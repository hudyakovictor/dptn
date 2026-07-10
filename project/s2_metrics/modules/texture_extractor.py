"""Texture extractor - извлечение текстурных метрик кожи и признаков силикона.

Исправлено:
- Используется alpha-маска из face_mask.png (ТОЛЬКО кожа)
- Квантизация GLCM по перцентилям [2, 98] (levels=33)
- CLAHE нормализация освещения
- Patch-based анализ (32x32, 64x64)
- Multi-scale LBP (R=1, R=2)
- Weighted aggregation
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import cv2

BACKEND_ROOT = Path(__file__).resolve().parents[3] / "backend"
DEEPUTIN_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(DEEPUTIN_ROOT) not in sys.path:
    sys.path.insert(0, str(DEEPUTIN_ROOT))

try:
    from metrics.texture_roi import compute as texture_roi_compute
    _HAS_TEXTURE_ROI = True
except ImportError:
    _HAS_TEXTURE_ROI = False

try:
    from shared.utils import image_quality_metrics
    _HAS_QUALITY_UTILS = True
except ImportError:
    _HAS_QUALITY_UTILS = False

# Попытка импортировать backend skin_authenticity
try:
    from pipeline.skin_authenticity.preprocess import (
        load_face_crop, clean_skin_mask, resize_canonical, make_variants
    )
    from pipeline.skin_authenticity.blocks import extract_all_blocks
    from pipeline.skin_authenticity.patches import iter_patches
    from pipeline.skin_authenticity.scorer import SkinAuthenticityScorer
    from pipeline.skin_authenticity.fusion import estimate_quality, block_weights_for_quality
    _HAS_SKIN_AUTHENTICITY = True
except ImportError:
    _HAS_SKIN_AUTHENTICITY = False

# Метрики, которые деградируют при шуме/блюре (из стресс-теста)
QUALITY_SENSITIVE_METRICS = {
    "glcm_dissimilarity_d5_a0",
    "glcm_dissimilarity_d5_a45",
    "glcm_dissimilarity_d5_a135",
    "glcm_dissimilarity_d5_avg",
    "glcm_dissimilarity_d3_a0",
    "glcm_dissimilarity_d3_a135",
    "glcm_dissimilarity_d3_avg",
    "glcm_dissimilarity_d2_a0",
    "glcm_dissimilarity_d2_range",
    "glcm_homogeneity_d5_a0",
    "glcm_homogeneity_d5_a45",
    "glcm_homogeneity_d5_a135",
    "glcm_homogeneity_d5_avg",
    "glcm_homogeneity_d3_a0",
    "homo_local_var_w15_cv",
    "homo_local_var_w31_cv",
    "homo_local_std_w9_mean",
    "homo_local_std_w15_mean",
    "morph_tophat_r4_std",
    "morph_tophat_r8_std",
    "morph_grad_r8_mean",
    "grad_sobel_mag_p90",
    "grad_sobel_mag_skewness",
    "local_entropy_median",
    "local_entropy_iqr",
    "entropy_w7_median",
    "entropy_w9_median",
    "entropy_w11_median",
    "entropy_w15_median",
    "entropy_w15_mean",
    "entropy_w21_median",
    "entropy_w21_mean",
    "lbp_ror_r1_std",
    "lbp_uniform_r1_mean",
    "lbp_complexity_ratio",
    "skin_brightness_std",
    "residual_bio_iqr",
    "residual_bio_p10",
    "residual_bio_p90",
    "residual_bio_mean_abs",
}

# Пороги для фильтрации (из stress test: noise_s5 sigma=32, blur_s5 15x15)
QUALITY_THRESHOLDS = {
    "noise_level_high": 25.0,
    "sharpness_low": 50.0,
    "jpeg_blockiness_high": 1.5,
    "overall_quality_low": 0.4,
}


class TextureExtractor:
    """Извлечение текстурных метрик кожи и silicone-break признаков.

    Исправления:
    - Использует alpha-маску из face_mask.png (ТОЛЬКО кожа)
    - Квантизация GLCM по перцентилям [2, 98] (levels=33)
    - CLAHE нормализация освещения
    - Patch-based анализ (32x32, 64x64)
    - Multi-scale LBP (R=1, R=2)
    """

    def __init__(self):
        self._quality_metrics = None
        self._quality_sensitive_excluded = False
        self._skin_scorer = None
        if _HAS_SKIN_AUTHENTICITY:
            try:
                self._skin_scorer = SkinAuthenticityScorer()
            except Exception:
                pass

    def extract(self, ctx: Any, exclude_sensitive: bool = True) -> dict[str, float]:
        """
        Извлекает все текстурные метрики с учётом качества фото.

        Args:
            ctx: Контекст с данными изображения и маской лица
            exclude_sensitive: Исключить метрики, чувствительные к шуму/блюру

        Returns:
            Словарь {имя_метрики: значение}
        """
        result = {}

        # Сначала извлекаем метрики качества
        quality = self._extract_quality_metrics(ctx)
        result.update(quality)

        # Попытка использовать backend skin_authenticity pipeline
        skin_result = self._extract_via_skin_authenticity(ctx)
        if skin_result:
            result.update(skin_result)
        else:
            # Fallback: используем улучшенный.extract_skin_metrics
            result.update(self.extract_skin_metrics(ctx))
            result.update(self.extract_texture_break_metrics(ctx))

        # Фильтруем чувствительные метрики если качество низкое
        if exclude_sensitive and self._should_exclude_sensitive(quality):
            result = self._filter_sensitive_metrics(result)
            self._quality_sensitive_excluded = True

        # texture_unreliable: флаг для downstream потребителей
        sigma_est = result.get("texture_noise_sigma", 0.0)
        noise_level = quality.get("noise_level", 0.0)
        sharpness = quality.get("sharpness_score", 1000.0)
        result["texture_unreliable"] = bool(
            sigma_est > 15.0
            or noise_level > 25.0
            or sharpness < 50.0
        )

        # Сохраняем веса фич для downstream (аномалия, классификатор)
        self._feature_weights = {
            k: self._feature_weight(k, quality) for k in result
        }
        result["texture_feature_weights_json"] = str(self._feature_weights)

        return result

    def _extract_via_skin_authenticity(self, ctx: Any) -> dict[str, float] | None:
        """Попытка использовать backend skin_authenticity pipeline."""
        if not _HAS_SKIN_AUTHENTICITY or self._skin_scorer is None:
            return None

        face_mask_path = getattr(ctx, 'face_mask_path', None)
        if not face_mask_path:
            return None

        try:
            result = self._skin_scorer.score_path(Path(face_mask_path))
            if result is None:
                return None

            # Конвертируем результат в формат для deeputin
            metrics = {
                "silicone_prob": result.synthetic_score,
                "skin_confidence": result.confidence,
                "skin_verdict": 1.0 if result.verdict == "silicone" else 0.0,
                "skin_quality_index": result.quality.get("overall_score", 0.5),
                "skin_cohort": 0.0 if result.cohort == "modern_live" else 1.0,
            }

            # Добавляем block features если есть
            for block_name, features in result.block_features.items():
                for feat_name, feat_value in features.items():
                    metrics[f"skin_{block_name}_{feat_name}"] = float(feat_value)

            return metrics
        except Exception:
            return None

    def _extract_quality_metrics(self, ctx: Any) -> dict[str, float]:
        """Извлечение метрик качества фото (noise, blur, sharpness, jpeg)."""
        if not hasattr(ctx, 'image_rgb') or ctx.image_rgb is None:
            return {}

        image = ctx.image_rgb
        if image.size == 0:
            return {}

        result = {}

        # Используем проверенную реализацию из shared/utils.py
        if _HAS_QUALITY_UTILS:
            try:
                bbox = getattr(ctx, 'face_bbox', None)
                if bbox and len(bbox) == 4:
                    quality = image_quality_metrics(image, bbox=tuple(bbox))
                else:
                    quality = image_quality_metrics(image)
                result.update(quality)
                return result
            except Exception:
                pass

        # Fallback: собственная реализация
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.astype(np.uint8)

            # Noise level (median absolute deviation)
            median_filtered = cv2.medianBlur(gray, 3)
            noise_level = float(np.mean(np.abs(gray.astype(np.float32) - median_filtered.astype(np.float32))))
            result["noise_level"] = noise_level

            # Sharpness (Laplacian variance)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = float(np.var(laplacian))
            result["sharpness_score"] = sharpness

            # JPEG blockiness
            h, w = gray.shape[:2]
            if h > 16 and w > 16:
                boundary = gray[:, 7::8].astype(np.float32)
                inside = gray[:, 3::8].astype(np.float32)
                if boundary.size > 0 and inside.size > 0:
                    blockiness = float(np.mean(np.abs(boundary - inside))) / 10.0 + 1.0
                else:
                    blockiness = 1.0
            else:
                blockiness = 1.0
            result["jpeg_blockiness"] = blockiness

            # Overall quality score
            sharpness_normalized = np.clip(sharpness / 5000.0, 0.0, 1.0)
            noise_penalty = np.clip(1.0 - noise_level / 35.0, 0.0, 1.0)
            result["overall_quality"] = float(sharpness_normalized * 0.7 + noise_penalty * 0.3)

        except Exception:
            pass

        return result

    def _should_exclude_sensitive(self, quality: dict[str, float]) -> bool:
        """Определяет, нужно ли исключать чувствительные к качеству метрики."""
        if not quality:
            return False

        noise = quality.get("noise_level", 0.0)
        sharpness = quality.get("sharpness_score", 0.0)
        blockiness = quality.get("jpeg_blockiness", 1.0)
        overall = quality.get("overall_quality", 1.0)

        # Исключаем если шум высокий
        if noise > QUALITY_THRESHOLDS["noise_level_high"]:
            return True

        # Исключаем если резкость низкая
        if sharpness < QUALITY_THRESHOLDS["sharpness_low"]:
            return True

        # Исключаем если JPEG-блоки видны
        if blockiness > QUALITY_THRESHOLDS["jpeg_blockiness_high"]:
            return True

        # Исключаем если общее качество низкое
        if overall < QUALITY_THRESHOLDS["overall_quality_low"]:
            return True

        return False

    def _feature_weight(self, feature_name: str, quality: dict[str, float]) -> float:
        """Вес фичи 0..1. 1.0 = полностью надёжна, 0.0 = бесполезна."""
        if feature_name not in QUALITY_SENSITIVE_METRICS:
            # Не-чувствительные фичи всегда полны
            return 1.0
        noise = quality.get("noise_level", 0.0)
        sharpness = quality.get("sharpness_score", 1000.0)
        # Вес падает линейно с ростом шума
        noise_weight = max(0.0, 1.0 - (noise - 5.0) / 30.0)
        # Вес падает с падением sharpness
        sharp_weight = max(0.0, min(1.0, sharpness / 200.0))
        return noise_weight * sharp_weight

    def _filter_sensitive_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        """Удаляет метрики, чувствительные к шуму/блюру."""
        return {k: v for k, v in metrics.items() if k not in QUALITY_SENSITIVE_METRICS}

    def extract_skin_metrics(self, ctx: Any) -> dict[str, float]:
        """Извлечение метрик кожи (цвет, текстура, блеск) с использованием alpha-маски."""
        if not hasattr(ctx, 'image_rgb') or ctx.image_rgb is None:
            return {}

        image = ctx.image_rgb
        if image.size == 0:
            return {}

        # Получаем alpha-маску из face_mask.png
        skin_mask = self._get_skin_mask(ctx)
        if skin_mask is None:
            # Fallback: используем все пиксели
            skin_mask = np.ones(image.shape[:2], dtype=np.uint8)

        result = {}
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image.astype(np.uint8)
            gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)

            # Применяем CLAHE для нормализации освещения
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray_clahe = clahe.apply(gray_u8)

            # Wavelet denoising: убираем плёночное зерно и JPEG-артефакты
            # до вычисления текстурных фич, не размывая микроструктуру пор
            from skimage.restoration import estimate_sigma, denoise_wavelet
            sigma_est = estimate_sigma(gray_clahe, channel_axis=None)
            result["texture_noise_sigma"] = float(sigma_est)
            if sigma_est > 2.0:
                gray_denoised = denoise_wavelet(
                    gray_clahe, sigma=sigma_est,
                    method="BayesShrink", mode="soft",
                    rescale_sigma=True,
                )
                gray_clahe = np.clip(gray_denoised * 255, 0, 255).astype(np.uint8)

            # Используем ТОЛЬКО пиксели кожи
            skin_pixels = gray_u8[skin_mask > 0]
            skin_pixels_clahe = gray_clahe[skin_mask > 0]

            if skin_pixels.size == 0:
                return result

            # Gray stats (только кожа)
            result["texture_gray_mean"] = float(np.mean(skin_pixels))
            result["texture_gray_std"] = float(np.std(skin_pixels))

            # Entropy (только кожа)
            hist, _ = np.histogram(skin_pixels, bins=32, range=(0, 255), density=True)
            hist = hist[hist > 0]
            result["texture_entropy"] = float(-np.sum(hist * np.log2(hist))) if hist.size > 0 else 0.0

            # Laplacian (CLAHE-normalized)
            result["texture_laplacian_var"] = float(cv2.Laplacian(gray_clahe, cv2.CV_64F).var())

            # LBP multi-scale (R=1, R=2)
            from skimage.feature import local_binary_pattern
            lbp1 = local_binary_pattern(gray_clahe, P=8, R=1, method="uniform")
            lbp2 = local_binary_pattern(gray_clahe, P=8, R=2, method="uniform")
            
            # LBP stats только на коже
            lbp1_skin = lbp1[skin_mask > 0]
            lbp2_skin = lbp2[skin_mask > 0]
            
            result["texture_lbp_uniformity"] = float(np.std(lbp1_skin)) if lbp1_skin.size > 0 else 0.0
            result["lbp_uniform_r5_std"] = float(np.std(lbp2_skin)) if lbp2_skin.size > 0 else 0.0

            # lbp_complexity_ratio: доля non-uniform паттернов (R=2, method=nri_uniform)
            # uniform patterns: коды 0..9 для P=8 → count / total
            if lbp2_skin.size > 0:
                nri2 = local_binary_pattern(gray_clahe, P=8, R=2, method="nri_uniform")
                nri2_skin = nri2[skin_mask > 0]
                n_total = nri2_skin.size
                if n_total > 0:
                    hist_nri, _ = np.histogram(nri2_skin, bins=60, range=(0, 60), density=False)
                    # uniform patterns: P+2 patterns (codes 0..P+1 = 0..10)
                    n_uniform = hist_nri[:10].sum()  # codes 0..9
                    result["lbp_complexity_ratio"] = float(1.0 - n_uniform / n_total)
                else:
                    result["lbp_complexity_ratio"] = 0.0
            else:
                result["lbp_complexity_ratio"] = 0.0

            # skin_brightness_std: std яркости в области кожи (для классификатора)
            result["skin_brightness_std"] = float(np.std(skin_pixels.astype(np.float64))) if skin_pixels.size > 0 else 0.0

            # GLCM — квантизация по перцентилям [2, 98] (levels=33)
            from skimage.feature import graycomatrix, graycoprops
            glcm_features = self._compute_glcm_with_percentiles(gray_clahe, skin_mask)
            result.update(glcm_features)

            # homo_local_var_w15_cv, w31_cv — local std / mean в окне (только кожа)
            from scipy.ndimage import uniform_filter
            for w_name, w in [("w15", 15), ("w31", 31)]:
                gray_f = gray_clahe.astype(np.float64)
                local_m = uniform_filter(gray_f, size=w)
                local_m_sq = uniform_filter(gray_f ** 2, size=w)
                local_var = np.maximum(local_m_sq - local_m ** 2, 0)
                local_std = np.sqrt(local_var)
                
                # Только пиксели кожи
                skin_local_m = local_m[skin_mask > 0]
                skin_local_std = local_std[skin_mask > 0]
                
                # Защита от деления на ноль
                valid = skin_local_m > 1.0
                if valid.any():
                    cv_vals = skin_local_std[valid] / skin_local_m[valid]
                    cv_vals = np.clip(cv_vals, 0.0, 10.0)  # Ограничиваем взрывные значения
                    result[f"homo_local_var_{w_name}_cv"] = float(np.mean(cv_vals))
                else:
                    result[f"homo_local_var_{w_name}_cv"] = 0.0

            # contrast_weber_mean
            result["contrast_weber_mean"] = result.get("texture_glcm_contrast", 0.0) / max(result.get("texture_gray_mean", 1.0), 1.0)

            # color_b_mean — blue channel mean (только кожа)
            if len(image.shape) == 3 and image.shape[2] >= 3:
                skin_b = image[:, :, 2][skin_mask > 0]
                result["color_b_mean"] = float(np.mean(skin_b)) if skin_b.size > 0 else 0.0

            # morph_tophat_r4_std, r8_std (только кожа)
            from skimage.morphology import white_tophat, disk
            for r_name, r in [("r4", 4), ("r8", 8)]:
                tophat = white_tophat(gray_clahe, disk(r))
                tophat_skin = tophat[skin_mask > 0]
                result[f"morph_tophat_{r_name}_std"] = float(np.std(tophat_skin.astype(np.float64))) if tophat_skin.size > 0 else 0.0

            # grad_sobel_mag_skewness (только кожа)
            from scipy.stats import skew
            sobel_x = cv2.Sobel(gray_clahe, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray_clahe, cv2.CV_64F, 0, 1, ksize=3)
            mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
            mag_skin = mag[skin_mask > 0]
            result["grad_sobel_mag_skewness"] = float(skew(mag_skin.ravel())) if mag_skin.size > 0 else 0.0

            # residual_bio_iqr — IQR of residual (только кожа)
            from scipy.ndimage import uniform_filter
            bio_approx = uniform_filter(gray_clahe.astype(np.float64), size=15)
            residual = gray_clahe.astype(np.float64) - bio_approx
            residual_skin = residual[skin_mask > 0]
            if residual_skin.size > 0:
                result["residual_bio_iqr"] = float(np.percentile(residual_skin, 75) - np.percentile(residual_skin, 25))
            else:
                result["residual_bio_iqr"] = 0.0

            # FFT — patch-based, не zero-padded (исправляет артефакты)
            # Делим область кожи на патчи и считаем FFT на каждом
            skin_coords = np.argwhere(skin_mask > 0)
            if skin_coords.size >= 64 * 64:
                y0, x0 = skin_coords[:, 0].min(), skin_coords[:, 1].min()
                y1, x1 = skin_coords[:, 0].max() + 1, skin_coords[:, 1].max() + 1
                crop = gray_clahe[y0:y1, x0:x1]
                crop_mask = skin_mask[y0:y1, x0:x1]
                
                # Patch-based FFT (64x64 patches, 50% overlap)
                patch_size = 64
                stride = 32
                high_ratios = []
                peak_ratios = []
                for py in range(0, crop.shape[0] - patch_size + 1, stride):
                    for px in range(0, crop.shape[1] - patch_size + 1, stride):
                        patch = crop[py:py+patch_size, px:px+patch_size].astype(np.float32)
                        patch_mask = crop_mask[py:py+patch_size, px:px+patch_size]
                        if patch_mask.sum() < patch_size * patch_size * 0.5:
                            continue  # недостаточно кожи в патче
                        
                        # Fill non-skin with mean of skin pixels
                        patch_skin = patch[patch_mask > 0]
                        if patch_skin.size == 0:
                            continue
                        patch_mean = patch_skin.mean()
                        patch_filled = patch.copy()
                        patch_filled[patch_mask == 0] = patch_mean
                        
                        patch_filled = patch_filled - patch_filled.mean()
                        f = np.fft.fft2(patch_filled)
                        fshift = np.fft.fftshift(f)
                        magnitude = np.abs(fshift)
                        
                        h, w = magnitude.shape
                        cy, cx = h // 2, w // 2
                        yy, xx = np.ogrid[:h, :w]
                        radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
                        
                        low = magnitude[radius <= 4].sum()
                        high = magnitude[radius > 8].sum()
                        total = magnitude.sum() + 1e-6
                        
                        if low > 0:
                            high_ratios.append(high / low)
                        peak_ratios.append(magnitude.max() / total)
                
                if high_ratios:
                    result["texture_fft_highfreq_ratio"] = float(np.mean(high_ratios))
                    result["texture_fft_highfreq_std"] = float(np.std(high_ratios))
                else:
                    result["texture_fft_highfreq_ratio"] = 0.0
                    result["texture_fft_highfreq_std"] = 0.0
                    
                if peak_ratios:
                    result["texture_fft_peak_ratio"] = float(np.mean(peak_ratios))
                    result["texture_fft_peak_std"] = float(np.std(peak_ratios))
                else:
                    result["texture_fft_peak_ratio"] = 0.0
                    result["texture_fft_peak_std"] = 0.0
            else:
                result["texture_fft_highfreq_ratio"] = 0.0
                result["texture_fft_highfreq_std"] = 0.0
                result["texture_fft_peak_ratio"] = 0.0
                result["texture_fft_peak_std"] = 0.0

            # Edge density (только кожа)
            edges = cv2.Canny(gray_clahe, 40, 120)
            edges_skin = edges[skin_mask > 0]
            result["texture_edge_density"] = float(edges_skin.mean() / 255.0) if edges_skin.size > 0 else 0.0

            # Color stats (только кожа)
            if len(image.shape) == 3 and image.shape[2] >= 3:
                color_pixels = image[skin_mask > 0].astype(np.float32)
                if color_pixels.size > 0:
                    result["texture_specular_ratio"] = float(np.mean((color_pixels.mean(axis=1) > 205) & (color_pixels.std(axis=1) < 28)))
                    result["texture_saturation"] = float(np.mean(np.max(color_pixels, axis=1) - np.min(color_pixels, axis=1)) / 255.0)
                    result["texture_color_std"] = float(np.std(color_pixels))
                else:
                    result["texture_specular_ratio"] = 0.0
                    result["texture_saturation"] = 0.0
                    result["texture_color_std"] = 0.0

            # Color mean (for color_b_mean alias)
            if len(image.shape) == 3 and image.shape[2] >= 3:
                skin_all = image[skin_mask > 0]
                if skin_all.size > 0:
                    result["color_mean"] = float(np.mean(skin_all))
                    result["color_variance"] = float(np.std(skin_all))
                else:
                    result["color_mean"] = 0.0
                    result["color_variance"] = 0.0

            # Albedo viability (L*a*b* analysis) — по ТЗ
            if len(image.shape) == 3 and image.shape[2] >= 3:
                lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                l_chan = lab[:, :, 0][skin_mask > 0]
                a_chan = lab[:, :, 1][skin_mask > 0]
                b_chan = lab[:, :, 2][skin_mask > 0]
                if a_chan.size > 0:
                    result["albedo_a_mean"] = float(np.mean(a_chan))
                    result["albedo_a_std"] = float(np.std(a_chan))
                    result["albedo_b_mean"] = float(np.mean(b_chan))
                    result["albedo_b_std"] = float(np.std(b_chan))
                    result["albedo_l_mean"] = float(np.mean(l_chan))
                    result["albedo_l_std"] = float(np.std(l_chan))
                    # viability index: вариативность a* (гемоглобин) / яркость
                    result["albedo_viability_index"] = float(np.std(a_chan) / (np.mean(l_chan) + 1e-6))
                else:
                    result["albedo_a_mean"] = result["albedo_a_std"] = 0.0
                    result["albedo_b_mean"] = result["albedo_b_std"] = 0.0
                    result["albedo_l_mean"] = result["albedo_l_std"] = 0.0
                    result["albedo_viability_index"] = 0.0

            # Pore analysis — morphological top-hat at multiple radii
            from skimage.morphology import white_tophat, disk
            for r_name, r in [("r2", 2), ("r4", 4), ("r6", 6), ("r8", 8)]:
                tophat = white_tophat(gray_clahe, disk(r))
                tophat_skin = tophat[skin_mask > 0]
                if tophat_skin.size > 0:
                    result[f"pore_tophat_{r_name}_mean"] = float(np.mean(tophat_skin))
                    result[f"pore_tophat_{r_name}_std"] = float(np.std(tophat_skin))
                    # density of pore-like structures
                    threshold = np.mean(tophat_skin) + np.std(tophat_skin)
                    pore_count = np.sum(tophat_skin > threshold)
                    skin_area_cm2 = (skin_mask > 0).sum() * 0.01  # rough calibration
                    result[f"pore_density_{r_name}"] = float(pore_count / max(skin_area_cm2, 1.0))
                else:
                    result[f"pore_tophat_{r_name}_mean"] = 0.0
                    result[f"pore_tophat_{r_name}_std"] = 0.0
                    result[f"pore_density_{r_name}"] = 0.0

            # Specular highlight analysis
            if len(image.shape) == 3 and image.shape[2] >= 3:
                hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                v_skin = hsv[:, :, 2][skin_mask > 0]
                s_skin = hsv[:, :, 1][skin_mask > 0]
                if v_skin.size > 0:
                    # high value, low saturation = specular
                    specular_mask = (v_skin > 220) & (s_skin < 40)
                    result["specular_ratio"] = float(np.mean(specular_mask))
                    # specular patch size (compact vs spread)
                    specular_pixels = v_skin[specular_mask]
                    if specular_pixels.size > 0:
                        result["specular_intensity_mean"] = float(np.mean(specular_pixels))
                        result["specular_intensity_std"] = float(np.std(specular_pixels))
                    else:
                        result["specular_intensity_mean"] = 0.0
                        result["specular_intensity_std"] = 0.0
                else:
                    result["specular_ratio"] = 0.0
                    result["specular_intensity_mean"] = 0.0
                    result["specular_intensity_std"] = 0.0

            # Texture ROI метрики
            if _HAS_TEXTURE_ROI and hasattr(ctx, 'uv_coords') and ctx.uv_coords is not None:
                try:
                    roi_metrics = texture_roi_compute(ctx)
                    for mv in roi_metrics:
                        if mv.value is not None and isinstance(mv.value, (int, float)):
                            result[mv.spec.name] = float(mv.value)
                except Exception:
                    pass

            # Multi-scale texture analysis (гауссова пирамида, 3 уровня)
            result.update(self._extract_multiscale_texture(gray_clahe, skin_mask))

        except Exception:
            pass

        return result

    def _extract_multiscale_texture(
        self, gray_clahe: np.ndarray, skin_mask: np.ndarray
    ) -> dict[str, float]:
        """Multi-scale текстурный анализ: 3 уровня гауссовой пирамиды.
        Уровень 0: оригинал (поры, micro-texture)
        Уровень 1: x0.5 (морфология, морщины)
        Уровень 2: x0.25 (глубокие тени, глобальная структура)
        """
        from skimage.transform import pyramid_gaussian
        from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
        from skimage.morphology import white_tophat, disk
        from scipy.ndimage import uniform_filter

        result = {}
        try:
            pyramid = list(pyramid_gaussian(
                gray_clahe.astype(np.float64) / 255.0,
                downscale=2, multichannel=False,
            ))
            pyramid = pyramid[:3]  # 3 уровня

            skin_mask_f = skin_mask.astype(bool)
            level_weights = [0.5, 0.3, 0.2]

            for level, (img_f, w) in enumerate(zip(pyramid, level_weights)):
                img_u8 = np.clip(img_f * 255, 0, 255).astype(np.uint8)
                # Resize mask to match level dimensions
                if img_u8.shape != skin_mask_f.shape:
                    mask_level = cv2.resize(
                        skin_mask_f.astype(np.uint8),
                        (img_u8.shape[1], img_u8.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                else:
                    mask_level = skin_mask_f

                skin_px = img_u8[mask_level]
                if skin_px.size < 100:
                    continue

                prefix = f"ms_l{level}_"

                # Gray stats
                result[f"{prefix}gray_mean"] = float(np.mean(skin_px)) * w
                result[f"{prefix}gray_std"] = float(np.std(skin_px)) * w

                # LBP uniformity (R=1 на уровне 0, R=2 на уровне 1)
                if level <= 1:
                    R = 1 if level == 0 else 2
                    lbp = local_binary_pattern(img_u8, P=8, R=R, method="uniform")
                    lbp_skin = lbp[mask_level]
                    if lbp_skin.size > 0:
                        result[f"{prefix}lbp_std"] = float(np.std(lbp_skin)) * w

                # GLCM contrast (d=3 на уровне 0, d=5 на уровне 1)
                if level <= 1:
                    d = 3 if level == 0 else 5
                    try:
                        from skimage.feature import graycomatrix, graycoprops
                        # Квантизация по перцентилям
                        p2, p98 = np.percentile(skin_px, [2, 98])
                        if p98 - p2 > 1:
                            quantized = np.clip(
                                (img_u8.astype(float) - p2) / (p98 - p2) * 32, 0, 32
                            ).astype(np.uint8)
                        else:
                            quantized = img_u8
                        glcm = graycomatrix(quantized, [d], [0], levels=33, symmetric=True, normed=True)
                        contrast = graycoprops(glcm, "contrast")[0, 0]
                        result[f"{prefix}glcm_contrast"] = float(contrast) * w
                    except Exception:
                        pass

                # Tophat (pore analysis на уровне 0)
                if level == 0:
                    for r_name, r in [("r2", 2), ("r4", 4)]:
                        try:
                            tophat = white_tophat(img_u8, disk(r))
                            tophat_skin = tophat[mask_level]
                            if tophat_skin.size > 0:
                                result[f"{prefix}pore_{r_name}_std"] = float(np.std(tophat_skin)) * w
                        except Exception:
                            pass

                # Local variability
                for w_name, win in [("w15", 15), ("w31", 31)]:
                    img_f64 = img_u8.astype(np.float64)
                    local_m = uniform_filter(img_f64, size=win)
                    local_m_sq = uniform_filter(img_f64 ** 2, size=win)
                    local_var = np.maximum(local_m_sq - local_m ** 2, 0)
                    local_std = np.sqrt(local_var)
                    skin_local_m = local_m[mask_level]
                    skin_local_std = local_std[mask_level]
                    valid = skin_local_m > 0.01
                    if valid.any():
                        cv_vals = skin_local_std[valid] / skin_local_m[valid]
                        cv_vals = np.clip(cv_vals, 0.0, 10.0)
                        result[f"{prefix}local_var_{w_name}_cv"] = float(np.mean(cv_vals)) * w

            # Weighted aggregation: итоговые фичи = сумма(level_i * weight_i)
            # Уже учтено в умножении на w выше

        except Exception:
            pass

        return result

    def extract_texture_break_metrics(self, ctx: Any) -> dict[str, float]:
        """Извлечение метрик synthetic-break и silicone-suspicion."""
        if not hasattr(ctx, 'image_rgb') or ctx.image_rgb is None:
            return {}

        image = ctx.image_rgb
        if image.size == 0:
            return {}

        result = {}
        try:
            gray = np.mean(image, axis=2) if len(image.shape) == 3 else image.astype(np.float32)

            from scipy import ndimage
            sobel_x = ndimage.sobel(gray, axis=1)
            sobel_y = ndimage.sobel(gray, axis=0)
            gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)

            result["gradient_magnitude_mean"] = float(np.mean(gradient_mag))
            result["gradient_magnitude_max"] = float(np.max(gradient_mag))
            result["gradient_magnitude_p95"] = float(np.percentile(gradient_mag, 95))

            from skimage.filters.rank import entropy
            from skimage.morphology import disk
            try:
                entropy_img = entropy(gray.astype(np.uint8), disk(5))
                result["local_entropy_median"] = float(np.median(entropy_img))
                result["local_entropy_iqr"] = float(np.percentile(entropy_img, 75) - np.percentile(entropy_img, 25))
            except Exception:
                pass

            laplacian = ndimage.laplace(gray)
            result["laplacian_variance"] = float(np.var(laplacian))
        except ImportError:
            pass
        except Exception:
            pass

        return result

    def get_quality_summary(self) -> dict[str, Any]:
        """Возвращает последнюю оценку качества."""
        return {
            "quality_sensitive_excluded": self._quality_sensitive_excluded,
            "thresholds": QUALITY_THRESHOLDS,
        }

    def _get_skin_mask(self, ctx: Any) -> np.ndarray | None:
        """Получает alpha-маску кожи из face_mask.png."""
        face_mask_path = getattr(ctx, 'face_mask_path', None)
        if not face_mask_path:
            return None

        try:
            img = cv2.imread(str(face_mask_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                return None

            if img.ndim == 3 and img.shape[2] == 4:
                alpha = img[:, :, 3]
                skin_mask = (alpha > 30).astype(np.uint8)
                return skin_mask
            else:
                return None
        except Exception:
            return None

    def _compute_glcm_with_percentiles(
        self, gray_u8: np.ndarray, skin_mask: np.ndarray
    ) -> dict[str, float]:
        """Вычисляет GLCM метрики с квантизацией по перцентилям [2, 98] (levels=33)."""
        from skimage.feature import graycomatrix, graycoprops

        result = {}

        # Получаем пиксели кожи
        skin_pixels = gray_u8[skin_mask > 0]
        if skin_pixels.size == 0:
            return result

        # Квантизация по перцентилям [2, 98] (как в backend)
        lo, hi = np.percentile(skin_pixels.astype(np.float64), [2, 98])
        span = max(hi - lo, 1e-6)
        norm = np.clip((gray_u8.astype(np.float64) - lo) / span, 0.0, 1.0)
        quantized = (norm * 32).astype(np.uint8)  # levels=33

        distances = [1, 2, 3, 5]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        levels = 33

        glcm = graycomatrix(
            quantized, distances=distances, angles=angles,
            levels=levels, symmetric=True, normed=True
        )

        angle_names = {0: "a0", 1: "a45", 2: "a90", 3: "a135"}
        for d_idx, d in enumerate(distances):
            for a_idx, a_name in angle_names.items():
                dissim = float(graycoprops(glcm, "dissimilarity")[d_idx, a_idx])
                homo = float(graycoprops(glcm, "homogeneity")[d_idx, a_idx])
                result[f"glcm_dissimilarity_d{d}_{a_name}"] = dissim
                result[f"glcm_homogeneity_d{d}_{a_name}"] = homo

        # glcm_dissimilarity_d5_avg, d3_avg — среднее по углам
        for d in [3, 5]:
            vals = [result.get(f"glcm_dissimilarity_d{d}_a{a}", 0.0) for a in [0, 45, 90, 135]]
            result[f"glcm_dissimilarity_d{d}_avg"] = float(np.mean(vals))

        # glcm_dissimilarity_d2_range = max - min по углам для d=2
        d2_vals = [result.get(f"glcm_dissimilarity_d2_a{a}", 0.0) for a in [0, 45, 90, 135]]
        result["glcm_dissimilarity_d2_range"] = float(max(d2_vals) - min(d2_vals))

        # Базовые GLCM
        result["texture_glcm_contrast"] = float(graycoprops(glcm, "contrast")[0, 0])
        result["texture_glcm_homogeneity"] = float(graycoprops(glcm, "homogeneity")[0, 0])
        result["texture_glcm_energy"] = float(graycoprops(glcm, "energy")[0, 0])

        return result
