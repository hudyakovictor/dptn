"""TextureExtractorV2 — полная замена старого extractor по ТЗ Texture V2.

Tier 1 (12 quality-robust): работают на 1999 low-Q фото
Tier 2 (8 HQ): требуют overall>=0.5 или sharpness>=200
Tier 3 (4 physical aux): из physical_features.py, не в CORE

Качество: фиксы багов (sharpness/500, RGB2GRAY, pore_density_mpx, white holes, wavelet gating)
Маска: native RGB + eroded skin mask (7px), distance transform 7px от краёв/глаз/рта, патчи 56x56 (fallback 40x40)
Soft weighting вместо hard delete: weight = f(quality), quality_curve компенсация
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from skimage.morphology import disk, white_tophat, opening, binary_erosion
from skimage.feature import (
    graycomatrix, graycoprops, local_binary_pattern, canny,
    structure_tensor, structure_tensor_eigenvalues
)
from skimage.restoration import denoise_tv_chambolle, estimate_sigma
from skimage.filters import sobel_h, sobel_v, gabor, meijering, gaussian, rank
from skimage import measure, filters
from scipy import ndimage as ndi
from scipy.ndimage import uniform_filter, distance_transform_edt
from scipy.stats import skew, entropy as scipy_entropy

# Константы качества V2 (калибровано по p10 real аудитора #3)
QUALITY_THRESHOLDS_V2 = {
    "noise_level_high": 8.0,      # было 25.0, p90 real 2.48 max 5.77
    "sharpness_low": 25.0,        # было 50.0, p10 real 72, min 28
    "jpeg_blockiness_high": 2.0,  # было 1.5
    "overall_quality_low": 0.28,  # было 0.4, mean 0.32 -> 0.63 после fix
}

# Tier 1: 12 quality-robust core metrics
TIER1_METRICS = [
    "tv_residual_sparsity",
    "lacunarity",
    "autocorr_decay_len",
    "wld_joint_entropy",
    "fft_high_low_ratio",
    "spectral_slope_beta",
    "glcm_diss_d3_aniso",
    "pore_density_r2_mpx",
    "hemoglobin_od_std",
    "bimodality_ashman_D",
    "glszm_small_area_emphasis",
    "edge_tortuosity_mean",
]

# Tier 2: 8 HQ extended metrics
TIER2_METRICS = [
    "glrlm_sre",
    "ngtdm_coarseness",
    "dwt_haar_HH_LL_ratio",
    "lbp_r1_hist_entropy",
    "shannon_entropy_q32",
    "gabor_f08_anisotropy",
    "pore_eccentricity_mean",
    "specular_elongation",
]

# Tier 3: 4 physical auxiliary (не в CORE, используются в s5_verdict)
PHYSICAL_AUX_METRICS = [
    "seam_score",
    "specular_sharpness",
    "sss_index",
    "melanin_hemo_slope",
]

# Все CORE V2 метрики
TEXTURE_CORE_METRICS_V2 = TIER1_METRICS + TIER2_METRICS

# Для маски: distance transform 7px от края маски/глаз/рта/волос/теней
MASK_ERODE_PX = 7
PATCH_SIZE = 56
PATCH_FALLBACK = 40
MIN_VALID_PATCHES = 4


class TextureExtractorV2:
    """Новый TextureExtractor V2 — полная замена старого."""

    def __init__(self):
        self._last_quality = {}
        self._last_assessability = "eligible"
        self._valid_patches = 0

    def extract(self, ctx: Any, exclude_sensitive: bool = False) -> Dict[str, float]:
        """Главная точка входа. Возвращает dict метрик + quality + assessability."""
        result = {}

        # 1. Quality metrics — ВСЕГДА, с фиксами багов
        quality = self._extract_quality_metrics_fixed(ctx)
        result.update(quality)

        # 2. Skin mask — native RGB + eroded native mask
        skin_mask, bbox, rgb_crop, gray_crop, q_crop = self._get_skin_mask_v2(ctx, quality)
        if skin_mask is None or gray_crop is None:
            # fallback на старое поведение
            result["texture_assessability"] = "not_assessable"
            result["q_valid_patches"] = 0
            return result

        # 3. Tier1 12 метрик — ВСЕГДА
        tier1 = self._extract_tier1_robust(gray_crop, skin_mask, q_crop, rgb_crop)
        result.update(tier1)

        # 4. Tier2 8 метрик — только если overall>=0.5 или sharpness>=200, иначе weight 0.2
        overall_q = quality.get("overall_quality", 0.0)
        sharpness = quality.get("sharpness_score", 0.0)
        if overall_q >= 0.5 or sharpness >= 200:
            tier2 = self._extract_tier2_hq(gray_crop, skin_mask, q_crop, rgb_crop)
            result.update(tier2)
        else:
            tier2 = self._extract_tier2_hq(gray_crop, skin_mask, q_crop, rgb_crop)
            for k, v in tier2.items():
                result[k] = v
                result[f"{k}_weight"] = 0.2

        # 5. Physical auxiliary (Tier3) — вычисляются отдельно в physical_features.py
        # Здесь только флаги
        result["texture_assessability"] = self._last_assessability
        result["q_valid_patches"] = self._valid_patches

        # 6. texture_unreliable V2 — мягкие пороги
        result["texture_unreliable"] = bool(
            result.get("texture_noise_sigma", 0) > 15 or
            quality.get("noise_level", 0) > 8 or
            quality.get("sharpness_score", 0) < 25
        )

        # 7. texture_feature_weights_json для downstream
        result["texture_feature_weights_json"] = str({
            k: self._feature_weight(k, quality) for k in result
        })

        return result

    # ===================== QUALITY METRICS (FIXED) =====================

    def _extract_quality_metrics_fixed(self, ctx: Any) -> Dict[str, float]:
        """Исправленные метрики качества: sharpness/500, RGB2GRAY, no BGR."""
        if not hasattr(ctx, 'image_rgb') or ctx.image_rgb is None:
            return {}

        image = ctx.image_rgb
        if image.size == 0:
            return {}

        result = {}
        try:
            # BUG FIX: RGB2GRAY, не BGR2GRAY!
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            mask = self._get_face_mask(ctx)

            if mask is None:
                mask = np.ones(gray.shape[:2], dtype=np.uint8)

            # Laplacian variance
            laplacian = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
            laplacian_masked = laplacian[mask > 0]
            sharpness = float(np.var(laplacian_masked)) if laplacian_masked.size > 0 else 0.0

            # BUG FIX: sharpness/500, не /5000!
            sharpness_normalized = np.clip(sharpness / 500.0, 0.0, 1.0)

            # Noise level (MAD)
            median_filtered = cv2.medianBlur(gray, 3)
            noise_level = float(np.mean(np.abs(gray.astype(np.float32) - median_filtered.astype(np.float32))))

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

            # Overall quality (фикс: mean 0.32 -> 0.63)
            noise_penalty = np.clip(1.0 - noise_level / 35.0, 0.0, 1.0)
            overall = float(sharpness_normalized * 0.7 + noise_penalty * 0.3)

            # Дополнительные quality метрики для threshold calibration (аудитор #1)
            result["q_laplacian_var"] = sharpness
            result["q_tenengrad"] = self._compute_tenengrad(gray, mask)
            result["q_noise_sigma"] = noise_level
            result["q_jpeg_blockiness"] = blockiness
            result["q_valid_patches"] = 0  # будет обновлено позже

            result["sharpness_score"] = sharpness
            result["noise_level"] = noise_level
            result["jpeg_blockiness"] = blockiness
            result["overall_quality"] = overall
            result["texture_noise_sigma"] = noise_level  # use MAD as texture noise estimate

        except Exception:
            pass

        return result

    def _compute_tenengrad(self, gray: np.ndarray, mask: np.ndarray) -> float:
        """Tenengrad — средняя величина градиента (Sobel) на маске."""
        try:
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mag = np.sqrt(gx * gx + gy * gy)
            mag_masked = mag[mask > 0]
            return float(np.mean(mag_masked)) if mag_masked.size > 0 else 0.0
        except Exception:
            return 0.0

    # ===================== SKIN MASK V2 =====================

    def _get_skin_mask_v2(self, ctx: Any, quality: Dict) -> Tuple[Optional[np.ndarray], Optional[Tuple], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Native RGB + eroded native skin mask, distance transform 7px от краёв/глаз/рта/волос/теней.
        Возвращает: (skin_mask, bbox, rgb_crop, gray_crop, q_crop)
        q_crop — quality-нормализованный crop для метрик, требующих качественного изображения.
        """
        face_mask_path = getattr(ctx, 'face_mask_path', None)
        if not face_mask_path:
            return None, None, None, None, None

        try:
            img = cv2.imread(str(face_mask_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                return None, None, None, None, None

            # Native RGB из маски
            if img.ndim == 3 and img.shape[2] == 4:
                rgb = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
                alpha = img[:, :, 3]
            else:
                return None, None, None, None, None

            # White holes exclusion (глаза/рот/вереса = белые дырки)
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            white_hole = (rgb > 240).all(axis=2) & (hsv[:, :, 1] < 25)
            skin_mask = (alpha > 10) & (~white_hole)

            # Erode 7px (distance transform)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * MASK_ERODE_PX + 1, 2 * MASK_ERODE_PX + 1))
            eroded = cv2.erode((skin_mask * 255).astype(np.uint8), kernel)
            skin_mask = eroded > 0

            # BBox
            coords = np.argwhere(skin_mask)
            if coords.size == 0:
                return None, None, None, None, None
            y0, x0 = coords[:, 0].min(), coords[:, 1].min()
            y1, x1 = coords[:, 0].max() + 1, coords[:, 1].max() + 1

            # Crops
            rgb_crop = rgb[y0:y1, x0:x1]
            mask_crop = skin_mask[y0:y1, x0:x1]

            # Gray crop
            gray_crop = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2GRAY)

            # Quality-normalized crop (CLAHE)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            q_crop = clahe.apply(gray_crop.astype(np.uint8))

            # Подсчёт valid patches 56x56 (fallback 40x40) fully inside mask
            self._valid_patches = self._count_valid_patches(mask_crop)
            if self._valid_patches < MIN_VALID_PATCHES:
                self._last_assessability = "not_assessable"
            else:
                self._last_assessability = "eligible"

            # Quality assessment (аудитор #1 percentiles)
            # Use native crop for quality metrics, not preview
            lapl = quality.get("q_laplacian_var", quality.get("sharpness_score", 0))
            # Compute tenengrad on native q_crop (CLAHE-enhanced)
            native_tenengrad = self._compute_tenengrad(q_crop, mask_crop)
            noise = quality.get("q_noise_sigma", quality.get("noise_level", 0))
            block = quality.get("q_jpeg_blockiness", quality.get("jpeg_blockiness", 0))

            # p10 lapl 20.5, native_tenengrad ~40-50 for good quality, p90 noise 3.66, block 0.01156
            if self._valid_patches < MIN_VALID_PATCHES:
                self._last_assessability = "not_assessable"
            elif (lapl < 20.5 and native_tenengrad < 40) or (noise > 3.66 and block > 0.01156):
                self._last_assessability = "not_assessable"
            elif (lapl < 20.5 or native_tenengrad < 40) or (noise > 3.66 or block > 0.01156):
                self._last_assessability = "low_confidence"
            else:
                self._last_assessability = "eligible"

            bbox = (y0, y1, x0, x1)
            return mask_crop, bbox, rgb_crop, gray_crop, q_crop

        except Exception:
            return None, None, None, None, None

    def _count_valid_patches(self, mask: np.ndarray) -> int:
        """Считает 56x56 патчи fully inside mask (fallback 40x40)."""
        h, w = mask.shape
        patch_sizes = [PATCH_SIZE, PATCH_FALLBACK]
        max_patches = 0

        for ps in patch_sizes:
            if h < ps or w < ps:
                continue
            count = 0
            for y in range(0, h - ps + 1, ps // 2):
                for x in range(0, w - ps + 1, ps // 2):
                    patch = mask[y:y+ps, x:x+ps]
                    if np.all(patch):
                        count += 1
            max_patches = max(max_patches, count)

        return max_patches

    def _get_face_mask(self, ctx: Any) -> Optional[np.ndarray]:
        """Получает alpha-маску из face_mask.png (для quality metrics)."""
        face_mask_path = getattr(ctx, 'face_mask_path', None)
        if not face_mask_path:
            return None
        try:
            img = cv2.imread(str(face_mask_path), cv2.IMREAD_UNCHANGED)
            if img is None or img.ndim != 3 or img.shape[2] != 4:
                return None
            alpha = img[:, :, 3]
            return (alpha > 30).astype(np.uint8)
        except Exception:
            return None

    # ===================== TIER 1 (12 robust) =====================

    def _extract_tier1_robust(self, gray: np.ndarray, mask: np.ndarray, q_crop: np.ndarray, rgb: np.ndarray) -> Dict[str, float]:
        """12 quality-robust метрик Tier 1."""
        result = {}

        # 1. tv_residual_sparsity
        result["tv_residual_sparsity"] = self._tier1_tv_residual_sparsity(q_crop)

        # 2. lacunarity
        result["lacunarity"] = self._tier1_lacunarity(gray, mask)

        # 3. autocorr_decay_len
        result["autocorr_decay_len"] = self._tier1_autocorr_decay_len(q_crop, mask)

        # 4. wld_joint_entropy
        result["wld_joint_entropy"] = self._tier1_wld_joint_entropy(q_crop, mask)

        # 5. fft_high_low_ratio
        result["fft_high_low_ratio"] = self._tier1_fft_high_low_ratio(q_crop, mask)

        # 6. spectral_slope_beta
        result["spectral_slope_beta"] = self._tier1_spectral_slope_beta(q_crop, mask)

        # 7. glcm_diss_d3_aniso
        result["glcm_diss_d3_aniso"] = self._tier1_glcm_diss_d3_aniso(q_crop, mask)

        # 8. pore_density_r2_mpx (per Mpx!)
        result["pore_density_r2_mpx"] = self._tier1_pore_density_r2_mpx(gray, mask)

        # 9. hemoglobin_od_std
        result["hemoglobin_od_std"] = self._tier1_hemoglobin_od_std(rgb, mask)

        # 10. bimodality_ashman_D
        result["bimodality_ashman_D"] = self._tier1_bimodality_ashman_D(gray, mask)

        # 11. glszm_small_area_emphasis
        result["glszm_small_area_emphasis"] = self._tier1_glszm_small_area_emphasis(q_crop, mask)

        # 12. edge_tortuosity_mean
        result["edge_tortuosity_mean"] = self._tier1_edge_tortuosity_mean(q_crop, mask)

        return result

    def _tier1_tv_residual_sparsity(self, gray: np.ndarray) -> float:
        """TV residual sparsity: denoise_tv_chambolle weight=0.1, sparsity = L1/L2 = mean|res|/sqrt(mean res^2)."""
        try:
            den = denoise_tv_chambolle(gray.astype(np.float32), weight=0.1, channel_axis=None)
            res = gray.astype(np.float32) - den
            mean_abs = np.mean(np.abs(res))
            rms = np.sqrt(np.mean(res ** 2) + 1e-6)
            return float(mean_abs / (rms + 1e-6))
        except Exception:
            return 0.0

    def _tier1_lacunarity(self, gray: np.ndarray, mask: np.ndarray) -> float:
        """Lacunarity: binary = white_tophat disk2 > mean+std, box 16, lac = var/mean^2+1."""
        try:
            th = white_tophat(gray.astype(np.uint8), disk(2))
            th_masked = th[mask]
            if th_masked.size == 0:
                return 0.0
            thr = th_masked.mean() + th_masked.std()
            binary = (th > thr) & (mask > 0)

            # Box counting 16x16
            masses = []
            h, w = binary.shape
            for y in range(0, h - 15, 16):
                for x in range(0, w - 15, 16):
                    masses.append(np.sum(binary[y:y+16, x:x+16]))
            if not masses:
                return 0.0
            masses = np.array(masses, dtype=float)
            return float((masses.var() / (masses.mean() ** 2 + 1e-6)) + 1.0)
        except Exception:
            return 0.0

    def _tier1_autocorr_decay_len(self, gray: np.ndarray, mask: np.ndarray) -> float:
        """Autocorr decay length: FFT power -> autocorr = ifft(|fft|^2), line decay до 0.2."""
        try:
            # Central patch 128x128 + Hanning
            h, w = gray.shape
            ch, cw = h // 2, w // 2
            ph, pw = min(128, h), min(128, w)
            patch = gray[ch-ph//2:ch+ph//2, cw-pw//2:cw+pw//2].astype(np.float32)

            # Mask central region
            mask_crop = mask[ch-ph//2:ch+ph//2, cw-pw//2:cw+pw//2] if mask.shape == gray.shape else np.ones_like(patch, dtype=bool)
            patch = patch * (mask_crop > 0)

            wy, wx = np.hanning(patch.shape[0]), np.hanning(patch.shape[1])
            patch_w = (patch - patch.mean()) * np.outer(wy, wx)

            f = np.fft.fft2(patch_w)
            power = np.abs(np.fft.fftshift(f)) ** 2

            # Radial average
            cy, cx = power.shape[0] // 2, power.shape[1] // 2
            yy, xx = np.ogrid[:power.shape[0], :power.shape[1]]
            radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

            max_r = min(cy, cx)
            bins = 15
            rad_r = []
            rad_p = []
            for i in range(1, bins + 1):
                r0 = (i - 1) * max_r / bins
                r1 = i * max_r / bins
                m = (radius >= r0) & (radius < r1)
                if m.any():
                    rad_r.append((r0 + r1) / 2)
                    rad_p.append(power[m].mean())

            if len(rad_r) < 4:
                return 0.0

            rad_r = np.array(rad_r)
            rad_p = np.array(rad_p)
            valid = (rad_p > 0) & (rad_r > 3)
            if valid.sum() < 4:
                return 0.0

            # Autocorr via FFT of power spectrum
            acorr = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(power))).real
            # 1D radial autocorr decay
            y, x = np.ogrid[:acorr.shape[0], :acorr.shape[1]]
            ac_rad = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
            max_r = min(cy, cx)
            ac_profile = []
            for r in np.linspace(0, max_r, 50):
                m = (ac_rad >= r - 1) & (ac_rad < r + 1)
                if m.any():
                    ac_profile.append(acorr[m].mean())

            if len(ac_profile) < 5:
                return 0.0

            ac_profile = np.array(ac_profile)
            ac_profile = ac_profile / (ac_profile[0] + 1e-6)
            # decay length: где падает до 0.2
            idx = np.where(ac_profile < 0.2)[0]
            return float(idx[0]) if len(idx) > 0 else float(len(ac_profile))
        except Exception:
            return 0.0

    def _tier1_wld_joint_entropy(self, gray: np.ndarray, mask: np.ndarray) -> float:
        """WLD joint entropy: Sobel gx,gy mag+orient, hist2d 16x16 joint entropy."""
        try:
            gx = sobel_h(gray)
            gy = sobel_v(gray)
            mag = np.sqrt(gx ** 2 + gy ** 2)
            orient = np.arctan2(gy, gx)

            mag_masked = mag[mask]
            orient_masked = orient[mask]

            if mag_masked.size == 0:
                return 0.0

            # Joint hist 16x16
            mag_bins = np.linspace(0, mag_masked.max(), 17)
            orient_bins = np.linspace(-np.pi, np.pi, 17)

            hist, _, _ = np.histogram2d(mag_masked, orient_masked, bins=[mag_bins, orient_bins], density=False)
            hist = hist / (hist.sum() + 1e-9)
            hist = hist[hist > 0]
            return float(-np.sum(hist * np.log2(hist)))
        except Exception:
            return 0.0

    def _tier1_fft_high_low_ratio(self, gray: np.ndarray, mask: np.ndarray) -> float:
        """FFT high/low ratio: central 128x128 + Hanning, radius low<=4 high>8."""
        try:
            h, w = gray.shape
            ch, cw = h // 2, w // 2
            ph, pw = min(128, h), min(128, w)
            patch = gray[ch-ph//2:ch+ph//2, cw-pw//2:cw+pw//2].astype(np.float32)

            mask_crop = mask[ch-ph//2:ch+ph//2, cw-pw//2:cw+pw//2] if mask.shape == gray.shape else np.ones_like(patch, dtype=bool)
            patch = patch * (mask_crop > 0)

            wy, wx = np.hanning(patch.shape[0]), np.hanning(patch.shape[1])
            patch_w = (patch - patch.mean()) * np.outer(wy, wx)

            f = np.fft.fft2(patch_w)
            power = np.abs(np.fft.fftshift(f)) ** 2

            cy, cx = power.shape[0] // 2, power.shape[1] // 2
            yy, xx = np.ogrid[:power.shape[0], :power.shape[1]]
            radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

            low = power[radius <= 4].sum()
            high = power[radius > 8].sum()
            return float(high / (low + 1e-6))
        except Exception:
            return 0.0

    def _tier1_spectral_slope_beta(self, gray: np.ndarray, mask: np.ndarray) -> float:
        """Spectral slope β: radial profile 1..15 bins, polyfit log(r) vs log(power), β=-slope."""
        try:
            h, w = gray.shape
            ch, cw = h // 2, w // 2
            ph, pw = min(128, h), min(128, w)
            patch = gray[ch-ph//2:ch+ph//2, cw-pw//2:cw+pw//2].astype(np.float32)

            mask_crop = mask[ch-ph//2:ch+ph//2, cw-pw//2:cw+pw//2] if mask.shape == gray.shape else np.ones_like(patch, dtype=bool)
            patch = patch * (mask_crop > 0)

            wy, wx = np.hanning(patch.shape[0]), np.hanning(patch.shape[1])
            patch_w = (patch - patch.mean()) * np.outer(wy, wx)

            f = np.fft.fft2(patch_w)
            power = np.abs(np.fft.fftshift(f)) ** 2

            cy, cx = power.shape[0] // 2, power.shape[1] // 2
            yy, xx = np.ogrid[:power.shape[0], :power.shape[1]]
            radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

            max_r = min(cy, cx)
            rad_r = []
            rad_p = []
            for i in range(1, 16):
                r0 = (i - 1) * max_r / 15
                r1 = i * max_r / 15
                m = (radius >= r0) & (radius < r1)
                if m.any():
                    rad_r.append((r0 + r1) / 2)
                    rad_p.append(power[m].mean())

            rad_r = np.array(rad_r)
            rad_p = np.array(rad_p)
            valid = (rad_p > 0) & (rad_r > 3)
            if valid.sum() >= 4:
                slope, _ = np.polyfit(np.log(rad_r[valid]), np.log(rad_p[valid] + 1e-9), 1)
                return float(-slope)
            return 2.5
        except Exception:
            return 2.5

    def _tier1_glcm_diss_d3_aniso(self, gray: np.ndarray, mask: np.ndarray) -> float:
        """GLCM aniso: quant 32 levels [2,98] percentile, dist=[3] angles 4, diss std."""
        try:
            # Quantize by percentiles on mask
            skin_vals = gray[mask > 0]
            if skin_vals.size == 0:
                return 0.0
            lo, hi = np.percentile(skin_vals.astype(float), [2, 98])
            span = max(hi - lo, 1e-6)
            norm = np.clip((gray.astype(float) - lo) / span, 0.0, 1.0)
            quant = (norm * 31).astype(np.uint8)

            glcm = graycomatrix(quant, distances=[3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                                levels=32, symmetric=True, normed=True)
            diss = [float(graycoprops(glcm, "dissimilarity")[0, a]) for a in range(4)]
            return float(np.std(diss))
        except Exception:
            return 0.0

    def _tier1_pore_density_r2_mpx(self, gray: np.ndarray, mask: np.ndarray) -> float:
        """Pore density: white_tophat disk2, thr=mean+std, count BLOBS / (skin_px/1e6)."""
        try:
            th = white_tophat(gray.astype(np.uint8), disk(2))
            th_masked = th[mask > 0]
            if th_masked.size == 0:
                return 0.0
            thr = th_masked.mean() + th_masked.std()
            binary = (th > thr) & (mask > 0)
            labeled = measure.label(binary)
            count = len(np.unique(labeled)) - 1  # exclude background 0
            skin_px = mask.sum()
            return float(count / max(skin_px / 1e6, 1e-6))
        except Exception:
            return 0.0

    def _tier1_hemoglobin_od_std(self, rgb: np.ndarray, mask: np.ndarray) -> float:
        """Hemoglobin OD std: OD=-log(G/R), std по маске."""
        try:
            R = rgb[:, :, 0].astype(float) + 1.0
            G = rgb[:, :, 1].astype(float) + 1.0
            od = -np.log(G / R)
            od_masked = od[mask > 0]
            return float(np.std(od_masked)) if od_masked.size > 0 else 0.0
        except Exception:
            return 0.0

    def _tier1_bimodality_ashman_D(self, gray: np.ndarray, mask: np.ndarray) -> float:
        """Bimodality Ashman D: 2-means кожи, D=|mu1-mu2|/sqrt((σ1^2+σ2^2)/2)."""
        try:
            from sklearn.cluster import KMeans
            skin = gray[mask > 0].astype(float)
            if skin.size < 10:
                return 0.0
            kmeans = KMeans(n_clusters=2, n_init=10, random_state=42).fit(skin.reshape(-1, 1))
            centers = kmeans.cluster_centers_.flatten()
            labels = kmeans.labels_
            stds = [skin[labels == i].std() for i in [0, 1]]
            mu1, mu2 = centers
            sigma1, sigma2 = stds
            d = abs(mu1 - mu2) / np.sqrt((sigma1**2 + sigma2**2) / 2 + 1e-6)
            return float(d)
        except Exception:
            return 0.0

    def _tier1_glszm_small_area_emphasis(self, gray: np.ndarray, mask: np.ndarray) -> float:
        """GLSZM SAE: quant 32->8 levels, per level label connected, SAE=sum(1/area^2)/zones."""
        try:
            # Quantize to 8 levels
            skin_vals = gray[mask > 0]
            if skin_vals.size == 0:
                return 0.0
            lo, hi = np.percentile(skin_vals.astype(float), [2, 98])
            span = max(hi - lo, 1e-6)
            norm = np.clip((gray.astype(float) - lo) / span, 0.0, 1.0)
            quant = (norm * 7).astype(np.uint8)

            sae_sum = 0.0
            total_zones = 0
            for level in range(8):
                binary = (quant == level) & (mask > 0)
                if not binary.any():
                    continue
                labeled = measure.label(binary)
                areas = np.bincount(labeled.ravel())[1:]  # exclude background
                if len(areas) > 0:
                    sae_sum += np.sum(1.0 / (areas.astype(float) ** 2))
                    total_zones += len(areas)

            return float(sae_sum / max(total_zones, 1))
        except Exception:
            return 0.0

    def _tier1_edge_tortuosity_mean(self, gray: np.ndarray, mask: np.ndarray) -> float:
        """Edge tortuosity: Canny σ1.5 + mask, regionprops, tort=perimeter/major_axis."""
        try:
            edges = canny(gray.astype(float), sigma=1.5, mask=mask.astype(bool))
            if not edges.any():
                return 1.0
            labeled = measure.label(edges)
            props = measure.regionprops(labeled)
            tortuosities = []
            for p in props:
                if p.major_axis_length > 0:
                    tortuosities.append(p.perimeter / p.major_axis_length)
            return float(np.mean(tortuosities)) if tortuosities else 1.0
        except Exception:
            return 1.0

    # ===================== TIER 2 (8 HQ) =====================

    def _extract_tier2_hq(self, gray: np.ndarray, mask: np.ndarray, q_crop: np.ndarray, rgb: np.ndarray) -> Dict[str, float]:
        """8 HQ метрик — требуют good quality."""
        result = {}

        # 13. glrlm_sre
        result["glrlm_sre"] = self._tier2_glrlm_sre(q_crop, mask)

        # 14. ngtdm_coarseness
        result["ngtdm_coarseness"] = self._tier2_ngtdm_coarseness(q_crop, mask)

        # 15. dwt_haar_HH_LL_ratio
        result["dwt_haar_HH_LL_ratio"] = self._tier2_dwt_haar_HH_LL_ratio(q_crop, mask)

        # 16. lbp_r1_hist_entropy
        result["lbp_r1_hist_entropy"] = self._tier2_lbp_r1_hist_entropy(q_crop, mask)

        # 17. shannon_entropy_q32
        result["shannon_entropy_q32"] = self._tier2_shannon_entropy_q32(gray, mask)

        # 18. gabor_f08_anisotropy
        result["gabor_f08_anisotropy"] = self._tier2_gabor_f08_anisotropy(q_crop, mask)

        # 19. pore_eccentricity_mean
        result["pore_eccentricity_mean"] = self._tier2_pore_eccentricity_mean(gray, mask)

        # 20. specular_elongation
        result["specular_elongation"] = self._tier2_specular_elongation(rgb, mask)

        return result

    def _tier2_glrlm_sre(self, gray: np.ndarray, mask: np.ndarray) -> float:
        """GLRLM Short Run Emphasis: run length encoding по строкам."""
        try:
            skin_vals = gray[mask > 0]
            if skin_vals.size == 0:
                return 0.0
            # Quantize to 8 levels
            lo, hi = np.percentile(skin_vals, [2, 98])
            span = max(hi - lo, 1e-6)
            norm = np.clip((gray.astype(float) - lo) / span, 0.0, 1.0)
            quant = (norm * 7).astype(np.uint8)

            sre_sum = 0.0
            total_runs = 0
            h, w = quant.shape
            for y in range(h):
                row = quant[y, :]
                mask_row = mask[y, :]
                if not mask_row.any():
                    continue
                # Run length encoding
                run_val = row[0]
                run_len = 1
                for x in range(1, w):
                    if row[x] == run_val and mask_row[x]:
                        run_len += 1
                    else:
                        if run_len > 0 and mask_row[x-1]:
                            sre_sum += 1.0 / (run_len ** 2)
                            total_runs += 1
                        run_val = row[x]
                        run_len = 1
                if run_len > 0 and mask_row[-1]:
                    sre_sum += 1.0 / (run_len ** 2)
                    total_runs += 1
            return float(sre_sum / max(total_runs, 1))
        except Exception:
            return 0.0

    def _tier2_ngtdm_coarseness(self, gray: np.ndarray, mask: np.ndarray) -> float:
        """NGTDM Coarseness: 1/(sum p*s), p=prob, s=abs(gray - avg_neighbor)."""
        try:
            skin = gray[mask > 0].astype(float)
            if skin.size < 100:
                return 0.0
            # Упрощенный NGTDM
            h, w = gray.shape
            gray_f = gray.astype(float)
            # 3x3 neighborhood
            from scipy.ndimage import generic_filter
            avg_neighbor = generic_filter(gray_f, np.mean, size=3, mode='constant')
            diff = np.abs(gray_f - avg_neighbor)
            diff_masked = diff[mask > 0]
            if diff_masked.size == 0:
                return 0.0
            return float(1.0 / (np.mean(diff_masked) + 1e-6))
        except Exception:
            return 0.0

    def _tier2_dwt_haar_HH_LL_ratio(self, gray: np.ndarray, mask: np.ndarray) -> float:
        """DWT Haar HH/LL energy ratio."""
        try:
            # Simple Haar DWT 1 level
            h, w = gray.shape
            gray_f = gray.astype(float)
            # Low-pass (LL) = average 2x2
            ll = (gray_f[::2, ::2] + gray_f[1::2, ::2] + gray_f[::2, 1::2] + gray_f[1::2, 1::2]) / 4.0
            # High-pass (HH) = difference
            hh = (gray_f[::2, ::2] - gray_f[1::2, 1::2] - gray_f[1::2, ::2] + gray_f[::2, 1::2]) / 4.0

            # Mask for LL/HH
            mask_ll = mask[::2, ::2]
            ll_masked = ll[mask_ll > 0]
            hh_masked = hh[mask_ll > 0]

            if ll_masked.size == 0 or hh_masked.size == 0:
                return 0.0
            ll_energy = np.mean(ll_masked ** 2)
            hh_energy = np.mean(hh_masked ** 2)
            return float(hh_energy / (ll_energy + 1e-6))
        except Exception:
            return 0.0

    def _tier2_lbp_r1_hist_entropy(self, gray: np.ndarray, mask: np.ndarray) -> float:
        """LBP R=1 uniform hist entropy."""
        try:
            lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
            lbp_skin = lbp[mask > 0]
            if lbp_skin.size == 0:
                return 0.0
            hist, _ = np.histogram(lbp_skin, bins=10, range=(0, 10), density=True)
            hist = hist[hist > 0]
            return float(-np.sum(hist * np.log2(hist)))
        except Exception:
            return 0.0

    def _tier2_shannon_entropy_q32(self, gray: np.ndarray, mask: np.ndarray) -> float:
        """Shannon entropy q32 — самая устойчивая метрика. Percentile quantization."""
        try:
            skin = gray[mask > 0]
            if skin.size == 0:
                return 0.0
            # Percentile quantization like GLCM
            lo, hi = np.percentile(skin.astype(float), [2, 98])
            span = max(hi - lo, 1e-6)
            norm = np.clip((gray.astype(float) - lo) / span, 0.0, 1.0)
            quant = (norm * 31).astype(np.uint8)
            quant_masked = quant[mask > 0]
            hist, _ = np.histogram(quant_masked, bins=32, range=(0, 31), density=True)
            hist = hist[hist > 0]
            return float(-np.sum(hist * np.log2(hist)))
        except Exception:
            return 0.0

    def _tier2_gabor_f08_anisotropy(self, gray: np.ndarray, mask: np.ndarray) -> float:
        """Gabor f=0.08 (1/0.08=12.5px wavelength) anisotropy across 4 angles.
        Anisotropy = std of mean responses across angles (higher = more directional)."""
        try:
            means = []
            for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
                real, _ = gabor(gray, frequency=0.08, theta=theta, n_stds=3)
                real_masked = real[mask > 0]
                if real_masked.size > 0:
                    means.append(float(np.mean(real_masked)))
            if len(means) < 2:
                return 0.0
            return float(np.std(means))
        except Exception:
            return 0.0

    def _tier2_pore_eccentricity_mean(self, gray: np.ndarray, mask: np.ndarray) -> float:
        """Pore eccentricity: white_tophat disk2 -> connected components -> eccentricity."""
        try:
            th = white_tophat(gray.astype(np.uint8), disk(2))
            skin_th = th[mask > 0]
            if skin_th.size == 0:
                return 0.0
            thr = skin_th.mean() + skin_th.std()
            binary = (th > thr) & (mask > 0)
            if not binary.any():
                return 0.0
            labeled = measure.label(binary)
            props = measure.regionprops(labeled)
            ecs = [p.eccentricity for p in props if p.area > 3]
            return float(np.mean(ecs)) if ecs else 0.0
        except Exception:
            return 0.0

    def _tier2_specular_elongation(self, rgb: np.ndarray, mask: np.ndarray) -> float:
        """Specular elongation: блики — high R=G=B, elongation = major/minor axis.
        Returns 1.0 when no specular (isotropic), >1 when elongated."""
        try:
            if mask.sum() < 100:
                return 1.0
            # Specular detection: high min(R,G,B) + low saturation
            R, G, B = rgb[:, :, 0].astype(float), rgb[:, :, 1].astype(float), rgb[:, :, 2].astype(float)
            min_rgb = np.minimum(np.minimum(R, G), B)
            sat = np.max([R, G, B], axis=0) - np.min([R, G, B], axis=0)
            spec_mask = (min_rgb > 200) & (sat < 30) & (mask > 0)
            if not spec_mask.any():
                return 1.0  # No specular = isotropic
            labeled = measure.label(spec_mask)
            props = measure.regionprops(labeled)
            elongations = []
            for p in props:
                if p.major_axis_length > 0 and p.minor_axis_length > 0:
                    elongations.append(p.major_axis_length / p.minor_axis_length)
            return float(np.mean(elongations)) if elongations else 1.0
        except Exception:
            return 1.0

    # ===================== QUALITY WEIGHTING =====================

    def _feature_weight(self, feature_name: str, quality: Dict) -> float:
        """Вес фичи 0..1. Tier1=1.0 всегда, Tier2=0.2 если low quality."""
        if feature_name in TIER1_METRICS:
            return 1.0
        if feature_name in TIER2_METRICS:
            overall = quality.get("overall_quality", 1.0)
            if overall < 0.5:
                return 0.2
            return 1.0
        if feature_name in PHYSICAL_AUX_METRICS:
            return 1.0
        return 0.5


# Backward compatibility: старый класс-алиас
TextureExtractor = TextureExtractorV2