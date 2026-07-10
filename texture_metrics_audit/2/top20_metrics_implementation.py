#!/usr/bin/env python3
"""
Top 20 NEW metrics implementation - correct version
Quality-robust metrics found on simple-test (196 real + 100 silicone)
All via scikit-image, fixes for 1999 false positive

Metrics not in current CORE 20, with thresholds from medians.

Usage:
    from top20_metrics_implementation import Top20Extractor
    extractor = Top20Extractor()
    metrics = extractor.extract_all(face_bgr)  # BGR 500x424 crop
    # metrics dict with 20 values + quality
    decision = extractor.classify(metrics)  # real/silicone + scores

Each function has docstring with formula, skimage usage, and threshold.
"""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.morphology import disk, white_tophat
from skimage.filters.rank import entropy as rank_entropy
from scipy.ndimage import uniform_filter
from typing import Dict, Tuple

class Top20Extractor:
    """
    Extract 20 quality-robust metrics that were NOT in original CORE
    """
    def __init__(self):
        # thresholds from simple-test medians (real vs silicone)
        # rule: if metric > thresh => real (or < for some)
        self.thresholds = {
            # 1
            "rank_entropy_std": {"thresh": 0.63, "higher_is_real": False, "real_med": 0.569, "sil_med": 0.701},
            # 2
            "fft_high_low_ratio": {"thresh": 0.05, "higher_is_real": True, "real_med": 0.087, "sil_med": 0.039},
            # 3
            "glcm_diss_d3_aniso": {"thresh": 0.06, "higher_is_real": False, "real_med": 0.040, "sil_med": 0.072},
            # 4 same as 3
            "glcm_diss_d3_std": {"thresh": 0.06, "higher_is_real": False, "real_med": 0.040, "sil_med": 0.072},
            # 5
            "fft_highfreq_ratio": {"thresh": 0.06, "higher_is_real": True, "real_med": 0.106, "sil_med": 0.044},
            # 6
            "spectral_slope_beta": {"thresh": 3.3, "higher_is_real": False, "real_med": 2.79, "sil_med": 3.36},
            # 7
            "glcm_homo_d3_mean": {"thresh": 0.53, "higher_is_real": False, "real_med": 0.510, "sil_med": 0.556},
            # 8
            "lbp_r1_hist_entropy": {"thresh": 3.19, "higher_is_real": True, "real_med": 3.243, "sil_med": 3.142},
            # 9
            "pore_density_r2": {"thresh": 0.12, "higher_is_real": True, "real_med": 0.128, "sil_med": 0.114},
            # 10
            "glcm_diss_d3_mean": {"thresh": 1.44, "higher_is_real": True, "real_med": 1.515, "sil_med": 1.373},
            # 11
            "pore_density_r4": {"thresh": 0.118, "higher_is_real": True, "real_med": 0.123, "sil_med": 0.113},
            # 12
            "glcm_energy_d1_mean": {"thresh": 0.119, "higher_is_real": False, "real_med": 0.115, "sil_med": 0.123},
            # 13
            "glcm_corr_d3_mean": {"thresh": 0.959, "higher_is_real": False, "real_med": 0.956, "sil_med": 0.961},
            # 14
            "fft_peak_ratio": {"thresh": 0.00025, "higher_is_real": True, "real_med": 0.0003, "sil_med": 0.0002},
            # 15
            "glcm_energy_d3_mean": {"thresh": 0.10, "higher_is_real": False, "real_med": 0.098, "sil_med": 0.103},
            # 16
            "glcm_contr_d3_mean": {"thresh": 5.86, "higher_is_real": True, "real_med": 5.925, "sil_med": 5.798},
            # 17
            "hist_entropy": {"thresh": 3.96, "higher_is_real": False, "real_med": 3.885, "sil_med": 4.035},
            # 18
            "homo_std_w15_mean": {"thresh": 9.95, "higher_is_real": True, "real_med": 10.208, "sil_med": 9.694},
            # 19
            "homo_cv_w31_std": {"thresh": 0.095, "higher_is_real": True, "real_med": 0.099, "sil_med": 0.091},
            # 20
            "homo_cv_w15_mean": {"thresh": 0.08, "higher_is_real": True, "real_med": 0.081, "sil_med": 0.079},
        }

    # ---------- helpers ----------
    @staticmethod
    def create_skin_mask(h: int, w: int):
        """Central ellipse 70% width 80% height - no background"""
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (w//2, h//2), (int(w*0.35), int(h*0.40)), 0, 0, 360, 1, -1)
        return mask.astype(bool)

    @staticmethod
    def quality_metrics(gray: np.ndarray) -> Dict:
        """Blur + overall quality for compensation"""
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        # overall approx 0..1
        overall = float(np.clip(blur/400*0.7+0.3, 0, 1))
        return {"blur": blur, "overall": overall}

    @staticmethod
    def quantize_by_percentiles(gray_u8: np.ndarray, mask: np.ndarray, levels: int = 32):
        """Quantization by [2,98] percentiles as in original code, robust to illumination"""
        skin_pixels = gray_u8[mask]
        if skin_pixels.size == 0:
            return gray_u8
        lo, hi = np.percentile(skin_pixels, [2, 98])
        span = max(hi - lo, 1e-6)
        norm = np.clip((gray_u8.astype(float) - lo) / span, 0, 1)
        quant = (norm * (levels - 1)).astype(np.uint8)
        return quant

    # ---------- 20 metrics implementations ----------

    def metric_rank_entropy_std(self, gray_u8: np.ndarray, mask: np.ndarray) -> float:
        """
        [01] rank_entropy_std
        What: std of local rank entropy in disk(5) window
        skimage: from skimage.filters.rank import entropy as rank_entropy, from skimage.morphology import disk
        Formula: entropy_img = rank_entropy(gray_u8, disk(5)); std = std(entropy_img[mask])
        Real vs Silicone: Real 0.569 vs Sil 0.701, Sil higher (more heterogeneous entropy due to wax?)
        Threshold: <0.63 = real
        Quality robust: corr -0.44 (moderate), CV not measured but rank filter relatively stable
        Why not in CORE: CORE had only skewness, not std
        """
        try:
            ent_img = rank_entropy(gray_u8, disk(5))
            vals = ent_img[mask]
            if vals.size == 0:
                return 0.0
            return float(np.std(vals))
        except Exception:
            return 0.0

    def metric_fft_high_low_ratio(self, gray_u8: np.ndarray, mask: np.ndarray) -> float:
        """
        [02] fft_high_low_ratio = high_power / low_power
        What: high = power radius>8, low = radius<=4 in FFT of central 128x128 patch with Hanning window
        skimage: np.fft.fft2 + np.hanning window (not skimage, but numpy)
        Real vs Sil: Real 0.087 vs Sil 0.039, real higher (more high freq = pores)
        Threshold: >0.05 = real
        Quality robust: corr +0.30 (low), CV 0.455 moderate
        Why not in CORE: No FFT at all in CORE!
        """
        try:
            coords = np.argwhere(mask)
            if coords.shape[0] < 64*64:
                return 0.0
            y0, y1 = coords[:, 0].min(), coords[:, 0].max()+1
            x0, x1 = coords[:, 1].min(), coords[:, 1].max()+1
            crop = gray_u8[y0:y1, x0:x1].astype(float)
            ch, cw = crop.shape[0]//2, crop.shape[1]//2
            ph, pw = min(128, crop.shape[0]), min(128, crop.shape[1])
            if ph < 16 or pw < 16:
                return 0.0
            patch = crop[ch-ph//2:ch+ph//2, cw-pw//2:cw+pw//2]
            wy, wx = np.hanning(patch.shape[0]), np.hanning(patch.shape[1])
            patch_w = (patch - patch.mean()) * np.outer(wy, wx)
            f = np.fft.fft2(patch_w)
            power = np.abs(np.fft.fftshift(f))**2
            h_, w_ = power.shape
            cy, cx = h_//2, w_//2
            yy, xx = np.ogrid[:h_, :w_]
            radius = np.sqrt((yy-cy)**2 + (xx-cx)**2)
            low = power[radius <= 4].sum()
            high = power[radius > 8].sum()
            return float(high / (low + 1e-6)) if low > 0 else 0.0
        except Exception:
            return 0.0

    def metric_glcm_diss_d3_aniso(self, quant: np.ndarray, mask: np.ndarray) -> float:
        """
        [03] glcm_diss_d3_aniso = std(dissimilarity) across 4 angles, distance=3
        What: Measures anisotropy of texture - silicone stamped pores are regular across angles -> high std, real skin isotropic -> low std
        skimage: graycomatrix(distances=[3], angles=[0, pi/4, pi/2, 3pi/4], levels=32, symmetric=True, normed=True) + graycoprops(..., "dissimilarity")
        Real vs Sil: Real 0.040 vs Sil 0.072, Sil higher
        Threshold: <0.06 = real
        Quality robust: corr -0.11 (! very robust), CV 0.226 (stable) -> TOP metric
        Why not in CORE: CORE had only mean a0, a135 etc single angles, not aniso/std across angles
        """
        try:
            coords = np.argwhere(mask)
            y0, y1 = coords[:, 0].min(), coords[:, 0].max()+1
            x0, x1 = coords[:, 1].min(), coords[:, 1].max()+1
            qcrop = quant[y0:y1, x0:x1]
            if qcrop.size < 100:
                return 0.0
            glcm = graycomatrix(qcrop, distances=[3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=32, symmetric=True, normed=True)
            diss = [float(graycoprops(glcm, "dissimilarity")[0, a]) for a in range(4)]
            return float(np.std(diss))
        except Exception:
            return 0.0

    def metric_fft_highfreq_ratio(self, gray_u8: np.ndarray, mask: np.ndarray) -> float:
        """
        [05] fft_highfreq_ratio = high_power / total_power
        Similar to [02] but high/total
        Real 0.106 vs Sil 0.044, >0.06=real
        """
        try:
            coords = np.argwhere(mask)
            y0, y1 = coords[:, 0].min(), coords[:, 0].max()+1
            x0, x1 = coords[:, 1].min(), coords[:, 1].max()+1
            crop = gray_u8[y0:y1, x0:x1].astype(float)
            ch, cw = crop.shape[0]//2, crop.shape[1]//2
            ph, pw = min(128, crop.shape[0]), min(128, crop.shape[1])
            if ph < 16 or pw < 16:
                return 0.0
            patch = crop[ch-ph//2:ch+ph//2, cw-pw//2:cw+pw//2]
            wy, wx = np.hanning(patch.shape[0]), np.hanning(patch.shape[1])
            patch_w = (patch - patch.mean()) * np.outer(wy, wx)
            f = np.fft.fft2(patch_w)
            power = np.abs(np.fft.fftshift(f))**2
            h_, w_ = power.shape
            cy, cx = h_//2, w_//2
            yy, xx = np.ogrid[:h_, :w_]
            radius = np.sqrt((yy-cy)**2 + (xx-cx)**2)
            high = power[radius > 8].sum()
            total = power.sum() + 1e-6
            return float(high / total)
        except Exception:
            return 0.0

    def metric_spectral_slope_beta(self, gray_u8: np.ndarray, mask: np.ndarray) -> float:
        """
        [06] spectral_slope_beta = -slope(log(power) vs log(radius))
        What: 1/f^β, real skin β=2.2-2.6 (natural 1/f), silicone β>3.1 (too smooth, fast drop high freq)
        Real 2.79 vs Sil 3.36, <3.3=real
        Quality robust corr -0.35, CV 0.362 moderate
        Why not in CORE: no spectral slope at all
        """
        try:
            coords = np.argwhere(mask)
            y0, y1 = coords[:, 0].min(), coords[:, 0].max()+1
            x0, x1 = coords[:, 1].min(), coords[:, 1].max()+1
            crop = gray_u8[y0:y1, x0:x1].astype(float)
            ch, cw = crop.shape[0]//2, crop.shape[1]//2
            ph, pw = min(128, crop.shape[0]), min(128, crop.shape[1])
            if ph < 16 or pw < 16:
                return 2.5
            patch = crop[ch-ph//2:ch+ph//2, cw-pw//2:cw+pw//2]
            wy, wx = np.hanning(patch.shape[0]), np.hanning(patch.shape[1])
            patch_w = (patch - patch.mean()) * np.outer(wy, wx)
            f = np.fft.fft2(patch_w)
            power = np.abs(np.fft.fftshift(f))**2
            h_, w_ = power.shape
            cy, cx = h_//2, w_//2
            yy, xx = np.ogrid[:h_, :w_]
            radius = np.sqrt((yy-cy)**2 + (xx-cx)**2)
            max_r = min(h_, w_) // 2
            rad_r = []
            rad_p = []
            for i in range(1, 15):
                r0 = i * max_r / 15
                r1 = (i+1) * max_r / 15
                m = (radius >= r0) & (radius < r1)
                if m.any():
                    rad_r.append((r0+r1)/2)
                    rad_p.append(power[m].mean())
            rad_r = np.array(rad_r); rad_p = np.array(rad_p)
            valid = (rad_p > 0) & (rad_r > 3)
            if valid.sum() >= 4:
                slope, _ = np.polyfit(np.log(rad_r[valid]), np.log(rad_p[valid]+1e-9), 1)
                return float(-slope)
            else:
                return 2.5
        except Exception:
            return 2.5

    def metric_glcm_homo_d3_mean(self, quant: np.ndarray, mask: np.ndarray) -> float:
        """
        [07] homogeneity dist3 mean 4 angles
        Real 0.510 vs Sil 0.556, <0.53=real (sil more homogeneous wax)
        corr -0.42, CV 0.185 stable
        """
        try:
            coords = np.argwhere(mask)
            y0, y1 = coords[:, 0].min(), coords[:, 0].max()+1
            x0, x1 = coords[:, 1].min(), coords[:, 1].max()+1
            qcrop = quant[y0:y1, x0:x1]
            if qcrop.size < 100:
                return 0.0
            glcm = graycomatrix(qcrop, distances=[3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=32, symmetric=True, normed=True)
            homo = [float(graycoprops(glcm, "homogeneity")[0, a]) for a in range(4)]
            return float(np.mean(homo))
        except:
            return 0.0

    def metric_lbp_hist_entropy(self, gray_u8: np.ndarray, mask: np.ndarray) -> float:
        """
        [08] LBP R1 uniform histogram entropy
        Real 3.243 vs Sil 3.142, >3.19=real (real more diverse LBP patterns)
        corr +0.42, CV 0.086 stable
        skimage: local_binary_pattern(P=8,R=1,method="uniform") -> hist 10 bins -> entropy
        """
        try:
            from skimage.feature import local_binary_pattern
            lbp = local_binary_pattern(gray_u8, P=8, R=1, method="uniform")
            vals = lbp[mask]
            if vals.size == 0:
                return 0.0
            hist, _ = np.histogram(vals, bins=10, range=(0, 10), density=True)
            hist = hist[hist > 0]
            return float(-np.sum(hist * np.log2(hist))) if hist.size > 0 else 0.0
        except:
            return 0.0

    def metric_pore_density(self, gray_u8: np.ndarray, mask: np.ndarray, r: int = 2) -> float:
        """
        [09/11] pore_density_r2 / r4
        What: density of pores via white_tophat disk r, count > mean+std / area
        Real r2 0.128 vs Sil 0.114, >0.12=real; r4 0.123 vs 0.113 >0.118=real
        corr +0.09/+0.12 (! very robust), CV 0.242/0.233 stable
        skimage: white_tophat(gray, disk(r))
        Why not in CORE: CORE had only std, not density
        Quality compensation: + k*(0.5-overall) not needed due low corr, but can add
        """
        try:
            th = white_tophat(gray_u8, disk(r))
            vals = th[mask]
            if vals.size == 0:
                return 0.0
            thr = np.mean(vals) + np.std(vals)
            dens = np.sum(vals > thr) / max(mask.sum(), 1)
            return float(dens)
        except:
            return 0.0

    def metric_glcm_diss_d3_mean(self, quant: np.ndarray, mask: np.ndarray) -> float:
        """
        [10] dissimilarity dist3 mean
        Real 1.515 vs Sil 1.373, >1.44=real
        """
        try:
            coords = np.argwhere(mask)
            y0, y1 = coords[:, 0].min(), coords[:, 0].max()+1
            x0, x1 = coords[:, 1].min(), coords[:, 1].max()+1
            qcrop = quant[y0:y1, x0:x1]
            if qcrop.size < 100:
                return 0.0
            glcm = graycomatrix(qcrop, distances=[3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=32, symmetric=True, normed=True)
            diss = [float(graycoprops(glcm, "dissimilarity")[0, a]) for a in range(4)]
            return float(np.mean(diss))
        except:
            return 0.0

    def metric_glcm_energy(self, quant: np.ndarray, mask: np.ndarray, d: int = 1) -> float:
        """
        [12/15] energy (ASM) = sum(p^2), homogeneity measure, silicone higher
        Real d1 0.115 vs Sil 0.123 <0.119=real; d3 0.098 vs 0.103 <0.10=real
        corr -0.30/-0.18 robust, CV 0.157 stable
        Why not in CORE: no energy at all!
        """
        try:
            coords = np.argwhere(mask)
            y0, y1 = coords[:, 0].min(), coords[:, 0].max()+1
            x0, x1 = coords[:, 1].min(), coords[:, 1].max()+1
            qcrop = quant[y0:y1, x0:x1]
            if qcrop.size < 100:
                return 0.0
            glcm = graycomatrix(qcrop, distances=[d], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=32, symmetric=True, normed=True)
            energ = [float(graycoprops(glcm, "energy")[0, a]) for a in range(4)]
            return float(np.mean(energ))
        except:
            return 0.0

    def metric_hist_entropy(self, gray_u8: np.ndarray, mask: np.ndarray) -> float:
        """
        [17] histogram entropy 32 bins
        Real 3.885 vs Sil 4.035, <3.96=real (real slightly less diverse brightness?)
        corr +0.06 super robust, CV 0.007 super stable!
        Why not in CORE: no hist entropy, only texture_entropy (different)
        """
        try:
            pixels = gray_u8[mask]
            if pixels.size == 0:
                return 0.0
            hist, _ = np.histogram(pixels, bins=32, range=(0, 255), density=False)
            prob = hist / hist.sum()
            prob = prob[prob > 0]
            return float(-np.sum(prob * np.log2(prob))) if prob.size>0 else 0.0
        except:
            return 0.0

    def metric_homo_cv(self, gray_u8: np.ndarray, mask: np.ndarray, w: int = 15, stat: str = "mean") -> float:
        """
        [18-20] homo_cv = local_std / local_mean via uniform_filter
        w15 mean Real 0.081 vs Sil 0.079 >0.08=real
        w31 std Real 0.099 vs Sil 0.091 >0.095=real
        w15/w31 small, but CV 0.047-0.096 very stable
        Why not in CORE: CORE had cv w15/w31 but removed for low quality, here we keep with weight
        """
        try:
            gray_f = gray_u8.astype(float)
            lm = uniform_filter(gray_f, size=w)
            lm_sq = uniform_filter(gray_f**2, size=w)
            lvar = np.maximum(lm_sq - lm**2, 0)
            lstd = np.sqrt(lvar)
            vm = mask & (lm > 1)
            if not vm.any():
                return 0.0
            cv = lstd[vm] / lm[vm]
            cv = np.clip(cv, 0, 5)
            if stat == "mean":
                return float(np.mean(cv))
            else:
                return float(np.std(cv))
        except:
            return 0.0

    def metric_glcm_corr(self, quant: np.ndarray, mask: np.ndarray, d: int = 3) -> float:
        """
        [13] correlation dist3 mean, Real 0.956 vs Sil 0.961 <0.959=real
        """
        try:
            coords = np.argwhere(mask)
            y0, y1 = coords[:, 0].min(), coords[:, 0].max()+1
            x0, x1 = coords[:, 1].min(), coords[:, 1].max()+1
            qcrop = quant[y0:y1, x0:x1]
            if qcrop.size < 100:
                return 0.0
            glcm = graycomatrix(qcrop, distances=[d], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=32, symmetric=True, normed=True)
            corr = [float(graycoprops(glcm, "correlation")[0, a]) for a in range(4)]
            return float(np.mean(corr))
        except:
            return 0.0

    # ---------- main extract ----------

    def extract_all(self, img_bgr: np.ndarray) -> Dict[str, float]:
        """Extract all 20 metrics + quality"""
        h, w = img_bgr.shape[:2]
        mask = self.create_skin_mask(h, w)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)

        # quality for compensation
        q = self.quality_metrics(gray_u8)
        overall = q["overall"]

        # quantized once for GLCM
        quant = self.quantize_by_percentiles(gray_u8, mask, levels=32)

        res = {}
        res["overall_quality"] = overall
        res["blur"] = q["blur"]

        # 1
        res["rank_entropy_std"] = self.metric_rank_entropy_std(gray_u8, mask)
        # 2
        res["fft_high_low_ratio"] = self.metric_fft_high_low_ratio(gray_u8, mask)
        # 3,4
        res["glcm_diss_d3_aniso"] = self.metric_glcm_diss_d3_aniso(quant, mask)
        res["glcm_diss_d3_std"] = res["glcm_diss_d3_aniso"]  # same
        # 5
        res["fft_highfreq_ratio"] = self.metric_fft_highfreq_ratio(gray_u8, mask) if "fft_highfreq_ratio" not in res else 0
        # actually we computed high_low, need highfreq separately - reuse same function but we already have highfreq via same calc? Let's recompute quickly via same method
        # For simplicity call same as high_low but with high/total
        # We'll use dedicated func
        # To avoid duplication, we computed high_low already, now compute highfreq via same logic but we need function
        # We'll just call metric that includes both
        # Let's compute both in one go for efficiency - but here separate
        # For brevity, we reuse: fft_highfreq = metric_fft_highfreq (we have method but we called high_low)
        # Actually metric_fft_high_low_ratio and metric_fft_highfreq_ratio share code, we already have high_low, let's compute highfreq via same
        # Workaround: call the same function that returns both - we have only high_low, so we compute highfreq as part of same
        # Let's compute quickly:
        try:
            coords = np.argwhere(mask)
            y0, y1 = coords[:, 0].min(), coords[:, 0].max() + 1
            x0, x1 = coords[:, 1].min(), coords[:, 1].max() + 1
            crop = gray_u8[y0:y1, x0:x1].astype(float)
            ch, cw = crop.shape[0]//2, crop.shape[1]//2
            ph, pw = min(128, crop.shape[0]), min(128, crop.shape[1])
            patch = crop[ch-ph//2:ch+ph//2, cw-pw//2:cw+pw//2]
            wy, wx = np.hanning(patch.shape[0]), np.hanning(patch.shape[1])
            patch_w = (patch - patch.mean()) * np.outer(wy, wx)
            f = np.fft.fft2(patch_w)
            power = np.abs(np.fft.fftshift(f))**2
            h_, w_ = power.shape
            cy, cx = h_//2, w_//2
            yy, xx = np.ogrid[:h_, :w_]
            radius = np.sqrt((yy-cy)**2 + (xx-cx)**2)
            high = power[radius > 8].sum()
            total = power.sum() + 1e-6
            res["fft_highfreq_ratio"] = float(high / total)
        except:
            res["fft_highfreq_ratio"] = 0.0

        # 6
        res["spectral_slope_beta"] = self.metric_spectral_slope_beta(gray_u8, mask)
        # 7
        res["glcm_homo_d3_mean"] = self.metric_glcm_homo_d3_mean(quant, mask) if hasattr(self, 'metric_glcm_homo_d3_mean') else self.metric_glcm_corr(quant, mask, d=3)  # fallback
        # Actually implement homo
        try:
            coords = np.argwhere(mask)
            y0, y1 = coords[:, 0].min(), coords[:, 0].max() + 1
            x0, x1 = coords[:, 1].min(), coords[:, 1].max() + 1
            qcrop = quant[y0:y1, x0:x1]
            glcm = graycomatrix(qcrop, distances=[3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=32, symmetric=True, normed=True)
            homo = [float(graycoprops(glcm, "homogeneity")[0, a]) for a in range(4)]
            res["glcm_homo_d3_mean"] = float(np.mean(homo))
        except:
            res["glcm_homo_d3_mean"] = 0.0

        # 8
        res["lbp_r1_hist_entropy"] = self.metric_lbp_hist_entropy(gray_u8, mask)
        # 9,11
        res["pore_density_r2"] = self.metric_pore_density(gray_u8, mask, r=2)
        res["pore_density_r4"] = self.metric_pore_density(gray_u8, mask, r=4)
        # 10,12,13,15,16
        res["glcm_diss_d3_mean"] = self.metric_glcm_diss_d3_mean(quant, mask)
        res["glcm_energy_d1_mean"] = self.metric_glcm_energy(quant, mask, d=1)
        res["glcm_corr_d3_mean"] = self.metric_glcm_corr(quant, mask, d=3)
        res["glcm_energy_d3_mean"] = self.metric_glcm_energy(quant, mask, d=3)
        res["glcm_contr_d3_mean"] = self.metric_glcm_contr(quant, mask, d=3) if hasattr(self, 'metric_glcm_contr') else 0.0
        # implement contr quickly
        try:
            coords = np.argwhere(mask)
            y0, y1 = coords[:, 0].min(), coords[:, 0].max()+1
            x0, x1 = coords[:,1].min(), coords[:,1].max()+1
            qcrop=quant[y0:y1,x0:x1]
            glcm=graycomatrix(qcrop, distances=[3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=32, symmetric=True, normed=True)
            contr=[float(graycoprops(glcm,"contrast")[0,a]) for a in range(4)]
            res["glcm_contr_d3_mean"]=float(np.mean(contr))
        except:
            res["glcm_contr_d3_mean"]=0.0

        res["fft_peak_ratio"] = 0.0  # placeholder
        try:
            coords = np.argwhere(mask)
            y0, y1 = coords[:, 0].min(), coords[:, 0].max()+1
            x0, x1 = coords[:, 1].min(), coords[:, 1].max()+1
            crop = gray_u8[y0:y1, x0:x1].astype(float)
            ch, cw = crop.shape[0]//2, crop.shape[1]//2
            ph, pw = min(128, crop.shape[0]), min(128, crop.shape[1])
            patch = crop[ch-ph//2:ch+ph//2, cw-pw//2:cw+pw//2]
            wy, wx = np.hanning(patch.shape[0]), np.hanning(patch.shape[1])
            patch_w = (patch - patch.mean()) * np.outer(wy, wx)
            f = np.fft.fft2(patch_w)
            power = np.abs(np.fft.fftshift(f))**2
            res["fft_peak_ratio"] = float(power.max() / (power.sum()+1e-6))
        except:
            res["fft_peak_ratio"]=0.0

        res["hist_entropy"] = self.metric_hist_entropy(gray_u8, mask)
        res["homo_std_w15_mean"] = self.metric_homo_cv(gray_u8, mask, w=15, stat="std_mean")
        res["homo_cv_w31_std"] = self.metric_homo_cv(gray_u8, mask, w=31, stat="cv_std")
        res["homo_cv_w15_mean"] = self.metric_homo_cv(gray_u8, mask, w=15, stat="cv_mean")

        return res

    # helper wrappers for brevity
    def metric_glcm_diss_d3_mean(self, quant, mask):
        try:
            coords=np.argwhere(mask)
            y0,y1=coords[:,0].min(), coords[:,0].max()+1
            x0,x1=coords[:,1].min(), coords[:,1].max()+1
            qcrop=quant[y0:y1,x0:x1]
            glcm=graycomatrix(qcrop, distances=[3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=32, symmetric=True, normed=True)
            diss=[float(graycoprops(glcm,"dissimilarity")[0,a]) for a in range(4)]
            return float(np.mean(diss))
        except:
            return 0.0

    def metric_hist_entropy(self, gray_u8, mask):
        try:
            pixels=gray_u8[mask]
            hist,_=np.histogram(pixels, bins=32, range=(0,255))
            prob=hist/hist.sum()
            prob=prob[prob>0]
            return float(-np.sum(prob*np.log2(prob))) if prob.size>0 else 0.0
        except:
            return 0.0

    def metric_homo_cv(self, gray_u8, mask, w=15, stat="cv_mean"):
        try:
            gray_f=gray_u8.astype(float)
            lm=uniform_filter(gray_f, size=w)
            lm_sq=uniform_filter(gray_f**2, size=w)
            lvar=np.maximum(lm_sq-lm**2,0)
            lstd=np.sqrt(lvar)
            vm=mask & (lm>1)
            if not vm.any():
                return 0.0
            if "cv" in stat:
                cv=lstd[vm]/lm[vm]
                cv=np.clip(cv,0,5)
                if "mean" in stat:
                    return float(np.mean(cv))
                else:
                    return float(np.std(cv))
            else:
                if "mean" in stat:
                    return float(np.mean(lstd[vm]))
                else:
                    return float(np.std(lstd[vm]))
        except:
            return 0.0

    def classify(self, metrics: Dict) -> Dict:
        """Simple voting with thresholds + quality compensation"""
        overall = metrics.get("overall_quality", 0.5)
        # quality compensation for sensitive metrics
        # pore density correction
        pore_r2_corr = metrics.get("pore_density_r2",0) + 0.0  # already robust, no comp needed
        pore_r4_corr = metrics.get("pore_density_r4",0)

        votes=0
        total=0
        decisions={}
        for name, cfg in self.thresholds.items():
            if name not in metrics:
                continue
            val=metrics[name]
            thresh=cfg["thresh"]
            higher=cfg["higher_is_real"]
            # quality compensation for metrics with corr>0.5 (not in top20 robust, but for completeness)
            # For robust list corr<0.45, no compensation needed
            is_real = (val>thresh) if higher else (val<thresh)
            decisions[name]=bool(is_real)
            if is_real:
                votes+=1
            total+=1

        # combined robust score (norm)
        # normalize each metric 0..1 based on thresholds? Simple voting
        voting_score = votes/max(total,1)
        # final label: need >=3 of 4 top robust
        top4 = ["fft_high_low_ratio","fft_highfreq_ratio","spectral_slope_beta","glcm_diss_d3_aniso"]
        top_votes=sum(1 for n in top4 if decisions.get(n,False))
        label = "real" if top_votes>=3 else "silicone"  # 3/4

        # if low quality and predicted silicone, mark uncertain
        is_low = overall<0.35
        if is_low and label=="silicone":
            # check if close to threshold
            label_uncertain = "uncertain"
        else:
            label_uncertain = label

        return {
            "label": label,
            "label_uncertain_aware": label_uncertain,
            "voting_score": voting_score,
            "top4_votes": top_votes,
            "decisions": decisions,
            "overall_quality": overall,
            "is_low_quality": is_low
        }

# Example integration into DPTN
# In s2_metrics/modules/texture_extractor.py:
# from top20_metrics_implementation import Top20Extractor
# extractor = Top20Extractor()
# metrics = extractor.extract_all(face_bgr)
# result = extractor.classify(metrics)
# texture_suspicion = 1 - result["voting_score"]  # 0 real, 1 silicone

if __name__=="__main__":
    import sys
    if len(sys.argv)<2:
        print("Usage: python top20_metrics_implementation.py <image_path>")
        sys.exit(0)
    img=cv2.imread(sys.argv[1])
    ext=Top20Extractor()
    m=ext.extract_all(img)
    print(m)
    print(ext.classify(m))
