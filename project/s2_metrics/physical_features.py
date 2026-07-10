from __future__ import annotations

import numpy as np
import cv2
from skimage.restoration import estimate_sigma, denoise_wavelet
from skimage.filters import gaussian
from skimage.feature import local_binary_pattern
from skimage.color import rgb2lab
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class PhysicalTextureFeatures:
    """Результат извлечения физических признаков силикона."""
    sss_index: float           # Subsurface scattering (R-B diff на ухе vs щеке)
    specular_sharpness: float  # Резкость бликов (σ градиента на периметре)
    pore_periodicity: float    # Энтропия углового спектра пор
    lbp_nonuniform_ratio: float # Доля non-uniform LBP паттернов
    spectral_slope: float      # β в 1/f^β
    hemoglobin_index: float    # a* в CIELAB на щеке
    seam_score: float          # Резкий скачок текстуры на границе лица
    wrinkle_anisotropy: float = 0.0    # Anisotropy ratio ( доминирующая ориентация / uniform )
    wrinkle_dominant_angle: float = 0.0 # Доминирующий угол морщин (0-180°)


class PhysicalTextureExtractor:
    """
    Извлекает 7 физически обоснованных признаков силикона (из 5.txt §4.3):
    1. Subsurface Scattering (SSS) на тонких участках (ухо/веко/крыло носа)
    2. Резкость спекулярных бликов
    3. Периодичность пор (штамповка)
    4. Доля non-uniform LBP паттернов
    5. Спектральный наклон (1/f^β)
    6. Гемоглобиновый индекс (CIELAB a*)
    7. Шов/стык на границе лица
    """
    
    def __init__(self, face_scale_mm: float = 1.0):
        self.face_scale_mm = face_scale_mm  # масштаб: 1 единица 3DDFA = N мм
    
    def extract(self, image: np.ndarray, landmarks: np.ndarray, 
                seg_mask: np.ndarray) -> PhysicalTextureFeatures:
        """
        image: RGB (H, W, 3) uint8
        landmarks: (68, 2) или (N, 2) лэндмарки
        seg_mask: (H, W) bool - маска кожи (без глаз/рта/волос)
        """
        # Предобработка
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Wavelet denoising перед физическими признаками
        sigma_est = estimate_sigma(gray, channel_axis=None)
        if sigma_est > 2.0:
            gray = np.clip(
                denoise_wavelet(gray, sigma=sigma_est, method="BayesShrink",
                                mode="soft", rescale_sigma=True) * 255,
                0, 255,
            ).astype(np.uint8)

        albedo = self._normalize_illumination(image)
        skin_mask = seg_mask.astype(bool)
        
        # ROIs для анализа
        ear_roi = self._get_ear_roi(landmarks, image.shape[:2])
        cheek_roi = self._get_cheek_roi(landmarks, image.shape[:2])
        forehead_roi = self._get_forehead_roi(landmarks, image.shape[:2])
        nose_roi = self._get_nose_roi(landmarks, image.shape[:2])
        chin_roi = self._get_chin_roi(landmarks, image.shape[:2])

        # Масштаб мм/пиксель
        mm_per_pixel = self._compute_mm_per_pixel(landmarks)

        # 1. Subsurface Scattering (SSS)
        sss_index = self._compute_sss(albedo, ear_roi, cheek_roi, skin_mask)
        
        # 2. Specular sharpness
        specular_sharpness = self._compute_specular_sharpness(albedo, forehead_roi, nose_roi, skin_mask)
        
        # 3. Pore periodicity
        pore_periodicity = self._compute_pore_periodicity(gray, cheek_roi, forehead_roi, skin_mask, mm_per_pixel)
        
        # 4. LBP non-uniform ratio
        lbp_ratio = self._compute_lbp_nonuniform(gray, cheek_roi, forehead_roi, skin_mask)
        
        # 5. Spectral slope
        spectral_slope = self._compute_spectral_slope(gray, cheek_roi, forehead_roi, skin_mask, mm_per_pixel)
        
        # 6. Hemoglobin index
        hemoglobin_index = self._compute_hemoglobin_index(albedo, cheek_roi, skin_mask)
        
        # 7. Seam score
        seam_score = self._compute_seam_score(gray, landmarks, skin_mask)
        
        # 8. Wrinkle anisotropy (Langer's lines)
        wrinkle_anisotropy, wrinkle_dominant_angle = self._compute_wrinkle_anisotropy(
            gray, skin_mask, forehead_roi, cheek_roi
        )

        return PhysicalTextureFeatures(
            sss_index=sss_index,
            specular_sharpness=specular_sharpness,
            pore_periodicity=pore_periodicity,
            lbp_nonuniform_ratio=lbp_ratio,
            spectral_slope=spectral_slope,
            hemoglobin_index=hemoglobin_index,
            seam_score=seam_score,
            wrinkle_anisotropy=wrinkle_anisotropy,
            wrinkle_dominant_angle=wrinkle_dominant_angle,
        )
    
    def _normalize_illumination(self, image: np.ndarray) -> np.ndarray:
        """Нормализация освещения: деление на размытое изображение (σ≈40px)."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        blur = gaussian(gray, sigma=40, preserve_range=True)
        albedo = gray / (blur + 1e-6)
        return np.clip(albedo * 255, 0, 255).astype(np.uint8)
    
    def _get_ear_roi(self, landmarks: np.ndarray, img_shape: Tuple) -> np.ndarray:
        """ROI мочки уха (точки 0-16 для профиля, ищем самую левую/правую точку)."""
        h, w = img_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        # Упрощенно: область вокруг лобуля уха
        leftmost = landmarks[:, 0].min()
        rightmost = landmarks[:, 0].max()
        # Берем область за ухом
        if rightmost - leftmost > 0:
            ear_x = int(rightmost + (rightmost - leftmost) * 0.15)
            ear_y = int(landmarks[8, 1])  # подбородок как ориентир
            cv2.ellipse(mask, (ear_x, ear_y), (30, 40), 0, 0, 360, 255, -1)
        return mask > 0
    
    def _get_cheek_roi(self, landmarks: np.ndarray, img_shape: Tuple) -> np.ndarray:
        """ROI щеки (между глазом и углом рта)."""
        h, w = img_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        # Точки щеки: примерно 1-5 и 11-15
        cheek_pts = np.vstack([landmarks[1:6], landmarks[11:16]])
        if len(cheek_pts) > 3:
            cv2.fillPoly(mask, [cheek_pts.astype(np.int32)], 255)
        return mask > 0
    
    def _get_forehead_roi(self, landmarks: np.ndarray, img_shape: Tuple) -> np.ndarray:
        """ROI лба (над бровями)."""
        h, w = img_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        # Брови: точки 17-26
        brow_y = landmarks[17:27, 1].min()
        forehead_top = max(0, int(brow_y - (landmarks[8, 1] - brow_y) * 0.5))
        cv2.rectangle(mask, (int(landmarks[:,0].min()), forehead_top), 
                      (int(landmarks[:,0].max()), int(brow_y)), 255, -1)
        return mask > 0
    
    def _get_nose_roi(self, landmarks: np.ndarray, img_shape: Tuple) -> np.ndarray:
        """ROI носа (для бликов)."""
        h, w = img_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        nose_pts = landmarks[27:36]
        if len(nose_pts) > 3:
            cv2.fillPoly(mask, [nose_pts.astype(np.int32)], 255)
        return mask > 0
    
    def _get_chin_roi(self, landmarks: np.ndarray, img_shape: Tuple) -> np.ndarray:
        """ROI подбородка."""
        h, w = img_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        chin_pts = landmarks[6:11]
        if len(chin_pts) > 3:
            cv2.fillPoly(mask, [chin_pts.astype(np.int32)], 255)
        return mask > 0

    def _compute_mm_per_pixel(self, landmarks: np.ndarray) -> float:
        """
        Оценка масштаба мм/пиксель через interpupillary distance.
        Среднее расстояние между зрачками взрослого человека: ~63 мм.
        landmarks[36] = left eye outer corner, landmarks[45] = right eye outer corner.
        """
        if self.face_scale_mm != 1.0:
            return self.face_scale_mm
        try:
            # Interpupillary distance (approximate by eye corners)
            left_eye = landmarks[36]
            right_eye = landmarks[45]
            ipd_pixels = np.sqrt(
                (right_eye[0] - left_eye[0])**2 + (right_eye[1] - left_eye[1])**2
            )
            if ipd_pixels < 10:
                return 1.0  # fallback
            # Reference IPD: ~63mm for adult
            mm_per_pixel = 63.0 / ipd_pixels
            return float(np.clip(mm_per_pixel, 0.05, 2.0))  # sanity check
        except Exception:
            return 1.0

    def _compute_sss(self, albedo: np.ndarray, ear_roi: np.ndarray, 
                     cheek_roi: np.ndarray, skin_mask: np.ndarray) -> float:
        """
        Subsurface Scattering: тонкие участки (ухо) просвечивают красным.
        Реальная кожа: R - B на ухе на 12-20% выше чем на щеке.
        Силикон: разница <5%.
        """
        ear_mask = ear_roi & skin_mask
        cheek_mask = cheek_roi & skin_mask
        
        if ear_mask.sum() < 50 or cheek_mask.sum() < 50:
            return 0.0
        
        # Берем RGB из albedo (но albedo это grayscale, нужно оригинальное изображение)
        # Здесь упрощенно: используем intensity
        ear_mean = albedo[ear_mask].mean()
        cheek_mean = albedo[cheek_mask].mean()
        
        if cheek_mean > 0:
            diff = (ear_mean - cheek_mean) / cheek_mean
            return float(np.clip(diff, -1.0, 1.0))
        return 0.0
    
    def _compute_specular_sharpness(self, albedo: np.ndarray, forehead_roi: np.ndarray,
                                     nose_roi: np.ndarray, skin_mask: np.ndarray) -> float:
        """
        Резкость края спекулярных бликов.
        Реальная кожа: блики размытые, σ градиента 2-3 px.
        Силикон: блики зеркальные, σ < 1px.
        """
        h, w = albedo.shape[:2]
        mask = (forehead_roi | nose_roi) & skin_mask
        if mask.sum() < 100:
            return 0.0
        
        # Выделяем блики: I > μ + 3σ на albedo
        roi_vals = albedo[mask]
        mean_val = roi_vals.mean()
        std_val = roi_vals.std()
        highlight_mask = (albedo > mean_val + 3 * std_val) & mask
        
        if highlight_mask.sum() < 20:
            return 0.0
        
        # Градиент на периметре бликов
        grad_x = cv2.Sobel(albedo.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(albedo.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Морфология для получения периметра
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(highlight_mask.astype(np.uint8), kernel)
        perimeter = (dilated - highlight_mask.astype(np.uint8)).astype(bool)
        
        if perimeter.sum() > 0:
            edge_sharpness = grad_mag[perimeter].mean()
            return float(edge_sharpness)
        return 0.0
    
    def _compute_pore_periodicity(self, gray: np.ndarray, cheek_roi: np.ndarray,
                                   forehead_roi: np.ndarray, skin_mask: np.ndarray,
                                   mm_per_pixel: float = 1.0) -> float:
        """
        Периодичность пор: штамповка дает регулярные пики в полярном спектре.
        Энтропия по угловой координате: низкая = регулярный рисунок.
        mm_per_pixel используется для конвертации frequency bins → cycles/mm.
        """
        combined_roi = (cheek_roi | forehead_roi) & skin_mask
        if combined_roi.sum() < 1000:
            return 1.0  # высокая энтропия = естественная кожа
        
        # Извлекаем патчи 64x64 внутри ROI
        ys, xs = np.where(combined_roi)
        if len(xs) < 64:
            return 1.0
        
        # Берем центральный патч
        cy, cx = int(ys.mean()), int(xs.mean())
        patch = gray[cy-32:cy+32, cx-32:cx+32]
        if patch.shape != (64, 64):
            return 1.0
        
        # Окно Ханна
        hanning = np.hanning(64)
        window = np.outer(hanning, hanning)
        patch_w = patch * window
        
        # FFT
        f = np.fft.fft2(patch_w)
        fshift = np.fft.fftshift(f)
        magnitude = np.log(np.abs(fshift) + 1e-6)
        
        # Полярные координаты
        center = (32, 32)
        y_grid, x_grid = np.ogrid[:64, :64]
        theta = np.arctan2(y_grid - center[0], x_grid - center[1]) + np.pi  # 0..2π
        radius = np.sqrt((y_grid - center[0])**2 + (x_grid - center[1])**2)

        # Конвертация радиуса в cycles/mm
        # patch_size = 64 pixels, total field of view = 64 * mm_per_pixel mm
        # frequency_resolution = 1 / (64 * mm_per_pixel) cycles/mm per pixel
        freq_res = 1.0 / (64.0 * mm_per_pixel + 1e-6)
        radius_mm = radius * freq_res  # now in cycles/mm

        # Биннинг по углу (36 бинов по 10°)
        n_bins = 36
        angular_profile = np.zeros(n_bins)
        for i in range(n_bins):
            angle_mask = (theta >= i * 2*np.pi/n_bins) & (theta < (i+1) * 2*np.pi/n_bins)
            # Средние частоты: 0.05-0.3 cycles/mm (поры ~3-20 мм^-1)
            freq_mask = (radius_mm >= 0.05) & (radius_mm <= 0.3)
            combined = angle_mask & freq_mask
            if combined.any():
                angular_profile[i] = magnitude[combined].mean()
        
        # Энтропия углового профиля
        probs = angular_profile / (angular_profile.sum() + 1e-6)
        entropy = -np.sum(probs * np.log(probs + 1e-6))
        max_entropy = np.log(n_bins)
        
        # Нормализованная энтропия: 0 = идеально периодический, 1 = хаотичный
        norm_entropy = entropy / max_entropy
        return float(norm_entropy)
    
    def _compute_lbp_nonuniform(self, gray: np.ndarray, cheek_roi: np.ndarray,
                                 forehead_roi: np.ndarray, skin_mask: np.ndarray) -> float:
        """
        Доля non-uniform LBP паттернов (R=2,3,5).
        Реальная кожа: 30-45% non-uniform.
        Силикон: <20% (слишком регулярная текстура).
        """
        combined_roi = (cheek_roi | forehead_roi) & skin_mask
        if combined_roi.sum() < 500:
            return 0.5
        
        # Мультирадиусный LBP
        radii = [2, 3, 5]
        nonuniform_ratios = []
        
        for r in radii:
            n_points = 8 * r
            lbp = local_binary_pattern(gray, n_points, r, method='nri_uniform')
            
            roi_lbp = lbp[combined_roi]
            if len(roi_lbp) == 0:
                continue
            
            # Uniform patterns: 0..n_points (включая n_points для non-uniform)
            # nri_uniform дает 0..n_points где n_points = non-uniform
            uniform_mask = roi_lbp < n_points
            nonuniform_ratio = 1.0 - (uniform_mask.sum() / len(roi_lbp))
            nonuniform_ratios.append(nonuniform_ratio)
        
        return float(np.mean(nonuniform_ratios)) if nonuniform_ratios else 0.5
    
    def _compute_spectral_slope(self, gray: np.ndarray, cheek_roi: np.ndarray,
                                 forehead_roi: np.ndarray, skin_mask: np.ndarray,
                                 mm_per_pixel: float = 1.0) -> float:
        """
        Спектральный наклон β в 1/f^β.
        Реальная кожа: β≈2.2-2.6.
        Силикон: β>2.8 (слишком гладко) или β<1.8 с пиками (штамповка).
        mm_per_pixel используется для конвертации frequency → cycles/mm.
        """
        combined_roi = (cheek_roi | forehead_roi) & skin_mask
        ys, xs = np.where(combined_roi)
        if len(xs) < 64:
            return 2.5
        
        cy, cx = int(ys.mean()), int(xs.mean())
        patch = gray[cy-32:cy+32, cx-32:cx+32]
        if patch.shape != (64, 64):
            return 2.5
        
        # Radially averaged power spectrum
        f = np.fft.fft2(patch)
        fshift = np.fft.fftshift(f)
        power = np.abs(fshift)**2
        
        # Radial bins
        center = (32, 32)
        y_grid, x_grid = np.ogrid[:64, :64]
        radius = np.sqrt((y_grid - center[0])**2 + (x_grid - center[1])**2)

        # Конвертация в cycles/mm
        freq_res = 1.0 / (64.0 * mm_per_pixel + 1e-6)
        radius_mm = radius * freq_res

        max_r_mm = 32.0 * freq_res  # maximum frequency in cycles/mm
        n_bins = 20
        radial_power = np.zeros(n_bins)
        radial_counts = np.zeros(n_bins)
        
        for i in range(n_bins):
            r1 = i * max_r_mm / n_bins
            r2 = (i + 1) * max_r_mm / n_bins
            mask = (radius_mm >= r1) & (radius_mm < r2)
            if mask.any():
                radial_power[i] = power[mask].mean()
                radial_counts[i] = mask.sum()
        
        # Fit log-log linear regression for r > 2 bins (skip DC and very low freq)
        valid = (radial_counts > 0) & (np.arange(n_bins) >= 2)
        if valid.sum() < 4:
            return 2.5
        
        log_r = np.log(np.arange(n_bins)[valid] * max_r_mm / n_bins + 1e-6)
        log_p = np.log(radial_power[valid] + 1e-6)
        
        slope, _ = np.polyfit(log_r, log_p, 1)
        beta = -slope
        return float(np.clip(beta, 1.0, 4.0))
    
    def _compute_hemoglobin_index(self, albedo: np.ndarray, cheek_roi: np.ndarray,
                                   skin_mask: np.ndarray) -> float:
        """
        Гемоглобиновый индекс: a* в CIELAB на щеке.
        Реальная кожа: a* = 8-15.
        Силикон: a* <5 или >20 с низкой дисперсией.
        """
        # albedo это уже нормализованное, но нужно RGB для LAB
        # Здесь упрощенная версия: используем разность каналов
        return 0.0  # placeholder - нужно исходное RGB изображение
    
    def _compute_seam_score(self, gray: np.ndarray, landmarks: np.ndarray,
                             skin_mask: np.ndarray) -> float:
        """
        Шов по границе челюсти/за ушами.
        Скользящее окно по периметру лица, расстояние между статистиками внутри/снаружи.
        """
        # Упрощенная версия: проверяем границу маски
        kernel = np.ones((5,5), np.uint8)
        dilated = cv2.dilate(skin_mask.astype(np.uint8), kernel, iterations=2)
        eroded = cv2.erode(skin_mask.astype(np.uint8), kernel, iterations=2)
        boundary = (dilated - eroded).astype(bool)
        
        if boundary.sum() < 50:
            return 0.0
        
        inside_vals = gray[skin_mask]
        boundary_vals = gray[boundary]
        outside_vals = gray[~skin_mask & (dilated > 0)]
        
        if len(inside_vals) == 0 or len(outside_vals) == 0:
            return 0.0
        
        # GLCM contrast или просто разница средних
        mean_inside = inside_vals.mean()
        mean_outside = outside_vals.mean()
        seam = abs(mean_inside - mean_outside) / 255.0
        
        return float(np.clip(seam, 0.0, 1.0))

    def _compute_wrinkle_anisotropy(
        self,
        gray: np.ndarray,
        skin_mask: np.ndarray,
        forehead_roi: np.ndarray,
        cheek_roi: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Anisotropy wrinkle direction (Langer's lines).
        Computes gradient orientation histogram within skin regions.
        Returns (anisotropy_ratio, dominant_angle).
        - anisotropy_ratio: 1.0 = uniform, >1.0 = anisotropic (dominant direction)
        - dominant_angle: 0-180 degrees (wrinkle direction)
        """
        try:
            # Combine forehead and cheek ROIs for wrinkle analysis
            analysis_mask = (forehead_roi | cheek_roi) & skin_mask.astype(bool)
            if analysis_mask.sum() < 500:
                return 0.0, 0.0

            # Extract region of interest
            coords = np.argwhere(analysis_mask)
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0) + 1
            roi = gray[y0:y1, x0:x1].astype(np.float64)
            roi_mask = analysis_mask[y0:y1, x0:x1]

            # Sobel gradients
            sobel_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)

            # Gradient orientation (0-180 degrees, undirected)
            angles = np.arctan2(sobel_y, sobel_x)  # -pi to pi
            angles_deg = np.degrees(angles) % 180  # 0-180 degrees

            # Weight by gradient magnitude (stronger edges = more reliable orientation)
            magnitudes = np.sqrt(sobel_x**2 + sobel_y**2)

            # Only use skin pixels with sufficient gradient
            valid = roi_mask & (magnitudes > 5.0)
            if valid.sum() < 100:
                return 0.0, 0.0

            angles_valid = angles_deg[valid]
            mag_valid = magnitudes[valid]

            # Build weighted histogram (18 bins = 10 degrees each)
            n_bins = 18
            hist, bin_edges = np.histogram(
                angles_valid, bins=n_bins, range=(0, 180), weights=mag_valid
            )

            # Normalize
            hist = hist / (hist.sum() + 1e-6)

            # Anisotropy ratio: max(bin) / mean(bin)
            # 1.0 = perfectly uniform, higher = more anisotropic
            mean_hist = hist.mean()
            anisotropy = float(hist.max() / (mean_hist + 1e-6))

            # Dominant angle: center of the tallest bin
            dominant_bin = int(np.argmax(hist))
            dominant_angle = float(bin_edges[dominant_bin] + 5.0)  # center of 10-degree bin

            return float(np.clip(anisotropy, 0.0, 10.0)), float(dominant_angle % 180)

        except Exception:
            return 0.0, 0.0