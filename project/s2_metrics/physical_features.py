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
    """Результат извлечения физических признаков силикона (Tier 3 V2)."""
    sss_index: float           # Subsurface scattering: R-B diff на ухе vs щеке
    specular_sharpness: float  # Резкость бликов (σ градиента на периметре)
    specular_dispersion: float # Разброс центроидов бликов (mean pairwise dist)
    pore_periodicity: float    # Энтропия углового спектра пор
    lbp_nonuniform_ratio: float # Доля non-uniform LBP паттернов
    spectral_slope: float      # β в 1/f^β
    hemoglobin_od_std: float   # OD=-log(G/R) std по маске (аналог a* в CIELAB)
    seam_score: float          # Скачок текстуры по периметру челюсти/за ушами
    melanin_hemo_slope: float  # slope log(R) vs log(R/G) (V2 aux)
    wrinkle_anisotropy: float = 0.0    # Anisotropy ratio (доминантная ориентация / uniform)
    wrinkle_dominant_angle: float = 0.0 # Доминирующий угол морщин (0-180°)


class PhysicalTextureExtractor:
    """
    Извлекает 7 физически обоснованных признаков силикона (по ТЗ V2, из 5.txt §4.3):
    1. Subsurface Scattering (SSS) на тонких участках (ухо/веко/крыло носа)
    2. Резкость спекулярных бликов + дисперсия центроидов
    3. Периодичность пор (штамповка)
    4. Доля non-uniform LBP паттернов
    5. Спектральный наклон (1/f^β)
    6. Гемоглобиновый индекс (OD=-log(G/R) std, коррелирует с CIELAB a*)
    7. Шов/стык на границе лица (distance transform)
    8. Melanin-hemo slope (log R vs log R/G) — V2 aux
    """

    def __init__(self, face_scale_mm: float = 1.0):
        self.face_scale_mm = face_scale_mm  # масштаб: 1 единица 3DDFA = N мм
    
    def extract(self, image: np.ndarray, landmarks: np.ndarray, 
                seg_mask: np.ndarray, overall_quality: float = 1.0) -> PhysicalTextureFeatures:
        """
        image: RGB (H, W, 3) uint8
        landmarks: (68, 2) или (N, 2) лэндмарки
        seg_mask: (H, W) bool - маска кожи (без глаз/рта/волос)
        overall_quality: качество 0..1 для wavelet gating
        """
        # Предобработка
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Wavelet denoising ТОЛЬКО если overall_quality > 0.5 (фикс баг P1)
        if overall_quality > 0.5:
            sigma_est = estimate_sigma(gray, channel_axis=None)
            if sigma_est > 2.0:
                gray = np.clip(
                    denoise_wavelet(gray, sigma=sigma_est, method="BayesShrink",
                                    mode="soft", rescale_sigma=True) * 255,
                    0, 255,
                ).astype(np.uint8)

        # Illumination normalization
        albedo = self._normalize_illumination(image)
        skin_mask = seg_mask.astype(bool)
        
        # ROIs
        ear_roi = self._get_ear_roi(landmarks, image.shape[:2])
        cheek_roi = self._get_cheek_roi(landmarks, image.shape[:2])
        forehead_roi = self._get_forehead_roi(landmarks, image.shape[:2])
        nose_roi = self._get_nose_roi(landmarks, image.shape[:2])
        jaw_roi = self._get_jaw_roi(landmarks, image.shape[:2])

        # Масштаб мм/пиксель
        mm_per_pixel = self._compute_mm_per_pixel(landmarks)

        # 1. Subsurface Scattering (SSS) - R-B diff на ухе vs щеке
        sss_index = self._compute_sss(image, ear_roi, cheek_roi, skin_mask)
        
        # 2. Specular sharpness + dispersion
        specular_sharpness, specular_dispersion = self._compute_specular_sharpness_dispersion(
            albedo, forehead_roi, nose_roi, skin_mask
        )
        
        # 3. Pore periodicity
        pore_periodicity = self._compute_pore_periodicity(
            gray, cheek_roi, forehead_roi, skin_mask, mm_per_pixel
        )
        
        # 4. LBP non-uniform ratio
        lbp_ratio = self._compute_lbp_nonuniform(
            gray, cheek_roi, forehead_roi, skin_mask
        )
        
        # 5. Spectral slope
        spectral_slope = self._compute_spectral_slope(
            gray, cheek_roi, forehead_roi, skin_mask, mm_per_pixel
        )
        
        # 6. Hemoglobin OD std (replaces placeholder)
        hemoglobin_od_std = self._compute_hemoglobin_od_std(image, cheek_roi, skin_mask)
        
        # 7. Seam score (distance transform + gradient)
        seam_score = self._compute_seam_score(gray, jaw_roi, landmarks, skin_mask)
        
        # 8. Melanin-hemo slope (V2 aux)
        melanin_hemo_slope = self._compute_melanin_hemo_slope(image, cheek_roi, skin_mask)
        
        # 9. Wrinkle anisotropy (bonus)
        wrinkle_anisotropy, wrinkle_dominant_angle = self._compute_wrinkle_anisotropy(
            gray, skin_mask, forehead_roi, cheek_roi
        )

        return PhysicalTextureFeatures(
            sss_index=sss_index,
            specular_sharpness=specular_sharpness,
            specular_dispersion=specular_dispersion,
            pore_periodicity=pore_periodicity,
            lbp_nonuniform_ratio=lbp_ratio,
            spectral_slope=spectral_slope,
            hemoglobin_od_std=hemoglobin_od_std,
            seam_score=seam_score,
            melanin_hemo_slope=melanin_hemo_slope,
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
        """
        ROI мочки уха (используем лобулярную точку, НЕ подбородок landmarks[8]).
        Для 68-точечной модели: точка 0 - край челюсти слева, 16 - справа.
        Ухо примерно за точкой 0/16 на уровне носа (точка 30).
        """
        h, w = img_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if len(landmarks) >= 17:
            leftmost = landmarks[:, 0].min()
            rightmost = landmarks[:, 0].max()
            face_width = rightmost - leftmost
            # Уровень уха ~ уровня носа (точка 30) или чуть выше
            nose_y = landmarks[30, 1] if len(landmarks) > 30 else landmarks[27, 1]
            
            # Правое ухо (для фронтального лица)
            ear_x = int(rightmost + face_width * 0.12)
            ear_y = int(nose_y)
            cv2.ellipse(mask, (ear_x, ear_y), (int(face_width * 0.08), int(face_width * 0.1)), 0, 0, 360, 255, -1)
            
            # Левое ухо
            ear_x_l = int(leftmost - face_width * 0.12)
            cv2.ellipse(mask, (ear_x_l, ear_y), (int(face_width * 0.08), int(face_width * 0.1)), 0, 0, 360, 255, -1)
        
        return mask > 0
    
    def _get_cheek_roi(self, landmarks: np.ndarray, img_shape: Tuple) -> np.ndarray:
        """ROI щеки (ТОЛЬКО чистая щека, БЕЗ губ/носогубных складок)."""
        h, w = img_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if len(landmarks) >= 68:
            # Правая щека: точки 12-14 (угол рта) до 35 (крыло носа), но НЕ включая губы
            # Используем точки 1,2,3,13,14,15 (нижняя челюсть) + область под глазом
            # Берем треугольник: зрачок (36-41) -> крыло носа (31,35) -> угол челюсти (13)
            # ЧИСТАЯ ЩЕКА: между глазом и углом рта, ВЫШЕ носа
            right_cheek_pts = np.array([
                landmarks[36],   # левый глаз внешний угол
                landmarks[39],   # левый глаз внутренний угол  
                landmarks[31],   # носик
                landmarks[35],   # правое крыло носа
                landmarks[13],   # правый угол челюсти
                landmarks[14],   # правый угол челюсти
            ], dtype=np.int32)
            
            left_cheek_pts = np.array([
                landmarks[42],   # правый глаз внешний угол
                landmarks[45],   # правый глаз внутренний угол
                landmarks[31],   # носик
                landmarks[35],   # правое крыло носа
                landmarks[3],    # левый угол челюсти
                landmarks[2],    # левый угол челюсти
            ], dtype=np.int32)
            
            cv2.fillPoly(mask, [right_cheek_pts], 255)
            cv2.fillPoly(mask, [left_cheek_pts], 255)
        
        return mask > 0
    
    def _get_forehead_roi(self, landmarks: np.ndarray, img_shape: Tuple) -> np.ndarray:
        """ROI лба (над бровями, ИСКЛЮЧАЯ волосы - используем distance transform от верхней границы маски)."""
        h, w = img_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if len(landmarks) >= 68:
            # Брови: точки 17-26
            brow_y = landmarks[17:27, 1].min()
            # Верх лица: точка 8 (подбородок) дает масштаб
            chin_y = landmarks[8, 1]
            forehead_top = max(0, int(brow_y - (chin_y - brow_y) * 0.45))
            left_x = int(landmarks[:, 0].min())
            right_x = int(landmarks[:, 0].max())
            cv2.rectangle(mask, (left_x, forehead_top), (right_x, int(brow_y)), 255, -1)
        
        return mask > 0
    
    def _get_nose_roi(self, landmarks: np.ndarray, img_shape: Tuple) -> np.ndarray:
        """ROI носа (для бликов)."""
        h, w = img_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        if len(landmarks) >= 36:
            nose_pts = landmarks[27:36]
            if len(nose_pts) > 3:
                cv2.fillPoly(mask, [nose_pts.astype(np.int32)], 255)
        return mask > 0
    
    def _get_jaw_roi(self, landmarks: np.ndarray, img_shape: Tuple) -> np.ndarray:
        """ROI челюсти/линии подбородка (для seam_score)."""
        h, w = img_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        if len(landmarks) >= 17:
            # Точки 0-16 - линия челюсти
            jaw_pts = landmarks[0:17].astype(np.int32)
            # Расширяем вниз
            jaw_pts_ext = np.vstack([
                jaw_pts,
                [[jaw_pts[-1, 0], jaw_pts[-1, 1] + 30]],
                [[jaw_pts[0, 0], jaw_pts[0, 1] + 30]]
            ])
            cv2.fillPoly(mask, [jaw_pts_ext], 255)
        return mask > 0

    def _compute_mm_per_pixel(self, landmarks: np.ndarray) -> float:
        """
        Оценка масштаба мм/пиксель через interpupillary distance.
        Среднее расстояние между зрачками взрослого: ~63 мм.
        landmarks[36] = левый глаз внешний угол, landmarks[45] = правый глаз внешний угол.
        """
        if self.face_scale_mm != 1.0:
            return self.face_scale_mm
        try:
            left_eye = landmarks[36]
            right_eye = landmarks[45]
            ipd_pixels = np.sqrt(
                (right_eye[0] - left_eye[0])**2 + (right_eye[1] - left_eye[1])**2
            )
            if ipd_pixels < 15:
                return 1.0  # fallback
            mm_per_pixel = 63.0 / ipd_pixels
            # ТЕСНЫЙ клип: для 120px лица IPD~40px -> 1.57, для 600px IPD~200px -> 0.315
            return float(np.clip(mm_per_pixel, 0.15, 1.2))  # было 0.05-2.0
        except Exception:
            return 1.0

    def _compute_sss(self, image: np.ndarray, ear_roi: np.ndarray, 
                     cheek_roi: np.ndarray, skin_mask: np.ndarray) -> float:
        """
        Subsurface Scattering: тонкие участки (ухо) просвечивают красным.
        Реальная кожа: R - B на ухе на 12-20% выше чем на щеке.
        Силикон: разница <5%.
        ИСПРАВЛЕНО: используем R-B diff из RGB, НЕ grayscale intensity!
        """
        ear_mask = ear_roi & skin_mask
        cheek_mask = cheek_roi & skin_mask
        
        if ear_mask.sum() < 50 or cheek_mask.sum() < 50:
            return 0.0
        
        R = image[:, :, 0].astype(float)
        B = image[:, :, 2].astype(float)
        
        ear_rb = (R[ear_mask] - B[ear_mask]).mean()
        cheek_rb = (R[cheek_mask] - B[cheek_mask]).mean()
        
        if cheek_rb != 0:
            diff = (ear_rb - cheek_rb) / (abs(cheek_rb) + 1e-6)
            return float(np.clip(diff, -1.0, 1.0))
        return 0.0
    
    def _compute_specular_sharpness_dispersion(self, albedo: np.ndarray, forehead_roi: np.ndarray,
                                                nose_roi: np.ndarray, skin_mask: np.ndarray) -> Tuple[float, float]:
        """
        Резкость края спекулярных бликов + дисперсия центроидов.
        Реальная кожа: блики размытые, σ градиента 2-3 px, centroids разбросаны.
        Силикон: блики зеркальные, σ < 1px, centroids кластеризованы.
        """
        h, w = albedo.shape[:2]
        mask = (forehead_roi | nose_roi) & skin_mask
        if mask.sum() < 100:
            return 0.0, 0.0
        
        # Выделяем блики: I > μ + 3σ на albedo
        roi_vals = albedo[mask]
        mean_val = roi_vals.mean()
        std_val = roi_vals.std()
        highlight_mask = (albedo > mean_val + 3 * std_val) & mask
        
        if highlight_mask.sum() < 20:
            return 0.0, 0.0
        
        # Градиент на периметре бликов
        grad_x = cv2.Sobel(albedo.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(albedo.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Морфология для получения периметра
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(highlight_mask.astype(np.uint8), kernel)
        perimeter = (dilated - highlight_mask.astype(np.uint8)).astype(bool)
        
        sharpness = 0.0
        centroids = []
        
        if perimeter.sum() > 0:
            edge_sharpness = grad_mag[perimeter].mean()
            sharpness = float(edge_sharpness)
        
        # Centroids of individual highlights
        labeled = cv2.connectedComponents(highlight_mask.astype(np.uint8))[1]
        for i in range(1, labeled.max() + 1):
            coords = np.argwhere(labeled == i)
            if len(coords) > 5:
                centroid = coords.mean(axis=0)[::-1]  # (x, y)
                centroids.append(centroid)
        
        dispersion = 0.0
        if len(centroids) > 1:
            centroids = np.array(centroids)
            # Mean pairwise distance
            from scipy.spatial.distance import pdist
            dists = pdist(centroids)
            dispersion = float(dists.mean())
        
        return sharpness, dispersion

    def _compute_pore_periodicity(self, gray: np.ndarray, cheek_roi: np.ndarray,
                                   forehead_roi: np.ndarray, skin_mask: np.ndarray,
                                   mm_per_pixel: float = 1.0) -> float:
        """
        Периодичность пор: штамповка дает регулярные пики в полярном спектре.
        Энтропия по угловой координате: низкая = регулярный рисунок (silicone).
        mm_per_pixel используется для конвертации frequency bins -> cycles/mm.
        """
        combined_roi = (cheek_roi | forehead_roi) & skin_mask
        if combined_roi.sum() < 1000:
            return 1.0  # высокая энтропия = естественная кожа
        
        ys, xs = np.where(combined_roi)
        if len(xs) < 64:
            return 1.0
        
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
        
        radii = [2, 3, 5]
        nonuniform_ratios = []
        
        for r in radii:
            n_points = 8 * r
            lbp = local_binary_pattern(gray, n_points, r, method='nri_uniform')
            
            roi_lbp = lbp[combined_roi]
            if len(roi_lbp) == 0:
                continue
            
            # Uniform patterns: 0..n_points-1, non-uniform = n_points
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
        
        center = (32, 32)
        y_grid, x_grid = np.ogrid[:64, :64]
        radius = np.sqrt((y_grid - center[0])**2 + (x_grid - center[1])**2)

        # Конвертация в cycles/mm
        freq_res = 1.0 / (64.0 * mm_per_pixel + 1e-6)
        radius_mm = radius * freq_res

        max_r_mm = 32.0 * freq_res
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
    
    def _compute_hemoglobin_od_std(self, image: np.ndarray, cheek_roi: np.ndarray,
                                    skin_mask: np.ndarray) -> float:
        """
        Гемоглобиновый индекс: OD=-log(G/R) std по маске.
        Реальная кожа: OD std > 0.05 (вариация гемоглобина).
        Силикон: OD std ~ 0 (равномерный материал).
        Это аналог CIELAB a* но из RGB напрямую, коррелирует 0.07 с ground truth.
        """
        cheek_mask = cheek_roi & skin_mask
        if cheek_mask.sum() < 50:
            return 0.0
        
        R = image[:, :, 0].astype(float) + 1.0
        G = image[:, :, 1].astype(float) + 1.0
        od = -np.log(G / R)
        od_masked = od[cheek_mask]
        return float(np.std(od_masked)) if od_masked.size > 0 else 0.0
    
    def _compute_seam_score(self, gray: np.ndarray, jaw_roi: np.ndarray,
                            landmarks: np.ndarray, skin_mask: np.ndarray) -> float:
        """
        Шов по границе челюсти/за ушами.
        Distance transform от границы маски + GLCM contrast на boundary.
        Работает даже на low-Q.
        """
        # Distance transform от границы кожи
        skin_uint8 = skin_mask.astype(np.uint8)
        dt = cv2.distanceTransform(skin_uint8, cv2.DIST_L2, 3)
        
        # Boundary: pixels with distance < 5px from edge
        boundary = (dt > 0) & (dt < 5) & skin_mask
        
        if boundary.sum() < 50:
            return 0.0
        
        inside = (dt >= 5) & skin_mask
        outside = (~skin_mask) & (dt < 10)
        
        if inside.sum() == 0 or outside.sum() == 0:
            return 0.0
        
        # GLCM contrast на boundary vs inside
        from skimage.feature import graycomatrix, graycoprops
        
        # Quantize
        skin_vals = gray[skin_mask]
        if skin_vals.size == 0:
            return 0.0
        lo, hi = np.percentile(skin_vals.astype(float), [2, 98])
        span = max(hi - lo, 1e-6)
        norm = np.clip((gray.astype(float) - lo) / span, 0.0, 1.0)
        quant = (norm * 31).astype(np.uint8)
        
        # Boundary patch
        b_coords = np.argwhere(boundary)
        if len(b_coords) > 0:
            y0, x0 = b_coords.min(axis=0)
            y1, x1 = b_coords.max(axis=0) + 1
            patch = quant[y0:y1, x0:x1]
            if patch.shape[0] > 3 and patch.shape[1] > 3:
                glcm = graycomatrix(patch, distances=[3], angles=[0, np.pi/4], 
                                   levels=32, symmetric=True, normed=True)
                contrast = float(graycoprops(glcm, "contrast").mean())
                
                # Inside reference
                i_coords = np.argwhere(inside)
                if len(i_coords) > 0:
                    y0, x0 = i_coords.min(axis=0)
                    y1, x1 = i_coords.max(axis=0) + 1
                    patch_in = quant[y0:y1, x0:x1]
                    if patch_in.shape[0] > 3 and patch_in.shape[1] > 3:
                        glcm_in = graycomatrix(patch_in, distances=[3], angles=[0, np.pi/4],
                                              levels=32, symmetric=True, normed=True)
                        contrast_in = float(graycoprops(glcm_in, "contrast").mean())
                        seam = abs(contrast - contrast_in) / max(contrast_in, 1e-6)
                        return float(np.clip(seam, 0.0, 1.0))
        
        return 0.0
    
    def _compute_melanin_hemo_slope(self, image: np.ndarray, cheek_roi: np.ndarray,
                                     skin_mask: np.ndarray) -> float:
        """
        Melanin-hemo slope: log(R) vs log(R/G) slope.
        corr -0.07 robust (из аудита).
        """
        cheek_mask = cheek_roi & skin_mask
        if cheek_mask.sum() < 100:
            return 0.0
        
        R = image[:, :, 0].astype(float) + 1.0
        G = image[:, :, 1].astype(float) + 1.0
        
        logR = np.log(R[cheek_mask])
        logRG = np.log(R[cheek_mask] / G[cheek_mask])
        
        if len(logR) > 10:
            slope, _ = np.polyfit(logRG, logR, 1)
            return float(slope)
        return 0.0
    
    def _compute_wrinkle_anisotropy(
        self,
        gray: np.ndarray,
        skin_mask: np.ndarray,
        forehead_roi: np.ndarray,
        cheek_roi: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Anisotropy wrinkle direction (Langer's lines).
        Returns (anisotropy_ratio, dominant_angle).
        - anisotropy_ratio: 1.0 = uniform, >1.0 = anisotropic (dominant direction)
        - dominant_angle: 0-180 degrees (wrinkle direction)
        """
        try:
            analysis_mask = (forehead_roi | cheek_roi) & skin_mask.astype(bool)
            if analysis_mask.sum() < 500:
                return 0.0, 0.0

            coords = np.argwhere(analysis_mask)
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0) + 1
            roi = gray[y0:y1, x0:x1].astype(np.float64)
            roi_mask = analysis_mask[y0:y1, x0:x1]

            sobel_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)

            angles = np.arctan2(sobel_y, sobel_x)  # -pi to pi
            angles_deg = np.degrees(angles) % 180  # 0-180 degrees

            magnitudes = np.sqrt(sobel_x**2 + sobel_y**2)

            valid = roi_mask & (magnitudes > 5.0)
            if valid.sum() < 100:
                return 0.0, 0.0

            angles_valid = angles_deg[valid]
            mag_valid = magnitudes[valid]

            n_bins = 18
            hist, bin_edges = np.histogram(
                angles_valid, bins=n_bins, range=(0, 180), weights=mag_valid
            )

            hist = hist / (hist.sum() + 1e-6)

            mean_hist = hist.mean()
            anisotropy = float(hist.max() / (mean_hist + 1e-6))

            dominant_bin = int(np.argmax(hist))
            dominant_angle = float(bin_edges[dominant_bin] + 5.0)  # center of 10-degree bin

            return float(np.clip(anisotropy, 0.0, 10.0)), float(dominant_angle % 180)

        except Exception:
            return 0.0, 0.0