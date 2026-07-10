"""
DEEPUTIN — Практические патчи для текущего кода
Копировать/адаптировать в существующие файлы проекта
"""

# =============================================================================
# ПАТЧ 1: s1_extraction/engine.py — интеграция 3DDFA-V3 (минимальный вариант)
# =============================================================================

from pathlib import Path
import sys
import pickle
import numpy as np
import cv2

# Добавляем путь к 3DDFA-V3 (предполагается, что склонирован в core/3ddfa_v3)
TDDFA_ROOT = Path(__file__).resolve().parents[2] / "core" / "3ddfa_v3"
if str(TDDFA_ROOT) not in sys.path:
    sys.path.insert(0, str(TDDFA_ROOT))

# Пытаемся импортировать 3DDFA-V3
try:
    from tddfa_v3 import TDDFA_V3  # псевдоним — заменить на реальный импорт
    _HAS_TDDFA = True
except ImportError:
    _HAS_TDDFA = False
    print("[WARN] 3DDFA-V3 не найдена, используем placeholder")


class ExtractionEngine:
    def __init__(self, input_dir, output_dir, dataset, limit=None, config=None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.dataset = dataset
        self.limit = limit
        self.config = config or {}

        # Инициализация 3DDFA-V3
        self.tddfa = None
        if _HAS_TDDFA:
            checkpoint = self.config.get("tddfa_checkpoint", TDDFA_ROOT / "weights" / "3ddfa_v3.pth")
            cfg_path = self.config.get("tddfa_config", TDDFA_ROOT / "configs" / "tddfa_v3.yml")
            try:
                self.tddfa = TDDFA_V3(str(cfg_path), str(checkpoint))
            except Exception as e:
                print(f"[WARN] Не удалось инициализировать 3DDFA-V3: {e}")

    def _build_reconstruction(self, image_rgb, bbox, photo_path):
        """Возвращает реальную 3D реконструкцию или placeholder."""
        if self.tddfa is not None:
            try:
                rec = self._tddfa_reconstruct(image_rgb, bbox)
                if rec is not None:
                    return rec
            except Exception as e:
                print(f"[WARN] Ошибка 3D реконструкции {photo_path.name}: {e}")

        # Fallback на placeholder (только если 3DDFA сломалась)
        return build_placeholder_reconstruction(image_rgb.shape, bbox, 0, 0, 0)

    def _tddfa_reconstruct(self, image_rgb, bbox):
        """
        Адаптер под 3DDFA-V3.
        Нужно адаптировать под реальный API библиотеки.
        """
        x, y, w, h = bbox
        crop = image_rgb[y:y+h, x:x+w]

        # Предполагаемый вызов 3DDFA-V3:
        # param, roi_box = self.tddfa(crop, return_roi=True)
        # vertices = self.tddfa.recon_vers(param, roi_box)
        # landmarks = self.tddfa.recon_lmks(param, roi_box)

        # Заглушка с пояснением — заменить на реальный вызов:
        raise NotImplementedError(
            "Замените этот метод на реальный вызов 3DDFA-V3. "
            "Нужно получить: vertices (35709x3), landmarks_68, pose_params, alpha, exp"
        )

        # Пример структуры результата:
        # return {
        #     "vertices": vertices,               # (35709, 3) — мировые координаты
        #     "triangles": triangles,             # (70789, 3) — индексы вершин
        #     "landmarks_68": lmks68,             # (68, 2) — 2D на изображении
        #     "landmarks_106": lmks106,           # (106, 2)
        #     "texture_map": tex_map,             # (256, 256, 3) — UV текстура
        #     "pose": {
        #         "R": pose_R,                    # (3, 3) — rotation matrix
        #         "t": pose_t,                    # (3,) — translation
        #         "scale": pose_scale,
        #         "yaw": yaw,
        #         "pitch": pitch,
        #         "roll": roll,
        #     },
        #     "alpha": alpha,                     # (199,) — identity PCA
        #     "exp": exp,                         # (29,) — expression PCA
        #     "vertices_canon": vertices_canon,   # (35709, 3) — выровненный меш
        # }


# =============================================================================
# ПАТЧ 2: s2_metrics/modules/texture_extractor.py — убрать pose, добавить FFT/LBP
# =============================================================================

class TextureExtractorFixed:
    """Исправленный extractor без pose-зависимости."""

    def __init__(self):
        self.quality_sensitive_metrics = set()  # убрали старый список
        self._skin_scorer = None

    def extract(self, ctx, exclude_sensitive=False):
        """
        ctx должен содержать:
            - image_rgb: np.ndarray (H, W, 3)
            - face_mask: np.ndarray (H, W) bool — маска кожи
            - quality: QualityMetrics
        """
        rgb = ctx.image_rgb
        mask = ctx.face_mask if hasattr(ctx, 'face_mask') else np.ones(rgb.shape[:2], dtype=bool)
        quality = getattr(ctx, 'quality', None)

        # Извлекаем только кожные пиксели
        skin_pixels = rgb[mask]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray_skin = gray[mask]

        metrics = {}

        # --- Базовые метрики (инвариантные к pose) ---
        metrics["skin_mean_r"] = float(np.mean(skin_pixels[:, 0]))
        metrics["skin_mean_g"] = float(np.mean(skin_pixels[:, 1]))
        metrics["skin_mean_b"] = float(np.mean(skin_pixels[:, 2]))
        metrics["skin_std"] = float(np.std(gray_skin))

        # --- FFT-анализ (по TZ required) ---
        fft_metrics = self._fft_skin_analysis(gray, mask)
        metrics.update(fft_metrics)

        # --- LBP-сложность (по TZ required) ---
        lbp_metrics = self._lbp_complexity(gray, mask)
        metrics.update(lbp_metrics)

        # --- GLCM (существующий код, но на кожной маске) ---
        glcm_metrics = self._glcm_on_mask(gray, mask)
        metrics.update(glcm_metrics)

        # --- Albedo / Color viability (по TZ required) ---
        albedo_metrics = self._albedo_analysis(rgb, mask)
        metrics.update(albedo_metrics)

        # --- Quality-aware confidence ---
        if quality is not None:
            metrics["_quality_confidence"] = self._quality_confidence(quality)
            if quality.overall_quality < 0.25:
                metrics["_quality_warning"] = "low_quality_early_photo"

        return metrics

    def _fft_skin_analysis(self, gray, mask, patch_size=64):
        """FFT по патчам кожи. Силикон имеет регулярные высокие частоты."""
        coords = np.argwhere(mask)
        if len(coords) < patch_size * patch_size:
            return {"fft_highfreq_ratio": 0.0, "fft_peak_regularity": 0.0}

        y0, x0 = coords[:, 0].min(), coords[:, 1].min()
        y1, x1 = coords[:, 0].max() + 1, coords[:, 1].max() + 1
        crop = gray[y0:y1, x0:x1]
        mh, mw = crop.shape

        ratios = []
        regularities = []

        for py in range(0, mh - patch_size, patch_size // 2):
            for px in range(0, mw - patch_size, patch_size // 2):
                patch = crop[py:py+patch_size, px:px+patch_size].astype(np.float32)
                if patch.std() < 1.0:
                    continue

                patch = patch - patch.mean()
                spectrum = np.fft.fftshift(np.fft.fft2(patch))
                power = np.abs(spectrum) ** 2

                h, w = power.shape
                cy, cx = h // 2, w // 2
                yy, xx = np.ogrid[:h, :w]
                radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

                low = power[radius <= 4].sum()
                high = power[radius > 8].sum()
                total = power.sum() + 1e-6

                ratios.append(high / (low + 1e-6))

                # Регулярность: силикон даёт острые пики на фиксированных частотах
                peaks = (power > power.mean() * 3).sum()
                regularities.append(peaks / (patch_size * patch_size))

        if not ratios:
            return {"fft_highfreq_ratio": 0.0, "fft_peak_regularity": 0.0}

        return {
            "fft_highfreq_ratio_mean": float(np.mean(ratios)),
            "fft_highfreq_ratio_std": float(np.std(ratios)),
            "fft_peak_regularity": float(np.mean(regularities)),
        }

    def _lbp_complexity(self, gray, mask, P=8, R=1):
        """LBP-энтропия. Силикон = низкая энтропия."""
        from skimage.feature import local_binary_pattern

        coords = np.argwhere(mask)
        if len(coords) < 100:
            return {"lbp_entropy_r1": 0.0, "lbp_entropy_r2": 0.0}

        y0, x0 = coords[:, 0].min(), coords[:, 1].min()
        y1, x1 = coords[:, 0].max() + 1, coords[:, 1].max() + 1
        crop = gray[y0:y1, x0:x1]

        lbp = local_binary_pattern(crop, P=P, R=R, method="uniform")
        hist, _ = np.histogram(lbp, bins=P + 2, range=(0, P + 2), density=True)
        entropy_r1 = float(-np.sum(hist * np.log2(hist + 1e-10)))

        # Multi-scale
        lbp_r2 = local_binary_pattern(crop, P=16, R=2, method="uniform")
        hist_r2, _ = np.histogram(lbp_r2, bins=18, range=(0, 18), density=True)
        entropy_r2 = float(-np.sum(hist_r2 * np.log2(hist_r2 + 1e-10)))

        return {
            "lbp_entropy_r1": entropy_r1,
            "lbp_entropy_r2": entropy_r2,
            "lbp_complexity_ratio": entropy_r2 / (entropy_r1 + 1e-6),
        }

    def _albedo_analysis(self, rgb, mask):
        """Анализ цвета и бликов. Силикон = спекулярные блики, плоский albedo."""
        skin = rgb[mask]
        if len(skin) == 0:
            return {"albedo_a_std": 0.0, "specular_ratio": 0.0}

        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        a_channel = lab[:, :, 1][mask]

        # Спекулярность: яркие пиксели с низкой насыщенностью
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        v_channel = hsv[:, :, 2][mask]
        s_channel = hsv[:, :, 1][mask]
        specular = (v_channel > 200) & (s_channel < 30)
        specular_ratio = float(specular.sum() / max(len(skin), 1))

        return {
            "albedo_a_std": float(np.std(a_channel)),
            "albedo_a_mean": float(np.mean(a_channel)),
            "specular_ratio": specular_ratio,
            "skin_brightness_std": float(np.std(v_channel)),
        }

    def _glcm_on_mask(self, gray, mask):
        """Упрощённый GLCM на маске кожи."""
        from skimage.feature import graycomatrix, graycoprops

        coords = np.argwhere(mask)
        if len(coords) < 100:
            return {"glcm_contrast": 0.0, "glcm_homogeneity": 0.0}

        y0, x0 = coords[:, 0].min(), coords[:, 1].min()
        y1, x1 = coords[:, 0].max() + 1, coords[:, 1].max() + 1
        crop = gray[y0:y1, x0:x1]

        levels = 16
        quantized = np.floor(crop.astype(np.float32) / (256 / levels)).astype(np.uint8)
        glcm = graycomatrix(quantized, distances=[1], angles=[0], levels=levels, symmetric=True, normed=True)

        return {
            "glcm_contrast": float(graycoprops(glcm, "contrast")[0, 0]),
            "glcm_homogeneity": float(graycoprops(glcm, "homogeneity")[0, 0]),
            "glcm_energy": float(graycoprops(glcm, "energy")[0, 0]),
        }

    def _quality_confidence(self, quality):
        """Уверенность на основе качества фото."""
        q = quality.overall_quality
        if q > 0.7:
            return 0.95
        elif q > 0.4:
            return 0.8
        elif q > 0.25:
            return 0.6
        else:
            return 0.4


# =============================================================================
# ПАТЧ 3: s4_compare/engine.py — ICP выравнивание и anchor-сравнение
# =============================================================================

import numpy as np
from scipy.spatial.distance import cdist


def procrustes_align(v_source, v_target, bone_indices=None):
    """
    Выравнивает v_source к v_target по костным вершинам (или по всем).
    Returns: v_source_aligned, R, t, scale
    """
    if bone_indices is not None:
        src = v_source[bone_indices]
        tgt = v_target[bone_indices]
    else:
        src = v_source
        tgt = v_target

    # Центрирование
    src_mean = src.mean(axis=0)
    tgt_mean = tgt.mean(axis=0)
    src_c = src - src_mean
    tgt_c = tgt - tgt_mean

    # Масштабирование
    src_scale = np.linalg.norm(src_c)
    tgt_scale = np.linalg.norm(tgt_c)
    scale = tgt_scale / src_scale
    src_c = src_c * scale

    # Поворот (SVD)
    H = src_c.T @ tgt_c
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Применяем к полному мешу
    v_aligned = (v_source - src_mean) @ R * scale + tgt_mean
    return v_aligned, R, tgt_mean, scale


def compute_heatmap(v_a, v_b, triangles, bone_zones, soft_zones):
    """
    Возвращает per-vertex разницу с учётом зональных порогов.
    """
    diff = np.linalg.norm(v_a - v_b, axis=1)  # мм (предполагаем, что vertices в мм)

    heat = np.zeros_like(diff)

    # Костные зоны: порог 2 мм
    for zone_name, indices in bone_zones.items():
        zone_diff = diff[indices]
        heat[indices] = np.clip(zone_diff / 2.0, 0, 1)  # 0..1, где 1 = >2мм

    # Мягкие ткани: порог 5 мм
    for zone_name, indices in soft_zones.items():
        zone_diff = diff[indices]
        heat[indices] = np.clip(zone_diff / 5.0, 0, 1)

    return heat


def compare_with_anchor(photo, anchor, calibration_reference):
    """
    Сравнивает фото с anchor (калибровочным или ранним).
    Возвращает excess_distance после вычитания калибровочного шума.
    """
    # Загружаем реконструкции
    rec_photo = load_reconstruction(photo)
    rec_anchor = load_reconstruction(anchor)

    # Выравниваем
    v_photo_aligned, _, _, _ = procrustes_align(
        rec_photo["vertices_canon"],
        rec_anchor["vertices_canon"],
        bone_indices=BONE_VERTEX_INDICES
    )

    # Разница
    raw_distance = np.linalg.norm(v_photo_aligned - rec_anchor["vertices_canon"], axis=1).mean()

    # Вычитаем ожидаемый шум
    pose_gap = compute_pose_gap(rec_photo["pose"], rec_anchor["pose"])
    expected_noise = calibration_reference.get_expected_noise(pose_gap)

    excess = max(0.0, raw_distance - expected_noise)
    return {
        "raw_distance": float(raw_distance),
        "expected_noise": float(expected_noise),
        "excess_distance": float(excess),
        "heatmap": compute_heatmap(v_photo_aligned, rec_anchor["vertices_canon"], ...),
    }


# =============================================================================
# ПАТЧ 4: test/retrain_classifier.py — без pose, с quality-stratified CV
# =============================================================================

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np


def train_quality_aware_classifier(X, y, qualities, feature_names):
    """
    Обучает классификатор с учётом качества фото.
    Стратификация по качеству + метке.
    """
    # Кодируем качество в бины
    quality_bins = np.digitize(qualities, bins=[0.0, 0.25, 0.5, 0.75, 1.0])
    stratify_labels = [f"{q}_{label}" for q, label in zip(quality_bins, y)]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, test_idx in skf.split(X, stratify_labels):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
        )
        clf.fit(X_train_s, y_train)
        scores.append(clf.score(X_test_s, y_test))

    print(f"Stratified CV accuracy: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
    return clf, scaler


# =============================================================================
# ПАТЧ 5: s5_verdict/modules/chronology.py — биологические impossibility checks
# =============================================================================

from datetime import timedelta

BIOLOGICAL_LIMITS = {
    "rhinoplasty_healing": timedelta(days=180),
    "facelift_healing": timedelta(days=90),
    "implant_settling": timedelta(days=365),
    "max_bone_shift_mm": 2.0,  # за любой период у взрослого
}


def check_biological_impossibility(points, metric_series):
    """
    points: list[ChronologyPoint]
    metric_series: dict[str, list[float]] — метрики в хронологическом порядке
    """
    flags = []

    for metric_name, values in metric_series.items():
        if len(values) < 2:
            continue

        for i in range(1, len(values)):
            gap = points[i].date - points[i - 1].date
            if gap.days <= 0:
                continue

            delta = abs(values[i] - values[i - 1])

            # Костные метрики не должны меняться быстро
            if metric_name.startswith("bone_") and delta > BIOLOGICAL_LIMITS["max_bone_shift_mm"]:
                if gap.days < 365:
                    flags.append({
                        "type": "IMPOSSIBLE_BONE_CHANGE",
                        "metric": metric_name,
                        "delta_mm": float(delta),
                        "gap_days": gap.days,
                        "photo_a": points[i - 1].photo_id,
                        "photo_b": points[i].photo_id,
                    })

            # Инверсия асимметрии
            if metric_name == "bone_asymmetry_x":
                if values[i - 1] > 2.0 and values[i] < -2.0:
                    flags.append({
                        "type": "IMPOSSIBLE_ASYMMETRY_INVERSION",
                        "photo_a": points[i - 1].photo_id,
                        "photo_b": points[i].photo_id,
                    })

    return flags


# =============================================================================
# ПАТЧ 6: shared/utils.py — канонические зоны лица (vertex indices)
# =============================================================================

# Это примерные индексы для Basel Face Model (35709 вершин).
# Нужно уточнить под реальную топологию 3DDFA-V3.

BONE_VERTEX_INDICES = {
    "nasion": list(range(8000, 8200)),          # переносица
    "orbit_L": list(range(12000, 12200)),       # левая глазница
    "orbit_R": list(range(14000, 14200)),       # правая глазница
    "zygomatic_L": list(range(16000, 16200)),   # левая скуловая
    "zygomatic_R": list(range(18000, 18200)),   # правая скуловая
    "gonion_L": list(range(20000, 20200)),      # левый угол челюсти
    "gonion_R": list(range(22000, 22200)),      # правый угол челюсти
    "pogonion": list(range(24000, 24200)),      # подбородок
    "ramus_L": list(range(26000, 26200)),       # левая ветвь челюсти
    "ramus_R": list(range(28000, 28200)),       # правая ветвь челюсти
}

SOFT_VERTEX_INDICES = {
    "cheek_L": list(range(30000, 30500)),
    "cheek_R": list(range(31000, 31500)),
    "nasolabial_L": list(range(32000, 32200)),
    "nasolabial_R": list(range(33000, 33200)),
    "forehead": list(range(34000, 34500)),
    "lip_upper": list(range(35000, 35200)),
    "lip_lower": list(range(35300, 35500)),
}

ALL_BONE_INDICES = []
for indices in BONE_VERTEX_INDICES.values():
    ALL_BONE_INDICES.extend(indices)
ALL_BONE_INDICES = np.array(list(set(ALL_BONE_INDICES)))


def get_zone_vertices(zone_name: str) -> np.ndarray:
    """Возвращает индексы вершин для зоны."""
    if zone_name in BONE_VERTEX_INDICES:
        return np.array(BONE_VERTEX_INDICES[zone_name])
    if zone_name in SOFT_VERTEX_INDICES:
        return np.array(SOFT_VERTEX_INDICES[zone_name])
    raise ValueError(f"Unknown zone: {zone_name}")
