from __future__ import annotations
from datetime import date
from pathlib import Path

# --- Quality Thresholds ---
BLUR_THRESHOLD_DEFAULT = 65.0
NOISE_THRESHOLD_DEFAULT = 2.5

# --- Visibility & Z-Buffer ---
VISIBILITY_ANGLE_DEG = 82.0
Z_TOLERANCE_RATIO = 0.005 # 0.5% of Z-span
MIN_ZONE_VERTICES = 80

# --- Geometry & Scoring ---
ALIGNMENT_MIN_RANK = 4
TRIMMED_KEEP_RATIO = 0.90
MIN_KEEP_N = 50
FACE_SCALE_Y_FACTOR = 0.7

# --- Chronology & Eras ---
REFERENCE_PERIOD_END = "2002-12-31"

ERAS = {
    "ERA_1_BASELINE": {"start": "1999-01-01", "end": "2011-12-31", "label": "Original Baseline"},
    "ERA_2_EARLY": {"start": "2012-01-01", "end": "2014-12-31", "label": "Early Anomaly / First Udmurt"},
    "ERA_3_UDMURT": {"start": "2015-01-01", "end": "2021-09-08", "label": "Main Udmurt Period"},
    "ERA_4_TRANSITION": {"start": "2021-09-09", "end": "2023-09-30", "label": "Transition Zone"},
    "ERA_5_VASILICH": {"start": "2023-10-01", "end": "2099-12-31", "label": "Main Vasilich Period"},
}

RTR_RATIO = 0.75
RTR_MIN_ABS_DELTA = 0.15
IMPOSSIBLE_AGE_REVERSAL_DAYS = 180

# Дата рождения субъекта по умолчанию (переопределяется через SUBJECT_BIRTH_YEAR).
SUBJECT_BIRTH_DATE = date(1952, 10, 7)


def subject_birth_year() -> int:
    """Год рождения субъекта: env SUBJECT_BIRTH_YEAR или SUBJECT_BIRTH_DATE.year."""
    import os

    for key in (
        "SUBJECT_BIRTH_YEAR",
        "DUTIN_CASE_SUBJECT_BIRTH_YEAR",
        "DUTIN_SUBJECT_BIRTH_YEAR",
    ):
        raw = os.environ.get(key)
        if raw:
            return int(raw)
    return SUBJECT_BIRTH_DATE.year


def exif_spoof_age_threshold() -> float:
    import os

    return float(os.environ.get("DUTIN_EXIF_SPOOF_AGE_THRESHOLD", "12.0"))

# Chronology Flag Types as Constants
CHRONO_FLAG_IMPOSSIBLE = "impossible_short"
CHRONO_FLAG_RETURN = "return"
CHRONO_FLAG_TRANSITION = "transition"

# --- Texture & Silicone ---
# SILICONE_RAW_BIAS: порог сырого mask-score перед sigmoid (калибровка по main dataset).
# p50(raw)≈0.13 → p50(prob)≈8%; типичная «восковая» кожа ≈0.35 raw → ~35% prob.
SILICONE_RAW_BIAS = 0.43
SILICONE_SIGMOID_SCALE = 8.0
# Legacy alias (config hash / runtime metadata)
SILICONE_SIGMOID_BIAS = SILICONE_RAW_BIAS
RELIABILITY_MIN = 0.1
RELIABILITY_MAX = 1.0

# --- Bayesian & Calibration ---
SNR_UNCERTAIN_THRESHOLD = 1.0
SNR_SIGNAL_THRESHOLD = 2.0
MIN_SUCCESSFUL_PAIRS_FOR_CALIBRATION = 30
MIN_PAIRS_PER_BUCKET_FOR_CALIBRATION = 5
# Legacy pipeline/verdict.py only; w2 inference uses fuzzy_bayes/config.py priors.
PRIOR_SAME_PERSON = 0.65
PRIOR_IDENTITY_SWAP = 0.02

# 3DDFA 106-landmark indices for inter-pupillary distance scaling
LANDMARK_IPD_LEFT = 74
LANDMARK_IPD_RIGHT = 83

# --- Compare band scoring (iteration 4) ---
MESH_PROFILE_REF = 4.5
BONE_ERR_INTERNAL_REF = 0.10
BAND_WEIGHT_PROFILE = 0.45
BAND_WEIGHT_ID = 0.35
BAND_WEIGHT_BONE = 0.20

# --- Artifact Versioning ---
ARTIFACT_VERSION = "2.1.0"
RUNTIME_CONFIG_HASH_VERSION = "v2"

# --- Exclusion Lists ---
# Photo IDs excluded from automated forensic analysis (not zone names).
EXCLUDED_PHOTO_IDS = [
    "main-2012_05_07-a1b2c3d4",  # Example: problematic lighting
]
# Backward-compatible alias used by older modules.
EXCLUDED_FROM_ANALYSIS = EXCLUDED_PHOTO_IDS

# Zone names excluded from geometry scoring (soft tissue / unstable regions).
GEOMETRY_EXCLUDED_ZONES = frozenset(
    {
        "upper_lip",
        "lower_lip",
        "mouth",
        "mouth_corner_L",
        "mouth_corner_R",
        "cheek_lower_L",
        "cheek_lower_R",
        "right_eye",
        "left_eye",
        "skin",
    }
)


# --- Zone Weights ---
# Веса зон по реальным ключам из BUCKET_METRIC_KEYS.
# Приоритет на неизменные костные структуры согласно ТЗ.
# [FIX-1] Расширено до полного набора 21 зоны с анатомически обоснованными весами.
ZONE_WEIGHTS = {
    # === Костные структуры (максимальный приоритет, вес 1.0) ===
    # Эти зоны формируются в раннем возрасте и не меняются на протяжении жизни
    "nose_projection_ratio": 1.0,   # Проекция носа (переносица) — костная основа
    "orbit_depth_L_ratio": 1.0,     # Глубина левой глазницы — костная структура
    "orbit_depth_R_ratio": 1.0,     # Глубина правой глазницы — костная структура
    "jaw_width_ratio": 0.95,        # Ширина челюсти — костная структура
    "cranial_face_index": 0.95,     # Краниальный индекс — соотношение черепа и лица

    # === Костно-связочные зоны (высокий приоритет, вес 0.8-0.9) ===
    "gonial_angle_L": 0.85,         # Угол нижней челюсти L — гониальный угол
    "gonial_angle_R": 0.85,         # Угол нижней челюсти R
    

    # === Зоны симметрии и асимметрии (средний приоритет, вес 0.7) ===
    # Используются для выявления структурных несоответствий
    "chin_offset_asymmetry": 0.7,   # Асимметрия подбородка
    "nasal_frontal_index": 0.7,     # Индекс переносицы
    
    "palpebral_fissure_asymmetry_ratio": 0.72,
    "orbit_depth_asymmetry_ratio": 0.72,
    "orbit_vertical_asymmetry_ratio": 0.7,

    # === Мягкие ткани и текстура (низкий приоритет, вес 0.2-0.4) ===
    # Подвержены временным изменениям, но полезны для детекции синтетики
    "texture_silicone_prob": 0.3,   # Вероятность силикона (мягкие ткани)
    "nose_width_ratio": 0.25,       # Ширина носа (включает мягкие ткани крыльев)
    "texture_specular_gloss": 0.22,
    "interorbital_ratio": 0.85,     # Межглазничное расстояние
    "orbital_asymmetry_index": 0.75,

    # === Extended 3D zone metrics (2026-05) ===
    "nasal_length_ratio": 0.88,
    "nose_bridge_length_ratio": 0.9,
    "subnasale_projection_ratio": 0.85,
    "nasion_zone_depth_ratio": 0.92,
    "midface_depth_index": 0.88,
    
    "bigonial_width_ratio": 0.9,
    "intercanthal_width_ratio": 0.15,
    
    "gonial_width_asymmetry": 0.75,
    "orbit_width_L_ratio": 0.85,
    "orbit_width_R_ratio": 0.85,
    "brow_ridge_projection_L_ratio": 0.88,
    "brow_ridge_projection_R_ratio": 0.88,
    "temporal_depth_L_ratio": 0.82,
    "temporal_depth_R_ratio": 0.82,
    
    "palpebral_fissure_length_L_ratio": 0.78,
    "palpebral_fissure_length_R_ratio": 0.78,
    "mandibular_body_length_L_ratio": 0.88,
    "mandibular_body_length_R_ratio": 0.88,
    "mandibular_ramus_length": 0.84,
    "orbit_fossa_spread_L": 0.82,
    "orbit_fossa_spread_R": 0.82,
}

# --- Collinearity Adjustments (Iteration 5) ---
# Downweight highly correlated factors (|rho| > 0.85) to prevent double-counting likelihoods.
# Final effective weights for keys below are the values in this block (not the entries above).
ZONE_WEIGHTS.update({
    "gnathion_midline_deviation_ratio": 0.3,
    "nasion_zone_depth_ratio": 0.3,
    "orbital_asymmetry_index": 0.3,
    "temporal_depth_L_ratio": 0.3,
    "temporal_depth_R_ratio": 0.3,
    "orbital_perimeter_symmetry": 0.3,
    "orbit_vertical_asymmetry_ratio": 0.3,
})

# Веса для stability UI слоя extract_saved (без texture/diagnostic).
METRIC_EXTRACT_WEIGHTS: dict[str, float] = {
    k: float(v)
    for k, v in ZONE_WEIGHTS.items()
    if not str(k).startswith("texture_") and not str(k).startswith("glcm_")
}

# Привести ZONE_WEIGHTS в соответствие с текущим реестром метрик:
# удалить ключи, которых нет в metrics.registry (чтобы избежать рассинхронизации).
try:
    from metrics.registry import all_specs
    _registry_names = {s.name for s in all_specs() if s.scope == 'single'}
    for _k in list(ZONE_WEIGHTS.keys()):
        if _k not in _registry_names:
            ZONE_WEIGHTS.pop(_k, None)
except Exception:
    # В тестовой/рантайм среде импорт может быть недоступен — оставить оригинальные веса.
    pass

# Обновить METRIC_EXTRACT_WEIGHTS после приведения ZONE_WEIGHTS
METRIC_EXTRACT_WEIGHTS = {
    k: float(v)
    for k, v in ZONE_WEIGHTS.items()
    if not str(k).startswith("texture_") and not str(k).startswith("glcm_")
}
