from __future__ import annotations

import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class CohortBaseline:
    """Baseline статистика когорты для anomaly detection."""
    mean: np.ndarray
    std: np.ndarray
    median: np.ndarray
    mad: np.ndarray
    n_samples: int


@dataclass
class TextureAnomalyResult:
    """Результат детекции аномалии текстуры."""
    anomaly_score: float  # 0..1
    max_z: float
    mean_z: float
    threshold: float
    quality_adjusted: bool
    feature_flags: Dict[str, float]
    interpretation: str


class CohortTextureAnomalyDetector:
    """
    Вместо "real vs silicone" используем:
    - Каждая когорта (эпоха) имеет свой texture baseline.
    - Фото, выбивающееся из baseline своей когорты — anomaly.
    - Anomaly + high quality = подозрение на силикон.
    """
    
    COHORT_ERAS = [
        (1999, 2005, "early_scan"),      # сканы 1999-2005
        (2005, 2012, "early_digital"),   # ранние цифровые 2005-2012
        (2012, 2021, "udmurt_era"),      # эпоха UDMURT 2012-2021
        (2021, 2030, "vas_era"),         # эпоха VAS 2021+
    ]
    
    # Маппинг: internal_name -> texture_extractor_name
    # Если extractor_name нет в metrics — используется fallback_value (0.0)
    FEATURE_MAP = {
        "fft_highfreq_ratio_mean": ("texture_fft_highfreq_ratio", 0.0),
        "fft_peak_regularity":     ("texture_fft_peak_ratio", 0.0),
        "lbp_entropy_r1":          ("texture_lbp_uniformity", 0.0),
        "lbp_entropy_r2":          ("lbp_uniform_r5_std", 0.0),
        "lbp_complexity_ratio":    (None, 0.0),  # отсутствует в extractor
        "albedo_a_std":            ("albedo_a_std", 0.0),
        "specular_ratio":          ("specular_ratio", 0.0),
        "skin_brightness_std":     (None, 0.0),  # отсутствует в extractor
        "glcm_contrast":           ("texture_glcm_contrast", 0.0),
        "glcm_homogeneity":        ("texture_glcm_homogeneity", 0.0),
    }
    
    def __init__(self, contamination: float = 0.1):
        self.scaler = StandardScaler()
        self.lof = LocalOutlierFactor(
            n_neighbors=20,
            contamination=contamination,
            novelty=True,
            metric="mahalanobis",
        )
        # Internal feature names (used for scoring)
        self.texture_features = list(self.FEATURE_MAP.keys())
        self.cohort_baselines: Dict[str, CohortBaseline] = {}
    
    def get_cohort_key(self, year: int) -> str:
        for start, end, key in self.COHORT_ERAS:
            if start <= year < end:
                return key
        return "vas_era" if year >= 2021 else "early_scan"
    
    def fit_cohort(self, cohort_records: List[Dict], cohort_key: str) -> CohortBaseline:
        """
        Строит baseline для когорты.
        cohort_records: список записей с texture metrics.
        """
        X = self._extract_features(cohort_records)
        X_scaled = self.scaler.fit_transform(X)
        self.lof.fit(X_scaled)
        
        # Статистики когорты
        baseline = CohortBaseline(
            mean=X.mean(axis=0),
            std=X.std(axis=0) + 1e-6,
            median=np.median(X, axis=0),
            mad=np.median(np.abs(X - np.median(X, axis=0)), axis=0) + 1e-6,
            n_samples=len(X),
        )
        self.cohort_baselines[cohort_key] = baseline
        return baseline
    
    def score(self, texture_metrics: Dict, cohort_key: str, quality: float) -> TextureAnomalyResult:
        """
        Returns anomaly score относительно baseline когорты.
        Учитывает quality: низкое качество = широкий допуск.
        """
        x = self._extract_single(texture_metrics)
        
        baseline = self.cohort_baselines.get(cohort_key)
        if baseline is None:
            return TextureAnomalyResult(
                anomaly_score=0.0,
                max_z=0.0,
                mean_z=0.0,
                threshold=2.0,
                quality_adjusted=True,
                feature_flags={},
                interpretation="no_cohort_baseline",
            )
        
        # Robust z-score от median
        robust_z = np.abs(x - baseline.median) / baseline.mad
        
        # Quality-adjusted threshold
        # High quality (0.9): threshold = 2.0
        # Low quality (0.2): threshold = 5.0
        threshold = 2.0 + (1.0 - quality) * 3.0
        
        max_z = float(robust_z.max())
        mean_z = float(robust_z.mean())
        
        # Anomaly score: 0..1
        anomaly_score = float(np.clip(mean_z / threshold, 0, 1))
        
        # Feature-level breakdown
        feature_flags = {}
        for i, feat_name in enumerate(self.texture_features):
            if robust_z[i] > threshold:
                feature_flags[feat_name] = float(robust_z[i])
        
        interpretation = self._interpret(anomaly_score, quality, feature_flags)
        
        return TextureAnomalyResult(
            anomaly_score=anomaly_score,
            max_z=max_z,
            mean_z=mean_z,
            threshold=threshold,
            quality_adjusted=True,
            feature_flags=feature_flags,
            interpretation=interpretation,
        )
    
    def _extract_features(self, records: List[Dict]) -> np.ndarray:
        X = []
        for record in records:
            x = self._extract_single(record)
            X.append(x)
        return np.array(X)
    
    def _extract_single(self, texture_metrics: Dict) -> np.ndarray:
        x = []
        for feat in self.texture_features:
            extractor_name, fallback = self.FEATURE_MAP[feat]
            if extractor_name is None:
                x.append(fallback)
            else:
                x.append(texture_metrics.get(extractor_name, fallback))
        return np.array(x, dtype=float)
    
    def _interpret(self, score: float, quality: float, flags: Dict) -> str:
        if quality < 0.3:
            return "low_quality_cannot_assess"
        if score < 0.3:
            return "texture_consistent_with_cohort"
        if score < 0.7:
            return "texture_slightly_atypical"
        # flags keys are internal names (specular_ratio, lbp_entropy_r1, etc.)
        if "specular_ratio" in flags and "lbp_entropy_r1" in flags:
            return "synthetic_material_suspected: specular + low_entropy"
        if "fft_peak_regularity" in flags:
            return "synthetic_material_suspected: regular_microrelief"
        return "texture_anomaly: deviates from cohort baseline"