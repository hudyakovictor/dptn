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


class CohortTextureAnomalyDetectorV2:
    """
    Texture V2 anomaly detector:
    - Каждая когорта (эпоха) имеет свой texture baseline
    - Фото, выбивающееся из baseline своей когорты — anomaly
    - Anomaly + high quality = подозрение на силикон
    - Quality-adjusted thresholds
    - 12 baseline'ов: era × quality_class
    """

    COHORT_ERAS = [
        (1999, 2005, "early_scan"),      # сканы 1999-2005
        (2005, 2012, "early_digital"),   # ранние цифровые 2005-2012
        (2012, 2021, "udmurt_era"),      # эпоха UDMURT 2012-2021
        (2021, 2030, "vas_era"),         # эпоха VAS 2021+
    ]

    Q_CLASSES = ["low", "mid", "high"]

    # FEATURE_MAP_V2: internal_name -> (extractor_name, fallback)
    # fallback=None означает not_available если метрика отсутствует, не 0.0!
    FEATURE_MAP_V2 = {
        "fft_high_low_ratio": ("fft_high_low_ratio", 0.09),
        "spectral_beta": ("spectral_slope_beta", 2.5),
        "glcm_aniso": ("glcm_diss_d3_aniso", 0.04),
        "pore_density": ("pore_density_r2_mpx", 0.12),
        "tv_sparsity": ("tv_residual_sparsity", 0.6),
        "lacunarity": ("lacunarity", 2.2),
        "autocorr_decay": ("autocorr_decay_len", 70),
        "wld_entropy": ("wld_joint_entropy", 5.1),
        "hemoglobin_std": ("hemoglobin_od_std", 0.08),
        "bimodality": ("bimodality_ashman_D", 2.8),
        "glszm_sae": ("glszm_small_area_emphasis", 0.5),
        "edge_tortuosity": ("edge_tortuosity_mean", 1.12),
        # Tier2 (weight 0.2 if low quality)
        "glrlm_sre": ("glrlm_sre", 0.7),
        "ngtdm_coarseness": ("ngtdm_coarseness", 0.0004),
        "dwt_ratio": ("dwt_haar_HH_LL_ratio", 0.0003),
        "lbp_hist_entropy": ("lbp_r1_hist_entropy", 3.2),
        "shannon_q32": ("shannon_entropy_q32", 3.9),
        "gabor_aniso": ("gabor_f08_anisotropy", 0.65),
        "pore_eccentricity": ("pore_eccentricity_mean", 0.54),
        "specular_elongation": ("specular_elongation", 0.55),
        # Tier3 physical aux
        "seam_score": ("seam_score", 0.1),
        "specular_sharpness": ("specular_sharpness", 0.1),
        "sss_index": ("sss_index", 0.1),
    }

    def __init__(self, contamination: float = 0.1):
        self.scaler = StandardScaler()
        self.lof = LocalOutlierFactor(
            n_neighbors=20,
            contamination=contamination,
            novelty=True,
            metric="mahalanobis",
        )
        self.texture_features = list(self.FEATURE_MAP_V2.keys())
        self.cohort_baselines: Dict[str, CohortBaseline] = {}

    def get_cohort_key(self, year: int, overall_quality: float) -> str:
        """Возвращает ключ когорты: era (e.g., early_scan). 
        Используем только эпоху для получения достаточного n_samples в когорте."""
        for start, end, key in self.COHORT_ERAS:
            if start <= year < end:
                return key
        return "vas_era"

    def fit_cohort(self, cohort_records: List[Dict], cohort_key: str) -> CohortBaseline:
        """Строит baseline для когорты."""
        X = self._extract_features(cohort_records)
        X_scaled = self.scaler.fit_transform(X)

        n_samples, n_features = X.shape
        
        # Dynamic n_neighbors: min(2, n_samples-1) for small cohorts, 20 for larger
        n_neighbors = min(20, max(2, n_samples - 1))
        
        # Use euclidean metric for small cohorts (Mahalanobis needs n_samples > n_features)
        metric = "mahalanobis" if n_samples > n_features else "euclidean"
        
        self.lof.set_params(n_neighbors=n_neighbors, metric=metric)
        self.lof.fit(X_scaled)

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
                threshold=2.5,
                quality_adjusted=True,
                feature_flags={},
                interpretation="no_cohort_baseline",
            )

        # Robust z-score от median, 1.4826*MAD
        robust_z = np.abs(x - baseline.median) / (baseline.mad * 1.4826)

        # Quality-adjusted threshold: 2.5 + (1-quality)*1.0
        # High quality (0.9): threshold = 2.5
        # Low quality (0.2): threshold = 3.3
        threshold = 2.5 + (1.0 - quality) * 1.0

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
            extractor_name, fallback = self.FEATURE_MAP_V2[feat]
            if extractor_name is None:
                x.append(fallback)
            else:
                x.append(texture_metrics.get(extractor_name, fallback))
        return np.array(x, dtype=float)

    def _interpret(self, score: float, quality: float, flags: Dict) -> str:
        if quality < 0.35:
            return "low_quality_cannot_assess"
        if score < 0.3:
            return "texture_consistent_with_cohort"
        if score < 0.7:
            return "texture_slightly_atypical"
        # Tier3 physical flags
        if "seam_score" in flags and "specular_sharpness" in flags:
            return "synthetic_material_suspected: seam + sharp_specular"
        if "sss_index" in flags:
            return "synthetic_material_suspected: low_sss"
        return "texture_anomaly: deviates from cohort baseline"


# Backward compatibility
CohortTextureAnomalyDetector = CohortTextureAnomalyDetectorV2
TextureAnomalyResult = TextureAnomalyResult