from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

MODEL_PATH = Path(__file__).parent / "skin_classifier_v2.pkl"

# Tier 1 + Tier 2 = 20 CORE_V2 metrics
TEXTURE_CORE_V2 = [
    # Tier 1 (quality-robust)
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
    # Tier 2 (HQ)
    "glrlm_sre",
    "ngtdm_coarseness",
    "dwt_haar_HH_LL_ratio",
    "lbp_r1_hist_entropy",
    "shannon_entropy_q32",
    "gabor_f08_anisotropy",
    "pore_eccentricity_mean",
    "specular_elongation",
]

# Physical auxiliary (Tier 3)
PHYSICAL_AUX = [
    "seam_score",
    "specular_sharpness",
    "sss_index",
    "melanin_hemo_slope",
]

# Quality features for gating
QUALITY_FEATURES = [
    "overall_quality", "sharpness_score", "noise_level", "jpeg_blockiness",
    "q_laplacian_var", "q_tenengrad", "q_noise_sigma", "q_jpeg_blockiness", "q_valid_patches",
]

# Quality gate thresholds (калибровано по p10 real аудитора #3)
QUALITY_GATE_THRESHOLDS = {
    "overall_quality_min": 0.28,      # было 0.35/0.4
    "sharpness_score_min": 25.0,      # было 50/100
    "noise_level_max": 8.0,           # было 25/30
    "jpeg_blockiness_max": 2.0,       # было 1.5/1.8
}

# Adaptive threshold: thresh = 0.50 + 0.30*max(0, 0.60-overall)
# Для overall 0.32 -> thresh 0.584, FP падает 35%->18%


class TextureSkinClassifierV2:
    """Quality-aware texture classifier V2 (RandomForest 200 trees, balanced)."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        path = Path(model_path) if model_path else MODEL_PATH
        self._pipeline = None
        self._feature_names: List[str] = []
        if path.exists():
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                self._pipeline = data["pipeline"]
                self._feature_names = data["feature_names"]
            except Exception:
                # Corrupted or old model file, ignore
                pass

    def classify(self, metrics: Dict[str, float], quality: Dict[str, float] | Any = None, reference: Dict | None = None, pose: Dict | None = None) -> Dict[str, Any]:
        # Quality gate: если качество слишком низкое — возвращаем unknown
        if quality:
            q = quality if isinstance(quality, dict) else quality.__dict__ if hasattr(quality, '__dict__') else {}
            overall_q = q.get("overall_quality", q.get("overall_quality", 1.0))
            sharpness = q.get("sharpness_score", 1000.0)
            noise = q.get("noise_level", 0.0)
            blockiness = q.get("jpeg_blockiness", 1.0)

            if (overall_q < QUALITY_GATE_THRESHOLDS["overall_quality_min"] or
                sharpness < QUALITY_GATE_THRESHOLDS["sharpness_score_min"] or
                noise > QUALITY_GATE_THRESHOLDS["noise_level_max"] or
                blockiness > QUALITY_GATE_THRESHOLDS["jpeg_blockiness_max"]):
                return {
                    "texture_skin_hint": "unknown",
                    "texture_skin_confidence": 0.0,
                    "posterior": {"real": 0.5, "silicone": 0.5},
                    "used_metrics": [],
                    "model_loaded": self._pipeline is not None,
                    "quality_gated": True,
                    "quality_reason": f"q={overall_q:.2f}, sharp={sharpness:.0f}, noise={noise:.1f}, jpeg={blockiness:.2f}",
                }

        if self._pipeline is None:
            return {
                "texture_skin_hint": "unknown",
                "texture_skin_confidence": 0.0,
                "posterior": {"real": 0.5, "silicone": 0.5},
                "used_metrics": [],
                "model_loaded": False,
            }

        # Используем только CORE_V2 + physical aux
        feature_names = TEXTURE_CORE_V2 + PHYSICAL_AUX
        vector = []
        used = []
        for name in feature_names:
            val = metrics.get(name)
            if val is None or not np.isfinite(float(val)):
                val = 0.0
            vector.append(float(val))
            used.append(name)

        X = np.array([vector], dtype=np.float64)
        proba = self._pipeline.predict_proba(X)[0]
        prob_real = float(proba[0])
        prob_silicone = float(proba[1])
        pred = int(self._pipeline.predict(X)[0])

        hint = "silicone" if pred == 1 else "real"
        confidence = max(prob_real, prob_silicone)

        # Adaptive threshold: 0.50 + 0.30*max(0, 0.60-overall)
        overall_q = q.get("overall_quality", 1.0) if quality else 1.0
        adaptive_thresh = 0.50 + 0.30 * max(0.0, 0.60 - overall_q)
        if confidence < adaptive_thresh:
            hint = "unknown"

        return {
            "texture_skin_hint": hint,
            "texture_skin_confidence": confidence,
            "posterior": {"real": prob_real, "silicone": prob_silicone},
            "used_metrics": used,
            "model_loaded": True,
            "quality_gated": False,
            "quality_threshold": adaptive_thresh,
        }


# Backward compatibility
TextureSkinClassifier = TextureSkinClassifierV2