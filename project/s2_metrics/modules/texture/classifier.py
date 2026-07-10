from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

MODEL_PATH = Path(__file__).parent / "skin_classifier_model.pkl"

# Core texture features (NO pose features!)
# These are the features that actually measure skin properties, not image quality/pose
TEXTURE_FEATURES = [
    # GLCM (multi-distance, multi-angle)
    "glcm_dissimilarity_d5_a0", "glcm_homogeneity_d5_a0", "glcm_dissimilarity_d3_a0",
    "glcm_dissimilarity_d5_a45", "glcm_homogeneity_d5_a45", "glcm_dissimilarity_d5_a135",
    "glcm_homogeneity_d5_a135", "glcm_dissimilarity_d3_a135", "glcm_homogeneity_d3_a0",
    "glcm_dissimilarity_d2_a0", "glcm_dissimilarity_d2_range", "glcm_dissimilarity_d5_avg",
    "glcm_dissimilarity_d3_avg",
    
    # Local variance (homo)
    "homo_local_var_w15_cv", "homo_local_var_w31_cv",
    
    # LBP (multi-scale)
    "texture_lbp_uniformity", "lbp_uniform_r5_std", "lbp_complexity_ratio",
    
    # Morphology (pores)
    "morph_tophat_r4_std", "morph_tophat_r8_std",
    
    # Gradient/Sobel
    "grad_sobel_mag_skewness",
    
    # Residual (biological texture)
    "residual_bio_iqr",
    
    # FFT (high-frequency structure)
    "texture_fft_highfreq_ratio", "texture_fft_highfreq_std", "texture_fft_peak_ratio", "texture_fft_peak_std",
    
    # Color/Albedo
    "color_b_mean", "albedo_a_std", "albedo_a_mean", "albedo_viability_index",
    
    # Specular
    "specular_ratio", "skin_brightness_std",
    
    # Edge density
    "texture_edge_density",
    
    # Saturation
    "texture_saturation", "texture_color_std",
    
    # Entropy
    "texture_entropy",
]

# Quality features for gating (not for classification directly)
QUALITY_FEATURES = [
    "overall_quality", "sharpness_score", "noise_level", "jpeg_blockiness",
]

# Features that indicate the model should NOT classify (quality too low)
QUALITY_GATE_THRESHOLDS = {
    "overall_quality_min": 0.35,      # ниже этого — unknown
    "sharpness_score_min": 100.0,     # variance of laplacian
    "noise_level_max": 30.0,          # MAD
    "jpeg_blockiness_max": 1.8,       # blockiness index
}


class TextureSkinClassifier:
    """Quality-aware texture classifier (real vs silicone)."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        if model_path and str(model_path).endswith(".csv"):
            model_path = None
        path = Path(model_path) if model_path else MODEL_PATH
        self._pipeline = None
        self._feature_names: list[str] = []
        if path.exists():
            with open(path, "rb") as f:
                data = pickle.load(f)
            self._pipeline = data["pipeline"]
            self._feature_names = data["feature_names"]

    def classify(self, metrics: dict[str, float], quality: dict[str, float] | Any = None, reference: dict | None = None, pose: dict | None = None) -> dict[str, Any]:
        # Quality gate: if image quality too low, return unknown
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

        # Use only the features the model was trained on
        vector = []
        used = []
        for name in self._feature_names:
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
        if confidence < 0.55:
            hint = "unknown"

        return {
            "texture_skin_hint": hint,
            "texture_skin_confidence": confidence,
            "posterior": {"real": prob_real, "silicone": prob_silicone},
            "used_metrics": used,
            "model_loaded": True,
            "quality_gated": False,
        }