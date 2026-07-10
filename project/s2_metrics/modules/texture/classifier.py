from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

MODEL_PATH = Path(__file__).parent / "skin_classifier_model.pkl"

TOP20_FEATURES = [
    "glcm_dissimilarity_d5_a0", "glcm_homogeneity_d5_a0", "glcm_dissimilarity_d3_a0",
    "homo_local_var_w15_cv", "contrast_weber_mean", "homo_local_var_w31_cv",
    "color_b_mean", "glcm_homogeneity_d3_a0", "glcm_dissimilarity_d3_a135",
    "glcm_dissimilarity_d2_a0", "lbp_uniform_r5_std", "glcm_dissimilarity_d5_avg",
    "glcm_dissimilarity_d3_avg", "morph_tophat_r4_std", "glcm_dissimilarity_d5_a135",
    "glcm_dissimilarity_d2_range", "grad_sobel_mag_skewness", "residual_bio_iqr",
    "morph_tophat_r8_std", "glcm_dissimilarity_d5_a45",
]

POSE_FEATURES = ["yaw", "pitch", "roll"]

TOP23_FEATURES = TOP20_FEATURES + POSE_FEATURES


class TextureSkinClassifier:
    """sklearn LogisticRegression real vs silicone classifier (LOO-CV 96.70%)."""

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
        if self._pipeline is None:
            return {
                "texture_skin_hint": "unknown",
                "texture_skin_confidence": 0.0,
                "posterior": {"real": 0.5, "silicone": 0.5},
                "used_metrics": [],
                "model_loaded": False,
            }

        vector = []
        used = []
        for name in self._feature_names:
            if name in POSE_FEATURES and pose is not None:
                val = pose.get(name)
            else:
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
        }
