from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np

from .models import NoiseDiscountResult


class CalibratedNoiseDiscount:
    """Вычитает ожидаемый шум из raw distance."""

    def __init__(self, calibration_models: Dict[str, Any]):
        self.models = calibration_models

    def discount(self, bucket: str, raw_distance: float, 
                 pose_gap_deg: float, quality: float) -> NoiseDiscountResult:
        """
        Returns NoiseDiscountResult with:
            - excess_distance: raw - expected (главная метрика)
            - expected_noise: что вычли
            - confidence: насколько надёжна калибровка
            - is_significant: превышает ли excess CI
        """
        model = self.models.get(bucket)
        if model is None:
            return NoiseDiscountResult(
                excess_distance=raw_distance,
                expected_noise=0.0,
                confidence=0.3,
                is_significant=raw_distance > 2.0,
                model_bucket=bucket,
                model_samples=0,
            )
        
        expected, ci = model.predict(pose_gap_deg, quality)
        excess = max(0.0, raw_distance - expected)
        
        confidence = min(0.95, 0.4 + 0.5 * (1 - np.exp(-model.sample_count / 50)))
        if model.mad > 1.5:
            confidence *= 0.7
        
        is_significant = excess > ci
        
        return NoiseDiscountResult(
            excess_distance=float(excess),
            expected_noise=float(expected),
            confidence=float(confidence),
            is_significant=bool(is_significant),
            model_bucket=bucket,
            model_samples=model.sample_count,
        )