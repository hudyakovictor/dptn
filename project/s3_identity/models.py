from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class PoseNoiseModel:
    """Модель шума как функции от pose_gap."""
    bucket: str
    intercept: float = 0.0
    slope: float = 0.0
    curvature: float = 0.0
    p05: float = 0.0
    p95: float = 0.0
    mad: float = 0.0
    sample_count: int = 0
    quality_breakdown: Dict[str, Dict] = field(default_factory=dict)

    def predict(self, pose_gap_deg: float, quality: float = 0.5) -> Tuple[float, float]:
        """Returns (expected_noise, confidence_interval)."""
        base = self.intercept + self.slope * pose_gap_deg + self.curvature * (pose_gap_deg ** 2)
        quality_factor = 1.0 + max(0, 0.7 - quality) * 2.0
        ci = self.p95 * quality_factor
        return base, ci


@dataclass
class CalibrationBucketHealth:
    """Health monitoring для корзины калибровки."""
    bucket: str
    status: str = "unknown"
    photo_count: int = 0
    pose_coverage: Dict[str, float] = field(default_factory=dict)
    quality_coverage: Dict[str, float] = field(default_factory=dict)
    residual_check: Dict[str, float] = field(default_factory=dict)
    false_anomaly_rate: float = 0.0
    warnings: List[str] = field(default_factory=list)


@dataclass
class NoiseDiscountResult:
    """Результат применения калибровочной скидки шума."""
    excess_distance: float
    expected_noise: float
    confidence: float
    is_significant: bool
    model_bucket: str
    model_samples: int


@dataclass
class CalibrationPair:
    """Пара для калибровочной выборки."""
    pose_gap: float
    geom_dist: float
    tex_dist: float
    quality: float
    age_gap: float
    scale_diff: float
    per_metric_dists: Dict[str, float] = field(default_factory=dict)