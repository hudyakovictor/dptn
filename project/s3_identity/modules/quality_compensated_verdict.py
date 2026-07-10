from __future__ import annotations

from typing import Any

import numpy as np


class QualityCompensatedVerdict:
    """Композитный verdict с учётом качества, шума и reference."""

    def compute(self, metrics, quality, reference) -> dict[str, Any]:
        geom = self._metrics(metrics, "geometry")
        tex = self._metrics(metrics, "texture")
        q = self._quality(quality)
        ref_thresholds = self._ref_get(reference, "thresholds", {}) or {}
        noise = self._ref_get(reference, "pairwise_noise", {}) or {}
        bucket = self._bucket(metrics)

        geometry_distance = self._distance(geom, reference, bucket, channel="geometry")
        texture_distance = self._distance(tex, reference, bucket, channel="texture")
        quality_penalty = float(np.clip(1.0 - q, 0.0, 1.0))
        noise_discount = self._noise_discount(noise, bucket, geom, tex)
        geometry_distance = max(0.0, geometry_distance - min(geometry_distance * 0.5, noise_discount))
        texture_distance = max(0.0, texture_distance - min(texture_distance * 0.45, noise_discount * 0.8))

        identity_score = float(np.clip(1.0 - (geometry_distance * 0.62 + texture_distance * 0.18 + quality_penalty * 0.15), 0.0, 1.0))
        synthetic_score = float(np.clip((texture_distance * 0.72 + quality_penalty * 0.18 + noise_discount * 0.15), 0.0, 1.0))
        different_score = float(np.clip((geometry_distance * 0.78 + max(0.0, geometry_distance - texture_distance) * 0.15), 0.0, 1.0))
        uncertain_score = float(np.clip(1.0 - max(identity_score, synthetic_score, different_score), 0.0, 1.0))

        posterior = {
            "H0_SAME": identity_score,
            "H1_SYNTHETIC": synthetic_score,
            "H2_DIFFERENT": different_score,
            "H_UNCERTAIN": uncertain_score + quality_penalty * 0.05,
        }
        total = sum(posterior.values()) or 1.0
        posterior = {key: value / total for key, value in posterior.items()}
        hypothesis = max(posterior, key=posterior.get)
        confidence = float(posterior[hypothesis])

        notes = []
        if q < 0.25:
            notes.append("low_quality")
        if geometry_distance > ref_thresholds.get("geometry_distance", 1.0):
            notes.append("geometry_far")
        if texture_distance > ref_thresholds.get("texture_suspicion", 0.65):
            notes.append("texture_suspicious")
        if noise_discount > 0.2:
            notes.append("noise_discounted")

        return {
            "verdict": hypothesis,
            "posterior": posterior,
            "confidence": confidence,
            "evidence": {
                "geometry_distance": geometry_distance,
                "texture_distance": texture_distance,
                "quality": q,
                "quality_penalty": quality_penalty,
                "noise_discount": noise_discount,
            },
            "notes": notes,
        }

    def _metrics(self, record: Any, kind: str) -> dict[str, float]:
        if record is None:
            return {}
        if hasattr(record, kind):
            value = getattr(record, kind)
            if isinstance(value, dict):
                return self._numeric_map(value)
        if isinstance(record, dict):
            value = record.get(kind)
            if isinstance(value, dict):
                return self._numeric_map(value)
        return {}

    def _numeric_map(self, payload: dict[str, Any]) -> dict[str, float]:
        return {str(k): float(v) for k, v in payload.items() if isinstance(v, (int, float))}

    def _quality(self, quality: Any) -> float:
        if quality is None:
            return 0.0
        if hasattr(quality, "overall_quality"):
            return float(getattr(quality, "overall_quality", 0.0) or 0.0)
        if isinstance(quality, dict):
            return float(quality.get("overall_quality", 0.0) or 0.0)
        if isinstance(quality, (int, float)):
            return float(quality)
        return 0.0

    def _bucket(self, record: Any) -> str:
        if hasattr(record, "bucket"):
            bucket = getattr(record, "bucket")
            return str(getattr(bucket, "value", bucket))
        if isinstance(record, dict):
            return str(record.get("bucket", "unknown"))
        return "unknown"

    def _ref_get(self, reference: Any, key: str, default: Any = None) -> Any:
        if reference is None:
            return default
        if isinstance(reference, dict):
            return reference.get(key, default)
        if hasattr(reference, key):
            return getattr(reference, key)
        return default

    def _distance(self, metrics: dict[str, float], reference: Any, bucket: str, *, channel: str) -> float:
        if not metrics:
            return 0.0
        stats = self._ref_get(reference, "global_stats", {}) or {}
        noise = (self._ref_get(reference, "pairwise_noise", {}) or {}).get(bucket, {})
        values = []
        for key, value in metrics.items():
            stat = stats.get(key, {})
            scale = max(float(stat.get("mad", 0.0) or stat.get("std", 1.0) or 1.0), 1e-6)
            ref = float(stat.get("median", stat.get("mean", 0.0)))
            delta = abs(float(value) - ref) / scale
            delta = max(0.0, delta - min(float(noise.get(key, {}).get("mad", 0.0) or 0.0) / scale, 0.8))
            if channel == "texture":
                delta *= 0.9
            values.append(delta)
        return float(np.mean(values) if values else 0.0)

    def _noise_discount(self, noise: dict[str, Any], bucket: str, geom: dict[str, float], tex: dict[str, float]) -> float:
        bucket_noise = noise.get(bucket, {})
        keys = sorted(set(geom) | set(tex))
        discounts = []
        for key in keys:
            entry = bucket_noise.get(key, {})
            val = float(entry.get("mad", 0.0) or entry.get("std", 0.0) or 0.0)
            if val > 0:
                discounts.append(val)
        return float(np.mean(discounts) if discounts else 0.0)
