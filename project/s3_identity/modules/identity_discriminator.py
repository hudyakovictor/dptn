from __future__ import annotations

from typing import Any

import numpy as np


class IdentityDiscriminator:
    """PUT / OTHER discrimination with reference-aware scoring."""

    def classify(self, metrics, reference) -> dict[str, Any]:
        geometry = self._metrics(metrics, "geometry")
        texture = self._metrics(metrics, "texture")
        quality = self._quality(metrics)
        bucket = self._bucket(metrics)
        thresholds = self._ref_get(reference, "thresholds", {}) or {}
        global_stats = self._ref_get(reference, "global_stats", {}) or {}
        age_profiles = self._ref_get(reference, "age_profiles", {}) or {}
        pairwise_noise = self._ref_get(reference, "pairwise_noise", {}) or {}

        geometry_distance = self._distance(geometry, global_stats, pairwise_noise, bucket)
        texture_distance = self._distance(texture, global_stats, pairwise_noise, bucket, channel="texture")
        texture_suspicion = self._texture_suspicion(texture, reference, bucket)
        identity_score = float(np.clip(1.0 - geometry_distance / 3.0, 0.0, 1.0))
        identity_score = float(np.clip(identity_score + max(0.0, 0.15 - texture_suspicion * 0.12), 0.0, 1.0))
        skin_score = float(np.clip(texture_suspicion + max(0.0, quality - 0.4) * 0.15, 0.0, 1.0))

        if self._age_years(metrics) is not None and bucket in age_profiles:
            age_years = self._age_years(metrics) or 0.0
            identity_score = float(np.clip(identity_score - self._age_shift(age_profiles[bucket], age_years) * 0.15, 0.0, 1.0))

        posterior = {
            "PUT": identity_score * (0.75 + quality * 0.25),
            "OTHER": (1.0 - identity_score) * (0.8 + texture_suspicion * 0.2),
            "UNCERTAIN": max(0.05, abs(identity_score - 0.5) * 0.35 + (1.0 - quality) * 0.2),
            "SILICONE": skin_score * (0.7 + (1.0 - quality) * 0.3),
        }
        total = sum(posterior.values()) or 1.0
        posterior = {key: value / total for key, value in posterior.items()}
        hint = max(("PUT", "OTHER", "UNCERTAIN"), key=lambda key: posterior.get(key, 0.0))
        if posterior.get("SILICONE", 0.0) > 0.32 and texture_suspicion > thresholds.get("texture_suspicion", 0.65):
            skin_hint = "silicone"
        else:
            skin_hint = "real"

        evidence = {
            "geometry_distance": geometry_distance,
            "texture_distance": texture_distance,
            "texture_suspicion": texture_suspicion,
            "quality": quality,
        }
        notes = []
        if geometry_distance > thresholds.get("geometry_distance", 1.0):
            notes.append("geometry_far_from_baseline")
        if texture_suspicion > thresholds.get("texture_suspicion", 0.65):
            notes.append("texture_suspicious")
        if quality < 0.25:
            notes.append("low_quality")

        return {
            "identity_hint": hint,
            "identity_confidence": float(posterior.get(hint, 0.0)),
            "skin_hint": skin_hint,
            "skin_confidence": float(posterior.get("SILICONE" if skin_hint == "silicone" else "PUT", 0.0)),
            "posterior": posterior,
            "evidence": evidence,
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

    def _bucket(self, record: Any) -> str:
        if hasattr(record, "bucket"):
            bucket = getattr(record, "bucket")
            return str(getattr(bucket, "value", bucket))
        if isinstance(record, dict):
            return str(record.get("bucket", "unknown"))
        return "unknown"

    def _quality(self, record: Any) -> float:
        if hasattr(record, "quality"):
            quality = getattr(record, "quality")
            if quality is not None and hasattr(quality, "overall_quality"):
                return float(getattr(quality, "overall_quality", 0.0) or 0.0)
        if isinstance(record, dict):
            quality = record.get("quality", {})
            if isinstance(quality, dict):
                return float(quality.get("overall_quality", 0.0) or 0.0)
        return 0.0

    def _age_years(self, record: Any) -> float | None:
        if hasattr(record, "age_years"):
            value = getattr(record, "age_years")
            if isinstance(value, (int, float)):
                return float(value)
        if isinstance(record, dict):
            value = record.get("age_years")
            if isinstance(value, (int, float)):
                return float(value)
        return None

    def _distance(
        self,
        metrics: dict[str, float],
        global_stats: dict[str, Any],
        pairwise_noise: dict[str, Any],
        bucket: str,
        *,
        channel: str = "geometry",
    ) -> float:
        if not metrics:
            return 0.0
        selected = [key for key in metrics if key in global_stats]
        if not selected:
            selected = list(metrics.keys())
        values = []
        noise = pairwise_noise.get(bucket, {})
        for key in selected:
            stat = global_stats.get(key, {})
            scale = max(float(stat.get("mad", 0.0) or stat.get("std", 1.0) or 1.0), 1e-6)
            ref = float(stat.get("median", stat.get("mean", 0.0)))
            delta = abs(float(metrics[key]) - ref) / scale
            noise_level = float(noise.get(key, {}).get("mad", 0.0) or noise.get(key, {}).get("std", 0.0) or 0.0)
            delta = max(0.0, delta - min(noise_level / scale, 0.9))
            if channel == "texture":
                delta *= 0.92
            values.append(delta)
        return float(np.mean(values) if values else 0.0)

    def _texture_suspicion(self, texture: dict[str, float], reference: Any, bucket: str) -> float:
        if not texture:
            return 0.0
        thresholds = self._ref_get(reference, "thresholds", {}) or {}
        stats = self._ref_get(reference, "global_stats", {}) or {}
        quality_curve = self._ref_get(reference, "quality_curve", {}) or {}
        values = []
        for key, value in texture.items():
            if key not in stats:
                continue
            stat = stats[key]
            scale = max(float(stat.get("mad", 0.0) or stat.get("std", 1.0) or 1.0), 1e-6)
            ref = float(stat.get("median", stat.get("mean", 0.0)))
            delta = abs(float(value) - ref) / scale
            if "texture" in key or "entropy" in key or "fft" in key:
                delta *= 1.1
            if key in quality_curve.get(bucket, {}):
                q = quality_curve[bucket][key]
                delta = max(0.0, delta - abs(float(q.get("corr", 0.0))) * 0.12)
            values.append(delta)
        suspicion = float(np.clip(np.mean(values) / 3.0 if values else 0.0, 0.0, 1.0))
        return float(np.clip(suspicion + 0.05 * (thresholds.get("texture_suspicion", 0.65) - 0.65), 0.0, 1.0))

    def _age_shift(self, profiles: dict[str, Any], age_years: float) -> float:
        values = []
        for key, profile in profiles.items():
            if not isinstance(profile, dict):
                continue
            slope = abs(float(profile.get("slope", 0.0)))
            corr = abs(float(profile.get("corr", 0.0)))
            values.append(slope * age_years * (1.0 + min(corr, 1.0)))
        return float(np.median(values) if values else 0.0)

    def _ref_get(self, reference: Any, key: str, default: Any = None) -> Any:
        if reference is None:
            return default
        if isinstance(reference, dict):
            return reference.get(key, default)
        if hasattr(reference, key):
            return getattr(reference, key)
        return default
