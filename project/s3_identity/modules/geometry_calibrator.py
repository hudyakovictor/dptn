from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


class GeometryCalibrator:
    """Строит robust baseline для геометрических метрик."""

    def build_reference(self, records) -> dict[str, Any]:
        items = [self._extract(record) for record in records or []]
        items = [item for item in items if item["geometry"]]
        if not items:
            return {
                "global_stats": {},
                "bucket_stats": {},
                "pairwise_noise": {},
                "age_profiles": {},
                "selected_metric_keys": [],
                "thresholds": {"geometry_distance": 1.0},
            }

        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            grouped[item["bucket"]].append(item)

        bucket_stats: dict[str, dict[str, dict[str, float]]] = {}
        pairwise_noise: dict[str, dict[str, dict[str, float]]] = {}
        age_profiles: dict[str, dict[str, dict[str, float]]] = {}
        global_stats: dict[str, dict[str, float]] = {}
        selected_metric_keys: list[str] = []

        for bucket, bucket_items in grouped.items():
            bucket_stats[bucket] = self._describe_metrics(bucket_items)
            pairwise_noise[bucket] = self._fit_pairwise_noise(bucket_items)
            age_profiles[bucket] = self._fit_age_profiles(bucket_items)

        global_stats = self._describe_metrics(items)
        for key, stats in global_stats.items():
            if stats.get("count", 0.0) >= max(3.0, len(items) / 5.0):
                selected_metric_keys.append(key)

        return {
            "global_stats": global_stats,
            "bucket_stats": bucket_stats,
            "pairwise_noise": pairwise_noise,
            "age_profiles": age_profiles,
            "selected_metric_keys": sorted(set(selected_metric_keys)),
            "thresholds": self._build_thresholds(global_stats),
        }

    def _extract(self, record: Any) -> dict[str, Any]:
        bucket = "unknown"
        age_years = None
        geometry = {}
        quality = 0.0

        if hasattr(record, "bucket"):
            bucket = str(getattr(getattr(record, "bucket"), "value", getattr(record, "bucket")))
        elif isinstance(record, dict):
            bucket = str(record.get("bucket", bucket))

        if hasattr(record, "age_years"):
            age_years = getattr(record, "age_years")
        elif isinstance(record, dict):
            age_years = record.get("age_years")

        if hasattr(record, "geometry"):
            geometry = getattr(record, "geometry") or {}
        elif isinstance(record, dict):
            geometry = record.get("geometry", {}) or {}

        if hasattr(record, "quality"):
            quality_obj = getattr(record, "quality")
            if quality_obj is not None and hasattr(quality_obj, "overall_quality"):
                quality = float(getattr(quality_obj, "overall_quality", 0.0) or 0.0)
        elif isinstance(record, dict):
            quality_obj = record.get("quality", {})
            if isinstance(quality_obj, dict):
                quality = float(quality_obj.get("overall_quality", 0.0) or 0.0)

        return {
            "bucket": bucket,
            "age_years": float(age_years) if isinstance(age_years, (int, float)) else None,
            "quality": quality,
            "geometry": self._numeric_map(geometry),
        }

    def _numeric_map(self, payload: Any) -> dict[str, float]:
        if not isinstance(payload, dict):
            return {}
        return {str(k): float(v) for k, v in payload.items() if isinstance(v, (int, float))}

    def _describe_metrics(self, items: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
        keys = sorted({key for item in items for key in item["geometry"].keys()})
        stats: dict[str, dict[str, float]] = {}
        for key in keys:
            values = np.asarray([item["geometry"][key] for item in items if key in item["geometry"]], dtype=float)
            if values.size == 0:
                continue
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median))) or 1e-6
            stats[key] = {
                "count": float(values.size),
                "mean": float(np.mean(values)),
                "std": float(np.std(values) or 1e-6),
                "median": median,
                "mad": mad,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        return stats

    def _fit_pairwise_noise(self, items: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
        if len(items) < 2:
            return {}
        ordered = sorted(items, key=lambda item: (item["age_years"] is None, item["age_years"] or 9999.0))
        deltas: dict[str, list[float]] = defaultdict(list)
        for left, right in zip(ordered[:-1], ordered[1:]):
            for key in sorted(set(left["geometry"]) & set(right["geometry"])):
                deltas[key].append(abs(float(left["geometry"][key]) - float(right["geometry"][key])))
        return {key: self._describe(values) for key, values in deltas.items() if values}

    def _fit_age_profiles(self, items: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
        profiles: dict[str, dict[str, float]] = {}
        points: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for item in items:
            age = item.get("age_years")
            if age is None:
                continue
            for key, value in item["geometry"].items():
                points[key].append((float(age), float(value)))
        for key, pairs in points.items():
            if len(pairs) < 4:
                continue
            ages = np.asarray([p[0] for p in pairs], dtype=float)
            vals = np.asarray([p[1] for p in pairs], dtype=float)
            if np.std(ages) < 1e-6 or np.std(vals) < 1e-6:
                continue
            slope, intercept = np.polyfit(ages, vals, 1)
            corr = float(np.corrcoef(ages, vals)[0, 1])
            if np.isfinite(slope) and np.isfinite(intercept) and np.isfinite(corr):
                profiles[key] = {
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "corr": corr,
                    "n": float(len(pairs)),
                }
        return profiles

    def _build_thresholds(self, stats: dict[str, dict[str, float]]) -> dict[str, float]:
        values = [v["mean"] + v["std"] for v in stats.values() if isinstance(v, dict)]
        return {
            "geometry_distance": float(np.clip(np.mean(values) if values else 1.0, 0.5, 3.0)),
            "geometry_mad": float(np.clip(np.median([v["mad"] for v in stats.values() if isinstance(v, dict)]) if stats else 1.0, 0.1, 2.0)),
        }

    def _describe(self, values: list[float]) -> dict[str, float]:
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return {"count": 0.0, "mean": 0.0, "std": 0.0, "median": 0.0, "mad": 0.0}
        med = float(np.median(arr))
        return {
            "count": float(arr.size),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr) or 1e-6),
            "median": med,
            "mad": float(np.median(np.abs(arr - med))) or 1e-6,
        }
