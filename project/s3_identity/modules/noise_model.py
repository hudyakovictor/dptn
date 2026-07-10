from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


class NoiseModel:
    """Оценивает шум по ракурсам, качеству и времени."""

    def fit(self, records) -> dict[str, Any]:
        items = [self._extract(record) for record in records or []]
        items = [item for item in items if item["metrics"]]
        if not items:
            return {"bucket_models": {}, "global_noise": {}, "quality_noise_curve": {}, "summary": {}}

        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            grouped[item["bucket"]].append(item)

        bucket_models: dict[str, dict[str, Any]] = {}
        quality_noise_curve: dict[str, dict[str, dict[str, float]]] = {}
        global_noise = self._describe_metric_block(items)
        for bucket, bucket_items in grouped.items():
            bucket_models[bucket] = {
                "metrics": self._describe_metric_block(bucket_items),
                "temporal_noise": self._fit_temporal_noise(bucket_items),
                "quality_noise": self._fit_quality_noise(bucket_items),
                "pairwise_noise": self._pairwise_noise(bucket_items),
            }
            quality_noise_curve[bucket] = bucket_models[bucket]["quality_noise"]

        summary = {
            "bucket_count": len(bucket_models),
            "metric_count": len(global_noise),
            "noisy_buckets": [bucket for bucket, model in bucket_models.items() if model["temporal_noise"]],
        }
        return {
            "bucket_models": bucket_models,
            "global_noise": global_noise,
            "quality_noise_curve": quality_noise_curve,
            "summary": summary,
        }

    def _extract(self, record: Any) -> dict[str, Any]:
        bucket = "unknown"
        age_years = None
        quality = 0.0
        metrics = {}

        if hasattr(record, "bucket"):
            bucket = str(getattr(getattr(record, "bucket"), "value", getattr(record, "bucket")))
        elif isinstance(record, dict):
            bucket = str(record.get("bucket", bucket))

        if hasattr(record, "age_years"):
            age_years = getattr(record, "age_years")
        elif isinstance(record, dict):
            age_years = record.get("age_years")

        if hasattr(record, "quality"):
            quality_obj = getattr(record, "quality")
            if quality_obj is not None and hasattr(quality_obj, "overall_quality"):
                quality = float(getattr(quality_obj, "overall_quality", 0.0) or 0.0)
        elif isinstance(record, dict):
            quality_obj = record.get("quality", {})
            if isinstance(quality_obj, dict):
                quality = float(quality_obj.get("overall_quality", 0.0) or 0.0)

        if hasattr(record, "geometry"):
            metrics.update(self._numeric_map(getattr(record, "geometry") or {}))
        if hasattr(record, "texture"):
            metrics.update(self._numeric_map(getattr(record, "texture") or {}))
        if isinstance(record, dict):
            metrics.update(self._numeric_map(record.get("geometry", {}) or {}))
            metrics.update(self._numeric_map(record.get("texture", {}) or {}))

        return {
            "bucket": bucket,
            "age_years": float(age_years) if isinstance(age_years, (int, float)) else None,
            "quality": quality,
            "metrics": metrics,
        }

    def _numeric_map(self, payload: Any) -> dict[str, float]:
        if not isinstance(payload, dict):
            return {}
        return {str(k): float(v) for k, v in payload.items() if isinstance(v, (int, float))}

    def _describe_metric_block(self, items: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
        keys = sorted({key for item in items for key in item["metrics"].keys()})
        result: dict[str, dict[str, float]] = {}
        for key in keys:
            arr = np.asarray([item["metrics"][key] for item in items if key in item["metrics"]], dtype=float)
            if arr.size == 0:
                continue
            med = float(np.median(arr))
            result[key] = {
                "count": float(arr.size),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr) or 1e-6),
                "median": med,
                "mad": float(np.median(np.abs(arr - med))) or 1e-6,
            }
        return result

    def _fit_temporal_noise(self, items: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
        if len(items) < 4:
            return {}
        ordered = sorted(items, key=lambda item: (item["age_years"] is None, item["age_years"] or 9999.0))
        deltas: dict[str, list[float]] = defaultdict(list)
        gaps: dict[str, list[float]] = defaultdict(list)
        for left, right in zip(ordered[:-1], ordered[1:]):
            age_gap = abs((left["age_years"] or 0.0) - (right["age_years"] or 0.0))
            for key in sorted(set(left["metrics"]) & set(right["metrics"])):
                deltas[key].append(abs(float(left["metrics"][key]) - float(right["metrics"][key])))
                gaps[key].append(age_gap)
        model: dict[str, dict[str, float]] = {}
        for key, values in deltas.items():
            if len(values) < 3:
                continue
            y = np.asarray(values, dtype=float)
            x = np.asarray(gaps[key], dtype=float)
            if np.std(x) < 1e-6 or np.std(y) < 1e-6:
                continue
            slope, intercept = np.polyfit(x, y, 1)
            corr = float(np.corrcoef(x, y)[0, 1])
            model[key] = {"slope": float(slope), "intercept": float(intercept), "corr": corr, "n": float(len(values))}
        return model

    def _fit_quality_noise(self, items: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
        if len(items) < 4:
            return {}
        quality = np.asarray([item["quality"] for item in items], dtype=float)
        model: dict[str, dict[str, float]] = {}
        for key in sorted({key for item in items for key in item["metrics"].keys()}):
            values = np.asarray([item["metrics"][key] for item in items if key in item["metrics"]], dtype=float)
            q = np.asarray([item["quality"] for item in items if key in item["metrics"]], dtype=float)
            if values.size < 4 or np.std(q) < 1e-6 or np.std(values) < 1e-6:
                continue
            slope, intercept = np.polyfit(q, values, 1)
            corr = float(np.corrcoef(q, values)[0, 1])
            model[key] = {"slope": float(slope), "intercept": float(intercept), "corr": corr, "n": float(values.size)}
        return model

    def _pairwise_noise(self, items: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
        if len(items) < 2:
            return {}
        deltas: dict[str, list[float]] = defaultdict(list)
        for left, right in zip(items[:-1], items[1:]):
            for key in sorted(set(left["metrics"]) & set(right["metrics"])):
                deltas[key].append(abs(float(left["metrics"][key]) - float(right["metrics"][key])))
        return {key: self._describe(values) for key, values in deltas.items() if values}

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
