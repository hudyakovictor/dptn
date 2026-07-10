from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


class TextureCalibrator:
    """Строит baseline для текстурных и шумовых признаков."""

    def build_reference(self, records) -> dict[str, Any]:
        items = [self._extract(record) for record in records or []]
        items = [item for item in items if item["texture"]]
        if not items:
            return {
                "global_stats": {},
                "bucket_stats": {},
                "pairwise_noise": {},
                "quality_curve": {},
                "selected_metric_keys": [],
                "thresholds": {"texture_suspicion": 0.65},
            }

        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            grouped[item["bucket"]].append(item)

        bucket_stats: dict[str, dict[str, dict[str, float]]] = {}
        pairwise_noise: dict[str, dict[str, dict[str, float]]] = {}
        quality_curve: dict[str, dict[str, dict[str, float]]] = {}
        global_stats = self._describe_metrics(items)
        selected_metric_keys = [key for key, stats in global_stats.items() if stats.get("count", 0.0) >= max(3.0, len(items) / 5.0)]

        for bucket, bucket_items in grouped.items():
            bucket_stats[bucket] = self._describe_metrics(bucket_items)
            pairwise_noise[bucket] = self._fit_pairwise_noise(bucket_items)
            quality_curve[bucket] = self._fit_quality_curve(bucket_items)

        return {
            "global_stats": global_stats,
            "bucket_stats": bucket_stats,
            "pairwise_noise": pairwise_noise,
            "quality_curve": quality_curve,
            "selected_metric_keys": sorted(set(selected_metric_keys)),
            "thresholds": self._build_thresholds(global_stats),
        }

    def _extract(self, record: Any) -> dict[str, Any]:
        bucket = "unknown"
        quality = 0.0
        texture = {}

        if hasattr(record, "bucket"):
            bucket = str(getattr(getattr(record, "bucket"), "value", getattr(record, "bucket")))
        elif isinstance(record, dict):
            bucket = str(record.get("bucket", bucket))

        if hasattr(record, "quality"):
            quality_obj = getattr(record, "quality")
            if quality_obj is not None and hasattr(quality_obj, "overall_quality"):
                quality = float(getattr(quality_obj, "overall_quality", 0.0) or 0.0)
        elif isinstance(record, dict):
            quality_obj = record.get("quality", {})
            if isinstance(quality_obj, dict):
                quality = float(quality_obj.get("overall_quality", 0.0) or 0.0)

        if hasattr(record, "texture"):
            texture = getattr(record, "texture") or {}
        elif isinstance(record, dict):
            texture = record.get("texture", {}) or {}

        return {
            "bucket": bucket,
            "quality": quality,
            "texture": self._numeric_map(texture),
        }

    def _numeric_map(self, payload: Any) -> dict[str, float]:
        if not isinstance(payload, dict):
            return {}
        return {str(k): float(v) for k, v in payload.items() if isinstance(v, (int, float))}

    def _describe_metrics(self, items: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
        keys = sorted({key for item in items for key in item["texture"].keys()})
        stats: dict[str, dict[str, float]] = {}
        for key in keys:
            values = np.asarray([item["texture"][key] for item in items if key in item["texture"]], dtype=float)
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
        ordered = sorted(items, key=lambda item: (item["quality"], item["bucket"]))
        deltas: dict[str, list[float]] = defaultdict(list)
        for left, right in zip(ordered[:-1], ordered[1:]):
            for key in sorted(set(left["texture"]) & set(right["texture"])):
                deltas[key].append(abs(float(left["texture"][key]) - float(right["texture"][key])))
        return {key: self._describe(values) for key, values in deltas.items() if values}

    def _fit_quality_curve(self, items: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
        curve: dict[str, dict[str, float]] = {}
        quality = np.asarray([item["quality"] for item in items], dtype=float)
        if quality.size < 4 or np.std(quality) < 1e-6:
            return curve
        for key in sorted({key for item in items for key in item["texture"].keys()}):
            values = np.asarray([item["texture"][key] for item in items if key in item["texture"]], dtype=float)
            if values.size < 4 or np.std(values) < 1e-6:
                continue
            q = np.asarray([item["quality"] for item in items if key in item["texture"]], dtype=float)
            if q.size < 4 or np.std(q) < 1e-6:
                continue
            slope, intercept = np.polyfit(q, values, 1)
            corr = float(np.corrcoef(q, values)[0, 1])
            if np.isfinite(slope) and np.isfinite(intercept) and np.isfinite(corr):
                curve[key] = {
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "corr": corr,
                    "n": float(q.size),
                }
        return curve

    def _build_thresholds(self, stats: dict[str, dict[str, float]]) -> dict[str, float]:
        entropy_like = [v["mean"] + v["std"] for k, v in stats.items() if "entropy" in k or "laplacian" in k or "fft" in k]
        return {
            "texture_suspicion": float(np.clip(np.mean(entropy_like) / 3.0 if entropy_like else 0.65, 0.3, 0.9)),
            "texture_mad": float(np.clip(np.median([v["mad"] for v in stats.values() if isinstance(v, dict)]) if stats else 1.0, 0.1, 2.5)),
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
