from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


class ShiftModel:
    """Оценивает возрастной и позовый сдвиг метрик."""

    def fit(self, records) -> dict[str, Any]:
        items = [self._extract(record) for record in records or []]
        items = [item for item in items if item["metrics"]]
        if not items:
            return {"shifts": {}, "summary": {}}

        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            grouped[item["bucket"]].append(item)

        shifts: dict[str, Any] = {}
        for bucket, bucket_items in grouped.items():
            shifts[bucket] = {
                "age_shift": self._fit_linear_shift(bucket_items, feature="age_years"),
                "yaw_shift": self._fit_linear_shift(bucket_items, feature="yaw"),
                "pitch_shift": self._fit_linear_shift(bucket_items, feature="pitch"),
                "roll_shift": self._fit_linear_shift(bucket_items, feature="roll"),
                "quality_shift": self._fit_linear_shift(bucket_items, feature="quality"),
            }

        summary = {
            "bucket_count": len(shifts),
            "metric_keys": sorted({key for item in items for key in item["metrics"].keys()}),
            "strong_drifts": self._strong_drifts(shifts),
        }
        return {"shifts": shifts, "summary": summary}

    def _extract(self, record: Any) -> dict[str, Any]:
        bucket = "unknown"
        age_years = None
        yaw = pitch = roll = 0.0
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

        pose = getattr(record, "pose", None) if hasattr(record, "pose") else (record.get("pose") if isinstance(record, dict) else None)
        if pose is not None:
            if isinstance(pose, dict):
                yaw = float(pose.get("yaw", 0.0) or 0.0)
                pitch = float(pose.get("pitch", 0.0) or 0.0)
                roll = float(pose.get("roll", 0.0) or 0.0)
            else:
                yaw = float(getattr(pose, "yaw", 0.0) or 0.0)
                pitch = float(getattr(pose, "pitch", 0.0) or 0.0)
                roll = float(getattr(pose, "roll", 0.0) or 0.0)

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
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "quality": quality,
            "metrics": metrics,
        }

    def _numeric_map(self, payload: Any) -> dict[str, float]:
        if not isinstance(payload, dict):
            return {}
        return {str(k): float(v) for k, v in payload.items() if isinstance(v, (int, float))}

    def _fit_linear_shift(self, items: list[dict[str, Any]], feature: str) -> dict[str, dict[str, float]]:
        if len(items) < 4:
            return {}
        model: dict[str, dict[str, float]] = {}
        for key in sorted({key for item in items for key in item["metrics"].keys()}):
            y = np.asarray([item["metrics"][key] for item in items if key in item["metrics"]], dtype=float)
            xk = np.asarray([item.get(feature) if item.get(feature) is not None else 0.0 for item in items if key in item["metrics"]], dtype=float)
            if y.size < 4 or np.std(xk) < 1e-6 or np.std(y) < 1e-6:
                continue
            A = np.column_stack([np.ones_like(xk), xk])
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            pred = A @ coef
            resid = y - pred
            ss_res = float(np.sum(resid ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2)) or 1e-6
            r2 = max(0.0, 1.0 - ss_res / ss_tot)
            model[key] = {
                "intercept": float(coef[0]),
                "slope": float(coef[1]),
                "r2": float(r2),
                "residual_std": float(np.std(resid) or 1e-6),
                "n": float(y.size),
            }
        return model

    def _strong_drifts(self, shifts: dict[str, Any]) -> list[str]:
        flags: list[str] = []
        for bucket, payload in shifts.items():
            for family, values in payload.items():
                if not isinstance(values, dict):
                    continue
                strong = [key for key, stats in values.items() if isinstance(stats, dict) and abs(float(stats.get("slope", 0.0))) > 0.01 and float(stats.get("r2", 0.0)) > 0.2]
                if strong:
                    flags.append(f"{bucket}:{family}:{len(strong)}")
        return flags
