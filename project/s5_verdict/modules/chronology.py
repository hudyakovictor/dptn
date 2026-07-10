from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from statistics import median

import numpy as np

from deeputin.shared.schemas import Stage1Record, Stage2Record


@dataclass
class ChronologyPoint:
    photo_id: str
    date: date | None
    bucket: str
    age_years: float | None
    chronology_score: float = 0.0
    flags: list[str] = field(default_factory=list)
    details: dict[str, float] = field(default_factory=dict)


@dataclass
class ChronologyResult:
    points: list[ChronologyPoint]
    summary_flags: list[str]
    anomaly_score: float


class ChronologyAnalyzer:
    """Лёгкий, но полезный движок временных аномалий."""

    def build(self, stage1_records: dict[str, Stage1Record], stage2_records: dict[str, Stage2Record]) -> ChronologyResult:
        ordered = sorted(
            [
                (photo_id, rec)
                for photo_id, rec in stage1_records.items()
                if photo_id in stage2_records and rec.date is not None
            ],
            key=lambda item: (item[1].date.isoformat(), item[0]),
        )
        points: list[ChronologyPoint] = []
        for photo_id, rec in ordered:
            points.append(
                ChronologyPoint(
                    photo_id=photo_id,
                    date=rec.date,
                    bucket=stage2_records[photo_id].bucket.value,
                    age_years=rec.age_years,
                )
            )

        if len(points) < 2:
            return ChronologyResult(points=points, summary_flags=[], anomaly_score=0.0)

        per_metric_series = self._series(stage2_records, ordered)
        per_photo_scores: dict[str, float] = {p.photo_id: 0.0 for p in points}
        per_photo_flags: dict[str, set[str]] = {p.photo_id: set() for p in points}

        for metric_name, values in per_metric_series.items():
            if len(values) < 3:
                continue
            deltas = np.abs(np.diff(values))
            baseline = float(np.median(deltas)) + 1e-6
            robust_scale = float(np.median(np.abs(deltas - baseline)) + 1e-6)
            age_trend = self._age_trend(points, values)
            for idx in range(1, len(values)):
                gap_days = max((points[idx].date - points[idx - 1].date).days if points[idx].date and points[idx - 1].date else 1, 1)
                rate = float(abs(values[idx] - values[idx - 1]) / gap_days)
                normalized = rate / max(baseline, 1e-6)
                if normalized > 2.2:
                    flag = f"spike:{metric_name}"
                    per_photo_scores[points[idx].photo_id] += 0.8
                    per_photo_flags[points[idx].photo_id].add(flag)
                    per_photo_flags[points[idx - 1].photo_id].add(flag)
                elif normalized > 1.3:
                    flag = f"elevated_change:{metric_name}"
                    per_photo_scores[points[idx].photo_id] += 0.35
                    per_photo_flags[points[idx].photo_id].add(flag)

                if age_trend and points[idx].age_years is not None and points[idx - 1].age_years is not None:
                    age_delta = float(points[idx].age_years - points[idx - 1].age_years)
                    if age_delta > 0:
                        direction = float(values[idx] - values[idx - 1])
                        if age_trend["slope"] > 0 and direction < -robust_scale * 0.8:
                            per_photo_scores[points[idx].photo_id] += 0.45
                            per_photo_flags[points[idx].photo_id].add(f"age_inversion:{metric_name}")
                        elif age_trend["slope"] < 0 and direction > robust_scale * 0.8:
                            per_photo_scores[points[idx].photo_id] += 0.45
                            per_photo_flags[points[idx].photo_id].add(f"age_inversion:{metric_name}")

            self._detect_return_to_baseline(metric_name, values, points, per_photo_scores, per_photo_flags)

        for point in points:
            point.chronology_score = float(np.clip(per_photo_scores[point.photo_id], 0.0, 3.0))
            point.flags = sorted(per_photo_flags[point.photo_id])
            point.details["age_years"] = float(point.age_years) if point.age_years is not None else 0.0

        summary_flags = self._summary_flags(points)
        anomaly_score = float(np.mean([p.chronology_score for p in points]))
        return ChronologyResult(points=points, summary_flags=summary_flags, anomaly_score=anomaly_score)

    def _age_trend(self, points: list[ChronologyPoint], values: list[float]) -> dict[str, float] | None:
        ages = []
        vals = []
        for point, value in zip(points, values):
            if point.age_years is None or not np.isfinite(value):
                continue
            ages.append(float(point.age_years))
            vals.append(float(value))
        if len(ages) < 4:
            return None
        age_arr = np.asarray(ages, dtype=float)
        val_arr = np.asarray(vals, dtype=float)
        if np.std(age_arr) < 1e-6 or np.std(val_arr) < 1e-6:
            return None
        slope = float(np.polyfit(age_arr, val_arr, 1)[0])
        corr = float(np.corrcoef(age_arr, val_arr)[0, 1])
        if not np.isfinite(slope) or not np.isfinite(corr):
            return None
        if abs(corr) < 0.35:
            return None
        return {"slope": slope, "corr": corr}

    def _series(
        self,
        stage2_records: dict[str, Stage2Record],
        ordered: list[tuple[str, Stage1Record]],
    ) -> dict[str, list[float]]:
        keys = sorted(
            {
                key
                for _, rec in ordered
                for key in list(stage2_records[rec.photo_id].geometry.keys())[:12] + list(stage2_records[rec.photo_id].texture.keys())[:8]
            }
        )
        series: dict[str, list[float]] = {key: [] for key in keys}
        for photo_id, _ in ordered:
            record = stage2_records[photo_id]
            merged = {**record.geometry, **record.texture}
            for key in keys:
                value = merged.get(key)
                if isinstance(value, (int, float)):
                    series[key].append(float(value))
                else:
                    series[key].append(float("nan"))
        return series

    def _detect_return_to_baseline(
        self,
        metric_name: str,
        values: list[float],
        points: list[ChronologyPoint],
        per_photo_scores: dict[str, float],
        per_photo_flags: dict[str, set[str]],
    ) -> None:
        if len(values) < 4:
            return
        arr = np.asarray(values, dtype=float)
        finite = np.isfinite(arr)
        if finite.sum() < 4:
            return
        arr = arr.copy()
        arr[~finite] = np.nanmedian(arr[finite])
        window = 3
        smoothed = np.convolve(arr, np.ones(window) / window, mode="same")
        overall = float(np.nanmedian(arr))
        before = np.abs(smoothed[:-2] - overall)
        after = np.abs(smoothed[2:] - overall)
        return_candidates = np.where((before > 0.75) & (after < 0.35))[0]
        for idx in return_candidates:
            photo = points[idx + 1].photo_id
            per_photo_scores[photo] += 0.6
            per_photo_flags[photo].add(f"return_to_baseline:{metric_name}")

    def _summary_flags(self, points: list[ChronologyPoint]) -> list[str]:
        if not points:
            return []
        scores = [p.chronology_score for p in points]
        if max(scores) > 1.5:
            return ["strong_temporal_break"]
        if median(scores) > 0.7:
            return ["multiple_temporal_anomalies"]
        if any(any(flag.startswith("age_inversion:") for flag in p.flags) for p in points):
            return ["age_inversion_detected"]
        return []
