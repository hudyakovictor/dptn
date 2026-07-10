from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np


class PersonaAggregator:
    """Агрегирует эпохи, гипотезы и аномалии в persona-профили."""

    def build(self, timeline, verdicts) -> list[dict]:
        rows = list(timeline or [])
        verdict_map = self._verdict_map(verdicts or [])
        if not rows:
            return []

        eras = [
            ("1998-2005", 1998, 2005),
            ("2005-2012", 2005, 2012),
            ("2012-2019", 2012, 2019),
            ("2019-2022", 2019, 2022),
            ("2022-2026", 2022, 2026),
        ]
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            year = self._year(row.get("date"))
            if year is None:
                continue
            for label, start, end in eras:
                if start <= year <= end:
                    grouped[label].append(row)
                    break

        personas: list[dict] = []
        for label, _, _ in eras:
            era_rows = grouped.get(label, [])
            if not era_rows:
                continue
            hypothesis_counts = Counter(self._hypothesis(row, verdict_map) for row in era_rows)
            anomaly_scores = np.asarray([float(row.get("anomaly_score", 0.0) or 0.0) for row in era_rows], dtype=float)
            confidence_scores = np.asarray([float(self._confidence(row, verdict_map)) for row in era_rows], dtype=float)
            dates = [self._year(row.get("date")) for row in era_rows if self._year(row.get("date")) is not None]
            trend = self._trend(dates, anomaly_scores.tolist()) if len(dates) >= 3 else 0.0
            personas.append(
                {
                    "era": label,
                    "count": len(era_rows),
                    "dominant_hypothesis": hypothesis_counts.most_common(1)[0][0],
                    "hypothesis_counts": dict(hypothesis_counts),
                    "mean_anomaly_score": float(np.mean(anomaly_scores) if anomaly_scores.size else 0.0),
                    "mean_confidence": float(np.mean(confidence_scores) if confidence_scores.size else 0.0),
                    "trend": float(trend),
                    "top_photos": sorted(era_rows, key=lambda row: float(row.get("anomaly_score", 0.0) or 0.0), reverse=True)[:5],
                }
            )
        return personas

    def _verdict_map(self, verdicts: list[Any]) -> dict[str, Any]:
        mapping = {}
        for item in verdicts:
            if hasattr(item, "photo_id"):
                mapping[str(getattr(item, "photo_id"))] = item
            elif isinstance(item, dict):
                photo_id = item.get("photo_id")
                if photo_id:
                    mapping[str(photo_id)] = item
        return mapping

    def _hypothesis(self, row: dict[str, Any], verdict_map: dict[str, Any]) -> str:
        photo_id = str(row.get("photo_id", ""))
        verdict = verdict_map.get(photo_id)
        if verdict is None:
            return str(row.get("hypothesis", "H_UNCERTAIN"))
        if hasattr(verdict, "verdict"):
            verdict_obj = getattr(verdict, "verdict")
            return str(getattr(verdict_obj, "hypothesis", "H_UNCERTAIN")).split(".")[-1]
        if isinstance(verdict, dict):
            if "verdict" in verdict and isinstance(verdict["verdict"], dict):
                hyp = verdict["verdict"].get("hypothesis", "H_UNCERTAIN")
                return str(getattr(hyp, "value", hyp))
            hyp = verdict.get("hypothesis", "H_UNCERTAIN")
            return str(getattr(hyp, "value", hyp))
        return "H_UNCERTAIN"

    def _confidence(self, row: dict[str, Any], verdict_map: dict[str, Any]) -> float:
        photo_id = str(row.get("photo_id", ""))
        verdict = verdict_map.get(photo_id)
        if verdict is None:
            return float(row.get("confidence", 0.0) or 0.0)
        if hasattr(verdict, "verdict"):
            verdict_obj = getattr(verdict, "verdict")
            return float(getattr(verdict_obj, "confidence", 0.0) or 0.0)
        if isinstance(verdict, dict):
            if "verdict" in verdict and isinstance(verdict["verdict"], dict):
                return float(verdict["verdict"].get("confidence", 0.0) or 0.0)
            return float(verdict.get("confidence", 0.0) or 0.0)
        return float(row.get("confidence", 0.0) or 0.0)

    def _year(self, date_value: Any) -> int | None:
        if not date_value:
            return None
        try:
            return int(str(date_value)[:4])
        except Exception:
            return None

    def _trend(self, xs: list[int], ys: list[float]) -> float:
        if len(xs) < 3 or len(xs) != len(ys):
            return 0.0
        x = np.asarray(xs, dtype=float)
        y = np.asarray(ys, dtype=float)
        if np.std(x) < 1e-6 or np.std(y) < 1e-6:
            return 0.0
        slope, _ = np.polyfit(x, y, 1)
        return float(slope)
