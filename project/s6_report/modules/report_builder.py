from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


class ReportBuilder:
    """Собирает структурированный аналитический отчёт."""

    def build(self, data) -> dict:
        verdicts = list(data.get("verdicts", []) or [])
        timeline = list(data.get("timeline", []) or [])
        chronology = data.get("chronology", {}) or {}
        personas = list(data.get("personas", []) or [])

        verdict_counts = Counter(self._hypothesis(item) for item in verdicts)
        anomaly_scores = np.asarray([float(row.get("anomaly_score", 0.0) or 0.0) for row in timeline], dtype=float)
        confidence_scores = np.asarray([float(self._confidence(item)) for item in verdicts], dtype=float)
        top_anomalies = sorted(timeline, key=lambda row: float(row.get("anomaly_score", 0.0) or 0.0), reverse=True)[:20]

        summary = {
            "photo_count": len(verdicts),
            "dominant_hypothesis": verdict_counts.most_common(1)[0][0] if verdict_counts else "H_UNCERTAIN",
            "mean_confidence": float(np.mean(confidence_scores) if confidence_scores.size else 0.0),
            "mean_anomaly": float(np.mean(anomaly_scores) if anomaly_scores.size else 0.0),
            "chronology_score": float(chronology.get("anomaly_score", 0.0) or 0.0),
        }

        theses = [
            f"Доминирующая гипотеза: {summary['dominant_hypothesis']}.",
            f"Средняя уверенность: {summary['mean_confidence']:.2f}.",
            f"Средний anomaly_score: {summary['mean_anomaly']:.2f}.",
        ]
        if personas:
            most_troubled = max(personas, key=lambda item: float(item.get("mean_anomaly_score", 0.0)))
            theses.append(
                f"Наиболее тревожная эпоха: {most_troubled['era']} ({float(most_troubled.get('mean_anomaly_score', 0.0)):.2f})."
            )

        return {
            "summary": summary,
            "verdict_counts": dict(verdict_counts),
            "top_anomalies": top_anomalies,
            "personas": personas,
            "theses": theses,
            "chronology_summary": self._chronology_summary(chronology),
            "timeline_brief": timeline[:200],
        }

    def _hypothesis(self, item: Any) -> str:
        if hasattr(item, "verdict"):
            verdict = getattr(item, "verdict")
            if hasattr(verdict, "hypothesis"):
                hyp = getattr(verdict, "hypothesis")
                return str(getattr(hyp, "value", hyp))
        if isinstance(item, dict):
            if "verdict" in item and isinstance(item["verdict"], dict):
                hyp = item["verdict"].get("hypothesis", "H_UNCERTAIN")
                return str(getattr(hyp, "value", hyp))
            hyp = item.get("hypothesis", "H_UNCERTAIN")
            return str(getattr(hyp, "value", hyp))
        return "H_UNCERTAIN"

    def _confidence(self, item: Any) -> float:
        if hasattr(item, "verdict"):
            verdict = getattr(item, "verdict")
            if hasattr(verdict, "confidence"):
                return float(getattr(verdict, "confidence", 0.0) or 0.0)
        if isinstance(item, dict):
            if "verdict" in item and isinstance(item["verdict"], dict):
                return float(item["verdict"].get("confidence", 0.0) or 0.0)
            return float(item.get("confidence", 0.0) or 0.0)
        return 0.0

    def _chronology_summary(self, chronology: dict[str, Any]) -> dict[str, Any]:
        points = chronology.get("points", []) if isinstance(chronology, dict) else []
        return {
            "point_count": len(points),
            "summary_flags": chronology.get("summary_flags", []) if isinstance(chronology, dict) else [],
            "anomaly_score": float(chronology.get("anomaly_score", 0.0) or 0.0) if isinstance(chronology, dict) else 0.0,
        }
