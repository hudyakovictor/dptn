from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .catalog import GEOMETRY_CORE_METRICS, load_geometry_metric_catalog


@dataclass
class GeometryRule:
    bucket: str
    metric_name: str
    face: str
    center: float
    spread: float
    weight: float


class GeometryIdentityResolver:
    """Быстрый three-face resolver по ключевым геометрическим метрикам."""

    def __init__(self, evidence_table: str | Path | None = None) -> None:
        self.rules = self._load_rules(evidence_table)

    def resolve(self, metrics: dict[str, float], reference: dict | None = None) -> dict[str, Any]:
        bucket = self._bucket(metrics, reference)
        scores = {"PUT": 0.0, "UDMURT": 0.0, "VAS": 0.0}
        hits = []
        for rule in self.rules:
            if rule.bucket not in {"all", bucket}:
                continue
            if rule.metric_name not in metrics:
                continue
            value = float(metrics[rule.metric_name])
            dist = abs(value - rule.center) / max(rule.spread, 1e-6)
            contrib = rule.weight / (1.0 + dist)
            scores[rule.face] += contrib
            hits.append(
                {
                    "bucket": rule.bucket,
                    "metric": rule.metric_name,
                    "face": rule.face,
                    "value": round(value, 6),
                    "center": round(rule.center, 6),
                    "spread": round(rule.spread, 6),
                    "contribution": round(contrib, 6),
                }
            )

        if reference and reference.get("thresholds"):
            scores["PUT"] *= 1.05
            scores["UDMURT"] *= 1.0
            scores["VAS"] *= 1.0

        best = max(scores, key=scores.get)
        ordered = sorted(scores.values(), reverse=True)
        confidence = float(np.clip(ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0], 0.0, 1.0))
        if confidence < 0.08:
            best = "UNCERTAIN"
        return {
            "identity_hint": best,
            "identity_confidence": confidence,
            "scores": scores,
            "pairwise_gap": self._pairwise_gap(scores),
            "hits": hits[:20],
            "selected_metric_keys": self.selected_metric_keys(metrics, bucket=bucket),
            "bucket": bucket,
        }

    def selected_metric_keys(self, metrics: dict[str, float], bucket: str | None = None) -> list[str]:
        names = [name for name in GEOMETRY_CORE_METRICS if name in metrics]
        if bucket:
            names.extend([rule.metric_name for rule in self.rules if rule.bucket in {"all", bucket} and rule.metric_name in metrics])
        return sorted(set(names))

    def _load_rules(self, evidence_table: str | Path | None) -> list[GeometryRule]:
        if evidence_table is None or not Path(evidence_table).exists():
            return self._load_rules_from_results() or self._fallback_rules()
        df = pd.read_csv(evidence_table)
        rules: list[GeometryRule] = []
        for _, row in df.iterrows():
            metric = str(row.get("метрика") or row.get("metric_name") or "").strip()
            if not metric:
                continue
            for face in ("PUT", "UDMURT", "VAS"):
                center, spread = self._parse_range(row.get(f"{face}_диапазон"))
                if center is None:
                    continue
                weight = abs(float(row.get(f"{face}_effect", row.get("max_effect", 1.0)) or 1.0))
                rules.append(
                    GeometryRule(
                        bucket="all",
                        metric_name=metric,
                        face=face,
                        center=center,
                        spread=max(spread, 1e-6),
                        weight=max(weight, 0.05),
                    )
                )
        return rules or self._fallback_rules()

    def _load_rules_from_results(self) -> list[GeometryRule]:
        root = Path(__file__).resolve().parents[4] / "results"
        if not root.exists():
            return []
        rules: list[GeometryRule] = []
        for csv_path in sorted(root.glob("ракурс_*_полный.csv")):
            bucket = csv_path.stem.replace("ракурс_", "").replace("_полный", "")
            try:
                df = pd.read_csv(csv_path, encoding="utf-8-sig")
            except Exception:
                continue
            for _, row in df.iterrows():
                metric = str(row.get("метрика") or "").strip()
                if not metric:
                    continue
                comment = str(row.get("комментарий") or "")
                weight_boost = 1.0
                if "Удмурт vs Василич" in comment:
                    weight_boost += 0.12
                if "Путин vs Удмурт" in comment:
                    weight_boost += 0.38
                if "Путин vs Василич" in comment:
                    weight_boost += 0.38
                for face in ("PUT", "UDMURT", "VAS"):
                    center, spread = self._parse_range(row.get(f"{face}_диапазон"))
                    if center is None:
                        continue
                    face_boost = 0.0
                    if face == "PUT":
                        face_boost = weight_boost
                    elif face in {"UDMURT", "VAS"} and weight_boost > 1.0:
                        face_boost = weight_boost * 0.65
                    rules.append(
                        GeometryRule(
                            bucket=bucket,
                            metric_name=metric,
                            face=face,
                            center=center,
                            spread=max(spread, 1e-6),
                            weight=max(0.05, weight_boost + face_boost),
                        )
                    )
        return rules

    def _fallback_rules(self) -> list[GeometryRule]:
        # Короткий запасной набор, если таблица недоступна.
        seeds = [
            ("all", "palpebral_aperture_aspect_ratio_L", "PUT", 0.49, 0.04, 1.2),
            ("all", "zone_rel_jaw_L_to_brow_ridge_L_distance_ratio", "UDMURT", 0.87, 0.02, 1.1),
            ("all", "zone_rel_jaw_L_to_brow_ridge_L_distance_ratio", "VAS", 0.88, 0.02, 0.9),
            ("all", "R_malar_peak_height_ratio", "VAS", 0.19, 0.03, 1.0),
            ("all", "L_malar_peak_height_ratio", "UDMURT", 0.19, 0.03, 1.0),
        ]
        return [GeometryRule(*seed) for seed in seeds]

    def _bucket(self, metrics: dict[str, float], reference: dict | None) -> str:
        if reference and isinstance(reference, dict):
            bucket = reference.get("bucket")
            if bucket:
                return str(bucket)
        return str(metrics.get("bucket", "all"))

    def _parse_range(self, text: Any) -> tuple[float | None, float]:
        if text is None or (isinstance(text, float) and math.isnan(text)):
            return None, 0.0
        s = str(text)
        match = re.search(r"([0-9.]+)\s*\[([0-9.]+)\s*[–-]\s*([0-9.]+)\]", s)
        if not match:
            try:
                value = float(s)
            except Exception:
                return None, 0.0
            return value, max(abs(value) * 0.1, 0.01)
        center = float(match.group(1))
        low = float(match.group(2))
        high = float(match.group(3))
        spread = max((high - low) / 2.0, 0.01)
        return center, spread

    def _pairwise_gap(self, scores: dict[str, float]) -> dict[str, float]:
        put = float(scores.get("PUT", 0.0))
        udmurt = float(scores.get("UDMURT", 0.0))
        vas = float(scores.get("VAS", 0.0))
        return {
            "PUT_vs_UDMURT": round(put - udmurt, 6),
            "PUT_vs_VAS": round(put - vas, 6),
            "UDMURT_vs_VAS": round(udmurt - vas, 6),
        }
