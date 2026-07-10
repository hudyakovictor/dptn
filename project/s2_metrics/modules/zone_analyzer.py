"""Zone analyzer - bucket-aware zone planning и zone quality assessment."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[4] / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

try:
    from metrics.zone_morphology import compute as zone_morphology_compute
    from metrics.zone_relations import compute as zone_relations_compute
    from metrics.types import MetricContext, MetricSpec
    _HAS_BACKEND = True
except ImportError:
    _HAS_BACKEND = False


# Bucket → recommended zones mapping
BUCKET_ZONE_HINTS: dict[str, list[str]] = {
    "frontal": [
        "orbit", "nose_bridge", "brow_ridge", "cheek", "jaw", "chin",
        "forehead", "temple", "periocular"
    ],
    "left_profile": [
        "orbit", "nose_bridge", "jaw", "mandible", "zygomatic", "temple"
    ],
    "right_profile": [
        "orbit", "nose_bridge", "jaw", "mandible", "zygomatic", "temple"
    ],
    "threequarter_light": [
        "orbit", "nose_bridge", "brow_ridge", "cheek", "jaw", "temple"
    ],
}


class ZoneAnalyzer:
    """Bucket-aware zone planner и zone quality assessment."""

    def __init__(self):
        pass

    def analyze(self, ctx: Any) -> dict[str, Any]:
        """
        Анализирует зоны и определяет какие зоны релевантны для данного ракурса.

        Args:
            ctx: MetricContext или dict с данными

        Returns:
            Словарь с результатами анализа:
            - recommended_zones: list[str] — рекомендованные зоны
            - zone_confidence: dict[str, float] — уверенность по зонам
            - zones_available: int — количество доступных зон
        """
        if not _HAS_BACKEND or not hasattr(ctx, 'pose_bucket'):
            return self._analyze_stub(ctx)

        bucket = ctx.pose_bucket
        recommended = BUCKET_ZONE_HINTS.get(bucket, BUCKET_ZONE_HINTS["frontal"])

        result = {
            "recommended_zones": recommended,
            "zone_confidence": {},
            "zones_available": 0,
        }

        # Check which annotation groups actually have data
        if hasattr(ctx, 'annotation_groups') and ctx.annotation_groups:
            result["zones_available"] = len(ctx.annotation_groups)

        # Compute zone morphology if available
        if _HAS_BACKEND and hasattr(ctx, 'annotation_groups') and ctx.annotation_groups:
            try:
                morph_specs = [s for s in zone_morphology.specs() if s.scope == "single"]
                morph_values = zone_morphology.compute(ctx, morph_specs)
                for mv in morph_values:
                    if mv.value is not None:
                        result["zone_confidence"][mv.spec.name] = mv.confidence
            except Exception:
                pass

        return result

    def get_zone_weights(self, ctx: Any) -> dict[str, float]:
        """
        Возвращает веса зон для взвешенного агрегирования.

        Args:
            ctx: MetricContext

        Returns:
            Словарь {zone_name: weight} где weight ∈ [0, 1]
        """
        if not _HAS_BACKEND or not hasattr(ctx, 'pose_bucket'):
            return {}

        bucket = ctx.pose_bucket
        weights = {}

        recommended = BUCKET_ZONE_HINTS.get(bucket, BUCKET_ZONE_HINTS["frontal"])
        for zone in recommended:
            weights[zone] = 1.0  # базовый вес для рекомендованных зон

        # Уменьшаем веса для нерекомендованных зон
        if hasattr(ctx, 'annotation_groups') and ctx.annotation_groups:
            for zone_idx in range(len(ctx.annotation_groups)):
                zone_name = f"zone_{zone_idx}"
                if zone_name not in weights:
                    weights[zone_name] = 0.3  # пониженный вес для нерекомендованных

        return weights

    def _analyze_stub(self, ctx: Any) -> dict[str, Any]:
        """Заглушка когда backend недоступен."""
        return {
            "recommended_zones": [],
            "zone_confidence": {},
            "zones_available": 0,
        }
