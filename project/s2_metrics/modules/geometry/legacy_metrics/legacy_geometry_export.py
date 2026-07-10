from __future__ import annotations

import json
from typing import Any

from .common import ALL_BUCKETS, emit, metric
from .types import MetricContext, MetricValue

IMPLEMENTATION = "legacy_geometry_export.py"


def _provenance_metric_base(key: str) -> str:
    if key.endswith("_clipped"):
        return key[: -len("_clipped")]
    return key


def build_legacy_geometry_provenance_values(
    ctx: MetricContext,
    geometry_metrics: dict[str, Any],
) -> list[MetricValue]:
    """Export geometry correction journal rows into metrics.csv."""
    provenance = geometry_metrics.get("provenance")
    if not isinstance(provenance, dict) or not provenance:
        return []

    out: list[MetricValue] = []
    for prov_key, entry in sorted(provenance.items()):
        if not isinstance(entry, dict):
            continue
        raw_value = entry.get("raw_value")
        clipped_value = entry.get("clipped_value")
        corrections = entry.get("corrections")
        if raw_value is None:
            continue

        base = _provenance_metric_base(str(prov_key))
        notes_payload: dict[str, Any] = {"provenance_key": prov_key}
        if clipped_value is not None:
            notes_payload["clipped_value"] = clipped_value
        if corrections:
            notes_payload["corrections"] = corrections
        notes = json.dumps(notes_payload, ensure_ascii=False, separators=(",", ":"))

        spec = metric(
            name=f"{base}_raw",
            family="F0",
            group="geometry_provenance",
            zone="geometry",
            source_spaces=("raw",),
            unit="ratio",
            normalization="face_scale",
            implementation=IMPLEMENTATION,
            buckets=ALL_BUCKETS,
        )
        mv = emit(spec, raw_value, confidence=1.0, source_space="raw", notes=notes)
        if mv is not None:
            out.append(mv)
    return out
