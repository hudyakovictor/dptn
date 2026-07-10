from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

from .types import MetricValue

DEFAULT_SELECTED_PATH = Path(__file__).with_name("selected_metrics.json")


def load_selected_metrics(path: Path | None = None) -> dict:
    p = path or Path(os.environ.get("NEWWAP_SELECTED_METRICS", DEFAULT_SELECTED_PATH))
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def metric_keys_for_value(v: MetricValue) -> set[str]:
    keys = {v.spec.name, f"{v.spec.family}.{v.spec.group}.{v.spec.name}"}
    for space in v.spec.source_spaces:
        keys.add(f"{space}.{v.spec.name}")
        keys.add(f"{space}.{v.spec.family}.{v.spec.group}.{v.spec.name}")
    return keys


def filter_selected(values: Iterable[MetricValue], bucket: str, selected: dict | None = None) -> list[MetricValue]:
    cfg = selected if selected is not None else load_selected_metrics()
    if not cfg:
        return list(values)
    allowed = set(cfg.get(bucket, [])) | set(cfg.get("all", []))
    if not allowed:
        return []
    out = []
    for v in values:
        if metric_keys_for_value(v) & allowed:
            out.append(v)
    return out


def selected_only_enabled() -> bool:
    return os.environ.get("NEWWAP_METRICS_SELECTED_ONLY", "0").strip().lower() in {"1", "true", "yes"}
