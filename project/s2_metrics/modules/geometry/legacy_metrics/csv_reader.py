from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable

DEFAULT_SINGLE_SPACES: tuple[str, ...] = ("canon_bucket", "shape_neutral", "raw")


def _finite_float(value: object) -> float | None:
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def flat_metrics_from_csv(
    path: Path,
    *,
    scope: str = "single",
    prefer_spaces: Iterable[str] = DEFAULT_SINGLE_SPACES,
) -> dict[str, float]:
    """Load single-photo metrics from metrics.csv into a flat dict.

    When the same metric_name appears in several source_space rows, the first
  matching row in prefer_spaces wins.
    """
    if not path.is_file():
        return {}

    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}

    space_rank = {s: i for i, s in enumerate(prefer_spaces)}
    best: dict[str, tuple[int, float]] = {}

    for row in rows:
        if row.get("scope", "single") != scope:
            continue
        name = row.get("metric_name", "")
        if not name:
            continue
        val = _finite_float(row.get("value"))
        if val is None:
            continue
        space = row.get("source_space", "")
        rank = space_rank.get(space, len(space_rank) + 1)
        prev = best.get(name)
        if prev is None or rank < prev[0]:
            best[name] = (rank, val)

    return {name: val for name, (_rank, val) in best.items()}
