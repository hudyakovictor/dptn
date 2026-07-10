from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from .policy import metric_value_writable, spec_allowed_for_context
from .production import canonical_metric_name, is_production_metric
from .types import MetricContext, MetricValue

SINGLE_FIELDNAMES = [
    "pose_bucket",
    "metric_name",
    "value",
    "metric_group",
    "anatomical_zone",
    "source_space",
    "metric_family",
]

# Single-photo: geometry (canon + shape-neutral identity) + raw texture.
SINGLE_SOURCE_SPACES = frozenset({"canon_bucket", "raw", "shape_neutral"})

PAIR_FIELDNAMES = [
    "pose_bucket",
    "metric_name",
    "value",
    "metric_group",
    "anatomical_zone",
    "source_space",
    "metric_family",
    "pair_id",
    "reference_id",
]

PAIR_SOURCE_SPACES = frozenset({"pair_umeyama", "pair_icp"})

ERRORS_FIELDNAMES = [
    "photo_id",
    "pose_bucket",
    "module",
    "error_type",
    "message",
]


def _is_angle_metric(metric_name: str, unit: str) -> bool:
    if unit == "deg" or "_deg" in metric_name:
        return True
    if metric_name.startswith("canthal_tilt"):
        return True
    if metric_name.endswith("_angle") or metric_name.endswith("_angle_raw"):
        return True
    return metric_name in {
        "gonial_angle_L",
        "gonial_angle_R",
        "nasofacial_angle_ratio",
        "glabella_nasion_projection_angle",
        "glabella_subnasale_pogonion_angle",
        "brow_asymmetry_deg",
        "canthal_tilt_asymmetry_deg",
    }


def format_metric_csv_value(value: float, *, metric_name: str, unit: str = "") -> str:
    """Human-readable rounding for metrics.csv (compare-friendly)."""
    v = float(value)
    if metric_name.endswith("_vertex_count") or metric_name.endswith("_count"):
        return str(int(round(v)))
    if _is_angle_metric(metric_name, unit):
        return f"{v:.2f}"
    if metric_name.startswith("texture_") and abs(v) >= 1.0:
        return f"{v:.2f}"
    if abs(v) >= 1.0 or abs(v) == 0.0:
        return f"{v:.4f}"
    # Tiny ratios: 4dp would show 0.0000 — keep extra precision
    rounded4 = round(v, 4)
    if rounded4 == 0.0:
        return f"{v:.6f}".rstrip("0").rstrip(".")
    return f"{rounded4:.4f}"


def _csv_value(mv: MetricValue) -> str:
    return format_metric_csv_value(mv.value, metric_name=mv.spec.name, unit=mv.spec.unit)


def _emit_spaces(mv: MetricValue) -> tuple[str, ...]:
    spec = mv.spec
    if mv.source_space:
        return (mv.source_space,)
    return spec.source_spaces[:1] or ("canon_bucket",)


def write_metrics_csv(
    path: Path,
    values: Iterable[MetricValue],
    ctx: MetricContext,
    *,
    append: bool = False,
    allowed_names: set[str] | None = None,
) -> int:
    rows = list(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and path.exists() else "w"
    with path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SINGLE_FIELDNAMES)
        if mode == "w":
            writer.writeheader()
        n = 0
        seen: set[tuple[str, str]] = set()
        for mv in rows:
            if not metric_value_writable(mv, ctx):
                continue
            if allowed_names is not None and not is_production_metric(mv.spec.name, allowed_names):
                continue
            metric_name = (
                canonical_metric_name(mv.spec.name, allowed_names)
                if allowed_names is not None
                else mv.spec.name
            )
            for source_space in _emit_spaces(mv):
                if source_space not in SINGLE_SOURCE_SPACES:
                    continue
                key = (metric_name, source_space)
                if key in seen:
                    continue
                seen.add(key)
                writer.writerow({
                    "pose_bucket": ctx.pose_bucket,
                    "metric_name": metric_name,
                    "value": _csv_value(mv),
                    "metric_group": mv.spec.group,
                    "anatomical_zone": mv.spec.zone,
                    "source_space": source_space,
                    "metric_family": mv.spec.family,
                })
                n += 1
    return n


def write_metrics_errors_csv(
    path: Path,
    errors: Iterable[dict[str, str]],
    *,
    photo_id: str,
    pose_bucket: str,
) -> int:
    rows = list(errors)
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ERRORS_FIELDNAMES)
        if write_header:
            writer.writeheader()
        n = 0
        for err in rows:
            writer.writerow({
                "photo_id": photo_id,
                "pose_bucket": pose_bucket,
                "module": err.get("module", ""),
                "error_type": err.get("error_type", ""),
                "message": err.get("message", ""),
            })
            n += 1
    return n


def write_pair_metrics_csv(path: Path, values: Iterable[MetricValue], pair_ctx, *, append: bool = True) -> int:
    from .policy import spec_allowed_for_bucket

    rows = list(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and path.exists() else "w"
    bucket = pair_ctx.pose_bucket_a if pair_ctx.pose_bucket_a == pair_ctx.pose_bucket_b else pair_ctx.pose_bucket_a
    with path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PAIR_FIELDNAMES)
        if mode == "w":
            writer.writeheader()
        n = 0
        for mv in rows:
            if not spec_allowed_for_bucket(mv.spec, bucket):
                continue
            for source_space in _emit_spaces(mv):
                if source_space not in PAIR_SOURCE_SPACES:
                    continue
                writer.writerow({
                    "pose_bucket": bucket,
                    "metric_name": mv.spec.name,
                    "value": _csv_value(mv),
                    "metric_group": mv.spec.group,
                    "anatomical_zone": mv.spec.zone,
                    "source_space": source_space,
                    "metric_family": mv.spec.family,
                    "pair_id": pair_ctx.pair_id,
                    "reference_id": pair_ctx.photo_id_b,
                })
                n += 1
    return n
