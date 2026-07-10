from __future__ import annotations

from typing import Any, Iterable

from . import registry
from .common import ALL_BUCKETS, emit, metric
from .policy import (
    filter_metric_values_for_bucket,
    spec_production_ready,
    filter_bucket_production_metrics,
    filter_bucket_experimental_metrics,
    is_bucket_core_metric,
    is_bucket_experimental_metric,
)
from .types import MetricContext, MetricValue

EXISTING_BACKEND = "existing_backend.py"
IMPLEMENTATION = "production.py"

# Имена в пайплайне → канонические ключи из BUCKET_METRIC_KEYS / selected_metric_keys.
CANONICAL_ALIASES: dict[str, str] = {
    "texture_lbp_entropy": "texture_lbp_complexity",
}

PIPELINE_VALUE_ALIASES: dict[str, tuple[str, ...]] = {
    "texture_lbp_complexity": ("texture_lbp_complexity", "texture_lbp_entropy"),
}


def canonical_metric_name(name: str, allowed: set[str]) -> str:
    if name in allowed:
        return name
    if name.endswith("_raw"):
        return name
    return CANONICAL_ALIASES.get(name, name)


def is_production_metric(name: str, allowed: set[str]) -> bool:
    if name in allowed:
        return True
    if name.endswith("_raw"):
        return name[:-4] in allowed
    alias = CANONICAL_ALIASES.get(name)
    return alias in allowed if alias else False


def _pipeline_value(name: str, pipeline_metrics: dict[str, Any]) -> Any:
    if name in pipeline_metrics:
        return pipeline_metrics.get(name)
    for alt in PIPELINE_VALUE_ALIASES.get(name, ()):
        if alt in pipeline_metrics:
            return pipeline_metrics.get(alt)
    return None


def _production_source_space(name: str) -> str:
    return "raw" if name.startswith("texture_") else "canon_bucket"


def filter_production_values(values: Iterable[MetricValue], allowed: set[str]) -> list[MetricValue]:
    return [v for v in values if is_production_metric(v.spec.name, allowed)]


def fill_production_scalars_from_pipeline(
    ctx: MetricContext,
    values: list[MetricValue],
    allowed: set[str],
    pipeline_metrics: dict[str, Any],
) -> list[MetricValue]:
    """Backfill BUCKET_METRIC_KEYS scalars already computed in extract pipeline."""
    present = {
        (canonical_metric_name(v.spec.name, allowed), v.source_space or _production_source_space(v.spec.name))
        for v in values
    }
    out = list(values)
    for name in sorted(allowed):
        source_space = _production_source_space(name)
        key = (name, source_space)
        if key in present:
            continue
        val = _pipeline_value(name, pipeline_metrics)
        if val is None:
            continue
        spec = metric(
            name=name,
            family="F0",
            group="pipeline_bridge",
            zone="production",
            source_spaces=(source_space,),
            implementation=IMPLEMENTATION,
            buckets=ALL_BUCKETS,
        )
        mv = emit(spec, val, source_space=source_space)
        if mv is not None:
            out.append(mv)
            present.add(key)
    return out


def compute_production_photo_metrics(
    ctx: MetricContext,
    production_names: Iterable[str],
    *,
    pipeline_metrics: dict[str, Any] | None = None,
) -> list[MetricValue]:
    """Compute only bucket production metrics (BUCKET_METRIC_KEYS), not the full candidate catalog."""
    allowed = set(production_names)
    specs = [
        s
        for s in registry.specs_for_bucket(ctx.pose_bucket, scope="single")
        if s.implementation == EXISTING_BACKEND
        and is_production_metric(s.name, allowed)
        and spec_production_ready(s)
    ]
    values: list[MetricValue] = []
    if specs:
        from . import existing_backend

        values = existing_backend.compute(ctx, specs)
        values = filter_production_values(values, allowed)
        values = filter_metric_values_for_bucket(values, ctx.pose_bucket, yaw_deg=ctx.yaw_deg)
        values = filter_bucket_production_metrics(values, ctx.pose_bucket)
    if pipeline_metrics:
        values = fill_production_scalars_from_pipeline(ctx, values, allowed, pipeline_metrics)
    return values


def compute_bucket_experimental_metrics(
    ctx: MetricContext,
    production_names: Iterable[str],
    *,
    pipeline_metrics: dict[str, Any] | None = None,
) -> list[MetricValue]:
    """Compute bucket experimental metrics separately for logging/analysis."""
    allowed = set(production_names)
    specs = [
        s
        for s in registry.specs_for_bucket(ctx.pose_bucket, scope="single")
        if s.implementation == EXISTING_BACKEND
        and is_bucket_experimental_metric(s.name, ctx.pose_bucket)
        and spec_production_ready(s)
    ]
    values: list[MetricValue] = []
    if specs:
        from . import existing_backend

        values = existing_backend.compute(ctx, specs)
        values = filter_production_values(values, allowed)
        values = filter_metric_values_for_bucket(values, ctx.pose_bucket, yaw_deg=ctx.yaw_deg)
        values = filter_bucket_experimental_metrics(values, ctx.pose_bucket)
    if pipeline_metrics:
        values = fill_production_scalars_from_pipeline(ctx, values, allowed, pipeline_metrics)
    return values
