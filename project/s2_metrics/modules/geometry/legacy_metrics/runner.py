from __future__ import annotations

from collections import defaultdict

from . import registry
from .policy import (
    apply_runtime_confidence_gates,
    filter_metric_values_for_context,
    spec_production_ready,
)
from .types import MetricContext, MetricValue


def compute_single_photo_metrics(ctx: MetricContext) -> tuple[list[MetricValue], list[dict[str, str]]]:
    specs = [
        s
        for s in registry.specs_for_bucket(ctx.pose_bucket, scope="single")
        if spec_production_ready(s)
    ]
    by_impl: dict[str, list] = defaultdict(list)
    for spec in specs:
        by_impl[spec.implementation].append(spec)

    values: list[MetricValue] = []
    errors: list[dict[str, str]] = []
    modules = {m.__name__.split(".")[-1] + ".py": m for m in registry.load_modules()}
    for impl, impl_specs in by_impl.items():
        module = modules.get(impl)
        if module is None or not hasattr(module, "compute"):
            continue
        try:
            values.extend(module.compute(ctx, impl_specs))
        except Exception as exc:
            errors.append({
                "module": impl,
                "error_type": type(exc).__name__,
                "message": str(exc)[:500],
            })
            continue

    from .selection import filter_selected, selected_only_enabled

    values = filter_metric_values_for_context(values, ctx)
    values = apply_runtime_confidence_gates(ctx, values)
    if selected_only_enabled():
        values = filter_selected(values, ctx.pose_bucket)
    return values, errors
