from __future__ import annotations

from collections import defaultdict
from typing import Any

from . import registry
from .types import MetricValue, PairMetricContext


def compute_pair_metrics(
    ctx: PairMetricContext,
    *,
    errors: list[dict[str, Any]] | None = None,
    strict: bool = False,
) -> list[MetricValue]:
    from typing import Any

    bucket = ctx.pose_bucket_a if ctx.pose_bucket_a == ctx.pose_bucket_b else ctx.pose_bucket_a
    specs = registry.specs_for_bucket(bucket, scope="pair")
    by_impl: dict[str, list] = defaultdict(list)
    for spec in specs:
        by_impl[spec.implementation].append(spec)
    modules = {m.__name__.split(".")[-1] + ".py": m for m in registry.load_modules()}
    values: list[MetricValue] = []
    for impl, impl_specs in by_impl.items():
        module = modules.get(impl)
        if module is None or not hasattr(module, "compute_pair"):
            if errors is not None:
                errors.append({
                    "module": impl,
                    "spec_count": len(impl_specs),
                    "error": "module_not_found_or_no_compute_pair",
                })
            continue
        try:
            values.extend(module.compute_pair(ctx, impl_specs))
        except Exception as exc:
            if errors is not None:
                errors.append({
                    "module": impl,
                    "spec_count": len(impl_specs),
                    "error": str(exc),
                })
            if strict:
                raise
            continue
    from .selection import filter_selected, selected_only_enabled
    return filter_selected(values, bucket) if selected_only_enabled() else values
