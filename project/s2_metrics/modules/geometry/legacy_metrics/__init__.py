"""Pluggable metric registry for extraction/classifier candidates.

This package is intentionally separate from :mod:`pipeline.metrics` to avoid an
import-name collision with the legacy module.
"""

from .types import MetricSpec, MetricValue, MetricContext
from .runner import compute_single_photo_metrics
from .csv_writer import write_metrics_csv, write_metrics_errors_csv

__all__ = [
    "MetricSpec",
    "MetricValue",
    "MetricContext",
    "compute_single_photo_metrics",
    "write_metrics_csv",
    "write_metrics_errors_csv",
]
