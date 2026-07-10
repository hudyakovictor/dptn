from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

SourceSpace = Literal[
    "raw",
    "canon_bucket",
    "shape_neutral",
    "unit_template",
    "pair_umeyama",
    "pair_icp",
]
MetricScope = Literal["single", "pair", "template", "chronology"]
Side = Literal["L", "R", "B", "NA"]


@dataclass(frozen=True)
class MetricSpec:
    name: str
    family: str
    group: str
    zone: str
    side: Side = "NA"
    buckets: tuple[str, ...] = ("frontal",)
    source_spaces: tuple[SourceSpace, ...] = ("canon_bucket",)
    scope: MetricScope = "single"
    unit: str = "ratio"
    normalization: str = "none"
    expression_sensitive: bool = False
    pose_sensitive: str = "medium"
    required_sources: tuple[str, ...] = ()
    implementation: str = ""
    tags: tuple[str, ...] = ()


@dataclass
class MetricValue:
    spec: MetricSpec
    value: float | int | str | None
    confidence: float = 1.0
    visibility: float | None = None
    quality_gate: str = "pass"
    notes: str = ""
    source_space: SourceSpace | None = None


@dataclass
class MetricContext:
    photo_id: str
    image_path: Path
    pose_bucket: str
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    recon: Any
    vertices_raw: np.ndarray
    vertices_canon: np.ndarray
    vertices_shape_neutral: np.ndarray | None
    normals_raw: np.ndarray | None
    normals_canon: np.ndarray | None
    normals_shape_neutral: np.ndarray | None
    triangles: np.ndarray
    annotation_groups: list[np.ndarray]
    macro_indices: dict[str, Any]
    landmarks_106: np.ndarray | None
    visibility_raw: Any | None = None
    visibility_canon: Any | None = None
    id_params: np.ndarray | None = None
    exp_params: np.ndarray | None = None
    shape_basis: np.ndarray | None = None
    image_rgb: np.ndarray | None = None
    uv_coords: np.ndarray | None = None
    quality: dict[str, Any] = field(default_factory=dict)
    geometry_metrics: dict[str, Any] = field(default_factory=dict)
    periocular_metrics: dict[str, Any] = field(default_factory=dict)
    texture_forensics: dict[str, Any] = field(default_factory=dict)
    texture_profile: dict[str, Any] = field(default_factory=dict)

@dataclass
class PairMetricContext:
    photo_id_a: str
    photo_id_b: str
    pair_id: str
    pose_bucket_a: str
    pose_bucket_b: str
    yaw_a_deg: float
    yaw_b_deg: float
    shared_idx: np.ndarray
    vertices_a_unit: np.ndarray
    vertices_b_unit_aligned: np.ndarray
    normals_a: np.ndarray | None
    triangles: np.ndarray | None
    macro_indices: dict[str, Any]
    visibility_weights: np.ndarray | None = None
    alignment: Any | None = None
