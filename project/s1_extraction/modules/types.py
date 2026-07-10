from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

@dataclass
class ReconstructionResult:
    image_path: Path
    vertices_world: np.ndarray
    vertices_camera: np.ndarray
    vertices_image: np.ndarray
    triangles: np.ndarray
    point_buffer: np.ndarray
    annotation_groups: List[np.ndarray]
    visible_idx_renderer: np.ndarray
    normals_world: np.ndarray
    normals_camera: np.ndarray
    rotation_matrix: np.ndarray
    translation: np.ndarray
    angles_deg: np.ndarray
    trans_params: Optional[np.ndarray] = None
    landmarks_106: Optional[np.ndarray] = None
    uv_coords: Optional[np.ndarray] = None
    pose_bucket: str = "frontal"          # ← НОВОЕ
    payload: Dict[str, Any] = field(default_factory=dict)

    @property
    def points(self) -> np.ndarray:
        """Alias for vertices_world."""
        return self.vertices_world

@dataclass(frozen=True)
class VisibilityResult:
    binary_mask: np.ndarray
    cosine_weights: np.ndarray
    facing_cosines: np.ndarray
    visible_count: int

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VisibilityResult):
            return NotImplemented
        return (
            self.visible_count == other.visible_count
            and np.array_equal(self.binary_mask, other.binary_mask)
            and np.array_equal(self.cosine_weights, other.cosine_weights)
            and np.array_equal(self.facing_cosines, other.facing_cosines)
        )

    def __hash__(self) -> int:
        return hash((self.visible_count, self.binary_mask.tobytes(), self.cosine_weights.tobytes(), self.facing_cosines.tobytes()))

@dataclass(frozen=True)
class AlignmentResult:
    rotation: np.ndarray
    translation: np.ndarray
    scale: float
    source_aligned: np.ndarray
    residual_before: float
    residual_after: float
    residual_before_sum: float = 0.0
    residual_after_sum: float = 0.0
    is_fallback: bool = False

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AlignmentResult):
            return NotImplemented
        return (
            self.scale == other.scale
            and self.residual_before == other.residual_before
            and self.residual_after == other.residual_after
            and self.residual_before_sum == other.residual_before_sum
            and self.residual_after_sum == other.residual_after_sum
            and self.is_fallback == other.is_fallback
            and np.array_equal(self.rotation, other.rotation)
            and np.array_equal(self.translation, other.translation)
            and np.array_equal(self.source_aligned, other.source_aligned)
        )

    def __hash__(self) -> int:
        return hash((
            self.scale,
            self.residual_before,
            self.residual_after,
            self.residual_before_sum,
            self.residual_after_sum,
            self.is_fallback,
            self.rotation.tobytes(),
            self.translation.tobytes(),
        ))

@dataclass(frozen=True)
class ZoneMetric:
    name: str
    status: str
    shared_vertex_count: int
    analysis_role: str
    bone_priority_class: str
    bone_weight: float
    raw_error: Optional[float] = None
    bounded_score: Optional[float] = None
    mean_weight: Optional[float] = None
    mean_signed_depth_delta: Optional[float] = None
    mean_signed_lateral_delta: Optional[float] = None
    mean_signed_vertical_delta: Optional[float] = None
    principal_shift_axis: Optional[str] = None
    dominant_shift_direction: Optional[str] = None
    delta_mm: Optional[float] = None
    delta_rel: Optional[float] = None
    metric: Optional[str] = None
    view: Optional[str] = None
    z_score: Optional[float] = None
    level_1_5: Optional[int] = None
    text_user: Optional[str] = None
    text_expert: Optional[str] = None

@dataclass(frozen=True)
class ComparisonResult:
    status: str
    shared_vertex_indices: np.ndarray
    score_raw: Optional[float]
    score_bounded: Optional[float]
    robust_score_raw: Optional[float]
    robust_score_bounded: Optional[float]
    provisional_band: str
    robust_provisional_band: str
    visibility_a: VisibilityResult
    visibility_b: VisibilityResult
    alignment: Optional[AlignmentResult]
    zones: List[ZoneMetric]
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ComparisonResult):
            return NotImplemented
        return (
            self.status == other.status
            and self.score_raw == other.score_raw
            and self.score_bounded == other.score_bounded
            and self.provisional_band == other.provisional_band
            and self.robust_provisional_band == other.robust_provisional_band
            and self.visibility_a == other.visibility_a
            and self.visibility_b == other.visibility_b
            and self.alignment == other.alignment
            and self.zones == other.zones
            and np.array_equal(self.shared_vertex_indices, other.shared_vertex_indices)
        )

    def __hash__(self) -> int:
        return hash((self.status, self.shared_vertex_indices.tobytes(), self.provisional_band))
