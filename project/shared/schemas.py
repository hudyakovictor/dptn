from __future__ import annotations

from datetime import date as dt_date
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class PipelineDataset(str, Enum):
    MAIN = "main"
    CALIBRATION = "calibration"


class PoseBucket(str, Enum):
    FRONTAL = "frontal"
    LEFT_THREEQUARTER_LIGHT = "left_threequarter_light"
    RIGHT_THREEQUARTER_LIGHT = "right_threequarter_light"
    LEFT_THREEQUARTER_MEDIUM = "left_threequarter_medium"
    RIGHT_THREEQUARTER_MEDIUM = "right_threequarter_medium"
    LEFT_THREEQUARTER_DEEP = "left_threequarter_deep"
    RIGHT_THREEQUARTER_DEEP = "right_threequarter_deep"
    LEFT_PROFILE = "left_profile"
    RIGHT_PROFILE = "right_profile"
    UNKNOWN = "unknown"


class Hypothesis(str, Enum):
    H0_SAME = "H0_SAME"
    H1_SYNTHETIC = "H1_SYNTHETIC"
    H2_DIFFERENT = "H2_DIFFERENT"
    H_UNCERTAIN = "H_UNCERTAIN"


class VerdictLabel(str, Enum):
    SAME = "same"
    SYNTHETIC = "synthetic"
    DIFFERENT = "different"
    UNCERTAIN = "uncertain"


class QualityMetrics(BaseModel):
    blur_value: float = 0.0
    noise_level: float = 0.0
    jpeg_blockiness: float = 0.0
    sharpness_score: float = 0.0
    overall_quality: float = 0.0
    is_motion_blurred: bool = False
    is_jpeg_blocky: bool = False
    is_over_smoothed: bool = False


class PoseEstimate(BaseModel):
    photo_id: str
    date: dt_date | None = None
    age_years: float | None = None
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    bucket: PoseBucket = PoseBucket.UNKNOWN
    pose_source: str = "filename"
    confidence: float = 0.5


class GeometryMetric(BaseModel):
    name: str
    value: float
    confidence: float = 1.0
    status: str = "ok"
    notes: str = ""


class TextureMetric(BaseModel):
    name: str
    value: float
    confidence: float = 1.0
    status: str = "ok"
    notes: str = ""


class Stage1Record(BaseModel):
    photo_id: str
    dataset: PipelineDataset
    source_path: str
    date: dt_date | None = None
    age_years: float | None = None
    pose: PoseEstimate
    quality: QualityMetrics
    face_bbox: list[int] = Field(default_factory=list)
    face_mask_path: str
    reconstruction_path: str
    image_size: list[int] = Field(default_factory=list)
    expression_flags: dict[str, bool] = Field(default_factory=dict)
    readiness: dict[str, str] = Field(default_factory=dict)


class Stage2Record(BaseModel):
    photo_id: str
    dataset: PipelineDataset
    bucket: PoseBucket
    quality: QualityMetrics
    geometry: dict[str, float] = Field(default_factory=dict)
    texture: dict[str, float] = Field(default_factory=dict)
    selected_metric_keys: list[str] = Field(default_factory=list)
    metric_notes: dict[str, str] = Field(default_factory=dict)
    geometry_identity_hint: str = "UNCERTAIN"
    geometry_identity_confidence: float = 0.0
    texture_skin_hint: str = "unknown"
    texture_skin_confidence: float = 0.0
    quality_summary: dict[str, float] = Field(default_factory=dict)


class CalibrationReference(BaseModel):
    dataset: PipelineDataset = PipelineDataset.CALIBRATION
    generated_at: str
    photo_count: int
    bucket_stats: dict[str, dict[str, dict[str, float]]] = Field(default_factory=dict)
    pairwise_noise: dict[str, dict[str, dict[str, float]]] = Field(default_factory=dict)
    age_profiles: dict[str, dict[str, dict[str, float]]] = Field(default_factory=dict)
    global_stats: dict[str, dict[str, float]] = Field(default_factory=dict)
    selected_metric_keys: list[str] = Field(default_factory=list)
    thresholds: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class PairEvidence(BaseModel):
    pair_id: str
    photo_a: str
    photo_b: str
    bucket: str
    date_gap_days: int = 0
    age_gap_years: float = 0.0
    pose_gap_deg: float = 0.0
    geometry_distance: float = 0.0
    texture_distance: float = 0.0
    age_explained_distance: float = 0.0
    quality_penalty: float = 1.0
    chronology_penalty: float = 1.0
    noise_discount: float = 0.0
    metric_overlap: int = 0
    synthetic_suspicion: float = 0.0
    different_suspicion: float = 0.0
    same_suspicion: float = 0.0
    anomaly_flags: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ForensicVerdict(BaseModel):
    photo_id: str
    hypothesis: Hypothesis
    posterior: dict[str, float] = Field(default_factory=dict)
    confidence: float = 0.0
    reasoning: list[str] = Field(default_factory=list)
    evidence: dict[str, float] = Field(default_factory=dict)


class Stage3Record(BaseModel):
    photo_id: str
    dataset: PipelineDataset
    identity_hint: str
    identity_confidence: float
    skin_hint: str
    skin_confidence: float
    geometry_distance: float
    texture_suspicion: float
    notes: list[str] = Field(default_factory=list)


class Stage4Record(BaseModel):
    photo_id: str
    dataset: PipelineDataset
    pairwise: list[PairEvidence] = Field(default_factory=list)
    chronology_flags: list[str] = Field(default_factory=list)
    strongest_neighbor: str | None = None


class Stage5Record(BaseModel):
    photo_id: str
    dataset: PipelineDataset
    verdict: ForensicVerdict
    chronology_score: float = 0.0
    anomaly_score: float = 0.0


class Stage6Record(BaseModel):
    dataset: PipelineDataset
    generated_at: str
    summary: dict[str, Any] = Field(default_factory=dict)
    chronology_summary: dict[str, Any] = Field(default_factory=dict)
    personas: list[dict[str, Any]] = Field(default_factory=list)
    theses: list[str] = Field(default_factory=list)
    statistics: dict[str, Any] = Field(default_factory=dict)
    top_anomalies: list[dict[str, Any]] = Field(default_factory=list)
    verdict_counts: dict[str, int] = Field(default_factory=dict)
    timeline_brief: list[dict[str, Any]] = Field(default_factory=list)


MetricsRecord = Stage2Record
ReportBundle = Stage6Record
