from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from core.policy import POSE_YAW_BILATERAL_OFF_DEG

from .common import ALL_BUCKETS
from .types import MetricContext, MetricSpec, MetricValue

POSE_POLICY = {
    "frontal": {"visible_sides": {"L", "R", "B", "NA"}, "allow_bilateral": True, "allow_mirror": True},
    "left_threequarter_light": {"visible_sides": {"L", "B", "NA"}, "hidden_side": "R", "allow_bilateral": "conditional"},
    "right_threequarter_light": {"visible_sides": {"R", "B", "NA"}, "hidden_side": "L", "allow_bilateral": "conditional"},
    "left_threequarter_mid": {"visible_sides": {"L", "B", "NA"}, "hidden_side": "R", "allow_bilateral": False},
    "right_threequarter_mid": {"visible_sides": {"R", "B", "NA"}, "hidden_side": "L", "allow_bilateral": False},
    "left_threequarter_deep": {"visible_sides": {"L", "B", "NA"}, "profile_mode": True, "allow_bilateral": False},
    "right_threequarter_deep": {"visible_sides": {"R", "B", "NA"}, "profile_mode": True, "allow_bilateral": False},
    "left_profile": {"visible_sides": {"L", "NA"}, "profile_mode": True, "allow_bilateral": False},
    "right_profile": {"visible_sides": {"R", "NA"}, "profile_mode": True, "allow_bilateral": False},
    "unclassified": {"visible_sides": {"L", "R", "B", "NA"}, "allow_bilateral": "conditional"},
}

PRODUCTION_STATUSES = frozenset({"implemented", "partial", "legacy"})
BLOCKED_STATUSES = frozenset({"planned", "abandoned"})
MIN_PRODUCTION_CONFIDENCE = 0.35

DIAGNOSTIC_TEXTURE_KEYS = frozenset({
    "texture_glcm_contrast",
    "texture_glcm_homogeneity",
    "texture_block_median_score",
    "texture_mask_score",
    "texture_reliability",
    "texture_natural_score",
    "texture_silicone_prob",
    "texture_pore_density",
})


CONDITIONAL_BILATERAL_PAIR_BLOCK = frozenset({
    "interorbital_ratio",
    "intercanthal_width_ratio",
    "nose_width_ratio",
    "jaw_width_ratio",
    "bigonial_width_ratio",
    "bizygomatic_depth_asymmetry",
    "bilateral_center_depth_asymmetry",
})

CONDITIONAL_ASYMMETRY_BLOCK = frozenset({
    "orbital_asymmetry_index",
    "chin_offset_asymmetry",
    "gonial_width_asymmetry",
    "palpebral_fissure_asymmetry_ratio",
    "orbit_depth_asymmetry_ratio",
    "orbit_vertical_asymmetry_ratio",
    "canthal_tilt_asymmetry_deg",
})

PROFILE_FRONTAL_ONLY_METRICS = frozenset({
    "interorbital_ratio",
    "intercanthal_width_ratio",
    "jaw_width_ratio",
    "bigonial_width_ratio",
    "nose_width_ratio",
    "cranial_face_index",
    "orbital_asymmetry_index",
    "chin_offset_asymmetry",
    "gonial_width_asymmetry",
    "bizygomatic_depth_asymmetry",
    "bilateral_center_depth_asymmetry",
    "palpebral_fissure_asymmetry_ratio",
    "orbit_depth_asymmetry_ratio",
    "orbit_vertical_asymmetry_ratio",
    "canthal_tilt_asymmetry_deg",
    "mean_orbital_depth",
    "depth_asymmetry",
})


def spec_status(spec: MetricSpec) -> str | None:
    for tag in spec.tags:
        if tag in PRODUCTION_STATUSES or tag in BLOCKED_STATUSES:
            return tag
    return None


def is_diagnostic_texture_key(name: str) -> bool:
    n = str(name)
    if n in DIAGNOSTIC_TEXTURE_KEYS:
        return True
    if n.startswith("roi_") and n.endswith("_raw"):
        return True
    return n.startswith("texture_") and any(x in n for x in ("glcm", "block", "mask", "reliability"))


def spec_production_ready(spec: MetricSpec) -> bool:
    status = spec_status(spec)
    if status in BLOCKED_STATUSES:
        return False
    return True


def _conditional_bilateral_allowed(spec: MetricSpec, yaw_deg: float | None) -> bool:
    if spec.name in CONDITIONAL_BILATERAL_PAIR_BLOCK or spec.name in CONDITIONAL_ASYMMETRY_BLOCK:
        return False
    if spec.family == "F10":
        return False
    if yaw_deg is not None and abs(float(yaw_deg)) >= POSE_YAW_BILATERAL_OFF_DEG:
        return False
    return True


def spec_allowed_for_bucket(
    spec: MetricSpec,
    bucket: str,
    *,
    yaw_deg: float | None = None,
) -> bool:
    if bucket not in spec.buckets and "unclassified" not in spec.buckets:
        return False
    policy = POSE_POLICY.get(bucket, POSE_POLICY["unclassified"])
    side = spec.side

    hidden = policy.get("hidden_side")
    if hidden and side == hidden:
        return False

    if policy.get("profile_mode"):
        if side == "B":
            return False
        if spec.family == "F10":
            return False
        if spec.name in PROFILE_FRONTAL_ONLY_METRICS:
            return False
        if spec.name.startswith("mirror_"):
            return False

    allow_bilateral = policy.get("allow_bilateral")
    if side == "B":
        if allow_bilateral is False:
            return False
        if allow_bilateral == "conditional" and not _conditional_bilateral_allowed(spec, yaw_deg):
            return False

    if "mirror" in spec.tags and not policy.get("allow_mirror", False):
        return False

    return side in policy.get("visible_sides", {"L", "R", "B", "NA"}) or side == "NA"


def spec_allowed_for_context(spec: MetricSpec, ctx: MetricContext) -> bool:
    return spec_allowed_for_bucket(spec, ctx.pose_bucket, yaw_deg=ctx.yaw_deg)


def filter_metric_values_for_bucket(
    values,
    bucket: str,
    *,
    yaw_deg: float | None = None,
):
    """Keep only metrics valid for the photo pose bucket."""
    return [v for v in values if spec_allowed_for_bucket(v.spec, bucket, yaw_deg=yaw_deg)]


def filter_metric_values_for_context(values, ctx: MetricContext) -> list[MetricValue]:
    return [v for v in values if spec_allowed_for_context(v.spec, ctx)]


def metric_value_writable(mv: MetricValue, ctx: MetricContext) -> bool:
    if not spec_production_ready(mv.spec):
        return False
    if not spec_allowed_for_context(mv.spec, ctx):
        return False
    if mv.quality_gate == "blocked":
        return False
    if float(mv.confidence) < MIN_PRODUCTION_CONFIDENCE:
        return False
    if mv.value is None:
        return False
    return True


def apply_runtime_confidence_gates(ctx: MetricContext, values: list[MetricValue]) -> list[MetricValue]:
    """Adjust confidence/quality_gate from pose and expression without changing csv schema."""
    exp_norm = 0.0
    if ctx.exp_params is not None:
        try:
            import numpy as np

            exp_norm = float(np.linalg.norm(ctx.exp_params))
        except Exception:
            exp_norm = 0.0

    out: list[MetricValue] = []
    for mv in values:
        conf = float(mv.confidence)
        qg = mv.quality_gate

        if mv.spec.expression_sensitive and exp_norm > 1.5:
            conf = min(conf, 0.25)
            qg = "degraded"

        policy = POSE_POLICY.get(ctx.pose_bucket, POSE_POLICY["unclassified"])
        if policy.get("allow_bilateral") == "conditional" and mv.spec.side == "B":
            yaw = abs(float(ctx.yaw_deg))
            if yaw > 15.0:
                conf = min(conf, max(0.35, 1.0 - (yaw - 15.0) / 20.0))
                if yaw >= POSE_YAW_BILATERAL_OFF_DEG:
                    qg = "blocked"

        if mv.spec.name in CONDITIONAL_BILATERAL_PAIR_BLOCK | CONDITIONAL_ASYMMETRY_BLOCK:
            if policy.get("allow_bilateral") == "conditional" or policy.get("profile_mode"):
                conf = min(conf, 0.3)
                qg = "degraded"

        out.append(
            MetricValue(
                spec=mv.spec,
                value=mv.value,
                confidence=conf,
                visibility=mv.visibility,
                quality_gate=qg,
                notes=mv.notes,
                source_space=mv.source_space,
            )
        )
    return out


_BUCKET_CORE_CACHE: dict[str, frozenset[str]] = {}
_BUCKET_EXP_CACHE: dict[str, frozenset[str]] = {}
_BUCKET_DISABLED_CACHE: dict[str, frozenset[str]] = {}


def _load_identity_scoring_config() -> dict:
    repo = Path(__file__).resolve().parents[1]
    path = repo / "metrics" / "identity_scoring_config.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _get_bucket_core_keep(bucket: str) -> frozenset[str]:
    if bucket not in _BUCKET_CORE_CACHE:
        cfg = _load_identity_scoring_config()
        key = f"{bucket}_core_keep"
        _BUCKET_CORE_CACHE[bucket] = frozenset(cfg.get(key, []))
    return _BUCKET_CORE_CACHE[bucket]


def _get_bucket_experimental(bucket: str) -> frozenset[str]:
    if bucket not in _BUCKET_EXP_CACHE:
        cfg = _load_identity_scoring_config()
        key = f"{bucket}_experimental"
        _BUCKET_EXP_CACHE[bucket] = frozenset(cfg.get(key, []))
    return _BUCKET_EXP_CACHE[bucket]


def _get_bucket_disabled(bucket: str) -> frozenset[str]:
    if bucket not in _BUCKET_DISABLED_CACHE:
        cfg = _load_identity_scoring_config()
        key = f"{bucket}_disabled"
        _BUCKET_DISABLED_CACHE[bucket] = frozenset(cfg.get(key, []))
    return _BUCKET_DISABLED_CACHE[bucket]


def filter_bucket_production_metrics(values: Iterable[MetricValue], bucket: str) -> list[MetricValue]:
    core_keep = _get_bucket_core_keep(bucket)
    disabled = _get_bucket_disabled(bucket)

    result: list[MetricValue] = []
    for v in values:
        if disabled and v.spec.name in disabled:
            continue
        if core_keep and v.spec.name not in core_keep:
            continue
        result.append(v)
    return result


def filter_bucket_experimental_metrics(values: Iterable[MetricValue], bucket: str) -> list[MetricValue]:
    exp = _get_bucket_experimental(bucket)
    if not exp:
        return []
    return [v for v in values if v.spec.name in exp]


def is_bucket_core_metric(name: str, bucket: str) -> bool:
    return name in _get_bucket_core_keep(bucket)


def is_bucket_experimental_metric(name: str, bucket: str) -> bool:
    return name in _get_bucket_experimental(bucket)


def is_bucket_disabled_metric(name: str, bucket: str) -> bool:
    return name in _get_bucket_disabled(bucket)


# Legacy aliases for frontal (backward compat during transition)
filter_frontal_production_metrics = lambda values: filter_bucket_production_metrics(values, "frontal")
filter_frontal_experimental_metrics = lambda values: filter_bucket_experimental_metrics(values, "frontal")
is_frontal_core_metric = lambda name: is_bucket_core_metric(name, "frontal")
is_frontal_experimental_metric = lambda name: is_bucket_experimental_metric(name, "frontal")
is_frontal_disabled_metric = lambda name: is_bucket_disabled_metric(name, "frontal")
