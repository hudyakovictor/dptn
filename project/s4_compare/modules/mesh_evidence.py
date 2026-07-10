from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass
class MeshComparison:
    geometry_distance: float
    texture_distance: float
    age_explained_distance: float
    pose_gap_deg: float
    same_suspicion: float
    different_suspicion: float
    synthetic_suspicion: float
    confidence: float
    notes: list[str]


class MeshEvidence:
    """Geometry + texture evidence with age-aware compensation."""

    def compare(self, a, b, reference: dict | None = None) -> dict[str, Any]:
        a_geom = self._metrics(a, "geometry")
        b_geom = self._metrics(b, "geometry")
        a_tex = self._metrics(a, "texture")
        b_tex = self._metrics(b, "texture")
        a_quality = self._quality(a)
        b_quality = self._quality(b)
        bucket = self._bucket(a, b)
        age_gap_years = self._age_gap_years(a, b)
        pose_gap_deg = self._pose_gap_deg(a, b)

        geo, geo_overlap, geo_discount = self._distance(a_geom, b_geom, reference, bucket, channel="geometry")
        tex, tex_overlap, tex_discount = self._distance(a_tex, b_tex, reference, bucket, channel="texture")
        age_explained = self._age_explained(reference, bucket, age_gap_years, a_geom, a_tex)
        geo = max(0.0, geo - min(geo * 0.55, age_explained))
        tex = max(0.0, tex - min(tex * 0.45, age_explained * 0.75))
        quality_penalty = float(np.clip(1.0 - ((a_quality + b_quality) / 2.0), 0.0, 1.0))
        channel_gap = abs(geo - tex)

        same = float(
            np.clip(
                1.0
                - (
                    geo * 0.56
                    + tex * 0.30
                    + pose_gap_deg / 130.0
                    + channel_gap * 0.10
                    + quality_penalty * 0.12
                )
                / 3.0,
                0.0,
                1.0,
            )
        )
        diff = float(
            np.clip(
                (
                    geo * 0.82
                    + tex * 0.22
                    + max(0.0, age_gap_years - age_explained) * 0.10
                    + channel_gap * 0.15
                )
                / 2.7,
                0.0,
                1.0,
            )
        )
        syn = float(
            np.clip(
                (
                    tex * 0.78
                    + max(0.0, channel_gap - 0.35) * 0.22
                    + quality_penalty * 0.18
                )
                / 2.8,
                0.0,
                1.0,
            )
        )
        confidence = float(np.clip(max(same, diff, syn), 0.0, 1.0))
        notes = []
        if age_explained > 0:
            notes.append("age_explained")
        if pose_gap_deg > 25.0:
            notes.append("pose_gap_high")
        if self._ref_get(reference, "pairwise_noise"):
            notes.append("noise_compensated")
        if quality_penalty > 0.35:
            notes.append("quality_penalty")
        if channel_gap > 0.45:
            notes.append("cross_modal_gap")

        comparison = MeshComparison(
            geometry_distance=float(geo),
            texture_distance=float(tex),
            age_explained_distance=float(age_explained),
            pose_gap_deg=float(pose_gap_deg),
            same_suspicion=same,
            different_suspicion=diff,
            synthetic_suspicion=syn,
            confidence=confidence,
            notes=notes,
        )

        return {
            **asdict(comparison),
            "bucket": bucket,
            "age_gap_years": float(age_gap_years),
            "metric_overlap": int(max(geo_overlap, tex_overlap)),
            "noise_discount": float(max(geo_discount, tex_discount)),
            "quality_penalty": quality_penalty,
        }

    def _metrics(self, record: Any, kind: str) -> dict[str, float]:
        if record is None:
            return {}
        if hasattr(record, kind):
            value = getattr(record, kind)
            if isinstance(value, dict):
                return self._numeric_map(value)
        if isinstance(record, dict):
            direct = record.get(kind)
            if isinstance(direct, dict):
                return self._numeric_map(direct)
            if kind == "geometry":
                return self._pick_map(record, ("geometry", "geom", "mesh", "metrics"))
            return self._pick_map(record, ("texture", "tex", "metrics"))
        return {}

    def _pick_map(self, payload: dict[str, Any], keys: tuple[str, ...]) -> dict[str, float]:
        for key in keys:
            value = payload.get(key)
            if isinstance(value, dict):
                return self._numeric_map(value)
        return {
            str(k): float(v)
            for k, v in payload.items()
            if isinstance(k, str)
            and isinstance(v, (int, float))
            and (k.startswith("geometry_") or k.startswith("texture_") or k.startswith("mesh_") or k.endswith("_span") or k.endswith("_ratio"))
        }

    def _numeric_map(self, payload: dict[str, Any]) -> dict[str, float]:
        return {str(k): float(v) for k, v in payload.items() if isinstance(v, (int, float))}

    def _distance(
        self,
        a: dict[str, float],
        b: dict[str, float],
        reference: dict | None,
        bucket: str,
        *,
        channel: str,
    ) -> tuple[float, int, float]:
        keys = sorted(set(a) & set(b))
        if not keys:
            return 0.0, 0, 0.0
        selected_keys = self._ref_get(reference, "selected_metric_keys", [])
        if selected_keys:
            selected = set(selected_keys)
            filtered = [key for key in keys if key in selected]
            if len(filtered) >= 3:
                keys = filtered
        ref_stats = self._ref_get(reference, "global_stats", {}) or {}
        noise_bucket = (self._ref_get(reference, "pairwise_noise", {}) or {}).get(bucket, {})
        vals = []
        discounts = []
        for key in keys:
            va = float(a[key])
            vb = float(b[key])
            ref = ref_stats.get(key, {})
            scale = max(ref.get("mad", 0.0) or ref.get("std", 0.0) or 1.0, 1e-6)
            diff = abs(va - vb) / scale
            if key.startswith("texture_"):
                diff *= 0.85
            elif key.startswith("mesh_") or key.endswith("_span") or key.startswith("face_"):
                diff *= 1.12
            noise = float(noise_bucket.get(key, {}).get("mad", 0.0) or noise_bucket.get(key, {}).get("std", 0.0) or 0.0)
            if noise > 0:
                discount = min(noise / max(scale, 1e-6), 0.85)
                discounts.append(discount)
                diff = max(0.0, diff - discount)
            vals.append(diff)
        base = float(np.median(vals))
        if channel == "geometry":
            base *= 1.05
        else:
            base *= 0.95
        return max(0.0, base), len(keys), float(np.mean(discounts) if discounts else 0.0)

    def _age_explained(self, reference: dict | None, bucket: str, age_gap_years: float, a_geom: dict[str, float], a_tex: dict[str, float]) -> float:
        if not reference or age_gap_years <= 0:
            return 0.0
        profiles = (self._ref_get(reference, "age_profiles", {}) or {}).get(bucket, {})
        shifts = []
        for key in list(a_geom.keys()) + list(a_tex.keys()):
            profile = profiles.get(key)
            if not profile:
                continue
            slope = abs(float(profile.get("slope", 0.0)))
            corr = abs(float(profile.get("corr", 0.0)))
            shifts.append(slope * age_gap_years * (1.0 + min(corr, 1.0)))
        return float(np.median(shifts)) if shifts else 0.0

    def _bucket(self, a: Any, b: Any) -> str:
        if hasattr(a, "bucket"):
            bucket = getattr(a, "bucket")
            if bucket:
                return str(getattr(bucket, "value", bucket))
        if isinstance(a, dict):
            bucket = a.get("bucket")
            if bucket:
                return str(bucket)
        if hasattr(b, "bucket"):
            bucket = getattr(b, "bucket")
            if bucket:
                return str(getattr(bucket, "value", bucket))
        if isinstance(b, dict):
            bucket = b.get("bucket")
            if bucket:
                return str(bucket)
        return "unknown"

    def _age_gap_years(self, a: Any, b: Any) -> float:
        da = self._age_years(a)
        db = self._age_years(b)
        if da is None or db is None:
            return 0.0
        return abs(da - db)

    def _age_years(self, record: Any) -> float | None:
        if hasattr(record, "age_years"):
            value = getattr(record, "age_years")
            if isinstance(value, (int, float)):
                return float(value)
        if not isinstance(record, dict):
            return None
        if "age_years" in record and isinstance(record["age_years"], (int, float)):
            return float(record["age_years"])
        pose = record.get("pose")
        if isinstance(pose, dict) and isinstance(pose.get("age_years"), (int, float)):
            return float(pose["age_years"])
        return None

    def _pose_gap_deg(self, a: Any, b: Any) -> float:
        pa = self._pose(a)
        pb = self._pose(b)
        if pa is None or pb is None:
            return 0.0
        dy = abs(pa[0] - pb[0])
        dp = abs(pa[1] - pb[1])
        dr = abs(pa[2] - pb[2])
        return float(np.sqrt((1.4 * dy) ** 2 + dp ** 2 + (0.6 * dr) ** 2))

    def _pose(self, record: Any) -> tuple[float, float, float] | None:
        if hasattr(record, "pose"):
            pose = getattr(record, "pose")
            if pose is not None:
                return (
                    float(getattr(pose, "yaw", 0.0) or 0.0),
                    float(getattr(pose, "pitch", 0.0) or 0.0),
                    float(getattr(pose, "roll", 0.0) or 0.0),
                )
        if not isinstance(record, dict):
            return None
        pose = record.get("pose")
        if isinstance(pose, dict):
            return (
                float(pose.get("yaw", 0.0) or 0.0),
                float(pose.get("pitch", 0.0) or 0.0),
                float(pose.get("roll", 0.0) or 0.0),
            )
        return None

    def _quality(self, record: Any) -> float:
        if hasattr(record, "quality"):
            quality = getattr(record, "quality")
            if quality is not None and hasattr(quality, "overall_quality"):
                return float(getattr(quality, "overall_quality", 0.0) or 0.0)
        if isinstance(record, dict):
            quality = record.get("quality")
            if isinstance(quality, dict):
                value = quality.get("overall_quality", 0.0)
                if isinstance(value, (int, float)):
                    return float(value)
        return 0.0

    def _ref_get(self, reference: Any, key: str, default: Any = None) -> Any:
        if reference is None:
            return default
        if isinstance(reference, dict):
            return reference.get(key, default)
        if hasattr(reference, key):
            return getattr(reference, key)
        return default
