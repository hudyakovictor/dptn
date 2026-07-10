from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from ..shared.logging import setup_logger
from ..shared.schemas import CalibrationReference, PairEvidence, PipelineDataset, Stage1Record, Stage2Record
from ..shared.utils import load_json, save_json

logger = setup_logger("deeputin.s4")


class CompareEngine:
    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}

    def build_pairwise_evidence(self, main_root: str | Path, reference_path: str | Path | None = None) -> list[PairEvidence]:
        root = Path(main_root)
        records = self._load_stage2_records(root)
        if not records:
            logger.warning("Нет stage2 записей для сравнения в %s", root)
            return []

        reference = CalibrationReference.model_validate(load_json(reference_path)) if reference_path and Path(reference_path).exists() else None
        stage1_records = self._load_stage1_records(root)
        grouped: dict[str, list[Stage2Record]] = defaultdict(list)
        for record in records:
            grouped[record.bucket.value].append(record)

        evidence: list[PairEvidence] = []
        pair_index: dict[str, list[dict[str, object]]] = defaultdict(list)
        for bucket, bucket_records in grouped.items():
            ordered = sorted(bucket_records, key=lambda rec: self._sort_key(rec, stage1_records))
            window = max(1, int(self.config.get("comparison_window", 2)))
            for idx, current in enumerate(ordered):
                for offset in range(1, min(window, idx) + 1):
                    a = ordered[idx - offset]
                    b = current
                    pair = self._compare_pair(a, b, reference, stage1_records)
                    evidence.append(pair)
                    pair_index[a.photo_id].append(pair.model_dump())
                    pair_index[b.photo_id].append(pair.model_dump())
        save_json([p.model_dump() for p in evidence], root / "pairs.json")
        save_json(pair_index, root / "pair_index.json")
        return evidence

    def _load_stage2_records(self, root: Path) -> list[Stage2Record]:
        records: list[Stage2Record] = []
        for path in sorted(root.glob("*/metrics.json")):
            payload = load_json(path)
            if payload:
                records.append(Stage2Record.model_validate(payload))
        return records

    def _load_stage1_records(self, root: Path) -> dict[str, Stage1Record]:
        records: dict[str, Stage1Record] = {}
        for path in sorted(root.glob("*/info.json")):
            payload = load_json(path)
            if not payload:
                continue
            record = Stage1Record.model_validate(payload)
            records[record.photo_id] = record
        return records

    def _sort_key(self, record: Stage2Record, stage1_records: dict[str, Stage1Record]) -> tuple[str, str]:
        stage1 = stage1_records.get(record.photo_id)
        date_value = stage1.date.isoformat() if stage1 and stage1.date else "9999-99-99"
        return (date_value, record.photo_id)

    def _compare_pair(
        self,
        a: Stage2Record,
        b: Stage2Record,
        reference: CalibrationReference | None,
        stage1_records: dict[str, Stage1Record],
    ) -> PairEvidence:
        stage1_a = stage1_records.get(a.photo_id)
        stage1_b = stage1_records.get(b.photo_id)
        age_gap_years = abs(float(stage1_a.age_years) - float(stage1_b.age_years)) if stage1_a and stage1_b and stage1_a.age_years is not None and stage1_b.age_years is not None else 0.0
        geometry_distance, geometry_noise_discount, geometry_overlap = self._normalized_distance(
            a.geometry, b.geometry, reference, a.bucket.value, channel="geometry", age_gap_years=age_gap_years
        )
        texture_distance, texture_noise_discount, texture_overlap = self._normalized_distance(
            a.texture, b.texture, reference, a.bucket.value, channel="texture", age_gap_years=age_gap_years
        )
        qa = float(a.quality.overall_quality)
        qb = float(b.quality.overall_quality)
        quality_penalty = float(np.clip(1.15 - ((qa + qb) / 2.0), 0.55, 1.35))
        pose_gap_deg = self._pose_gap_deg(stage1_a, stage1_b)
        date_gap_days = abs(self._date_to_ord(stage1_a) - self._date_to_ord(stage1_b))
        chronology_penalty = float(np.clip(1.0 + min(date_gap_days / 180.0, 2.0) * 0.18 + min(pose_gap_deg / 45.0, 1.0) * 0.12, 1.0, 1.65))
        synthetic_suspicion = float(np.clip((texture_distance * 0.85 + texture_noise_discount * 0.2) / 3.0, 0.0, 1.0))
        different_suspicion = float(np.clip((geometry_distance * 0.9 + geometry_noise_discount * 0.15) / 3.0, 0.0, 1.0))
        same_raw = (geometry_distance * 0.62 + texture_distance * 0.28 + pose_gap_deg / 120.0 + quality_penalty * 0.15)
        same_suspicion = float(np.clip(1.0 - same_raw / 3.6, 0.0, 1.0))
        age_explained_distance = float(
            self._age_explained_distance(
                a.geometry,
                b.geometry,
                a.texture,
                b.texture,
                reference,
                a.bucket.value,
                age_gap_years,
            )
        )
        if age_explained_distance > 0:
            geometry_distance = float(max(0.0, geometry_distance - min(geometry_distance * 0.55, age_explained_distance)))
            texture_distance = float(max(0.0, texture_distance - min(texture_distance * 0.45, age_explained_distance * 0.75)))
        anomaly_flags: list[str] = []
        if synthetic_suspicion > 0.7 and geometry_distance < 1.0:
            anomaly_flags.append("geometry_stable_texture_break")
        if date_gap_days < 90 and geometry_distance > 1.5 and pose_gap_deg < 22.5:
            anomaly_flags.append("short_gap_identity_shift")
        if chronology_penalty > 1.15 and same_suspicion < 0.4:
            anomaly_flags.append("chrono_pressure")
        if geometry_distance > 1.3 and texture_distance > 1.0 and abs(geometry_distance - texture_distance) > 0.8:
            anomaly_flags.append("cross_modal_disagreement")
        if pose_gap_deg > 35.0 and date_gap_days <= 90:
            anomaly_flags.append("pose_inconsistent_neighbor")
        if geometry_noise_discount > 0.2 or texture_noise_discount > 0.2:
            anomaly_flags.append("calibration_discounted")
        pair_id = f"{a.photo_id}__{b.photo_id}"
        return PairEvidence(
            pair_id=pair_id,
            photo_a=a.photo_id,
            photo_b=b.photo_id,
            bucket=a.bucket.value,
            date_gap_days=int(date_gap_days),
            age_gap_years=float(age_gap_years),
            pose_gap_deg=float(pose_gap_deg),
            geometry_distance=float(geometry_distance),
            texture_distance=float(texture_distance),
            age_explained_distance=float(age_explained_distance),
            quality_penalty=quality_penalty,
            chronology_penalty=chronology_penalty,
            noise_discount=float(max(geometry_noise_discount, texture_noise_discount)),
            metric_overlap=int(max(geometry_overlap, texture_overlap)),
            synthetic_suspicion=synthetic_suspicion,
            different_suspicion=different_suspicion,
            same_suspicion=same_suspicion,
            anomaly_flags=anomaly_flags,
            notes=[
                f"bucket={a.bucket.value}",
                "pairwise evidence built from consecutive photos in the same bucket",
            ],
        )

    def _normalized_distance(
        self,
        a_metrics: dict[str, float],
        b_metrics: dict[str, float],
        reference: CalibrationReference | None,
        bucket: str,
        *,
        channel: str,
        age_gap_years: float = 0.0,
    ) -> tuple[float, float, int]:
        keys = sorted(set(a_metrics) & set(b_metrics))
        if not keys:
            return 0.0, 0.0, 0
        noise_bucket = reference.pairwise_noise.get(bucket, {}) if reference is not None else {}
        ref_stats = reference.global_stats if reference is not None else {}
        weighted = []
        discounts = []
        for key in keys:
            va = float(a_metrics[key])
            vb = float(b_metrics[key])
            ref = ref_stats.get(key, {})
            scale = max(ref.get("mad", 0.0) or ref.get("std", 0.0) or 1.0, 1e-6)
            base = abs(va - vb) / scale
            if key.startswith("texture_"):
                key_weight = 0.85
            elif key.startswith("mesh_") or key.endswith("_span") or key.startswith("face_"):
                key_weight = 1.15
            else:
                key_weight = 1.0
            weighted.append(base * key_weight)
            noise_entry = noise_bucket.get(key, {})
            noise_level = float(noise_entry.get("mad", 0.0) or noise_entry.get("std", 0.0) or 0.0)
            if noise_level > 0:
                discounts.append(min(noise_level / max(scale, 1e-6), 0.8))
        raw_distance = float(np.median(weighted) if weighted else 0.0)
        noise_discount = float(np.mean(discounts) if discounts else 0.0)
        age_shift = self._expected_age_shift(reference, bucket, channel, age_gap_years, keys)
        if age_shift > 0:
            raw_distance = max(0.0, raw_distance - min(raw_distance * 0.6, age_shift))
        if channel == "geometry":
            raw_distance *= 1.05
        else:
            raw_distance *= 0.95
        return float(max(0.0, raw_distance - min(raw_distance * 0.7, noise_discount))), noise_discount, len(keys)

    def _weighted_distance(self, a: dict[str, float], b: dict[str, float]) -> float:
        keys = sorted(set(a) & set(b))
        if not keys:
            return 0.0
        vals = []
        for key in keys:
            va = float(a[key])
            vb = float(b[key])
            scale = max(abs(va), abs(vb), 1.0)
            vals.append(abs(va - vb) / scale)
        return float(np.mean(vals) * 3.0)

    def _date_to_ord(self, record: Stage1Record | None) -> int:
        if record is None or record.date is None:
            return 0
        try:
            return record.date.toordinal()
        except Exception:
            return 0

    def _pose_gap_deg(self, a: Stage1Record | None, b: Stage1Record | None) -> float:
        if a is None or b is None:
            return 0.0
        pose_a = a.pose
        pose_b = b.pose
        dy = abs(float(pose_a.yaw) - float(pose_b.yaw))
        dp = abs(float(pose_a.pitch) - float(pose_b.pitch))
        dr = abs(float(pose_a.roll) - float(pose_b.roll))
        return float(np.sqrt((1.4 * dy) ** 2 + dp ** 2 + (0.6 * dr) ** 2))

    def _expected_age_shift(
        self,
        reference: CalibrationReference | None,
        bucket: str,
        channel: str,
        age_gap_years: float,
        keys: list[str],
    ) -> float:
        if reference is None or age_gap_years <= 0:
            return 0.0
        age_profiles = reference.age_profiles.get(bucket, {})
        shifts = []
        for key in keys:
            if channel == "texture" and not key.startswith("texture_"):
                continue
            profile = age_profiles.get(key)
            if not profile:
                continue
            slope = float(profile.get("slope", 0.0))
            corr = abs(float(profile.get("corr", 0.0)))
            weight = 1.0 + min(corr, 1.0)
            shifts.append(abs(slope) * age_gap_years * weight)
        if not shifts:
            return 0.0
        return float(np.median(shifts))

    def _age_explained_distance(
        self,
        a_geometry: dict[str, float],
        b_geometry: dict[str, float],
        a_texture: dict[str, float],
        b_texture: dict[str, float],
        reference: CalibrationReference | None,
        bucket: str,
        age_gap_years: float,
    ) -> float:
        if reference is None or age_gap_years <= 0:
            return 0.0
        geom_shift = self._expected_age_shift(reference, bucket, "geometry", age_gap_years, list(a_geometry.keys()))
        tex_shift = self._expected_age_shift(reference, bucket, "texture", age_gap_years, list(a_texture.keys()))
        return float(max(geom_shift, tex_shift))
