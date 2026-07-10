from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

from ..shared.logging import setup_logger
from ..shared.schemas import CalibrationReference, PipelineDataset, Stage1Record, Stage2Record, Stage3Record
from ..shared.utils import ensure_dir, load_json, save_json, subject_age_years_at
from .calibration_builder import PoseAwareCalibrationBuilder
from .noise_discount import CalibratedNoiseDiscount
from .health_monitor import CalibrationHealthMonitor

logger = setup_logger("deeputin.s3")


class CalibrationEngine:
    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}

    def build_reference(self, calibration_root: str | Path) -> CalibrationReference | None:
        root = Path(calibration_root)
        records = self._load_stage2_records(root)
        if not records:
            logger.warning("Нет calibration stage2 записей в %s", calibration_root)
            return None

        stage1_records = self._load_stage1_records(root)
        
        # Convert Stage2Record to dict for new calibration builder
        calibration_data = []
        for record in records:
            stage1 = stage1_records.get(record.photo_id)
            calibration_data.append({
                "bucket": record.bucket.value,
                "pose": {"yaw": record.pose.yaw if hasattr(record, 'pose') else 0, 
                         "pitch": record.pose.pitch if hasattr(record, 'pose') else 0, 
                         "roll": record.pose.roll if hasattr(record, 'pose') else 0},
                "quality": record.quality.overall_quality if hasattr(record.quality, 'overall_quality') else 0.5,
                "age_years": stage1.age_years if stage1 else None,
                "geometry": dict(record.geometry),
                "texture": dict(record.texture),
            })

        # Build new pose-aware calibration
        builder = PoseAwareCalibrationBuilder(min_pairs=self.config.get("min_calibration_pairs", 10))
        pose_models = builder.build(calibration_data)

        # Health monitoring
        monitor = CalibrationHealthMonitor()
        health_results = monitor.check(pose_models, calibration_data)
        health_summary = monitor.summary(health_results)
        logger.info(f"Calibration health: {health_summary}")

        # Build legacy reference format for backward compatibility
        bucket_stats: dict[str, dict[str, dict[str, float]]] = {}
        pairwise_noise: dict[str, dict[str, dict[str, float]]] = {}
        age_profiles: dict[str, dict[str, dict[str, float]]] = {}
        global_stats: dict[str, dict[str, float]] = {}
        selected_metric_keys: list[str] = []

        grouped: dict[str, list[Stage2Record]] = defaultdict(list)
        for record in records:
            grouped[record.bucket.value].append(record)

        for bucket, bucket_records in grouped.items():
            bucket_stats[bucket] = {}
            merged = self._merge_metric_maps(bucket_records)
            for metric_name, values in merged.items():
                bucket_stats[bucket][metric_name] = self._describe(values)
            pairwise_noise[bucket] = self._build_pairwise_noise(bucket_records, stage1_records)
            age_profiles[bucket] = self._build_age_profiles(bucket_records, stage1_records)

        merged_all = self._merge_metric_maps(records)
        for metric_name, values in merged_all.items():
            stats = self._describe(values)
            global_stats[metric_name] = stats
            if stats["count"] >= max(3, len(records) // 5):
                selected_metric_keys.append(metric_name)

        thresholds = self._build_thresholds(global_stats)
        reference = CalibrationReference(
            generated_at=datetime.utcnow().isoformat(timespec="seconds"),
            photo_count=len(records),
            bucket_stats=bucket_stats,
            pairwise_noise=pairwise_noise,
            age_profiles=age_profiles,
            global_stats=global_stats,
            selected_metric_keys=sorted(selected_metric_keys),
            thresholds=thresholds,
            notes=[
                "Stage 3 строит baseline только по калибровочным фото, где человек считается оригинальным.",
                "Все геометрические и текстурные шумы интерпретируются как вариативность съёмки и реконструкции.",
                f"Pose-aware calibration: {health_summary['healthy']} healthy, {health_summary['degraded']} degraded, {health_summary['insufficient']} insufficient buckets.",
            ],
        )
        save_json(reference.model_dump(), Path(calibration_root) / "calibration_reference.json")
        
        # Also save new calibration models
        self._save_pose_models(pose_models, Path(calibration_root))
        
        return reference

    def _save_pose_models(self, models: Dict, path: Path) -> None:
        """Save pose-aware calibration models."""
        import json
        data = {}
        for bucket, model in models.items():
            data[bucket] = {
                "bucket": model.bucket,
                "intercept": model.intercept,
                "slope": model.slope,
                "curvature": model.curvature,
                "p05": model.p05,
                "p95": model.p95,
                "mad": model.mad,
                "sample_count": model.sample_count,
                "quality_breakdown": model.quality_breakdown,
            }
        with open(path / "pose_calibration_models.json", "w") as f:
            json.dump(data, f, indent=2)

    def save_reference(self, reference: CalibrationReference, path: str | Path) -> Path:
        return save_json(reference.model_dump(), path)

    def annotate_main_dataset(self, main_root: str | Path, reference: CalibrationReference) -> list[Stage3Record]:
        root = Path(main_root)
        records = self._load_stage2_records(root)
        stage1_records = self._load_stage1_records(root)
        annotated: list[Stage3Record] = []
        for record in records:
            identity_distance = self._distance_to_reference(record, reference, stage1_records.get(record.photo_id))
            texture_suspicion = self._texture_suspicion(record, reference, stage1_records.get(record.photo_id))
            stage2_identity_hint = getattr(record, "geometry_identity_hint", "UNCERTAIN")
            stage2_skin_hint = getattr(record, "texture_skin_hint", "unknown")
            identity_hint = (
                stage2_identity_hint
                if stage2_identity_hint in {"PUT", "UDMURT", "VAS", "OTHER"}
                else ("PUT" if identity_distance < 1.0 else ("UNCERTAIN" if identity_distance < 1.8 else "OTHER"))
            )
            skin_hint = stage2_skin_hint if stage2_skin_hint in {"real", "silicone"} else ("real" if texture_suspicion < reference.thresholds.get("texture_suspicion", 0.65) else "silicone")
            stage3 = Stage3Record(
                photo_id=record.photo_id,
                dataset=record.dataset,
                identity_hint=identity_hint,
                identity_confidence=float(np.clip(1.0 - identity_distance / 3.0, 0.1, 0.99)),
                skin_hint=skin_hint,
                skin_confidence=float(np.clip(abs(reference.thresholds.get("texture_suspicion", 0.65) - texture_suspicion) + 0.35, 0.1, 0.99)),
                geometry_distance=float(identity_distance),
                texture_suspicion=float(texture_suspicion),
                notes=[
                    f"bucket={record.bucket.value}",
                    "identity_hint основан на расстоянии до calibration baseline",
                    f"stage2_identity_hint={stage2_identity_hint}",
                    f"stage2_skin_hint={stage2_skin_hint}",
                ],
            )
            photo_dir = root / record.photo_id
            save_json(stage3.model_dump(), photo_dir / "identity.json")
            annotated.append(stage3)
        return annotated

    def _load_stage2_records(self, root: Path) -> list[Stage2Record]:
        records: list[Stage2Record] = []
        for photo_dir in sorted(root.iterdir()):
            if not photo_dir.is_dir():
                continue
            info = load_json(photo_dir / "info.json")
            geo = load_json(photo_dir / "geometry_metrics.json")
            tex = load_json(photo_dir / "texture_metrics.json")
            if info and isinstance(geo, dict):
                records.append(Stage2Record(
                    photo_id=info.get("photo_id", photo_dir.name),
                    dataset=info.get("dataset", "main"),
                    bucket=info.get("pose", {}).get("bucket", "unknown"),
                    quality=info.get("quality", {}),
                    geometry=geo,
                    texture=tex or {},
                ))
        return records

    def _merge_metric_maps(self, records: list[Stage2Record]) -> dict[str, list[float]]:
        merged: dict[str, list[float]] = defaultdict(list)
        for record in records:
            for key, value in {**record.geometry, **record.texture}.items():
                if isinstance(value, (int, float)) and np.isfinite(float(value)):
                    merged[key].append(float(value))
        return merged

    def _describe(self, values: list[float]) -> dict[str, float]:
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return {"count": 0, "mean": 0.0, "std": 1.0, "median": 0.0, "mad": 1.0}
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med))) or 1e-6
        return {
            "count": float(arr.size),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr) or 1e-6),
            "median": med,
            "mad": mad,
        }

    def _build_thresholds(self, global_stats: dict[str, dict[str, float]]) -> dict[str, float]:
        texture_keys = [k for k in global_stats if k.startswith("texture_")]
        geom_keys = [k for k in global_stats if k.startswith("face_") or k.startswith("mesh_") or k.endswith("_span")]
        texture_scores = [global_stats[k]["mean"] + global_stats[k]["std"] for k in texture_keys if global_stats[k]["std"] > 0]
        geom_scores = [global_stats[k]["mean"] + global_stats[k]["std"] for k in geom_keys if global_stats[k]["std"] > 0]
        texture_suspicion = float(np.clip(np.mean(texture_scores) if texture_scores else 0.65, 0.3, 0.9))
        geometry_cutoff = float(np.clip(np.mean(geom_scores) if geom_scores else 1.0, 0.5, 2.5))
        return {
            "texture_suspicion": texture_suspicion,
            "geometry_distance": geometry_cutoff,
        }

    def _distance_to_reference(self, record: Stage2Record, reference: CalibrationReference, stage1: Stage1Record | None = None) -> float:
        stats = reference.global_stats
        noise = reference.pairwise_noise.get(record.bucket.value, {})
        age_profiles = reference.age_profiles.get(record.bucket.value, {})
        age_years = float(stage1.age_years) if stage1 and stage1.age_years is not None else None
        diffs: list[float] = []
        for key, value in {**record.geometry, **record.texture}.items():
            ref = stats.get(key)
            if not ref:
                continue
            scale = max(ref.get("std", 1.0) or 1e-6, ref.get("mad", 1.0) or 1e-6)
            expected_noise = float(noise.get(key, {}).get("mad", 0.0) or noise.get(key, {}).get("mean", 0.0) or 0.0)
            expected = ref.get("median", ref.get("mean", 0.0))
            profile = age_profiles.get(key, {})
            if age_years is not None and profile:
                expected = float(profile.get("intercept", expected)) + float(profile.get("slope", 0.0)) * age_years
            delta = max(0.0, abs(float(value) - expected) - expected_noise)
            diffs.append(delta / scale)
        return float(np.mean(diffs) if diffs else 0.0)

    def _texture_suspicion(self, record: Stage2Record, reference: CalibrationReference, stage1: Stage1Record | None = None) -> float:
        texture_values = []
        noise = reference.pairwise_noise.get(record.bucket.value, {})
        age_profiles = reference.age_profiles.get(record.bucket.value, {})
        age_years = float(stage1.age_years) if stage1 and stage1.age_years is not None else None
        for key, value in record.texture.items():
            if not key.startswith("texture_"):
                continue
            ref = reference.global_stats.get(key)
            if not ref:
                continue
            scale = max(ref.get("std", 1.0) or 1e-6, ref.get("mad", 1.0) or 1e-6)
            expected_noise = float(noise.get(key, {}).get("mad", 0.0) or noise.get(key, {}).get("mean", 0.0) or 0.0)
            expected = ref.get("median", ref.get("mean", 0.0))
            profile = age_profiles.get(key, {})
            if age_years is not None and profile:
                expected = float(profile.get("intercept", expected)) + float(profile.get("slope", 0.0)) * age_years
            delta = max(0.0, abs(float(value) - expected) - expected_noise)
            texture_values.append(delta / scale)
        if not texture_values:
            return 0.0
        return float(np.clip(np.mean(texture_values) / 3.0, 0.0, 1.0))

    def _load_stage1_records(self, root: Path) -> dict[str, Stage1Record]:
        records: dict[str, Stage1Record] = {}
        for path in sorted(root.glob("*/info.json")):
            payload = load_json(path)
            if payload:
                record = Stage1Record.model_validate(payload)
                records[record.photo_id] = record
        return records

    def _build_pairwise_noise(self, records: list[Stage2Record], stage1_records: dict[str, Stage1Record]) -> dict[str, dict[str, float]]:
        if len(records) < 2:
            return {}
        ordered = sorted(
            records,
            key=lambda rec: (
                stage1_records.get(rec.photo_id).date.isoformat() if stage1_records.get(rec.photo_id) and stage1_records.get(rec.photo_id).date else "9999-99-99",
                rec.photo_id,
            ),
        )
        deltas: dict[str, list[float]] = defaultdict(list)
        for left, right in zip(ordered[:-1], ordered[1:]):
            for key, value in {**left.geometry, **left.texture}.items():
                other = {**right.geometry, **right.texture}.get(key)
                if other is None:
                    continue
                if isinstance(value, (int, float)) and isinstance(other, (int, float)):
                    deltas[key].append(abs(float(value) - float(other)))
        return {key: self._describe(values) for key, values in deltas.items() if values}

    def _build_age_profiles(self, records: list[Stage2Record], stage1_records: dict[str, Stage1Record]) -> dict[str, dict[str, float]]:
        age_points: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for record in records:
            stage1 = stage1_records.get(record.photo_id)
            if stage1 is None or stage1.age_years is None:
                continue
            age = float(stage1.age_years)
            for key, value in {**record.geometry, **record.texture}.items():
                if isinstance(value, (int, float)) and np.isfinite(float(value)):
                    age_points[key].append((age, float(value)))
        profiles: dict[str, dict[str, float]] = {}
        for key, pairs in age_points.items():
            if len(pairs) < 4:
                continue
            ages = np.asarray([p[0] for p in pairs], dtype=float)
            vals = np.asarray([p[1] for p in pairs], dtype=float)
            if np.std(ages) < 1e-6 or np.std(vals) < 1e-6:
                continue
            slope, intercept = np.polyfit(ages, vals, 1)
            corr = float(np.corrcoef(ages, vals)[0, 1])
            if not np.isfinite(slope) or not np.isfinite(intercept) or not np.isfinite(corr):
                continue
            profiles[key] = {
                "slope": float(slope),
                "intercept": float(intercept),
                "corr": float(corr),
                "n": float(len(pairs)),
            }
        return profiles
