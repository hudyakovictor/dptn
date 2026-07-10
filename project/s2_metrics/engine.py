from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import os

from ..shared.logging import setup_logger
from ..shared.schemas import PipelineDataset, Stage1Record, Stage2Record
from ..shared.utils import ensure_dir, load_json, load_pickle, load_rgba_png, save_json
from .modules import GeometryIdentityResolver, TextureSkinClassifier, load_geometry_metric_catalog, load_texture_metric_catalog
from .modules.geometry_extractor import GeometryExtractor
from .modules.texture.extractor_v2 import TextureExtractorV2
from .texture_anomaly import CohortTextureAnomalyDetectorV2
from .physical_features import PhysicalTextureExtractor


logger = setup_logger("deeputin.s2")


class MetricsEngine:
    def __init__(self, output_dir: str | Path, dataset: PipelineDataset, config: dict | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.dataset = dataset
        self.config = config or {}
        
        # Use env var or relative paths - NO HARDCODED ABSOLUTE PATHS
        data_root = Path(os.environ.get("DPTN_DATA_ROOT", Path(__file__).resolve().parents[2] / "data"))
        
        geometry_table = self.config.get(
            "geometry_evidence_table",
            data_root / "imgtest" / "metrics_test" / "METRIC_EVIDENCE_TABLE.csv",
        )
        texture_leaderboard = self.config.get(
            "texture_leaderboard",
            data_root / "imgtest" / "unified_test" / "clean_feature_leaderboard.csv",
        )
        
        # Fallback to relative paths if configured paths don't exist
        if not geometry_table.exists():
            geometry_table = Path(__file__).resolve().parents[2] / "data" / "imgtest" / "metrics_test" / "METRIC_EVIDENCE_TABLE.csv"
        if not texture_leaderboard.exists():
            texture_leaderboard = Path(__file__).resolve().parents[2] / "data" / "imgtest" / "unified_test" / "clean_feature_leaderboard.csv"
        
        self.geometry_resolver = GeometryIdentityResolver(geometry_table)
        self.texture_classifier = TextureSkinClassifier(texture_leaderboard)
        self.geometry_catalog = load_geometry_metric_catalog()
        self.texture_catalog = load_texture_metric_catalog(texture_leaderboard)
        self.texture_extractor = TextureExtractorV2()
        self.geometry_extractor = GeometryExtractor()
        self.cohort_detector = CohortTextureAnomalyDetectorV2()
        self.physical_extractor = PhysicalTextureExtractor()

    def run(self) -> list[Stage2Record]:
        stage1_records = self._load_stage1_records()
        if not stage1_records:
            logger.warning("Нет stage1 записей для этапа 2 в %s", self.output_dir)
            return []

        raw_records: list[tuple[Stage1Record, Stage2Record]] = []
        cohort_groups: dict[str, list[dict]] = {}
        
        for index, record in enumerate(stage1_records, start=1):
            try:
                metrics = self._process_one(record)
                raw_records.append((record, metrics))
                logger.info("[s2] %s/%s %s", index, len(stage1_records), record.photo_id)
                
                year = record.date.year if record.date else 2000
                quality = float(record.quality.overall_quality) if record.quality else 0.5
                cohort_key = self.cohort_detector.get_cohort_key(year, quality)
                if cohort_key not in cohort_groups:
                    cohort_groups[cohort_key] = []
                cohort_groups[cohort_key].append(metrics.texture)
            except Exception as exc:
                logger.exception("[s2] Ошибка на %s: %s", record.photo_id, exc)

        for cohort_key, cohort_textures in cohort_groups.items():
            if len(cohort_textures) >= 3:
                try:
                    self.cohort_detector.fit_cohort(cohort_textures, cohort_key)
                    logger.info("[s2] Cohort '%s': fitted on %d samples", cohort_key, len(cohort_textures))
                except Exception as exc:
                    logger.warning("[s2] Cohort '%s' fit failed: %s", cohort_key, exc)

        records: list[Stage2Record] = []
        for stage1_record, stage2_record in raw_records:
            year = stage1_record.date.year if stage1_record.date else 2000
            quality = float(stage1_record.quality.overall_quality) if stage1_record else 0.5
            cohort_key = self.cohort_detector.get_cohort_key(year, quality)
            try:
                anomaly_result = self.cohort_detector.score(stage2_record.texture, cohort_key, quality)
                stage2_record.metric_notes["texture_anomaly_score"] = str(anomaly_result.anomaly_score)
                stage2_record.metric_notes["texture_anomaly_interpretation"] = anomaly_result.interpretation
                stage2_record.metric_notes["texture_anomaly_max_z"] = str(anomaly_result.max_z)
                if anomaly_result.feature_flags:
                    stage2_record.metric_notes["texture_anomaly_flags"] = ",".join(anomaly_result.feature_flags.keys())
            except Exception:
                stage2_record.metric_notes["texture_anomaly_score"] = "0.0"
                stage2_record.metric_notes["texture_anomaly_interpretation"] = "computation_error"
            records.append(stage2_record)

        save_json([r.model_dump() for r in records], self.output_dir / "stage2_manifest.json")
        return records

    def _load_stage1_records(self) -> list[Stage1Record]:
        records: list[Stage1Record] = []
        for info_path in sorted(self.output_dir.glob("*/info.json")):
            payload = load_json(info_path)
            if not payload:
                continue
            records.append(Stage1Record.model_validate(payload))
        return records

    def _process_one(self, record: Stage1Record) -> Stage2Record:
        photo_dir = Path(record.face_mask_path).parent
        info_path = photo_dir / "info.json"
        info = Stage1Record.model_validate(load_json(info_path))
        reconstruction = load_pickle(photo_dir / "reconstruction.pkl")
        rgba = load_rgba_png(photo_dir / "face_mask.png")

        geometry = self.geometry_extractor.extract(reconstruction)

        # Legacy geometry metrics (optional)
        try:
            from .modules.geometry.legacy_metrics.context import build_metric_context
            from .modules.geometry.legacy_metrics.runner import compute_single_photo_metrics

            legacy_ctx = build_metric_context(
                photo_id=record.photo_id,
                image_path=photo_dir / "face_crop.jpg",
                reconstruction=reconstruction,
                adapter=None,
                pose_bucket=info.pose.bucket.value,
                quality=info.quality.model_dump() if info.quality else {},
                geometry_metrics=geometry,
            )
            legacy_values, legacy_errors = compute_single_photo_metrics(legacy_ctx)
            for mv in legacy_values:
                if mv.value is not None and isinstance(mv.value, (int, float)):
                    geometry[mv.spec.name] = float(mv.value)
        except Exception as exc:
            logger.warning("Legacy metrics computation failed: %s", exc)

        # REMOVED: Old CSV bucket filtering that was dropping bone metrics
        # allowed_geo = load_old_csv_bucket_metrics(bucket_name)
        # if allowed_geo: geometry = {k: v for k, v in geometry.items() if k in allowed_geo}

        # Texture context with NATIVE RGB + native mask + ppIOD + face_min_dim
        # The TextureExtractorV2 reads face_mask_path and extracts native crops internally
        class TextureCtx:
            image_rgb = rgba[:, :, :3]  # Preview RGB (424x500) - fallback only
            face_bbox = info.face_bbox
            face_mask_path = photo_dir / "face_mask.png"  # Native mask path for V2 extractor
            # ppIOD and face_min_dim can be derived from reconstruction if needed
            pp_iod = info.pose.iod if hasattr(info.pose, 'iod') else None
            face_min_dim = min(info.face_bbox[2], info.face_bbox[3]) if info.face_bbox else None

        texture_ctx = TextureCtx()
        texture = self.texture_extractor.extract(texture_ctx, exclude_sensitive=False)

        # Extract assessability fields from texture (strings, not floats)
        texture_assessability = texture.pop("texture_assessability", "eligible")
        q_valid_patches = texture.pop("q_valid_patches", 0)

        geometry_hint = self.geometry_resolver.resolve(geometry)
        texture_hint = self.texture_classifier.classify(texture, info.quality)

        physical_features = {}
        try:
            landmarks_68 = reconstruction.get("landmarks_68")
            if (not landmarks_68 or len(landmarks_68) == 0):
                landmarks_68 = reconstruction.get("landmarks_106")
            if landmarks_68 is not None and len(landmarks_68) > 0 and rgba is not None:
                landmarks = np.array(landmarks_68, dtype=np.float32)
                if landmarks.ndim == 2 and landmarks.shape[1] >= 2:
                    image_rgb = rgba[:, :, :3]
                    # FIX: seg_mask threshold >10 not >128 (preserves skin edges)
                    seg_mask = rgba[:, :, 3] > 10 if rgba.shape[2] == 4 else np.ones(rgba.shape[:2], dtype=bool)
                    overall_q = float(info.quality.overall_quality) if info.quality else 1.0
                    pf = self.physical_extractor.extract(image_rgb, landmarks, seg_mask, overall_q)
                    # Tier 3 physical auxiliary metrics
                    physical_features = {
                        "seam_score": pf.seam_score,
                        "specular_sharpness": pf.specular_sharpness,
                        "specular_dispersion": pf.specular_dispersion,
                        "sss_index": pf.sss_index,
                        "melanin_hemo_slope": pf.melanin_hemo_slope,
                    }
        except Exception:
            pass

        # Merge Tier 3 physical aux into texture
        texture.update(physical_features)

        texture_weights_json = texture.pop("texture_feature_weights_json", None)
        metric_notes = {
            "geometry_space": "3ddfa_v3_canonical",
            "texture_source": "face_mask.png (native)",
            "geometry_identity_hint": geometry_hint.get("identity_hint", "UNCERTAIN"),
            "texture_skin_hint": texture_hint.get("texture_skin_hint", "unknown"),
            "geometry_catalog_size": str(len(self.geometry_catalog)),
            "texture_catalog_size": str(len(self.texture_catalog)),
            "quality_sensitive_excluded": "false",
        }
        if texture_weights_json:
            metric_notes["texture_feature_weights_json"] = texture_weights_json
        for k, v in physical_features.items():
            metric_notes[f"physical_{k}"] = str(v)

        selected_keys = sorted(set(geometry) | set(texture) | set(geometry_hint.get("selected_metric_keys", [])) | set(texture_hint.get("used_metrics", [])))
        stage2 = Stage2Record(
            photo_id=info.photo_id,
            dataset=info.dataset,
            bucket=info.pose.bucket,
            quality=info.quality,
            geometry=geometry,
            texture=texture,
            selected_metric_keys=selected_keys,
            metric_notes=metric_notes,
            geometry_identity_hint=str(geometry_hint.get("identity_hint", "UNCERTAIN")),
            geometry_identity_confidence=float(geometry_hint.get("identity_confidence", 0.0)),
            texture_skin_hint=str(texture_hint.get("texture_skin_hint", "unknown")),
            texture_skin_confidence=float(texture_hint.get("texture_skin_confidence", 0.0)),
            texture_assessability=texture_assessability,
            quality_summary={
                "overall_quality": float(info.quality.overall_quality),
                "blur_value": float(info.quality.blur_value),
                "noise_level": float(info.quality.noise_level),
                "jpeg_blockiness": float(info.quality.jpeg_blockiness),
                "sharpness_score": float(info.quality.sharpness_score),
                "quality_sensitive_excluded": False,
            },
        )
        save_json(stage2.geometry, photo_dir / "geometry_metrics.json")
        save_json(stage2.texture, photo_dir / "texture_metrics.json")
        return stage2