from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from ..shared.logging import setup_logger
from ..shared.schemas import PipelineDataset, Stage1Record, Stage2Record
from ..shared.utils import ensure_dir, load_json, load_pickle, load_rgba_png, save_json
from .modules import GeometryIdentityResolver, TextureSkinClassifier, load_geometry_metric_catalog, load_texture_metric_catalog
from .modules.geometry_extractor import GeometryExtractor
from .modules.texture.aliases import project_texture_aliases
from .modules.texture_extractor import TextureExtractor
from .texture_anomaly import CohortTextureAnomalyDetector
from .physical_features import PhysicalTextureExtractor

logger = setup_logger("deeputin.s2")


class MetricsEngine:
    def __init__(self, output_dir: str | Path, dataset: PipelineDataset, config: dict | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.dataset = dataset
        self.config = config or {}
        root = Path(__file__).resolve().parents[2]
        geometry_table = self.config.get(
            "geometry_evidence_table",
            root / "imgtest" / "futureplan" / "stage3_geomety_info" / "assets" / "METRIC_EVIDENCE_TABLE.csv",
        )
        texture_leaderboard = self.config.get(
            "texture_leaderboard",
            root / "imgtest" / "unified_test" / "clean_feature_leaderboard.csv",
        )
        self.geometry_resolver = GeometryIdentityResolver(geometry_table)
        self.texture_classifier = TextureSkinClassifier(texture_leaderboard)
        self.geometry_catalog = load_geometry_metric_catalog()
        self.texture_catalog = load_texture_metric_catalog(texture_leaderboard)
        self.texture_extractor = TextureExtractor()
        self.geometry_extractor = GeometryExtractor()
        self.cohort_detector = CohortTextureAnomalyDetector()
        self.physical_extractor = PhysicalTextureExtractor()

    def run(self) -> list[Stage2Record]:
        stage1_records = self._load_stage1_records()
        if not stage1_records:
            logger.warning("Нет stage1 записей для этапа 2 в %s", self.output_dir)
            return []

        # Pass 1: extract all metrics and group by cohort for anomaly detection
        raw_records: list[tuple[Stage1Record, Stage2Record]] = []
        cohort_groups: dict[str, list[dict]] = {}
        for index, record in enumerate(stage1_records, start=1):
            try:
                metrics = self._process_one(record)
                raw_records.append((record, metrics))
                logger.info("[s2] %s/%s %s", index, len(stage1_records), record.photo_id)
                year = record.date.year if record.date else 2000
                cohort_key = self.cohort_detector.get_cohort_key(year)
                if cohort_key not in cohort_groups:
                    cohort_groups[cohort_key] = []
                cohort_groups[cohort_key].append(metrics.texture)
            except Exception as exc:
                logger.exception("[s2] Ошибка на %s: %s", record.photo_id, exc)

        # Fit cohort baselines
        for cohort_key, cohort_textures in cohort_groups.items():
            if len(cohort_textures) >= 3:
                try:
                    self.cohort_detector.fit_cohort(cohort_textures, cohort_key)
                    logger.info("[s2] Cohort '%s': fitted on %d samples", cohort_key, len(cohort_textures))
                except Exception as exc:
                    logger.warning("[s2] Cohort '%s' fit failed: %s", cohort_key, exc)

        # Pass 2: score each photo against its cohort and finalize
        records: list[Stage2Record] = []
        for stage1_record, stage2_record in raw_records:
            year = stage1_record.date.year if stage1_record.date else 2000
            cohort_key = self.cohort_detector.get_cohort_key(year)
            quality = float(stage1_record.quality.overall_quality) if stage1_record else 0.5
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

        # Extract geometry from real 3DDFA-V3 reconstruction
        geometry = self.geometry_extractor.extract(reconstruction)

        # TextureExtractor: извлекаем все метрики с face_mask_path для alpha-маски
        class TextureCtx:
            image_rgb = rgba[:, :, :3]
            face_bbox = info.face_bbox
            face_mask_path = photo_dir / "face_mask.png"
        texture_ctx = TextureCtx()
        texture = self.texture_extractor.extract(texture_ctx, exclude_sensitive=False)

        texture.update(project_texture_aliases(texture))
        geometry_hint = self.geometry_resolver.resolve(geometry)
        texture_hint = self.texture_classifier.classify(texture, info.quality)

        # Extract physical texture features (SSS, specular, pores, etc.)
        physical_features = {}
        try:
            landmarks_68 = reconstruction.get("landmarks_68")
            if (not landmarks_68 or len(landmarks_68) == 0):
                landmarks_68 = reconstruction.get("landmarks_106")
            if landmarks_68 is not None and len(landmarks_68) > 0 and rgba is not None:
                landmarks = np.array(landmarks_68, dtype=np.float32)
                if landmarks.ndim == 2 and landmarks.shape[1] >= 2:
                    image_rgb = rgba[:, :, :3]
                    seg_mask = rgba[:, :, 3] > 128 if rgba.shape[2] == 4 else np.ones(rgba.shape[:2], dtype=bool)
                    pf = self.physical_extractor.extract(image_rgb, landmarks, seg_mask)
                    physical_features = {
                        "sss_index": pf.sss_index,
                        "specular_sharpness": pf.specular_sharpness,
                        "pore_periodicity": pf.pore_periodicity,
                        "lbp_nonuniform_ratio": pf.lbp_nonuniform_ratio,
                        "spectral_slope": pf.spectral_slope,
                        "hemoglobin_index": pf.hemoglobin_index,
                        "seam_score": pf.seam_score,
                        "wrinkle_anisotropy": pf.wrinkle_anisotropy,
                        "wrinkle_dominant_angle": pf.wrinkle_dominant_angle,
                    }
        except Exception:
            pass

        # Добавляем флаг фильтрации
        texture_weights_json = texture.pop("texture_feature_weights_json", None)
        metric_notes = {
            "geometry_space": "3ddfa_v3_canonical",
            "texture_source": "face_mask.png",
            "geometry_identity_hint": geometry_hint.get("identity_hint", "UNCERTAIN"),
            "texture_skin_hint": texture_hint.get("texture_skin_hint", "unknown"),
            "geometry_catalog_size": str(len(self.geometry_catalog)),
            "texture_catalog_size": str(len(self.texture_catalog)),
            "quality_sensitive_excluded": str(self.texture_extractor._quality_sensitive_excluded),
        }
        if texture_weights_json:
            metric_notes["texture_feature_weights_json"] = texture_weights_json
        for k, v in physical_features.items():
            metric_notes[f"physical_{k}"] = str(v)
        if self.texture_extractor._quality_sensitive_excluded:
            metric_notes["quality_filter_reason"] = "low_quality_detected"

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
            quality_summary={
                "overall_quality": float(info.quality.overall_quality),
                "blur_value": float(info.quality.blur_value),
                "noise_level": float(info.quality.noise_level),
                "jpeg_blockiness": float(info.quality.jpeg_blockiness),
                "sharpness_score": float(info.quality.sharpness_score),
                "quality_sensitive_excluded": self.texture_extractor._quality_sensitive_excluded,
            },
        )
        save_json(stage2.model_dump(), photo_dir / "metrics.json")
        save_json(
            {
                "photo_id": info.photo_id,
                "geometry_hint": geometry_hint,
                "texture_hint": texture_hint,
                "selected_metric_keys": selected_keys,
                "quality_sensitive_excluded": self.texture_extractor._quality_sensitive_excluded,
            },
            photo_dir / "stage2_hints.json",
        )
        return stage2

    def _filter_geometry_metrics(self, geometry: dict[str, float]) -> dict[str, float]:
        if not self.geometry_metric_whitelist:
            return geometry
        return {key: value for key, value in geometry.items() if key in self.geometry_metric_whitelist}