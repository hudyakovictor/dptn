from __future__ import annotations

import pickle
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from ..shared.logging import setup_logger
from ..shared.schemas import PipelineDataset, PoseEstimate, QualityMetrics, Stage1Record
from ..shared.utils import (
    build_placeholder_reconstruction,
    clamp_bbox,
    create_face_mask_rgba,
    detect_face_bbox,
    ensure_dir,
    expand_bbox,
    image_quality_metrics,
    list_images,
    load_json,
    parse_date_from_name,
    pose_hint_from_name,
    save_json,
    save_pickle,
    save_face_mask_png,
    stable_photo_id,
    subject_age_years_at,
    fallback_face_bbox,
    classify_pose_bucket,
)

logger = setup_logger("deeputin.s1")


class ExtractionEngine:
    def __init__(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        dataset: PipelineDataset,
        limit: int | None = None,
        config: dict | None = None,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.output_dir = ensure_dir(output_dir)
        self.dataset = dataset
        self.limit = limit
        self.config = config or {}

    def run(self) -> list[Stage1Record]:
        photos = list_images(self.input_dir)
        if self.limit is not None:
            photos = photos[: self.limit]
        records: list[Stage1Record] = []
        if not photos:
            logger.warning("Нет фото для этапа 1 в %s", self.input_dir)
            return records

        for index, photo_path in enumerate(photos, start=1):
            try:
                record = self._process_one(photo_path)
                records.append(record)
                logger.info("[s1] %s/%s %s -> %s", index, len(photos), photo_path.name, record.pose.bucket.value)
            except Exception as exc:
                logger.exception("[s1] Ошибка на %s: %s", photo_path.name, exc)
        save_json([r.model_dump() for r in records], self.output_dir / "stage1_manifest.json")
        return records

    def _process_one(self, photo_path: Path) -> Stage1Record:
        photo_id = stable_photo_id(photo_path)
        photo_dir = ensure_dir(self.output_dir / photo_id)
        image_bgr = cv2.imread(str(photo_path))
        if image_bgr is None:
            raise RuntimeError(f"Не удалось прочитать изображение: {photo_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        bbox = detect_face_bbox(image_bgr)
        if bbox is None:
            bbox = fallback_face_bbox(image_bgr.shape)
        bbox = expand_bbox(clamp_bbox(bbox, image_bgr.shape), image_bgr.shape, margin=0.16)
        crop_rgb, face_rgba = create_face_mask_rgba(image_rgb, bbox)

        mask_path = save_face_mask_png(face_rgba, photo_dir / "face_mask.png")
        reconstruction = self._build_reconstruction(image_rgb.shape, bbox, photo_path)
        reconstruction_path = save_pickle(reconstruction, photo_dir / "reconstruction.pkl")

        photo_date = parse_date_from_name(photo_path.stem)
        age_years = subject_age_years_at(photo_date)
        yaw, pitch, roll, bucket_hint, pose_source = pose_hint_from_name(photo_path.stem)
        bucket = classify_pose_bucket(yaw, pitch, roll, fallback=bucket_hint)

        quality = QualityMetrics(**image_quality_metrics(image_bgr, bbox))
        pose = PoseEstimate(
            photo_id=photo_id,
            date=photo_date,
            age_years=age_years,
            yaw=float(yaw),
            pitch=float(pitch),
            roll=float(roll),
            bucket=bucket,
            pose_source=pose_source,
            confidence=0.6 if bucket != bucket.UNKNOWN else 0.3,
        )

        expression_flags = self._expression_flags(reconstruction, quality)
        record = Stage1Record(
            photo_id=photo_id,
            dataset=self.dataset,
            source_path=str(photo_path),
            date=photo_date,
            age_years=age_years,
            pose=pose,
            quality=quality,
            face_bbox=list(map(int, bbox)),
            face_mask_path=str(mask_path),
            reconstruction_path=str(reconstruction_path),
            image_size=[int(image_rgb.shape[1]), int(image_rgb.shape[0])],
            expression_flags=expression_flags,
            readiness={
                "geometry": "available",
                "texture": "available",
            },
        )
        save_json(record.model_dump(), photo_dir / "info.json")
        return record

    def _build_reconstruction(self, image_shape: tuple[int, int, int], bbox: tuple[int, int, int, int], photo_path: Path) -> dict:
        yaw, pitch, roll, _, _ = pose_hint_from_name(photo_path.stem)
        return build_placeholder_reconstruction(image_shape, bbox, yaw, pitch, roll)

    def _expression_flags(self, reconstruction: dict, quality: QualityMetrics) -> dict[str, bool]:
        landmarks = reconstruction.get("landmarks", {})
        left = np.asarray(landmarks.get("mouth_left", [0.0, 0.0]), dtype=float)
        right = np.asarray(landmarks.get("mouth_right", [0.0, 0.0]), dtype=float)
        chin = np.asarray(landmarks.get("chin", [0.0, 0.0]), dtype=float)
        nose = np.asarray(landmarks.get("nose", [0.0, 0.0]), dtype=float)
        mouth_width = float(np.linalg.norm(left - right))
        mouth_drop = float(chin[1] - nose[1])
        smile_excluded = mouth_width > 40.0 and quality.overall_quality > 0.25
        jaw_excluded = mouth_drop < 18.0
        return {
            "smile_excluded": bool(smile_excluded),
            "jaw_excluded": bool(jaw_excluded),
            "neutralized": bool(smile_excluded or jaw_excluded),
        }
