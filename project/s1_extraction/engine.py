from __future__ import annotations

import pickle
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from ..shared.logging import setup_logger
from ..shared.schemas import PipelineDataset, PoseEstimate, QualityMetrics, Stage1Record, PoseBucket
from ..shared.utils import (
    clamp_bbox,
    create_face_mask_rgba,
    detect_face_bbox,
    ensure_dir,
    expand_bbox,
    image_quality_metrics,
    list_images,
    load_json,
    parse_date_from_name,
    save_json,
    save_pickle,
    save_face_mask_png,
    stable_photo_id,
    subject_age_years_at,
    fallback_face_bbox,
    classify_pose_bucket,
)
from .modules.reconstruction import ReconstructionAdapter, resolve_reconstruction

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

        # Initialize 3DDFA-V3 adapter
        s1_config = self.config.get("s1", {})
        self.reconstruction_adapter = ReconstructionAdapter(
            device=s1_config.get("device", "auto"),
            detector_device=s1_config.get("detector_device", "auto"),
            backbone=s1_config.get("backbone", "resnet50"),
        )
        self.neutral_expression = s1_config.get("neutral_expression", False)
        self.identity_only = s1_config.get("identity_only", False)

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

        # Run full 3DDFA-V3 reconstruction with disk caching
        reconstruction_result = resolve_reconstruction(
            adapter=self.reconstruction_adapter,
            image_path=photo_path,
            entry_dir=photo_dir,
            neutral_expression=self.neutral_expression,
            identity_only=self.identity_only,
        )

        # Save face mask from 3DDFA seg_visible or fallback to oval
        face_rgba = self._create_face_mask_from_reconstruction(image_rgb, reconstruction_result)
        mask_path = save_face_mask_png(face_rgba, photo_dir / "face_mask.png")

        # Save full reconstruction to pickle
        reconstruction_dict = self._reconstruction_to_dict(reconstruction_result)
        reconstruction_path = save_pickle(reconstruction_dict, photo_dir / "reconstruction.pkl")

        # Extract pose from 3DDFA (not from filename!)
        angles_deg = reconstruction_result.angles_deg
        pitch, yaw, roll = float(angles_deg[0]), float(angles_deg[1]), float(angles_deg[2])
        bucket = reconstruction_result.pose_bucket

        photo_date = parse_date_from_name(photo_path.stem)
        age_years = subject_age_years_at(photo_date)

        quality = QualityMetrics(**image_quality_metrics(image_bgr))

        pose = PoseEstimate(
            photo_id=photo_id,
            date=photo_date,
            age_years=age_years,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            bucket=PoseBucket(bucket),
            pose_source="3ddfa_v3",
            confidence=0.8 if bucket != "unknown" else 0.5,
        )

        expression_flags = self._expression_flags(reconstruction_result)
        record = Stage1Record(
            photo_id=photo_id,
            dataset=self.dataset,
            source_path=str(photo_path),
            date=photo_date,
            age_years=age_years,
            pose=pose,
            quality=quality,
            face_bbox=list(map(int, self._estimate_bbox_from_landmarks(reconstruction_result))),
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

    def _create_face_mask_from_reconstruction(self, image_rgb: np.ndarray, recon) -> np.ndarray:
        """Create face mask from 3DDFA segmentation (seg_visible) or fallback to landmarks bbox."""
        # Try to use seg_visible from 3DDFA for precise skin mask
        seg_visible = recon.payload.get("seg_visible")
        if seg_visible is not None:
            # seg_visible contains 8-part segmentation: [right_eye, left_eye, right_eyebrow, left_eyebrow, nose, up_lip, down_lip, skin]
            # We want the skin part (index 7) + nose + eyebrows for face mask
            skin_mask = seg_visible[7] if len(seg_visible) > 7 else None
            if skin_mask is not None and skin_mask.sum() > 100:
                # Create RGBA from segmentation
                h, w = image_rgb.shape[:2]
                mask_resized = cv2.resize(skin_mask.astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_LINEAR)
                mask_blurred = cv2.GaussianBlur(mask_resized, (15, 15), 0)
                alpha = mask_blurred.astype(np.uint8)
                rgba = np.dstack([image_rgb, alpha])
                return rgba

        # Fallback: use landmarks to create bbox-based oval mask
        landmarks_106 = recon.landmarks_106
        if landmarks_106 is not None and len(landmarks_106) > 0:
            x_min, y_min = landmarks_106[:, 0].min(), landmarks_106[:, 1].min()
            x_max, y_max = landmarks_106[:, 0].max(), landmarks_106[:, 1].max()
            bbox = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
            bbox = expand_bbox(clamp_bbox(bbox, image_rgb.shape), image_rgb.shape, margin=0.15)
            _, face_rgba = create_face_mask_rgba(image_rgb, bbox)
            return face_rgba

        # Ultimate fallback: oval from center
        h, w = image_rgb.shape[:2]
        bbox = (w // 4, h // 4, w // 2, h // 2)
        _, face_rgba = create_face_mask_rgba(image_rgb, bbox)
        return face_rgba

    def _estimate_bbox_from_landmarks(self, recon) -> tuple[int, int, int, int]:
        """Estimate face bbox from 3DDFA landmarks."""
        landmarks_106 = recon.landmarks_106
        if landmarks_106 is not None and len(landmarks_106) > 0:
            x_min, y_min = landmarks_106[:, 0].min(), landmarks_106[:, 1].min()
            x_max, y_max = landmarks_106[:, 0].max(), landmarks_106[:, 1].max()
            bbox = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
            return expand_bbox(clamp_bbox(bbox, (recon.image_size[1], recon.image_size[0]) if hasattr(recon, 'image_size') else (512, 512)), (512, 512), margin=0.15)
        return (0, 0, 512, 512)

    def _reconstruction_to_dict(self, recon) -> dict:
        """Convert ReconstructionResult to serializable dict for pickle."""
        return {
            "space": "3ddfa_v3_canonical",
            "image_shape": [int(recon.vertices_image.shape[0]), int(recon.vertices_image.shape[1])] if recon.vertices_image is not None else [512, 512],
            "vertices": recon.vertices_world.tolist() if recon.vertices_world is not None else [],
            "vertices_canonical": recon.vertices_camera.tolist() if recon.vertices_camera is not None else [],
            "triangles": recon.triangles.tolist() if recon.triangles is not None else [],
            "normals": recon.normals_world.tolist() if recon.normals_world is not None else [],
            "landmarks_106": recon.landmarks_106.tolist() if recon.landmarks_106 is not None else [],
            "landmarks_68": [],  # not extracted by default
            "pose": {
                "yaw": float(recon.angles_deg[1]),
                "pitch": float(recon.angles_deg[0]),
                "roll": float(recon.angles_deg[2]),
            },
            "rotation_matrix": recon.rotation_matrix.tolist() if recon.rotation_matrix is not None else [],
            "translation": recon.translation.tolist() if recon.translation is not None else [],
            "mesh_quality": {
                "vertex_count": int(recon.vertices_world.shape[0]) if recon.vertices_world is not None else 0,
                "face_count": int(recon.triangles.shape[0]) if recon.triangles is not None else 0,
                "visible_vertices": int(np.count_nonzero(recon.visible_idx_renderer)) if recon.visible_idx_renderer is not None else 0,
            },
            "annotation_groups": [g.tolist() for g in recon.annotation_groups] if recon.annotation_groups else [],
            "seg_visible": recon.payload.get("seg_visible"),
            "id_params": recon.payload.get("id_params", []).tolist() if isinstance(recon.payload.get("id_params"), np.ndarray) else [],
            "exp_params": recon.payload.get("exp_params", []).tolist() if isinstance(recon.payload.get("exp_params"), np.ndarray) else [],
        }

    def _expression_flags(self, recon) -> dict[str, bool]:
        exp_params = recon.payload.get("exp_params")
        if exp_params is None or len(exp_params) == 0:
            return {"smile_excluded": False, "jaw_excluded": False, "neutralized": False}

        exp_np = np.asarray(exp_params, dtype=float)
        # exp[0] = jaw open, exp[1], exp[2] = smile
        jaw_open = float(abs(exp_np[0])) if len(exp_np) > 0 else 0.0
        smile_intensity = float(max(abs(exp_np[1]), abs(exp_np[2]))) if len(exp_np) > 2 else 0.0

        smile_excluded = smile_intensity > 2.0  # threshold on PCA coeff
        jaw_excluded = jaw_open > 0.8

        return {
            "smile_excluded": bool(smile_excluded),
            "jaw_excluded": bool(jaw_excluded),
            "neutralized": bool(smile_excluded or jaw_excluded),
        }