from __future__ import annotations

import pickle
import sys
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

FACE_CROP_WIDTH = 424
FACE_CROP_HEIGHT = 500
FACE_MASK_FILENAME = "face_mask.png"
FACE_CROP_FILENAME = "face_crop.jpg"
THUMB_FILENAME = "thumb.jpg"
UV_TEXTURE_FILENAME = "uv_texture.png"
UV_CONFIDENCE_FILENAME = "uv_confidence.png"
MESH_OBJ_FILENAME = "mesh.obj"
MESH_MTL_FILENAME = "mesh.mtl"


def _resize_letterbox(bgr: np.ndarray, tw: int, th: int) -> np.ndarray:
    """Вписать в tw×th без искажения пропорций (чёрные поля по краям)."""
    h, w = bgr.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((th, tw, 3), dtype=np.uint8)
    scale = min(tw / w, th / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    y0 = (th - nh) // 2
    x0 = (tw - nw) // 2
    canvas[y0: y0 + nh, x0: x0 + nw] = resized
    return canvas


def _resize_letterbox_gray(gray: np.ndarray, tw: int, th: int) -> np.ndarray:
    h, w = gray.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((th, tw), dtype=np.uint8)
    scale = min(tw / w, th / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((th, tw), dtype=np.uint8)
    y0 = (th - nh) // 2
    x0 = (tw - nw) // 2
    canvas[y0: y0 + nh, x0: x0 + nw] = resized
    return canvas


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

        # Save face mask (424x500 letterboxed RGBA crop with skin alpha mask) + face_crop.jpg + thumb.jpg
        mask_path, crop_path, thumb_path = self._save_face_assets(image_bgr, reconstruction_result, photo_dir)

        # Save full reconstruction to pickle
        reconstruction_dict = self._reconstruction_to_dict(reconstruction_result)
        reconstruction_path = save_pickle(reconstruction_dict, photo_dir / "reconstruction.pkl")

        # Save UV texture + confidence map
        uv_paths = self._save_uv_assets(image_bgr, reconstruction_dict, photo_dir)

        # Save 3D mesh (OBJ + MTL)
        mesh_paths = self._save_mesh_assets(reconstruction_dict, photo_dir)

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

    def _save_face_assets(self, image_bgr: np.ndarray, recon, photo_dir: Path) -> tuple[Path, Path, Path]:
        """Сохраняет face_mask.png (424x500 RGBA letterbox crop), face_crop.jpg, thumb.jpg."""
        seg_visible = recon.payload.get("seg_visible")
        trans_params = recon.payload.get("trans_params")
        h, w = image_bgr.shape[:2]

        # Build skin alpha mask from 3DDFA segmentation
        mask = None
        if seg_visible is not None:
            # seg channels: [right_eye, left_eye, right_eyebrow, left_eyebrow, nose, up_lip, down_lip, skin]
            skin_224 = np.maximum(seg_visible[7], seg_visible[4]).copy()  # skin + nose
            if seg_visible.shape[0] > 6:
                excluded_224 = np.maximum.reduce([
                    seg_visible[0], seg_visible[1], seg_visible[2], seg_visible[3],
                    seg_visible[5], seg_visible[6],
                ])
                exclusion_weight = 1.0 / (1.0 + np.exp(-10 * (excluded_224 - 0.5)))
                skin_224 *= (1.0 - exclusion_weight)
            skin_224_uint8 = np.clip(skin_224 * 255, 0, 255).astype(np.uint8)

            # Project from 224x224 to original image
            try:
                sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core" / "3ddfa_v3"))
                from util.io import back_resize_crop_img
                from PIL import Image as PILImage
                mask_rgb = np.stack((skin_224_uint8, skin_224_uint8, skin_224_uint8), axis=-1)
                blank = np.zeros((h, w, 3), dtype=np.uint8)
                full_mask_rgb = back_resize_crop_img(mask_rgb, trans_params, blank, resample_method=PILImage.BILINEAR)
                mask = full_mask_rgb[:, :, 0]
            except Exception:
                mask = cv2.resize(skin_224_uint8, (w, h), interpolation=cv2.INTER_LINEAR)

        if mask is None:
            # Fallback: oval from landmarks
            landmarks = recon.landmarks_106
            if landmarks is not None and len(landmarks) > 0:
                x_min, y_min = landmarks[:, 0].min(), landmarks[:, 1].min()
                x_max, y_max = landmarks[:, 0].max(), landmarks[:, 1].max()
                bbox = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
                bbox = expand_bbox(clamp_bbox(bbox, image_bgr.shape), image_bgr.shape, margin=0.15)
                _, face_rgba = create_face_mask_rgba(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), bbox)
                mask = face_rgba[:, :, 3]
            else:
                mask = np.zeros((h, w), dtype=np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Find bounding box of mask
        coords = cv2.findNonZero((mask > 10).astype(np.uint8))
        if coords is None:
            mask_path = save_face_mask_png(np.dstack([cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), np.zeros((h, w), dtype=np.uint8)]), photo_dir / FACE_MASK_FILENAME)
            return mask_path, Path(""), Path("")

        x, y, bw, bh = cv2.boundingRect(coords)
        target_aspect = FACE_CROP_WIDTH / FACE_CROP_HEIGHT
        crop_w = int(max(bw * 1.25, bh * 1.25 * target_aspect, 1))
        crop_h = int(max(bh * 1.25, crop_w / target_aspect, 1))
        cx = x + bw / 2.0
        cy = y + bh / 2.0
        x1 = int(round(cx - crop_w / 2.0))
        y1 = int(round(cy - crop_h / 2.0))
        x2 = x1 + crop_w
        y2 = y1 + crop_h
        if x1 < 0:
            x2 -= x1
            x1 = 0
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if x2 > w:
            x1 = max(0, x1 - (x2 - w))
            x2 = w
        if y2 > h:
            y1 = max(0, y1 - (y2 - h))
            y2 = h

        face_crop_bgr = image_bgr[y1:y2, x1:x2].copy()
        face_crop_mask = mask[y1:y2, x1:x2]
        face_crop_bgr = _resize_letterbox(face_crop_bgr, FACE_CROP_WIDTH, FACE_CROP_HEIGHT)
        face_crop_mask = _resize_letterbox_gray(face_crop_mask, FACE_CROP_WIDTH, FACE_CROP_HEIGHT)

        # Save face_mask.png (RGBA)
        face_crop_rgba = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2BGRA)
        face_crop_rgba[:, :, 3] = face_crop_mask
        mask_path = photo_dir / FACE_MASK_FILENAME
        cv2.imwrite(str(mask_path), face_crop_rgba)

        # Save face_crop.jpg (BGR preview)
        crop_path = photo_dir / FACE_CROP_FILENAME
        cv2.imwrite(str(crop_path), face_crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        # Save thumb.jpg (100x100)
        thumb_path = photo_dir / THUMB_FILENAME
        thumb = cv2.resize(face_crop_bgr, (100, 100), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(thumb_path), thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

        return mask_path, crop_path, thumb_path

    def _save_uv_assets(self, image_bgr: np.ndarray, recon_dict: dict, photo_dir: Path) -> dict[str, Path]:
        """Сохраняет uv_texture.png и uv_confidence.png, если доступен UV-модуль."""
        paths = {}
        try:
            core_repo = Path(__file__).resolve().parents[2] / "core"
            sys.path.insert(0, str(core_repo))
            from uv_module.hd_uv_generator import HDUVConfig, HDUVTextureGenerator

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            uv_gen = HDUVTextureGenerator(HDUVConfig(uv_size=768))
            uv_tex_analysis, uv_tex_beauty, uv_mask_visible, uv_confidence, aux = uv_gen.generate(image_rgb, recon_dict)

            # Save UV texture (use beauty version for preview)
            uv_tex_path = photo_dir / UV_TEXTURE_FILENAME
            # uv_tex_beauty is uint8, need to save as RGB PNG
            from PIL import Image as PILImage
            PILImage.fromarray(uv_tex_beauty.astype(np.uint8), mode="RGB").save(str(uv_tex_path))
            paths["uv_texture"] = uv_tex_path

            # Save UV confidence
            uv_conf_path = photo_dir / UV_CONFIDENCE_FILENAME
            conf_uint8 = np.clip(uv_confidence * 255, 0, 255).astype(np.uint8)
            PILImage.fromarray(conf_uint8, mode="L").save(str(uv_conf_path))
            paths["uv_confidence"] = uv_conf_path
        except Exception as exc:
            logger.warning("UV texture generation not available: %s", exc)
        return paths

    def _save_mesh_assets(self, recon_dict: dict, photo_dir: Path) -> dict[str, Path]:
        """Сохраняет mesh.obj и mesh.mtl из 3DDFA-реконструкции."""
        paths = {}
        try:
            vertices = np.asarray(recon_dict.get("vertices", []), dtype=np.float32)
            triangles = np.asarray(recon_dict.get("triangles", []), dtype=np.int32)
            normals = np.asarray(recon_dict.get("normals", []), dtype=np.float32)
            if len(vertices) == 0 or len(triangles) == 0:
                return paths

            # Write MTL file
            mtl_path = photo_dir / MESH_MTL_FILENAME
            mtl_content = f"""# DEEPUTIN 3DDFA-V3 mesh
newmtl face_material
Ka 0.2 0.2 0.2
Kd 0.8 0.8 0.8
Ks 0.0 0.0 0.0
d 1.0
illum 2
map_Kd {UV_TEXTURE_FILENAME}
"""
            mtl_path.write_text(mtl_content)
            paths["mesh_mtl"] = mtl_path

            # Write OBJ file
            obj_path = photo_dir / MESH_OBJ_FILENAME
            lines = [f"mtllib {MESH_MTL_FILENAME}\n"]
            # Vertices
            for v in vertices:
                lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            # Texture coordinates (placeholder)
            for v in vertices:
                lines.append(f"vt 0.0 0.0\n")
            # Normals
            for n in normals:
                lines.append(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
            # Faces
            lines.append("usemtl face_material\n")
            for t in triangles:
                lines.append(f"f {t[0]+1}/{t[0]+1}/{t[0]+1} {t[1]+1}/{t[1]+1}/{t[1]+1} {t[2]+1}/{t[2]+1}/{t[2]+1}\n")
            obj_path.write_text("".join(lines))
            paths["mesh_obj"] = obj_path

        except Exception as exc:
            logger.warning("Mesh export error: %s", exc)
        return paths

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
            "trans_params": recon.trans_params.tolist() if recon.trans_params is not None else None,
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