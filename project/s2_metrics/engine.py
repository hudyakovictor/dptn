from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from ..shared.logging import setup_logger
from ..shared.schemas import PipelineDataset, Stage1Record, Stage2Record
from ..shared.utils import ensure_dir, load_json, load_pickle, load_rgba_png, save_json
from .modules import GEOMETRY_CORE_METRICS, GeometryIdentityResolver, TextureSkinClassifier, load_geometry_metric_catalog, load_texture_metric_catalog
from .modules.geometry.aliases import project_geometry_aliases
from .modules.texture.aliases import project_texture_aliases
from .modules.texture_extractor import TextureExtractor

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
        self.geometry_metric_whitelist = set(GEOMETRY_CORE_METRICS)
        self.texture_extractor = TextureExtractor()

    def run(self) -> list[Stage2Record]:
        stage1_records = self._load_stage1_records()
        if not stage1_records:
            logger.warning("Нет stage1 записей для этапа 2 в %s", self.output_dir)
            return []

        records: list[Stage2Record] = []
        for index, record in enumerate(stage1_records, start=1):
            try:
                metrics = self._process_one(record)
                records.append(metrics)
                logger.info("[s2] %s/%s %s", index, len(stage1_records), record.photo_id)
            except Exception as exc:
                logger.exception("[s2] Ошибка на %s: %s", record.photo_id, exc)

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
        geometry = self._geometry_metrics(info, reconstruction)

        # TextureExtractor: извлекаем все метрики с face_mask_path для alpha-маски
        class TextureCtx:
            image_rgb = rgba[:, :, :3]
            face_bbox = info.face_bbox
            face_mask_path = photo_dir / "face_mask.png"
        texture_ctx = TextureCtx()
        texture = self.texture_extractor.extract(texture_ctx, exclude_sensitive=False)

        geometry.update(project_geometry_aliases(geometry, info))
        geometry = self._filter_geometry_metrics(geometry)
        texture.update(project_texture_aliases(texture))
        geometry_hint = self.geometry_resolver.resolve(geometry)
        texture_hint = self.texture_classifier.classify(texture, info.quality)

        # Добавляем флаг фильтрации
        metric_notes = {
            "geometry_space": "placeholder_canon",
            "texture_source": "face_mask.png",
            "geometry_identity_hint": geometry_hint.get("identity_hint", "UNCERTAIN"),
            "texture_skin_hint": texture_hint.get("texture_skin_hint", "unknown"),
            "geometry_catalog_size": str(len(self.geometry_catalog)),
            "texture_catalog_size": str(len(self.texture_catalog)),
            "quality_sensitive_excluded": str(self.texture_extractor._quality_sensitive_excluded),
        }
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

    def _geometry_metrics(self, info: Stage1Record, reconstruction: dict | None) -> dict[str, float]:
        bbox = np.asarray(info.face_bbox, dtype=float)
        if bbox.size != 4:
            raise RuntimeError(f"Некорректный bbox для {info.photo_id}")
        x, y, w, h = bbox.tolist()
        rec = reconstruction or {}
        landmarks = rec.get("landmarks", {})
        def pt(name: str) -> np.ndarray:
            return np.asarray(landmarks.get(name, [x + w / 2.0, y + h / 2.0]), dtype=float)
        left_eye = pt("left_eye")
        right_eye = pt("right_eye")
        nose = pt("nose")
        mouth_left = pt("mouth_left")
        mouth_right = pt("mouth_right")
        chin = pt("chin")
        forehead = pt("forehead")
        cheek_left = pt("cheek_left")
        cheek_right = pt("cheek_right")
        vertices = np.asarray(rec.get("vertices", []), dtype=float)
        z_span = float(np.ptp(vertices[:, 2])) if vertices.size and vertices.shape[1] >= 3 else 0.0
        mesh_width = float(np.ptp(vertices[:, 0])) if vertices.size and vertices.shape[1] >= 3 else float(w)
        mesh_height = float(np.ptp(vertices[:, 1])) if vertices.size and vertices.shape[1] >= 3 else float(h)

        face_width = float(np.linalg.norm(right_eye - left_eye))
        face_height = float(np.linalg.norm(chin - forehead))
        mouth_width = float(np.linalg.norm(mouth_right - mouth_left))
        cheek_span = float(np.linalg.norm(cheek_right - cheek_left))
        eye_to_mouth = float(np.linalg.norm(((left_eye + right_eye) / 2.0) - ((mouth_left + mouth_right) / 2.0)))
        symmetry_proxy = 1.0 - min(1.0, abs((cheek_left[0] + cheek_right[0]) / 2.0 - (x + w / 2.0)) / max(w, 1.0))

        return {
            "face_width_px": face_width,
            "face_height_px": face_height,
            "face_aspect_ratio": float(face_width / max(face_height, 1e-6)),
            "mesh_width_span": mesh_width,
            "mesh_height_span": mesh_height,
            "mesh_depth_span": z_span,
            "cheekbone_span": cheek_span,
            "orbit_span": face_width,
            "jaw_span": mouth_width,
            "nose_bridge_length": float(np.linalg.norm(forehead - nose)),
            "chin_projection": float((chin[1] - nose[1]) / max(h, 1e-6)),
            "eye_to_mouth_ratio": float(eye_to_mouth / max(face_height, 1e-6)),
            "symmetry_proxy": float(np.clip(symmetry_proxy, 0.0, 1.0)),
            "landmark_dispersion": float(np.std(np.array([left_eye, right_eye, nose, mouth_left, mouth_right, chin]), axis=0).mean()),
            "mesh_vertex_count": float(len(vertices)) if vertices.size else 0.0,
            "mesh_face_count": float(len(rec.get("faces", []))) if rec else 0.0,
        }

    def _texture_metrics(self, rgba: np.ndarray) -> dict[str, float]:
        alpha = rgba[:, :, 3].astype(np.float32) / 255.0
        rgb = rgba[:, :, :3].astype(np.float32)
        mask = alpha > 0.1
        if not np.any(mask):
            mask = np.ones(alpha.shape, dtype=bool)
        gray = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2GRAY)
        pixels = gray[mask]
        color_pixels = rgb[mask]
        if pixels.size == 0:
            pixels = gray.reshape(-1)
            color_pixels = rgb.reshape(-1, 3)

        gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
        laplacian_var = float(cv2.Laplacian(gray_u8, cv2.CV_64F).var())
        entropy = float(_entropy(gray_u8[mask]))
        lbp = local_binary_pattern(gray_u8, P=8, R=1, method="uniform")
        lbp_values = lbp[mask]
        lbp_hist, _ = np.histogram(lbp_values, bins=int(lbp.max() + 1), density=True)
        lbp_uniformity = float(np.sum(lbp_hist[:-1])) if lbp_hist.size > 1 else 0.0
        glcm_features = _glcm_features(gray_u8, mask)
        fft_features = _fft_features(gray_u8, mask)
        edges = cv2.Canny(gray_u8, 40, 120)
        edge_density = float(edges[mask].mean() / 255.0)
        specular_ratio = float(np.mean((color_pixels.mean(axis=1) > 205) & (color_pixels.std(axis=1) < 28)))
        saturation = float(np.mean(_rgb_to_saturation(color_pixels)))
        return {
            "texture_gray_mean": float(np.mean(pixels)),
            "texture_gray_std": float(np.std(pixels)),
            "texture_entropy": entropy,
            "texture_laplacian_var": laplacian_var,
            "texture_lbp_uniformity": lbp_uniformity,
            "texture_fft_highfreq_ratio": fft_features["highfreq_ratio"],
            "texture_fft_peak_ratio": fft_features["peak_ratio"],
            "texture_glcm_contrast": glcm_features["contrast"],
            "texture_glcm_homogeneity": glcm_features["homogeneity"],
            "texture_glcm_energy": glcm_features["energy"],
            "texture_edge_density": edge_density,
            "texture_specular_ratio": specular_ratio,
            "texture_saturation": saturation,
            "texture_color_std": float(np.std(color_pixels)),
        }


def _entropy(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    hist, _ = np.histogram(values, bins=32, range=(0, 255), density=True)
    hist = hist[hist > 0]
    if hist.size == 0:
        return 0.0
    return float(-np.sum(hist * np.log2(hist)))


def _glcm_features(gray_u8: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return {"contrast": 0.0, "homogeneity": 0.0, "energy": 0.0}
    ys = coords[:, 0]
    xs = coords[:, 1]
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop = gray_u8[y0:y1, x0:x1]
    if crop.size == 0:
        return {"contrast": 0.0, "homogeneity": 0.0, "energy": 0.0}
    levels = 16
    quantized = np.floor(crop.astype(np.float32) / (256 / levels)).astype(np.uint8)
    glcm = graycomatrix(quantized, distances=[1], angles=[0], levels=levels, symmetric=True, normed=True)
    return {
        "contrast": float(graycoprops(glcm, "contrast")[0, 0]),
        "homogeneity": float(graycoprops(glcm, "homogeneity")[0, 0]),
        "energy": float(graycoprops(glcm, "energy")[0, 0]),
    }


def _fft_features(gray_u8: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return {"highfreq_ratio": 0.0, "peak_ratio": 0.0}
    y0, y1 = int(coords[:, 0].min()), int(coords[:, 0].max()) + 1
    x0, x1 = int(coords[:, 1].min()), int(coords[:, 1].max()) + 1
    crop = gray_u8[y0:y1, x0:x1].astype(np.float32)
    if crop.size == 0:
        return {"highfreq_ratio": 0.0, "peak_ratio": 0.0}
    crop = crop - float(np.mean(crop))
    spectrum = np.fft.fftshift(np.fft.fft2(crop))
    power = np.abs(spectrum) ** 2
    h, w = power.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    high = power[radius > min(h, w) * 0.25].sum()
    total = power.sum() + 1e-6
    peak_ratio = float(power.max() / total)
    return {"highfreq_ratio": float(high / total), "peak_ratio": peak_ratio}


def _rgb_to_saturation(color_pixels: np.ndarray) -> np.ndarray:
    if color_pixels.size == 0:
        return np.array([], dtype=float)
    maxc = color_pixels.max(axis=1)
    minc = color_pixels.min(axis=1)
    sat = (maxc - minc) / np.clip(maxc, 1e-6, None)
    return np.clip(sat, 0.0, 1.0)
