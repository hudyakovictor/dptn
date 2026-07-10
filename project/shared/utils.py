from __future__ import annotations

import json
import math
import os
import pickle
import re
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from hashlib import md5
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .schemas import PipelineDataset, PoseBucket

SUBJECT_BIRTHDATE = date(1952, 10, 7)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
_FACE_CASCADE = None


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def load_json(path: str | Path, default: Any | None = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(data: Any, path: str | Path) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(_json_ready(data), fh, ensure_ascii=False, indent=2, sort_keys=True)
    return p


def save_pickle(data: Any, path: str | Path) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("wb") as fh:
        pickle.dump(data, fh)
    return p


def load_pickle(path: str | Path, default: Any | None = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    with p.open("rb") as fh:
        return pickle.load(fh)


def save_text(text: str, path: str | Path) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(text, encoding="utf-8")
    return p


def load_yaml(path: str | Path, default: Any | None = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    import yaml

    return yaml.safe_load(p.read_text(encoding="utf-8"))


def save_yaml(data: Any, path: str | Path) -> Path:
    import yaml

    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(yaml.safe_dump(_json_ready(data), sort_keys=False, allow_unicode=True), encoding="utf-8")
    return p


def list_images(path: str | Path) -> list[Path]:
    root = Path(path)
    if not root.exists():
        return []
    return sorted(
        [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES],
        key=lambda p: p.name.lower(),
    )


def stable_photo_id(path: str | Path) -> str:
    return Path(path).stem


def parse_date_from_name(name: str) -> date | None:
    patterns = [
        r"(?P<y>19\d{2}|20\d{2})[_-](?P<m>\d{1,2})[_-](?P<d>\d{1,2})",
        r"(?P<y>19\d{2}|20\d{2})(?P<m>\d{2})(?P<d>\d{2})",
    ]
    for pattern in patterns:
        match = re.search(pattern, name)
        if not match:
            continue
        try:
            return date(int(match.group("y")), int(match.group("m")), int(match.group("d")))
        except ValueError:
            continue
    return None


def subject_age_years_at(photo_date: date | None, birthdate: date = SUBJECT_BIRTHDATE) -> float | None:
    if photo_date is None:
        return None
    delta = photo_date - birthdate
    return round(delta.days / 365.2425, 2)


def detect_face_bbox(image_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _FACE_CASCADE = cv2.CascadeClassifier(cascade_path)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5, minSize=(64, 64))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
    return int(x), int(y), int(w), int(h)


def fallback_face_bbox(image_shape: tuple[int, int, int] | tuple[int, int]) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    side = int(min(width, height) * 0.72)
    x = max(0, (width - side) // 2)
    y = max(0, (height - side) // 2)
    return x, y, side, side


def clamp_bbox(bbox: tuple[int, int, int, int], image_shape: tuple[int, int, int] | tuple[int, int]) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    x, y, w, h = bbox
    x = max(0, min(int(x), width - 1))
    y = max(0, min(int(y), height - 1))
    w = max(1, min(int(w), width - x))
    h = max(1, min(int(h), height - y))
    return x, y, w, h


def expand_bbox(bbox: tuple[int, int, int, int], image_shape: tuple[int, int, int] | tuple[int, int], margin: float = 0.18) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    dx = int(w * margin)
    dy = int(h * margin)
    return clamp_bbox((x - dx, y - dy, w + 2 * dx, h + 2 * dy), image_shape)


def crop_image(image_rgb: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = bbox
    return image_rgb[y : y + h, x : x + w]


def create_face_mask_rgba(image_rgb: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    x, y, w, h = bbox
    crop = image_rgb[y : y + h, x : x + w]
    if crop.size == 0:
        raise ValueError("Empty crop for face mask")
    pil = Image.fromarray(crop.astype(np.uint8))
    mask = Image.new("L", pil.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, pil.size[0] - 1, pil.size[1] - 1), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=max(2, min(pil.size) // 20)))
    alpha = np.asarray(mask, dtype=np.uint8)
    rgba = np.dstack([crop, alpha])
    return crop, rgba


def image_quality_metrics(image_bgr: np.ndarray, bbox: tuple[int, int, int, int] | None = None) -> dict[str, Any]:
    if bbox is not None:
        x, y, w, h = bbox
        face = image_bgr[y : y + h, x : x + w]
    else:
        face = image_bgr
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    blur_value = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    median = cv2.medianBlur(gray, 3)
    noise_level = float(np.mean(np.abs(gray.astype(np.float32) - median.astype(np.float32))))
    blockiness = _jpeg_blockiness(gray)
    sharpness = float(np.clip(blur_value / max(_sharpness_denominator(min(gray.shape[:2])), 1e-6), 0.0, 1.0))
    overall_quality = float(np.clip((sharpness * 0.7) + ((1.0 - noise_level / 35.0) * 0.3), 0.0, 1.0))
    is_motion_blurred = bool(_motion_blur_detected(gray))
    is_jpeg_blocky = bool(blockiness > 1.3)
    is_over_smoothed = bool(sharpness > 0.88 and noise_level < 6.5 and not is_motion_blurred)
    return {
        "blur_value": blur_value,
        "noise_level": noise_level,
        "jpeg_blockiness": blockiness,
        "sharpness_score": sharpness,
        "overall_quality": overall_quality,
        "is_motion_blurred": is_motion_blurred,
        "is_jpeg_blocky": is_jpeg_blocky,
        "is_over_smoothed": is_over_smoothed,
    }


def pose_hint_from_name(stem: str) -> tuple[float, float, float, PoseBucket, str]:
    name = stem.lower()
    mapping: list[tuple[str, float, PoseBucket]] = [
        ("left_threequarter_deep", -67.5, PoseBucket.LEFT_THREEQUARTER_DEEP),
        ("right_threequarter_deep", 67.5, PoseBucket.RIGHT_THREEQUARTER_DEEP),
        ("left_threequarter_medium", -45.0, PoseBucket.LEFT_THREEQUARTER_MEDIUM),
        ("right_threequarter_medium", 45.0, PoseBucket.RIGHT_THREEQUARTER_MEDIUM),
        ("left_threequarter_light", -22.5, PoseBucket.LEFT_THREEQUARTER_LIGHT),
        ("right_threequarter_light", 22.5, PoseBucket.RIGHT_THREEQUARTER_LIGHT),
        ("left_profile", -90.0, PoseBucket.LEFT_PROFILE),
        ("right_profile", 90.0, PoseBucket.RIGHT_PROFILE),
        ("frontal", 0.0, PoseBucket.FRONTAL),
    ]
    for token, yaw, bucket in mapping:
        if token in name:
            return yaw, 0.0, 0.0, bucket, "filename"
    return 0.0, 0.0, 0.0, PoseBucket.UNKNOWN, "filename"


def classify_pose_bucket(yaw: float, pitch: float, roll: float = 0.0, fallback: PoseBucket = PoseBucket.UNKNOWN) -> PoseBucket:
    """
    Classify pose into 9 buckets based on yaw angle.
    Based on 3DDFA-V3 yaw convention: positive = right turn, negative = left turn.
    Thresholds from audit spec:
    - frontal: |yaw| < 10
    - threequarter_light: 10-33
    - threequarter_medium: 33-56
    - threequarter_deep: 56-78
    - profile: 78+
    """
    ay = abs(float(yaw))
    if ay < 10:
        return PoseBucket.FRONTAL
    if yaw < 0:
        if ay < 33:
            return PoseBucket.LEFT_THREEQUARTER_LIGHT
        if ay < 56:
            return PoseBucket.LEFT_THREEQUARTER_MEDIUM
        if ay < 78:
            return PoseBucket.LEFT_THREEQUARTER_DEEP
        return PoseBucket.LEFT_PROFILE
    if yaw > 0:
        if ay < 33:
            return PoseBucket.RIGHT_THREEQUARTER_LIGHT
        if ay < 56:
            return PoseBucket.RIGHT_THREEQUARTER_MEDIUM
        if ay < 78:
            return PoseBucket.RIGHT_THREEQUARTER_DEEP
        return PoseBucket.RIGHT_PROFILE
    return fallback


def build_placeholder_reconstruction(
    image_shape: tuple[int, int, int],
    bbox: tuple[int, int, int, int],
    yaw: float,
    pitch: float,
    roll: float,
) -> dict[str, Any]:
    x, y, w, h = bbox
    gx, gy = 12, 14
    vertices: list[list[float]] = []
    for iy in range(gy):
        fy = (iy / max(gy - 1, 1)) * 2.0 - 1.0
        for ix in range(gx):
            fx = (ix / max(gx - 1, 1)) * 2.0 - 1.0
            depth = 1.0 - 0.42 * (fx * fx + fy * fy)
            depth += 0.05 * math.tanh(yaw / 60.0) * fx
            depth -= 0.04 * math.tanh(pitch / 40.0) * fy
            depth += 0.02 * math.sin(math.radians(roll))
            vertices.append(
                [
                    float(x + ((fx + 1.0) / 2.0) * w),
                    float(y + ((fy + 1.0) / 2.0) * h),
                    float(depth * max(w, h) * 0.12),
                ]
            )

    faces: list[list[int]] = []
    for iy in range(gy - 1):
        for ix in range(gx - 1):
            i0 = iy * gx + ix
            i1 = i0 + 1
            i2 = i0 + gx
            i3 = i2 + 1
            faces.append([i0, i2, i1])
            faces.append([i1, i2, i3])

    vertices_np = np.asarray(vertices, dtype=np.float32)
    face_centroid = vertices_np.mean(axis=0)
    normals = vertices_np - face_centroid
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.clip(norm, 1e-6, None)

    landmarks = {
        "left_eye": [float(x + 0.33 * w), float(y + 0.38 * h)],
        "right_eye": [float(x + 0.67 * w), float(y + 0.38 * h)],
        "nose": [float(x + 0.50 * w), float(y + 0.55 * h)],
        "mouth_left": [float(x + 0.38 * w), float(y + 0.74 * h)],
        "mouth_right": [float(x + 0.62 * w), float(y + 0.74 * h)],
        "chin": [float(x + 0.50 * w), float(y + 0.92 * h)],
        "forehead": [float(x + 0.50 * w), float(y + 0.12 * h)],
        "cheek_left": [float(x + 0.24 * w), float(y + 0.58 * h)],
        "cheek_right": [float(x + 0.76 * w), float(y + 0.58 * h)],
    }

    return {
        "space": "placeholder_canon",
        "image_shape": list(image_shape[:2]),
        "bbox": list(map(int, bbox)),
        "vertices": vertices_np.tolist(),
        "faces": faces,
        "normals": normals.tolist(),
        "landmarks": landmarks,
        "pose": {"yaw": float(yaw), "pitch": float(pitch), "roll": float(roll)},
        "mesh_quality": {
            "vertex_count": int(vertices_np.shape[0]),
            "face_count": int(len(faces)),
        },
    }


def _json_ready(data: Any) -> Any:
    if isinstance(data, dict):
        return {str(k): _json_ready(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_json_ready(v) for v in data]
    if isinstance(data, tuple):
        return [_json_ready(v) for v in data]
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, (np.float32, np.float64)):
        return float(data)
    if isinstance(data, (np.int32, np.int64, np.integer)):
        return int(data)
    if is_dataclass(data):
        return _json_ready(asdict(data))
    if hasattr(data, "model_dump"):
        return _json_ready(data.model_dump())
    if isinstance(data, (datetime, date)):
        return data.isoformat()
    return data


def _sharpness_denominator(min_face_dim: int) -> float:
    dim = max(int(min_face_dim), 64)
    return 400.0 * float(np.clip(dim / 224.0, 0.35, 2.5))


def _motion_blur_detected(gray: np.ndarray) -> bool:
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    var_x = max(float(np.var(sobel_x)), 1e-5)
    var_y = max(float(np.var(sobel_y)), 1e-5)
    ratio = max(var_x / var_y, var_y / var_x)
    return ratio > 3.0 and min(var_x, var_y) < 100.0


def _jpeg_blockiness(gray: np.ndarray) -> float:
    if gray.shape[1] <= 16 or gray.shape[0] <= 16:
        return 1.0
    boundary = gray[:, 7::8]
    inside = gray[:, 3::8]
    if boundary.size == 0 or inside.size == 0:
        return 1.0
    min_w = min(boundary.shape[1], inside.shape[1])
    boundary = boundary[:, :min_w]
    inside = inside[:, :min_w]
    return float(np.mean(np.abs(boundary.astype(np.float32) - inside.astype(np.float32)))) / 10.0 + 1.0


def md5_file(path: str | Path) -> str:
    hasher = md5()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_face_mask_png(rgba: np.ndarray, path: str | Path) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    Image.fromarray(rgba.astype(np.uint8), mode="RGBA").save(p)
    return p


def load_rgba_png(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGBA"))


def ensure_dataset_enum(value: str | PipelineDataset) -> PipelineDataset:
    if isinstance(value, PipelineDataset):
        return value
    try:
        return PipelineDataset(value)
    except Exception:
        return PipelineDataset.MAIN
