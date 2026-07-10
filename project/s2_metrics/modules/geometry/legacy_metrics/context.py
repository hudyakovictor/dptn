from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .types import MetricContext

_ATTR_KEY_MAP = {
    "vertices_world": "vertices",
    "vertices_canon": "vertices_canonical",
    "normals_world": "normals",
    "angles_deg": "pose",
    "visibility": "seg_visible",
}

# Zone names matching 3DDFA-V3 annotation_groups order (same as geometry_extractor.py)
ANNOTATION_ZONE_NAMES = (
    "right_eye", "left_eye", "right_eyebrow", "left_eyebrow",
    "nose", "upper_lip", "lower_lip", "skin"
)

# Derived macro zones from base annotation zones
MACRO_ZONES = {
    "forehead": ["skin"],
    "brow_ridge_L": ["left_eyebrow"],
    "brow_ridge_R": ["right_eyebrow"],
    "orbit_L": ["left_eye"],
    "orbit_R": ["right_eye"],
    "nose_bridge_tip": ["nose"],
    "nose_wing_L": ["skin"],
    "nose_wing_R": ["skin"],
    "cheekbone_L": ["skin"],
    "cheekbone_R": ["skin"],
    "chin": ["skin"],
    "jaw_L": ["skin"],
    "jaw_R": ["skin"],
    "upper_lip": ["upper_lip"],
    "lower_lip": ["lower_lip"],
}


def _build_macro_indices(annotation_groups: list, vertices: np.ndarray) -> dict[str, np.ndarray]:
    """Build zone name -> vertex indices mapping from annotation_groups."""
    macro = {}
    # Base annotation zones
    for i, zone_name in enumerate(ANNOTATION_ZONE_NAMES):
        if i < len(annotation_groups):
            idx = annotation_groups[i]
            if idx is not None and len(idx) > 0:
                macro[zone_name] = np.asarray(idx, dtype=np.int64)
    
    # Derived macro zones (union of base zones)
    for macro_name, base_zones in MACRO_ZONES.items():
        indices = []
        for bz in base_zones:
            if bz in macro:
                indices.append(macro[bz])
        if indices:
            macro[macro_name] = np.concatenate(indices)
    return macro


def _get(reconstruction: Any, attr_name: str, default: Any = None) -> Any:
    if isinstance(reconstruction, dict):
        if attr_name == "payload":
            return {
                "id_params": reconstruction.get("id_params"),
                "exp_params": reconstruction.get("exp_params"),
            }
        key = _ATTR_KEY_MAP.get(attr_name, attr_name)
        val = reconstruction.get(key, default)
        if attr_name == "angles_deg" and isinstance(val, dict):
            return [val.get("pitch", 0), val.get("yaw", 0), val.get("roll", 0)]
        return val
    return getattr(reconstruction, attr_name, default)


def _neutral_shape(adapter: Any, recon: Any) -> np.ndarray | None:
    payload = _get(recon, "payload", {}) or {}
    if "id_params" not in payload or not hasattr(adapter, "_model") or adapter._model is None:
        return None
    try:
        import torch

        id_tensor = torch.tensor(payload["id_params"], dtype=torch.float32).unsqueeze(0).to(adapter.runtime_device)
        exp = payload.get("exp_params")
        if exp is None:
            # Ask model for zero expression with a conservative fallback length.
            exp_tensor = torch.zeros((1, 64), dtype=torch.float32, device=adapter.runtime_device)
        else:
            exp_tensor = torch.zeros_like(torch.tensor(exp, dtype=torch.float32).unsqueeze(0).to(adapter.runtime_device))
        return adapter._model.compute_shape(id_tensor, exp_tensor).detach().cpu().numpy()[0]
    except Exception:
        return None



def _shape_basis(adapter: Any, vertex_count: int, id_dim: int | None) -> np.ndarray | None:
    """Best-effort extraction of 3DMM identity basis from adapter assets/model.

    Different 3DDFA/BFM assets use different key names. Return shape as
    (vertex_count, 3, n_coeff) when possible.
    """
    if adapter is None:
        return None
    candidates = []
    assets = getattr(adapter, "_face_model_assets", None)
    if isinstance(assets, dict):
        for key in ("id_base", "shapePC", "w_shape", "base_id", "u_id", "idBase", "shape_basis"):
            if key in assets:
                candidates.append(assets[key])
    model = getattr(adapter, "_model", None)
    if model is not None:
        for key in ("id_base", "shapePC", "w_shape", "base_id", "u_id", "idBase", "shape_basis"):
            if hasattr(model, key):
                candidates.append(getattr(model, key))
    for arr in candidates:
        try:
            if hasattr(arr, "detach"):
                arr = arr.detach().cpu().numpy()
            arr = np.asarray(arr, dtype=float)
            if arr.ndim == 3 and arr.shape[0] == vertex_count and arr.shape[1] == 3:
                return arr
            if arr.ndim == 2:
                # Common shapes: (3N, C), (C, 3N), (N*3, C)
                if arr.shape[0] == vertex_count * 3:
                    return arr.reshape(vertex_count, 3, arr.shape[1])
                if arr.shape[1] == vertex_count * 3:
                    return arr.T.reshape(vertex_count, 3, arr.shape[0])
                if id_dim and arr.shape[1] == id_dim and arr.shape[0] >= vertex_count * 3:
                    return arr[: vertex_count * 3].reshape(vertex_count, 3, arr.shape[1])
        except Exception:
            continue
    return None

def build_metric_context(
    *,
    photo_id: str,
    image_path: Path,
    reconstruction: Any,
    adapter: Any | None,
    pose_bucket: str,
    quality: dict[str, Any] | None = None,
    geometry_metrics: dict[str, Any] | None = None,
    periocular_metrics: dict[str, Any] | None = None,
    texture_forensics: dict[str, Any] | None = None,
    texture_profile: dict[str, Any] | None = None,
) -> MetricContext:
    angles = np.asarray(_get(reconstruction, "angles_deg", np.zeros(3)), dtype=float).reshape(-1)
    pitch = float(angles[0]) if len(angles) > 0 else 0.0
    yaw = float(angles[1]) if len(angles) > 1 else 0.0
    roll = float(angles[2]) if len(angles) > 2 else 0.0
    vertices_raw = np.asarray(_get(reconstruction, "vertices_world"), dtype=float)
    vertices_canon = vertices_raw.copy()

    vertices_shape_neutral = _neutral_shape(adapter, reconstruction) if adapter is not None else None
    visibility_raw = _get(reconstruction, "visibility", None)

    payload = _get(reconstruction, "payload", {}) or {}
    id_arr_tmp = None if payload.get("id_params") is None else np.asarray(payload.get("id_params"), dtype=float)
    basis = _shape_basis(adapter, len(vertices_raw), None if id_arr_tmp is None else int(id_arr_tmp.size))
    image_rgb = None
    try:
        from PIL import Image
        image_rgb = np.asarray(Image.open(image_path).convert("RGB"))
    except Exception:
        image_rgb = None

    from .topology_utils import recompute_vertex_normals

    triangles = np.asarray(_get(reconstruction, "triangles", np.zeros((0, 3))), dtype=np.int64)
    annotation_groups = list(_get(reconstruction, "annotation_groups", []) or [])
    normals_raw = _get(reconstruction, "normals_world", None)
    normals_canon = (
        recompute_vertex_normals(vertices_canon, triangles)
        if vertices_canon is not None
        else normals_raw
    )
    normals_shape_neutral = (
        recompute_vertex_normals(vertices_shape_neutral, triangles)
        if vertices_shape_neutral is not None
        else None
    )

    macro_indices = _build_macro_indices(annotation_groups, vertices_canon)

    return MetricContext(
        photo_id=photo_id,
        image_path=image_path,
        pose_bucket=str(pose_bucket),
        yaw_deg=yaw,
        pitch_deg=pitch,
        roll_deg=roll,
        recon=reconstruction,
        vertices_raw=vertices_raw,
        vertices_canon=vertices_canon,
        vertices_shape_neutral=vertices_shape_neutral,
        normals_raw=normals_raw,
        normals_canon=normals_canon,
        normals_shape_neutral=normals_shape_neutral,
        triangles=triangles,
        annotation_groups=annotation_groups,
        macro_indices=macro_indices,
        landmarks_106=_get(reconstruction, "landmarks_106", None),
        visibility_raw=visibility_raw,
        visibility_canon=visibility_raw,
        id_params=id_arr_tmp,
        exp_params=None if payload.get("exp_params") is None else np.asarray(payload.get("exp_params"), dtype=float),
        shape_basis=basis,
        uv_coords=_get(reconstruction, "uv_coords", None),
        image_rgb=image_rgb,
        quality=quality or {},
        geometry_metrics=geometry_metrics or {},
        periocular_metrics=periocular_metrics or {},
        texture_forensics=texture_forensics or {},
        texture_profile=texture_profile or {},
    )
