from __future__ import annotations

import numpy as np
from .types import ReconstructionResult, VisibilityResult
try:
    from core.constants import Z_TOLERANCE_RATIO
except ImportError:
    from .core_constants import Z_TOLERANCE_RATIO
from typing import Dict, List

def compute_software_zbuffer_mask(vertices_camera: np.ndarray, resolution: int = 512) -> np.ndarray:
    """
    [CORE-03] Software Z-buffer for occlusion detection.
    [ITER-1.4] Unified tolerance (0.005) for higher forensic precision.
    """
    if vertices_camera.ndim != 2 or vertices_camera.shape[1] != 3:
        return np.zeros((vertices_camera.shape[0],), dtype=bool)

    finite_mask = np.isfinite(vertices_camera).all(axis=1)
    if not np.any(finite_mask):
        return np.zeros((vertices_camera.shape[0],), dtype=bool)

    valid_vertices = vertices_camera[finite_mask]
    x, y, z = valid_vertices[:, 0], valid_vertices[:, 1], valid_vertices[:, 2]

    x_span = max(float(x.max() - x.min()), 1e-6)
    y_span = max(float(y.max() - y.min()), 1e-6)

    x_idx = np.clip(((x - x.min()) / x_span) * (resolution - 1), 0, resolution - 1).astype(np.int32)
    y_idx = np.clip(((y - y.min()) / y_span) * (resolution - 1), 0, resolution - 1).astype(np.int32)

    z_buffer = np.full((resolution, resolution), np.inf, dtype=np.float32)
    np.minimum.at(z_buffer, (y_idx, x_idx), z)

    # [ITER-1.4] Unified tolerance
    z_min, z_max = float(z.min()), float(z.max())
    epsilon = max((z_max - z_min) * Z_TOLERANCE_RATIO, 1e-6)
    
    visible_valid = z <= (z_buffer[y_idx, x_idx] + epsilon)

    visible_mask = np.zeros((vertices_camera.shape[0],), dtype=bool)
    visible_mask[finite_mask] = visible_valid
    return visible_mask

def compute_visibility(reconstruction: ReconstructionResult, angle_threshold_deg: float) -> VisibilityResult:
    """
    Combines normal-based facing check and Z-buffer occlusion check.
    """
    # Normals camera-space (simplified normalization for example)
    normals = reconstruction.normals_camera
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / (norms + 1e-8)
    
    view_direction = np.array([0, 0, 1], dtype=np.float32)
    facing_cosines = np.sum(normals * view_direction, axis=1)
    
    cosine_threshold = float(np.cos(np.deg2rad(angle_threshold_deg)))
    binary_mask = facing_cosines >= cosine_threshold
    
    # Occlusion check
    zbuffer_mask = compute_software_zbuffer_mask(reconstruction.vertices_camera)
    binary_mask &= zbuffer_mask
    
    # Optional: combine with renderer-provided mask
    if hasattr(reconstruction, "visible_idx_renderer"):
        binary_mask &= np.asarray(reconstruction.visible_idx_renderer, dtype=bool)

    cosine_weights = np.clip((facing_cosines - cosine_threshold) / max(1e-6, 1.0 - cosine_threshold), 0.0, 1.0)
    cosine_weights *= binary_mask.astype(np.float32)

    # V-01: Gradient fade for yaw 45-60 degrees to prevent geometric "hallucinations" in profile shots
    try:
        angles = getattr(reconstruction, "angles_deg", None)
        if angles is not None and len(angles) > 0:
            yaw = float(angles[1])
            yaw_abs = abs(yaw)
            if 45.0 <= yaw_abs <= 60.0:
                fade = float((60.0 - yaw_abs) / 15.0)
                x_coords = reconstruction.vertices_camera[:, 0]
                if yaw > 0:
                    turning_away = x_coords < 0
                else:
                    turning_away = x_coords > 0
                cosine_weights[turning_away] *= fade
            elif yaw_abs > 60.0:
                x_coords = reconstruction.vertices_camera[:, 0]
                if yaw > 0:
                    turning_away = x_coords < 0
                else:
                    turning_away = x_coords > 0
                cosine_weights[turning_away] *= 0.0
    except Exception:
        pass

    return VisibilityResult(
        binary_mask=binary_mask,
        cosine_weights=cosine_weights,
        facing_cosines=facing_cosines,
        visible_count=int(np.count_nonzero(binary_mask)),
    )


def get_visible_zones(yaw: float, pitch: float) -> List[str]:
    """
    Определяет список видимых зон лица на основе углов поворота головы (в градусах).
    Согласно ТЗ: предотвращение геометрических «галлюцинаций» в профильных снимках.
    """
    # Базовые зоны, видимые всегда (фронтально)
    visible = [
        "nasal_bridge", "chin", "forehead"
    ]
    
    # В проектной конвенции отрицательный yaw = left_profile, положительный = right_profile.
    # При сильном правом профиле видима правая сторона; при левом — левая.
    if yaw < 45.0:
        visible.extend(["left_eye", "left_zygomatic", "left_cheek"])
        
    if yaw > -45.0:
        visible.extend(["right_eye", "right_zygomatic", "right_cheek"])
        
    # Дополнительные проверки для сильного наклона (pitch)
    if pitch > -30.0: # Голова не сильно наклонена вниз
        visible.extend(["jawline", "lower_lip"])
        
    return visible


def filter_metrics_by_pose(metrics: Dict[str, float], yaw: float, pitch: float) -> Dict[str, float]:
    """Оставляет в словаре метрик только те зоны, которые физически видны в данном ракурсе."""
    visible_zones = get_visible_zones(yaw, pitch)
    return {k: v for k, v in metrics.items() if k in visible_zones}


class VisibilityComputer:
    """
    Вычислитель видимости 3D меша.
    Обёртка над функциями visibility.py для совместимости с s1_extraction.modules.
    """
    
    def compute(self, aligned_mesh: dict) -> dict:
        """
        Вычисляет видимость для выровненного меша.
        
        Args:
            aligned_mesh: Словарь с данными выровненного меша (vertices, faces, normals)
            
        Returns:
            Словарь с результатами вычисления видимости (binary_mask, visible_ratio, occlusion_ratio)
        """
        # Если нет данных о вершинах, возвращаем базовый каркас
        if 'vertices' not in aligned_mesh or not aligned_mesh['vertices']:
            return {"binary_mask": None, "visible_ratio": 1.0, "occlusion_ratio": 0.0}
        
        vertices = np.array(aligned_mesh['vertices'])
        if vertices.size == 0:
            return {"binary_mask": None, "visible_ratio": 1.0, "occlusion_ratio": 0.0}
        
        # Вычисляем видимость через Z-buffer
        binary_mask = compute_software_zbuffer_mask(vertices)
        
        # Вычисляем соотношение видимых/скрытых вершин
        total_vertices = len(binary_mask)
        visible_vertices = np.sum(binary_mask)
        visible_ratio = visible_vertices / total_vertices if total_vertices > 0 else 1.0
        occlusion_ratio = 1.0 - visible_ratio
        
        return {
            "binary_mask": binary_mask,
            "visible_ratio": float(visible_ratio),
            "occlusion_ratio": float(occlusion_ratio)
        }
    
    def compute_binary_mask(self, aligned_mesh: dict) -> dict:
        """
        Вычисляет бинарную маску видимости.
        
        Args:
            aligned_mesh: Словарь с данными выровненного меша
            
        Returns:
            Словарь с бинарной маской
        """
        result = self.compute(aligned_mesh)
        return {"binary_mask": result["binary_mask"]}
    
    def compute_occlusion_ratio(self, aligned_mesh: dict) -> float:
        """
        Вычисляет коэффициент перекрытия.
        
        Args:
            aligned_mesh: Словарь с данными выровненного меша
            
        Returns:
            Коэффициент перекрытия (0.0 - нет перекрытия, 1.0 - полностью скрыто)
        """
        result = self.compute(aligned_mesh)
        return result["occlusion_ratio"]
    
    def summarize_visibility(self, aligned_mesh: dict) -> dict:
        """
        Создаёт краткую сводку по видимости зон.
        
        Args:
            aligned_mesh: Словарь с данными выровненного меша
            
        Returns:
            Словарь с краткой сводкой
        """
        result = self.compute(aligned_mesh)
        return {
            "visible_ratio": result["visible_ratio"],
            "occlusion_ratio": result["occlusion_ratio"],
            "total_vertices": len(result["binary_mask"]) if result["binary_mask"] is not None else 0,
            "visible_vertices": int(np.sum(result["binary_mask"])) if result["binary_mask"] is not None else 0
        }
