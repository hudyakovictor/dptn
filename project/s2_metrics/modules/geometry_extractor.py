"""Geometry extractor - извлечение геометрических метрик из 3DDFA-V3 реконструкции."""
from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional

from deeputin.shared.schemas import PoseBucket


# Zone names matching 3DDFA-V3 annotation_groups order:
# [right_eye, left_eye, right_eyebrow, left_eyebrow, nose, upper_lip, lower_lip, skin]
ANNOTATION_ZONE_NAMES = (
    "right_eye", "left_eye", "right_eyebrow", "left_eyebrow", 
    "nose", "upper_lip", "lower_lip", "skin"
)

# Derived macro zones (from zones.py in old backend)
MACRO_ZONES = {
    "forehead": ["skin"],  # upper part of skin
    "brow_ridge_L": ["left_eyebrow"],
    "brow_ridge_R": ["right_eyebrow"],
    "orbit_L": ["left_eye"],
    "orbit_R": ["right_eye"],
    "nose_bridge_tip": ["nose"],
    "nose_wing_L": ["skin"],  # left part of nose area
    "nose_wing_R": ["skin"],  # right part of nose area
    "cheekbone_L": ["skin"],  # left cheek
    "cheekbone_R": ["skin"],  # right cheek
    "chin": ["skin"],  # lower part of skin
    "jaw_L": ["skin"],  # left jaw
    "jaw_R": ["skin"],  # right jaw
    "upper_lip": ["upper_lip"],
    "lower_lip": ["lower_lip"],
}


def get_zone_vertices(vertices: np.ndarray, annotation_groups: List[np.ndarray], zone_name: str) -> np.ndarray:
    """Get vertices for a specific zone by name."""
    try:
        idx = ANNOTATION_ZONE_NAMES.index(zone_name)
        zone_indices = annotation_groups[idx]
        if zone_indices is not None and len(zone_indices) > 0:
            return vertices[zone_indices]
    except (ValueError, IndexError):
        pass
    return np.zeros((0, 3), dtype=np.float32)


def compute_zone_metrics(vertices: np.ndarray, annotation_groups: List[np.ndarray], 
                         face_scale: float, visible_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute geometric metrics for all anatomical zones."""
    metrics = {}
    
    if vertices.size == 0 or face_scale <= 0:
        return metrics
    
    # Precompute face bbox for normalization
    v_min = vertices.min(axis=0)
    v_max = vertices.max(axis=0)
    face_bbox = v_max - v_min
    
    for i, zone_name in enumerate(ANNOTATION_ZONE_NAMES):
        if i >= len(annotation_groups):
            continue
            
        zone_indices = annotation_groups[i]
        if zone_indices is None or len(zone_indices) == 0:
            continue
            
        zone_verts = vertices[zone_indices]
        if zone_verts.size == 0:
            continue
        
        # Apply visibility filter if provided
        if visible_mask is not None and len(visible_mask) == len(vertices):
            zone_visible = visible_mask[zone_indices]
            zone_verts = zone_verts[zone_visible]
            if zone_verts.size == 0:
                continue
        
        prefix = f"zone_{zone_name}"
        
        # Bounding box
        z_min = zone_verts.min(axis=0)
        z_max = zone_verts.max(axis=0)
        bbox = z_max - z_min
        
        # Centroid
        centroid = zone_verts.mean(axis=0)
        
        # Normal stats (using PCA on zone vertices for normal approximation)
        if len(zone_verts) >= 3:
            centered = zone_verts - centroid
            try:
                _, _, vh = np.linalg.svd(centered, full_matrices=False)
                normal = vh[-1]
                normal = normal / (np.linalg.norm(normal) + 1e-8)
                normal_mean = normal
                normal_var = float(np.mean(np.abs(centered @ normal)))
            except np.linalg.LinAlgError:
                normal_mean = np.array([0.0, 0.0, 1.0])
                normal_var = 0.0
        else:
            normal_mean = np.array([0.0, 0.0, 1.0])
            normal_var = 0.0
        
        # Span ratios
        metrics[f"{prefix}_bbox_volume_ratio"] = float(np.prod(bbox) / (np.prod(face_bbox) + 1e-10))
        metrics[f"{prefix}_centroid_x"] = float(centroid[0] / face_scale)
        metrics[f"{prefix}_centroid_y"] = float(centroid[1] / face_scale)
        metrics[f"{prefix}_centroid_z"] = float(centroid[2] / face_scale)
        metrics[f"{prefix}_span_x"] = float(bbox[0] / face_scale)
        metrics[f"{prefix}_span_y"] = float(bbox[1] / face_scale)
        metrics[f"{prefix}_span_z"] = float(bbox[2] / face_scale)
        metrics[f"{prefix}_normal_mean_x"] = float(normal_mean[0])
        metrics[f"{prefix}_normal_mean_y"] = float(normal_mean[1])
        metrics[f"{prefix}_normal_mean_z"] = float(normal_mean[2])
        metrics[f"{prefix}_normal_variance"] = float(normal_var / face_scale)
        metrics[f"{prefix}_depth_std_ratio"] = float(np.std(zone_verts[:, 2]) / face_scale)
        
        # Plane residual (how flat the zone is)
        if len(zone_verts) >= 3:
            try:
                centered = zone_verts - centroid
                _, _, vh = np.linalg.svd(centered, full_matrices=False)
                plane_normal = vh[-1]
                plane_normal = plane_normal / (np.linalg.norm(plane_normal) + 1e-8)
                residuals = np.abs(centered @ plane_normal)
                metrics[f"{prefix}_plane_residual_std_ratio"] = float(np.std(residuals) / face_scale)
                metrics[f"{prefix}_plane_residual_p95_ratio"] = float(np.percentile(residuals, 95) / face_scale)
            except np.linalg.LinAlgError:
                metrics[f"{prefix}_plane_residual_std_ratio"] = 0.0
                metrics[f"{prefix}_plane_residual_p95_ratio"] = 0.0
        
        # Convexity index (ratio of volume to convex hull volume approximation)
        if len(zone_verts) >= 4:
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(zone_verts)
                hull_volume = hull.volume
                bbox_volume = np.prod(bbox)
                if bbox_volume > 1e-10:
                    metrics[f"{prefix}_convexity_index"] = float(hull_volume / bbox_volume)
            except Exception:
                pass
    
    return metrics


def compute_bone_anchors(vertices: np.ndarray, annotation_groups: List[np.ndarray]) -> Dict[str, np.ndarray]:
    """Compute stable bone anchor points (centroids of bone zones)."""
    anchors = {}
    
    # Key bone zones (stable across age/expression)
    bone_zone_names = [
        "nose", "left_eyebrow", "right_eyebrow", 
        "left_eye", "right_eye"
    ]
    
    for zone_name in bone_zone_names:
        zone_verts = get_zone_vertices(vertices, annotation_groups, zone_name)
        if zone_verts.size > 0:
            anchors[zone_name] = zone_verts.mean(axis=0)
    
    return anchors


def compute_macro_metrics(vertices: np.ndarray, annotation_groups: List[np.ndarray],
                          face_scale: float, visible_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute macro-zone metrics (bone priority zones)."""
    metrics = {}
    
    if vertices.size == 0 or face_scale <= 0:
        return metrics
    
    # Nasion depth (nose bridge)
    nose_verts = get_zone_vertices(vertices, annotation_groups, "nose")
    if nose_verts.size > 0:
        metrics["bone_nasion_depth"] = float(nose_verts[:, 2].mean() / face_scale)
    
    # Orbital depths
    for side, eye_name in [("L", "left_eye"), ("R", "right_eye")]:
        eye_verts = get_zone_vertices(vertices, annotation_groups, eye_name)
        if eye_verts.size > 0:
            metrics[f"bone_orbit_{side}_depth"] = float(eye_verts[:, 2].mean() / face_scale)
    
    # Zygomatic width
    left_cheek = get_zone_vertices(vertices, annotation_groups, "skin")  # left part
    right_cheek = get_zone_vertices(vertices, annotation_groups, "skin")  # right part
    # Use overall skin zone span as proxy
    skin_verts = get_zone_vertices(vertices, annotation_groups, "skin")
    if skin_verts.size > 0:
        x_span = skin_verts[:, 0].max() - skin_verts[:, 0].min()
        metrics["bone_zygomatic_width"] = float(x_span / face_scale)
    
    # Gonial angle (jaw angle) - approximate from lower skin
    if skin_verts.size > 0:
        lower_skin = skin_verts[skin_verts[:, 1] > skin_verts[:, 1].mean()]  # lower half
        if len(lower_skin) > 10:
            left_lower = lower_skin[lower_skin[:, 0] < 0]
            right_lower = lower_skin[lower_skin[:, 0] > 0]
            if len(left_lower) > 5 and len(right_lower) > 5:
                left_jaw = left_lower[left_lower[:, 0].argmin()]
                right_jaw = right_lower[right_lower[:, 0].argmax()]
                chin = skin_verts[skin_verts[:, 1].argmax()]
                
                # Angle between left_jaw-chin-right_jaw
                v1 = left_jaw - chin
                v2 = right_jaw - chin
                v1 = v1 / (np.linalg.norm(v1) + 1e-8)
                v2 = v2 / (np.linalg.norm(v2) + 1e-8)
                angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
                metrics["bone_gonial_angle"] = float(np.degrees(angle))
    
    # Chin projection
    if skin_verts.size > 0:
        chin_tip = skin_verts[skin_verts[:, 1].argmax()]
        nose_bridge = nose_verts[nose_verts[:, 2].argmax()] if nose_verts.size > 0 else skin_verts[skin_verts[:, 2].argmax()]
        metrics["bone_chin_projection"] = float(np.linalg.norm(chin_tip - nose_bridge) / face_scale)
    
    return metrics


def compute_asymmetry(vertices: np.ndarray, visible_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute mirror asymmetry metrics."""
    metrics = {}
    
    if vertices.size == 0:
        return metrics
    
    # Mirror vertices across YZ plane
    mirrored = vertices.copy()
    mirrored[:, 0] = -mirrored[:, 0]
    
    # Align and compute residual
    if visible_mask is not None:
        valid = visible_mask
    else:
        valid = np.ones(len(vertices), dtype=bool)
    
    if valid.sum() < 10:
        return metrics
    
    v_valid = vertices[valid]
    m_valid = mirrored[valid]
    
    # Procrustes alignment
    try:
        # Center
        v_centered = v_valid - v_valid.mean(axis=0)
        m_centered = m_valid - m_valid.mean(axis=0)
        
        # SVD for rotation
        H = m_centered.T @ v_centered
        U, _, Vt = np.linalg.svd(H)
        R_mat = Vt.T @ U.T
        
        if np.linalg.det(R_mat) < 0:
            Vt[-1, :] *= -1
            R_mat = Vt.T @ U.T
        
        m_aligned = m_centered @ R_mat
        
        # Residual
        diff = np.linalg.norm(v_centered - m_aligned, axis=1)
        metrics["bone_asymmetry_x"] = float(diff.mean())
        metrics["bone_asymmetry_x_p95"] = float(np.percentile(diff, 95))
    except np.linalg.LinAlgError:
        pass
    
    return metrics


class GeometryExtractor:
    """Извлечение геометрических метрик из 3D реконструкции 3DDFA-V3."""

    def __init__(self):
        pass

    def extract(self, reconstruction: Dict[str, Any]) -> Dict[str, float]:
        """
        Извлекает все геометрические метрики из результата реконструкции.
        
        Args:
            reconstruction: Dict с данными реконструкции (vertices, annotation_groups, etc.)
            
        Returns:
            Словарь {имя_метрики: значение}
        """
        metrics = {}
        
        vertices = np.asarray(reconstruction.get("vertices", []), dtype=np.float32)
        vertices_canon = np.asarray(reconstruction.get("vertices_canonical", []), dtype=np.float32)
        annotation_groups = reconstruction.get("annotation_groups", [])
        visible_idx = reconstruction.get("visible_idx_renderer", None)
        
        # Use canonical vertices if available, else world
        if vertices_canon.size > 0:
            verts = vertices_canon
        elif vertices.size > 0:
            verts = vertices
        else:
            return metrics
        
        # Compute face scale (inter-zygomatic distance approximation)
        if verts.size > 0:
            x_span = verts[:, 0].max() - verts[:, 0].min()
            face_scale = max(x_span, 1.0)
        else:
            face_scale = 1.0
        
        # 1. Zone metrics (all 8 annotation zones)
        zone_metrics = compute_zone_metrics(verts, annotation_groups, face_scale, visible_idx)
        metrics.update(zone_metrics)
        
        # 2. Bone anchor metrics (stable zones)
        bone_metrics = compute_macro_metrics(verts, annotation_groups, face_scale, visible_idx)
        metrics.update(bone_metrics)
        
        # 3. Asymmetry metrics
        asym_metrics = compute_asymmetry(verts, visible_idx)
        metrics.update(asym_metrics)
        
        # 4. Mesh stats
        v_min = verts.min(axis=0)
        v_max = verts.max(axis=0)
        bbox = v_max - v_min
        metrics["mesh_bbox_width"] = float(bbox[0])
        metrics["mesh_bbox_height"] = float(bbox[1])
        metrics["mesh_bbox_depth"] = float(bbox[2])
        metrics["mesh_bbox_volume"] = float(np.prod(bbox))
        
        # Face scale
        metrics["face_scale"] = float(face_scale)
        
        # Symmetry
        x_coords = verts[:, 0]
        metrics["mesh_symmetry_x"] = float(abs(x_coords.mean()) / face_scale)
        
        # Vertex count
        metrics["mesh_vertex_count"] = float(len(verts))
        
        # Visible ratio
        if visible_idx is not None:
            metrics["visible_vertex_ratio"] = float(np.mean(visible_idx))
        
        # Expression params
        exp_params = reconstruction.get("payload", {}).get("exp_params", [])
        if len(exp_params) > 0:
            exp_np = np.asarray(exp_params)
            metrics["exp_magnitude"] = float(np.linalg.norm(exp_np))
            metrics["exp_jaw_open"] = float(abs(exp_np[0])) if len(exp_np) > 0 else 0.0
            metrics["exp_smile"] = float(max(abs(exp_np[1]), abs(exp_np[2]))) if len(exp_np) > 2 else 0.0
        
        # ID params (identity vector)
        id_params = reconstruction.get("payload", {}).get("id_params", [])
        if len(id_params) > 0:
            id_np = np.asarray(id_params)
            metrics["id_norm"] = float(np.linalg.norm(id_np))
            metrics["id_mean"] = float(np.mean(id_np))
            metrics["id_std"] = float(np.std(id_np))
        
        # Pose angles
        angles = reconstruction.get("angles_deg", np.zeros(3))
        metrics["pose_yaw"] = float(angles[1])
        metrics["pose_pitch"] = float(angles[0])
        metrics["pose_roll"] = float(angles[2])
        
        return metrics