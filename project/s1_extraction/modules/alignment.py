from __future__ import annotations

import numpy as np
from .types import AlignmentResult

# Импорт типов из бэкенда
try:
    from backend.pipeline.types import AlignmentResult as BackendAlignmentResult
except ImportError:
    pass

_CANONICAL_YAW_BY_VIEW_GROUP: dict[str, float] = {
    "frontal": 0.0,
    "left_threequarter_light": -22.5,
    "right_threequarter_light": 22.5,
    "left_threequarter_mid": -45.0,
    "right_threequarter_mid": 45.0,
    "left_threequarter_deep": -67.5,
    "right_threequarter_deep": 67.5,
    "left_profile": -90.0,
    "right_profile": 90.0,
}

CANONICAL_YAW_BY_VIEW_GROUP = _CANONICAL_YAW_BY_VIEW_GROUP


def canonical_angles_deg_for_bucket(view_group: str) -> np.ndarray:
    """Целевые углы после canonical: pitch=0, roll=0, yaw=bucket."""
    target_yaw = float(_CANONICAL_YAW_BY_VIEW_GROUP.get(view_group, 0.0))
    return np.array([0.0, target_yaw, 0.0], dtype=np.float32)


def canonical_angles_deg_preserve_pose(angles_deg: np.ndarray, view_group: str) -> np.ndarray:
    """Устаревшее: сохранял pitch/roll. Не используется в compare/extract."""
    pitch, yaw, roll = angles_deg
    target_yaw = float(_CANONICAL_YAW_BY_VIEW_GROUP.get(view_group, 0.0))
    return np.array([float(pitch), target_yaw, float(roll)], dtype=np.float32)

def rigid_umeyama(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray | None = None,
    allow_scale: bool = False,
) -> AlignmentResult:
    """
    [GEOM-02] Weighted Rigid Umeyama Alignment with Iterative Robust Outlier Rejection.
    """
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != 3 or target.shape[1] != 3:
        raise ValueError(f"source/target must be (N,3), got {source.shape} / {target.shape}")
    if source.shape[0] != target.shape[0]:
        raise ValueError(f"source/target length mismatch: {source.shape[0]} vs {target.shape[0]}")
    # ИСПРАВЛЕНИЕ №8: Требуется минимум 4 точки для избежания вырожденности ковариации
    if source.shape[0] < 4:
        raise ValueError("source/target must have at least 4 points for reliable 3D alignment")

    if weights is None:
        weights = np.ones(len(source), dtype=np.float32)

    # Iterative outlier rejection (RANSAC-like) to solve Errors 9 & 10 (expression shifts and shadow noise)
    active_weights = weights.copy()
    robust_iterations = 4
    
    for iter_idx in range(robust_iterations):
        weight_sum = float(np.sum(active_weights))
        if weight_sum <= 1e-8:
            # Fallback to standard weights if robust weights collapse
            active_weights = weights.copy()
            weight_sum = float(np.sum(active_weights))

        w = active_weights / (weight_sum + 1e-8)
        w = w[:, np.newaxis]

        source_mean = np.sum(source * w, axis=0)
        target_mean = np.sum(target * w, axis=0)

        centered_source = source - source_mean
        centered_target = target - target_mean

        m = centered_source.T @ (w * centered_target)
        
        # [ITER-1.1] Rank Check Guard
        if np.linalg.matrix_rank(m) < 3:
            raise ValueError(
                f"Degenerate alignment covariance (rank={np.linalg.matrix_rank(m)}) "
                f"for {source.shape[0]} shared points"
            )

        u, s, vh = np.linalg.svd(m)

        d = np.linalg.det(u @ vh)
        sign_matrix = np.diag([1.0, 1.0, np.sign(d)])
        rotation = u @ sign_matrix @ vh

        if allow_scale:
            var_source = np.sum(active_weights.flatten() * np.sum(centered_source**2, axis=1))
            if var_source > 1e-8:
                scale = float((s[0] + s[1] + s[2] * np.sign(d)) / var_source)
            else:
                scale = 1.0
        else:
            scale = 1.0

        translation = target_mean - scale * (source_mean @ rotation)
        source_aligned = scale * (source @ rotation) + translation
        
        # Calculate residuals
        residuals = np.linalg.norm(source_aligned - target, axis=1)
        
        if iter_idx < robust_iterations - 1:
            # Prune outliers (residuals > 2.5 * MAD)
            med = np.median(residuals)
            mad = np.median(np.abs(residuals - med))
            std_est = 1.4826 * mad + 1e-6
            inliers = residuals <= (2.5 * std_est)
            
            # Keep at least 60% of points to ensure stability
            if np.sum(inliers) < len(source) * 0.6:
                threshold = np.percentile(residuals, 60)
                inliers = residuals <= threshold
                
            active_weights = weights * inliers

    w_flat = np.asarray(weights, dtype=np.float64).reshape(-1)
    norm_before = np.linalg.norm(source - target, axis=1) * w_flat
    norm_after = np.linalg.norm(source_aligned - target, axis=1) * w_flat
    residual_before_sum = float(np.sum(norm_before))
    residual_after_sum = float(np.sum(norm_after))
    w_den = max(float(np.sum(w_flat)), 1e-8)
    residual_before = float(residual_before_sum / w_den)
    residual_after = float(residual_after_sum / w_den)

    return AlignmentResult(
        rotation=rotation,
        translation=translation,
        scale=scale,
        source_aligned=source_aligned,
        residual_before=residual_before,
        residual_after=residual_after,
        residual_before_sum=residual_before_sum,
        residual_after_sum=residual_after_sum,
    )

def euler_to_rotation_matrix(angles_rad: np.ndarray) -> np.ndarray:
    """
    Конвертирует углы Эйлера в матрицу вращения (ZYX convention).
    Ожидается формат 3DDFA_v3: [pitch, yaw, roll] в радианах.
    R = Rz(roll) @ Ry(yaw) @ Rx(pitch)
    """
    pitch, yaw, roll = angles_rad

    cx, sx = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)
    cz, sz = np.cos(roll),  np.sin(roll)

    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    
    return rot_z @ rot_y @ rot_x

def align_canonical_pair_for_view_group(
    vertices_a: np.ndarray,
    angles_a: np.ndarray,
    translation_a: np.ndarray,
    vertices_b: np.ndarray,
    angles_b: np.ndarray,
    translation_b: np.ndarray,
    view_group: str | None,
    shared_idx: np.ndarray,
    weights: np.ndarray,
) -> dict[str, np.ndarray | AlignmentResult]:
    """
    [ITER-1.1] Aligns a pair of faces to a canonical pose.
    Uses average pitch and roll of the pair for better stability.
    """
    # 1. Determine canonical yaw
    target_yaw = _CANONICAL_YAW_BY_VIEW_GROUP.get((view_group or "").strip(), 0.0)
    
    # 2. Use average pitch and roll from the pair instead of 0
    avg_pitch = (angles_a[0] + angles_b[0]) / 2.0
    avg_roll = (angles_a[2] + angles_b[2]) / 2.0
    
    target_angles_deg = np.array([avg_pitch, target_yaw, avg_roll], dtype=np.float32)
    target_angles_rad = np.deg2rad(target_angles_deg)
    
    # Rotation matrices for current poses
    R_a = euler_to_rotation_matrix(np.deg2rad(angles_a))
    R_b = euler_to_rotation_matrix(np.deg2rad(angles_b))
    
    # Rotation matrix for target pose
    R_target = euler_to_rotation_matrix(target_angles_rad)
    
    # Alignment rotations
    R_align_a = R_a @ R_target.T
    R_align_b = R_b @ R_target.T
    
    # Center and align (mesh centroid, not camera translation parameter)
    centroid_a = vertices_a.mean(axis=0)
    centroid_b = vertices_b.mean(axis=0)
    va_canon = (vertices_a - centroid_a) @ R_align_a
    vb_canon = (vertices_b - centroid_b) @ R_align_b
    
    # Final rigid alignment. Scale fitting is forbidden for forensic/heatmap
    # interpretation; otherwise a larger face can be shrunk into a smaller one.
    alignment = rigid_umeyama(va_canon[shared_idx], vb_canon[shared_idx], weights=weights, allow_scale=False)
    aligned_a = (va_canon * alignment.scale) @ alignment.rotation + alignment.translation
    
    return {
        "target_angles_deg": target_angles_deg,
        "vertices_a_canonical": va_canon,
        "vertices_b_canonical": vb_canon,
        "alignment_mode": "forensic_no_scale",
        "alignment": alignment,
        "vertices_a_aligned": aligned_a,
    }

def gpa_unit_scale(points: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """
    [GEOM-01] Generalized Procrustes Analysis: Centering and Unit Scaling.
    Returns (centered_unit_points, scale, centroid).
    """
    if points.size == 0:
        return points, 1.0, np.zeros(3)
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    scale = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
    if scale < 1e-8:
        return centered, 1.0, centroid
    return centered / scale, scale, centroid

def euler_to_rotation_matrix_applied(angles_rad: np.ndarray) -> np.ndarray:
    """Матрица, которую 3DDFA реально применяет к вершинам: pts @ R (см. recon.compute_rotation)."""
    return euler_to_rotation_matrix(angles_rad).T


def canonicalize_vertices_for_bucket(
    vertices: np.ndarray,
    angles_deg: np.ndarray,
    view_group: str,
    rotation_matrix_applied: np.ndarray | None = None,
) -> np.ndarray:
    """
    Приводит меш к bucket pose: pitch=0, roll=0, yaw=целевой для view_group.
    3DDFA: vertices = base @ R_applied — undo через R_applied.T, затем target.

    rotation_matrix_applied: фактическая матрица из 3DDFA (recon.rotation_matrix).
    Без неё — fallback на euler; при большом |pitch| euler ≠ compute_rotation.
    """
    target_yaw = float(_CANONICAL_YAW_BY_VIEW_GROUP.get(view_group, 0.0))
    r_tgt = euler_to_rotation_matrix_applied(
        np.deg2rad(np.array([0.0, target_yaw, 0.0], dtype=np.float64))
    )
    if rotation_matrix_applied is not None:
        r_cur = np.asarray(rotation_matrix_applied, dtype=np.float64).reshape(3, 3)
    else:
        pitch, yaw, roll = angles_deg
        r_cur = euler_to_rotation_matrix_applied(
            np.deg2rad(np.array([pitch, yaw, roll], dtype=np.float64))
        )
    # raw = base @ r_cur  →  raw @ r_cur.T @ r_tgt = base @ r_tgt
    r_align = r_cur.T @ r_tgt

    centroid = vertices.mean(axis=0)
    aligned_vertices = (vertices - centroid) @ r_align + centroid

    return aligned_vertices


def rigid_umeyama_robust(src: np.ndarray, dst: np.ndarray, allow_scale: bool = True) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Legacy Umeyama helper for align_and_score_gpa (unused in compare/calibration path).
    Production forensic alignment uses rigid_umeyama() with allow_scale=False (A-01).
    Исправлена проблема сингулярности SVD и двойного масштабирования.
    
    :param src: Массив вершин A (N, 3)
    :param dst: Массив вершин B (N, 3)
    :return: (Rotation Matrix, Translation Vector, Scale)
    """
    assert src.shape == dst.shape, "Массивы вершин должны совпадать по размеру"
    num_pts = src.shape[0]

    # 1. Центрирование
    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean

    # Расчет дисперсии (для масштаба)
    src_var = np.mean(np.sum(src_c ** 2, axis=1))

    # 2. Матрица ковариации
    H = (src_c.T @ dst_c) / num_pts

    # ЗАЩИТА ОТ СИНГУЛЯРНОСТИ (Исправление бага M-01)
    if np.linalg.matrix_rank(H) < 3:
        raise np.linalg.LinAlgError("Матрица ковариации вырождена (rank < 3). Сравнение невозможно, точки коллинеарны.")

    # 3. SVD разложение
    U, S, Vt = np.linalg.svd(H)

    # 4. Расчет матрицы вращения с защитой от отражения (Reflection)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    # 5. Расчет масштаба (Только один раз!)
    scale = 1.0
    if allow_scale:
        scale = np.sum(S) / (src_var + 1e-10)

    # 6. Вектор трансляции
    t = dst_mean - scale * (src_mean @ R.T)

    return R, t, scale


def align_and_score_gpa(verts_a: np.ndarray, verts_b: np.ndarray, mask: np.ndarray):
    """
    Обобщенный Прокрустов Анализ только по валидным (shared) костным ориентирам.
    """
    valid_a = verts_a[mask]
    valid_b = verts_b[mask]
    
    # Вызываем исправленный Умеяма с разрешением масштабирования
    R, t, scale = rigid_umeyama_robust(valid_a, valid_b, allow_scale=False)
    
    # Применяем трансформацию ко ВСЕМ вершинам A
    verts_a_aligned = (scale * (verts_a @ R.T)) + t
    
    # Теперь сырая ошибка считается в правильном метрическом пространстве
    raw_errors = np.linalg.norm(verts_a_aligned - verts_b, axis=1)
    return verts_a_aligned, raw_errors


class AlignmentEngine:
    """
    Движок выравнивания 3D мешей.
    Обёртка над функциями alignment.py для совместимости с s1_extraction.modules.
    """
    
    def align(self, reconstruction: dict, pose_info) -> dict:
        """
        Выравнивает реконструкцию по информации о позе.
        
        Args:
            reconstruction: Словарь с данными реконструкции (vertices, faces, landmarks)
            pose_info: Информация о позе (yaw, pitch, roll, bucket)
            
        Returns:
            Выровненная реконструкция
        """
        # Если нет данных о вершинах, возвращаем как есть
        if 'vertices' not in reconstruction or not reconstruction['vertices']:
            return reconstruction
        
        vertices = np.array(reconstruction['vertices'])
        if vertices.size == 0:
            return reconstruction
        
        # Получаем bucket из pose_info
        bucket = getattr(pose_info, 'bucket', 'unknown') if hasattr(pose_info, 'bucket') else 'unknown'
        
        # Получаем целевые углы для canonical
        target_angles = canonical_angles_deg_for_bucket(bucket)
        
        # Применяем выравнивание (пока заглушка - просто возвращаем как есть)
        # В будущем здесь будет вызов align_canonical_pair_for_view_group
        return reconstruction
