"""
Строит forensic zone indices (17 зон) из 8 базовых annotation_groups 3DDFA-V3.

3DDFA-V3 даёт 8 зон:
  [0] right_eye, [1] left_eye, [2] right_eyebrow, [3] left_eyebrow,
  [4] nose, [5] up_lip, [6] down_lip, [7] skin

Мы разбиваем skin на анатомические подзоны используя пространственные
координаты вершин и 68-landmark keypoints.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional


# Anatomical zone names for forensic comparison
FORENSIC_ZONE_NAMES = [
    "nasion", "orbit_L", "orbit_R",
    "zygomatic_L", "zygomatic_R",
    "gonion_L", "gonion_R",
    "pogonion", "ramus_L", "ramus_R",
    "cheek_L", "cheek_R",
    "nasolabial_L", "nasolabial_R",
    "lip_upper", "lip_lower", "forehead",
]


def build_forensic_zone_indices(
    vertices: np.ndarray,
    annotation_groups: List[np.ndarray],
    landmarks_106: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Строит mapping forensic_zone_name -> vertex indices.

    Args:
        vertices: (N, 3) vertex positions in canonical space
        annotation_groups: list of 8 arrays with vertex indices
        landmarks_106: (106, 2) or (106, 3) optional landmarks for precise zone placement

    Returns:
        dict mapping zone name to np.ndarray of vertex indices
    """
    n_verts = len(vertices)
    zone_indices: Dict[str, np.ndarray] = {}

    # Extract basic zones
    skin_idx = np.asarray(annotation_groups[7], dtype=int) if len(annotation_groups) > 7 else np.array([], dtype=int)
    nose_idx = np.asarray(annotation_groups[4], dtype=int) if len(annotation_groups) > 4 else np.array([], dtype=int)
    right_eye_idx = np.asarray(annotation_groups[0], dtype=int) if len(annotation_groups) > 0 else np.array([], dtype=int)
    left_eye_idx = np.asarray(annotation_groups[1], dtype=int) if len(annotation_groups) > 1 else np.array([], dtype=int)
    up_lip_idx = np.asarray(annotation_groups[5], dtype=int) if len(annotation_groups) > 5 else np.array([], dtype=int)
    down_lip_idx = np.asarray(annotation_groups[6], dtype=int) if len(annotation_groups) > 6 else np.array([], dtype=int)

    if skin_idx.size == 0:
        # Fallback: all vertices
        return {name: np.array([], dtype=int) for name in FORENSIC_ZONE_NAMES}

    # Compute face coordinate bounds from skin vertices
    skin_verts = vertices[skin_idx]
    face_min = skin_verts.min(axis=0)
    face_max = skin_verts.max(axis=0)
    face_center = (face_min + face_max) / 2.0
    face_range = face_max - face_min
    face_range[face_range == 0] = 1.0  # avoid division by zero

    # Normalize skin vertex positions to [0, 1] range
    skin_norm = (skin_verts - face_min) / face_range

    # Key Y-axis thresholds (normalized 0=top, 1=bottom of face)
    # Forehead: top 25% of face
    # Mid-face: 25-60%
    # Lower face: 60-100%
    forehead_y_thresh = 0.25
    midface_y_thresh = 0.60

    # Key X-axis thresholds (normalized 0=left, 1=right)
    center_x = 0.5
    side_x_thresh = 0.35  # Below this = left side, above 0.65 = right side

    # --- 1. NOSE (nasion) ---
    # Nasion = top of nose bridge, use nose zone vertices in upper region
    if nose_idx.size > 0:
        nose_verts = vertices[nose_idx]
        nose_norm = (nose_verts - face_min) / face_range
        # Nasion: nose vertices in top 40% of face
        nasion_mask = nose_norm[:, 1] < 0.40
        zone_indices["nasion"] = nose_idx[nasion_mask] if nasion_mask.any() else nose_idx[:len(nose_idx)//3]
    else:
        zone_indices["nasion"] = np.array([], dtype=int)

    # --- 2. ORBITS ---
    zone_indices["orbit_R"] = right_eye_idx
    zone_indices["orbit_L"] = left_eye_idx

    # --- 3. ZYGOMATIC (cheekbones) ---
    # Skin vertices in mid-face, lateral position
    zyg_mask_L = (skin_norm[:, 1] > 0.30) & (skin_norm[:, 1] < 0.55) & (skin_norm[:, 0] < side_x_thresh)
    zyg_mask_R = (skin_norm[:, 1] > 0.30) & (skin_norm[:, 1] < 0.55) & (skin_norm[:, 0] > (1.0 - side_x_thresh))
    zone_indices["zygomatic_L"] = skin_idx[zyg_mask_L]
    zone_indices["zygomatic_R"] = skin_idx[zyg_mask_R]

    # --- 4. GONION (jaw angles) ---
    # Skin vertices in lower face, extreme lateral
    gon_mask_L = (skin_norm[:, 1] > 0.70) & (skin_norm[:, 0] < 0.20)
    gon_mask_R = (skin_norm[:, 1] > 0.70) & (skin_norm[:, 0] > 0.80)
    zone_indices["gonion_L"] = skin_idx[gon_mask_L]
    zone_indices["gonion_R"] = skin_idx[gon_mask_R]

    # --- 5. POGONION (chin tip) ---
    # Skin vertices at bottom center
    pog_mask = (skin_norm[:, 1] > 0.80) & (skin_norm[:, 0] > 0.35) & (skin_norm[:, 0] < 0.65)
    zone_indices["pogonion"] = skin_idx[pog_mask]

    # --- 6. RAMUS (jawline sides) ---
    # Skin vertices along jawline, between gonion and chin
    ram_mask_L = (skin_norm[:, 1] > 0.65) & (skin_norm[:, 1] < 0.85) & (skin_norm[:, 0] > 0.15) & (skin_norm[:, 0] < 0.40)
    ram_mask_R = (skin_norm[:, 1] > 0.65) & (skin_norm[:, 1] < 0.85) & (skin_norm[:, 0] > 0.60) & (skin_norm[:, 0] < 0.85)
    zone_indices["ramus_L"] = skin_idx[ram_mask_L]
    zone_indices["ramus_R"] = skin_idx[ram_mask_R]

    # --- 7. CHEEKS (soft tissue, below cheekbones) ---
    # Skin vertices in mid-lower face, medial to zygomatic
    cheek_mask_L = (skin_norm[:, 1] > 0.45) & (skin_norm[:, 1] < 0.70) & (skin_norm[:, 0] > 0.15) & (skin_norm[:, 0] < 0.40)
    cheek_mask_R = (skin_norm[:, 1] > 0.45) & (skin_norm[:, 1] < 0.70) & (skin_norm[:, 0] > 0.60) & (skin_norm[:, 0] < 0.85)
    zone_indices["cheek_L"] = skin_idx[cheek_mask_L]
    zone_indices["cheek_R"] = skin_idx[cheek_mask_R]

    # --- 8. NASOLABIAL (fold area) ---
    # Skin vertices along nasolabial fold: medial mid-face
    naso_mask_L = (skin_norm[:, 1] > 0.50) & (skin_norm[:, 1] < 0.75) & (skin_norm[:, 0] > 0.30) & (skin_norm[:, 0] < 0.45)
    naso_mask_R = (skin_norm[:, 1] > 0.50) & (skin_norm[:, 1] < 0.75) & (skin_norm[:, 0] > 0.55) & (skin_norm[:, 0] < 0.70)
    zone_indices["nasolabial_L"] = skin_idx[naso_mask_L]
    zone_indices["nasolabial_R"] = skin_idx[naso_mask_R]

    # --- 9. LIPS ---
    zone_indices["lip_upper"] = up_lip_idx
    zone_indices["lip_lower"] = down_lip_idx

    # --- 10. FOREHEAD ---
    # Skin vertices in top 25% of face
    forehead_mask = skin_norm[:, 1] < forehead_y_thresh
    zone_indices["forehead"] = skin_idx[forehead_mask]

    # Ensure all zones are non-empty (fallback to nearest skin vertices if empty)
    for zone_name in FORENSIC_ZONE_NAMES:
        if zone_indices.get(zone_name) is None or len(zone_indices[zone_name]) == 0:
            # Fallback: use closest skin vertices to expected zone center
            zone_indices[zone_name] = _fallback_zone(skin_idx, skin_verts, zone_name, face_center, face_range)

    return zone_indices


def _fallback_zone(
    skin_idx: np.ndarray,
    skin_verts: np.ndarray,
    zone_name: str,
    face_center: np.ndarray,
    face_range: np.ndarray,
) -> np.ndarray:
    """Fallback: pick closest 5% of skin vertices to expected zone center."""
    # Approximate zone centers in normalized coordinates
    zone_centers_norm = {
        "nasion": (0.5, 0.35, 0.5),
        "orbit_L": (0.35, 0.38, 0.5),
        "orbit_R": (0.65, 0.38, 0.5),
        "zygomatic_L": (0.25, 0.42, 0.5),
        "zygomatic_R": (0.75, 0.42, 0.5),
        "gonion_L": (0.10, 0.80, 0.5),
        "gonion_R": (0.90, 0.80, 0.5),
        "pogonion": (0.50, 0.90, 0.5),
        "ramus_L": (0.25, 0.75, 0.5),
        "ramus_R": (0.75, 0.75, 0.5),
        "cheek_L": (0.30, 0.58, 0.5),
        "cheek_R": (0.70, 0.58, 0.5),
        "nasolabial_L": (0.38, 0.62, 0.5),
        "nasolabial_R": (0.62, 0.62, 0.5),
        "lip_upper": (0.50, 0.68, 0.5),
        "lip_lower": (0.50, 0.73, 0.5),
        "forehead": (0.50, 0.12, 0.5),
    }

    center_norm = np.array(zone_centers_norm.get(zone_name, (0.5, 0.5, 0.5)))
    target = skin_center + center_norm * face_range

    dists = np.linalg.norm(skin_verts - target, axis=1)
    n_pick = max(50, len(skin_idx) // 20)  # ~5% of skin vertices
    closest = np.argsort(dists)[:n_pick]
    return skin_idx[closest]


def get_zone_types() -> Dict[str, str]:
    """Returns zone type mapping: 'bone' or 'soft'."""
    return {
        "nasion": "bone",
        "orbit_L": "bone",
        "orbit_R": "bone",
        "zygomatic_L": "bone",
        "zygomatic_R": "bone",
        "gonion_L": "bone",
        "gonion_R": "bone",
        "pogonion": "bone",
        "ramus_L": "bone",
        "ramus_R": "bone",
        "cheek_L": "soft",
        "cheek_R": "soft",
        "nasolabial_L": "soft",
        "nasolabial_R": "soft",
        "lip_upper": "soft",
        "lip_lower": "soft",
        "forehead": "soft",
    }


def get_bone_vertex_indices(
    vertices: np.ndarray,
    annotation_groups: list,
    landmarks_106: np.ndarray | None = None,
) -> np.ndarray:
    """Returns union of all bone zone vertex indices for Procrustes alignment."""
    zone_indices = build_forensic_zone_indices(vertices, annotation_groups, landmarks_106)
    zone_types = get_zone_types()
    bone_sets = [zone_indices[name] for name in FORENSIC_ZONE_NAMES if zone_types.get(name) == "bone" and name in zone_indices and len(zone_indices[name]) > 0]
    if not bone_sets:
        return np.arange(len(vertices), dtype=int)
    return np.unique(np.concatenate(bone_sets))
