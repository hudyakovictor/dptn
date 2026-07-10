from __future__ import annotations

import numpy as np
from typing import Dict


class DifferenceHeatmap:
    """Генерирует per-vertex heatmap с зональными порогами."""
    
    # Пороги в мм
    THRESHOLDS = {
        "nasion": 1.5,
        "orbit_L": 1.5,
        "orbit_R": 1.5,
        "zygomatic_L": 2.0,
        "zygomatic_R": 2.0,
        "gonion_L": 2.0,
        "gonion_R": 2.0,
        "pogonion": 2.0,
        "ramus_L": 2.5,
        "ramus_R": 2.5,
        "cheek_L": 5.0,
        "cheek_R": 5.0,
        "nasolabial_L": 4.0,
        "nasolabial_R": 4.0,
        "lip_upper": 6.0,
        "lip_lower": 6.0,
        "forehead": 4.0,
    }
    
    DEFAULT_THRESHOLD = 3.0
    
    def __init__(self, face_scale: float = 1.0):
        self.face_scale = face_scale
    
    def compute(self, verts_a: np.ndarray, verts_b: np.ndarray,
                zone_indices: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Returns: heatmap (N,) where 0=blue, 1=red
        """
        diff = np.linalg.norm(verts_a - verts_b, axis=1) * self.face_scale  # mm
        heat = np.zeros_like(diff)
        
        classified = np.zeros(len(diff), dtype=bool)
        
        for zone_name, indices in zone_indices.items():
            threshold = self.THRESHOLDS.get(zone_name, self.DEFAULT_THRESHOLD)
            zone_diff = diff[indices]
            zone_heat = np.clip(zone_diff / threshold, 0, 1)
            heat[indices] = zone_heat
            classified[indices] = True
        
        # Unclassified vertices
        unclassified = ~classified
        if unclassified.any():
            heat[unclassified] = np.clip(diff[unclassified] / self.DEFAULT_THRESHOLD, 0, 1)
        
        return heat
    
    def to_colormap(self, heat: np.ndarray,
                    blue_threshold: float = 0.25,
                    green_threshold: float = 0.5,
                    red_threshold: float = 0.75) -> np.ndarray:
        """
        Преобразует scalar heat в RGB colormap.
        0.0–0.25: синий→голубой
        0.25–0.50: голубой→зелёный
        0.50–0.75: зелёный→красный
        0.75+: тёмно-красный
        """
        rgb = np.zeros((len(heat), 3))
        
        for i, h in enumerate(heat):
            if h < blue_threshold:
                t = h / blue_threshold
                rgb[i] = [0, t, 1]
            elif h < green_threshold:
                t = (h - blue_threshold) / (green_threshold - blue_threshold)
                rgb[i] = [0, 1, 1 - t]
            elif h < red_threshold:
                t = (h - green_threshold) / (red_threshold - green_threshold)
                rgb[i] = [t, 1 - t, 0]
            else:
                t = min(1.0, (h - red_threshold) / 0.25)
                rgb[i] = [0.5 + 0.5 * t, 0, 0]
        
        return rgb
    
    def count_zone_violations(self, heat: np.ndarray, zone_indices: Dict[str, np.ndarray],
                              zone_types: Dict[str, str], heat_threshold: float = 0.5) -> int:
        """Считает, сколько костных зон превышают порог."""
        violations = 0
        for zone_name, indices in zone_indices.items():
            if zone_types.get(zone_name) == "bone":
                if heat[indices].mean() > heat_threshold:
                    violations += 1
        return violations