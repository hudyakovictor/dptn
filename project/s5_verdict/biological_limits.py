from __future__ import annotations

from datetime import timedelta
from typing import List, Dict, Tuple
import numpy as np


class BiologicalConstraintChecker:
    """Проверяет биологическую возможность изменений."""
    
    CONSTRAINTS = {
        # Хирургические лимиты (минимальное время на заживление + видимость)
        "rhinoplasty": {
            "min_gap": timedelta(days=180),
            "affected_metrics": ["bone_nasion_depth", "landmark_nose_chin_distance", "nose_width", "nose_tip_projection"],
            "max_change_mm": 5.0,
            "description": "Ринопластика требует минимум 6 месяцев на заживление. Форма носа не может измениться кардинально за меньший срок.",
        },
        "facelift": {
            "min_gap": timedelta(days=90),
            "affected_metrics": ["soft_cheek_volume", "soft_nasolabial_depth", "jawline_definition", "marionette_depth"],
            "max_change_mm": 8.0,
            "description": "Подтяжка лица (SMAS) требует 3+ месяцев на спад отека. Резкие изменения мягких тканей раньше — невозможны.",
        },
        "blepharoplasty": {
            "min_gap": timedelta(days=60),
            "affected_metrics": ["landmark_eye_width_L", "landmark_eye_width_R", "eyelid_height_L", "eyelid_height_R"],
            "max_change_mm": 4.0,
            "description": "Блефаропластика: минимум 2 месяца на заживление век.",
        },
        "zygomatic_implant": {
            "min_gap": timedelta(days=365),
            "affected_metrics": ["bone_zygomatic_width", "face_scale", "zygomatic_arch_height_L", "zygomatic_arch_height_R"],
            "max_change_mm": 10.0,
            "description": "Импланты скул требуют до года на оссеоинтеграцию и спад отека.",
        },
        "chin_implant": {
            "min_gap": timedelta(days=180),
            "affected_metrics": ["bone_chin_projection", "chin_soft_volume", "menton_position"],
            "max_change_mm": 8.0,
            "description": "Имплант подбородка: 6+ месяцев на финальную форму.",
        },
        
        # Естественные лимиты
        "bone_growth_adult": {
            "min_gap": timedelta(days=365),
            "affected_metrics": ["bone_zygomatic_width", "bone_nasion_depth", "bone_gonial_angle", "bone_mandible_width", "bone_interorbital_distance"],
            "max_change_mm": 0.5,  # У взрослого кости не растут
            "description": "Костные структуры взрослого человека (старше 25 лет) практически не изменяются. Изменение >0.5 мм за год невозможно без травмы/операции.",
        },
        "asymmetry_inversion": {
            "min_gap": timedelta(days=365*5),
            "affected_metrics": ["bone_asymmetry_x", "bone_asymmetry_y", "bone_asymmetry_z"],
            "max_change_mm": 999,  # Любое изменение знака — подозрительно
            "description": "Инверсия асимметрии (левая скуловая кость вдруг 'длиннее' правой) невозможна естественным путём.",
        },
    }
    
    def check(self, photo_a: Dict, photo_b: Dict, 
              metrics_a: Dict, metrics_b: Dict) -> List[Dict]:
        """
        photo: {date, age_years, ...}
        metrics: {metric_name: value}
        """
        flags = []
        gap = abs((photo_a["date"] - photo_b["date"]).days)
        
        for constraint_name, constraint in self.CONSTRAINTS.items():
            for metric_name in constraint["affected_metrics"]:
                if metric_name not in metrics_a or metric_name not in metrics_b:
                    continue
                
                val_a = metrics_a[metric_name]
                val_b = metrics_b[metric_name]
                delta = abs(val_a - val_b)
                
                # Проверка 1: Слишком быстрое изменение
                if gap < constraint["min_gap"].days and delta > constraint["max_change_mm"]:
                    flags.append({
                        "type": "BIOLOGICALLY_IMPOSSIBLE",
                        "constraint": constraint_name,
                        "metric": metric_name,
                        "delta_mm": float(delta),
                        "gap_days": gap,
                        "min_required_days": constraint["min_gap"].days,
                        "description": constraint["description"],
                        "severity": "CRITICAL" if delta > constraint["max_change_mm"] * 2 else "HIGH",
                    })
                
                # Проверка 2: Инверсия асимметрии
                if constraint_name == "asymmetry_inversion":
                    if (val_a > 2.0 and val_b < -2.0) or (val_a < -2.0 and val_b > 2.0):
                        flags.append({
                            "type": "ASYMMETRY_INVERSION",
                            "constraint": constraint_name,
                            "metric": metric_name,
                            "val_a": float(val_a),
                            "val_b": float(val_b),
                            "description": constraint["description"],
                            "severity": "CRITICAL",
                        })
        
        return flags