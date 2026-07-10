from __future__ import annotations

import numpy as np
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class ExpressionAnalysis:
    """Результат анализа мимики через 3DDFA exp-вектор."""
    flags: Dict[str, bool]
    intensities: Dict[str, float]
    neutral_score: float
    expression_label: str
    excluded_zones: List[str]


class ExpressionAnalyzer3D:
    """Анализирует expression через 3DDFA exp-параметры (29-D)."""
    
    # Индексы exp-параметров, отвечающих за конкретные движения
    # (уточнить под реальную модель 3DDFA-V3)
    EXP_INDICES = {
        "jaw_open": [0, 1, 2],
        "smile_L": [3, 4, 5],
        "smile_R": [6, 7, 8],
        "brow_raise": [9, 10, 11],
        "brow_frown": [12, 13, 14],
        "eye_squint": [15, 16, 17],
        "lip_pucker": [18, 19, 20],
        "nostril_dilate": [21, 22, 23],
    }
    
    # Адаптивные пороги (в std от нейтрального)
    THRESHOLDS = {
        "jaw_open": 1.5,
        "smile": 1.2,
        "brow_raise": 1.0,
        "brow_frown": 1.0,
    }
    
    # Зоны лица для исключения при разных мимиках
    ZONE_EXCLUSIONS = {
        "jaw_open": ["lip_upper", "lip_lower", "chin_soft", "nasolabial_L", "nasolabial_R"],
        "smile": ["lip_upper", "lip_lower", "cheek_L", "cheek_R", "nasolabial_L", "nasolabial_R"],
        "brow_raise": ["forehead", "brow_L", "brow_R"],
        "brow_frown": ["glabella", "brow_L", "brow_R"],
    }
    
    def analyze(self, exp_vector: np.ndarray, face_scale: float = 1.0) -> ExpressionAnalysis:
        """
        face_scale: масштаб лица (межзрачковое расстояние) для нормализации (опционально).
        3DDFA exp параметры обычно в диапазоне [-1, 1], нормализация не требуется.
        """
        flags = {}
        intensities = {}
        
        # 3DDFA exp параметры уже в нормализованных координатах
        exp_norm = exp_vector
        
        # Jaw open
        jaw_intensity = np.linalg.norm(exp_norm[self.EXP_INDICES["jaw_open"]])
        intensities["jaw_open"] = float(jaw_intensity)
        flags["jaw_open"] = jaw_intensity > self.THRESHOLDS["jaw_open"]
        
        # Smile (симметричный или асимметричный)
        smile_L = np.linalg.norm(exp_norm[self.EXP_INDICES["smile_L"]])
        smile_R = np.linalg.norm(exp_norm[self.EXP_INDICES["smile_R"]])
        smile_max = max(smile_L, smile_R)
        smile_asym = abs(smile_L - smile_R) / (smile_max + 1e-6)
        
        intensities["smile"] = float(smile_max)
        intensities["smile_asymmetry"] = float(smile_asym)
        flags["smile"] = smile_max > self.THRESHOLDS["smile"]
        flags["asymmetric_smile"] = smile_asym > 0.3 and smile_max > 0.8
        
        # Brows
        brow_raise = np.linalg.norm(exp_norm[self.EXP_INDICES["brow_raise"]])
        brow_frown = np.linalg.norm(exp_norm[self.EXP_INDICES["brow_frown"]])
        intensities["brow_raise"] = float(brow_raise)
        intensities["brow_frown"] = float(brow_frown)
        flags["brow_raise"] = brow_raise > self.THRESHOLDS["brow_raise"]
        flags["brow_frown"] = brow_frown > self.THRESHOLDS["brow_frown"]
        
        # Neutral score
        neutral_score = np.linalg.norm(exp_norm)
        flags["is_neutral"] = neutral_score < 0.5 and not any([
            flags["jaw_open"], flags["smile"], flags["brow_raise"], flags["brow_frown"]
        ])
        
        # Excluded zones
        excluded = self.get_excluded_zones(flags)
        
        return ExpressionAnalysis(
            flags=flags,
            intensities=intensities,
            neutral_score=float(neutral_score),
            expression_label=self._label(flags),
            excluded_zones=excluded,
        )
    
    def _label(self, flags: Dict) -> str:
        if flags.get("is_neutral"):
            return "neutral"
        active = [k for k, v in flags.items() if v and k != "is_neutral"]
        return "+".join(active) if active else "undefined"
    
    def get_excluded_zones(self, flags: Dict[str, bool]) -> List[str]:
        """Возвращает зоны, которые нужно исключить из анализа."""
        excluded = []
        
        if flags.get("jaw_open"):
            excluded.extend(["lip_upper", "lip_lower", "chin_soft", "nasolabial_L", "nasolabial_R"])
        
        if flags.get("smile"):
            excluded.extend(["lip_upper", "lip_lower", "cheek_L", "cheek_R", "nasolabial_L", "nasolabial_R"])
        
        if flags.get("brow_raise"):
            excluded.extend(["forehead", "brow_L", "brow_R"])
        
        if flags.get("brow_frown"):
            excluded.extend(["glabella", "brow_L", "brow_R"])
        
        # Убираем дубликаты
        return list(set(excluded))
    
    def get_zone_weights(self, flags: Dict[str, bool]) -> Dict[str, float]:
        """
        Возвращает веса зон для взвешенного сравнения.
        Костные зоны всегда 1.0, мягкие зоны понижаются при мимике.
        """
        excluded = self.get_excluded_zones(flags)
        weights = {}
        
        # Все костные зоны остаются 1.0
        bone_zones = ["nasion", "orbit_L", "orbit_R", "zygomatic_L", "zygomatic_R", 
                      "gonion_L", "gonion_R", "pogonion", "ramus_L", "ramus_R"]
        for zone in bone_zones:
            weights[zone] = 1.0
        
        # Мягкие зоны
        soft_zones = ["cheek_L", "cheek_R", "nasolabial_L", "nasolabial_R", 
                      "lip_upper", "lip_lower", "forehead", "brow_L", "brow_R", 
                      "glabella", "chin_soft"]
        for zone in soft_zones:
            if zone in excluded:
                weights[zone] = 0.1 if "jaw_open" in flags or "smile" in flags else 0.3
            else:
                weights[zone] = 1.0
        
        return weights


class ExpressionNormalizedComparator:
    """
    Вместо исключения зон — 'выпрямляет' меш до нейтрального expression.
    Использует exp-параметры 3DDFA.
    """
    
    def normalize_to_neutral(self, vertices: np.ndarray, 
                             exp_vector: np.ndarray,
                             exp_basis: np.ndarray) -> np.ndarray:
        """
        vertices: (N, 3) — текущий меш
        exp_vector: (29,) — expression params
        exp_basis: (N*3, 29) — basis vectors для expression
        
        Returns: vertices_neutral — меш без expression
        """
        # Деформируем обратно: вычитаем вклад expression
        delta = (exp_basis @ exp_vector).reshape(-1, 3)
        neutral = vertices - delta
        return neutral
    
    def compare_expression_robust(self, rec_a: dict, rec_b: dict) -> dict:
        """
        Сравнивает два фото независимо от expression:
        1. Нормализуем оба до neutral
        2. Сравниваем neutral меши
        3. Если neutral близки, а original далеки — различия из-за мимики
        """
        v_a_neutral = self.normalize_to_neutral(
            rec_a["vertices"], rec_a["exp"], rec_a["exp_basis"]
        )
        v_b_neutral = self.normalize_to_neutral(
            rec_b["vertices"], rec_b["exp"], rec_b["exp_basis"]
        )
        
        # Сравниваем neutral
        diff_neutral = np.linalg.norm(v_a_neutral - v_b_neutral, axis=1).mean()
        
        # Сравниваем оригинальные (с мимикой)
        diff_original = np.linalg.norm(rec_a["vertices"] - rec_b["vertices"], axis=1).mean()
        
        # Если neutral близки, а original далеки — различия из-за мимики
        expression_contribution = diff_original - diff_neutral
        
        return {
            "diff_neutral": float(diff_neutral),
            "diff_original": float(diff_original),
            "expression_contribution": float(expression_contribution),
            "is_expression_only": diff_neutral < 1.0 and diff_original > 2.0,
        }