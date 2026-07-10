from __future__ import annotations

from typing import List, Dict, Any


class CrossModalTextureRules:
    """
    Самый сильный сигнал маски — когда геометрия говорит 'тот же человек',
    а текстура 'аномалия'.
    """
    
    RULES = [
        {
            "name": "mask_on_similar_skull",
            "conditions": {
                "geometry_excess": "< 1.0",      # Кости совпадают
                "texture_anomaly": "> 0.7",      # Текстура аномальна
                "quality": "> 0.5",               # Фото качественное
                "specular_flag": True,
            },
            "h1_boost": 0.35,
            "description": "Геометрия совпадает, но текстура указывает на синтетический материал. Вероятна маска на похожем черепе.",
        },
        {
            "name": "vas_modern_mask",
            "conditions": {
                "year": ">= 2023",
                "texture_anomaly": "> 0.6",
                "lbp_entropy_r1": "< 3.0",
                "albedo_a_std": "< 5.0",
            },
            "h1_boost": 0.25,
            "description": "Современная маска VAS: высокая детализация, но отсутствие биологической вариативности.",
        },
        {
            "name": "udmurt_early_mask",
            "conditions": {
                "year": "2012-2021",
                "specular_ratio": "> 0.15",
                "fft_peak_regularity": "> 0.6",
            },
            "h1_boost": 0.20,
            "description": "Ранние маски UDMURT: характерные зеркальные блики и регулярный микрорельеф.",
        },
    ]
    
    def evaluate(self, geometry_excess: float, texture_score: Dict,
                 photo_year: int, quality: float) -> List[Dict]:
        triggered = []
        for rule in self.RULES:
            if self._match(rule, geometry_excess, texture_score, photo_year, quality):
                triggered.append({
                    "rule": rule["name"],
                    "h1_boost": rule["h1_boost"],
                    "description": rule["description"],
                })
        return triggered
    
    def _match(self, rule, geom_excess, texture, year, quality):
        cond = rule["conditions"]
        if "geometry_excess" in cond and geom_excess >= 1.0:
            return False
        if "texture_anomaly" in cond and texture.get("anomaly_score", 0) < 0.6:
            return False
        if "quality" in cond and quality < 0.5:
            return False
        if "year" in cond:
            year_str = cond["year"]
            if ">=" in year_str:
                if year < int(year_str.replace(">= ", "")):
                    return False
            elif "-" in year_str:
                start, end = map(int, year_str.split("-"))
                if not (start <= year <= end):
                    return False
        # Check feature-specific conditions from texture feature_flags
        feature_flags = texture.get("feature_flags", {})
        if "specular_flag" in cond and cond["specular_flag"]:
            if not feature_flags.get("specular_ratio") and not feature_flags.get("specular_sharpness"):
                return False
        if "specular_ratio" in cond:
            threshold = float(cond["specular_ratio"].replace("> ", ""))
            if feature_flags.get("specular_ratio", 0.0) < threshold:
                return False
        if "lbp_entropy_r1" in cond:
            threshold = float(cond["lbp_entropy_r1"].replace("< ", ""))
            if feature_flags.get("lbp_entropy_r1", 10.0) > threshold:
                return False
        if "albedo_a_std" in cond:
            threshold = float(cond["albedo_a_std"].replace("< ", ""))
            if feature_flags.get("albedo_a_std", 10.0) > threshold:
                return False
        if "fft_peak_regularity" in cond:
            threshold = float(cond["fft_peak_regularity"].replace("> ", ""))
            if feature_flags.get("fft_peak_regularity", 0.0) < threshold:
                return False
        return True