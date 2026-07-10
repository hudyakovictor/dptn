from __future__ import annotations

from typing import List, Dict, Any


class CrossModalTextureRulesV2:
    """
    Cross-modal rules V2: использует новые фичи TextureExtractorV2.
    
    Правила срабатывают когда геометрия говорит 'тот же человек',
    а текстура показывает аномалии синтетического материала.
    """

    RULES = [
        {
            "name": "mask_on_similar_skull",
            "conditions": {
                "geometry_excess": "< 1.0",       # Кости совпадают (нормализованное расстояние < 1)
                "texture_anomaly": "> 0.7",       # Текстура аномальна относительно когорты
                "quality": "> 0.5",                # Фото достаточно качественное
                "specular_elongation": "> 1.8",    # Блики вытянутые (silicone specular)
            },
            "h1_boost": 0.35,
            "description": "Геометрия совпадает, но текстура указывает на синтетический материал. Вероятна маска на похожем черепе.",
        },
        {
            "name": "modern_vas_mask",
            "conditions": {
                "era": "vas_era",                  # Эпоха VAS (2021+)
                "texture_anomaly": "> 0.6",
                "lbp_r1_hist_entropy": "< 3.0",    # Низкая энтропия LBP = регулярная текстура
                "pore_density_r2_mpx": "< 0.05",   # Очень низкая плотность пор
            },
            "h1_boost": 0.25,
            "description": "Современная маска VAS: высокая детализация, но отсутствие биологической вариативности пор и LBP.",
        },
        {
            "name": "early_udmurt_mask",
            "conditions": {
                "era": "udmurt_era",               # Эпоха UDMURT (2012-2021)
                "specular_sharpness": "> 2.5",     # Острые блики
                "specular_dispersion": "> 0.15",   # Рассеянные блики
                "sss_index": "< 0.05",             # Низкий SSS (силикон не просвечивает)
            },
            "h1_boost": 0.20,
            "description": "Ранние маски UDMURT: характерные зеркальные блики и отсутствие subsurface scattering.",
        },
        {
            "name": "early_scan_silicone",
            "conditions": {
                "era": "early_scan",               # Сканы 1999-2005
                "texture_anomaly": "> 0.8",
                "hemoglobin_od_std": "< 0.02",     # Нет гемоглобина
                "tv_residual_sparsity": "> 0.85",  # Слишком гладко (TV residual)
            },
            "h1_boost": 0.30,
            "description": "Ранние сканы 1999-2005: аномально гладкая текстура без гемоглобинного шума.",
        },
        {
            "name": "seam_detected",
            "conditions": {
                "seam_score": "> 0.15",            # Четкий скачок текстуры по границе
                "geometry_excess": "< 1.2",        # Геометрия близка
            },
            "h1_boost": 0.40,
            "description": "Обнаружен шов/стык текстуры по границе челюсти или за ушами — признак накладки маски.",
        },
        {
            "name": "pore_anomaly_stamping",
            "conditions": {
                "pore_periodicity": "< 0.3",       # Низкая энтропия = регулярная штамповка
                "pore_eccentricity_mean": "> 0.7", # Поры вытянутые в одну сторону
                "texture_anomaly": "> 0.5",
            },
            "h1_boost": 0.25,
            "description": "Регулярная периодичность пор и их эксцентриситет указывают на штамповку силикона.",
        },
    ]

    def evaluate(self, geometry_excess: float, texture_score: Dict,
                 photo_year: int, quality: float, era: str) -> List[Dict]:
        """Возвращает список сработавших правил с H1 boost."""
        triggered = []
        for rule in self.RULES:
            if self._match(rule, geometry_excess, texture_score, photo_year, quality, era):
                triggered.append({
                    "rule": rule["name"],
                    "h1_boost": rule["h1_boost"],
                    "description": rule["description"],
                })
        return triggered

    def _match(self, rule, geom_excess, texture, year, quality, era):
        cond = rule["conditions"]
        
        if "geometry_excess" in cond:
            if geom_excess >= 1.0:
                return False
        
        if "texture_anomaly" in cond:
            if texture.get("anomaly_score", 0) < 0.6:
                return False
        
        if "quality" in cond:
            if quality < 0.5:
                return False

        if "era" in cond:
            if era != cond["era"]:
                return False

        # Feature-specific conditions from texture feature_flags
        feature_flags = texture.get("feature_flags", {})
        
        if "specular_elongation" in cond:
            threshold = float(cond["specular_elongation"].replace("> ", ""))
            if feature_flags.get("specular_elongation", 0.0) < threshold:
                return False
        
        if "specular_sharpness" in cond:
            threshold = float(cond["specular_sharpness"].replace("> ", ""))
            if feature_flags.get("specular_sharpness", 0.0) < threshold:
                return False
        
        if "specular_dispersion" in cond:
            threshold = float(cond["specular_dispersion"].replace("> ", ""))
            if feature_flags.get("specular_dispersion", 0.0) < threshold:
                return False
        
        if "sss_index" in cond:
            threshold = float(cond["sss_index"].replace("< ", ""))
            if feature_flags.get("sss_index", 1.0) > threshold:
                return False
        
        if "lbp_r1_hist_entropy" in cond:
            threshold = float(cond["lbp_r1_hist_entropy"].replace("< ", ""))
            if feature_flags.get("lbp_r1_hist_entropy", 10.0) > threshold:
                return False
        
        if "pore_density_r2_mpx" in cond:
            threshold = float(cond["pore_density_r2_mpx"].replace("< ", ""))
            if feature_flags.get("pore_density_r2_mpx", 1.0) > threshold:
                return False
        
        if "hemoglobin_od_std" in cond:
            threshold = float(cond["hemoglobin_od_std"].replace("< ", ""))
            if feature_flags.get("hemoglobin_od_std", 1.0) > threshold:
                return False
        
        if "tv_residual_sparsity" in cond:
            threshold = float(cond["tv_residual_sparsity"].replace("> ", ""))
            if feature_flags.get("tv_residual_sparsity", 0.0) < threshold:
                return False
        
        if "seam_score" in cond:
            threshold = float(cond["seam_score"].replace("> ", ""))
            if feature_flags.get("seam_score", 0.0) < threshold:
                return False
        
        if "pore_periodicity" in cond:
            threshold = float(cond["pore_periodicity"].replace("< ", ""))
            if feature_flags.get("pore_periodicity", 1.0) > threshold:
                return False
        
        if "pore_eccentricity_mean" in cond:
            threshold = float(cond["pore_eccentricity_mean"].replace("> ", ""))
            if feature_flags.get("pore_eccentricity_mean", 0.0) < threshold:
                return False

        return True


# Backward compatibility
CrossModalTextureRules = CrossModalTextureRulesV2