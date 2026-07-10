from __future__ import annotations

import numpy as np
from typing import List, Dict, Any
from ..s2_metrics.cross_modal_rules import CrossModalTextureRules


class H1SyntheticDetector:
    """Комплексный детектор синтетики H1."""
    
    def __init__(self):
        self.texture_rules = TextureRules()
        self.geometry_rules = GeometryRules()
        self.cross_modal = CrossModalTextureRules()
    
    def evaluate(self, photo: Dict, anchor_comparison: Dict,
                 texture_anomaly: Dict, biological_flags: List[Dict]) -> Dict:
        """
        Собирает evidence для H1 из всех источников.
        """
        evidence = []
        score = 0.0
        
        # 1. Texture evidence
        tex_evidence = self.texture_rules.evaluate(photo, texture_anomaly)
        if tex_evidence:
            evidence.extend(tex_evidence)
            score += sum(e["weight"] for e in tex_evidence)
        
        # 2. Geometry evidence (маска на похожем черепе)
        geom_evidence = self.geometry_rules.evaluate(anchor_comparison)
        if geom_evidence:
            evidence.extend(geom_evidence)
            score += sum(e["weight"] for e in geom_evidence)
        
        # 3. Cross-modal (самый сильный)
        cross = self.cross_modal.evaluate(
            anchor_comparison.get("excess_distance", 999),
            texture_anomaly,
            photo.get("year", 2000),
            photo.get("quality", 0.5)
        )
        if cross:
            evidence.extend(cross)
            score += sum(e.get("h1_boost", 0) for e in cross) * 2  # Удваиваем вес
        
        # 4. Biological impossibility (косвенно поддерживает H1)
        bio_h1 = [f for f in biological_flags if f.get("severity") == "CRITICAL"]
        if bio_h1:
            evidence.append({
                "type": "biological_impossibility_supports_h1",
                "weight": 0.15,
                "description": "Биологически невозможные изменения косвенно поддерживают гипотезу подмены.",
            })
            score += 0.15
        
        # 5. Prosthetic modifier (зоны натяжения маски)
        prosthetic = self._prosthetic_zones(anchor_comparison)
        if prosthetic:
            evidence.append(prosthetic)
            score += prosthetic.get("weight", 0)
        
        # Нормализация
        h1_probability = float(np.clip(score, 0.0, 1.0))
        
        return {
            "h1_probability": h1_probability,
            "evidence": evidence,
            "confidence": self._confidence(evidence),
            "is_triggered": h1_probability > 0.65,
        }
    
    def _prosthetic_zones(self, anchor_comparison: Dict) -> Dict:
        """
        Маска на другом черепе создаёт артефакты в зонах натяжения:
        - Скуловые дуги (маска 'съезжает')
        - Переносица (натяжение)
        - Угол челюсти (лишний объём)
        """
        violations = anchor_comparison.get("bone_zone_violations", 0)
        heatmap_max = anchor_comparison.get("heatmap_max", 0)
        
        if violations >= 3 and heatmap_max > 0.7:
            return {
                "type": "prosthetic_zone_tension",
                "weight": 0.25,
                "violations": violations,
                "heatmap_max": heatmap_max,
                "description": f"Обнаружены артефакты натяжения маски в {violations} костных зонах. Характерно для силиконовой маски на чужом черепе.",
            }
        return None
    
    def _confidence(self, evidence: List[Dict]) -> float:
        """Уверенность растёт с количеством независимых evidence."""
        if not evidence:
            return 0.0
        # Независимые источники = выше confidence
        sources = set(e.get("type", "unknown") for e in evidence)
        base = min(0.9, 0.3 + 0.15 * len(sources))
        return float(base)


class TextureRules:
    """Правила на основе текстурных метрик."""
    
    def evaluate(self, photo: Dict, texture_anomaly: Dict) -> List[Dict]:
        evidence = []
        flags = texture_anomaly.get("feature_flags", {})
        
        if "specular_ratio" in flags and "lbp_entropy_r1" in flags:
            evidence.append({
                "type": "texture_silicone_signature",
                "weight": 0.20,
                "description": "Комбинация зеркальных бликов (specular) и низкой энтропии пор (LBP) — классическая сигнатура силикона.",
            })
        
        if "fft_peak_regularity" in flags:
            evidence.append({
                "type": "texture_regular_microrelief",
                "weight": 0.15,
                "description": "Регулярный микрорельеф по FFT указывает на пресс-форму маски, а не естественную кожу.",
            })
        
        if photo.get("quality", 0) > 0.6 and texture_anomaly.get("anomaly_score", 0) > 0.8:
            evidence.append({
                "type": "texture_anomaly_high_quality",
                "weight": 0.20,
                "description": "Высокая текстурная аномалия при отличном качестве фото исключает артефакты шума.",
            })
        
        return evidence


class GeometryRules:
    """Геометрические маркеры маски на похожем черепе."""
    
    def evaluate(self, anchor_comparison: Dict) -> List[Dict]:
        evidence = []
        
        # Маска не меняет кость, но создаёт "мягкую оболочку" неестественной толщины
        if anchor_comparison.get("excess_distance", 0) < 1.0:
            # Кости совпадают — но проверим soft tissue
            heatmap_mean = anchor_comparison.get("heatmap_mean", 0)
            if heatmap_mean > 0.4:
                evidence.append({
                    "type": "geometry_mask_soft_tissue",
                    "weight": 0.15,
                    "description": "Костные структуры совпадают, но мягкие ткани имеют аномальную толщину. Характерно для маски на похожем черепе.",
                })
        
        return evidence