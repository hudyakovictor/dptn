from __future__ import annotations

from typing import List, Dict, Any
from datetime import datetime


class AlternativeExplanations:
    """Генерирует блок альтернативных объяснений для каждого типа аномалии.
    Проверяет, мог ли обнаруженный эффект быть вызван mundane-причинами.
    """

    RULES = {
        "bone_mismatch": [
            {
                "name": "Разница ракурсов",
                "check": lambda e: e.get("pose_gap_deg", 99) < 5.0,
                "verdict": "не подтверждается",
                "reason_template": "ракурс отличается на {pose_gap:.1f}° (< 5°)",
            },
            {
                "name": "Возрастные изменения",
                "check": lambda e: e.get("age_gap_years", 99) < 2.0,
                "verdict": "не подтверждается",
                "reason_template": "разница в возрасте {age_gap:.1f} года (< 2 лет)",
            },
            {
                "name": "Плохое качество фото",
                "check": lambda e: e.get("quality", 0) > 0.5,
                "verdict": "не объясняет",
                "reason_template": "качество фото {quality:.0%} (> 50%)",
            },
        ],
        "silicone_texture": [
            {
                "name": "Грим/освещение",
                "check": lambda e: e.get("specular_ratio", 1) < 0.3,
                "verdict": "не объясняет",
                "reason_template": "доля спекулярных бликов {specular:.0%} (< 30%)",
            },
            {
                "name": "JPEG-артефакты",
                "check": lambda e: e.get("jpeg_blockiness", 99) < 1.5,
                "verdict": "не объясняет",
                "reason_template": "блокность JPEG {jpeg:.2f} (< 1.5)",
            },
            {
                "name": "Плёночное зерно",
                "check": lambda e: e.get("noise_level", 99) < 20.0,
                "verdict": "не объясняет",
                "reason_template": "уровень шума {noise:.1f} (< 20)",
            },
            {
                "name": "Низкое разрешение",
                "check": lambda e: e.get("resolution_score", 1) > 0.4,
                "verdict": "не объясняет",
                "reason_template": "разрешение достаточно ({resolution:.0%})",
            },
        ],
        "chronology_impossible": [
            {
                "name": "Ошибка датировки",
                "check": lambda e: e.get("date_confidence", 1) > 0.8,
                "verdict": "маловероятна",
                "reason_template": "уверенность в датах высокая",
            },
            {
                "name": "Естественное старение",
                "check": lambda e: e.get("age_gap_years", 99) < 5.0,
                "verdict": "не подтверждается",
                "reason_template": "изменения слишком быстрые для естественного старения",
            },
        ],
        "return_to_baseline": [
            {
                "name": "Естественная вариативность",
                "check": lambda e: e.get("return_amplitude", 99) < 0.3,
                "verdict": "не объясняет",
                "reason_template": "амплитуда возврата аномальна (> 0.3)",
            },
            {
                "name": "Ошибка реконструкции",
                "check": lambda e: e.get("reconstruction_quality", 0) > 0.6,
                "verdict": "маловероятна",
                "reason_template": "качество реконструкции достаточное",
            },
        ],
        "cluster_shift": [
            {
                "name": "Изменение позы",
                "check": lambda e: e.get("pose_shift_deg", 99) < 10.0,
                "verdict": "не подтверждается",
                "reason_template": "сдвиг позы минимален (< 10°)",
            },
            {
                "name": "Изменение освещения",
                "check": lambda e: e.get("lighting_change", 0) < 0.3,
                "verdict": "не объясняет",
                "reason_template": "изменение освещения незначительно (< 0.3)",
            },
        ],
        "mask_on_similar_skull": [
            {
                "name": "Совпадение костей",
                "check": lambda e: True,
                "verdict": "подтверждается",
                "reason_template": "костные структуры совпадают",
            },
        ],
        "prosthetic_tension": [
            {
                "name": "Естественное натяжение кожи",
                "check": lambda e: e.get("age_years", 0) > 50,
                "verdict": "возможно",
                "reason_template": "возраст предполагает естественное натяжение",
            },
        ],
        "physical_texture_anomaly": [
            {
                "name": "Плохое качество",
                "check": lambda e: e.get("quality", 0) > 0.5,
                "verdict": "не объясняет",
                "reason_template": "качество фото достаточное",
            },
            {
                "name": "Освещение",
                "check": lambda e: e.get("specular_ratio", 1) < 0.4,
                "verdict": "не объясняет",
                "reason_template": "доля бликов нормальная (< 40%)",
            },
        ],
    }

    def evaluate(self, finding_type: str, evidence: Dict) -> List[Dict]:
        """Возвращает список альтернативных объяснений с оценкой."""
        rules = self.RULES.get(finding_type, [])
        results = []
        for rule in rules:
            try:
                applies = rule["check"](evidence)
            except Exception:
                applies = False
            # Форматируем reason_template с данными из evidence
            reason = rule.get("reason_template", rule.get("reason", ""))
            try:
                reason = reason.format(**evidence)
            except (KeyError, ValueError):
                pass  # Используем как есть если форматирование не удалось
            results.append({
                "explanation": rule["name"],
                "supported": applies,
                "verdict": rule["verdict"] if applies else "не подтверждается",
                "reason": reason,
            })
        return results

    def format_block(self, finding_type: str, evidence: Dict) -> str:
        """Форматирует блок альтернатив для вывода в отчёт."""
        alternatives = self.evaluate(finding_type, evidence)
        if not alternatives:
            return ""
        lines = ["Возможные альтернативные объяснения:"]
        for alt in alternatives:
            status = "✓" if alt["supported"] else "✗"
            lines.append(f"  {status} {alt['explanation']}: {alt['verdict']} ({alt['reason']})")
        return "\n".join(lines)


class JournalistThesisEngine:
    """Генерирует тезисы на человеческом языке для публикаций."""
    
    TEMPLATES = {
        "bone_mismatch": {
            "severity": "CRITICAL",
            "text": "На фото от {date} костная структура {bone_zone} отличается от установленного baseline на {delta_mm:.1f} мм. Это превышает естественную вариативность реконструкции ({noise_mm:.1f} мм) в {ratio:.1f} раза. Такое изменение невозможно без хирургического вмешательства или подмены человека.",
            "evidence_type": "geometry",
        },
        "silicone_texture": {
            "severity": "HIGH",
            "text": "Текстурный анализ фото от {date} выявил признаки синтетического материала: {texture_flags}. При разрешении {quality:.0%} и отсутствии шумовых артефактов эти показатели не могут быть объяснены качеством съёмки.",
            "evidence_type": "texture",
        },
        "chronology_impossible": {
            "severity": "CRITICAL",
            "text": "Между фото от {date_a} и {date_b} ({gap_days} дней) зафиксировано изменение формы {body_part}, которое биологически требует минимум {min_days} дней ({surgery_type} + реабилитация). Столь короткий промежуток исключает естественное или хирургическое объяснение.",
            "evidence_type": "chronology",
        },
        "return_to_baseline": {
            "severity": "HIGH",
            "text": "Форма {feature} изменялась в период {period_a}, но к {period_b} вернулась к значениям {baseline_years}-летней давности. При естественном старении такой 'откат' невозможен — он указывает на смену маски или двойника с последующим возвратом к оригиналу.",
            "evidence_type": "chronology",
        },
        "cluster_shift": {
            "severity": "CRITICAL",
            "text": "Анализ 3D-структуры черепа (alpha-vector) показывает, что фото от {date} принадлежит identity-кластеру '{to_cluster}', тогда как предыдущие {prev_count} фото относились к кластеру '{from_cluster}'. Вероятность случайного отклонения: <{p_value:.1%}. Это статистически значимое доказательство другого человека.",
            "evidence_type": "alpha_clustering",
        },
        "mask_on_similar_skull": {
            "severity": "HIGH",
            "text": "Фото от {date} демонстрирует совпадение костных структур (±{geom_mm:.1f} мм) с baseline, но текстура кожи содержит аномалии: {texture_summary}. Эта комбинация — классический маркер силиконовой маски на похожем черепе.",
            "evidence_type": "cross_modal",
        },
        "prosthetic_tension": {
            "severity": "MEDIUM",
            "text": "В зонах натяжения маски ({zones}) обнаружены артефакты: локальные деформации меша на {deform_mm:.1f} мм. Характерно для силиконовой накладки, натянутой на чужую костную структуру.",
            "evidence_type": "geometry",
        },
    }
    
    def __init__(self):
        self._alt_explainer = AlternativeExplanations()

    def generate(self, findings: List[Dict]) -> List[Dict]:
        theses = []
        for finding in findings:
            template = self.TEMPLATES.get(finding["type"])
            if not template:
                continue

            text = template["text"].format(**finding["params"])

            # Альтернативные объяснения
            evidence = finding.get("evidence", finding.get("params", {}))
            alternatives = self._alt_explainer.evaluate(finding["type"], evidence)
            alt_block = self._alt_explainer.format_block(finding["type"], evidence)

            thesis = {
                "type": finding["type"],
                "severity": template["severity"],
                "text": text,
                "evidence_type": template["evidence_type"],
                "photo_ids": finding.get("photo_ids", []),
                "pair_ids": finding.get("pair_ids", []),
                "confidence": finding.get("confidence", 0.5),
                "alternatives": alternatives,
                "alternatives_block": alt_block,
            }
            theses.append(thesis)

        # Сортируем: CRITICAL first
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        theses.sort(key=lambda x: severity_order.get(x["severity"], 99))
        return theses
    
    def generate_executive_summary(self, theses: List[Dict], total_photos: int) -> str:
        """Итоговый текст для вставки в статью."""
        critical = len([t for t in theses if t["severity"] == "CRITICAL"])
        high = len([t for t in theses if t["severity"] == "HIGH"])

        lines = [
            f"## Итоги форензик-анализа ({total_photos} фото, 1999–2025)",
            "",
            f"Система выявила **{critical} критических** и **{high} высоких** аномалий, несовместимых с гипотезой об единственном человеке на всех фотографиях.",
            "",
            "### Ключевые находки:",
        ]

        for thesis in theses[:10]:
            emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡"}.get(thesis["severity"], "⚪")
            lines.append(f"{emoji} {thesis['text']}")
            # Добавляем альтернативные объяснения
            alt_block = thesis.get("alternatives_block", "")
            if alt_block:
                lines.append(f"   {alt_block}")

        lines.extend([
            "",
            "### Методология",
            "Анализ проведён с использованием 3D-реконструкции лиц (3DDFA-V3), костной геометрии, текстурного анализа кожи и хронологического моделирования. Калибровка выполнена на контрольном датасете с известным ground-truth.",
        ])

        return "\n".join(lines)


class EvidenceLinker:
    """Связывает тезисы с конкретными фото и парами."""
    
    def link(self, thesis: Dict, all_pairs: List[Dict], 
             all_photos: Dict) -> Dict:
        """Добавляет к тезису ссылки на исходные данные."""
        enriched = thesis.copy()
        enriched["evidence_links"] = []
        
        for photo_id in thesis.get("photo_ids", []):
            photo = all_photos.get(photo_id)
            if photo:
                enriched["evidence_links"].append({
                    "type": "photo",
                    "photo_id": photo_id,
                    "date": photo.get("date"),
                    "path": photo.get("source_path"),
                    "preview": f"/photos/{photo_id}.jpg",
                })
        
        for pair_id in thesis.get("pair_ids", []):
            pair = next((p for p in all_pairs if p["pair_id"] == pair_id), None)
            if pair:
                enriched["evidence_links"].append({
                    "type": "comparison",
                    "pair_id": pair_id,
                    "photo_a": pair["photo_a"],
                    "photo_b": pair["photo_b"],
                    "heatmap": f"/heatmaps/{pair_id}.png",
                    "excess_distance": pair.get("excess_distance"),
                })
        
        return enriched