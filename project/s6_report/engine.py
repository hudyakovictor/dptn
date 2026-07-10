from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ..shared.logging import setup_logger
from ..shared.schemas import PipelineDataset, Stage5Record, Stage6Record
from ..shared.utils import load_json, save_json, save_text
from .journalist_engine import JournalistThesisEngine, EvidenceLinker

logger = setup_logger("deeputin.s6")


class ReportEngine:
    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}
        self.journalist = JournalistThesisEngine()
        self.evidence_linker = EvidenceLinker()

    def build_report(self, main_root: str | Path) -> Stage6Record:
        root = Path(main_root)
        verdicts = self._load_verdicts(root)
        timeline = load_json(root / "timeline.json", default=[]) or []
        chronology = load_json(root / "chronology.json", default={}) or {}
        if not verdicts:
            report = Stage6Record(
                dataset=PipelineDataset.MAIN,
                generated_at=__import__("datetime").datetime.utcnow().isoformat(timespec="seconds"),
                summary={"message": "Нет verdicts для отчёта"},
                chronology_summary=self._chronology_summary(chronology),
            )
            save_json(report.model_dump(), root / "report.json")
            save_text(self._to_markdown(report), root / "report.md")
            return report

        stats = self._aggregate_statistics(verdicts, timeline)
        # Загружаем stage2_records для persona atlas
        stage2_records = {}
        for photo_dir in root.iterdir():
            if not photo_dir.is_dir():
                continue
            info = load_json(photo_dir / "info.json")
            geo = load_json(photo_dir / "geometry_metrics.json")
            tex = load_json(photo_dir / "texture_metrics.json")
            if info and isinstance(geo, dict):
                try:
                    from ..shared.schemas import Stage2Record
                    stage2_records[info.get("photo_id", photo_dir.name)] = Stage2Record(
                        photo_id=info.get("photo_id", photo_dir.name),
                        dataset=info.get("dataset", "main"),
                        bucket=info.get("pose", {}).get("bucket", "unknown"),
                        quality=info.get("quality", {}),
                        geometry=geo,
                        texture=tex or {},
                    )
                except Exception:
                    pass
        personas = self._build_personas(timeline, verdicts, stage2_records=stage2_records or None)
        theses = self._generate_theses(stats, personas)
        
        # Генерация журналистских тезисов на русском языке
        journalist_findings = self._build_journalist_findings(verdicts, timeline, chronology)
        journalist_theses = self.journalist.generate(journalist_findings)
        executive_summary = self.journalist.generate_executive_summary(journalist_theses, len(verdicts))
        
        # Связывание тезисов с исходными фото
        all_pairs = load_json(root / "pairs.json", default=[]) or []
        all_photos = {v.photo_id: {"date": None} for v in verdicts}
        for v in verdicts:
            stage1_path = root / v.photo_id / "info.json"
            s1_data = load_json(stage1_path, default={})
            if s1_data:
                all_photos[v.photo_id]["date"] = s1_data.get("date")
                all_photos[v.photo_id]["source_path"] = str(root / v.photo_id)
        
        linked_theses = []
        for thesis in journalist_theses:
            linked = self.evidence_linker.link(thesis, all_pairs, all_photos)
            linked_theses.append(linked)
        
        report = Stage6Record(
            dataset=PipelineDataset.MAIN,
            generated_at=__import__("datetime").datetime.utcnow().isoformat(timespec="seconds"),
            summary={
                "total_photos": len(verdicts),
                "dominant_hypothesis": stats["dominant_hypothesis"],
                "top_bucket": stats["top_bucket"],
                "executive_summary": executive_summary,
            },
            chronology_summary=self._chronology_summary(chronology),
            personas=personas,
            theses=theses,
            statistics=stats,
            top_anomalies=sorted(timeline, key=lambda row: float(row.get("anomaly_score", 0.0)), reverse=True)[:20],
            verdict_counts=stats["verdict_counts"],
            timeline_brief=timeline[:200],
        )
        save_json(report.model_dump(), root / "report.json")
        save_text(self._to_markdown(report), root / "report.md")
        
        # Сохраняем журналистские тезисы отдельно
        save_json({
            "executive_summary": executive_summary,
            "theses": linked_theses,
            "total_photos": len(verdicts),
        }, root / "journalist_report.json")
        
        return report

    def _load_verdicts(self, root: Path) -> list[Stage5Record]:
        payload = load_json(root / "verdicts.json", default=[]) or []
        return [Stage5Record.model_validate(item) for item in payload]

    def _aggregate_statistics(self, verdicts: list[Stage5Record], timeline: list[dict[str, object]]) -> dict[str, object]:
        counts = Counter(v.verdict.hypothesis.value for v in verdicts)
        dominant = counts.most_common(1)[0][0] if counts else "H_UNCERTAIN"
        bucket_counter = Counter(row.get("bucket", "unknown") for row in timeline if row.get("bucket"))
        return {
            "verdict_counts": dict(counts),
            "dominant_hypothesis": dominant,
            "top_bucket": bucket_counter.most_common(1)[0][0] if bucket_counter else "unknown",
            "mean_confidence": float(np.mean([v.verdict.confidence for v in verdicts]) if verdicts else 0.0),
            "photo_count": len(verdicts),
        }

    def _build_personas(
        self,
        timeline: list[dict[str, object]],
        verdicts: list[Stage5Record],
        stage2_records: dict | None = None,
    ) -> list[dict[str, object]]:
        """Persona atlas через HDBSCAN кластеризацию по костным метрикам."""
        try:
            import hdbscan
        except ImportError:
            # Fallback: эпохальная группировка
            return self._build_personas_era_fallback(timeline, verdicts)

        # Собираем костные метрики из stage2_records
        if not stage2_records:
            return self._build_personas_era_fallback(timeline, verdicts)

        photo_ids = []
        bone_vectors = []
        for v in verdicts:
            rec = stage2_records.get(v.photo_id)
            if rec is None:
                continue
            geom = rec.geometry
            if not geom:
                continue
            # Берём топ-10 костных метрик (bone_*, mesh_*)
            bone_keys = sorted([k for k in geom if k.startswith("bone_") or k.startswith("mesh_")])[:10]
            if len(bone_keys) < 3:
                continue
            vec = [geom.get(k, 0.0) for k in bone_keys]
            bone_vectors.append(vec)
            photo_ids.append(v.photo_id)

        if len(bone_vectors) < 5:
            return self._build_personas_era_fallback(timeline, verdicts)

        X = np.array(bone_vectors, dtype=float)
        # Z-score нормализация
        mu = np.median(X, axis=0)
        sigma = np.std(X, axis=0) + 1e-6
        X_norm = (X - mu) / sigma

        # HDBSCAN кластеризация
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=max(3, len(X) // 10),
            min_samples=2,
            metric="euclidean",
        )
        labels = clusterer.fit_predict(X_norm)

        # Группируем по кластерам
        clusters: dict[int, list[int]] = defaultdict(list)
        for i, label in enumerate(labels):
            clusters[label].append(i)

        personas = []
        for cluster_id, indices in sorted(clusters.items()):
            if cluster_id == -1:
                # Шум — отдельная персона
                persona_label = "unclustered"
            else:
                persona_label = f"persona_{cluster_id}"

            cluster_photo_ids = [photo_ids[i] for i in indices]
            cluster_vectors = X[indices]

            # Характерные признаки: mean ± std по костным зонам
            mean_vec = np.mean(cluster_vectors, axis=0)
            std_vec = np.std(cluster_vectors, axis=0)

            # Находим ключевые зоны (самые стабильные = lowest CV)
            cv = std_vec / (np.abs(mean_vec) + 1e-6)
            stable_zones = np.argsort(cv)[:5]

            # Даты кластера
            cluster_dates = []
            cluster_hypotheses = []
            for v in verdicts:
                if v.photo_id in cluster_photo_ids:
                    # Найти дату в timeline
                    for row in timeline:
                        if row.get("photo_id") == v.photo_id:
                            cluster_dates.append(row.get("date"))
                            cluster_hypotheses.append(v.verdict.hypothesis.value)
                            break

            valid_dates = [d for d in cluster_dates if d]
            date_range = f"{min(valid_dates)} — {max(valid_dates)}" if valid_dates else "unknown"
            dominant_hyp = Counter(cluster_hypotheses).most_common(1)[0][0] if cluster_hypotheses else "H_UNCERTAIN"

            personas.append({
                "persona_id": persona_label,
                "cluster_id": int(cluster_id),
                "photo_count": len(indices),
                "date_range": date_range,
                "dominant_hypothesis": dominant_hyp,
                "mean_anomaly_score": float(np.mean([
                    float(v.verdict.evidence.get("anomaly", 0.0))
                    for v in verdicts if v.photo_id in cluster_photo_ids
                ])),
                "characteristic_features": [
                    {"zone": f"metric_{i}", "mean": float(mean_vec[i]), "std": float(std_vec[i])}
                    for i in stable_zones
                ],
                "photo_ids": cluster_photo_ids[:5],  # representative examples
                "noise_fraction": float(np.mean(labels == -1)),
            })

        return personas

    def _build_personas_era_fallback(
        self, timeline: list[dict[str, object]], verdicts: list[Stage5Record]
    ) -> list[dict[str, object]]:
        """Fallback: эпохальная группировка (если HDBSCAN недоступен)."""
        eras = [
            ("1998-2005", 1998, 2005),
            ("2005-2012", 2005, 2012),
            ("2012-2019", 2012, 2019),
            ("2019-2022", 2019, 2022),
            ("2022-2026", 2022, 2026),
        ]
        items_by_year: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in timeline:
            date_str = row.get("date")
            if not date_str:
                continue
            try:
                year = int(str(date_str)[:4])
            except Exception:
                continue
            for label, start, end in eras:
                if start <= year <= end:
                    items_by_year[label].append(row)
                    break
        personas = []
        for label, _, _ in eras:
            rows = items_by_year.get(label, [])
            if not rows:
                continue
            hypothesis = Counter(row.get("hypothesis", "H_UNCERTAIN") for row in rows).most_common(1)[0][0]
            personas.append(
                {
                    "era": label,
                    "dominant_hypothesis": hypothesis,
                    "count": len(rows),
                    "mean_anomaly_score": float(np.mean([float(row.get("anomaly_score", 0.0)) for row in rows])),
                }
            )
        return personas

    def _generate_theses(self, stats: dict[str, object], personas: list[dict[str, object]]) -> list[str]:
        theses = [
            f"Доминирующая гипотеза по датасету: {stats.get('dominant_hypothesis', 'H_UNCERTAIN')}.",
            f"Средняя уверенность по вердиктам: {float(stats.get('mean_confidence', 0.0)):.2f}.",
        ]
        if personas:
            top_era = max(personas, key=lambda item: float(item.get("mean_anomaly_score", 0.0)))
            theses.append(
                f"Самая тревожная эпоха: {top_era['era']} со средним anomaly_score {float(top_era['mean_anomaly_score']):.2f}."
            )
        return theses

    def _build_journalist_findings(
        self,
        verdicts: list[Stage5Record],
        timeline: list[dict[str, object]],
        chronology: dict[str, object],
    ) -> list[dict]:
        """Конвертирует verdicts/timeline в формат для JournalistThesisEngine."""
        findings = []
        
        # Группируем по гипотезам
        h1_photos = [v for v in verdicts if v.verdict.hypothesis.value == "H1_SYNTHETIC"]
        h2_photos = [v for v in verdicts if v.verdict.hypothesis.value == "H2_DIFFERENT"]
        
        # H1: силиконовые маски
        if h1_photos:
            for v in h1_photos[:10]:  # Топ-10
                evidence = v.verdict.evidence
                if evidence.get("h1_probability", 0) > 0.5:
                    # Ищем pair_id для этого фото
                    pair_ids = []
                    for row in timeline:
                        if row.get("photo_id") == v.photo_id:
                            pair_ids.append(f"{v.photo_id}__anchor")
                            break
                    
                    findings.append({
                        "type": "silicone_texture",
                        "params": {
                            "date": "N/A",
                            "texture_flags": f"anomaly_score={evidence.get('h1_probability', 0):.2f}",
                            "quality": evidence.get("quality", 0.5),
                        },
                        "photo_ids": [v.photo_id],
                        "pair_ids": pair_ids,
                        "confidence": evidence.get("h1_confidence", 0.5),
                    })
        
        # H2: другие люди
        if h2_photos:
            for v in h2_photos[:10]:
                evidence = v.verdict.evidence
                pair_ids = []
                for row in timeline:
                    if row.get("photo_id") == v.photo_id:
                        pair_ids.append(f"{v.photo_id}__anchor")
                        break
                
                findings.append({
                    "type": "bone_mismatch",
                    "params": {
                        "date": "N/A",
                        "bone_zone": "composite",
                        "delta_mm": evidence.get("geometry", 0) * 100,
                        "noise_mm": 1.5,
                        "ratio": evidence.get("geometry", 0) / 0.015 if evidence.get("geometry", 0) > 0 else 1.0,
                    },
                    "photo_ids": [v.photo_id],
                    "pair_ids": pair_ids,
                    "confidence": v.verdict.confidence,
                })
        
        # Хронологические аномалии
        chronology_points = chronology.get("points", [])
        for point in chronology_points:
            if point.get("flags"):
                for flag in point["flags"]:
                    if "spike" in flag or "age_inversion" in flag:
                        findings.append({
                            "type": "chronology_impossible",
                            "params": {
                                "date_a": point.get("date", "N/A"),
                                "date_b": point.get("date", "N/A"),
                                "gap_days": 0,
                                "body_part": "face geometry",
                                "min_days": 180,
                                "surgery_type": "biological constraint",
                            },
                            "photo_ids": [point.get("photo_id", "")],
                            "pair_ids": [],
                            "confidence": 0.7,
                        })
                        break
        
        return findings

    def _chronology_summary(self, chronology: dict[str, object]) -> dict[str, object]:
        points = chronology.get("points", []) if isinstance(chronology, dict) else []
        summary_flags = chronology.get("summary_flags", []) if isinstance(chronology, dict) else []
        anomaly_score = chronology.get("anomaly_score", 0.0) if isinstance(chronology, dict) else 0.0
        return {
            "point_count": len(points),
            "summary_flags": summary_flags,
            "anomaly_score": float(anomaly_score or 0.0),
        }

    def _to_markdown(self, report: Stage6Record) -> str:
        lines = [
            "# DEEPUTIN Forensic Report",
            "",
            f"- Total photos: {report.summary.get('total_photos', 0)}",
            f"- Dominant hypothesis: {report.summary.get('dominant_hypothesis', 'H_UNCERTAIN')}",
            f"- Top bucket: {report.summary.get('top_bucket', 'unknown')}",
            f"- Chronology anomaly score: {report.chronology_summary.get('anomaly_score', 0.0):.2f}",
            "",
        ]
        
        # Executive summary (если есть)
        exec_summary = report.summary.get("executive_summary")
        if exec_summary:
            lines.append(exec_summary)
            lines.append("")
        
        lines.append("## Theses")
        for thesis in report.theses:
            lines.append(f"- {thesis}")
        lines.append("")
        lines.append("## Verdict counts")
        for key, value in report.verdict_counts.items():
            lines.append(f"- {key}: {value}")

        # Ограничения методологии
        lines.extend(self._limitations_section(report))

        return "\n".join(lines)

    def _limitations_section(self, report: Stage6Record) -> list[str]:
        """Генерирует секцию 'Ограничения методологии'."""
        total = report.summary.get("total_photos", 0)
        low_quality = report.summary.get("low_quality_count", 0)
        cal_ref = report.summary.get("calibration_photo_count", 0)

        lines = [
            "",
            "## Ограничения методологии",
            "",
            "### Калибровка",
            f"- Калибровочный датасет: {cal_ref} фото (рекомендуется минимум 30 для робастности)",
            "- Пороги классификации установлены эмпирически и могут требовать перенастройки",
            "",
            "### Качество данных",
            f"- Всего фото: {total}",
            f"- Низкое качество (texture_unreliable): {low_quality} фото",
            "- Фото до 2005 года имеют повышенный уровень шума и JPEG-артефактов",
            "- Сканы документов могут содержать артефакты сканирования",
            "",
            "### Алгоритмические ограничения",
            "- Change point detection требует минимум 5 фото в кластере",
            "- Persona atlas зависит от качества 3D-реконструкции (3DDFA-V3)",
            "- Альтернативные объяснения проверяются автоматически, но не исключают ручную верификацию",
            "- Вероятности калиброваны через Platt scaling (Platt-like) — для изотонической регрессии нужен размеченный датасет",
            "",
            "### Рекомендации",
            "- Расширить калибровочный датасет до 30+ фронтальных фото",
            "- Разметить 100+ фото для blind test (H0/H1/H2 ground truth)",
            "- Провести кросс-валидацию на независимом датасете",
        ]
        return lines
