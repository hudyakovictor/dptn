from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ..shared.logging import setup_logger
from ..shared.schemas import PipelineDataset, Stage5Record, Stage6Record
from ..shared.utils import load_json, save_json, save_text

logger = setup_logger("deeputin.s6")


class ReportEngine:
    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}

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
        personas = self._build_personas(timeline, verdicts)
        theses = self._generate_theses(stats, personas)
        report = Stage6Record(
            dataset=PipelineDataset.MAIN,
            generated_at=__import__("datetime").datetime.utcnow().isoformat(timespec="seconds"),
            summary={
                "total_photos": len(verdicts),
                "dominant_hypothesis": stats["dominant_hypothesis"],
                "top_bucket": stats["top_bucket"],
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

    def _build_personas(self, timeline: list[dict[str, object]], verdicts: list[Stage5Record]) -> list[dict[str, object]]:
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
            "## Theses",
        ]
        for thesis in report.theses:
            lines.append(f"- {thesis}")
        lines.append("")
        lines.append("## Verdict counts")
        for key, value in report.verdict_counts.items():
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)
