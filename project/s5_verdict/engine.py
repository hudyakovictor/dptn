from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from ..shared.logging import setup_logger
from ..shared.schemas import ForensicVerdict, Hypothesis, PairEvidence, Stage1Record, Stage2Record, Stage3Record, Stage5Record
from ..shared.utils import load_json, save_json
from .modules.chronology import ChronologyAnalyzer

logger = setup_logger("deeputin.s5")


class VerdictEngine:
    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}

    def build_verdicts(self, main_root: str | Path) -> tuple[list[Stage5Record], list[dict[str, object]]]:
        root = Path(main_root)
        pair_index = self._load_pair_index(root)
        stage1_records = self._load_stage1_records(root)
        stage2_records = self._load_stage2_records(root)
        stage3_records = self._load_stage3_records(root)
        chronology = ChronologyAnalyzer().build(stage1_records, stage2_records)
        chronology_map = {point.photo_id: point for point in chronology.points}
        chronology_flags = set(chronology.summary_flags)
        save_json(
            {
                "summary_flags": chronology.summary_flags,
                "anomaly_score": chronology.anomaly_score,
                "points": [
                    {
                        "photo_id": point.photo_id,
                        "date": point.date.isoformat() if point.date else None,
                        "bucket": point.bucket,
                        "age_years": point.age_years,
                        "chronology_score": point.chronology_score,
                        "flags": point.flags,
                        "details": point.details,
                    }
                    for point in chronology.points
                ],
            },
            root / "chronology.json",
        )

        verdicts: list[Stage5Record] = []
        timeline: list[dict[str, object]] = []

        for record in stage2_records:
            stage1 = stage1_records.get(record.photo_id)
            stage3 = stage3_records.get(record.photo_id)
            pairs = pair_index.get(record.photo_id, [])
            chronology_point = chronology_map.get(record.photo_id)
            verdict = self._verdict_for_photo(record, stage1, stage3, pairs, stage2_records, stage1_records, chronology_point, chronology, chronology_flags)
            stage5 = Stage5Record(
                photo_id=record.photo_id,
                dataset=record.dataset,
                verdict=verdict,
                chronology_score=float(chronology_point.chronology_score if chronology_point else verdict.evidence.get("chronology", 0.0)),
                anomaly_score=float(verdict.evidence.get("anomaly", 0.0)),
            )
            verdicts.append(stage5)
            photo_dir = root / record.photo_id
            save_json(stage5.model_dump(), photo_dir / "verdict.json")
            timeline.append(
                {
                    "photo_id": record.photo_id,
                    "bucket": record.bucket.value,
                    "date": stage1.date.isoformat() if stage1 and stage1.date else None,
                    "age_years": stage1.age_years if stage1 else None,
                    "anomaly_score": stage5.anomaly_score,
                    "hypothesis": verdict.hypothesis.value,
                    "confidence": verdict.confidence,
                }
            )

        save_json([v.model_dump() for v in verdicts], root / "verdicts.json")
        save_json(timeline, root / "timeline.json")
        return verdicts, timeline

    def _load_pair_index(self, root: Path) -> dict[str, list[PairEvidence]]:
        payload = load_json(root / "pair_index.json", default={}) or {}
        result: dict[str, list[PairEvidence]] = defaultdict(list)
        for photo_id, items in payload.items():
            for item in items:
                result[photo_id].append(PairEvidence.model_validate(item))
        return result

    def _load_stage1_records(self, root: Path) -> dict[str, Stage1Record]:
        records: dict[str, Stage1Record] = {}
        for path in sorted(root.glob("*/info.json")):
            payload = load_json(path)
            if payload:
                record = Stage1Record.model_validate(payload)
                records[record.photo_id] = record
        return records

    def _load_stage2_records(self, root: Path) -> dict[str, Stage2Record]:
        records: dict[str, Stage2Record] = {}
        for path in sorted(root.glob("*/metrics.json")):
            payload = load_json(path)
            if payload:
                record = Stage2Record.model_validate(payload)
                records[record.photo_id] = record
        return records

    def _load_stage3_records(self, root: Path) -> dict[str, Stage3Record]:
        records: dict[str, Stage3Record] = {}
        for path in sorted(root.glob("*/identity.json")):
            payload = load_json(path)
            if payload:
                record = Stage3Record.model_validate(payload)
                records[record.photo_id] = record
        return records

    def _verdict_for_photo(
        self,
        record: Stage2Record,
        stage1: Stage1Record | None,
        stage3: Stage3Record | None,
        pairs: list[PairEvidence],
        stage2_records: dict[str, Stage2Record],
        stage1_records: dict[str, Stage1Record],
        chronology_point,
        chronology,
        chronology_flags: set[str],
    ) -> ForensicVerdict:
        avg_geom = float(np.mean([p.geometry_distance for p in pairs]) if pairs else 0.0)
        avg_tex = float(np.mean([p.texture_distance for p in pairs]) if pairs else 0.0)
        avg_same = float(np.mean([p.same_suspicion for p in pairs]) if pairs else 0.0)
        avg_diff = float(np.mean([p.different_suspicion for p in pairs]) if pairs else 0.0)
        avg_syn = float(np.mean([p.synthetic_suspicion for p in pairs]) if pairs else 0.0)
        avg_age_gap = float(np.mean([p.age_gap_years for p in pairs]) if pairs else 0.0)
        avg_age_explained = float(np.mean([p.age_explained_distance for p in pairs]) if pairs else 0.0)
        avg_anomaly = float(np.mean([len(p.anomaly_flags) for p in pairs]) if pairs else 0.0)
        quality = float(stage1.quality.overall_quality) if stage1 else 0.5
        quality_penalty = float(np.clip(1.0 - quality, 0.0, 1.0))
        identity_hint = stage3.identity_hint if stage3 else "PUT"
        skin_hint = stage3.skin_hint if stage3 else "real"
        chronology_score = float(chronology_point.chronology_score if chronology_point else self._chronology_score(record.photo_id, stage2_records, stage1_records))
        identity_boost = 0.12 if identity_hint == "PUT" else 0.0
        silicone_boost = 0.2 if skin_hint == "silicone" else 0.0
        priors = {
            Hypothesis.H0_SAME.value: 0.52,
            Hypothesis.H1_SYNTHETIC.value: 0.18,
            Hypothesis.H2_DIFFERENT.value: 0.20,
            Hypothesis.H_UNCERTAIN.value: 0.10,
        }
        likelihoods = {
            Hypothesis.H0_SAME.value: max(1e-6, np.exp(-(avg_geom * 0.85 + avg_tex * 0.25 + chronology_score * 0.7 + avg_age_gap * 0.08)) * (0.65 + 0.35 * quality + identity_boost)),
            Hypothesis.H1_SYNTHETIC.value: max(1e-6, np.clip((avg_tex * 0.6 + avg_syn * 0.5 + silicone_boost) * (0.8 + 0.2 * quality_penalty), 0.01, 2.0)),
            Hypothesis.H2_DIFFERENT.value: max(1e-6, np.clip((avg_geom * 0.7 + avg_diff * 0.6 + max(0.0, avg_age_gap - avg_age_explained) * 0.12) * (0.9 + 0.1 * chronology_score) + (0.15 if identity_hint == "OTHER" else 0.0), 0.01, 2.0)),
            Hypothesis.H_UNCERTAIN.value: max(1e-6, np.clip(1.1 - quality + avg_anomaly * 0.12 + abs(avg_same - avg_diff) * 0.15, 0.01, 2.0)),
        }
        if "age_inversion_detected" in chronology_flags or "strong_temporal_break" in chronology_flags:
            likelihoods[Hypothesis.H2_DIFFERENT.value] *= 1.18
            likelihoods[Hypothesis.H_UNCERTAIN.value] *= 1.12
            likelihoods[Hypothesis.H0_SAME.value] *= 0.88
        if chronology.summary_flags and chronology_score > 0.9:
            likelihoods[Hypothesis.H_UNCERTAIN.value] *= 1.15
        posterior = self._bayes(priors, likelihoods)
        hypothesis = max(posterior, key=posterior.get)
        sorted_scores = sorted(posterior.values(), reverse=True)
        confidence = float((sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else sorted_scores[0])
        confidence = float(np.clip(confidence + max(0.0, sorted_scores[0] - 0.5) * 0.35, 0.0, 1.0))
        if confidence < 0.08 or quality < 0.18 or sorted_scores[0] < 0.35:
            hypothesis = Hypothesis.H_UNCERTAIN.value
        if chronology.summary_flags and chronology_score > 0.85 and hypothesis == Hypothesis.H0_SAME.value:
            hypothesis = Hypothesis.H_UNCERTAIN.value
        if avg_syn > 0.72 and avg_geom < 0.9 and hypothesis == Hypothesis.H0_SAME.value:
            hypothesis = Hypothesis.H1_SYNTHETIC.value
        if avg_diff > 0.75 and chronology_score > 0.35 and hypothesis == Hypothesis.H0_SAME.value:
            hypothesis = Hypothesis.H2_DIFFERENT.value
        reasoning = [
            f"quality={quality:.2f}",
            f"avg_geom={avg_geom:.2f}",
            f"avg_tex={avg_tex:.2f}",
            f"chronology_score={chronology_score:.2f}",
            f"avg_same={avg_same:.2f}",
            f"avg_diff={avg_diff:.2f}",
            f"avg_syn={avg_syn:.2f}",
            f"avg_age_gap={avg_age_gap:.2f}",
            f"avg_age_explained={avg_age_explained:.2f}",
            f"chronology_flags={','.join(chronology_point.flags) if chronology_point else ''}",
            f"chronology_summary_flags={','.join(sorted(chronology_flags))}",
            f"skin_hint={skin_hint}",
            f"identity_hint={identity_hint}",
        ]
        evidence = {
            "geometry": avg_geom,
            "texture": avg_tex,
            "quality": quality,
            "chronology": float(avg_anomaly),
            "anomaly": float(avg_anomaly + avg_tex * 0.5),
            "same_suspicion": avg_same,
            "different_suspicion": avg_diff,
            "synthetic_suspicion": avg_syn,
            "age_gap_years": avg_age_gap,
            "age_explained_distance": avg_age_explained,
            "chronology_score": chronology_score,
            "chronology_flags": float(len(chronology_point.flags)) if chronology_point else 0.0,
            "chronology_anomaly_score": float(chronology.anomaly_score),
        }
        return ForensicVerdict(
            photo_id=record.photo_id,
            hypothesis=Hypothesis(hypothesis),
            posterior=posterior,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            reasoning=reasoning,
            evidence=evidence,
        )

    def _bayes(self, priors: dict[str, float], likelihoods: dict[str, float]) -> dict[str, float]:
        scores = {key: priors[key] * likelihoods[key] for key in priors}
        total = sum(scores.values()) or 1.0
        return {key: value / total for key, value in scores.items()}

    def _chronology_score(
        self,
        photo_id: str,
        stage2_records: dict[str, Stage2Record],
        stage1_records: dict[str, Stage1Record],
    ) -> float:
        current = stage1_records.get(photo_id)
        if current is None or current.date is None:
            return 0.0
        current_record = stage2_records.get(photo_id)
        if current_record is None:
            return 0.0
        neighbors = []
        for other_id, other in stage1_records.items():
            if other_id == photo_id or other.date is None:
                continue
            if other_record := stage2_records.get(other_id):
                if other_record.bucket != current_record.bucket:
                    continue
                gap_days = abs((other.date - current.date).days)
                if gap_days <= 365:
                    neighbors.append((gap_days, other_record))
        if not neighbors:
            return 0.0
        gaps = []
        for gap_days, other_record in neighbors:
            geom = self._rough_distance(current_record.geometry, other_record.geometry)
            tex = self._rough_distance(current_record.texture, other_record.texture)
            gaps.append(np.clip((geom + tex) * (1.0 + gap_days / 365.0), 0.0, 6.0))
        return float(np.mean(gaps) / 6.0)

    def _rough_distance(self, a: dict[str, float], b: dict[str, float]) -> float:
        keys = sorted(set(a) & set(b))
        if not keys:
            return 0.0
        vals = []
        for key in keys:
            va = float(a[key])
            vb = float(b[key])
            scale = max(abs(va), abs(vb), 1.0)
            vals.append(abs(va - vb) / scale)
        return float(np.mean(vals) * 3.0)
