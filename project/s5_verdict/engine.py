from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from ..shared.logging import setup_logger
from ..shared.schemas import ForensicVerdict, Hypothesis, PairEvidence, Stage1Record, Stage2Record, Stage3Record, Stage5Record
from ..shared.utils import load_json, load_pickle, save_json
from .modules.chronology import ChronologyAnalyzer
from .biological_limits import BiologicalConstraintChecker
from .alpha_tracker import AlphaStabilityTracker
from .baseline_return import BaselineReturnDetector
from .h1_engine import H1SyntheticDetector
from .calibration_analysis import FittedPlattCalibrator, compute_ece_mce

logger = setup_logger("deeputin.s5")


class VerdictEngine:
    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}
        self.bio_checker = BiologicalConstraintChecker()
        self.alpha_tracker = AlphaStabilityTracker()
        self.baseline_detector = BaselineReturnDetector()
        self.h1_detector = H1SyntheticDetector()
        self._calibrator = FittedPlattCalibrator()

    def build_verdicts(self, main_root: str | Path) -> tuple[list[Stage5Record], list[dict[str, object]]]:
        root = Path(main_root)
        pair_index = self._load_pair_index(root)
        stage1_records = self._load_stage1_records(root)
        stage2_records = self._load_stage2_records(root)
        stage3_records = self._load_stage3_records(root)
        chronology = ChronologyAnalyzer().build(stage1_records, stage2_records)
        chronology_map = {point.photo_id: point for point in chronology.points}
        chronology_flags = set(chronology.summary_flags)

        # --- Baseline return detection: compute for all photos upfront ---
        baseline_returns = self._compute_baseline_returns(stage2_records, stage1_records)

        # --- Alpha stability: cluster identity vectors ---
        alpha_timeline = self._compute_alpha_clusters(stage1_records, root)

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

        for record in stage2_records.values():
            stage1 = stage1_records.get(record.photo_id)
            stage3 = stage3_records.get(record.photo_id)
            pairs = pair_index.get(record.photo_id, [])
            chronology_point = chronology_map.get(record.photo_id)
            verdict = self._verdict_for_photo(record, stage1, stage3, pairs, stage2_records, stage1_records, chronology_point, chronology, chronology_flags, baseline_returns, alpha_timeline)
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
        
        # Calibration report
        if verdicts:
            y_true = np.array([1 if v.hypothesis.value == "H0_SAME" else 0 for v in verdicts])
            y_prob = np.array([v.posterior.get("H0_SAME", 0.5) for v in verdicts])
            if len(np.unique(y_true)) > 1:
                cal_metrics = compute_ece_mce(y_true, y_prob, n_bins=5)
                save_json({
                    "ece": cal_metrics.ece,
                    "mce": cal_metrics.mce,
                    "n_bins": cal_metrics.n_bins,
                    "total_samples": cal_metrics.total_samples,
                    "bins": [
                        {
                            "bin_lower": b.bin_lower,
                            "bin_upper": b.bin_upper,
                            "mean_predicted": b.mean_predicted,
                            "mean_observed": b.mean_observed,
                            "count": b.count,
                        }
                        for b in cal_metrics.bins
                    ],
                }, root / "calibration_report.json")
        
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
        baseline_returns: dict[str, list[dict]] | None = None,
        alpha_timeline: dict | None = None,
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
        texture_unreliable = bool(record.quality_summary.get("texture_unreliable", False))
        identity_hint = stage3.identity_hint if stage3 else "PUT"
        skin_hint = stage3.skin_hint if stage3 else "real"
        chronology_score = float(chronology_point.chronology_score if chronology_point else self._chronology_score(record.photo_id, stage2_records, stage1_records))
        identity_boost = 0.12 if identity_hint == "PUT" else 0.0
        silicone_boost = 0.2 if skin_hint == "silicone" else 0.0
        
        # --- Новые модули: биологические ограничения ---
        bio_flags = self._check_biological_constraints(record, stage1, stage2_records, stage1_records)
        
        # --- Новые модули: H1 детекция синтетики ---
        h1_result = self._detect_h1_synthetic(record, stage1, pairs, bio_flags)
        
        # --- Baseline return flags ---
        photo_baseline_flags = baseline_returns.get(record.photo_id, []) if baseline_returns else []
        has_baseline_return = len(photo_baseline_flags) > 0
        
        # --- Alpha cluster info ---
        photo_alpha_anomaly = False
        if alpha_timeline and alpha_timeline.get("anomalies"):
            for anom in alpha_timeline["anomalies"]:
                if anom.get("photo_id") == record.photo_id:
                    photo_alpha_anomaly = True
                    break

        # --- Physical texture features (from metric_notes) ---
        phys = {}
        for k, v in record.metric_notes.items():
            if k.startswith("physical_"):
                try:
                    phys[k.replace("physical_", "")] = float(v)
                except (ValueError, TypeError):
                    pass
        silicone_physical_boost = (
            phys.get("sss_index", 0.0) * 0.30
            + phys.get("specular_sharpness", 0.0) * 0.20
            + phys.get("seam_score", 0.0) * 0.25
            + phys.get("spectral_slope", 0.0) * 0.15
            + phys.get("lbp_nonuniform_ratio", 0.0) * 0.10
            + phys.get("wrinkle_anisotropy", 0.0) * 0.10
        )

        priors = {
            Hypothesis.H0_SAME.value: 0.52,
            Hypothesis.H1_SYNTHETIC.value: 0.18,
            Hypothesis.H2_DIFFERENT.value: 0.20,
            Hypothesis.H_UNCERTAIN.value: 0.10,
        }
        likelihoods = {
            Hypothesis.H0_SAME.value: max(1e-6, np.exp(-(avg_geom * 0.85 + avg_tex * 0.25 + chronology_score * 0.7 + avg_age_gap * 0.08)) * (0.65 + 0.35 * quality + identity_boost)),
            Hypothesis.H1_SYNTHETIC.value: max(1e-6, np.clip((avg_tex * 0.6 + avg_syn * 0.5 + silicone_boost + silicone_physical_boost * 0.35 + h1_result.get("h1_probability", 0.0) * 0.4) * (0.8 + 0.2 * quality_penalty), 0.01, 2.0)),
            Hypothesis.H2_DIFFERENT.value: max(1e-6, np.clip((avg_geom * 0.7 + avg_diff * 0.6 + max(0.0, avg_age_gap - avg_age_explained) * 0.12 + len(bio_flags) * 0.08) * (0.9 + 0.1 * chronology_score) + (0.15 if identity_hint == "OTHER" else 0.0), 0.01, 2.0)),
            Hypothesis.H_UNCERTAIN.value: max(1e-6, np.clip(1.1 - quality + avg_anomaly * 0.12 + abs(avg_same - avg_diff) * 0.15, 0.01, 2.0)),
        }
        if "age_inversion_detected" in chronology_flags or "strong_temporal_break" in chronology_flags:
            likelihoods[Hypothesis.H2_DIFFERENT.value] *= 1.18
            likelihoods[Hypothesis.H_UNCERTAIN.value] *= 1.12
            likelihoods[Hypothesis.H0_SAME.value] *= 0.88
        if has_baseline_return:
            likelihoods[Hypothesis.H1_SYNTHETIC.value] *= 1.25
            likelihoods[Hypothesis.H2_DIFFERENT.value] *= 1.15
            likelihoods[Hypothesis.H0_SAME.value] *= 0.82
        if photo_alpha_anomaly:
            likelihoods[Hypothesis.H1_SYNTHETIC.value] *= 1.20
            likelihoods[Hypothesis.H2_DIFFERENT.value] *= 1.18
            likelihoods[Hypothesis.H0_SAME.value] *= 0.80
        if chronology.summary_flags and chronology_score > 0.9:
            likelihoods[Hypothesis.H_UNCERTAIN.value] *= 1.15
        if texture_unreliable:
            # Текстура ненадёжна — ослабляем её вклад в H1/H2
            likelihoods[Hypothesis.H1_SYNTHETIC.value] *= 0.80
            likelihoods[Hypothesis.H2_DIFFERENT.value] *= 0.85
            likelihoods[Hypothesis.H_UNCERTAIN.value] *= 1.25
        posterior_raw = self._bayes(priors, likelihoods)
        posterior = self._calibrator.calibrate(posterior_raw, quality)
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
        if h1_result.get("is_triggered") and hypothesis == Hypothesis.H0_SAME.value:
            hypothesis = Hypothesis.H1_SYNTHETIC.value
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
            f"bio_flags={len(bio_flags)}",
            f"h1_probability={h1_result.get('h1_probability', 0.0):.3f}",
            f"h1_triggered={h1_result.get('is_triggered', False)}",
            f"baseline_return_count={len(photo_baseline_flags)}",
            f"alpha_cluster_anomaly={photo_alpha_anomaly}",
            f"texture_unreliable={texture_unreliable}",
            f"phys_sss={phys.get('sss_index', 0.0):.3f}",
            f"phys_specular={phys.get('specular_sharpness', 0.0):.3f}",
            f"phys_seam={phys.get('seam_score', 0.0):.3f}",
            f"phys_spectral={phys.get('spectral_slope', 0.0):.3f}",
            f"phys_physical_boost={silicone_physical_boost:.3f}",
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
            "biological_flags_count": len(bio_flags),
            "h1_probability": h1_result.get("h1_probability", 0.0),
            "h1_confidence": h1_result.get("confidence", 0.0),
            "baseline_return_flags": float(len(photo_baseline_flags)),
            "alpha_cluster_anomaly": float(photo_alpha_anomaly),
            "texture_unreliable": float(texture_unreliable),
            "phys_sss_index": phys.get("sss_index", 0.0),
            "phys_specular_sharpness": phys.get("specular_sharpness", 0.0),
            "phys_seam_score": phys.get("seam_score", 0.0),
            "phys_spectral_slope": phys.get("spectral_slope", 0.0),
            "phys_silicone_physical_boost": silicone_physical_boost,
        }
        return ForensicVerdict(
            photo_id=record.photo_id,
            hypothesis=Hypothesis(hypothesis),
            posterior=posterior,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            reasoning=reasoning,
            evidence=evidence,
        )

    def _compute_alpha_clusters(
        self,
        stage1_records: dict[str, Stage1Record],
        root: Path,
    ) -> dict | None:
        """Загружает id_params из reconstruction.pkl и кластеризует через AlphaStabilityTracker."""
        records_with_alpha = []
        for photo_id, s1 in stage1_records.items():
            try:
                rec_path = root / photo_id / "reconstruction.pkl"
                if not rec_path.exists():
                    continue
                rec = load_pickle(rec_path)
                id_params = rec.get("id_params")
                if id_params is None:
                    continue
                id_vec = np.array(id_params, dtype=np.float32).flatten()
                if id_vec.size == 0:
                    continue
                records_with_alpha.append({
                    "photo_id": photo_id,
                    "date": s1.date,
                    "alpha": id_vec,
                    "quality": float(s1.quality.overall_quality) if s1 else 0.5,
                })
            except Exception:
                continue
        
        if len(records_with_alpha) < 5:
            return None
        
        try:
            return self.alpha_tracker.build_timeline(records_with_alpha)
        except Exception:
            return None

    def _compute_baseline_returns(
        self,
        stage2_records: dict[str, Stage2Record],
        stage1_records: dict[str, Stage1Record],
    ) -> dict[str, list[dict]]:
        """Вычисляет baseline return flags для каждого фото."""
        from collections import defaultdict
        bucket_series: dict[str, list[tuple[str, float, object]]] = defaultdict(list)
        for photo_id, s2 in stage2_records.items():
            s1 = stage1_records.get(photo_id)
            if s1 is None or s1.date is None:
                continue
            avg_geom = float(np.mean(list(s2.geometry.values()))) if s2.geometry else 0.0
            bucket_series[s2.bucket.value].append((photo_id, avg_geom, s1.date))
        
        result: dict[str, list[dict]] = defaultdict(list)
        for bucket, series in bucket_series.items():
            series_sorted = sorted(series, key=lambda x: x[2])
            if len(series_sorted) < 10:
                continue
            metric_vals = [s[1] for s in series_sorted]
            dates_list = [s[2] for s in series_sorted]
            photo_ids = [s[0] for s in series_sorted]
            flags_all = self.baseline_detector.detect(metric_vals, dates_list)
            for flag in flags_all:
                dev_idx = None
                ret_idx = None
                for idx, (pid, _, d) in enumerate(series_sorted):
                    if d == flag.get("deviation_start"):
                        dev_idx = idx
                    if d == flag.get("return_date"):
                        ret_idx = idx
                if dev_idx is not None:
                    result[photo_ids[dev_idx]].append(flag)
                if ret_idx is not None:
                    result[photo_ids[ret_idx]].append(flag)
        return dict(result)

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

    def _check_biological_constraints(
        self,
        record: Stage2Record,
        stage1: Stage1Record | None,
        stage2_records: dict[str, Stage2Record],
        stage1_records: dict[str, Stage1Record],
    ) -> list[dict]:
        """Проверяет биологическую невозможность изменений между фото."""
        bio_flags = []
        if stage1 is None or stage1.date is None:
            return bio_flags
        
        current_metrics = record.geometry
        current_date = stage1.date
        
        # Сравниваем с предыдущими фото того же bucket
        for other_id, other_stage1 in stage1_records.items():
            if other_id == record.photo_id:
                continue
            if other_stage1.date is None:
                continue
            other_record = stage2_records.get(other_id)
            if other_record is None:
                continue
            if other_record.bucket != record.bucket:
                continue
            
            try:
                photo_a = {"date": current_date}
                photo_b = {"date": other_stage1.date}
                flags = self.bio_checker.check(photo_a, photo_b, current_metrics, other_record.geometry)
                bio_flags.extend(flags)
            except Exception:
                continue
        
        return bio_flags

    def _detect_h1_synthetic(
        self,
        record: Stage2Record,
        stage1: Stage1Record | None,
        pairs: list[PairEvidence],
        bio_flags: list[dict],
    ) -> dict:
        """Детекция синтетики H1 через комплексный анализ."""
        # Формируем anchor_comparison из pairs
        anchor_comparison = {}
        if pairs:
            best_pair = max(pairs, key=lambda p: p.confidence if hasattr(p, 'confidence') else 0.5)
            anchor_comparison = {
                "excess_distance": best_pair.geometry_distance,
                "heatmap_mean": best_pair.texture_distance,
                "heatmap_max": max(p.geometry_distance for p in pairs),
                "bone_zone_violations": sum(1 for p in pairs if p.geometry_distance > 1.0),
            }
        
        # Формируем texture_anomaly
        texture_anomaly = {
            "anomaly_score": float(np.mean([p.texture_distance for p in pairs]) if pairs else 0.0),
            "feature_flags": {},
        }
        
        # Формируем photo info
        photo_info = {
            "year": stage1.date.year if stage1 and stage1.date else 2000,
            "quality": float(stage1.quality.overall_quality) if stage1 else 0.5,
        }
        
        return self.h1_detector.evaluate(photo_info, anchor_comparison, texture_anomaly, bio_flags)
