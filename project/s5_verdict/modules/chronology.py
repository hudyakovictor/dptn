from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from statistics import median
from typing import Dict, List, Optional, Set

import numpy as np

from deeputin.shared.schemas import Stage1Record, Stage2Record


# Biological limits (from audit document)
BIOLOGICAL_LIMITS = {
    "rhinoplasty_healing_days": 180,
    "facelift_healing_days": 90,
    "blepharoplasty_healing_days": 60,
    "implant_settling_days": 365,
    "max_bone_change_mm_per_year": 0.5,
    "max_asymmetry_inversion_days": 365 * 5,
    "max_bone_change_mm_short_gap": 2.0,  # max bone change for < 1 year gap
    "short_gap_days_threshold": 365,
}

# Bone-stable metrics that should not change rapidly
BONE_METRICS = {
    "bone_nasion_depth",
    "bone_orbit_L_depth",
    "bone_orbit_R_depth",
    "bone_zygomatic_width",
    "bone_gonial_angle",
    "bone_chin_projection",
    "bone_asymmetry_x",
    "mesh_symmetry_x",
    "id_norm",  # identity vector norm
    "id_mean",
    "id_std",
}


@dataclass
class ChronologyPoint:
    photo_id: str
    date: date | None
    bucket: str
    age_years: float | None
    chronology_score: float = 0.0
    flags: list[str] = field(default_factory=list)
    details: dict[str, float] = field(default_factory=dict)


@dataclass
class ChronologyResult:
    points: list[ChronologyPoint]
    summary_flags: list[str]
    anomaly_score: float


class ChronologyAnalyzer:
    """Chronology analyzer with biological impossibility checks."""

    def build(self, stage1_records: dict[str, Stage1Record], stage2_records: dict[str, Stage2Record]) -> ChronologyResult:
        ordered = sorted(
            [
                (photo_id, rec)
                for photo_id, rec in stage1_records.items()
                if photo_id in stage2_records and rec.date is not None
            ],
            key=lambda item: (item[1].date.isoformat(), item[0]),
        )
        points: list[ChronologyPoint] = []
        for photo_id, rec in ordered:
            points.append(
                ChronologyPoint(
                    photo_id=photo_id,
                    date=rec.date,
                    bucket=stage2_records[photo_id].bucket.value,
                    age_years=rec.age_years,
                )
            )

        if len(points) < 2:
            return ChronologyResult(points=points, summary_flags=[], anomaly_score=0.0)

        per_metric_series = self._series(stage2_records, ordered)
        per_photo_scores: dict[str, float] = {p.photo_id: 0.0 for p in points}
        per_photo_flags: dict[str, Set[str]] = {p.photo_id: set() for p in points}

        for metric_name, values in per_metric_series.items():
            if len(values) < 3:
                continue
            deltas = np.abs(np.diff(values))
            baseline = float(np.median(deltas)) + 1e-6
            robust_scale = float(np.median(np.abs(deltas - baseline)) + 1e-6)
            age_trend = self._age_trend(points, values)
            for idx in range(1, len(values)):
                gap_days = max((points[idx].date - points[idx - 1].date).days if points[idx].date and points[idx - 1].date else 1, 1)
                rate = float(abs(values[idx] - values[idx - 1]) / gap_days)
                normalized = rate / max(baseline, 1e-6)
                if normalized > 2.2:
                    flag = f"spike:{metric_name}"
                    per_photo_scores[points[idx].photo_id] += 0.8
                    per_photo_flags[points[idx].photo_id].add(flag)
                    per_photo_flags[points[idx - 1].photo_id].add(flag)
                elif normalized > 1.3:
                    flag = f"elevated_change:{metric_name}"
                    per_photo_scores[points[idx].photo_id] += 0.35
                    per_photo_flags[points[idx].photo_id].add(flag)

                # Age inversion
                if age_trend and points[idx].age_years is not None and points[idx - 1].age_years is not None:
                    age_delta = float(points[idx].age_years - points[idx - 1].age_years)
                    if age_delta > 0:
                        direction = float(values[idx] - values[idx - 1])
                        if age_trend["slope"] > 0 and direction < -robust_scale * 0.8:
                            per_photo_scores[points[idx].photo_id] += 0.45
                            per_photo_flags[points[idx].photo_id].add(f"age_inversion:{metric_name}")
                        elif age_trend["slope"] < 0 and direction > robust_scale * 0.8:
                            per_photo_scores[points[idx].photo_id] += 0.45
                            per_photo_flags[points[idx].photo_id].add(f"age_inversion:{metric_name}")

            self._detect_return_to_baseline(metric_name, values, points, per_photo_scores, per_photo_flags)

        # Biological impossibility checks (NEW)
        self._check_biological_impossibilities(points, per_metric_series, per_photo_scores, per_photo_flags)

        # Change point detection (ruptures Pelt)
        self._detect_change_points(points, per_metric_series, per_photo_scores, per_photo_flags)

        # Cross-metric return-to-baseline: если geometry + texture одновременно вернулись
        self._detect_cross_metric_return(points, per_metric_series, per_photo_scores, per_photo_flags)

        for point in points:
            point.chronology_score = float(np.clip(per_photo_scores[point.photo_id], 0.0, 3.0))
            point.flags = sorted(per_photo_flags[point.photo_id])
            point.details["age_years"] = float(point.age_years) if point.age_years is not None else 0.0

        summary_flags = self._summary_flags(points)
        anomaly_score = float(np.mean([p.chronology_score for p in points]))
        return ChronologyResult(points=points, summary_flags=summary_flags, anomaly_score=anomaly_score)

    def _age_trend(self, points: list[ChronologyPoint], values: list[float]) -> dict[str, float] | None:
        ages = []
        vals = []
        for point, value in zip(points, values):
            if point.age_years is None or not np.isfinite(value):
                continue
            ages.append(float(point.age_years))
            vals.append(float(value))
        if len(ages) < 4:
            return None
        age_arr = np.asarray(ages, dtype=float)
        val_arr = np.asarray(vals, dtype=float)
        if np.std(age_arr) < 1e-6 or np.std(val_arr) < 1e-6:
            return None
        slope = float(np.polyfit(age_arr, val_arr, 1)[0])
        corr = float(np.corrcoef(age_arr, val_arr)[0, 1])
        if not np.isfinite(slope) or not np.isfinite(corr):
            return None
        if abs(corr) < 0.35:
            return None
        return {"slope": slope, "corr": corr}

    def _series(
        self,
        stage2_records: dict[str, Stage2Record],
        ordered: list[tuple[str, Stage1Record]],
    ) -> dict[str, list[float]]:
        # Get all metric keys from first record to determine what to track
        if not ordered:
            return {}
        first_rec = stage2_records[ordered[0][0]]
        all_keys = set(first_rec.geometry.keys()) | set(first_rec.texture.keys())
        # Prioritize bone metrics and key texture metrics
        priority_keys = [k for k in all_keys if k.startswith(("bone_", "mesh_", "id_", "exp_", "texture_"))]
        keys = sorted(priority_keys) if priority_keys else sorted(all_keys)
        
        series: dict[str, list[float]] = {key: [] for key in keys}
        for photo_id, _ in ordered:
            record = stage2_records[photo_id]
            merged = {**record.geometry, **record.texture}
            for key in keys:
                value = merged.get(key)
                if isinstance(value, (int, float)):
                    series[key].append(float(value))
                else:
                    series[key].append(float("nan"))
        return series

    def _check_biological_impossibilities(
        self,
        points: List[ChronologyPoint],
        per_metric_series: Dict[str, List[float]],
        per_photo_scores: Dict[str, float],
        per_photo_flags: Dict[str, Set[str]],
    ) -> None:
        """Check for biologically impossible changes."""
        if len(points) < 2:
            return

        for idx in range(1, len(points)):
            p_prev = points[idx - 1]
            p_curr = points[idx]
            
            if p_prev.date is None or p_curr.date is None:
                continue
            
            gap_days = (p_curr.date - p_prev.date).days
            if gap_days <= 0:
                continue

            # Check each bone metric
            for metric_name, values in per_metric_series.items():
                is_bone_metric = any(bm in metric_name for bm in ["bone_", "id_norm", "id_mean", "id_std", "mesh_symmetry"])
                
                if idx >= len(values) or idx - 1 >= len(values):
                    continue
                    
                val_prev = values[idx - 1]
                val_curr = values[idx]
                
                if not np.isfinite(val_prev) or not np.isfinite(val_curr):
                    continue

                delta = abs(val_curr - val_prev)

                # 1. Bone metrics should not change rapidly
                if is_bone_metric:
                    # For short gaps (< 1 year), bone change should be minimal
                    if gap_days < BIOLOGICAL_LIMITS["short_gap_days_threshold"]:
                        if delta > BIOLOGICAL_LIMITS["max_bone_change_mm_short_gap"]:
                            per_photo_scores[p_curr.photo_id] += 1.2
                            per_photo_flags[p_curr.photo_id].add(f"IMPOSSIBLE_BONE_CHANGE:{metric_name}:delta={delta:.3f}:gap_days={gap_days}")
                            per_photo_flags[p_prev.photo_id].add(f"IMPOSSIBLE_BONE_CHANGE:{metric_name}:delta={delta:.3f}:gap_days={gap_days}")
                    
                    # Max bone change per year
                    max_allowed_per_year = BIOLOGICAL_LIMITS["max_bone_change_mm_per_year"]
                    max_allowed = max_allowed_per_year * (gap_days / 365.25)
                    if delta > max_allowed * 5:  # 5x the annual limit
                        per_photo_scores[p_curr.photo_id] += 1.0
                        per_photo_flags[p_curr.photo_id].add(f"IMPOSSIBLE_BONE_RATE:{metric_name}:delta={delta:.3f}:gap_days={gap_days}")

                # 2. Asymmetry inversion (mirror flip)
                if metric_name in ["bone_asymmetry_x", "mesh_symmetry_x"]:
                    if val_prev * val_curr < 0 and abs(val_prev) > 1.0 and abs(val_curr) > 1.0:
                        # Sign flipped with significant magnitude
                        per_photo_scores[p_curr.photo_id] += 1.5
                        per_photo_flags[p_curr.photo_id].add(f"IMPOSSIBLE_ASYMMETRY_INVERSION:{metric_name}:{val_prev:.3f}->{val_curr:.3f}")

                # 3. Return to baseline (mask swap indicator)
                if idx >= 3 and idx + 1 < len(values):
                    # Check if value went away from baseline then returned
                    baseline = values[0] if np.isfinite(values[0]) else np.nanmedian([v for v in values[:3] if np.isfinite(v)])
                    if np.isfinite(baseline):
                        deviation = abs(val_curr - baseline)
                        earlier_vals = [values[i] for i in range(max(0, idx - 2), idx) if np.isfinite(values[i])]
                        if earlier_vals:
                            earlier_deviation = max(abs(v - baseline) for v in earlier_vals)
                            robust_scale = float(np.nanmedian(np.abs(np.array(earlier_vals) - baseline)))
                            if earlier_deviation > 2 * deviation and earlier_deviation > robust_scale:
                                per_photo_scores[p_curr.photo_id] += 1.0
                                per_photo_flags[p_curr.photo_id].add(f"RETURN_TO_BASELINE:{metric_name}:earlier_dev={earlier_deviation:.3f}:current_dev={deviation:.3f}")

                # 4. Surgical healing limits
                if metric_name in ["bone_nasion_depth", "bone_zygomatic_width"] and gap_days < BIOLOGICAL_LIMITS["rhinoplasty_healing_days"]:
                    if delta > 3.0:  # 3mm change in nose/zygoma before healing
                        per_photo_scores[p_curr.photo_id] += 0.8
                        per_photo_flags[p_curr.photo_id].add(f"PRE_HEALING_BONE_CHANGE:{metric_name}:delta={delta:.3f}:gap={gap_days}days")

    def _detect_return_to_baseline(
        self,
        metric_name: str,
        values: list[float],
        points: list[ChronologyPoint],
        per_photo_scores: dict[str, float],
        per_photo_flags: dict[str, Set[str]],
    ) -> None:
        if len(values) < 4:
            return
        arr = np.asarray(values, dtype=float)
        finite = np.isfinite(arr)
        if finite.sum() < 4:
            return
        arr = arr.copy()
        arr[~finite] = np.nanmedian(arr[finite])
        window = 3
        smoothed = np.convolve(arr, np.ones(window) / window, mode="same")
        overall = float(np.nanmedian(arr))
        before = np.abs(smoothed[:-2] - overall)
        after = np.abs(smoothed[2:] - overall)
        return_candidates = np.where((before > 0.75) & (after < 0.35))[0]
        for idx in return_candidates:
            photo = points[idx + 1].photo_id
            per_photo_scores[photo] += 0.6
            per_photo_flags[photo].add(f"return_to_baseline:{metric_name}")

    def _detect_change_points(
        self,
        points: list[ChronologyPoint],
        per_metric_series: dict[str, list[float]],
        per_photo_scores: dict[str, float],
        per_photo_flags: dict[str, Set[str]],
    ):
        """Change point detection через ruptures Pelt на мульти-метрическом профиле."""
        try:
            import ruptures as rpt
        except ImportError:
            return

        if len(points) < 5:
            return

        # Строим нормализованный профиль (N, n_metrics)
        metric_names = sorted(per_metric_series.keys())
        profiles = []
        for mname in metric_names:
            vals = per_metric_series[mname]
            if len(vals) < 3:
                continue
            arr = np.array(vals, dtype=float)
            # Z-score нормализация
            mu, sigma = np.median(arr), np.std(arr) + 1e-6
            profiles.append((arr - mu) / sigma)

        if not profiles:
            return

        # Выравниваем длины (берём минимум)
        min_len = min(len(p) for p in profiles)
        profile_matrix = np.column_stack([p[:min_len] for p in profiles])

        # Pelt с косинусным расстоянием
        try:
            algo = rpt.Pelt(model="cos", min_size=2).fit(profile_matrix)
            penalty = np.log(min_len) * profile_matrix.shape[1] * 0.5
            change_points = algo.predict(pen=penalty)
        except Exception:
            return

        # change_points — индексы после которых происходит разрыв
        # Последний индекс = len(profile) — это не change point
        for cp_idx in change_points:
            if cp_idx >= len(points) or cp_idx <= 0:
                continue
            # Флаг на фото после change point
            photo_id = points[cp_idx].photo_id
            per_photo_scores[photo_id] += 0.5
            per_photo_flags[photo_id].add("change_point_detected")

            # Вычисляем before/after средние для ключевых метрик
            for mname in metric_names[:5]:  # топ-5 метрик
                vals = per_metric_series.get(mname, [])
                if len(vals) <= cp_idx:
                    continue
                before = np.mean(vals[:cp_idx])
                after = np.mean(vals[cp_idx:])
                delta = abs(after - before)
                if delta > 0.3:  # значимый сдвиг
                    per_photo_flags[photo_id].add(f"structural_break:{mname}")

    def _detect_cross_metric_return(
        self,
        points: list[ChronologyPoint],
        per_metric_series: dict[str, list[float]],
        per_photo_scores: dict[str, float],
        per_photo_flags: dict[str, Set[str]],
    ):
        """Cross-metric return-to-baseline: если geometry + texture одновременно
        вернулись к baseline — это сильный сигнал (маска снята, двойник вернулся).
        """
        if len(points) < 5:
            return

        # Группируем метрики по типам
        geom_metrics = {k: v for k, v in per_metric_series.items() if k.startswith("bone_") or k.startswith("mesh_")}
        tex_metrics = {k: v for k, v in per_metric_series.items() if k.startswith("texture_") or k.startswith("glcm_") or k.startswith("lbp_")}

        if not geom_metrics or not tex_metrics:
            return

        # Для каждой пары (geometry, texture) проверяем одновременный возврат
        for g_name, g_vals in geom_metrics.items():
            for t_name, t_vals in tex_metrics.items():
                min_len = min(len(g_vals), len(t_vals))
                if min_len < 5:
                    continue

                g_arr = np.array(g_vals[:min_len], dtype=float)
                t_arr = np.array(t_vals[:min_len], dtype=float)

                # Нормализация
                g_mu, g_sigma = np.median(g_arr), np.std(g_arr) + 1e-6
                t_mu, t_sigma = np.median(t_arr), np.std(t_arr) + 1e-6
                g_norm = (g_arr - g_mu) / g_sigma
                t_norm = (t_arr - t_mu) / t_sigma

                # Ищем окно (3 фото), где обе метрики отклоняются > 1.0, а затем возвращаются < 0.3
                for idx in range(2, min_len - 1):
                    # Проверяем: было отклонение на idx-2..idx-1
                    g_was偏离 = np.mean(np.abs(g_norm[idx-2:idx])) > 1.0
                    t_was偏离 = np.mean(np.abs(t_norm[idx-2:idx])) > 1.0
                    # Проверяем: вернулось к baseline на idx
                    g_returned = abs(g_norm[idx]) < 0.3
                    t_returned = abs(t_norm[idx]) < 0.3

                    if g_was偏离 and t_was偏离 and g_returned and t_returned:
                        photo_id = points[idx].photo_id
                        per_photo_scores[photo_id] += 0.7
                        per_photo_flags[photo_id].add(f"CROSS_RETURN:{g_name}+{t_name}")

    def _summary_flags(self, points: list[ChronologyPoint]) -> list[str]:
        if not points:
            return []
        scores = [p.chronology_score for p in points]
        flags = []
        if max(scores) > 1.5:
            flags.append("strong_temporal_break")
        if median(scores) > 0.7:
            flags.append("multiple_temporal_anomalies")
        if any(any(flag.startswith("age_inversion:") or flag.startswith("IMPOSSIBLE_") or flag.startswith("PRE_HEALING_") or flag.startswith("RETURN_TO_BASELINE") for flag in p.flags) for p in points):
            flags.append("biological_impossibility_detected")
        if any("return_to_baseline:" in flag for p in points for flag in p.flags):
            flags.append("mask_swap_indicator")
        if any("change_point_detected" in flag for p in points for flag in p.flags):
            flags.append("structural_break_detected")
        if any("CROSS_RETURN:" in flag for p in points for flag in p.flags):
            flags.append("cross_metric_return_detected")
        return flags