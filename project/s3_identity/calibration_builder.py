from __future__ import annotations

import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Optional
from scipy.optimize import curve_fit
from sklearn.linear_model import TheilSenRegressor
from sklearn.preprocessing import RobustScaler

from .models import PoseNoiseModel, CalibrationBucketHealth, CalibrationPair


class PoseAwareCalibrationBuilder:
    """Строит калибровку с учётом pose gap и quality."""

    def __init__(self, min_pairs: int = 10):
        self.min_pairs = min_pairs

    def build(self, calibration_records: List[Dict]) -> Dict[str, PoseNoiseModel]:
        """
        Для каждого bucket строит модель шума от pose_gap.
        Использует robust statistics, не средние.
        """
        from collections import defaultdict
        bucket_pairs = defaultdict(list)

        grouped = self._group_by_bucket(calibration_records)

        for bucket, records in grouped.items():
            pairs = self._generate_pairs(bucket, records)
            bucket_pairs[bucket].extend(pairs)

        models = {}
        for bucket, pairs in bucket_pairs.items():
            if len(pairs) < self.min_pairs:
                models[bucket] = self._fallback_model(bucket, pairs)
                continue

            models[bucket] = self._fit_robust_model(bucket, pairs)

        return models

    def _group_by_bucket(self, records: List[Dict]) -> Dict[str, List[Dict]]:
        grouped = defaultdict(list)
        for r in records:
            bucket = r.get("bucket", "unknown")
            grouped[bucket].append(r)
        return grouped

    def _generate_pairs(self, bucket: str, records: List[Dict]) -> List[CalibrationPair]:
        pairs = []
        for i, a in enumerate(records):
            for b in records[i+1:]:
                pose_gap = self._pose_gap(a["pose"], b["pose"])
                geom_dist = self._geometry_distance(a, b)
                tex_dist = self._texture_distance(a, b)
                quality = min(a.get("quality", 0.5), b.get("quality", 0.5))
                age_a = a.get("age_years")
                age_b = b.get("age_years")
                age_gap = abs(age_a - age_b) if (age_a is not None and age_b is not None) else 0.0
                scale_a = a.get("face_scale", 1.0)
                scale_b = b.get("face_scale", 1.0)
                scale_diff = abs(scale_a - scale_b)
                # Per-metric distances для TheilSen regression
                per_metric = self._per_metric_distances(a, b)
                pairs.append(CalibrationPair(
                    pose_gap=pose_gap,
                    geom_dist=geom_dist,
                    tex_dist=tex_dist,
                    quality=quality,
                    age_gap=age_gap,
                    scale_diff=scale_diff,
                    per_metric_dists=per_metric,
                ))
        return pairs

    def _per_metric_distances(self, a: Dict, b: Dict) -> Dict[str, float]:
        """Вычисляет попарные расстояния для каждой метрики отдельно."""
        result = {}
        # Geometry metrics
        geom_a = a.get("geometry", {})
        geom_b = b.get("geometry", {})
        for key in set(geom_a.keys()) & set(geom_b.keys()):
            result[f"geom_{key}"] = abs(geom_a[key] - geom_b[key])
        # Texture metrics
        tex_a = a.get("texture", {})
        tex_b = b.get("texture", {})
        for key in set(tex_a.keys()) & set(tex_b.keys()):
            result[f"tex_{key}"] = abs(tex_a[key] - tex_b[key])
        return result

    def _pose_gap(self, pose_a: Dict, pose_b: Dict) -> float:
        dy = abs(pose_a.get("yaw", 0) - pose_b.get("yaw", 0))
        dp = abs(pose_a.get("pitch", 0) - pose_b.get("pitch", 0))
        dr = abs(pose_a.get("roll", 0) - pose_b.get("roll", 0))
        return float(np.sqrt((1.4 * dy) ** 2 + dp ** 2 + (0.6 * dr) ** 2))

    def _geometry_distance(self, a: Dict, b: Dict) -> float:
        geom_a = a.get("geometry", {})
        geom_b = b.get("geometry", {})
        keys = set(geom_a.keys()) & set(geom_b.keys())
        if not keys:
            return 0.0
        dists = [abs(geom_a[k] - geom_b[k]) for k in keys]
        return float(np.median(dists))

    def _texture_distance(self, a: Dict, b: Dict) -> float:
        tex_a = a.get("texture", {})
        tex_b = b.get("texture", {})
        keys = set(tex_a.keys()) & set(tex_b.keys())
        if not keys:
            return 0.0
        dists = [abs(tex_a[k] - tex_b[k]) for k in keys]
        return float(np.median(dists))

    def _fit_robust_model(self, bucket: str, pairs: List[CalibrationPair]) -> PoseNoiseModel:
        """Robust polynomial fit с outlier rejection."""
        gaps = np.array([p.pose_gap for p in pairs])
        dists = np.array([p.geom_dist for p in pairs])

        median = np.median(dists)
        mad = np.median(np.abs(dists - median))
        mask = np.abs(dists - median) < 3.5 * mad

        gaps_clean = gaps[mask]
        dists_clean = dists[mask]

        def model(x, a, b, c):
            return a + b * x + c * x ** 2

        (a, b, c), _ = curve_fit(
            model, gaps_clean, dists_clean,
            p0=[0.1, 0.02, 0.001],
            bounds=([0, 0, 0], [5, 1, 0.1]),
            maxfev=5000
        )

        predictions = model(gaps_clean, a, b, c)
        residuals = dists_clean - predictions

        return PoseNoiseModel(
            bucket=bucket,
            intercept=float(a),
            slope=float(b),
            curvature=float(c),
            p05=float(np.percentile(residuals, 5)),
            p95=float(np.percentile(residuals, 95)),
            mad=float(np.median(np.abs(residuals - np.median(residuals)))),
            sample_count=len(gaps_clean),
        )

    def _fallback_model(self, bucket: str, pairs: List[CalibrationPair]) -> PoseNoiseModel:
        """Если мало пар — используем conservative estimates."""
        if not pairs:
            return PoseNoiseModel(bucket=bucket, intercept=0.5, p95=2.0, mad=1.0, sample_count=0)

        dists = [p.geom_dist for p in pairs]
        return PoseNoiseModel(
            bucket=bucket,
            intercept=float(np.median(dists)),
            p95=float(np.percentile(dists, 95)) if len(dists) > 5 else 2.0,
            mad=float(np.median(np.abs(np.array(dists) - np.median(dists)))) if len(dists) > 1 else 1.0,
            sample_count=len(pairs),
        )

    def fit_thesensen_per_metric(
        self, pairs: List[CalibrationPair], metric_keys: list[str] | None = None
    ) -> dict[str, dict]:
        """TheilSen регрессия для каждой метрики отдельно.
        Возвращает {metric: {slope, intercept, residuals_p05, residuals_p50, residuals_p95, r2, false_anomaly_rate}}.
        false_anomaly_rate = доля пар, где residual > p95 (ожидаемый уровень ~5%).
        """
        if len(pairs) < 10:
            return {}

        # Собираем доступные метрики из per_metric_dists
        if metric_keys is None:
            all_keys: set[str] = set()
            for p in pairs:
                all_keys.update(p.per_metric_dists.keys())
            metric_keys = sorted(all_keys)

        if not metric_keys:
            # Fallback: используем geom_dist и tex_dist
            metric_keys = ["geometry_distance", "texture_distance"]

        results = {}
        gaps = np.array([p.pose_gap for p in pairs]).reshape(-1, 1)

        for metric in metric_keys:
            if metric == "geometry_distance":
                y = np.array([p.geom_dist for p in pairs])
            elif metric == "texture_distance":
                y = np.array([p.tex_dist for p in pairs])
            elif metric.startswith("geom_") or metric.startswith("tex_"):
                y = np.array([p.per_metric_dists.get(metric, 0.0) for p in pairs])
            else:
                continue

            if np.std(y) < 1e-6:
                continue

            try:
                reg = TheilSenRegressor(max_subpopulation=1000, random_state=42)
                reg.fit(gaps, y)
                y_pred = reg.predict(gaps)
                residuals = y - y_pred

                # False anomaly rate: доля точек за пределами p95
                # В идеале ~5% (если калибровка корректна)
                p95_threshold = float(np.percentile(np.abs(residuals), 95))
                false_positives = np.sum(np.abs(residuals) > p95_threshold)
                false_anomaly_rate = float(false_positives / len(residuals)) if len(residuals) > 0 else 0.0

                results[metric] = {
                    "slope": float(reg.coef_[0]),
                    "intercept": float(reg.intercept_),
                    "residuals_p05": float(np.percentile(residuals, 5)),
                    "residuals_p50": float(np.percentile(residuals, 50)),
                    "residuals_p95": float(np.percentile(residuals, 95)),
                    "r2": float(reg.score(gaps, y)),
                    "false_anomaly_rate": false_anomaly_rate,
                    "sample_count": len(pairs),
                }
            except Exception:
                continue

        return results


class CalibrationHealthMonitor:
    """Мониторинг здоровья калибровочных корзин."""

    def check(
        self,
        models: Dict[str, PoseNoiseModel],
        calibration_records: List[Dict],
        per_metric_results: Dict[str, dict] | None = None,
    ) -> List[CalibrationBucketHealth]:
        results = []
        for bucket, model in models.items():
            bucket_recs = [r for r in calibration_records if r.get("bucket") == bucket]
            health = CalibrationBucketHealth(bucket=bucket)
            health.photo_count = len(bucket_recs)

            yaws = [abs(r["pose"].get("yaw", 0)) for r in bucket_recs]
            health.pose_coverage = {
                "min_yaw": min(yaws) if yaws else 0,
                "max_yaw": max(yaws) if yaws else 0,
                "range": max(yaws) - min(yaws) if yaws else 0,
            }

            qualities = [r.get("quality", 0.5) for r in bucket_recs]
            health.quality_coverage = {
                "min": min(qualities) if qualities else 0,
                "max": max(qualities) if qualities else 0,
                "mean": float(np.mean(qualities)) if qualities else 0,
            }

            # Residual quantile check: p95/p05 размах не должен быть слишком большим
            residual_range = model.p95 - model.p05
            health.residual_check = {
                "p05": model.p05,
                "p95": model.p95,
                "range": residual_range,
                "mad": model.mad,
            }

            # False anomaly rate из TheilSen per-metric results
            if per_metric_results:
                # Берём средний false_anomaly_rate по всем метрикам
                rates = [
                    v.get("false_anomaly_rate", 0.0)
                    for v in per_metric_results.values()
                    if isinstance(v, dict)
                ]
                health.false_anomaly_rate = float(np.mean(rates)) if rates else 0.0
            else:
                # Estimate: если p95 > 3*mad, значит калибровка нестабильна
                health.false_anomaly_rate = 0.05 if residual_range > 3.0 * model.mad else 0.0

            # Status determination
            if health.photo_count < 5:
                health.status = "insufficient"
                health.warnings.append(f"Only {health.photo_count} photos")
            elif model.mad > 2.0:
                health.status = "degraded"
                health.warnings.append(f"High MAD: {model.mad:.2f}")
            elif health.false_anomaly_rate > 0.10:
                health.status = "degraded"
                health.warnings.append(f"High false anomaly rate: {health.false_anomaly_rate:.1%}")
            elif residual_range > 5.0:
                health.status = "warning"
                health.warnings.append(f"Wide residual range: {residual_range:.2f}")
            else:
                health.status = "healthy"

            results.append(health)
        return results