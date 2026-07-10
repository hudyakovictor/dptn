from __future__ import annotations

from typing import Dict, List
import numpy as np

from .models import CalibrationBucketHealth, PoseNoiseModel


class CalibrationHealthMonitor:
    """Мониторинг здоровья корзин калибровки."""

    def check(self, models: Dict[str, PoseNoiseModel], 
              calibration_records: List[Dict]) -> List[CalibrationBucketHealth]:
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

            if health.photo_count < 5:
                health.status = "insufficient"
                health.warnings.append(f"Only {health.photo_count} photos")
            elif model.mad > 2.0:
                health.status = "degraded"
                health.warnings.append(f"High MAD: {model.mad:.2f}")
            else:
                health.status = "healthy"
            
            results.append(health)
        return results

    def summary(self, health_results: List[CalibrationBucketHealth]) -> Dict[str, int]:
        """Return summary counts by status."""
        summary = {"healthy": 0, "degraded": 0, "insufficient": 0, "total": len(health_results)}
        for h in health_results:
            if h.status in summary:
                summary[h.status] += 1
        return summary