from __future__ import annotations

from datetime import date
import numpy as np
from typing import List, Dict


class BaselineReturnDetector:
    """Детектирует 'возврат к baseline' — ключевой маркер маски."""
    
    def detect(self, metric_series: List[float], dates: List[date],
               threshold_ratio: float = 0.6) -> List[Dict]:
        """
        Если метрика отклонялась, а потом вернулась к значениям 
        5+ летней давности — это невозможно при естественном старении.
        """
        if len(metric_series) < 10:
            return []
        
        flags = []
        # Early baseline = среднее первых 20% фото
        early_end = max(3, len(metric_series) // 5)
        baseline = np.median(metric_series[:early_end])
        baseline_mad = np.median(np.abs(np.array(metric_series[:early_end]) - baseline))
        
        for i in range(early_end + 5, len(metric_series)):
            # Ищем период отклонения
            window = metric_series[max(0, i-5):i]
            window_median = np.median(window)
            
            # Отклонились от baseline?
            if abs(window_median - baseline) > 2 * baseline_mad:
                # Ищем возврат
                for j in range(i + 1, min(i + 20, len(metric_series))):
                    if abs(metric_series[j] - baseline) < baseline_mad:
                        gap_days = (dates[j] - dates[i]).days
                        flags.append({
                            "type": "RETURN_TO_BASELINE",
                            "deviation_start": dates[i],
                            "return_date": dates[j],
                            "gap_days": gap_days,
                            "metric_delta": float(window_median - baseline),
                            "description": f"Метрика отклонилась на {window_median-baseline:.2f} мм, но вернулась к baseline через {gap_days} дней. При естественном старении такого 'отката' не бывает.",
                        })
                        break
        
        return flags