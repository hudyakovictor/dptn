from __future__ import annotations

import numpy as np
from sklearn.cluster import HDBSCAN
from typing import List, Dict
from datetime import date


class AlphaStabilityTracker:
    """
    Отслеживает стабильность alpha-вектора (199-D identity) во времени.
    Один человек = стабильный alpha (±0.1-0.2 от шума).
    Другой человек = другой кластер alpha.
    """
    
    def __init__(self, noise_threshold: float = 0.15):
        self.noise_threshold = noise_threshold
    
    def build_timeline(self, records: List[Dict]) -> Dict:
        """
        records: [{photo_id, date, alpha: np.ndarray(199,), quality}]
        """
        # 1. Кластеризуем все alpha
        alphas = np.stack([r["alpha"] for r in records])
        dates = [r["date"] for r in records]
        
        # HDBSCAN для кластеризации без заданного числа кластеров
        clusterer = HDBSCAN(min_cluster_size=5, metric="euclidean")
        labels = clusterer.fit_predict(alphas)
        
        # 2. Анализируем хронологию кластеров
        timeline = []
        for i, (record, label) in enumerate(zip(records, labels)):
            timeline.append({
                "photo_id": record["photo_id"],
                "date": record["date"],
                "cluster": int(label),
                "alpha_norm": float(np.linalg.norm(record["alpha"])),
            })
        
        # 3. Ищем аномалии
        anomalies = self._find_temporal_anomalies(timeline)
        
        return {
            "clusters": self._describe_clusters(timeline, labels),
            "timeline": timeline,
            "anomalies": anomalies,
            "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
        }
    
    def _find_temporal_anomalies(self, timeline: List[Dict]) -> List[Dict]:
        """Ищет фото, которые 'прыгают' между кластерами во времени."""
        anomalies = []
        timeline_sorted = sorted(timeline, key=lambda x: x["date"])
        
        for i in range(1, len(timeline_sorted) - 1):
            prev_cluster = timeline_sorted[i-1]["cluster"]
            curr_cluster = timeline_sorted[i]["cluster"]
            next_cluster = timeline_sorted[i+1]["cluster"]
            
            # Фото из кластера X, окружённое кластером Y
            if curr_cluster != prev_cluster and curr_cluster != next_cluster and prev_cluster == next_cluster:
                anomalies.append({
                    "type": "CLUSTER_OUTLIER",
                    "photo_id": timeline_sorted[i]["photo_id"],
                    "date": timeline_sorted[i]["date"],
                    "isolated_cluster": curr_cluster,
                    "surrounding_cluster": prev_cluster,
                    "description": "Фото временно 'прыгает' в другой identity-кластер, потом возвращается. Возможна маска/ретушь.",
                })
            
            # Резкая смена кластера без возврата
            if curr_cluster != prev_cluster and i > 5:
                # Проверяем, не возвращается ли потом
                future_clusters = [t["cluster"] for t in timeline_sorted[i+1:i+10]]
                if prev_cluster not in future_clusters:
                    anomalies.append({
                        "type": "PERMANENT_CLUSTER_SHIFT",
                        "photo_id": timeline_sorted[i]["photo_id"],
                        "date": timeline_sorted[i]["date"],
                        "from_cluster": prev_cluster,
                        "to_cluster": curr_cluster,
                        "description": "Постоянная смена identity-кластера. Возможна смена двойника.",
                    })
        
        return anomalies
    
    def _describe_clusters(self, timeline, labels):
        clusters = {}
        for label in set(labels):
            if label == -1:
                continue
            items = [t for t, l in zip(timeline, labels) if l == label]
            dates = [t["date"] for t in items]
            clusters[int(label)] = {
                "count": len(items),
                "date_range": (min(dates), max(dates)) if dates else None,
                "era_label": self._infer_era(dates),
            }
        return clusters
    
    def _infer_era(self, dates):
        """Инферирует эпоху из дат кластера."""
        if not dates:
            return "unknown"
        years = [d.year for d in dates]
        avg_year = sum(years) / len(years)
        if avg_year < 2005:
            return "early_putin"
        elif avg_year < 2012:
            return "mature_putin"
        elif avg_year < 2021:
            return "udmurt_era"
        else:
            return "vas_era"