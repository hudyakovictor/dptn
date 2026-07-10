from __future__ import annotations

import json
from typing import List, Dict, Any
import numpy as np


class TimelineVisualizer:
    """Генерирует данные для интерактивного timeline-графика."""
    
    def build(self, chronology_points: List[Dict], 
              alpha_clusters: Dict,
              metric_series: Dict[str, List[float]]) -> Dict:
        """
        Returns JSON-структуру для фронтенда (Plotly/D3).
        """
        # Основная линия: anomaly score по времени
        main_timeline = []
        for point in chronology_points:
            main_timeline.append({
                "date": point["date"].isoformat() if hasattr(point["date"], 'isoformat') else str(point["date"]),
                "anomaly_score": point.get("chronology_score", 0.0),
                "cluster": alpha_clusters.get(point["photo_id"], -1),
                "bucket": point.get("bucket", "unknown"),
                "flags": point.get("flags", []),
            })
        
        # Метрики для overlay-графиков
        metric_traces = {}
        for metric_name, values in metric_series.items():
            if metric_name.startswith("bone_"):
                metric_traces[metric_name] = {
                    "type": "bone",
                    "values": values,
                    "threshold": 2.0,
                }
            elif metric_name.startswith("texture_") or metric_name.startswith("soft_"):
                metric_traces[metric_name] = {
                    "type": "texture",
                    "values": values,
                    "threshold": 0.7,
                }
        
        # Когорты (цветные полосы)
        cohort_bands = []
        for cluster_id, cluster_info in alpha_clusters.get("clusters", {}).items():
            if cluster_info.get("date_range"):
                start, end = cluster_info["date_range"]
                cohort_bands.append({
                    "cluster": cluster_id,
                    "era_label": cluster_info.get("era_label", "unknown"),
                    "start": start.isoformat() if hasattr(start, 'isoformat') else str(start),
                    "end": end.isoformat() if hasattr(end, 'isoformat') else str(end),
                })
        
        return {
            "main_timeline": main_timeline,
            "metric_traces": metric_traces,
            "cohort_bands": cohort_bands,
            "anomalies": alpha_clusters.get("anomalies", []),
        }
    
    def export_html(self, data: Dict, output_path: str):
        """Генерирует самодостаточный HTML с Plotly."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .critical {{ color: #d32f2f; font-weight: bold; }}
        .high {{ color: #f57c00; }}
    </style>
</head>
<body>
    <h1>DEEPUTIN Timeline</h1>
    <div id="timeline"></div>
    <script>
        const data = {json.dumps(data, default=str)};
        
        // Main anomaly trace
        const dates = data.main_timeline.map(d => d.date);
        const scores = data.main_timeline.map(d => d.anomaly_score);
        const clusters = data.main_timeline.map(d => d.cluster);
        
        const trace1 = {{
            x: dates,
            y: scores,
            mode: 'lines+markers',
            name: 'Anomaly Score',
            marker: {{
                color: clusters,
                colorscale: 'Viridis',
                showscale: true,
                colorbar: {{title: 'Identity Cluster'}}
            }}
        }};
        
        const layout = {{
            title: 'Хронологические аномалии и identity-кластеры',
            xaxis: {{title: 'Дата'}},
            yaxis: {{title: 'Anomaly Score'}},
            shapes: data.cohort_bands.map(b => ({{
                type: 'rect',
                x0: b.start, x1: b.end,
                y0: 0, y1: 1,
                yref: 'paper',
                fillcolor: 'rgba(100,100,100,0.1)',
                line: {{width: 0}},
                name: b.era_label
            }}))
        }};
        
        Plotly.newPlot('timeline', [trace1], layout);
    </script>
</body>
</html>
"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)