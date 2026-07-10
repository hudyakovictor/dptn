from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

TEXTURE_CORE_METRICS = [
    "glcm_dissimilarity_d5_a0",
    "glcm_homogeneity_d5_a0",
    "glcm_dissimilarity_d3_a0",
    "homo_local_var_w15_cv",
    "contrast_weber_mean",
    "homo_local_var_w31_cv",
    "color_b_mean",
    "glcm_homogeneity_d3_a0",
    "glcm_dissimilarity_d3_a135",
    "glcm_dissimilarity_d2_a0",
    "lbp_uniform_r5_std",
    "glcm_dissimilarity_d5_avg",
    "glcm_dissimilarity_d3_avg",
    "morph_tophat_r4_std",
    "glcm_dissimilarity_d5_a135",
    "glcm_dissimilarity_d2_range",
    "grad_sobel_mag_skewness",
    "residual_bio_iqr",
    "morph_tophat_r8_std",
    "glcm_dissimilarity_d5_a45",
]


def load_texture_metric_catalog(path: str | Path | None = None) -> list[dict[str, Any]]:
    if path is None:
        return [{"metric_name": name, "priority": idx + 1} for idx, name in enumerate(TEXTURE_CORE_METRICS)]
    p = Path(path)
    if not p.exists():
        return [{"metric_name": name, "priority": idx + 1} for idx, name in enumerate(TEXTURE_CORE_METRICS)]
    df = pd.read_csv(p)
    out = []
    for idx, row in df.head(20).iterrows():
        out.append(
            {
                "metric_name": str(row.get("name", "")),
                "auc": float(row.get("auc", 0.0) or 0.0),
                "direction": str(row.get("direction", "")),
                "real_mean": float(row.get("real_mean", 0.0) or 0.0),
                "real_std": float(row.get("real_std", 1.0) or 1.0),
                "silicone_mean": float(row.get("silicone_mean", 0.0) or 0.0),
                "silicone_std": float(row.get("silicone_std", 1.0) or 1.0),
                "priority": int(idx) + 1,
            }
        )
    return out
