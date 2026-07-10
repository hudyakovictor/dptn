from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

TEXTURE_CORE_METRICS = [
    # Tier 1 — Quality-Robust Core (12 метрик, работают на 1999 low-Q)
    "tv_residual_sparsity",
    "lacunarity",
    "autocorr_decay_len",
    "wld_joint_entropy",
    "fft_high_low_ratio",
    "spectral_slope_beta",
    "glcm_diss_d3_aniso",
    "pore_density_r2_mpx",
    "hemoglobin_od_std",
    "bimodality_ashman_D",
    "glszm_small_area_emphasis",
    "edge_tortuosity_mean",
    # Tier 2 — High-Quality Extended (8 метрик, overall>=0.5 или sharpness>=200)
    "glrlm_sre",
    "ngtdm_coarseness",
    "dwt_haar_HH_LL_ratio",
    "lbp_r1_hist_entropy",
    "shannon_entropy_q32",
    "gabor_f08_anisotropy",
    "pore_eccentricity_mean",
    "specular_elongation",
]

TEXTURE_CORE_METRICS_V2 = TEXTURE_CORE_METRICS  # alias для совместимости

# Physical auxiliary metrics (Tier 3) — не в CORE, используются в s5_verdict как boost
PHYSICAL_AUX_METRICS = [
    "seam_score",
    "specular_sharpness",
    "sss_index",
    "melanin_hemo_slope",
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
