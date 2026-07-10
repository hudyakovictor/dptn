#!/usr/bin/env python3
import json, numpy as np
from collections import defaultdict
from scipy.stats import pearsonr

with open("/home/user/dptn/analysis_raw.json") as f:
    data = json.load(f)

real = data["real"]
sil = data["silicone"]

# collect keys
keys = set()
for r in real+sil:
    keys.update([k for k in r.keys() if k not in ("path","label")])
keys = sorted(keys)

def get_vals(rows, key):
    return np.array([x.get(key, np.nan) for x in rows], dtype=float)

def auc_score(y_true, y_score):
    # simple AUC
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    mask = ~np.isnan(y_score)
    y_true = y_true[mask]; y_score = y_score[mask]
    if len(np.unique(y_true)) < 2: return 0.5
    # sort
    order = np.argsort(y_score)
    y_true = y_true[order]
    n_pos = (y_true==1).sum()
    n_neg = (y_true==0).sum()
    if n_pos==0 or n_neg==0: return 0.5
    # rank sum
    ranks = np.arange(1, len(y_true)+1)
    pos_ranks = ranks[y_true==1].sum()
    auc = (pos_ranks - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
    return auc

results = []
for k in keys:
    rv = get_vals(real, k)
    sv = get_vals(sil, k)
    rv = rv[~np.isnan(rv)]
    sv = sv[~np.isnan(sv)]
    if len(rv)<5 or len(sv)<5: continue
    mean_r = rv.mean(); std_r = rv.std()
    mean_s = sv.mean(); std_s = sv.std()
    # separation Cohen d
    pooled = np.sqrt((std_r**2 + std_s**2)/2 + 1e-9)
    d = (mean_s - mean_r) / pooled
    # AUC
    y_true = [0]*len(rv) + [1]*len(sv)
    y_score = list(rv) + list(sv)
    auc = auc_score(y_true, y_score)
    auc = max(auc, 1-auc)  # direction-agnostic
    # quality correlation within real
    # get sharpness
    sharp = get_vals(real, "sharpness_score")
    # align lengths
    vals_r_full = get_vals(real, k)
    valid = ~np.isnan(vals_r_full) & ~np.isnan(sharp)
    if valid.sum() > 5:
        try:
            corr, _ = pearsonr(vals_r_full[valid], sharp[valid])
            corr = abs(corr)
        except Exception:
            corr = 1.0
    else:
        corr = 1.0
    # quality correlation with noise
    noise = get_vals(real, "noise_level")
    valid2 = ~np.isnan(vals_r_full) & ~np.isnan(noise)
    if valid2.sum() > 5:
        try:
            corr_n, _ = pearsonr(vals_r_full[valid2], noise[valid2])
            corr_n = abs(corr_n)
        except Exception:
            corr_n = 1.0
    else:
        corr_n = 1.0
    quality_corr = max(corr, corr_n)
    # stability CV in real
    cv = std_r / (abs(mean_r)+1e-9)
    # score: high auc, low quality_corr, low cv
    score = auc * (1 - 0.5*quality_corr) / (1 + cv)
    results.append({
        "metric": k,
        "mean_real": mean_r,
        "mean_sil": mean_s,
        "std_real": std_r,
        "cohen_d": d,
        "auc": auc,
        "quality_corr": quality_corr,
        "cv_real": cv,
        "score": score
    })

results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)
print(f"{'Metric':30s}  AUC  d     qCorr  CV   Score   real→sil")
for r in results_sorted[:30]:
    print(f"{r['metric']:30s} {r['auc']:.3f} {r['cohen_d']:+.2f} {r['quality_corr']:.2f} {r['cv_real']:.2f} {r['score']:.3f}  {r['mean_real']:.3f} → {r['mean_sil']:.3f}")

with open("/home/user/dptn/metric_ranking.json","w") as f:
    json.dump(results_sorted, f, indent=2)
