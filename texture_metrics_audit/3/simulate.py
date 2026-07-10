#!/usr/bin/env python3
import json, numpy as np, random
from collections import defaultdict
with open("/home/user/dptn/analysis_raw.json") as f:
    data = json.load(f)

real = data["real"]; sil = data["silicone"]
# merge
all_rows = [(r,0) for r in real] + [(s,1) for s in sil]

metrics = ["glcm_homogeneity","glcm_dissimilarity","lbp_r2_std","lbp_r1_std","fft_hf_ratio","grad_mean","tophat_r4_std","tophat_r2_std","local_var_cv15","albedo_a_std","albedo_viability","gray_std"]

def get(row, k): return float(row.get(k, np.nan))

# quality predictors
quality_keys = ["sharpness_score","noise_level","overall_quality","lap_var"]

# fit quality correction on real only
# for each metric, fit linear met = a*sharp + b*noise + c
def fit_correction(metric):
    X=[]; y=[]
    for r in real:
        v = get(r, metric)
        if np.isnan(v): continue
        sharp = get(r, "sharpness_score")
        noise = get(r, "noise_level")
        oq = get(r, "overall_quality")
        if any(np.isnan(x) for x in [sharp,noise,oq]): continue
        X.append([1, sharp, noise, oq])
        y.append(v)
    X = np.array(X); y=np.array(y)
    if len(y)<5: return None
    # least squares
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef

corrections = {m: fit_correction(m) for m in metrics}

def corrected(row, metric):
    coef = corrections.get(metric)
    if coef is None: return get(row, metric)
    sharp = get(row, "sharpness_score"); noise=get(row,"noise_level"); oq=get(row,"overall_quality")
    pred = coef[0] + coef[1]*sharp + coef[2]*noise + coef[3]*oq
    return get(row, metric) - pred  # residual

# simple scoring: z-score vs real distribution, combine
# compute real mean/std for raw and corrected
stats = {}
for m in metrics:
    vals = np.array([get(r,m) for r in real]); vals=vals[~np.isnan(vals)]
    stats[m+"_raw"] = (vals.mean(), vals.std()+1e-9)
    vals_c = np.array([corrected(r,m) for r in real]); vals_c=vals_c[~np.isnan(vals_c)]
    stats[m+"_corr"] = (vals_c.mean(), vals_c.std()+1e-9)

def zscore(v, mean, std): return (v-mean)/std

# direction: silicone - real mean sign
directions = {}
for m in metrics:
    mr = np.nanmean([get(r,m) for r in real])
    ms = np.nanmean([get(s,m) for s in sil])
    directions[m] = np.sign(ms-mr)  # +1 if silicone higher

print("Directions (silicone - real):")
for m,d in directions.items():
    print(m, d)

# simulation: try weighting combinations, evaluate AUC, FPR@95TPR etc, 50+ random trials
def evaluate(weights, use_corrected=True):
    scores=[]; labels=[]
    for row,label in all_rows:
        s=0; wsum=0
        for m, w in weights.items():
            if w==0: continue
            if use_corrected:
                v = corrected(row, m)
                mean,std = stats[m+"_corr"]
            else:
                v = get(row,m)
                mean,std = stats[m+"_raw"]
            if np.isnan(v): continue
            z = zscore(v, mean, std) * directions[m]
            s += w*z
            wsum += abs(w)
        score = s / (wsum+1e-9)
        scores.append(score); labels.append(label)
    # AUC
    scores = np.array(scores); labels = np.array(labels)
    # simple auc
    order = np.argsort(scores)
    labels_sorted = labels[order]
    n_pos = labels_sorted.sum(); n_neg = len(labels_sorted)-n_pos
    if n_pos==0 or n_neg==0: return 0.5,1
    ranks = np.arange(1, len(labels_sorted)+1)
    pos_ranks = ranks[labels_sorted==1].sum()
    auc = (pos_ranks - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
    # FPR at TPR 0.9
    # threshold to get 90% TPR (detect silicone)
    sil_scores = scores[np.array(labels)==1]
    thresh = np.percentile(sil_scores, 10)  # 90% above
    real_scores = scores[np.array(labels)==0]
    fpr = (real_scores >= thresh).mean()
    return auc, fpr

# baseline single metrics
print("\nSingle metric (corrected) AUC / FPR@90TPR:")
for m in metrics:
    auc,fpr = evaluate({m:1}, use_corrected=True)
    print(f"{m:25s} AUC {auc:.3f} FPR {fpr:.3f}")

# random search 200 trials
best = (0, None, True)
for trial in range(200):
    use_corr = random.choice([True, False])
    # random weights sparse
    w = {}
    for m in metrics:
        if random.random() < 0.4:
            w[m] = random.uniform(-1,2)  # allow negative but direction already applied
            if w[m] < 0: w[m]=0
    if not w: continue
    auc,fpr = evaluate(w, use_corrected=use_corr)
    score = auc - 0.5*fpr  # penalize FPR
    if score > best[0]:
        best = (score, w.copy(), use_corr)
        print(f"Trial {trial:03d} {'corr' if use_corr else 'raw '} AUC={auc:.3f} FPR={fpr:.3f} score={score:.3f} weights={w}")

print("\nBEST:", best)
# final evaluation with bootstrap 50 simulations
w_best = best[1]; use_corr_best = best[2]
aucs=[]; fprs=[]
for sim in range(50):
    # bootstrap sample
    indices = np.random.choice(len(all_rows), size=len(all_rows), replace=True)
    sampled = [all_rows[i] for i in indices]
    # temporarily evaluate on sampled
    def eval_sample(weights):
        scores=[]; labels=[]
        for row,label in sampled:
            s=0; wsum=0
            for m, wgt in weights.items():
                if use_corr_best:
                    v = corrected(row, m)
                    mean,std = stats[m+"_corr"]
                else:
                    v = get(row,m)
                    mean,std = stats[m+"_raw"]
                if np.isnan(v): continue
                z = zscore(v, mean, std) * directions[m]
                s += wgt*z
                wsum += abs(wgt)
            scores.append(s/(wsum+1e-9))
            labels.append(label)
        scores=np.array(scores); labels=np.array(labels)
        order=np.argsort(scores); labels_sorted=labels[order]
        n_pos=labels_sorted.sum(); n_neg=len(labels_sorted)-n_pos
        if n_pos==0 or n_neg==0: return 0.5,1
        ranks=np.arange(1,len(labels_sorted)+1)
        pos_ranks=ranks[labels_sorted==1].sum()
        auc=(pos_ranks-n_pos*(n_pos+1)/2)/(n_pos*n_neg)
        sil_scores=scores[np.array(labels)==1]
        thresh=np.percentile(sil_scores,10)
        real_scores=scores[np.array(labels)==0]
        fpr=(real_scores>=thresh).mean()
        return auc,fpr
    auc,fpr = eval_sample(w_best)
    aucs.append(auc); fprs.append(fpr)

print(f"\nBootstrap 50 sims: AUC {np.mean(aucs):.3f} ± {np.std(aucs):.3f}, FPR {np.mean(fprs):.3f} ± {np.std(fprs):.3f}")
print("Weights:", w_best)
print("Use_corrected:", use_corr_best)

# save
with open("/home/user/dptn/best_model.json","w") as f:
    json.dump({"weights":w_best, "use_corrected":use_corr_best, "directions":directions, "stats": {k:[float(v[0]),float(v[1])] for k,v in stats.items()}, "corrections": {k: (v.tolist() if v is not None else None) for k,v in corrections.items()}, "bootstrap_auc_mean": float(np.mean(aucs)), "bootstrap_fpr_mean": float(np.mean(fprs))}, f, indent=2)
