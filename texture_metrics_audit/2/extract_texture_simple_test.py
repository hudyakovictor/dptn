#!/usr/bin/env python3
"""
Extract texture metrics from simple-test real vs silicone using scikit-image
Goal: find quality-robust metrics and thresholds.

Dataset:
- /home/user/dptn_new/simple-test/test-real/*.png  (196)
- /home/user/dptn_new/simple-test/test-silicone/*.png (100)

Output:
- /home/user/texture_metrics.csv
- /home/user/texture_stats.json
"""

from pathlib import Path
import cv2
import numpy as np
import json
from collections import defaultdict

from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters.rank import entropy as rank_entropy
from skimage.morphology import disk, white_tophat
from skimage.restoration import estimate_sigma
from scipy.ndimage import uniform_filter
from scipy.stats import skew
import warnings
warnings.filterwarnings("ignore")

REAL_DIR = Path("/home/user/dptn_new/simple-test/test-real")
SIL_DIR = Path("/home/user/dptn_new/simple-test/test-silicone")
OUT_CSV = Path("/home/user/texture_metrics.csv")
OUT_JSON = Path("/home/user/texture_stats.json")

def create_skin_mask(h, w):
    """Ellipse central mask 70% width, 80% height, centered"""
    mask = np.zeros((h,w), dtype=np.uint8)
    cx, cy = w//2, h//2
    # axes: 35% width, 40% height
    ax_x, ax_y = int(w*0.35), int(h*0.40)
    cv2.ellipse(mask, (cx,cy), (ax_x, ax_y), 0, 0, 360, 1, -1)
    return mask.astype(bool)

def image_quality_metrics(gray):
    """blur, noise, blockiness, overall"""
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    median = cv2.medianBlur(gray, 3)
    noise = float(np.mean(np.abs(gray.astype(np.float32) - median.astype(np.float32))))
    # jpeg blockiness: difference across 8px grid
    if gray.shape[0] > 16 and gray.shape[1] > 16:
        b = gray[:, 7::8].astype(np.float32)
        i = gray[:, 3::8].astype(np.float32)
        n = min(b.shape[1], i.shape[1])
        blockiness = float(np.mean(np.abs(b[:,:n]-i[:,:n])))/10.0+1.0
    else:
        blockiness = 1.0
    sharp_denom = 400.0 * np.clip(min(gray.shape)/224.0, 0.35, 2.5)
    sharpness = float(np.clip(blur / sharp_denom, 0, 1))
    overall = float(np.clip(sharpness*0.7 + (1-min(noise/35.0,1))*0.3,0,1))
    return {"blur":blur, "noise":noise, "blockiness":blockiness, "sharpness":sharpness, "overall":overall}

def glcm_features(gray_u8, mask, levels=32):
    """Quantile-based GLCM"""
    # mask -> crop to bounding box of mask
    coords = np.argwhere(mask)
    if coords.size==0:
        return {}
    y0,y1 = coords[:,0].min(), coords[:,0].max()+1
    x0,x1 = coords[:,1].min(), coords[:,1].max()+1
    crop = gray_u8[y0:y1, x0:x1]
    if crop.size==0:
        return {}
    # percentile quantization [2,98] like original
    skin_pixels = gray_u8[mask]
    if skin_pixels.size==0:
        return {}
    lo, hi = np.percentile(skin_pixels, [2,98])
    span = max(hi-lo, 1e-6)
    norm = np.clip((crop.astype(float)-lo)/span,0,1)
    quant = (norm* (levels-1)).astype(np.uint8)
    # need same for whole crop, but mask for GLCM? skimage greycomatrix doesn't support mask, so we fill non-skin with mean
    # for simplicity compute GLCM on whole quant crop (includes some background) - but we cropped tightly so ok
    try:
        glcm = graycomatrix(quant, distances=[1,3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=levels, symmetric=True, normed=True)
        # aggregate
        res={}
        for d_idx, d in enumerate([1,3]):
            # mean over angles
            diss = [float(graycoprops(glcm, "dissimilarity")[d_idx, a]) for a in range(4)]
            homo = [float(graycoprops(glcm, "homogeneity")[d_idx, a]) for a in range(4)]
            contr = [float(graycoprops(glcm, "contrast")[d_idx, a]) for a in range(4)]
            energ = [float(graycoprops(glcm, "energy")[d_idx, a]) for a in range(4)]
            corr = [float(graycoprops(glcm, "correlation")[d_idx, a]) for a in range(4)]
            res[f"glcm_diss_d{d}_mean"] = float(np.mean(diss))
            res[f"glcm_diss_d{d}_std"] = float(np.std(diss))
            res[f"glcm_homo_d{d}_mean"] = float(np.mean(homo))
            res[f"glcm_contr_d{d}_mean"] = float(np.mean(contr))
            res[f"glcm_energy_d{d}_mean"] = float(np.mean(energ))
            res[f"glcm_corr_d{d}_mean"] = float(np.mean(corr))
            # anisotropy: std across angles (regular texture has high anisotropy)
            res[f"glcm_diss_d{d}_aniso"] = float(np.std(diss))
        return res
    except Exception as e:
        return {}

def lbp_features(gray_u8, mask):
    res={}
    try:
        lbp1 = local_binary_pattern(gray_u8, P=8, R=1, method="uniform")
        lbp2 = local_binary_pattern(gray_u8, P=8, R=2, method="uniform")
        lbp1_skin = lbp1[mask]
        lbp2_skin = lbp2[mask]
        if lbp1_skin.size>0:
            # uniform patterns: 0..8 uniform, 9 non-uniform
            # std
            res["lbp_r1_std"] = float(np.std(lbp1_skin))
            res["lbp_r1_mean"] = float(np.mean(lbp1_skin))
            # histogram
            hist, _ = np.histogram(lbp1_skin, bins=10, range=(0,10), density=True)
            # entropy of lbp histogram
            hist = hist[hist>0]
            res["lbp_r1_hist_entropy"] = float(-np.sum(hist*np.log2(hist))) if hist.size>0 else 0.0
            # non-uniform ratio: code 9 is non-uniform in uniform method (P+1)
            # Actually uniform method: 0..P are uniform, P+1=9 is non-uniform
            non_uniform = np.sum(lbp1_skin==9) / lbp1_skin.size
            res["lbp_r1_nonuniform_ratio"] = float(non_uniform)
        if lbp2_skin.size>0:
            res["lbp_r2_std"] = float(np.std(lbp2_skin))
            res["lbp_r2_mean"] = float(np.mean(lbp2_skin))
            non_uniform2 = np.sum(lbp2_skin==9) / lbp2_skin.size
            res["lbp_r2_nonuniform_ratio"] = float(non_uniform2)
        # ror method for complexity
        lbp_ror = local_binary_pattern(gray_u8, P=8, R=1, method="ror")
        lbp_ror_skin = lbp_ror[mask]
        if lbp_ror_skin.size>0:
            res["lbp_ror_r1_std"] = float(np.std(lbp_ror_skin))
    except Exception as e:
        pass
    return res

def fft_features(gray_u8, mask):
    res={}
    try:
        coords = np.argwhere(mask)
        if coords.size<64*64:
            return res
        y0,y1 = coords[:,0].min(), coords[:,0].max()+1
        x0,x1 = coords[:,1].min(), coords[:,1].max()+1
        crop = gray_u8[y0:y1, x0:x1].astype(np.float32)
        # patch-based: central 128x128 if possible
        h,w = crop.shape
        # take central patch 128x128
        ch, cw = h//2, w//2
        ph = min(128, h)
        pw = min(128, w)
        patch = crop[ch-ph//2:ch+ph//2, cw-pw//2:cw+pw//2]
        if patch.size==0:
            return res
        # window
        wy = np.hanning(patch.shape[0])
        wx = np.hanning(patch.shape[1])
        window = np.outer(wy, wx)
        patch_w = (patch - patch.mean()) * window
        f = np.fft.fft2(patch_w)
        fshift = np.fft.fftshift(f)
        mag = np.abs(fshift)
        power = mag**2
        h_, w_ = mag.shape
        cy, cx = h_//2, w_//2
        yy, xx = np.ogrid[:h_, :w_]
        radius = np.sqrt((yy-cy)**2 + (xx-cx)**2)
        # masks
        low = power[radius<=4].sum()
        high = power[radius>8].sum()
        total = power.sum()+1e-6
        res["fft_highfreq_ratio"] = float(high/(low+1e-6)) if low>0 else 0.0
        res["fft_high_low_ratio"] = float(high/total)
        res["fft_peak_ratio"] = float(mag.max()/ (total+1e-6))
        # anisotropy: angular profile
        theta = np.arctan2(yy-cy, xx-cx) + np.pi # 0..2pi
        n_bins=12
        ang_profile=[]
        for i in range(n_bins):
            a0 = i*2*np.pi/n_bins
            a1 = (i+1)*2*np.pi/n_bins
            mask_ang = (theta>=a0) & (theta<a1) & (radius>=5) & (radius<=20)
            if mask_ang.any():
                ang_profile.append(power[mask_ang].mean())
            else:
                ang_profile.append(0)
        ang_profile = np.array(ang_profile)
        if ang_profile.sum()>0:
            ang_profile = ang_profile/ang_profile.sum()
            # entropy
            ang_profile = ang_profile[ang_profile>0]
            ent = -np.sum(ang_profile*np.log(ang_profile+1e-9))
            max_ent = np.log(n_bins)
            res["fft_angular_entropy"] = float(ent/max_ent) # 1=isotropic, 0=anisotropic
            res["fft_aniso"] = float(1.0 - ent/max_ent) # anisotropy
        # spectral slope beta
        # radial bins
        max_r = min(h_, w_)//2
        n_rad=15
        rad_power=[]
        rad_r=[]
        for i in range(1, n_rad):
            r0 = i*max_r/n_rad
            r1 = (i+1)*max_r/n_rad
            m = (radius>=r0) & (radius<r1)
            if m.any():
                rad_power.append(power[m].mean())
                rad_r.append((r0+r1)/2)
        rad_power = np.array(rad_power)
        rad_r = np.array(rad_r)
        valid = (rad_power>0) & (rad_r>3)
        if valid.sum()>=4:
            log_r = np.log(rad_r[valid])
            log_p = np.log(rad_power[valid]+1e-9)
            slope, _ = np.polyfit(log_r, log_p, 1)
            res["spectral_slope_beta"] = float(-slope)
        else:
            res["spectral_slope_beta"] = 2.5
    except Exception as e:
        # print(e)
        pass
    return res

def local_var_features(gray_u8, mask):
    res={}
    try:
        gray_f = gray_u8.astype(np.float64)
        for wname, w in [("w15",15), ("w31",31)]:
            local_m = uniform_filter(gray_f, size=w)
            local_m_sq = uniform_filter(gray_f**2, size=w)
            local_var = np.maximum(local_m_sq - local_m**2, 0)
            local_std = np.sqrt(local_var)
            valid_mask = mask & (local_m>1)
            if valid_mask.any():
                cv = local_std[valid_mask] / local_m[valid_mask]
                cv = np.clip(cv, 0, 5)
                res[f"homo_cv_{wname}_mean"] = float(np.mean(cv))
                res[f"homo_cv_{wname}_std"] = float(np.std(cv))
                # local std mean
                res[f"homo_std_{wname}_mean"] = float(np.mean(local_std[valid_mask]))
    except Exception:
        pass
    return res

def pore_features(gray_u8, mask):
    res={}
    try:
        for rname, r in [("r2",2), ("r4",4)]:
            tophat = white_tophat(gray_u8, disk(r))
            vals = tophat[mask]
            if vals.size>0:
                res[f"pore_tophat_{rname}_mean"] = float(np.mean(vals))
                res[f"pore_tophat_{rname}_std"] = float(np.std(vals))
                # density: count > mean+std
                thr = np.mean(vals) + np.std(vals)
                dens = np.sum(vals>thr) / max(mask.sum(),1)
                res[f"pore_density_{rname}"] = float(dens)
    except Exception:
        pass
    return res

def edge_features(gray_u8, mask):
    res={}
    try:
        edges = cv2.Canny(gray_u8, 40, 120)
        if mask.any():
            res["edge_density"] = float(edges[mask].mean()/255.0)
        # sobel skew
        sx = cv2.Sobel(gray_u8, cv2.CV_64F, 1,0, ksize=3)
        sy = cv2.Sobel(gray_u8, cv2.CV_64F, 0,1, ksize=3)
        mag = np.sqrt(sx**2+sy**2)
        mag_skin = mag[mask]
        if mag_skin.size>0:
            res["sobel_mean"] = float(np.mean(mag_skin))
            res["sobel_std"] = float(np.std(mag_skin))
            res["sobel_skew"] = float(skew(mag_skin.ravel())) if mag_skin.size>10 else 0.0
    except Exception:
        pass
    return res

def entropy_features(gray_u8, mask):
    res={}
    try:
        # histogram entropy 32 bins
        pixels = gray_u8[mask]
        if pixels.size>0:
            hist, _ = np.histogram(pixels, bins=32, range=(0,255), density=False)
            prob = hist / hist.sum()
            prob = prob[prob>0]
            ent = -np.sum(prob*np.log2(prob)) if prob.size>0 else 0.0
            res["hist_entropy"] = float(ent)
        # rank entropy disk 5
        # need uint8 image
        try:
            ent_img = rank_entropy(gray_u8, disk(5))
            if mask.any():
                res["rank_entropy_median"] = float(np.median(ent_img[mask]))
                res["rank_entropy_std"] = float(np.std(ent_img[mask]))
        except Exception:
            pass
    except Exception:
        pass
    return res

def process_image(path, label):
    img = cv2.imread(str(path))
    if img is None:
        return None
    # resize if too large? Keep as is
    h,w = img.shape[:2]
    mask = create_skin_mask(h,w)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # quality
    q = image_quality_metrics(gray)
    # metrics
    feats={}
    feats.update(q)
    feats.update(glcm_features(gray, mask))
    feats.update(lbp_features(gray, mask))
    feats.update(fft_features(gray, mask))
    feats.update(local_var_features(gray, mask))
    feats.update(pore_features(gray, mask))
    feats.update(edge_features(gray, mask))
    feats.update(entropy_features(gray, mask))
    feats["label"] = label
    feats["path"] = str(path.name)
    # estimate sigma via skimage
    try:
        sigma = estimate_sigma(gray, channel_axis=None)
        feats["sigma_est"] = float(sigma)
    except:
        feats["sigma_est"] = 0.0
    return feats

def main():
    all_feats=[]
    for p in REAL_DIR.glob("*.png"):
        f = process_image(p, "real")
        if f:
            all_feats.append(f)
    for p in SIL_DIR.glob("*.png"):
        f = process_image(p, "silicone")
        if f:
            all_feats.append(f)
    # save csv
    import csv
    if not all_feats:
        print("no feats")
        return
    keys = sorted(set(k for d in all_feats for k in d.keys() if k not in ["path","label"]))
    # ensure label first
    fieldnames = ["path","label"]+keys
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for d in all_feats:
            row = {k: d.get(k,"") for k in fieldnames}
            writer.writerow(row)
    print(f"Saved {len(all_feats)} to {OUT_CSV}")

    # compute stats per class
    import pandas as pd
    df = pd.DataFrame(all_feats)
    stats={}
    for col in keys:
        if df[col].dtype==object:
            continue
        try:
            vals_real = df[df["label"]=="real"][col].astype(float).dropna()
            vals_sil = df[df["label"]=="silicone"][col].astype(float).dropna()
            if len(vals_real)==0 or len(vals_sil)==0:
                continue
            mean_real = float(vals_real.mean())
            std_real = float(vals_real.std())
            median_real = float(vals_real.median())
            mean_sil = float(vals_sil.mean())
            std_sil = float(vals_sil.std())
            median_sil = float(vals_sil.median())
            # cohen's d
            pooled_std = np.sqrt((std_real**2 + std_sil**2)/2)
            cohen_d = (mean_real - mean_sil) / (pooled_std+1e-9)
            # separation: difference of medians over sum mad
            mad_real = float(np.median(np.abs(vals_real - median_real)) or 1e-6)
            mad_sil = float(np.median(np.abs(vals_sil - median_sil)) or 1e-6)
            sep = abs(median_real - median_sil) / (mad_real + mad_sil + 1e-9)
            # correlation with quality (overall) for real class only - to check quality bias
            # if metric correlates strongly with blur, it's not robust
            try:
                corr_quality = float(np.corrcoef(vals_real, df[df["label"]=="real"]["overall"].astype(float))[0,1])
            except:
                corr_quality = 0.0
            stats[col] = {
                "mean_real": mean_real,
                "std_real": std_real,
                "median_real": median_real,
                "mad_real": mad_real,
                "mean_sil": mean_sil,
                "std_sil": std_sil,
                "median_sil": median_sil,
                "mad_sil": mad_sil,
                "cohen_d": float(cohen_d),
                "sep_mad": float(sep),
                "corr_quality_real": corr_quality,
                "abs_cohen": float(abs(cohen_d)),
            }
        except Exception as e:
            # print(col, e)
            continue
    # sort by abs cohen
    sorted_stats = sorted(stats.items(), key=lambda x: x[1]["abs_cohen"], reverse=True)
    # save json
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({k:v for k,v in sorted_stats}, f, indent=2, ensure_ascii=False)
    print(f"Saved stats to {OUT_JSON}")
    # print top 20
    print("\nTop 20 discriminative (by |Cohen d|) + quality correlation (low is good):\n")
    for k,v in sorted_stats[:20]:
        print(f"{k:30s} d={v['cohen_d']:+6.2f} sep={v['sep_mad']:5.2f} corr_q={v['corr_quality_real']:+5.2f}  real_median={v['median_real']:.3f} sil_median={v['median_sil']:.3f}")

if __name__=="__main__":
    main()
