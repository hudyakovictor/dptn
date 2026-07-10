#!/usr/bin/env python3
"""
Find top 20 NEW metrics not in current TEXTURE_CORE_METRICS that best separate real vs silicone considering quality.

Current CORE 20:
glcm_dissimilarity_d5_a0, glcm_homogeneity_d5_a0, glcm_dissimilarity_d3_a0,
homo_local_var_w15_cv, contrast_weber_mean, homo_local_var_w31_cv, color_b_mean,
glcm_homogeneity_d3_a0, glcm_dissimilarity_d3_a135, glcm_dissimilarity_d2_a0,
lbp_uniform_r5_std, glcm_dissimilarity_d5_avg, glcm_dissimilarity_d3_avg,
morph_tophat_r4_std, glcm_dissimilarity_d5_a135, glcm_dissimilarity_d2_range,
grad_sobel_mag_skewness, residual_bio_iqr, morph_tophat_r8_std, glcm_dissimilarity_d5_a45

We need to find NEW metrics from scikit-image that are:
- discriminative (|Cohen d| high)
- quality-robust (|corr_quality| <0.45)
- stable (CV low from previous stability test)

Extended metrics to try:
- GLCM: energy, correlation, contrast for d=1,2,3,5; anisotropy for all
- LBP: r1,r2,r3 nonuniform, std, hist entropy, ror std, var
- FFT: high_low, highfreq, peak, angular entropy, aniso, beta, low_power, mid_power
- Local var: w7,w15,w31 mean/std/cv
- Pore: tophat r2,r4,r6,r8 mean/std/density
- Entropy: hist, rank median/std
- Gabor: using skimage.filters.gabor (real mean)
- Frangi, Hessian (maybe)
- Color: blue mean, saturation, etc but focus on texture
"""

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import gabor
from skimage.morphology import disk, white_tophat
from skimage.filters.rank import entropy as rank_entropy
from scipy.ndimage import uniform_filter
from scipy.stats import skew
import json

REAL_DIR = Path("/home/user/dptn_new/simple-test/test-real")
SIL_DIR = Path("/home/user/dptn_new/simple-test/test-silicone")

CURRENT_CORE = [
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

def mask_ellipse(h,w):
    mask=np.zeros((h,w), dtype=np.uint8)
    cv2.ellipse(mask, (w//2,h//2), (int(w*0.35), int(h*0.40)), 0,0,360,1,-1)
    return mask.astype(bool)

def extract_extended(gray, mask):
    res={}
    gray_u8 = np.clip(gray,0,255).astype(np.uint8)
    # quality
    blur = float(cv2.Laplacian(gray_u8, cv2.CV_64F).var())
    overall = float(np.clip(blur/400*0.7+0.3,0,1))

    # --- GLCM extended ---
    try:
        skin_pixels=gray_u8[mask]
        if skin_pixels.size>0:
            lo,hi=np.percentile(skin_pixels,[2,98])
            span=max(hi-lo,1e-6)
            norm=np.clip((gray_u8.astype(float)-lo)/span,0,1)
            quant=(norm*31).astype(np.uint8)
            coords=np.argwhere(mask)
            y0,y1=coords[:,0].min(), coords[:,0].max()+1
            x0,x1=coords[:,1].min(), coords[:,1].max()+1
            qcrop=quant[y0:y1,x0:x1]
            if qcrop.size>100:
                glcm=graycomatrix(qcrop, distances=[1,2,3,5], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=32, symmetric=True, normed=True)
                for d_idx, d in enumerate([1,2,3,5]):
                    for prop in ["contrast","dissimilarity","homogeneity","energy","correlation"]:
                        try:
                            vals=[float(graycoprops(glcm, prop)[d_idx,a]) for a in range(4)]
                            res[f"glcm_{prop}_d{d}_mean"]=float(np.mean(vals))
                            res[f"glcm_{prop}_d{d}_std"]=float(np.std(vals))
                            res[f"glcm_{prop}_d{d}_aniso"]=float(np.std(vals))
                            # also max-min range
                            res[f"glcm_{prop}_d{d}_range"]=float(np.max(vals)-np.min(vals))
                        except:
                            pass
    except Exception as e:
        pass

    # --- LBP extended ---
    try:
        for R,P in [(1,8),(2,8),(3,8),(2,16)]:
            lbp=local_binary_pattern(gray_u8, P=P, R=R, method="uniform")
            skin=lbp[mask]
            if skin.size>0:
                res[f"lbp_R{R}_P{P}_std"]=float(np.std(skin))
                res[f"lbp_R{R}_P{P}_mean"]=float(np.mean(skin))
                # non-uniform ratio (code P+1)
                nun=np.sum(skin==P+1)/skin.size
                res[f"lbp_R{R}_P{P}_nonuniform"]=float(nun)
                # hist entropy
                hist,_=np.histogram(skin, bins=P+2, range=(0,P+2), density=True)
                hist=hist[hist>0]
                ent=float(-np.sum(hist*np.log2(hist))) if hist.size>0 else 0
                res[f"lbp_R{R}_P{P}_hist_ent"]=float(ent)
        # ror
        lbp_ror=local_binary_pattern(gray_u8, P=8, R=1, method="ror")
        skin=lbp_ror[mask]
        if skin.size>0:
            res["lbp_ror_R1_std"]=float(np.std(skin))
            res["lbp_ror_R1_mean"]=float(np.mean(skin))
        # var
        lbp_var=local_binary_pattern(gray_u8, P=8, R=1, method="var")
        skin=lbp_var[mask]
        if skin.size>0:
            res["lbp_var_R1_mean"]=float(np.mean(skin))
            res["lbp_var_R1_std"]=float(np.std(skin))
    except:
        pass

    # --- FFT extended ---
    try:
        coords=np.argwhere(mask)
        y0,y1=coords[:,0].min(), coords[:,0].max()+1
        x0,x1=coords[:,1].min(), coords[:,1].max()+1
        crop=gray_u8[y0:y1,x0:x1].astype(float)
        ch,cw=crop.shape[0]//2,crop.shape[1]//2
        ph,pw=min(128,crop.shape[0]), min(128,crop.shape[1])
        if ph>=16 and pw>=16:
            patch=crop[ch-ph//2:ch+ph//2, cw-pw//2:cw+pw//2]
            wy,wx=np.hanning(patch.shape[0]), np.hanning(patch.shape[1])
            patch_w=(patch-patch.mean())*np.outer(wy,wx)
            f=np.fft.fft2(patch_w)
            fshift=np.fft.fftshift(f)
            power=np.abs(fshift)**2
            h_,w_=power.shape
            cy,cx=h_//2,w_//2
            yy,xx=np.ogrid[:h_,:w_]
            radius=np.sqrt((yy-cy)**2+(xx-cx)**2)
            # various bands
            for r_low,r_high,name in [(0,4,"low"),(4,8,"mid"),(8,20,"high"),(20,40,"vhigh")]:
                band=power[(radius>=r_low)&(radius<r_high)].sum()
                res[f"fft_power_{name}"]=float(band)
            low=power[radius<=4].sum()
            mid=power[(radius>4)&(radius<=8)].sum()
            high=power[radius>8].sum()
            total=power.sum()+1e-6
            res["fft_high_low_ratio"]=float(high/(low+1e-6)) if low>0 else 0
            res["fft_high_mid_ratio"]=float(high/(mid+1e-6)) if mid>0 else 0
            res["fft_high_total_ratio"]=float(high/total)
            res["fft_mid_low_ratio"]=float(mid/(low+1e-6)) if low>0 else 0
            # peak
            res["fft_peak_ratio"]=float(power.max()/total)
            # angular
            theta=np.arctan2(yy-cy, xx-cx)+np.pi
            n_bins=12
            ang=[]
            for i in range(n_bins):
                a0=i*2*np.pi/n_bins; a1=(i+1)*2*np.pi/n_bins
                ma=(theta>=a0)&(theta<a1)&(radius>=5)&(radius<=20)
                if ma.any():
                    ang.append(power[ma].mean())
                else:
                    ang.append(0)
            ang=np.array(ang)
            if ang.sum()>0:
                ang=ang/ang.sum()
                ang_nz=ang[ang>0]
                ent=-np.sum(ang_nz*np.log(ang_nz+1e-9))
                max_ent=np.log(n_bins)
                res["fft_angular_entropy"]=float(ent/max_ent)
                res["fft_aniso"]=float(1-ent/max_ent)
                res["fft_angular_std"]=float(np.std(ang))
            # spectral slope beta
            max_r=min(h_,w_)//2
            rad_power=[]; rad_r=[]
            for i in range(1,15):
                r0=i*max_r/15; r1=(i+1)*max_r/15
                m=(radius>=r0)&(radius<r1)
                if m.any():
                    rad_power.append(power[m].mean())
                    rad_r.append((r0+r1)/2)
            rad_power=np.array(rad_power); rad_r=np.array(rad_r)
            valid=(rad_power>0)&(rad_r>3)
            if valid.sum()>=4:
                slope,_=np.polyfit(np.log(rad_r[valid]), np.log(rad_power[valid]+1e-9),1)
                res["spectral_slope_beta"]=float(-slope)
            else:
                res["spectral_slope_beta"]=2.5
    except Exception as e:
        pass

    # --- Local var ---
    try:
        gray_f=gray_u8.astype(float)
        for wname,w in [("w7",7),("w15",15),("w31",31)]:
            lm=uniform_filter(gray_f, size=w)
            lm_sq=uniform_filter(gray_f**2, size=w)
            lvar=np.maximum(lm_sq-lm**2,0)
            lstd=np.sqrt(lvar)
            vm=mask & (lm>1)
            if vm.any():
                cv=lstd[vm]/lm[vm]
                cv=np.clip(cv,0,5)
                res[f"homo_cv_{wname}_mean"]=float(np.mean(cv))
                res[f"homo_cv_{wname}_std"]=float(np.std(cv))
                res[f"homo_std_{wname}_mean"]=float(np.mean(lstd[vm]))
                res[f"homo_std_{wname}_std"]=float(np.std(lstd[vm]))
    except:
        pass

    # --- Pore tophat ---
    try:
        for rname,r in [("r2",2),("r4",4),("r6",6),("r8",8)]:
            th=white_tophat(gray_u8, disk(r))
            vals=th[mask]
            if vals.size>0:
                res[f"pore_tophat_{rname}_mean"]=float(np.mean(vals))
                res[f"pore_tophat_{rname}_std"]=float(np.std(vals))
                thr=np.mean(vals)+np.std(vals)
                dens=np.sum(vals>thr)/max(mask.sum(),1)
                res[f"pore_tophat_{rname}_dens"]=float(dens)
                # also count > mean+2std (strong pores)
                thr2=np.mean(vals)+2*np.std(vals)
                dens2=np.sum(vals>thr2)/max(mask.sum(),1)
                res[f"pore_tophat_{rname}_dens2"]=float(dens2)
    except:
        pass

    # --- Entropy ---
    try:
        pixels=gray_u8[mask]
        if pixels.size>0:
            hist,_=np.histogram(pixels, bins=32, range=(0,255), density=False)
            prob=hist/hist.sum()
            prob=prob[prob>0]
            ent=-np.sum(prob*np.log2(prob)) if prob.size>0 else 0
            res["hist_entropy"]=float(ent)
            res["gray_mean"]=float(np.mean(pixels))
            res["gray_std"]=float(np.std(pixels))
        # rank entropy
        try:
            ent_img=rank_entropy(gray_u8, disk(5))
            if mask.any():
                res["rank_entropy_median"]=float(np.median(ent_img[mask]))
                res["rank_entropy_std"]=float(np.std(ent_img[mask]))
                res["rank_entropy_mean"]=float(np.mean(ent_img[mask]))
        except:
            pass
    except:
        pass

    # --- Gabor ---
    try:
        # frequency 0.1, 0.2
        for freq in [0.1, 0.2]:
            for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
                real, imag = gabor(gray_u8, frequency=freq, theta=theta)
                # mean of real part in mask
                vals_real=real[mask]
                if vals_real.size>0:
                    res[f"gabor_f{freq}_t{int(np.degrees(theta))}_mean"]=float(np.mean(vals_real))
                    res[f"gabor_f{freq}_t{int(np.degrees(theta))}_std"]=float(np.std(vals_real))
        # anisotropy of gabor responses
        # compute std across thetas for same freq
        for freq in [0.1, 0.2]:
            means=[]
            for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
                key=f"gabor_f{freq}_t{int(np.degrees(theta))}_mean"
                if key in res:
                    means.append(res[key])
            if means:
                res[f"gabor_f{freq}_aniso"]=float(np.std(means))
    except:
        pass

    # --- Edge ---
    try:
        edges=cv2.Canny(gray_u8,40,120)
        if mask.any():
            res["edge_density"]=float(edges[mask].mean()/255.0)
        sx=cv2.Sobel(gray_u8, cv2.CV_64F, 1,0, ksize=3)
        sy=cv2.Sobel(gray_u8, cv2.CV_64F, 0,1, ksize=3)
        mag=np.sqrt(sx**2+sy**2)
        mag_skin=mag[mask]
        if mag_skin.size>0:
            res["sobel_mean"]=float(np.mean(mag_skin))
            res["sobel_std"]=float(np.std(mag_skin))
            res["sobel_skew"]=float(skew(mag_skin.ravel())) if mag_skin.size>10 else 0.0
    except:
        pass

    res["overall_quality"]=overall
    res["blur"]=blur
    return res

def process_all():
    all_recs=[]
    for label, dir_path in [("real", REAL_DIR), ("silicone", SIL_DIR)]:
        for p in dir_path.glob("*.png"):
            img=cv2.imread(str(p))
            if img is None:
                continue
            h,w=img.shape[:2]
            mask=mask_ellipse(h,w)
            gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            metrics=extract_extended(gray, mask)
            metrics["label"]=label
            metrics["path"]=p.name
            all_recs.append(metrics)
    # save
    import csv
    keys=sorted(set(k for d in all_recs for k in d.keys() if k not in ["label","path"]))
    fieldnames=["path","label"]+keys
    with open("/home/user/extended_metrics.csv","w", newline="", encoding="utf-8") as f:
        writer=csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for d in all_recs:
            row={k:d.get(k,"") for k in fieldnames}
            writer.writerow(row)
    print(f"Saved {len(all_recs)} records with {len(keys)} metrics")

    # compute stats
    df=pd.DataFrame(all_recs)
    stats={}
    for col in keys:
        if df[col].dtype==object:
            continue
        try:
            vals_real=df[df.label=="real"][col].astype(float).dropna()
            vals_sil=df[df.label=="silicone"][col].astype(float).dropna()
            if len(vals_real)<5 or len(vals_sil)<5:
                continue
            mean_real=float(vals_real.mean())
            std_real=float(vals_real.std())
            median_real=float(vals_real.median())
            mean_sil=float(vals_sil.mean())
            std_sil=float(vals_sil.std())
            median_sil=float(vals_sil.median())
            pooled_std=np.sqrt((std_real**2+std_sil**2)/2)
            cohen=(mean_real-mean_sil)/(pooled_std+1e-9)
            mad_real=float(np.median(np.abs(vals_real-median_real)) or 1e-6)
            mad_sil=float(np.median(np.abs(vals_sil-median_sil)) or 1e-6)
            sep=abs(median_real-median_sil)/(mad_real+mad_sil+1e-9)
            # corr with quality for real
            try:
                corr_q=float(np.corrcoef(vals_real, df[df.label=="real"]["overall_quality"].astype(float))[0,1])
            except:
                corr_q=0.0
            stats[col]={
                "mean_real":mean_real,
                "median_real":median_real,
                "mean_sil":mean_sil,
                "median_sil":median_sil,
                "cohen_d":float(cohen),
                "abs_cohen":float(abs(cohen)),
                "sep_mad":float(sep),
                "corr_q":corr_q,
                "abs_corr_q":abs(corr_q),
                "is_in_core": col in CURRENT_CORE
            }
        except Exception as e:
            continue
    # sort by abs cohen
    sorted_all=sorted(stats.items(), key=lambda x: x[1]["abs_cohen"], reverse=True)
    # filter not in core and quality-robust (|corr|<0.45)
    new_robust=[(k,v) for k,v in sorted_all if not v["is_in_core"] and abs(v["corr_q"])<0.45]
    new_robust_sorted=sorted(new_robust, key=lambda x: x[1]["abs_cohen"], reverse=True)
    print(f"\nTop 20 NEW metrics not in CORE and quality-robust (|corr|<0.45):")
    for k,v in new_robust_sorted[:20]:
        print(f"{k:40s} |d|={v['abs_cohen']:5.2f} d={v['cohen_d']:+6.2f} corr_q={v['corr_q']:+5.2f} sep={v['sep_mad']:4.1f} real_med={v['median_real']:.3f} sil_med={v['median_sil']:.3f}")

    # also top 20 new regardless of corr (for final list)
    new_all=[(k,v) for k,v in sorted_all if not v["is_in_core"]]
    new_all_sorted=sorted(new_all, key=lambda x: x[1]["abs_cohen"], reverse=True)
    print(f"\nTop 20 NEW metrics (any corr) by |d|:")
    for k,v in new_all_sorted[:20]:
        print(f"{k:40s} |d|={v['abs_cohen']:5.2f} corr={v['corr_q']:+5.2f} real={v['median_real']:.3f} sil={v['median_sil']:.3f}")

    # save json
    with open("/home/user/extended_stats.json","w") as f:
        json.dump({"all": dict(sorted_all), "new_robust_top20": new_robust_sorted[:20], "new_all_top20": new_all_sorted[:20]}, f, indent=2)

if __name__=="__main__":
    process_all()
