#!/usr/bin/env python3
"""
50+ simulations to find stable metrics

Idea:
- Take 10 high-quality real images (overall>0.6 from simple-test)
- For each, generate 5 degraded versions:
  * blur 0, 1, 3, 5
  * jpeg q 95, 85, 70, 50
  * downscale 1.0, 0.7, 0.5, 0.3
  * noise 0, 10, 20
  * combined (blur+downscale+jpeg)
- Compute metrics for each degraded version
- Measure stability: CV = std/mean across degradations for same identity
- Also measure discriminability: difference real vs silicone after degradation

Metrics tested: 30+ candidates

Output: stability ranking
"""

from pathlib import Path
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.morphology import disk, white_tophat
from scipy.ndimage import uniform_filter
from scipy.stats import skew
import json
from collections import defaultdict

REAL_DIR = Path("/home/user/dptn_new/simple-test/test-real")
SIL_DIR = Path("/home/user/dptn_new/simple-test/test-silicone")

def create_mask(h,w):
    mask=np.zeros((h,w), dtype=np.uint8)
    cv2.ellipse(mask, (w//2,h//2), (int(w*0.35), int(h*0.40)), 0,0,360,1,-1)
    return mask.astype(bool)

def extract_metrics_set(gray, mask):
    """Extract 30+ metrics in one go"""
    res={}
    # ensure uint8
    gray_u8 = np.clip(gray,0,255).astype(np.uint8)
    # quality
    blur = float(cv2.Laplacian(gray_u8, cv2.CV_64F).var())
    # GLCM
    try:
        skin_pixels=gray_u8[mask]
        if skin_pixels.size>0:
            lo,hi=np.percentile(skin_pixels,[2,98])
            span=max(hi-lo,1e-6)
            norm=np.clip((gray_u8.astype(float)-lo)/span,0,1)
            quant=(norm*31).astype(np.uint8)
            # crop
            coords=np.argwhere(mask)
            y0,y1=coords[:,0].min(), coords[:,0].max()+1
            x0,x1=coords[:,1].min(), coords[:,1].max()+1
            qcrop=quant[y0:y1,x0:x1]
            if qcrop.size>100:
                glcm=graycomatrix(qcrop, distances=[1,3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=32, symmetric=True, normed=True)
                diss=[float(graycoprops(glcm,"dissimilarity")[0,a]) for a in range(4)]
                homo=[float(graycoprops(glcm,"homogeneity")[0,a]) for a in range(4)]
                contr=[float(graycoprops(glcm,"contrast")[0,a]) for a in range(4)]
                energ=[float(graycoprops(glcm,"energy")[0,a]) for a in range(4)]
                corr=[float(graycoprops(glcm,"correlation")[0,a]) for a in range(4)]
                res["glcm_diss_d1_mean"]=float(np.mean(diss[:1])) if len(diss)>=1 else 0
                # for d=3: second distance index 1
                diss_d3=[float(graycoprops(glcm,"dissimilarity")[1,a]) for a in range(4)]
                res["glcm_diss_d3_mean"]=float(np.mean(diss_d3))
                res["glcm_diss_d3_aniso"]=float(np.std(diss_d3))
                res["glcm_diss_d3_std"]=float(np.std(diss_d3))
                res["glcm_homo_d1_mean"]=float(np.mean(homo[:1]))
                res["glcm_homo_d3_mean"]=float(np.mean(homo[1:2]) if len(homo)>1 else homo[0])
                res["glcm_contr_d1_mean"]=float(np.mean(contr[:1]))
                res["glcm_contr_d3_mean"]=float(np.mean(contr[1:2]) if len(contr)>1 else contr[0])
                res["glcm_energy_d1_mean"]=float(np.mean(energ[:1]))
                res["glcm_corr_d1_mean"]=float(np.mean(corr[:1]))
    except Exception as e:
        pass
    # LBP
    try:
        lbp1=local_binary_pattern(gray_u8, P=8, R=1, method="uniform")
        lbp1_skin=lbp1[mask]
        if lbp1_skin.size>0:
            res["lbp_r1_std"]=float(np.std(lbp1_skin))
            non_uniform=np.sum(lbp1_skin==9)/lbp1_skin.size
            res["lbp_r1_nonuniform"]=float(non_uniform)
            # hist entropy
            hist,_=np.histogram(lbp1_skin, bins=10, range=(0,10), density=True)
            hist=hist[hist>0]
            res["lbp_r1_hist_ent"]=float(-np.sum(hist*np.log2(hist))) if hist.size>0 else 0
    except:
        pass
    try:
        lbp2=local_binary_pattern(gray_u8, P=8, R=2, method="uniform")
        lbp2_skin=lbp2[mask]
        if lbp2_skin.size>0:
            res["lbp_r2_std"]=float(np.std(lbp2_skin))
            non_uniform2=np.sum(lbp2_skin==9)/lbp2_skin.size
            res["lbp_r2_nonuniform"]=float(non_uniform2)
    except:
        pass
    # FFT
    try:
        coords=np.argwhere(mask)
        y0,y1=coords[:,0].min(), coords[:,0].max()+1
        x0,x1=coords[:,1].min(), coords[:,1].max()+1
        crop=gray_u8[y0:y1, x0:x1].astype(float)
        ch,cw=crop.shape[0]//2, crop.shape[1]//2
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
            low=power[radius<=4].sum()
            high=power[radius>8].sum()
            res["fft_high_low"]=float(high/(low+1e-6)) if low>0 else 0
            res["fft_highfreq"]=float(high/(power.sum()+1e-6))
            # slope
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
                res["beta"]=float(-slope)
            else:
                res["beta"]=2.5
            # aniso angular
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
                ang=ang[ang>0]
                ent=-np.sum(ang*np.log(ang+1e-9))
                max_ent=np.log(n_bins)
                res["fft_angular_ent"]=float(ent/max_ent)
                res["fft_aniso"]=float(1-ent/max_ent)
    except Exception as e:
        pass
    # local var
    try:
        gray_f=gray_u8.astype(float)
        for wname,w in [("w15",15),("w31",31)]:
            lm=uniform_filter(gray_f, size=w)
            lm_sq=uniform_filter(gray_f**2, size=w)
            lvar=np.maximum(lm_sq-lm**2,0)
            lstd=np.sqrt(lvar)
            vm=mask & (lm>1)
            if vm.any():
                cv=lstd[vm]/lm[vm]
                cv=np.clip(cv,0,5)
                res[f"homo_cv_{wname}"]=float(np.mean(cv))
    except:
        pass
    # pore tophat
    try:
        for rname,r in [("r2",2),("r4",4)]:
            th=white_tophat(gray_u8, disk(r))
            vals=th[mask]
            if vals.size>0:
                res[f"pore_{rname}_mean"]=float(np.mean(vals))
                res[f"pore_{rname}_std"]=float(np.std(vals))
                thr=np.mean(vals)+np.std(vals)
                dens=np.sum(vals>thr)/max(mask.sum(),1)
                res[f"pore_{rname}_dens"]=float(dens)
    except:
        pass
    # entropy
    try:
        pixels=gray_u8[mask]
        if pixels.size>0:
            hist,_=np.histogram(pixels, bins=32, range=(0,255), density=False)
            prob=hist/hist.sum()
            prob=prob[prob>0]
            ent=-np.sum(prob*np.log2(prob)) if prob.size>0 else 0
            res["hist_entropy"]=float(ent)
    except:
        pass
    # edge
    try:
        edges=cv2.Canny(gray_u8,40,120)
        if mask.any():
            res["edge_dens"]=float(edges[mask].mean()/255.0)
    except:
        pass
    res["blur"]=blur
    return res

def degrade_image(img, blur_k=0, jpeg_q=95, scale=1.0, noise_sigma=0):
    """Simulate quality degradation"""
    out=img.copy()
    # downscale
    if scale<1.0:
        h,w=out.shape[:2]
        nh, nw=int(h*scale), int(w*scale)
        if nh>10 and nw>10:
            out=cv2.resize(out, (nw,nh), interpolation=cv2.INTER_AREA)
            out=cv2.resize(out, (w,h), interpolation=cv2.INTER_CUBIC)
    # blur
    if blur_k>0:
        k=int(blur_k)*2+1
        out=cv2.GaussianBlur(out, (k,k), 0)
    # noise
    if noise_sigma>0:
        noise=np.random.normal(0, noise_sigma, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32)+noise,0,255).astype(np.uint8)
    # jpeg
    if jpeg_q<95:
        encode_param=[int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_q)]
        _, enc=cv2.imencode('.jpg', out, encode_param)
        out=cv2.imdecode(enc, 1)
    return out

def main():
    # pick 10 high-quality real images: overall high blur high
    # First compute quality for all real
    real_paths=list(REAL_DIR.glob("*.png"))[:30]  # limit 30 for speed
    # select 10 with highest blur
    quals=[]
    for p in real_paths:
        img=cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        blur=cv2.Laplacian(img, cv2.CV_64F).var()
        quals.append((blur,p))
    quals=sorted(quals, reverse=True)
    selected=[p for _,p in quals[:10]]
    print(f"Selected {len(selected)} high-quality real images for simulation")

    # degradation configs: 5 levels + combined = 50+ simulations (10 images * 5 degradations =50)
    degradations=[
        {"name":"high","blur_k":0, "jpeg_q":95, "scale":1.0, "noise":0},
        {"name":"mid_blur1","blur_k":1, "jpeg_q":95, "scale":1.0, "noise":0},
        {"name":"mid_blur3","blur_k":3, "jpeg_q":95, "scale":1.0, "noise":0},
        {"name":"mid_jpeg85","blur_k":0, "jpeg_q":85, "scale":1.0, "noise":0},
        {"name":"mid_jpeg70","blur_k":0, "jpeg_q":70, "scale":1.0, "noise":0},
        {"name":"low_scale07","blur_k":0, "jpeg_q":95, "scale":0.7, "noise":0},
        {"name":"low_scale05","blur_k":0, "jpeg_q":95, "scale":0.5, "noise":0},
        {"name":"low_noise10","blur_k":0, "jpeg_q":95, "scale":1.0, "noise":10},
        {"name":"low_combined","blur_k":2, "jpeg_q":75, "scale":0.6, "noise":10},
        {"name":"very_low","blur_k":4, "jpeg_q":60, "scale":0.4, "noise":15},
    ]
    # We need 50+ simulations: 10 images * 10 degradations =100 simulations

    all_records=[]

    for img_path in selected:
        img_bgr=cv2.imread(str(img_path))
        h,w=img_bgr.shape[:2]
        mask=create_mask(h,w)
        for deg in degradations:
            degraded_bgr=degrade_image(img_bgr, blur_k=deg["blur_k"], jpeg_q=deg["jpeg_q"], scale=deg["scale"], noise_sigma=deg["noise"])
            # gray for metrics
            gray=cv2.cvtColor(degraded_bgr, cv2.COLOR_BGR2GRAY)
            metrics=extract_metrics_set(gray, mask)
            metrics["source"]=img_path.name
            metrics["degradation"]=deg["name"]
            metrics["blur_k"]=deg["blur_k"]
            metrics["jpeg_q"]=deg["jpeg_q"]
            metrics["scale"]=deg["scale"]
            metrics["noise"]=deg["noise"]
            all_records.append(metrics)

    # Now compute stability per metric: CV across degradations per source
    from collections import defaultdict
    stability={}
    # group by metric
    metric_names=set(k for r in all_records for k in r.keys() if k not in ["source","degradation","blur_k","jpeg_q","scale","noise"])
    for mname in metric_names:
        # for each source, compute std/mean across degradations
        per_source_cv=[]
        per_source_range=[]
        for src in selected:
            src_name=src.name
            vals=[r[mname] for r in all_records if r["source"]==src_name and mname in r]
            if len(vals)<2:
                continue
            vals=np.array(vals, dtype=float)
            mean=np.mean(vals)
            std=np.std(vals)
            cv=std/(abs(mean)+1e-9)
            per_source_cv.append(cv)
            per_source_range.append(float(np.max(vals)-np.min(vals)))
        if per_source_cv:
            stability[mname]={
                "mean_cv": float(np.mean(per_source_cv)),
                "median_cv": float(np.median(per_source_cv)),
                "mean_range": float(np.mean(per_source_range)),
                "count": len(per_source_cv)
            }

    # sort by mean_cv ascending (more stable = lower CV)
    sorted_stable=sorted(stability.items(), key=lambda x: x[1]["mean_cv"])
    print("\n=== Stability ranking (lower CV = more stable) ===")
    for m,(stats) in sorted_stable[:30]:
        print(f"{m:25s} CV={stats['mean_cv']:.3f} range={stats['mean_range']:.3f}")

    # Save
    with open("/home/user/stability_50.json","w") as f:
        json.dump({"stability": stability, "sorted": sorted_stable}, f, indent=2)

    # Now also compute discriminability on simple-test using these metrics
    # Load previous csv for real vs silicone comparison
    import pandas as pd
    df=pd.read_csv("/home/user/texture_metrics.csv")
    # For each stable metric, compute Cohen d and quality correlation already done, but recompute discriminability
    # We have stability, now combine: stable + discriminative = good
    # Load stats.json for cohen
    import json as js
    with open("/home/user/texture_stats.json") as f:
        stats=json.load(f)
    # stats is dict metric->...
    # Merge
    combined=[]
    for mname, stab in stability.items():
        if mname in stats:
            cohen=stats[mname]["cohen_d"]
            abs_cohen=abs(cohen)
            corr=abs(stats[mname]["corr_quality_real"])
            # score: high cohen, low CV, low corr
            # quality-robust score = abs_cohen / (mean_cv+0.1) / (corr+0.5)
            score = abs_cohen / (stab["mean_cv"]+0.2) / (corr+0.5)
            combined.append((mname, abs_cohen, stab["mean_cv"], corr, score))
    combined=sorted(combined, key=lambda x: x[4], reverse=True)
    print("\n=== Combined stable + discriminative + quality-robust ranking ===")
    for mname, cohen, cv, corr, score in combined[:30]:
        print(f"{mname:30s} |d|={cohen:5.2f} CV={cv:.3f} corr_q={corr:.2f} SCORE={score:.3f}")

    with open("/home/user/combined_ranking_50.json","w") as f:
        json.dump(combined, f, indent=2)

if __name__=="__main__":
    main()
