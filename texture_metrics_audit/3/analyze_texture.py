#!/usr/bin/env python3
"""
Texture analysis for DEEPUTIN skin vs silicone
Simple-test dataset calibration
"""
import cv2
import numpy as np
from pathlib import Path
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.restoration import estimate_sigma, denoise_wavelet
from skimage.morphology import white_tophat, disk
from skimage.filters.rank import entropy as rank_entropy
from scipy import ndimage
from scipy.stats import skew, kurtosis
import json
from collections import defaultdict

REAL_DIR = Path("/home/user/dptn/simple-test/test-real")
SILICONE_DIR = Path("/home/user/dptn/simple-test/test-silicone")

def load_skin_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, None
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:,:,3]
        rgb = cv2.cvtColor(img[:,:,:3], cv2.COLOR_BGR2RGB)
        # skin mask: alpha > 10, and exclude pure white eyes/mouth holes (R>240,G>240,B>240)
        white = (rgb[:,:,0] > 240) & (rgb[:,:,1] > 240) & (rgb[:,:,2] > 240)
        mask = (alpha > 10) & (~white)
    else:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim==3 else img
        mask = np.ones(rgb.shape[:2], dtype=bool)
    return rgb, mask

def quality_metrics(gray):
    # noise
    median_filtered = cv2.medianBlur(gray, 3)
    noise = float(np.mean(np.abs(gray.astype(np.float32) - median_filtered.astype(np.float32))))
    # sharpness
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(np.var(lap))
    # jpeg blockiness approx
    h,w = gray.shape
    if h>16 and w>16:
        boundary = gray[:,7::8].astype(np.float32)
        inside = gray[:,3::8].astype(np.float32)
        blockiness = float(np.mean(np.abs(boundary.mean() - inside.mean()))) / 10.0 + 1.0
    else:
        blockiness = 1.0
    overall = np.clip(sharpness/5000.0,0,1)*0.7 + np.clip(1-noise/35.0,0,1)*0.3
    return {"noise_level":noise, "sharpness_score":sharpness, "jpeg_blockiness":blockiness, "overall_quality":float(overall)}

def extract_metrics(rgb, mask):
    out = {}
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # quality first
    q = quality_metrics(gray)
    out.update(q)
    skin_px = gray[mask]
    if skin_px.size < 100:
        return out
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_c = clahe.apply(gray)
    skin_clahe = gray_c[mask]
    out["gray_mean"] = float(skin_px.mean())
    out["gray_std"] = float(skin_px.std())
    # sigma
    try:
        sigma_est = estimate_sigma(gray_c, channel_axis=None)
        out["noise_sigma"] = float(sigma_est)
    except Exception:
        out["noise_sigma"] = 0.0
        sigma_est = 0
    # LBP multi-scale
    for R,P,name in [(1,8,"lbp_r1"), (2,8,"lbp_r2"), (3,16,"lbp_r3")]:
        try:
            lbp = local_binary_pattern(gray_c, P=P, R=R, method="uniform")
            v = lbp[mask]
            out[f"{name}_mean"] = float(v.mean())
            out[f"{name}_std"] = float(v.std())
            # uniformity entropy
            hist,_ = np.histogram(v, bins=P+2, range=(0,P+2), density=True)
            hist = hist[hist>0]
            out[f"{name}_entropy"] = float(-np.sum(hist*np.log2(hist))) if hist.size else 0
        except Exception:
            pass
    # GLCM
    try:
        lo,hi = np.percentile(skin_clahe, [2,98])
        span = max(hi-lo,1e-6)
        norm = np.clip((gray_c.astype(float)-lo)/span,0,1)
        quant = (norm*32).astype(np.uint8)
        glcm = graycomatrix(quant, distances=[1,2,3,5], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=33, symmetric=True, normed=True)
        out["glcm_contrast"] = float(graycoprops(glcm, "contrast").mean())
        out["glcm_dissimilarity"] = float(graycoprops(glcm, "dissimilarity").mean())
        out["glcm_homogeneity"] = float(graycoprops(glcm, "homogeneity").mean())
        out["glcm_energy"] = float(graycoprops(glcm, "energy").mean())
        out["glcm_correlation"] = float(graycoprops(glcm, "correlation").mean())
        # anisotropy: std across angles
        diss = graycoprops(glcm, "dissimilarity")
        out["glcm_anisotropy"] = float(diss.std(axis=1).mean())
    except Exception:
        pass
    # FFT patch-based
    try:
        ys,xs = np.where(mask)
        if len(ys)>0:
            y0,y1 = ys.min(), ys.max()
            x0,x1 = xs.min(), xs.max()
            crop = gray_c[y0:y1+1, x0:x1+1]
            crop_m = mask[y0:y1+1, x0:x1+1]
            ratios=[]
            for py in range(0, crop.shape[0]-64+1, 32):
                for px in range(0, crop.shape[1]-64+1, 32):
                    pm = crop_m[py:py+64, px:px+64]
                    if pm.sum() < 64*64*0.6: continue
                    patch = crop[py:py+64, px:px+64].astype(np.float32)
                    patch = patch - patch.mean()
                    f = np.fft.fftshift(np.fft.fft2(patch))
                    mag = np.abs(f)
                    h,w = mag.shape; cy, cx = h//2, w//2
                    yy,xx = np.ogrid[:h,:w]
                    r = np.sqrt((yy-cy)**2 + (xx-cx)**2)
                    low = mag[r<=4].sum()
                    high = mag[r>8].sum()
                    if low>1e-6: ratios.append(high/low)
            if ratios:
                out["fft_hf_ratio"] = float(np.mean(ratios))
                out["fft_hf_std"] = float(np.std(ratios))
    except Exception:
        pass
    # pores / tophat
    for r in [2,3,4,6]:
        try:
            th = white_tophat(gray_c, disk(r))
            v = th[mask]
            out[f"tophat_r{r}_mean"] = float(v.mean())
            out[f"tophat_r{r}_std"] = float(v.std())
        except Exception:
            pass
    # gradient
    try:
        sx = cv2.Sobel(gray_c, cv2.CV_64F, 1,0, ksize=3)
        sy = cv2.Sobel(gray_c, cv2.CV_64F, 0,1, ksize=3)
        mag = np.sqrt(sx**2+sy**2)
        v = mag[mask]
        out["grad_mean"] = float(v.mean())
        out["grad_std"] = float(v.std())
        out["grad_skew"] = float(skew(v.ravel()))
    except Exception:
        pass
    # local entropy
    try:
        ent = rank_entropy(gray_c, disk(5))
        v = ent[mask]
        out["entropy_median"] = float(np.median(v))
        out["entropy_iqr"] = float(np.percentile(v,75)-np.percentile(v,25))
    except Exception:
        pass
    # color / albedo LAB
    try:
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l = lab[:,:,0][mask]; a = lab[:,:,1][mask]; b = lab[:,:,2][mask]
        out["albedo_a_std"] = float(a.std())
        out["albedo_b_std"] = float(b.std())
        out["albedo_a_mean"] = float(a.mean())
        out["albedo_viability"] = float(a.std()/(l.mean()+1e-6))
        # blood/hemoglobin speckling: a* local std
        a_img = lab[:,:,1].astype(float)
        from scipy.ndimage import uniform_filter
        m = uniform_filter(a_img, size=7)
        m_sq = uniform_filter(a_img**2, size=7)
        local_std = np.sqrt(np.maximum(m_sq - m**2,0))
        out["albedo_a_local_std"] = float(local_std[mask].mean())
    except Exception:
        pass
    # specular HSV
    try:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        v = hsv[:,:,2][mask]; s = hsv[:,:,1][mask]
        spec = (v>220) & (s<40)
        out["specular_ratio"] = float(spec.mean()) if spec.size else 0
    except Exception:
        pass
    # fractal / micro-contrast
    try:
        # difference of Gaussians
        blur1 = cv2.GaussianBlur(gray_c, (0,0), 1.0)
        blur2 = cv2.GaussianBlur(gray_c, (0,0), 3.0)
        dog = (blur1.astype(float)-blur2.astype(float))[mask]
        out["dog_std"] = float(dog.std())
        # local variance cv
        from scipy.ndimage import uniform_filter
        gf = gray_c.astype(float)
        m = uniform_filter(gf, size=15)
        msq = uniform_filter(gf*gf, size=15)
        lv = np.sqrt(np.maximum(msq-m*m,0))
        vm = m[mask]; vs = lv[mask]
        valid = vm>1
        cv_vals = vs[valid]/vm[valid] if valid.any() else np.array([0])
        out["local_var_cv15"] = float(np.mean(np.clip(cv_vals,0,10)))
    except Exception:
        pass
    # wavelet energy ratio
    try:
        # simple high-freq energy via Laplacian
        lap = cv2.Laplacian(gray_c, cv2.CV_64F)
        out["lap_var"] = float(lap[mask].var()) if mask.sum()>0 else 0
    except Exception:
        pass
    return out

def process_folder(folder):
    rows=[]
    for p in sorted(folder.glob("*.png")):
        rgb,mask = load_skin_rgb(p)
        if rgb is None: continue
        m = extract_metrics(rgb, mask)
        m["path"] = str(p.name)
        rows.append(m)
    return rows

print("Processing real...")
real_rows = process_folder(REAL_DIR)
print(f"real: {len(real_rows)}")
print("Processing silicone...")
sil_rows = process_folder(SILICONE_DIR)
print(f"silicone: {len(sil_rows)}")

# Save raw
with open("/home/user/dptn/analysis_raw.json","w") as f:
    json.dump({"real":real_rows, "silicone":sil_rows}, f)

print("Done, running analysis...")
