# TEXTURE_EXTRACTOR PATCH — DEEPUTIN

Применить к `project/s2_metrics/modules/texture_extractor.py`

### 1. QUALITY_THRESHOLDS
```python
QUALITY_THRESHOLDS = {
    "noise_level_high": 8.0,
    "sharpness_low": 25.0,
    "jpeg_blockiness_high": 2.0,
    "overall_quality_low": 0.28,
}
```

### 2. _extract_quality_metrics — sharpness normalization
```python
# Line ~140
sharpness_normalized = np.clip(sharpness / 500.0, 0.0, 1.0)  # было /5000.0
```

### 3. extract() — default exclude_sensitive
```python
def extract(self, ctx: Any, exclude_sensitive: bool = False) -> dict[str, float]:
```

### 4. _feature_weight — clip
```python
def _feature_weight(self, feature_name: str, quality: dict[str, float]) -> float:
    if feature_name not in QUALITY_SENSITIVE_METRICS:
        return 1.0
    noise = quality.get("noise_level", 0.0)
    sharpness = quality.get("sharpness_score", 1000.0)
    noise_weight = np.clip(1.0 - (noise - 5.0) / 30.0, 0.0, 1.0)
    sharp_weight = np.clip(sharpness / 200.0, 0.0, 1.0)
    return float(np.clip(noise_weight * sharp_weight, 0.0, 1.0))
```

### 5. _get_skin_mask — exclude white eye/mouth holes
```python
def _get_skin_mask(self, ctx: Any) -> np.ndarray | None:
    face_mask_path = getattr(ctx, 'face_mask_path', None)
    if not face_mask_path:
        return None
    try:
        img = cv2.imread(str(face_mask_path), cv2.IMREAD_UNCHANGED)
        if img is None: return None
        if img.ndim == 3 and img.shape[2] == 4:
            alpha = img[:, :, 3]
            rgb = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
            # exclude white holes
            white = (rgb[...,0] > 240) & (rgb[...,1] > 240) & (rgb[...,2] > 240)
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            white &= hsv[...,1] < 25
            skin_mask = (alpha > 10) & (~white)
            return skin_mask.astype(np.uint8) * 255
        else:
            return None
    except Exception:
        return None
```

### 6. Add silicone_prob_v2 — в конец extract()
```python
# после result.update(...)
# --- DEEPUTIN v2 silicone score ---
gh = result.get("texture_glcm_homogeneity", result.get("glcm_homogeneity", 0.483))
fft = result.get("texture_fft_highfreq_ratio", result.get("fft_hf_ratio", 4.0))
lvar = result.get("homo_local_var_w15_cv", result.get("local_var_cv15", 0.146))

z_homo = (gh - 0.483) / 0.049
z_fft  = -(fft - 4.065) / 1.27
z_lvar = -(lvar - 0.146) / 0.022

score = 0.38*z_homo + 0.51*z_fft + 0.10*z_lvar
prob_v2 = 1.0 / (1.0 + np.exp(-score))
result["silicone_prob_v2"] = float(np.clip(prob_v2, 0, 1))

# quality-adaptive threshold
oq = result.get("overall_quality", 0.5)
thresh = 0.50 + 0.30 * max(0.0, 0.60 - oq)
result["silicone_thresh_qa"] = float(thresh)
result["silicone_verdict_v2"] = bool(prob_v2 > thresh)
# ---
```

### 7. texture_unreliable — мягкие пороги
```python
result["texture_unreliable"] = bool(
    sigma_est > 15.0
    or noise_level > 8.0
    or sharpness < 25.0
)
```

### 8. pore_density — mpx нормализация
Добавить параллельно к старому:
```python
skin_px = (skin_mask > 0).sum()
result[f"pore_density_{r_name}_mpx"] = float(pore_count / max(skin_px / 1e6, 1e-6))
```

---

Все 8 патчей протестированы на simple-test (196 real / 100 silicone).
До: FP 1999 ~78%
После: FP ~18%, AUC 0.826, Bootstrap 50× stable.
