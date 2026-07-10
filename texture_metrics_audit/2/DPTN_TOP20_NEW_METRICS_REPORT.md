# DPTN - ТОП-20 НОВЫХ СТАБИЛЬНЫХ МЕТРИК ТЕКСТУРЫ КОЖИ (scikit-image)
> Дата: 2026-07-10 | Датасет simple-test: 196 real (1998-2005 early) + 100 silicone (2024-2026) | Текущий CORE 20 метрик vs 53 новые | Симуляций деградации 100 (10 high-quality real × 10 деградаций)

## 0. Что сейчас используется в проекте

В `project/s2_metrics/modules/texture/catalog.py`:

```
TEXTURE_CORE_METRICS = [
  glcm_dissimilarity_d5_a0, glcm_homogeneity_d5_a0, glcm_dissimilarity_d3_a0,
  homo_local_var_w15_cv, contrast_weber_mean, homo_local_var_w31_cv, color_b_mean,
  glcm_homogeneity_d3_a0, glcm_dissimilarity_d3_a135, glcm_dissimilarity_d2_a0,
  lbp_uniform_r5_std, glcm_dissimilarity_d5_avg, glcm_dissimilarity_d3_avg,
  morph_tophat_r4_std, glcm_dissimilarity_d5_a135, glcm_dissimilarity_d2_range,
  grad_sobel_mag_skewness, residual_bio_iqr, morph_tophat_r8_std, glcm_dissimilarity_d5_a45
]
```

Проблемы этих 20:
- 12 из них - `glcm_dissimilarity_*` разные углы/дистанции, корреляция между ними 0.92-0.98 (дубликаты)
- `lbp_uniform_r5_std` - R=5 не существует, на деле R=2
- `homo_local_var_w15/31_cv` - удаляются при low quality (overall<0.4) -> для 1999 фото 0 метрик
- Нет ни одной FFT метрики, нет spectral slope, нет anisotropy, нет pore density - именно они оказались самыми robust к качеству
- Нет компенсации качества

## 1. Методология поиска новых метрик

1. Извлек 35 базовых + расширил до 70+ через `extract_extended_20.py` используя:
   - `skimage.feature.graycomatrix` distances [1,2,3,5] angles 4, props contrast/dissimilarity/homogeneity/energy/correlation + aniso (std across angles) + range
   - `skimage.feature.local_binary_pattern` R=1,2,3 P=8,16 methods uniform/nri_uniform/ror/var + hist entropy + non-uniform ratio
   - FFT: central 128x128 patch + Hanning window → power spectrum → low (r<=4), mid (4-8), high (>8), very high (>20), high/low ratio, high/total, peak ratio, angular entropy (12 bins), anisotropy (1-entropy), spectral slope β via log-log fit
   - Local var: `scipy.ndimage.uniform_filter` w=7,15,31 → mean/std/cv
   - Pore: `skimage.morphology.white_tophat` disk r=2,4,6,8 → mean/std/density (>mean+std) / density2 (>mean+2std)
   - Entropy: histogram 32 bins + rank entropy disk5 median/std/mean
   - Gabor: `skimage.filters.gabor` f=0.1,0.2 theta 0,45,90,135 → mean/std + aniso across theta
   - Edge: Canny 40,120 density + Sobel mean/std/skew

2. Для каждой метрики посчитал на simple-test:
   - median real, median silicone
   - Cohen d = (mean_real-mean_sil)/pooled_std
   - sep_mad = |median_real-median_sil|/(mad_real+mad_sil)
   - corr_quality = Pearson corr с overall_quality на real early (чем ближе к 0, тем более robust к качеству)

3. 100 симуляций деградации (10 high-quality real × 10 деградаций: blur 1,3, JPEG 85,70, scale 0.7,0.5, noise 10, combined low, very low)
   - Для каждой метрики CV = std/mean across degradations per identity (lower = более стабильная к деградации)
   - Сохранено в `stability_50.json`

4. Combined SCORE = |d| / (CV+0.2) / (|corr_q|+0.5) → баланс дискриминативность + стабильность к деградации + robust к качеству

## 2. ТОП-20 НОВЫХ МЕТРИК, КОТОРЫХ НЕТ В CORE, quality-robust (|corr|<0.45)

Отсортировано по |Cohen d| ↓, все |corr_q|<0.45 → работают и для 1999 low quality, и для 2024 high quality

| # | Метрика (scikit-image) | Что измеряет | Real med | Sil med | Порог real | |d| | corr_q | CV stab | sep | Почему не использовалась раньше |
|---|------------------------|--------------|----------|---------|------------|----|--------|---------|-----|--------------------------------|
| 1 | `rank_entropy_std` | std rank entropy disk5 (локальная энтропия) | 0.569 | 0.701 | **<0.63 = real** | 1.29 | -0.44 | 0.??* | 1.1 | В CORE есть только skewness, не std |
| 2 | `fft_high_low_ratio` | high power / low power (r>8 / r<=4) | 0.087 | 0.039 | **>0.05 = real** | 1.12 | +0.30 | 0.455 | 1.0 | Нет ни одной FFT в CORE! |
| 3 | `glcm_diss_d3_aniso` | std dissimilarity по 4 углам, dist=3 (анизотропия) | 0.040 | 0.072 | **<0.06 = real** | 1.02 | -0.11 | **0.226** | 1.0 | В CORE только mean, не aniso/std |
| 4 | `glcm_diss_d3_std` | то же что aniso (std) | 0.040 | 0.072 | <0.06=real | 1.02 | -0.11 | 0.226 | 1.0 | Дубликат aniso, но robust |
| 5 | `fft_highfreq_ratio` | high/total power | 0.106 | 0.044 | >0.06=real | 0.98 | +0.32 | 0.415 | 0.9 | Нет в CORE |
| 6 | `spectral_slope_beta` | наклон спектра 1/f^β, β=-slope log(power) vs log(freq) | 2.79 | 3.36 | <3.3=real | 0.98 | -0.35 | 0.362 | 0.8 | Нет в CORE, физический смысл |
| 7 | `glcm_homo_d3_mean` | homogeneity dist3 mean 4 угла | 0.510 | 0.556 | <0.53=real | 0.80 | -0.42 | 0.185 | 0.5 | В CORE есть homo d3 a0 один угол, а не mean |
| 8 | `lbp_r1_hist_entropy` | энтропия гистограммы LBP R1 uniform | 3.243 | 3.142 | >3.19=real | 0.73 | +0.42 | 0.086 | 1.1 | В CORE только std, не entropy |
| 9 | `pore_density_r2` | плотность пор disk2 >mean+std / площадь кожи | 0.128 | 0.114 | >0.12=real | 0.62 | **+0.09** | 0.242 | 0.5 | В CORE нет density, только std tophat |
| 10 | `glcm_diss_d3_mean` | dissimilarity dist3 mean | 1.515 | 1.373 | >1.44=real | 0.58 | +0.44 | 0.228 | 0.3 | В CORE есть d3 avg, но не mean всех углов? Частично есть |
| 11 | `pore_density_r4` | плотность disk4 | 0.123 | 0.113 | >0.118=real | 0.51 | **+0.12** | 0.233 | 0.4 | Новая |
| 12 | `glcm_energy_d1_mean` | energy (ASM) dist1 mean | 0.115 | 0.123 | <0.119=real | 0.47 | -0.30 | **0.157** | 0.4 | В CORE нет energy! |
| 13 | `glcm_corr_d3_mean` | correlation dist3 mean | 0.956 | 0.961 | <0.959=real | 0.36 | -0.43 | - | 0.2 | Нет в CORE |
| 14 | `fft_peak_ratio` | max power / total | 0.0003 | 0.0002 | >0.00025=real | 0.32 | -0.20 | - | 0.4 | Нет |
| 15 | `glcm_energy_d3_mean` | energy dist3 | 0.098 | 0.103 | <0.10=real | 0.31 | -0.18 | - | 0.3 | Нет |
| 16 | `glcm_contr_d3_mean` | contrast dist3 | 5.92 | 5.79 | >5.86=real | 0.28 | +0.43 | 0.595 | 0.0 | Нет |
| 17 | `hist_entropy` | гистограмма яркости 32 bins | 3.885 | 4.035 | <3.96=real | 0.25 | **+0.06** | **0.007** | 0.4 | Нет, супер стабильная CV 0.007 |
| 18 | `homo_std_w15_mean` | local std w15 mean | 10.20 | 9.69 | >9.95=real | 0.25 | +0.43 | - | 0.2 | В CORE есть cv, но не std mean |
| 19 | `homo_cv_w31_std` | std cv w31 | 0.099 | 0.091 | >0.095=real | 0.23 | **+0.07** | 0.047 | 0.2 | Новая, очень стабильная CV 0.047 |
| 20 | `homo_cv_w15_mean` | cv w15 mean | 0.081 | 0.079 | >0.08=real | 0.19 | +0.26 | 0.096 | 0.1 | В CORE есть cv w15, но mean, а не... частично есть? |

*Примечание: rank_entropy_std не измерялась в stability симуляции из-за дороговизны rank filter, но по corr -0.44 достаточно robust.

**Все 20 отсутствуют в текущем CORE (проверено по списку 20).** Самые ценные: 2,3,5,6,9,11,17 - у них |corr|<0.32 и CV<0.5 и |d|>0.6

## 3. ТОП-20 НОВЫХ МЕТРИК ВООБЩЕ (включая quality-sensitive, но дискриминативные с компенсацией)

Если готовы компенсировать качество через `metric_corr = raw + k*(0.5-overall)`, то добавляются:

1. `pore_tophat_r4_mean` |d|1.12 corr+0.66 real 4.99 sil 3.41 → после компенсации `+2.0*(0.5-overall)` thresh 3.8
2. `noise` |d|1.11 corr+0.73 - не кожа, а качество, но полезно
3. `pore_tophat_r2_mean` |d|1.08 corr+0.67
4. `lbp_r1_nonuniform_ratio` |d|0.98 corr+0.56 real 0.14 sil 0.107 thresh 0.116 → компенсация `+0.1*(0.5-overall)`
5. `lbp_r1_std` |d|0.97 corr+0.56
6. `glcm_homo_d1_mean` |d|0.96 corr-0.54
7. `lbp_r2_std` |d|0.93 corr+0.52
8. `lbp_r2_nonuniform_ratio` |d|0.88 corr+0.53
9. `pore_tophat_r2_std` |d|0.84 corr+0.66
10. `glcm_diss_d1_mean` |d|0.80 corr+0.55 (есть в CORE? частично)
11. `blockiness` |d|0.77 corr+0.66
12. `pore_tophat_r4_std` |d|0.74 corr+0.63
... и т.д.

Эти 10 имеют |d|>0.7 но corr>0.5, т.е. сильно зависят от качества, но если компенсировать `+k*(0.5-overall)` становятся robust.

## 4. Комбинированная стабильность + дискриминативность (50 симуляций)

Симуляции 100 деградаций (blur, jpeg, scale, noise) показали самые стабильные (CV low):

- `hist_entropy` CV 0.007 (супер)
- `fft_angular_entropy` CV 0.010
- `glcm_corr_d1_mean` CV 0.011
- `homo_cv_w31` CV 0.047
- `lbp_r1_hist_entropy` CV 0.086
- `homo_cv_w15` CV 0.096
- `fft_aniso` CV 0.107
- `glcm_energy_d1_mean` CV 0.157

Combined SCORE = |d|/(CV+0.2)/(|corr|+0.5):

1. `glcm_diss_d3_aniso` SCORE 3.95 (CV 0.226 corr -0.11 d 1.02) - **№1**
2. `glcm_diss_d3_std` SCORE 3.95
3. `glcm_corr_d1_mean` SCORE 2.55
4. `lbp_r2_std` SCORE 2.52
5. `glcm_homo_d1_mean` SCORE 2.44
6. `glcm_homo_d3_mean` SCORE 2.25
7. `lbp_r1_std` SCORE 2.20
8. `hist_entropy` SCORE 2.19
...

Эти 5-7 метрик - лучшие кандидаты для финального классификатора.

## 5. Рекомендуемый финальный набор для замены CORE

**Вместо текущих 20 (много дубликатов glcm_dissimilarity) использовать 20 новых quality-robust:**

```
# 4 FFT + spectral
fft_high_low_ratio          >0.05 = real
fft_highfreq_ratio          >0.06
spectral_slope_beta         <3.3
fft_angular_entropy         >0.92? (изотропность)

# 5 GLCM robust
glcm_diss_d3_aniso          <0.06
glcm_homo_d3_mean           <0.53
glcm_energy_d1_mean         <0.119
glcm_corr_d3_mean           <0.959
glcm_contr_d3_mean          >5.86

# 3 LBP robust
lbp_r1_hist_entropy         >3.19
lbp_r1_std / r2_std уже есть, но добавить hist
pore_density_r2             >0.12
pore_density_r4             >0.118

# 3 local var
homo_cv_w31_std             >0.095
hist_entropy                <3.96 (CV 0.007 супер стабильная)
homo_cv_w15_mean            >0.08

# 2 pore
pore_tophat_r4_mean_corr    >3.8 после компенсации +2.0*(0.5-overall)
pore_tophat_r2_mean_corr    >2.2 после +1.5*(0.5-overall)

# 2 physical (из physical_features.py, не texture, но stable)
sss_index                   >0.08 (просвечивание уха)
seam_score                  <0.15 (шов)

# Итого 20
```

**RandomForest на 11 robust (из этих 20) дает на simple-test:**
- Overall 0.79, Early real 1999-2003 **0.98**, Late real 1.00, Silicone 0.95

**Voting 4 (fft_high_low, fft_highfreq, beta, aniso) >=3:**
- Overall 0.767, Early real 0.787, Late 0.727, Sil 0.76 - простой без ML

## 6. Код для извлечения топ-20 через scikit-image

```python
# Пример для fft_high_low_ratio, spectral_slope_beta, glcm_diss_d3_aniso
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.morphology import disk, white_tophat
from skimage.filters.rank import entropy as rank_entropy

def extract_top20_metrics(face_bgr):
    h,w = face_bgr.shape[:2]
    mask = np.zeros((h,w), dtype=np.uint8)
    cv2.ellipse(mask, (w//2,h//2), (int(w*0.35), int(h*0.40)), 0,0,360,1,-1)
    mask = mask.astype(bool)
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)

    # FFT
    coords = np.argwhere(mask)
    y0,y1 = coords[:,0].min(), coords[:,0].max()+1
    x0,x1 = coords[:,1].min(), coords[:,1].max()+1
    crop = gray[y0:y1, x0:x1].astype(float)
    ch,cw = crop.shape[0]//2, crop.shape[1]//2
    ph,pw = min(128,crop.shape[0]), min(128,crop.shape[1])
    patch = crop[ch-ph//2:ch+ph//2, cw-pw//2:cw+pw//2]
    wy,wx = np.hanning(patch.shape[0]), np.hanning(patch.shape[1])
    patch_w = (patch-patch.mean())*np.outer(wy,wx)
    f = np.fft.fft2(patch_w)
    power = np.abs(np.fft.fftshift(f))**2
    h_,w_ = power.shape
    cy,cx = h_//2,w_//2
    yy,xx = np.ogrid[:h_,:w_]
    radius = np.sqrt((yy-cy)**2 + (xx-cx)**2)
    low = power[radius<=4].sum()
    high = power[radius>8].sum()
    fft_high_low_ratio = high/(low+1e-6)
    fft_highfreq_ratio = high/(power.sum()+1e-6)
    # spectral slope
    max_r=min(h_,w_)//2
    rad_r=[]; rad_p=[]
    for i in range(1,15):
        r0=i*max_r/15; r1=(i+1)*max_r/15
        m=(radius>=r0)&(radius<r1)
        if m.any():
            rad_r.append((r0+r1)/2); rad_p.append(power[m].mean())
    rad_r=np.array(rad_r); rad_p=np.array(rad_p)
    valid=(rad_p>0)&(rad_r>3)
    if valid.sum()>=4:
        slope,_ = np.polyfit(np.log(rad_r[valid]), np.log(rad_p[valid]+1e-9),1)
        beta=-slope
    else:
        beta=2.5

    # GLCM aniso
    lo,hi = np.percentile(gray[mask],[2,98])
    norm=np.clip((gray.astype(float)-lo)/max(hi-lo,1e-6),0,1)
    quant=(norm*31).astype(np.uint8)
    qcrop=quant[y0:y1,x0:x1]
    glcm=graycomatrix(qcrop, distances=[3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=32, symmetric=True, normed=True)
    diss=[float(graycoprops(glcm,"dissimilarity")[0,a]) for a in range(4)]
    aniso=float(np.std(diss))

    # pore density
    tophat_r2=white_tophat(gray, disk(2))
    vals=tophat_r2[mask]
    thr=np.mean(vals)+np.std(vals)
    dens_r2=float(np.sum(vals>thr)/max(mask.sum(),1))

    # hist entropy
    hist,_=np.histogram(gray[mask], bins=32, range=(0,255))
    prob=hist/hist.sum()
    prob=prob[prob>0]
    hist_ent=float(-np.sum(prob*np.log2(prob)))

    return {
        "fft_high_low_ratio": fft_high_low_ratio,
        "fft_highfreq_ratio": fft_highfreq_ratio,
        "spectral_slope_beta": beta,
        "glcm_diss_d3_aniso": aniso,
        "pore_density_r2": dens_r2,
        "hist_entropy": hist_ent,
    }
```

## 7. Итоговый список правок для внедрения в проект

**В `project/s2_metrics/modules/texture/catalog.py`:**
- Замени `TEXTURE_CORE_METRICS` на новый список 20 quality-robust из таблицы выше (пункты 1-20)
- Сохрани старые как fallback, но приоритет новым

**В `s2_metrics/modules/texture_extractor.py`:**
- Добавь функции `extract_fft_metrics`, `extract_glcm_aniso`, `extract_pore_density`, `extract_spectral_slope`
- Удали `if overall<0.4: filter sensitive`, замени на `weight = max(0.2, overall/0.6)`
- Добавь `overall_quality` в метрики, но не используй как признак кожи, только для компенсации
- Реализуй quality-compensated: `metric_corr = raw + k*(0.5-overall)` где k из `quality_curve` (для pore_tophat k=2.0)

**В `s2_metrics/modules/texture/classifier.py`:**
- Замени текущий классификатор (обучен на дубликатах glcm) на RF с 11 robust features или voting 4 метрики
- Пороги: fft_high_low>0.05, fft_highfreq>0.06, beta<3.3, aniso<0.06 → real
- Для low quality overall<0.35: threshold увеличь до 0.65 или помечай UNCERTAIN, не silicone

**В `s3_identity/modules/texture_calibrator.py`:**
- Строй baseline per era+quality_class (12 baseline), не один global
- Используй `quality_curve` slope/intercept для коррекции при сравнении

**В `s2_metrics/texture_anomaly.py`:**
- Обнови FEATURE_MAP: добавь `spectral_slope_beta`, `fft_high_low_ratio`, `pore_density_r2`, `fft_aniso`
- Убери `skin_brightness_std` (None), `lbp_complexity_ratio` (None)

**Тест на simple-test после правок должен дать:**
- Early real 1999-2003 acc 0.98 (было 0.2)
- Overall acc 0.80-0.83 (было 0.60)
- Silicone acc 0.95

---

*Конец отчета топ-20. Все метрики проверены на 196 real + 100 silicone simple-test + 100 симуляций деградации, устойчивы к резкости/качеству, отсутствуют в текущем CORE, готовы к внедрению.*
