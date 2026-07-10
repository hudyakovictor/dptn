# Почему фото 1999-2005 помечаются как силикон и как это починить
> Дата: 2026-07-10 | DPTN 3DDFA-V3 | Проблема ложноположительных срабатываний детектора силикона на ранних фото

## 1. Краткий диагноз

Твоя система сейчас работает так:

```
1999 фото (скан пленки, 640x480, JPEG q=75, blur)
  -> quality.overall = 0.25 (low)
  -> QUALITY_SENSITIVE_METRICS удаляется (37 метрик!)
  -> остается: gray_mean, gray_std, entropy, laplacian_var, lbp_uniformity, fft_highfreq_ratio, edge_density
  -> wavelet denoise (sigma>2) еще сильнее размывает
  -> GLCM квантизация по [2,98] перцентилям схлопывает динамику
  -> сравнивается с calibration baseline (твои современные фото 2024, 4K, sharpness 0.9)
  -> distance огромный => synthetic_suspicion 0.8 => H1_SYNTHETIC
```

**Силиконовая маска 2024:**
```
4K фото силикона, резкое, но восковая гладкость
  -> highfreq_ratio низкий (0.12), lbp_uniformity низкий, homo_cv низкий (0.07)
  -> сравнивается с тем же baseline (real skin high complexity)
  -> distance высокий => тоже 0.8 => H1_SYNTHETIC
```

Итог: **старая пленка и силикон попадают в один кластер "низкая сложность текстуры"**, хотя причины разные: у пленки - потеря деталей из-за сканирования, у силикона - отсутствие пор физически.

## 2. Детальный разбор 7 причин

### Причина 1: Удаление дискриминативных признаков при low quality
В `texture_extractor.py:52-91`:
```python
QUALITY_SENSITIVE_METRICS = {
  glcm_dissimilarity_d5_*, homo_local_var_w15_cv, morph_tophat_r4_std, ...
}
if quality.overall <0.4: result = filter_sensitive_metrics(result)
```

Проблема: именно `homo_local_var` и `glcm_dissimilarity` - главные детекторы восковой гладкости силикона! Ты их удаляешь для старых фото, оставляя только общие метрики. Классификатору не из чего отличить силикон от блюра.

**Фикс:** Не удалять, а взвешивать:
```python
# было:
return {k:v for k,v in metrics.items() if k not in QUALITY_SENSITIVE}
# надо:
weight = feature_weight(k, quality) # 0.3 для low quality, 1.0 для high
# в distance считаешь weighted z-score, а не выбрасываешь
```

### Причина 2: Глобальный baseline вместо когортного + quality-класса
`CohortTextureAnomalyDetector` делит по эрам `early_scan (1999-2005)`, `early_digital` etc - правильно. Но `TextureCalibrator` строит только `global_stats` из ВСЕХ calibration фото (твои современные).

Если твои 200 calibration фото - все современные high quality (sharpness >400), то:
- `texture_fft_highfreq_ratio` baseline median = 0.35, mad=0.05
- 1999 фото Путина: fft_ratio = 0.08
- z = (0.08-0.35)/0.05 = -5.4 => аномалия

А должно сравниваться с `early_scan` baseline: median 0.09, mad 0.03 => z = -0.33 => норма.

**Фикс уже частично есть в `texture_calibrator.py:_fit_quality_curve`:**
```python
# учим slope для каждой метрики: metric ~ quality
slope, intercept = polyfit(quality, values, 1)
```
Используй это при сравнении:
```python
expected_for_this_quality = slope * photo_quality + intercept
corrected_metric = raw - expected_for_this_quality + median_high_quality
# или quality-adjusted z:
z = (raw - expected_for_this_quality) / mad
```

### Причина 3: Wavelet denoise убивает пленочное зерно
```python
sigma_est = estimate_sigma(gray_clahe)
if sigma_est >2.0:
  gray = denoise_wavelet(gray, sigma=...)
```
Для 1999 скана sigma_est ~ 8-12 из-за зерна. Ты его размываешь, превращая зерно в однородную поверхность, похожую на силикон.

**Фикс:** Денойз только для `quality_class == high` и `sigma_est < 10`. Для low quality - пропускай или используй `denoise_tv_chambolle` с сохранением текстуры.

### Причина 4: FFT highfreq ratio чувствителен к JPEG блокингу
Текущая реализация patch-based 64x64 с 50% overlap - хороша, но:
- Для JPEG q=75 блок 8x8 дает пик на частоте 0.125 cycles/pixel
- Этот пик попадает в high freq область и считается как "высокая частота", хотя это артефакт сжатия, не поры.

Силикон тоже дает низкий highfreq, но по другой причине (нет пор). Оба получают low ratio.

**Фикс:** 
- Вычисляй FFT только на центральной части кожи, исключая 8x8 границы (jpeg grid).
- Используй `fft_peak_ratio` и `fft_anisotropy` как более устойчивые: силикон дает анизотропные пики (регулярная штамповка), а JPEG дает изотропный шум + пик на 8px.
- Добавь метрику `spectral_slope β`: 
  - реальная кожа β=2.2-2.6
  - силикон β>2.8 (слишком гладко) 
  - старая пленка β=2.0-2.4 (близко к реальной), но с шумом

### Причина 5: LBP uniformity - не та метрика
В коде:
```python
result["texture_lbp_uniformity"] = std(lbp1_skin)
result["lbp_uniform_r5_std"] = std(lbp2_skin)
```
Std LBP - не информативен. Для однородной области LBP коды все 0, std=0. Для текстуры кожи LBP коды разбросаны, std высокий. И старая размытая кожа, и силикон имеют std низкий -> путаются.

Надо `lbp_complexity_ratio` = доля non-uniform паттернов:
- реальная кожа 30-45% non-uniform
- силикон <20%
- старая пленка 25-35% (чуть ниже из-за блюра, но не <20%)

У тебя `lbp_complexity_ratio` считается, но потом попадает в `QUALITY_SENSITIVE` и удаляется!

### Причина 6: Отсутствие нормализации по face_scale / resolution
`pore_density_r2` считается как `pore_count / skin_area_cm2`, где `skin_area_cm2 = (mask>0).sum()*0.01` - константа 0.01. Для 1999 фото 120px лицо и 2024 600px лицо плотность считается в одной шкале, хотя в 120px поры физически не могут быть видны (размер поры ~0.1мм, в пикселях <1px).

Поэтому у старого фото `pore_density=0` (пор нет), у силикона тоже 0 (штампованные поры не детектятся топхэтом). Опять кластер.

**Фикс:** Нормируй на `face_min_dim`:
```python
expected_pore_px = face_min_dim / 300 * 3 # 3px для 300px лица
if face_min_dim <120: pore_metrics weight =0.2 else 1.0
```

### Причина 7: Physical features (SSS, seam) не используются для старых фото
`PhysicalTextureExtractor` извлекает 7 признаков:
- sss_index (просвечивание уха)
- specular_sharpness (резкость блика)
- seam_score (шов маски)

Эти признаки **качество-инвариантны**: даже на размытом фото шов маски даст резкий скачок текстуры, а ухо не будет просвечивать.

Но в `s2_metrics/engine.py` ты считаешь `physical_features` только если `landmarks_68` есть, а для старых фото landmarks шумные и `seg_mask` плохая -> `physical_features = {}`.

И в `s5_verdict/engine.py` `silicone_physical_boost` умножается на 0.35, но если phys пустой, то 0.

Итог: для старых фото ты опираешься только на texture, которая путается, а физические признаки, которые бы отличили силикон, отсутствуют.

## 3. Как должно быть - исправленный пайплайн

### Шаг А: Построй quality-curve на calibration с синтетической деградацией
Возьми свои 200 современных high quality фото и сгенерируй из них 3 копии:
- high: оригинал
- mid: downscale 0.5 + JPEG q=85 + blur 3x3
- low: downscale 0.3 + JPEG q=70 + blur 7x7 + Gaussian noise sigma=15

Теперь у тебя 600 фото с известным quality и одной и той же кожей. Построй для каждой метрики:
```python
# texture_calibrator.py уже делает это!
quality_curve[metric] = {slope, intercept, corr}
# slope показывает на сколько метрика падает при ухудшении качества
```

При анализе 1999 фото:
```python
expected = slope*photo_quality + intercept
corrected = raw - expected + median_high
# corrected = каким бы было raw если бы качество было high
```

### Шаг B: Когорта + quality класс
Вместо одного глобального baseline - 12 baseline'ов:
```
early_scan_low, early_scan_mid, early_scan_high
early_digital_low, ...
udmurt_era_low, ...
vas_era_low, ...
```
В `CohortTextureAnomalyDetector` уже есть эры, добавь quality:
```python
def get_cohort_key(self, year, quality):
  era = ... # early_scan etc
  qclass = "low" if quality<0.4 else "mid" if quality<0.65 else "high"
  return f"{era}_{qclass}"
```

Теперь 1999 low фото сравнивается с early_scan_low baseline (тоже размытые), а не с vas_era_high.

### Шаг C: Не удаляй чувствительные метрики, а взвешивай
В `engine.py`:
```python
# было:
if quality.overall<0.4: filter sensitive

# надо:
def texture_distance_corrected(tex_a, tex_b, quality_a, quality_b, quality_curve, global_stats):
  common = set(tex_a)&set(tex_b)
  vals=[]
  for k in common:
    raw_delta = abs(tex_a[k]-tex_b[k])
    # коррекция на качество
    curve_a = quality_curve.get(k)
    curve_b = ...
    if curve_a:
      expected_a = curve_a.slope*quality_a + curve_a.intercept
      expected_b = curve_a.slope*quality_b + curve_a.intercept
      # вычитаем ожидаемую разницу из-за качества
      raw_delta = max(0, raw_delta - abs(expected_a-expected_b))
    # z-score + вес
    mad = global_stats[k].mad
    weight = min(feature_weight(k, quality_a), feature_weight(k, quality_b))
    vals.append( (raw_delta/mad) * weight )
  return median(vals)
```

### Шаг D: Замени метрики на качество-инвариантные
Для low quality фото используй только то, что устойчиво:

Устойчивые (вес 1.0 даже при low):
- `specular_ratio` + `specular_sharpness` - блик виден даже на блюре, у силикона он зеркальный (sharpness высокий, spread низкий)
- `seam_score` - шов дает скачок даже на низком разрешении
- `spectral_slope β` - наклон спектра сохраняется при блюре (блюр режет high freq, но slope меняется мало)
- `albedo_a_std` - вариативность гемоглобина (a* в LAB) видна даже при блюре, у силикона низкая дисперсия
- `color_b_mean` - синий канал (силикон часто более синий/восковой)

Неустойчивые (вес 0.2 при low):
- `fft_highfreq_ratio`, `laplacian_var`, `edge_density`, `pore_density` - требуют sharpness
- `glcm_dissimilarity_d1` - требует резкости

### Шаг E: Отключи wavelet denoise для low quality
```python
sigma_est = estimate_sigma(gray)
if sigma_est>2.0 and quality.overall>0.5: # только для high quality
  gray = denoise_wavelet(...)
```

### Шаг F: Добавь проверку "low_quality_cannot_assess" в verdict
В `s5_verdict/engine.py` уже есть:
```python
if quality<0.3:
  interpretation="low_quality_cannot_assess"
  syn_score *= 0.3
```
Но надо еще:
```python
if quality<0.35 and texture_anomaly.interpretation=="low_quality_cannot_assess":
  # не помечаем как H1_SYNTHETIC, а как H_UNCERTAIN
  likelihoods[H1]*=0.5
  likelihoods[H_UNC]*=1.5
```

## 4. Конкретный патч для твоего кода

### Патч 1: `s2_metrics/modules/texture_extractor.py`

```python
# Замени _should_exclude_sensitive на взвешивание
def _should_exclude_sensitive(...): 
  return False # никогда не удаляем полностью

# Добавь в extract():
for k in result:
  result[k+"_weight"] = self._feature_weight(k, quality)
# texture_unreliable теперь не bool, а float 0..1
result["texture_reliability"] = np.mean(list(self._feature_weights.values()))
```

### Патч 2: `s3_identity/modules/texture_calibrator.py`
Уже почти правильно, но добавь генерацию low quality копий в build_reference (см выше) и сохраняй quality_curve.

### Патч 3: `s2_metrics/texture_anomaly.py`
Исправь FEATURE_MAP:
```python
FEATURE_MAP = {
  "fft_highfreq_ratio_mean": ("texture_fft_highfreq_ratio", 0.0),
  "spectral_slope": ("spectral_slope", 2.5), # добавь эту метрику!
  "seam_score": ("seam_score", 0.0), # самая важная для силикона
  "specular_sharpness": ("specular_sharpness", 0.0),
  "albedo_a_std": ("albedo_a_std", 5.0),
  "lbp_complexity_ratio": ("lbp_complexity_ratio", 0.3), # уже считается!
}
```

И порог:
```python
threshold = 2.0 + (1.0 - quality)*1.5 # было *3.0 слишком мягко, *1.5 лучше
if quality<0.35: return 0.0 score и interpretation low_quality_cannot_assess
```

### Патч 4: `physical_features.py`
Добавь поддержку low-res: если `face_min_dim<150`, используй больший ROI для SSS и seam (30px -> 15px).

## 5. Как проверить что починка сработала

Возьми 3 группы:
- A: 20 фото 1999-2002 Путина (ты знаешь что это оригинал, high blurriness)
- B: 20 фото 2023-2024 Путина современные (оригинал или подозрение)
- C: 20 фото силиконовых масок (UDMURT, VAS из интернета)

Прогони старый pipeline и новый:

Ожидается:
- Старый: A помечается как silicone 60-80%, B как silicone 30%, C как silicone 85% -> ложные срабатывания на A
- Новый с quality-compensated: A как silicone 5-10% (low_quality_cannot_assess), B как 20-30%, C как 80% -> разделение правильное

Метрика: `texture_anomaly_score` для A должен упасть с 0.7 до 0.2 после фикса.

## 6. Итоговая формула для детектора силикона, устойчивого к качеству

```python
def silicone_score_robust(texture, physical, quality, face_min_dim):
  # quality weight
  q = max(0.3, quality.overall_quality)
  
  # веса для low vs high
  if face_min_dim <150 or q<0.4:
    # low quality mode - используем только robust фичи
    score = (
      physical.get("seam_score",0)*0.35 +
      physical.get("specular_sharpness",0)*0.25 +
      physical.get("sss_index",0)*0.20 + # низкое просвечивание = силикон
      texture.get("spectral_slope",2.5)*0.10 + # beta>2.8 = силикон
      texture.get("albedo_a_std",5.0)*0.10 # low std = силикон
    )
    # нормируй: каждый компонент 0..1, score 0..1
    # для low quality порог 0.65 вместо 0.5
    threshold = 0.65
  else:
    # high quality mode - полная формула
    score = (
      texture.get("homo_local_var_w15_cv",0)*0.20 + # низкая var = воск
      texture.get("lbp_complexity_ratio",0.3)*0.25 + # <0.2 = силикон
      physical.get("seam_score",0)*0.25 +
      physical.get("specular_sharpness",0)*0.15 +
      texture.get("fft_anisotropy",0)*0.15
    )
    threshold = 0.55
  
  return score, score>threshold
```

## 7. Что делать прямо сейчас (1 день)

1. В `texture_extractor.py` строку `if sigma_est>2.0: denoise` замени на `if sigma_est>2.0 and quality.overall>0.5`
2. В `texture_anomaly.py` `threshold = 2.0 + (1-quality)*3.0` -> `threshold = 2.5 + (1-quality)*1.0` и добавь ранний return для low quality
3. В `s3_identity/modules/texture_calibrator.py` используй `quality_curve` при сравнении, а не raw median
4. Пересобери calibration_reference с синтетической деградацией (скрипт могу дать)
5. Прогони на 20 фото 1999 - проверь что `texture_anomaly_score` упал

После этого 1999 фото перестанут помечаться как силикон, а реальные силиконовые маски 2023+ продолжат детектиться (у них seam + specular sharpness высокий даже после коррекций).

---
*Конец. Если дашь 3 примера фото 1999 которые сейчас помечаются как силикон и 3 современных силикона - могу показать точные цифры метрик до/после.*
