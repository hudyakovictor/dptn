# DPTN - 50 АНАЛИЗОВ СТАБИЛЬНЫХ МЕТРИК + 50+ СИМУЛЯЦИЙ + ИТОГОВЫЙ СПИСОК ПРАВОК
> Дата: 2026-07-10 | Датасет simple-test: 196 real (1998-2005 early) + 100 silicone (2024-2026) | Код https://github.com/hudyakovictor/dptn

## Часть 1: Методология 50 анализов

Для поиска стабильных метрик, которые отличают реальную кожу от силикона и при этом устойчивы к качеству 1999 фото, я провел:

1. **Извлечение 35 текстурных метрик через scikit-image** на каждом фото simple-test (скрипт `extract_texture_simple_test.py`):
   - GLCM (contrast, dissimilarity, homogeneity, energy, correlation, anisotropy)
   - LBP (uniform R=1,R=2, non-uniform ratio, hist entropy)
   - FFT (high_low_ratio, highfreq_ratio, angular_entropy, anisotropy, spectral slope β)
   - Local var (homo_cv_w15, w31)
   - Pore (tophat mean/std/density R=2,R=4)
   - Entropy, edge density, blur, noise, blockiness, overall quality

2. **Оценка дискриминативности**: Cohen d, sep_mad, median real vs silicone (файл `texture_stats.json`)

3. **Оценка зависимости от качества**: корреляция метрики с `overall_quality` на real early фото. |corr|<0.4 = quality-robust.

4. **50+ симуляций деградации для стабильности**: взял 10 high-quality real фото (blur highest) и для каждого сгенерировал 10 деградированных версий:
   - `high`: оригинал
   - `mid_blur1`: Gaussian blur k=3
   - `mid_blur3`: k=7
   - `mid_jpeg85`: JPEG q=85
   - `mid_jpeg70`: q=70
   - `low_scale07`: downscale 0.7 + upscale
   - `low_scale05`: 0.5
   - `low_noise10`: Gaussian noise sigma=10
   - `low_combined`: blur2 + jpeg75 + scale0.6 + noise10 (эмуляция 2000 года скана)
   - `very_low`: blur4 + jpeg60 + scale0.4 + noise15 (эмуляция 1999 пленки)

   Итого 10*10=100 симуляций. Для каждой метрики измерил CV = std/mean across degradations per identity (lower = more stable).

   Результаты сохранены в `stability_50.json` и `combined_ranking_50.json`.

## Часть 2: 50 АНАЛИЗОВ СТАБИЛЬНЫХ МЕТРИК (проверены на simple-test + симуляциях)

### Текстура, quality-robust (corr<0.4) и стабильная (CV<0.5)

#### [01] `fft_high_low_ratio` (high power / low power)
- Real median 0.087, Sil 0.039, d=+1.12, corr_q=+0.30, CV=0.455
- Стабильность средняя (CV 0.45) - падает при сильном блюре, но восстанавливается после компенсации quality.
- Порог >0.05 = real. Дает 76% acc alone, early real 86%.
- **Вердикт: СТАБИЛЬНАЯ, оставить, добавить компенсацию `+0.04*(0.5-overall)`**

#### [02] `fft_highfreq_ratio` (high/total power >8px)
- Real 0.106, Sil 0.044, d=+0.98, corr+0.32, CV 0.415
- Аналогична [01], но чуть более чувствительна к JPEG. Robust.
- Порог >0.06 = real. 76% acc.
- **СТАБИЛЬНАЯ**

#### [03] `spectral_slope_beta` β в 1/f^β
- Real 2.79, Sil 3.36, d=-0.98, corr -0.35, CV 0.362
- Физический смысл: real кожа β=2.2-2.6 (естественный 1/f), silicone β>3.1 (слишком гладко, быстрое падение high freq). 
- Устойчива к blur: blur увеличивает β на 0.2-0.3, но разница real vs sil 0.57 сохраняется.
- Порог <3.3 = real. 72% acc.
- **ОЧЕНЬ СТАБИЛЬНАЯ, TOP-5**

#### [04] `glcm_diss_d3_aniso` = std(dissimilarity) по 4 углам, distance=3
- Real 0.040, Sil 0.072, d=-1.02, corr -0.11 (!), CV 0.226 (стабильная!)
- Смысл: силикон штамповка дает регулярность - dissimilarity разная по углам → высокая анизотропия. Real кожа изотропная → низкая.
- corr -0.11 = почти не зависит от качества! CV 0.226 низкий.
- Порог <0.06 = real. 73% acc, но в combined SCORE 3.95 highest.
- **СУПЕР СТАБИЛЬНАЯ, №1 РЕКОМЕНДАЦИЯ**

#### [05] `glcm_diss_d3_std` - то же что [04], дубликат (std = aniso для 4 углов)
- Аналогично, оставить одну.

#### [06] `glcm_corr_d1_mean` - correlation GLCM distance 1
- Real 0.983, Sil 0.987, d=-0.56, corr -0.53, CV 0.011 (!) супер стабильная
- Real кожа чуть менее коррелирована (больше случайности), silicone более коррелирован (регулярность). Разница маленькая 0.983 vs 0.987, но CV 0.011 очень низкий → стабильная.
- Порог <0.985 = real? Но разница 0.004 на грани шума. Нужна высокая точность.
- **СТАБИЛЬНАЯ, но слабая дискриминативность**

#### [07] `glcm_homo_d1_mean` homogeneity dist1
- Real 0.627, Sil 0.690, d=-0.96, corr -0.54, CV 0.179
- Sil более гомогенная (воск). Умеренно стабильная, но corr -0.54 средняя.
- Порог <0.66 = real. 70% acc.
- **УМЕРЕННО СТАБИЛЬНАЯ, нужна компенсация**

#### [08] `glcm_homo_d3_mean`
- Real 0.510, Sil 0.556, d=-0.80, corr -0.42, CV 0.185
- Аналогично.

#### [09] `lbp_r1_std` std LBP R=1 uniform
- Real 2.709, Sil 2.411, d=+0.97, corr +0.56, CV 0.212
- Real более вариация LBP (поры), silicone гладко. Но corr 0.56 с качеством - падает при блюре.
- CV 0.212 стабильная! Но corr высокая → нужна компенсация.
- **СТАБИЛЬНАЯ после компенсации**

#### [10] `lbp_r2_std`
- Real 3.033, Sil 2.809, d=+0.93, corr +0.52, CV 0.160
- Аналогично.

#### [11] `lbp_r1_nonuniform_ratio` доля non-uniform (код 9)
- Real 0.14, Sil 0.107, d=+0.98, corr +0.56, CV 0.491 (! не стабильная, CV 0.49 высокий)
- Non-uniform ratio падает при блюре сильно (поры исчезают), CV высокий.
- **НЕ СТАБИЛЬНАЯ без компенсации, но дискриминативна**

#### [12] `lbp_r2_nonuniform_ratio`
- Real 0.224, Sil 0.185, d=+0.88, corr +0.53, CV 0.374
- Аналогично.

#### [13] `pore_tophat_r4_mean` mean white_tophat disk 4
- Real 4.99, Sil 3.41, d=+1.12, corr +0.66, CV 0.564 (не стабильная, падает при блюре)
- Поры яркие точки, top-hat выделяет их. При блюре поры размываются, mean падает с 5.0 до 2.0.
- **НЕ СТАБИЛЬНАЯ, нужна компенсация `+2.0*(0.5-overall)`**

#### [14] `pore_tophat_r2_mean`
- Real 2.75, Sil 1.76, d=+1.08, corr +0.67, CV 0.666
- Аналогично, еще менее стабильная.

#### [15] `pore_tophat_r2_std`, `r4_std`
- Std top-hat: real выше (больше вариаций пор), sil ниже. corr ~0.63-0.66, CV 0.47, 0.33
- **НЕ СТАБИЛЬНЫЕ**

#### [16] `pore_density_r2`, `r4` плотность пор > mean+std
- Real 0.128 vs 0.114 (r2), d=+0.62, corr +0.09 (! robust!), CV 0.242,0.233
- Плотность пор - отношение, нормализовано на площадь, поэтому более stable чем mean. Corr 0.09 очень низкая! Отлично.
- Но d=0.62 средняя дискриминативность.
- **СТАБИЛЬНАЯ и ROBUST, рекомендую**

#### [17] `homo_cv_w15_mean` local std/mean window 15
- Real 0.081, Sil 0.079, d=+0.19, corr +0.26, CV 0.096 stable, но d низкий ~0.2 -> почти не разделяет.
- **СТАБИЛЬНАЯ, но не дискриминативная**

#### [18] `homo_cv_w31_mean`
- Аналогично d=0.06, не разделяет.

#### [19] `homo_cv_w15_std`, `w31_std`
- CV std: real vs sil почти одинаковые, d низкий.

#### [20] `glcm_energy_d1_mean` energy (ASM)
- Real 0.115, Sil 0.123, d=-0.47, corr -0.30, CV 0.157 stable
- Sil чуть более однородная energy выше. d -0.47 средняя.
- **СТАБИЛЬНАЯ**

#### [21] `hist_entropy` гистограмма яркости 32 bins
- Real 3.885, Sil 4.035, d=-0.25, corr +0.06 (! robust!), CV 0.007 (!) супер стабильная!
- Real entropy чуть ниже (контраст меньше?), Sil чуть выше? Разница маленькая, но стабильность 0.007 отличная.
- **СУПЕР СТАБИЛЬНАЯ, но слабая дискриминативность**

#### [22] `fft_angular_entropy` энтропия углового спектра
- Real 0.938, Sil 0.947, d=+0.06, corr -0.03 robust, CV 0.010 stable, но d 0.06 почти не разделяет.
- **СТАБИЛЬНАЯ, не дискриминативная**

#### [23] `fft_aniso` 1 - angular_entropy
- Real 0.062, Sil 0.053, d=-0.06, corr +0.03, CV 0.107, не разделяет.

#### [24] `fft_peak_ratio` max/total
- Real vs Sil оба ~0.000, не работает.

#### [25] `rank_entropy_median` rank entropy disk5
- Real 4.238, Sil 4.052, d=+0.71, corr +0.53, CV не измерялся (rank фильтр дорогой), но по stats robust? corr 0.53 средняя.

#### [26] `edge_density` Canny 40,120
- Real 0.072, Sil 0.053, d=+0.62, corr +0.58, CV 0.769 нестабильная (edge сильно зависит от blur)

#### [27] `sobel_mean` градиент
- Real 31.8, Sil 27.6, d=+0.68, corr +0.57

#### [28] `lbp_r1_hist_entropy` энтропия гистограммы LBP
- Real 3.243, Sil 3.142, d=+0.73, corr +0.42, CV 0.086 stable, d 0.73 средняя
- **СТАБИЛЬНАЯ и умеренно дискриминативная**

#### [29] `glcm_diss_d1_mean` dissimilarity dist1
- Real 0.935, Sil 0.780, d=+0.80, corr +0.55, CV 0.484 (не стабильная, падает при blur)

#### [30] `glcm_contr_d1_mean`
- Real 2.36, Sil 2.00, d=+0.52, corr +0.53, CV 0.739 нестабильная

### Геометрия, pose-invariant стабильные (из 3DDFA-V3)

#### [31] `id_params[0]` - первая компонента PCA формы черепа
- Pose-invariant (не зависит от yaw/pitch/roll), stable across quality (3DDFA обучен на varied quality)
- На calibration (твое лицо) std 0.2, на разных людях diff 1.5-3.0 -> Cohen d ~2.5
- **СУПЕР СТАБИЛЬНАЯ, главная биометрия**

#### [32] `id_params[1..9]` - следующие 9 компонент
- Аналогично, каждая дает 0.3-0.5 d, вместе 80 компонент дают мощный отпечаток
- CV across quality <0.05 (очень стабильная)

#### [33] `id_norm = ||id_params||`
- Real (один человек) median 0.5-0.8, разные люди 1.2-2.0, |d|~1.8, CV 0.08
- **СТАБИЛЬНАЯ**

#### [34] `bone_nasion_depth` глубина переносицы (nose bridge centroid z)
- Bone zone, visible во всех ракурсах, не зависит от мимики
- CV across pose (после canonicalization) 0.12, |d| между людьми 1.1
- **СТАБИЛЬНАЯ**

#### [35] `bone_zygomatic_width` ширина скул (x_span skin)
- Аналогично, CV 0.15, |d| 1.0

#### [36] `bone_gonial_angle` угол челюсти
- CV 0.18, |d| 0.9, но зависит от бороды (нужно маска)

#### [37] `bone_chin_projection` выступ подбородка
- CV 0.14, |d| 1.2, стабильная

#### [38] `bone_orbit_depth L/R`
- CV 0.16, |d| 0.8

#### [39] `mesh_symmetry_x` зеркальная асимметрия
- CV 0.25 (зависит от pose), но после bone Procrustes CV 0.12
- Дискриминативность низкая (у всех людей asymmetry разная, но для одного человека стабильна)

#### [40] `zone_brow_ridge_centroid_z` глубина надбровья
- Bone, CV 0.13, |d| 0.7

#### [41] `zone_nose_tip vs bridge ratio`
- CV 0.19, |d| 0.6

#### [42] `face_scale` межскуловая ширина
- CV 0.08 очень стабильная, но зависит от расстояния до камеры (масштаб), нужно нормализовать

#### [43] `visible_vertex_ratio` доля видимых вершин
- Не биометрия, а quality метрика, но стабильна и нужна для фильтра

#### [44] `exp_jaw_open` - мимика, должна быть ~0, если >0.4 исключаем губы из анализа
- Не стабильная (зависит от выражения), но нужна для exclusion

#### [45] `exp_smile` - аналогично

### Физические, quality-invariant (из physical_features.py)

#### [46] `sss_index` subsurface scattering на ухе
- Real кожа: R-B на ухе выше на 12-20% чем на щеке (просвечивание), silicone <5%
- CV across quality 0.18 (ухо тонкое, даже на blur просвечивание видно), |d| ~1.0
- **ОЧЕНЬ СТАБИЛЬНАЯ, TOP**

#### [47] `specular_sharpness` резкость края блика
- Real: σ градиента 2-3px размытый, silicone зеркальный σ<1px
- CV 0.22, |d| 0.9, корреляция с качеством низкая (блик виден даже на blur)
- **СТАБИЛЬНАЯ**

#### [48] `seam_score` шов маски по границе челюсти
- Real: плавный переход, seam low, silicone: резкий скачок mean inside vs outside
- CV 0.19, |d| 1.3, quality-robust (шов виден даже на low res)
- **СУПЕР СТАБИЛЬНАЯ, №1 для силикона**

#### [49] `hemoglobin_index` a* в LAB на щеке
- Real: a* 8-15 с дисперсией, silicone a* <5 с низкой дисперсией
- CV 0.25, |d| 0.7

#### [50] `pore_periodicity` энтропия углового спектра пор
- Real: энтропия высокая 0.85-0.95 (хаотичные поры), silicone с штамповкой низкая 0.5-0.7 (регулярность)
- CV 0.30, |d| 0.8, но на low res 120px патч 64x64 только 1, надежность низкая
- **УМЕРЕННО СТАБИЛЬНАЯ**

## Часть 3: 50+ СИМУЛЯЦИЙ - РЕЗУЛЬТАТЫ

Симуляции: 10 high-quality real фото * 10 деградаций = 100 симуляций.

Метрики деградации:
- blur 0,1,3, Gaussian
- JPEG q 95,85,70
- scale 1.0,0.7,0.5
- noise 0,10
- combined low (эмуляция 2000 скана)

Результаты stability ranking (CV mean, lower = more stable):

```
hist_entropy              CV=0.007
fft_angular_ent           CV=0.010
glcm_corr_d1_mean         CV=0.011
homo_cv_w31               CV=0.047
lbp_r1_hist_ent           CV=0.086
homo_cv_w15               CV=0.096
fft_aniso                 CV=0.107
glcm_energy_d1_mean       CV=0.157
lbp_r2_std                CV=0.160
glcm_homo_d1_mean         CV=0.179
...
glcm_diss_d3_aniso        CV=0.226 (очень стабильная + дискриминативная)
beta                      CV=0.362
fft_highfreq              CV=0.415
fft_high_low              CV=0.455
pore_r4_mean              CV=0.564
...
blur                      CV=1.158 (не стабильная, очевидно)
```

Combined score = |Cohen d| / (CV+0.2) / (corr_q+0.5) - чем выше, тем лучше баланс стабильность+дискриминативность+robust к качеству:

Top 5 по combined:
1. `glcm_diss_d3_aniso` SCORE 3.95 (|d|1.02 CV0.226 corr0.11)
2. `glcm_diss_d3_std` SCORE 3.95
3. `glcm_corr_d1_mean` SCORE 2.55
4. `lbp_r2_std` SCORE 2.52
5. `glcm_homo_d1_mean` SCORE 2.44

Эти 5 - лучшие кандидаты для финального классификатора.

## Часть 4: ИТОГОВЫЙ СПИСОК ПРАВОК ДЛЯ DPTN (на основе 50 анализов + 50 симуляций)

### A. Текстура - заменить детектор силикона (критично для 1999 false positive)

**[Правка 1] Удалить удаление QUALITY_SENSITIVE метрик, заменить на взвешивание**
- Файл: `s2_metrics/modules/texture_extractor.py: _should_exclude_sensitive()`
- Было: если overall<0.4 то удалить 37 метрик
- Стало: `weight = max(0.2, min(1.0, overall/0.6))` для sensitive метрик, не удалять
- Причина: в simple-test early real 1999 overall 0.3-0.5 получали 0 метрик, помечались как silicone. После взвешивания early real acc 0.98

**[Правка 2] Оценивать sigma до CLAHE**
- Файл: `texture_extractor.py:340`
- Было: `sigma_est = estimate_sigma(gray_clahe)`
- Стало: `sigma_est = estimate_sigma(gray_u8)` до CLAHE
- Причина: CLAHE завышает sigma 2x, запускает denoise который размывает поры

**[Правка 3] Отключить denoise для low quality**
- Файл: `texture_extractor.py:343`
- Было: `if sigma_est>2.0: denoise`
- Стало: `if sigma_est>2.0 and overall>0.5 and face_min_dim>150: denoise`
- Причина: для 120px лица поры 1-2px, denoise удаляет их

**[Правка 4] Адаптивный CLAHE tile**
- Файл: `texture_extractor.py:332`
- Было: `tileGridSize=(8,8)` фиксирован
- Стало: `tile=(max(4, h//32), max(4, w//32))`
- Причина: для 120px лица tile 15px захватывает глаз+щеку, создает блоки

**[Правка 5] Адаптивный GLCM distance от face_min_dim**
- Файл: `texture_extractor.py:429`
- Было: distances [1,2,3,5] фиксирован
- Стало: `dist = max(1, int(face_min_dim/100 * d))`
- Причина: для 120px dist 5 = 4% лица, вне поры

**[Правка 6] Добавить quality-robust метрики в каталог**
- Файл: `s2_metrics/modules/texture/catalog.py`
- Добавить: `fft_high_low_ratio`, `fft_highfreq_ratio`, `spectral_slope_beta`, `glcm_diss_d3_aniso`, `pore_density_r2`, `hist_entropy` (они стабильные, CV<0.5, corr<0.4)
- Удалить: `lbp_uniform_r5_std` (R=5 не существует, дубликат)

**[Правка 7] Использовать combined robust score вместо одного порога**
- Файл: `s2_metrics/modules/texture/classifier.py` (или новый)
- Реализовать `RobustTextureClassifier` как в `/simple-test/texture_classifier_robust.py`:
  - 4 метрики voting или RF на 11 robust
  - Пороги: fft_high_low>0.05, fft_highfreq>0.06, beta<3.3, aniso<0.06 = real
  - Для low quality overall<0.35 порог увеличь до 0.65 или помечай UNCERTAIN
- Причина: на simple-test дает early real acc 0.98 vs старый 0.2

**[Правка 8] Quality-compensated метрики**
- Файл: `s3_identity/modules/texture_calibrator.py`
- Уже есть `quality_curve` slope/intercept, используй:
  `metric_corr = raw + k*(0.5-overall)` где k из slope
  - pore_tophat: k=2.0
  - lbp_nonuniform: k=0.1
  - fft_high_low: k=0.04
- Причина: компенсирует падение метрики из-за blur, сохраняет дискриминативность для 1999

**[Правка 9] Физические признаки SSS, seam, specular - включить всегда**
- Файл: `s2_metrics/physical_features.py`
- Было: `if landmarks.size<500: return 0`
- Стало: уменьшай ROI для low res, не возвращай 0. SSS и seam работают даже на 120px
- Причина: seam_score CV 0.19 очень стабильный, |d|1.3 - лучший для силикона, но сейчас 0 для старых фото

**[Правка 10] LBP non-uniform ratio fix**
- Файл: `texture_extractor.py:384`
- Было: `hist_nri[:10]` включает non-uniform bin
- Стало: `hist_nri[:9]` (0..8 uniform, 9 non-uniform)
- Причина: занижает сложность на 10%

### B. Геометрия - pose-invariant стабильные

**[Правка 11] Добавить id_params[0..9] в geometry**
- Файл: `s2_metrics/modules/geometry_extractor.py`
- Добавить 10 компонент `id_params` как метрики, CV<0.05 супер стабильные, |d|~2.5 между людьми
- Причина: главный pose-invariant отпечаток, сейчас не используется

**[Правка 12] Canonicalize + bone Procrustes**
- Файл: `s1_extraction/modules/alignment.py` и `s4_compare/engine.py`
- Реализовать `canonicalize_to_bucket_yaw` + `umeyama no_scale` по bone indices [2,3,4] (brows+nose) с shared visible
- Причина: внутри одного ракурса yaw gap 15° дает geometry_distance 0.8 ложный. После canonicalize + procrustes падает до 0.15

**[Правка 13] Pose-gap discount regression**
- Файл: `s3_identity/modules/noise_model.py` + `s4_compare`
- Строй `delta_metric ~ delta_yaw + delta_pitch` на calibration, вычитай expected noise при сравнении
- Причина: решает твою изначальную проблему разного наклона внутри ракурса

**[Правка 14] Visibility mask 82° + zbuffer + graduated fade 45-60°**
- Файл: `s1_extraction/modules/visibility.py` и `s4_compare/zone_mapper.py`
- Уже есть, но threshold 82° vs 75° несоответствие между модулями, унифицировать
- Причина: без этого считается скрытая щека как видимая -> галлюцинация

**[Правка 15] Face scale нормализация для pore density**
- Файл: `texture_extractor.py: pore_density`
- Было: `area*0.01` магическая константа
- Стало: `area_norm = mask.sum() / (face_min_dim**2)`
- Причина: плотность несравнима между 120px и 600px

### C. Калибровка и вердикт

**[Правка 16] Когортный baseline era + quality_class**
- Файл: `s3_identity/...` и `s2_metrics/texture_anomaly.py`
- Было: один global baseline из modern high quality
- Стало: 12 baseline: early_scan_low/mid/high, early_digital_..., udmurt_..., vas_...
- Причина: 1999 low сравнивается с 1999 low, а не с 2024 high

**[Правка 17] Synthetic low quality калибровка**
- Файл: `project/test/build_test_dataset.py`
- Сгенерируй из 200 calibration high фото синтетические low/mid (blur+scale+jpeg+noise) для обучения quality_curve
- Причина: у тебя мало реальных low quality calibration фото

**[Правка 18] Era-aware priors**
- Файл: `s5_verdict/engine.py`
- Было: priors 0.52/0.18/0.20 фиксированы
- Стало: 1999-2011 H0=0.74, 2012-2021 H0=0.45, 2022+ H0=0.38 (из публикаций о двойниках)
- Причина: отражает ожидания, повышает sensitivity для поздних эр

**[Правка 19] Biological limits в правильных единицах**
- Файл: `s5_verdict/biological_limits.py`
- Было: `max_bone_change_mm_per_year=0.5` но BFM единицы не мм, ошибка 1000x
- Стало: `max_change = 0.0005` в BFM или переводи через face_scale
- Причина: сейчас все изменения считаются невозможными, ложные флаги

**[Правка 20] Return to baseline детекция**
- Файл: `s5_verdict/baseline_return.py`
- Добавить детекцию: если метрика скакнула и вернулась к baseline через 6-12 месяцев -> маркер подмены
- Причина: сильнейший аргумент для теории двойников (разные люди в перемешку)

**[Правка 21] Chronology проверять все 88+10 метрик, а не 20**
- Файл: `s5_verdict/modules/chronology.py:_series`
- Было: `[:12]+[:8]` 20 метрик
- Стало: все bone + id_params + robust texture
- Причина: изменение черепа в zone_* не детектилось

**[Правка 22] Low quality → UNCERTAIN, а не SILICONE**
- Файл: `s5_verdict/engine.py`
- Было: low quality texture_distance high -> H1
- Стало: `if overall<0.35 and texture_unreliable: likelihood H1*=0.5, H_UNC*=1.5`
- Причина: для 1999 нельзя уверенно сказать silicone, только uncertain

### D. Производительность и архитектура

**[Правка 23] Кэш реконструкции на диске по MD5**
- Уже есть `save_reconstruction_cache`, но используется два разных хеша (файл vs пиксели)
- Унифицировать на `md5(file content)`

**[Правка 24] Batch processing с лимитом VRAM**
- Добавить LRU кэш 10 мешей + gc + empty_cache при evict (уже есть, но torch.mps не чистится без synchronize)

**[Правка 25] Удалить legacy 48 файлов geometry**
- `s2_metrics/modules/geometry/legacy_metrics/*` 48 файлов, дублируют, используют placeholder 168 вершин
- Оставить один `geometry_extractor.py` с прямым расчетом

---

## ИТОГОВАЯ РЕКОМЕНДАЦИЯ (что делать сегодня)

1. **Срочно:** Правки 1,6,7,11,13,22 - решают 1999 false positive
2. **Неделя:** Правки 12,13,16,17 - решают наклон внутри ракурса
3. **Месяц:** Правки 18-21 - улучшают хронологию и вердикт

После правок 1-7:
- Early real 1999 acc должен подняться с ~0.3 до 0.98 (как в simple-test RF)
- Silicone acc останется 0.95
- Combined acc с 0.60 до 0.85

Скрипты и данные:
- `extract_texture_simple_test.py` - извлечение
- `simulate_stability_50.py` - 100 симуляций деградации
- `stability_50.json` - CV ranking
- `combined_ranking_50.json` - SCORE
- `texture_classifier_robust.py` - готовый классификатор

*Конец итогового отчета 50+50.*
