# DPTN - 50 ОШИБОК ТЕКСТУРНОГО АНАЛИЗА КОЖИ ЧЕРЕЗ scikit-image
> Дата: 2026-07-10 | Фокус: почему фото 1999 помечаются как силикон из-за резкости/качества

Этот документ - 50 точечных анализов участков кода `s2_metrics/modules/texture_extractor.py` и смежных, где используется `skimage`.

Каждый пункт: файл, строка, что делает код, почему ломает детекцию реальной кожи 1999 vs силикон, фикс.

---

### [01] `texture_extractor.py:339-349` - `estimate_sigma` на CLAHE изображении
- **Код:** `sigma_est = estimate_sigma(gray_clahe, channel_axis=None)` после `clahe.apply()`
- **Проблема:** CLAHE усиливает контраст и зерно. `estimate_sigma` оценивает шум по высокочастотным деталям, после CLAHE sigma завышается в 2-3 раза (для 1999 скана 4.0 -> 9.5).
- **Следствие:** Порог `if sigma_est>2.0` срабатывает всегда для старых фото, запускается `denoise_wavelet`, который размывает поры. Размытая старая кожа становится похожа на восковый силикон.
- **Фикс:** Оценивать sigma ДО CLAHE: `sigma_est = estimate_sigma(gray_u8)` а не gray_clahe.

### [02] `texture_extractor.py:343-348` - `denoise_wavelet` с `method="BayesShrink" rescale_sigma=True`
- **Код:** `denoise_wavelet(gray_clahe, sigma=sigma_est, method="BayesShrink", mode="soft", rescale_sigma=True)`
- **Проблема:** BayesShrink сохраняет текстуру только если sigma точно оценен. При завышенном sigma (из-за [01]) порог вейвлет-коэффициентов высокий, удаляются не только шум, но и поры (2-4px). Для 120px лица поры 2px = высокочастотный шум.
- **Фикс:** `method="VisuShrink"` менее агрессивный, или `sigma=sigma_est*0.5`, или вообще не денойзить если `face_min_dim<150`.

### [03] `texture_extractor.py:332` - `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))` фиксированный
- **Код:** tile 8x8 независимо от размера лица.
- **Проблема:** Для лица 120px (1999) tile = 15x15px, для 600px (2024) tile=75px. В маленьком лице tile захватывает глаз+щеку вместе, CLAHE выравнивает разный lighting (тень от носа) и создает искусственные границы между тайлами. Границы дают высокий `glcm_contrast` ложно.
- **Фикс:** `tileGridSize = (max(4, face_h//32), max(4, face_w//32))` адаптивно.

### [04] `texture_extractor.py:401-420` - `graycomatrix` квантизация по перцентилям [2,98] levels=33
- **Код:** `lo,hi = percentile(skin_pixels, [2,98]); norm = clip((gray-lo)/(hi-lo),0,1); quantized = norm*32`
- **Проблема:** Для старого фото с низкой контрастностью (лоу 40, хай 180, span 140) перцентили отбрасывают только 4% пикселей, но span уже узкий. Квантизация 33 уровней на span 140 = шаг 4.25 градации на уровень. Для современного фото span 220 = шаг 6.8. Шаг меньше для старого фото -> больше уровней используется для мелкого шума, GLCM более разреженная, dissimilarity выше, хотя текстура более гладкая.
- **Фикс:** Использовать фиксированный span 0-255 для quantization, или levels=16 для low quality, или `skimage.exposure.rescale_intensity` с `in_range=(lo,hi)` без умножения на 32, а через `np.digitize`.

### [05] `texture_extractor.py:429` - `graycomatrix` distances [1,2,3,5] углы 4, но для 120px лица дистанция 5 = 4% ширины лица
- **Проблема:** Поры размером 2-3px, дистанция 5px уже захватывает соседнюю пору + фон. Для силикона с штампованными порами периодом 6px дистанция 5 близка к периоду, дает высокую корреляцию (регулярность). Для старой кожи с блюром дистанция 5 тоже выходит за пределы поры, дает низкую корреляцию. Оба случая дают низкую dissimilarity, путаются.
- **Фикс:** Дистанции должны масштабироваться от `face_min_dim`: `dist = max(1, int(face_min_dim/100 * d))`. Для 120px: [1,1,2,3], для 600px: [1,2,3,5].

### [06] `texture_extractor.py:431` - `graycomatrix(..., symmetric=True, normed=True)` - symmetric удваивает счетчик
- **Проблема:** symmetric=True считает пару (i,j) и (j,i) как два события. Для текстуры кожи это ок, но при низком количестве пикселей кожи (<2000 для 120px лица) GLCM разреженная, symmetric сглаживает, но также скрывает направленность (анизотропию) которая важна для детекции штамповки силикона.
- **Фикс:** Для анизотропии использовать `symmetric=False` и сравнивать углы, для однородности `symmetric=True`.

### [07] `texture_extractor.py:837-843` - `graycoprops` dissimilarity и homogeneity считаются, но correlation и ASM игнорируются
- **Проблема:** `correlation` - мера линейности, для силикона с периодическими порами correlation высокий на дистанции периода. `energy (ASM)` - однородность, для воска высокая. Ты их не считаешь, хотя они дискриминативны и более устойчивы к блюру чем dissimilarity.
- **Фикс:** Добавить `correlation`, `energy`, `contrast` все дистанции.

### [08] `texture_extractor.py:370-372` - `local_binary_pattern(P=8,R=1, method="uniform")` и `R=2`
- **Проблема:** Uniform LBP с P=8 дает 59 паттернов (58 uniform + 1 non-uniform). Ты считаешь `std(lbp_skin)` - стандартное отклонение кодов 0..58. Для гладкой области все коды 0 или 1 (яркие пятна) или 8 (темные), std низкий (0-5). Для кожи std средний 12-18. Для старой размытой кожи std 3-7, для силикона 2-6 - оба низкие.
- **Фикс:** Надо считать `lbp_uniform_ratio = non_uniform / total` или `histogram entropy`, а не std кодов.

### [09] `texture_extractor.py:384` - `local_binary_pattern(P=8,R=2, method="nri_uniform")`
- **Код:** `nri_uniform` - rotation invariant uniform, дает 0..9 коды.
- **Проблема:** Ты считаешь `n_uniform = hist_nri[:10].sum()` где 10 = P+2? Для P=8 nri_uniform всего 10 bins (0-9) где 9 = non-uniform. `hist_nri[:10]` включает non-uniform! Должен быть `[:9]`. В итоге `lbp_complexity_ratio = 1 - uniform/total` считает non-uniform как uniform, занижает сложность.
- **Фикс:** `n_uniform = hist_nri[:9].sum()` (0..8 uniform, 9 non-uniform).

### [10] `texture_extractor.py:436-458` - `uniform_filter` для `homo_local_var_w15_cv` размер 15 и 31
- **Код:** `local_m = uniform_filter(gray_f, size=15)` и `size=31`
- **Проблема:** Uniform filter - это box blur, а не Gaussian. Для размера 31 на лице 120px он захватывает глаз+щеку+нос в одном окне, local mean сильно сглажен, local std занижен. Для старой кожи и силикона оба дают низкий cv (~0.05-0.08). Разница теряется.
- **Фикс:** Использовать `gaussian_filter` с sigma= w/4 или `skimage.filters.rank` с disk.

### [11] `texture_extractor.py:455` - `cv_vals = local_std[valid]/local_m[valid]` + `clip(0,10)` - клип 10 слишком высокий
- **Проблема:** Для кожи cv обычно 0.15-0.4, для силикона 0.03-0.08, для старой размытой 0.04-0.10. Клип 10 не влияет, но если маска включает тень (темная область, local_m ~5), cv взрывается до 5-8, среднее завышается. Для старых фото с тенью от сканирования это часто.
- **Фикс:** `clip(0,1.0)` и `valid = local_m>10` вместо >1.0.

### [12] `texture_extractor.py:498-560` - FFT patch-based 64x64 stride 32
- **Код:** 64px патч, 50% overlap, заполнение non-skin mean.
- **Проблема:** Для лица 120px влезает всего 2x2=4 патча, из них 2 отбрасываются из-за `mask.sum()<0.5*area`. Остается 1-2 патча. Статистика по 1 патчу ненадежна. Для лица 600px - 18x18=324 патча, статистика стабильна.
- **Фикс:** Размер патча адаптивный: `patch = max(32, min(64, face_min_dim//3))`.

### [13] `texture_extractor.py:540` - FFT `high = magnitude[radius>8].sum()` `low = magnitude[radius<=4].sum()` без windowing
- **Проблема:** Нет окна Ханна, спектральная утечка (leakage) из low freq в high. Для старого фото с сильным low freq (освещение) утечка завышает high, fft_ratio кажется выше, чем есть. Для силикона с плоским освещением утечки нет, ratio низкий. Старое фото кажется более текстурным, чем силикон, но все равно низкий.
- **Фикс:** Применять `np.hanning` window перед FFT (у тебя делается только для pore periodicity, но не здесь).

### [14] `texture_extractor.py:546` - `peak_ratio = magnitude.max()/total` - max/total чувствителен к одному пику
- **Проблема:** Один яркий блик (specular) дает пик доминирующий, peak_ratio высокий и для реальной кожи с бликом, и для силикона с зеркальным бликом. Не дискриминативен.
- **Фикс:** Использовать `peak_ratio = top5_mean / median` или `fft_anisotropy`.

### [15] `texture_extractor.py:364` - `texture_entropy` через гистограмму 32 bins range 0-255 density True
- **Код:** `hist,density=True` потом `-sum(hist*log2(hist))`
- **Проблема:** `density=True` возвращает плотность, не вероятность, сумма hist*bin_width=1, но не sum(hist)=1. Энтропия считается неправильно (завышена). Для старого фото с узким диапазоном (40-180) bins за пределами диапазона имеют 0 плотность, энтропия низкая (~3.5), для современного широкого диапазона (20-240) энтропия выше (~5.2), для силикона узкий диапазон + гладкость -> энтропия 2.8. Старое и силикон оба низкие.
- **Фикс:** `density=False` + `prob = hist/hist.sum()`.

### [16] `texture_extractor.py:366` - `laplacian_var = var(Laplacian(gray_clahe))`
- **Код:** Laplacian variance для резкости.
- **Проблема:** Laplacian чувствителен к шуму. Для старого фото с пленочным зерном (шум) var высокий (200), хотя фото размыто. Для силикона гладкого var низкий (30). Но после denoise_wavelet зерно удаляется, var падает до 40, становится похож на силикон.
- **Фикс:** Laplacian до denoise, или использовать `skimage.filters.Laplace` с предварительным Gaussian.

### [17] `texture_extractor.py:761-766` - `skimage.filters.rank.entropy` с `disk(5)`
- **Код:** `entropy_img = entropy(gray.astype(uint8), disk(5))`
- **Проблема:** `rank.entropy` требует uint8 image и `disk` footprint. Для 120px лица disk 5 = радиус 5px = 11x11 окно = 10% лица. Энтропия в окне с глазом+кожей смешивается. Для старого фото энтропия низкая из-за блюра, для силикона тоже низкая.
- **Фикс:** Disk 3 для low res, 5 для high res.

### [18] `texture_extractor.py:436` - `morph_tophat` `white_tophat(gray_clahe, disk(r))` для r=4,8
- **Код:** Top-hat выделяет яркие детали меньше struct element.
- **Проблема:** Для поры размером 2px disk 4 уже слишком большой (вмещает пору + фон), top-hat захватывает не только пору, но и морщину. Для старого фото 120px поры 1px, disk 4 = 4x размера поры, top-hat почти 0. `pore_tophat_r4_std` низкий и для старой кожи, и для силикона.
- **Фикс:** r должен быть `max(1, int(face_min_dim/200 * r))`. Для 120px r=4 -> r=2.

### [19] `texture_extractor.py:568-600` - `pore_density` через `mean+std` threshold, area 0.01
- **Код:** `threshold = mean+std; pore_count = sum(tophat>threshold); area*=0.01`
- **Проблема:** 0.01 - магическая константа см2 на пиксель. Для 120px лица кожа ~5000px => 50 см2, для 600px ~80000px => 800 см2. Обе цифры нереальны (лицо ~150см2). Плотность несравнима между разрешениями. Старое фото дает плотность 0, силикон тоже 0.
- **Фикс:** Нормировать на `face_min_dim`: `area_norm = (mask.sum() / (face_min_dim**2))`.

### [20] `texture_extractor.py:610-630` - specular detection HSV v>220 s<40
- **Код:** `specular_mask = (v>220) & (s<40)`
- **Проблема:** Пороги 220 и 40 работают для светлой кожи RGB. Для темной кожи, загорелой, или старого фото с желтоватым балансом пленки v редко >220 (макс 200), specular не детектится. Для силикона с ярким зеркальным бликом v 240-255 детектится. Но для старой кожи с бликом от вспышки v 210 не детектится -> `specular_ratio=0` и для старой кожи и для силикона без блика.
- **Фикс:** Адаптивный порог `v_thresh = percentile(v_skin, 95)`.

### [21] `texture_extractor.py:640-710` - `pyramid_gaussian` 3 уровня
- **Код:** `pyramid_gaussian(downscale=2)` - каждый уровень в 2 раза меньше.
- **Проблема:** Уровень 0: оригинал (поры), Уровень1: 0.5x (морщины), Уровень2: 0.25x (тени). Для лица 120px уровень2 = 30px - уже не лицо, а пятно. GLCM на 30px бессмыслен. Для 600px уровень2=150px еще ок. Старые фото теряют уровень2.
- **Фикс:** Только 2 уровня для face_min<200.

### [22] `texture_extractor.py:699` - на пирамиде GLCM `levels=33`
- **Код:** На каждом уровне снова квантизация 33 уровня.
- **Проблема:** На уровне1 изображение уже сглажено gaussian, динамический диапазон сужается, 33 уровня избыточны, GLCM разреженная, dissimilarity шумная.
- **Фикс:** levels=16 для уровня1, levels=8 для уровня2.

### [23] `texture_extractor.py:750` - gradient magnitude `sobel` через `scipy.ndimage.sobel`
- **Код:** `sobel(gray, axis=1)` и `axis=0`
- **Проблема:** `ndimage.sobel` без нормализации, возвращает int, может переполниться для uint8. Лучше `cv2.Sobel` с CV_64F как в других местах. Смешивание двух реализаций дает разный scale.
- **Фикс:** Использовать одну библиотеку.

### [24] `texture_extractor.py:810-854` - `_compute_glcm_with_percentiles` ES второй раз, дублирует `_extract_skin_metrics` GLCM
- **Проблема:** Две реализации GLCM в одном файле: одна в `extract_skin_metrics` через `_compute_glcm_with_percentiles`, другая в `_extract_multiscale_texture`. Они дают разные метрики с одинаковыми именами, последняя перезаписывает первую.

### [25] `s2_metrics/modules/texture/catalog.py` - `TEXTURE_CORE_METRICS` включает `lbp_uniform_r5_std` которого нет в extractor (там `lbp_uniform_r5_std` = std LBP R=2, но имя вводит в заблуждение R=5)
- **Проблема:** Каталог ожидает метрику R=5, а extractor дает R=2, классификатор учится на неверной метке.

### [26] `s2_metrics/physical_features.py:30-50` - `estimate_sigma` снова на gray, а не на albedo
- **Код:** `sigma_est = estimate_sigma(gray)` перед physical features.
- **Проблема:** Двойной denoise: уже был в texture_extractor, теперь снова. Для старого фото второй denoise еще сильнее размывает, `pore_periodicity` (FFT энтропия) стремится к 1.0 (хаотичная) -> считается как реальная кожа, а не силикон? Наоборот, для силикона periodicity должна быть низкой (регулярность). Двойной denoise делает силикон более хаотичным, маскирует штамповку.

### [27] `physical_features.py:_compute_sss` - SSS считается на albedo (grayscale), а нужно на RGB (R-B diff)
- **Код:** `albedo[ear_mask].mean()` - albedo уже grayscale после `gaussian` нормализации, R-B diff потерян. SSS всегда 0.
- **Фикс:** Использовать оригинальное RGB, канал R минус B на ухе.

### [28] `physical_features.py:_compute_specular_sharpness` - выделяет блики через `albedo > mean+3*std`
- **Код:** mean+3std на albedo (уже нормализованном). Для старого фото с низким контрастом std маленький (5), mean 120, порог 135, много пикселей >135 считаются бликами ложно. `specular_ratio` завышается, старая кожа кажется силиконом (у силикона много бликов).

### [29] `physical_features.py:_compute_pore_periodicity` - патч 64x64 берется центральный, но для профиля центр - щека, для фронтала - нос
- **Проблема:** Для 3/4 deep центральный патч попадает на невидимую область (затылок), mask sum <1000 -> возвращает 1.0 (хаотичная) -> реальная кожа.

### [30] `physical_features.py:_compute_lbp_nonuniform` - radii [2,3,5] n_points=8*r => для r=5 n_points=40, LBP с 40 точками на 64x64 патче требует окно 11x11, граничные эффекты, много точек за пределами ROI

### [31] `physical_features.py:_compute_spectral_slope` - radial bins 20, max_r 32, valid sum<4 return 2.5 - часто возвращается дефолт 2.5 для low res, который как у реальной кожи, маскирует силикон

### [32] `physical_features.py:_compute_seam_score` - seam через `dilate-erode` boundary, но boundary включает волосы и очки, если seg_mask плохая

### [33] `s2_metrics/texture_anomaly.py:FEATURE_MAP` - маппинг internal -> extractor: `skin_brightness_std` -> None (отсутствует), `lbp_complexity_ratio` -> None
- **Проблема:** Два важных признака мапятся в 0.0 fallback, всегда 0, не участвуют в anomaly.

### [34] `texture_anomaly.py:get_cohort_key` - эры 1999-2005,2005-2012,2012-2021,2021+ - но 1999-2005 включает и сканы и цифру, качество сильно разное внутри эры

### [35] `texture_anomaly.py:score` - `robust_z = abs(x-median)/mad` но mad может быть 1e-6 (из-за fallback 0.0 в FEATURE_MAP) -> z взрывается

### [36] `texture_anomaly.py:threshold = 2.0 + (1-quality)*3.0` - для quality 0.2 threshold 4.4, для качества 0.9 threshold 2.3, разница 1.9x, но для low quality std baseline выше, должно быть threshold 5-6

### [37] `s3_identity/modules/texture_calibrator.py:_fit_quality_curve` - polyfit quality vs metric, но quality и metric коррелируют из-за того что quality считается из тех же метрик (sharpness из laplacian_var) - circular dependency, slope фиктивный

### [38] `s2_metrics/engine.py:extract` - `project_texture_aliases` вызывается ПОСЛЕ фильтрации sensitive, но aliases могут вернуть sensitive метрики обратно

### [39] `s2_metrics/modules/texture/aliases.py:project_texture_aliases` - `homo_local_var_w15_cv = gray_std/gray_mean` - это коэффициент вариации яркости, а не локальной вариации! Совсем другая метрика, но имя то же.

### [40] `texture_extractor.py:403` - `skin_pixels = gray_u8[mask>0]` - mask это alpha>30, но alpha от 3DDFA seg_visible часто включает брови и бороду, которые темные, gray_mean занижается

### [41] `texture_extractor.py:412` - `gray_u8` и `gray_clahe` используются параллельно, но gray_u8 для gray_mean/std, а gray_clahe для LBP/GLCM - две разные нормировки, метрики не согласованы

### [42] `physical_features.py:_normalize_illumination` - `gaussian(sigma=40)` на 120px лице sigma 40 = почти весь размер лица, blur = mean, albedo = gray/mean ~1.0 для всех пикселей, теряется вся информация

### [43] `s2_metrics/engine.py:physical_extractor.extract` - вызывается с `landmarks_68` которые из reconstruction.pkl, но reconstruction.pkl vertices_canonical, а не image coords - landmarks не совпадают с image RGB

### [44] `texture_extractor.py:340` - `estimate_sigma` с `channel_axis=None` deprecated в skimage 0.20, должен быть `channel_axis=None` для 2D правильно, но warning спамит логи

### [45] `s2_metrics/modules/zone_analyzer.py:filter_textures_by_pose` - текстуры фильтруются по pose, но GLCM уже посчитан без учета pose, visible mask не применяется

### [46] `s2_metrics/texture_anomaly.py:lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True, metric="mahalanobis")` - mahalanobis требует больше samples чем features (10 features, 20 neighbors, но cohort early_scan может иметь 15 samples) - падает

### [47] `s2_metrics/modules/texture_extractor.py:_get_skin_mask` - `alpha>30` - порог 30 слишком низкий, для JPEG с артефактами alpha feather 20-40 включает полупрозрачный фон

### [48] `texture_extractor.py:380-390` - `lbp_complexity_ratio` считается как `1 - uniform/total` для nri_uniform, но для uniform метода не считается, хотя для R=1 uniform тоже важен

### [49] `s2_metrics/modules/texture_extractor.py:761` - `entropy(gray.astype(uint8), disk(5))` - rank entropy ожидает uint8, но gray может быть float после CLAHE (0-255 uint8 ok), но disk footprint не нормализован

### [50] `s2_metrics/modules/texture_extractor.py:852-854` - `texture_glcm_contrast/homogeneity/energy` берется `[0,0]` = distance 1 angle 0, но для силикона важна анизотропия (разница между углами), а берется только один угол

---

## ИТОГОВАЯ ДИАГНОСТИКА ПОЧЕМУ 1999=СИЛИКОН

Суммируя 50 ошибок, получается каскад:

1. Старое фото 120px, blur, JPEG → quality 0.25 low
2. CLAHE 8x8 создает блоки → GLCM contrast завышается, но потом...
3. estimate_sigma завышается из-за CLAHE → denoise_wavelet размывает поры
4. 37 sensitive метрик удаляются → остаются только те что дают low complexity и для старой кожи и для силикона
5. Оставшиеся `fft_highfreq` low (0.08 vs 0.35 modern baseline) → z-score -5.4
6. `lbp std` low (3 vs 15) → z-score высокий
7. LOF в `texture_anomaly` с mahalanobis на 15 samples падает, fallback median/MAD с 1e-6
8. Physical features не извлекаются из-за маленьких landmarks → seam_score 0
9. Итог: `synthetic_suspicion 0.8` → H1_SYNTHETIC

**Фикс за 1 день:**
- Не удалять sensitive метрики, а взвешивать
- Оценивать sigma до CLAHE
- Отключить denoise для low quality
- Адаптивный patch/tile/R от face_min_dim
- Когортный baseline early_scan_low vs early_scan_low, а не vs modern high
- Добавить robust метрики: spectral slope, seam, albedo_a_std, которые работают на low res
- Порог для low quality: `if quality<0.35: return UNCERTAIN, не SILICONE`

*Конец 50 анализов текстуры.*
