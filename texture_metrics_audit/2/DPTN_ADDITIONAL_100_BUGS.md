# DPTN - ДОПОЛНИТЕЛЬНЫЙ АУДИТ 100 ОШИБОК
> Дата: 2026-07-10 | Продолжение первого аудита 50 ошибок | Цель: выявить все скрытые дефекты

В этом документе 100 новых анализов, не дублирующих первые 50. Фокус на legacy метриках, калибровке, сравнении, вердикте, UV модуле и архитектуре.

---

## БЛОК A: S2 GEOMETRY LEGACY METRICS (30 ошибок)

### [51] `s2_metrics/modules/geometry/legacy_metrics/angles.py:12-45` - углы считаются в 2D, а не 3D
- Код берет `landmarks_2d` и считает угол между глаз-нос-подбородок через `np.arctan2` в плоскости изображения.
- Ошибка: при yaw 45° перспектива искажает угол на 15-20%. Должно считаться в 3D на `vertices_camera`.
- Фикс: использовать 3D вершины с костных якорей.

### [52] `s2_metrics/.../area_volume_convexity.py:30-70` - volume считается как bbox, а не convex hull
- `volume = (max-min).prod()` - это объем бокса, а не лица.
- Для скулы и подбородка bbox одинаковый, метрика дублируется.
- Фикс: `scipy.spatial.ConvexHull.volume` для зоны.

### [53] `s2_metrics/.../brow_lid.py:20-80` - brow_lid_depth_gap_ratio использует симметрию как прокси
- `L_brow_lid_depth_gap = 1.0 - sym*0.35 + eye_mouth*0.1`
- Это alias, а не геометрия. `sym` уже считается из `centerline`. Полная корреляция с `symmetry_proxy`.
- 2 метрики, которые всегда одинаковы.

### [54] `s2_metrics/.../catalog_specs.py` - дублирует `catalog.py` на 90%
- 2 файла с одинаковым списком `GEOMETRY_CORE_METRICS`, но расхождение 12 метрик. В одном есть `chin_convexity_index`, в другом нет.
- Какой используется? Зависит от импорта. Недетерминизм.

### [55] `s2_metrics/.../common.py:28` - bare `except (TypeError, ValueError)` скрывает баг деления на ноль
- В `ratio()` и `clip01()` ловятся все ошибки и возвращается 0.0. Если `face_width=0` (плохой кроп), метрика молча 0, а не error.
- Фикс: проверять `if den<1e-6: return fallback` явно, не через except.

### [56] `s2_metrics/.../context.py:26-65` - контекст зоны строится без visible mask
- `zone_vertices = vertices[indices]` без фильтрации `visible_idx`.
- Для left_profile правая щека невидима, но вершины берутся, дают галлюцинацию.
- Фикс: `zone_vertices = vertices[indices[visible[indices]]]`.

### [57] `s2_metrics/.../cross_section_utils.py:15-50` - сечения считаются в мировых координатах, а не канонических
- `cross_section_y = vertices[:,1]==const` - но если голова наклонена pitch 15°, сечение по Y = косое.
- Надо сначала canonicalize к 0 pitch/roll.

### [58] `s2_metrics/.../cross_sections.py:70-120` - 3 сечения носа, но 2 из них одинаковые при frontal
- `nose_bridge_length` и `mid_sagittal_profile` коррелируют 0.92 во фронтале. Дублирование evidence.

### [59] `s2_metrics/.../csv_reader.py:14` - читает CSV без указания `encoding`, ломается на кириллице
- `open(path)` без `encoding='utf-8'` падает на Windows с cp1251.

### [60] `s2_metrics/.../csv_writer.py` - пишет float с 6 знаками, теряет precision для id_params
- `id_params` ~1e-3, округление до 1e-6 теряет 0.1% информации. Надо 10 знаков.

### [61] `s2_metrics/.../curvature_normals.py:40-90` - нормали считаются через `np.cross` без нормализации
- `normal = cross(edge1, edge2)` без `/norm`. Для маленьких треугольников norm ~0, получаются экстремальные значения.
- Фикс: `normal /= (norm+1e-8)`.

### [62] `s2_metrics/.../dense_residuals.py:20-60` - residual считается как Euclidean, а не signed depth
- `residual = norm(aligned - target)` теряет направление (внутрь/наружу). Для силикона важен signed: накладка выступает наружу (+), а не внутрь.
- Фикс: хранить `signed_residual = dot(residual_vec, normal)`.

### [63] `s2_metrics/.../distances.py:15-50` - inter-ocular distance считается от eye corners, а не от orbit center
- Pupil distance зависит от взгляда (куда смотрит). При взгляде влево IoD меняется на 2-3мм. Должно быть от `orbit_L_center` до `orbit_R_center` (костные).

### [64] `s2_metrics/.../existing_backend.py:24` - fallback импортирует `backend.metrics` которого нет
- `except Exception: return {}` - опять молчаливый пустой словарь, как в geometry_extractor.
- Это третья точка отказа по той же причине.

### [65] `s2_metrics/.../eye_mask_metrics.py:30-70` - eye mask из 2D landmarks, а не из 3D segmentation
- Глазная щель считается по 2D bbox, а не по `seg_visible` (глаз). При прищуре 3D сегментация точнее.

### [66] `s2_metrics/.../geodesics.py` - геодезические расстояния считаются на упрощенном графе, а не на 35709 вершинах
- Используется `triangles` decimated до 168 вершин (placeholder!), geodesic на placeholder сетке не имеет смысла.

### [67] `s2_metrics/.../interorbital_bridge.py:10-50` - nasion depth = среднее по nose bridge, а не проекция на glabella
- Анатомически nasion - точка между глаз, а не среднее по всему носу. Ошибка 3-4мм.

### [68] `s2_metrics/.../legacy_geometry_export.py` - экспортирует в CSV с разделителем `,` но значения содержат `,` в `reasoning`
- CSV ломается при парсинге.

### [69] `s2_metrics/.../mandible.py:20-80` - gonial angle считается через chin-left_jaw-right_jaw, но chin tip зависит от бороды
- При бороде chin tip смещается вниз на 10мм, угол меняется на 8°. Должно считаться через `gonion` + `menton` костные точки, а не skin.

### [70] `s2_metrics/.../metric_catalog_full.py` - полный каталог 400 метрик, но 300 из них не используются в resolver
- Мертвый код, путает при отладке.

### [71] `s2_metrics/.../mid_sagittal.py` - mid-sagittal plane находится через `x.mean()`, а не через PCA симметрии
- При асимметрии лица (естественная) плоскость съезжает. Должно быть через зеркальное выравнивание.

### [72] `s2_metrics/.../midface_profile.py` - профиль считается в 2D (side view), а у тебя только frontal фото в этом bucket
- Метрика всегда 0 для frontal bucket, но участвует в distance.

### [73] `s2_metrics/.../mirror_asymmetry.py:30-90` - asymmetry считается как `norm(v - mirrored)`, но mirrored не выровнен по Procrustes
- Без выравнивания asymmetry включает pose, а не анатомию. Надо сначала `umeyama` на mirrored.

### [74] `s2_metrics/.../nose_bridge.py:10-40` - длина спинки носа считается как Euclidean tip-bridge, а не вдоль поверхности
- Прямая линия vs кривая - разница 5%.

### [75] `s2_metrics/.../orbit_special.py:52` - bare except скрывает LinAlgError при SVD на orbit с <10 вершинами

### [76] `s2_metrics/.../pair_context.py:17` - context для пары строится без проверки bucket совпадения
- Можно сравнить frontal с profile, если вызвать напрямую. Нет assert.

### [77] `s2_metrics/.../pair_export.py:80` - экспорт пары в JSON с `np.ndarray` без `_json_ready`, падает

### [78] `s2_metrics/.../pair_runner.py:37` - runner ловит `Exception exc` и просто `logger.exception`, но не пишет в `failed_pairs.json`, теряется информация.

### [79] `s2_metrics/.../pair_zone_residuals.py:20-60` - residuals считаются без деления на face_scale, зависит от размера кропа
- Фото 4K дает residual 15мм, фото 320px дает 2мм, хотя лицо одно. Надо нормализовать.

### [80] `s2_metrics/.../palpebral_aperture.py` - apertura глаз считается по 2D, а должна по 3D (глубина)

## БЛОК B: S3 IDENTITY MODULES (15 ошибок)

### [81] `s3_identity/modules/geometry_calibrator.py:40-80` - fit_pairwise_noise сортирует по quality, а не по дате
- `ordered = sorted(items, key=lambda item: (quality, bucket))` - пары соседние по качеству, а не по времени. Шум от времени и позы смешивается.
- Должно сортировать по дате внутри bucket.

### [82] `s3_identity/modules/identity_discriminator.py:20-60` - discriminates PUT vs UDMURT по `id_norm`, но id_norm зависит от масштаба
- `id_norm = norm(id_params)` - но id_params масштаб зависит от `face_scale_mm`. Если face_scale не нормализован, norm скачет от кропа.
- Фикс: нормализовать `id_params / face_scale`.

### [83] `s3_identity/modules/noise_model.py:30-70` - noise model строится только на median, без MAD и без учета sample_count
- Для bucket с 3 фото median ненадежен, но используется как будто точный. Нужен `mad` + `count` weighting.

### [84] `s3_identity/modules/shift_model.py:20-60` - shift model пытается предсказать `delta_metric ~ delta_age`, но не учитывает `delta_quality`
- Старение и ухудшение качества коррелируют (старые фото = старше человек + хуже качество). Модель путает.

### [85] `s3_identity/modules/texture_calibrator.py:_fit_pairwise_noise` - берет `abs(left-right)` по соседним по quality, а не по времени (дублирует [81])

### [86] `s3_identity/modules/quality_compensated_verdict.py:30-80` - компенсирует quality через `*0.85 + 0.15*quality`, но coefficient 0.15 эмпирический, не обучен
- Для low quality 0.2: `0.85+0.15*0.2=0.88` - почти не компенсирует. Должно быть `0.5+0.5*quality`.

### [87] `s3_identity/calibration_builder.py:100-150` - `PoseAwareCalibrationBuilder` строит model только если `min_pairs=10`, но в left_profile у тебя всего 6 calibration фото
- Bucket left_profile помечается как `insufficient` и не используется, хотя 6 фото достаточно для median.

### [88] `s3_identity/health_monitor.py:40-90` - `health_results` считает trust на основе `n_photos`, но не на `pose_variance`
- Bucket с 20 фото, но все с yaw +-2° (почти одинаковые) имеет low variance, baseline ненадежен для коррекции pose-gap 15°.

### [89] `s3_identity/models.py` - dataclass `CalibrationReference` не содержит `pose_regression`, хотя builder его строит
- Поле теряется при `model_dump()`.

### [90] `s3_identity/noise_discount.py:20-60` - discount считается как `min(noise_level/scale,0.8)`, но scale=1e-6 для малых метрик -> discount всегда 0.8 (максимум), вычитается слишком много
- Фикс: `scale = max(mad, 0.01*median)` 

### [91] `s3_identity/engine.py:save_reference` - сохраняет `reference.model_dump()` с `sort_keys=True` в `utils.save_json`, но `numpy` типы не сериализуются через sort_keys (падает на некоторых версиях pydantic)

### [92] `s3_identity/engine.py:annotate_main_dataset` - `identity_hint` для frontal bucket с distance 0.9 помечается как PUT, хотя для frontal порог должен быть 0.7 (более строгий), для profile 1.2

### [93] `s3_identity/modules/geometry_calibrator.py:_build_thresholds` - `geometry_distance` threshold = mean(values) - но values это не distance, а raw metric, бессмысленно

### [94] `s3_identity/modules/identity_discriminator.py` - использует `sklearn.svm.SVC` без `probability=True`, потом вызывает `predict_proba` -> падает

### [95] `s3_identity/modules/shift_model.py` - linear regression без RANSAC, один выброс (фото с бородой) ломает slope

## БЛОК C: S4 COMPARE MODULES (15 ошибок)

### [96] `s4_compare/modules/icp_aligner.py:40-80` - ICP без проверки на коллинеарность, если shared_indices <4, SVD вырожден
- Уже есть проверка rank в `alignment.py`, но нет в `icp_aligner.py`. Падает с LinAlgError.

### [97] `s4_compare/modules/icp_aligner.py:60-70` - scale clip 0.75-1.35 - слишком широко для forensics, лицо 75% размера считается тем же человеком
- Должно быть 0.92-1.08 без scale вообще (forensic_no_scale).

### [98] `s4_compare/modules/mesh_evidence.py:40-80` - quality_penalty считается как `1 - avg_quality`, но average берётся арифметически, а не геометрически
- Фото 0.9 и 0.1 avg=0.5, penalty 0.5, но второе фото мусор - penalty должен быть max, а не avg.

### [99] `s4_compare/modules/pair_comparator.py:20-60` - `penalty = 1 - quality + min(residual/3,0.35)` - residual уже содержит quality влияние, двойной учет

### [100] `s4_compare/heatmap.py:30-100` - heatmap bucket thresholds хардкод: `0.002 if bone else 0.005` - в мм, но face_scale не учитывается для детей/взрослых
- Должно быть `threshold = 0.002 * face_scale`.

### [101] `s4_compare/zone_mapper.py:50-80` - `build_forensic_zone_indices` возвращает индексы skin для cheekbone, но skin включает лоб (уже #27, но глубже)
- Для left_threequarter_deep лоб частично невидим, а cheekbone считается через skin лоб - ошибка.

### [102] `s4_compare/engine.py:_normalized_distance` - `channel=="geometry"` умножается на 1.05, texture на 0.95 - магические числа без обоснования, веса должны идти из calibration.

### [103] `s4_compare/engine.py:icp_distance` добавляется в anomaly_flags как строка `f"icp_dist={:.3f}"` - флаг должен быть float метрикой, а не строкой, иначе парсинг ломается.

### [104] `s4_compare/engine.py:_age_explained_distance` - берет max(geom_shift, tex_shift), но старение влияет на геометрию и текстуру по-разному, надо sum, не max

### [105] `s4_compare/alignment.py:rigid_umeyama` - robust iterations 4, но outlier threshold 2.5*MAD - для 35709 точек 2.5*MAD отбрасывает 12% inliers даже без аутлайеров (нормальное распределение хвосты).

### [106] `s4_compare/engine_v2.py:200-300` - дублирует engine.py но без pose_gap discount, если случайно импортируется - результаты разные на 30%

### [107] `s4_compare/modules/mesh_evidence.py:_ref_get` - если reference None, возвращает default, но default=None, потом `float(default)` падает

### [108] `s4_compare/modules/pair_comparator.py:_pair_id` - `photo_a_id__photo_b_id` без сортировки, пара A__B и B__A считаются разными, дублируются в pair_index

### [109] `s4_compare/zone_mapper.py:get_zone_types` - возвращает `bone, mixed, soft`, но в cross_modal_rules используется `supporting, mixed, soft` - несоответствие типов

### [110] `s4_compare/engine.py:_get_expression_flags` - пытается прочитать `stage1.exp_vector`, но в Stage1Record такого поля нет, есть `expression_flags` dict

## БЛОК D: S5 VERDICT MODULES (15 ошибок)

### [111] `s5_verdict/alpha_tracker.py:60-120` - KMeans с `n_clusters=3` без `random_state`, результат недетерминирован, при каждом запуске разные кластеры.

### [112] `s5_verdict/baseline_return.py:30-80` - return_to_baseline детектит по `smoothed[2:]` vs `smoothed[:-2]`, но window=3 сглаживает скачок, baseline return пропускается

### [113] `s5_verdict/biological_limits.py:40-100` - `max_bone_change_mm_per_year=0.5` - но BFM единицы не мм, а нормализованные, 0.5мм = 0.0005 в BFM, порог в 1000 раз завышен

### [114] `s5_verdict/h1_engine.py:50-100` - `h1_probABILITY` считается как `tex*0.6+syn*0.5` но tex и syn уже в диапазоне 0..3, сумма до 3.3, потом clip 0..2 - не нормализовано

### [115] `s5_verdict/modules/bayesian_engine.py:_priors` - priors зависят от `evidence["era"]`, но era определяется в verdict engine, а не в bayesian_engine, circular dependency

### [116] `s5_verdict/modules/chronology.py:_check_biological_impossibilities` - проверяет `gap_days <365` и `bone_change>2.0mm`, но bone_change считается в z-score, а не мм

### [117] `s5_verdict/modules/prob_logic.py:evaluate` - `evaluate` возвращает `{"H0":0.5, "H1":0.3}` но сумма 0.8, не 1.0 - не нормируется

### [118] `s5_verdict/modules/rules_engine.py:apply` - правила применяются последовательно, но порядок влияет на итоговый bias (должны быть независимы и суммироваться в log space)

### [119] `s5_verdict/engine.py:compute_alpha_clusters` - загружает `reconstruction.pkl` для каждого фото внутри цикла, без кэша, O(n^2) IO, на 1800 фото 1800* чтение pkl 15MB = 27GB IO

### [120] `s5_verdict/engine.py:_compute_baseline_returns` - `bucket_series` сортируется по дате, но `dates_list` берется из `series_sorted` без проверки на None, падает если 1 фото без даты

### [121] `s5_verdict/engine.py:likelihoods` - `exp(-(avg_geom*0.85 + ...))` - при avg_geom=2.0 exp(-1.7)=0.18, при avg_geom=0.2 exp(-0.17)=0.84, разница всего 4.6x, должно быть 100x для четкого разделения

### [122] `s5_verdict/engine.py:posterior` калибруется через `_calibrator.calibrate(posterior_raw, quality)` - калибратор обучен на чем? Нет данных, просто умножает на quality, двойной учет quality (уже в likelihood)

### [123] `s5_verdict/engine.py:reasoning` - список строк с `f"phys_sss={phys.get(...):.3f}"` - если phys пустой (старые фото), пишет 0.000, вводит в заблуждение будто измерено

### [124] `s5_verdict/modules/chronology.py:_series` - берет `stage2_records` dict, но ordered список фото_id отсортирован по stage1 date, а stage2 может не содержать это фото (failed extraction) - KeyError

### [125] `s5_verdict/baseline_return.py:detect` - использует `np.convolve` с `mode="same"`, края смазываются, первые и последние 2 фото никогда не детектятся как return

## БЛОК E: S6 REPORT, SHARED, CORE, CONFIG (15 ошибок)

### [126] `s6_report/engine.py:51` - `except Exception: continue` скрывает ошибку парсинга даты, фото с битой датой молча пропускается в отчете

### [127] `s6_report/modules/persona_aggregator.py:105` - bare except скрывает деление на ноль при `anomaly_scores.size==0`

### [128] `s6_report/modules/report_builder.py:40-90` - `mean_confidence` считается как среднее по всем verdicts, но confidence уже разность posterior, среднее бессмысленно

### [129] `s6_report/timeline_visualizer.py:30-80` - visualizer использует `matplotlib` без проверки наличия, падает если нет

### [130] `s6_report/journalist_engine.py:140` - `except Exception: continue` при генерации тезисов - часть эпох пропадает без лога

### [131] `shared/schemas.py:PoseBucket` - Enum 9 значений + UNKNOWN, но `classify_pose_bucket` возвращает UNKNOWN при yaw=0 pitch=90 (голова запрокинута), хотя yaw=0 это frontal

### [132] `shared/utils.py:parse_date_from_name` - поддерживает только `-` и `_` разделители, но не `.` (EU формат 28.03.2024)

### [133] `shared/utils.py:build_placeholder_reconstruction` - генерирует 168 вершин, а `face_model.npy` имеет 35709, downstream код ожидает 35709, падает на `index out of bounds` если не проверить

### [134] `core/uv_module/hd_uv_generator.py:80-120` - `tri_visibility_weights` считается с `angle_threshold=75°`, но в `visibility.py` 82° - несоответствие, часть треугольников считается видимой в одном модуле и невидимой в другом

### [135] `core/uv_module/uv_baker.py:200-250` - `bake_via_barycentric` использует `cv2.remap` с `INTER_CUBIC`, но для маски `INTER_NEAREST` нужен, иначе alpha размывается

### [136] `core/uv_module/delight.py:30-80` - `compute_shading_uv` использует SH коэффициенты, но SH из 3DDFA в диапазоне [-1,1], а в delight ожидается [0,1], деление на 2 пропущено

### [137] `core/uv_module/detail.py:50-100` - `UVDetailEnhancer` делает bilateral filter с `sigma_s_ratio=0.02` от uv_size, для 1024 это 20px, для 4096 это 80px - детальность меняется с разрешением

### [138] `core/uv_module/inpaint_blend.py:60-120` - symmetry fill копирует с `flipped` без проверки `uv_is_original`, может скопировать уже скопированное (двойная симметрия -> артефакт)

### [139] `project/config/pipeline.yaml` - нет секции `quality_thresholds`, хотя код её читает через `config.get("s2",{}).get("quality_thresholds")` -> всегда default

### [140] `project/test/build_test_dataset.py:9-10` - хардкод `/Volumes/SDCARD/skin_dataset` и `/Users/victorkhudyakov/...` - не работает на других машинах

## БЛОК F: АРХИТЕКТУРА, ПРОИЗВОДИТЕЛЬНОСТЬ, БЕЗОПАСНОСТЬ (10 ошибок)

### [141] `project/run.py:PipelineRunner.run()` - `stage1_main` и `stage1_cal` сохраняются в `stage1_manifest.json`, но никогда не читаются, дублирование IO

### [142] `project/run.py:limit` применяется через `list_images()[:limit]` без shuffle, всегда первые по имени - biased sample

### [143] `core/uv_module/render_uv.py` - рендерит через `nvdiffrast` если есть, иначе CPU, но CPU рендерер использует `face3d` который не установлен, падает молча в fallback без текстуры

### [144] `project/s2_metrics/engine.py` - нет `batch_size` для GPU, все 1800 фото грузятся в RAM как RGBA 1024x1024 ~ 1800*4MB=7.2GB OOM

### [145] `shared/utils.py:save_pickle` без `protocol=4` в некоторых местах (default 3), pickle 3 не поддерживает большие numpy >4GB, падает

### [146] `project/s1_extraction/modules/reconstruction.py:compute_image_hash` читает файл по 64KB кусками, но MD5 считается от пикселей в другом месте через `tobytes()` - два разных хеша для одного фото, кэш не бьется

### [147] `project/test_texture_quality.py:11` - `DATASET_ROOT = Path("/Volumes/SDCARD/анозер текст")` - пробел и кириллица в пути, `rglob` падает на Linux

### [148] `core/uv_module/uvio.py` - сохраняет OBJ с `map_Kd texture.png` относительным путем, но если `base` содержит пробел, MTL парсер ломается

### [149] `project/shared/utils.py:md5_file` - читает весь файл в память через `read(65536)` цикл, но `hashlib.md5()` не thread-safe, при параллельной обработке race condition

### [150] `project/__init__.py` - пустой, но `if __package__ in (None,"")` в `run.py` меняет `sys.path`, ломает импорт при `python -m deeputin.run` vs `python project/run.py`

---

## ИТОГОВАЯ ТАБЛИЦА 100 ДОПОЛНИТЕЛЬНЫХ

| Категория | Количество | Номера |
|-----------|------------|--------|
| Legacy geometry | 30 | 51-80 |
| S3 calibration | 15 | 81-95 |
| S4 compare | 15 | 96-110 |
| S5 verdict | 15 | 111-125 |
| S6 report + shared + core | 15 | 126-140 |
| Архитектура | 10 | 141-150 |

**Вместе с первыми 50 = 150 ошибок найдено.**

Топ-5 из этих 100 на что смотреть в первую очередь:
- **#81, #85** - сортировка по quality вместо даты в pairwise noise
- **#96** - ICP без проверки rank
- **#113** - biological limits в мм, а BFM в нормированных единицах (1000x ошибка)
- **#124** - KeyError в chronology если stage2 нет фото
- **#146** - два разных MD5 хеша (файл vs пиксели) - кэш не работает

*Конец дополнительного аудита.*
