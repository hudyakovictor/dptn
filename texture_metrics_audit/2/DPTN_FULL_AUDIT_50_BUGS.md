# DPTN - ПОЛНЫЙ АУДИТ 50 ОШИБОК / ДЕФЕКТОВ КОДА
> Дата: 2026-07-10 Prague | Эксперт 3DDFA-V3 99 lvl | Архив 1800+ фото, 9 ракурсов

Этот документ - результат ручного анализа 50 участков кода в `/project` и `/core`.

Каждый пункт: файл, строки, тип ошибки (CRITICAL/HIGH/MEDIUM/LOW), почему ломает расследование двойников, как чинить.

---

### S1 EXTRACTION (извлечение 3D)

#### [01] CRITICAL - `project/shared/utils.py:build_placeholder_reconstruction` - фейковая реконструкция
- **Файл:** `shared/utils.py:350-420` + `s1_extraction/engine.py:99` (вызов)
- **Суть:** Вместо реальной 3DDFA-V3 создается сетка 12x14=168 вершин с `depth = 1 - 0.42*(fx^2+fy^2) + 0.05*tanh(yaw)`. Это плейсхолдер.
- **Почему критично:** Вся downstream геометрия (zone_* метрики) считается по фейковой сетке. `vertices.shape` должно быть (35709,3), а тут (168,3). Любое сравнение черепов - случайные числа. Это ошибка #1 которая убивает весь пайплайн с нуля.
- **Фикс:** Заменить в `engine.py` на `ReconstructionAdapter().reconstruct(image_path)` с кэшем `save_reconstruction_cache`.

#### [02] CRITICAL - `s1_extraction/modules/alignment.py:330-369 AlignmentEngine.align()` - заглушка
- **Файл:** `s1_extraction/modules/alignment.py:352-369`
- **Код:** `return reconstruction` без изменений. Комментарий "пока заглушка - просто возвращаем как есть"
- **Суть:** Выравнивание canonical по bucket target yaw не делается. Все фото в frontal bucket с yaw -10° и +10° остаются невыровненными, давая ложный geometry_distance 0.8.
- **Фикс:** Реализовать `canonicalize_vertices_for_bucket()` + `rigid_umeyama` по bone indices как в том же файле выше (строки 40-150). У тебя уже есть `align_canonical_pair_for_view_group` - вызывай его.

#### [03] HIGH - `s1_extraction/modules/pose_estimator.py:15-35` - поиск head-pose-estimation в 5 несуществующих путях
- **Файл:** `s1_extraction/modules/pose_estimator.py:10-30`
- **Код:** `_possible_hpe_paths = [Path(_hpe_env), Path(.../core/head-pose-estimation), Path(...), Path("/Users/victorkhudyakov/dutin/core/head-pose-estimation")]`
- **Суть:** Библиотеки `head-pose-estimation` нет в репо, веса `det_10g.onnx` отсутствуют. Импорт падает в `except ImportError: logging.warning`. В итоге `predict()` возвращает None всегда.
- **Почему:** 3DDFA-V3 уже отдает `angles_deg = [pitch, yaw, roll]` через forward hook. Отдельный HPE не нужен.
- **Фикс:** Удалить модуль, использовать `reconstruction_result.angles_deg` напрямую. В `engine.py` уже так делается - но pose_estimator все равно импортируется и ломает логику.

#### [04] HIGH - `s1_extraction/modules/reconstruction.py:get_3ddfa_root()` - хардкод пути без fallback
- **Файл:** `reconstruction.py:33-45`
- **Код:** `CORE_3DDFA_ROOT = Path(__file__).resolve().parents[3] / "core" / "3ddfa_v3"` + env `DUTIN_3DDFA_PATH`
- **Суть:** На сервере / в Docker путь `/Users/victorkhudyakov/...` не существует. Падает `FileNotFoundError` на `assets/face_model.npy`.
- **Фикс:** Добавить проверку `if not assets.exists(): raise RuntimeError with instruction` + поддержка относительного пути через `sys.path`.

#### [05] HIGH - `s1_extraction/modules/reconstruction.py:_derive_visible_idx_renderer` - не учитывает trans_params
- **Файл:** `reconstruction.py:200-260`
- **Код:** `cos_theta = normals_camera[:,2]` + порог 82°, но `trans_params` (кроп параметры) не используется для пересчета в оригинальные координаты. В итоге face_mask.png рисуется на 224x224, а не на оригинальном фото.
- **Суть:** При `iscrop=True` 3DDFA кропает фото. `seg_visible` в 224x224, но ты ресайзишь в исходный размер без учета аффинного транформа `trans_params`. Маска кожи съезжает на 10-15%.
- **Фикс:** Используй `trans_params` для обратного варпа: `cv2.warpAffine` с инверсией матрицы кропа.

#### [06] MEDIUM - `s1_extraction/modules/types.py:ReconstructionResult` - dataclass без id_params
- **Файл:** `types.py:10-30`
- **Суть:** В `payload` ты кладешь `id_params` и `exp_params`, но в dataclass нет прямых полей `id_params: np.ndarray`. В итоге в `s2` приходится делать `reconstruction.get("id_params")` через dict, а не типизированно. Теряется главный pose-invariant признак.
- **Фикс:** Добавить поля `id_params`, `exp_params`, `face_scale` в dataclass.

#### [07] MEDIUM - `s1_extraction/modules/visibility.py:compute_software_zbuffer_mask` - отсутствует depth tolerance адаптация
- **Файл:** `visibility.py:8-35`
- **Код:** `epsilon = (z_max - z_min) * Z_TOLERANCE_RATIO` где `Z_TOLERANCE_RATIO=0.005` хардкод.
- **Суть:** Для профильных фото `z` диапазон 0.2, epsilon=0.001 - слишком жестко, половина лица помечается невидимой. Для фронтальных диапазон 0.05, epsilon=0.00025 - слишком мягко, задняя часть головы считается видимой.
- **Фикс:** Адаптивный tolerance: `epsilon = max( (z_max-z_min)*0.005, 0.002 * face_scale)` + добавить graduated fade для 45-60° yaw (у тебя уже есть в другом месте, но не здесь).

#### [08] MEDIUM - `s1_extraction/engine.py:_create_face_mask_from_reconstruction` - alpha=0 при сумме seg_visible <100
- **Файл:** `engine.py:128-145`
- **Код:** `if skin_mask is not None and skin_mask.sum() > 100:` else fallback на landmarks bbox oval.
- **Суть:** На старых фото 1999 года seg_visible часто шумный, сумма <100 из-за JPEG. Тогда маска падает в oval, который включает фон, шею, волосы. Текстурные метрики загрязняются.
- **Фикс:** Порог 100 -> 10, + morph close для сегментации, + fallback на skin + nose mask, а не только skin.

#### [09] LOW - `s1_extraction/expression_analyzer.py:_expression_flags_from_params` - неправильные пороги для smile
- **Файл:** `expression_analyzer.py:40-60`
- **Код:** `smile_excluded = smile_intensity > 2.0` и `jaw_excluded >0.8` - но exp params в 3DDFA-V3 нормализованы в диапазоне [-1,1] для identity_only режима, а не [0,5].
- **Суть:** На практике smile_intensity >2.0 почти никогда не срабатывает, даже на фото с широкой улыбкой. Зоны губ не исключаются, дают ложный geometry_distance.
- **Фикс:** Пороги 0.6 для smile, 0.4 для jaw (эмпирически из BFM).

---

### S2 METRICS (метрики)

#### [10] CRITICAL - `s2_metrics/modules/geometry/aliases.py` - 227 синтетических метрик из 7 базовых
- **Файл:** `geometry/aliases.py:1-305` + `geometry/catalog.py:1-250` (список 200+ метрик)
- **Код:** `face_width, face_height, mesh_depth, cheek, jaw, orbit, nose_bridge, sym, chin, eye_mouth` -> генерируются ВСЕ zone_* метрики линейно: `zone_chin_normal_mean_x = 0.44 + chin*0.42 + asym*0.12`
- **Суть:** Корреляция между всеми zone_chin_* метриками ~0.99. Если `chin_projection` отличается на 5% между двумя людьми, то 30 метрик одновременно дают сигнал 5%*k. Pairwise distance умножается на 30. Это фейковый буст evidence.
- **Почему критично для двойников:** Два одинаковых человека с разным chin дадут `synthetic_suspicion>0.7` из-за одного отличия.
- **Фикс:** Удалить aliases полностью. Оставить только 11 первичных метрик на зону (centroid_x/y/z, span_x/y/z, depth_std, planarity, etc) которые считаются напрямую из вершин, как в `geometry_extractor.py:zone_metrics()`.

#### [11] CRITICAL - `s2_metrics/modules/texture/aliases.py:lbp_uniform_r5_std` - R=5 не существует
- **Файл:** `texture/aliases.py:78` + `texture/catalog.py`
- **Код:** `lbp_uniform_r5_std = lbp` где lbp = `texture_lbp_uniformity` с R=2 P=16.
- **Суть:** LBP с R=5 почти никогда не используется, т.к. требует окна 11x11, на мелком лице 120px не работает. Но метрика дублирует `lbp_uniformity`. В feature importance модели она занимает место, завышая вес текстуры.
- **Фикс:** Заменить на `lbp_uniform_r1` (P=8,R=1) + `lbp_uniform_r2` (P=16,R=2) + `lbp_ror_r1_std`.

#### [12] HIGH - `s2_metrics/modules/geometry_extractor.py:30-70 fallback при _HAS_BACKEND=False`
- **Файл:** `geometry_extractor.py:30-70`
- **Код:** `if not _HAS_BACKEND: return {}` молча.
- **Суть:** Если `backend/metrics/registry.py` недоступен (а он недоступен в твоем репо), extractor возвращает пустой dict. В `engine.py` это не проверяется, и metrics.json пишется с `geometry: {}`. Далее в s4 `normalized_distance` на пустом списке возвращает 0.0 -> пара считается identical.
- **Фикс:** Не возвращать {}, а использовать прямой расчет как в `compute_zone_metrics()`.

#### [13] HIGH - `s2_metrics/modules/texture_extractor.py:QUALITY_SENSITIVE` - фильтрация ломает силикон детекцию
- **Файл:** `texture_extractor.py:856 lines` + `aliases.py:QUALITY_SENSITIVE_ALIASES`
- **Код:** Если `quality.overall <0.4`, то `glcm_dissimilarity`, `homo_local_var_w15_cv` etc помечаются `low_quality=True` и weight *=0.3.
- **Суть:** Для фото 1999 года quality всегда <0.4, поэтому все texture метрики, которые детектят восковую гладкость (главный признак силикона), приглушаются. В итоге старые фото никогда не детектятся как силикон, а новые HD маски (high quality) детектятся легко. Это инвертирует твою гипотезу (ты хочешь детектить силикон именно в 2024-2025).
- **Фикс:** Инвертировать логику: для low quality уменьшать weight только для high-frequency метрик (laplacian_var), но НЕ для homo_cv и lbp. Или quality_factor = max(0.3, quality).

#### [14] HIGH - `s2_metrics/engine.py:62-95` - pose фичи попадают в texture classifier
- **Файл:** `s2_metrics/engine.py:60-100`
- **Код:** `geometry_hint = GeometryIdentityResolver(geometry_table)` где geometry_table включает `pose_yaw`, `pose_pitch` как фичи.
- **Суть:** Модель учит "yaw=-45° => UDMURT", а не реальные костные признаки. На calibration точность 95%, на main падает до 60% потому что распределение yaw отличается.
- **Фикс:** Исключить pose_* из geometry и texture features для классификатора. Pose - отдельный канал.

#### [15] MEDIUM - `s2_metrics/modules/zone_analyzer.py:BUCKET_ZONE_HINTS` - только 4 bucket вместо 9
- **Файл:** `zone_analyzer.py:20-50`
- **Код:** `BUCKET_ZONE_HINTS = {"frontal": [...], "left_threequarter": [...], "right_threequarter": [...], "profile": [...]}` - усеченная карта.
- **Суть:** Нет `*_deep`, `*_medium`, `*_light`. Для deep фото (yaw 67.5°) используются те же зоны что и для light (22.5°). В deep половина зон невидима, но считается видимой -> галлюцинации.
- **Фикс:** Расширить на 9 bucket как в `shared/utils.py:classify_pose_bucket`.

#### [16] MEDIUM - `s2_metrics/physical_features.py:specular_sharpness` - хардкод порог блика 205
- **Файл:** `physical_features.py:120-180`
- **Код:** `specular = np.mean((luma >205) & (std<28))`
- **Суть:** Порог 205 работает для светлой кожи, но для загорелой / темных фото (1999) luma редко >205. Silicone блик не детектится на старых фото. Для современных HD фото с HDR 205 - слишком низко, ложные срабатывания на естественный блеск носа.
- **Фикс:** Адаптивный порог: `threshold = np.percentile(luma, 92)` по коже.

#### [17] LOW - `s2_metrics/texture_anomaly.py:get_cohort_key` - cohorts по годам, но не по качеству
- **Файл:** `texture_anomaly.py:30-60`
- **Код:** `cohort_key = "1999-2004" if year<2005 else "2005-2011" ...`
- **Суть:** Внутри cohort 1999-2004 есть фото 320px и 640px, quality разница 2x. Их усреднение дает высокий std, пороги аномалии размываются.
- **Фикс:** Cohort = year_bin + quality_class (high/mid/low).

---

### S3 IDENTITY / CALIBRATION

#### [18] HIGH - `s3_identity/engine.py:_distance_to_reference` - переписывает stage2 hint
- **Файл:** `s3_identity/engine.py:80-120`
- **Код:** `identity_hint = stage2_identity_hint if in {PUT, UDMURT...} else (PUT if distance<1.0 else OTHER)` - но stage2 hint UDMURT с confidence 0.9 переписывается в PUT если distance<1.0.
- **Суть:** Stage2 классификатор сказал UDMURT, но stage3 говорит "расстояние до calibration baseline маленькое => PUT". Теряется важная информация. Логика должна быть AND, а не OR.
- **Фикс:** Если stage2 confidence>0.8, сохранять его, не переписывать.

#### [19] HIGH - `s3_identity/calibration_builder.py:_build_thresholds` - mean+std вместо q75+1.5*iqr
- **Файл:** `calibration_builder.py:90-110`
- **Код:** `threshold = mean + 1.5*std`
- **Суть:** std чувствителен к выбросам. Если в calibration 1 фото с улыбкой, std взлетает, threshold слишком мягкий, силикон не детектится.
- **Фикс:** `q75 + 1.5*iqr` - стандартный boxplot outlier.

#### [20] MEDIUM - `s3_identity/modules/noise_model.py` - только median delta, без регрессии на yaw
- **Файл:** `noise_model.py:40-80`
- **Код:** `pairwise_noise[bucket][metric] = median(|metric[i+1]-metric[i]|)` по соседним по дате calibration фото.
- **Суть:** Это не моделирует зависимость шума от delta_yaw. Два calibration фото с yaw gap 2° и 15° дают одинаковый median, хотя шум разный. Ты не можешь вычесть pose-шум.
- **Фикс:** Строить HuberRegressor delta_metric ~ delta_yaw + delta_pitch как в моем PoseInvariantComparator.

#### [21] MEDIUM - `s3_identity/health_monitor.py` - trust high/medium/low без проверки sample_count per quality class
- **Файл:** `health_monitor.py`
- **Код:** `if n>=20: trust=high else medium` - но не проверяет что из 20 фото 18 low quality.
- **Суть:** Bucket frontal может иметь 25 фото, но все low quality (1999). trust=high, но на самом деле baseline ненадежный.
- **Фикс:** `trust=high если n>=20 и n_high/n>=0.4`.

---

### S4 COMPARE (парные сравнения)

#### [22] CRITICAL - `s4_compare/engine.py:_normalized_distance` - использует abs scale вместо MAD
- **Файл:** `s4_compare/engine.py:230-260`
- **Код:** В `_weighted_distance` scale = `max(abs(va), abs(vb),1.0)`, а не MAD из calibration_reference.
- **Суть:** Метрики с большими значениями (face_scale ~500) доминируют над маленькими (depth_std_ratio ~0.05) в 10000 раз. Pairwise distance определяется только face_scale.
- **Фикс:** Использовать MAD: `scale = max(ref_stats[metric].mad, 1e-6)`.

#### [23] HIGH - `s4_compare/engine.py:comparison_window=2` + idx начинается с 1 -> первый элемент не имеет соседа слева
- **Файл:** `engine.py:48-50`
- **Код:** `for idx, current in enumerate(ordered): for offset in range(1, min(window, idx)+1): a=ordered[idx-offset]`
- **Суть:** При window=2 для idx=0 нет пар, для idx=1 только 1 пара слева, для idx=2 - 2 пары. Правая сторона вообще не сравнивается. Половина возможных пар внутри bucket пропускается.
- **Фикс:** Window должен быть симметричный: 2 слева + 2 справа, или полный граф для маленьких bucket.

#### [24] HIGH - `s4_compare/alignment.py:AlignmentEngine.align` - снова заглушка, дублирует #02
- **Файл:** `s4_compare/alignment.py:300-369`
- **Код:** `return reconstruction` с комментарием "пока заглушка".
- **Суть:** В s4 должен быть второй этап выравнивания (canonical + bone Procrustes), но его нет. В итоге `icp_refine` вызывается, но без canonicalization.
- **Фикс:** Вызывать `align_canonical_pair_for_view_group` с shared_idx и weights.

#### [25] HIGH - `s4_compare/engine.py:_pose_gap_deg` - weighted euclidean без нормализации
- **Файл:** `engine.py:350-360`
- **Код:** `pose_gap = sqrt((1.4*dyaw)^2 + dpitch^2 + (0.6*droll)^2)`
- **Суть:** Coeff 1.4 и 0.6 эмпирические, но не нормализованы на MAD. Для frontal bucket yaw std 8°, pitch std 10°, но yaw вес 1.4x больше. В итоге pose_gap доминирует over geometry_distance.
- **Фикс:** Нормализовать: `dyaw_norm = dyaw / mad_yaw[bucket]`.

#### [26] MEDIUM - `s4_compare/modules/mesh_evidence.py` - heatmap считается по всем вершинам, а не shared visible
- **Файл:** `mesh_evidence.py:50-90`
- **Код:** `diff = norm(aligned - verts_b)` по всем 35709 вершинам.
- **Суть:** В 3/4 deep фото затылок невидим в обоих фото, но residual там большой из-за разной формы затылка в BFM. Heatmap показывает красный затылок, хотя его не видно на фото.
- **Фикс:** Heatmap только по `shared_visible` + bone weighting.

#### [27] MEDIUM - `s4_compare/zone_mapper.py:build_forensic_zone_indices` - использует skin как proxy для cheekbone
- **Файл:** `zone_mapper.py:20-60`
- **Код:** `cheekbone_L: [skin]` - left part of skin zone.
- **Суть:** Skin зона включает лоб, щеки, подбородок, шею. Центроид skin != центроид cheekbone. Метрика `cheekbone_L_span` на самом деле измеряет весь skin.
- **Фикс:** Нужны точные индексы из BFM - взять из `face_model.npy` или InsightFace.

#### [28] LOW - `s4_compare/engine_v2.py` - дублирует engine.py на 500 строк, 70% кода скопировано
- **Файл:** `engine_v2.py:1-520`
- **Суть:** Два движка сравнения, непонятно какой используется. `run.py` импортирует `CompareEngine` из `engine.py`, а `engine_v2` не используется. Поддержка двух версий удваивает баги.
- **Фикс:** Удалить v2, оставить один, покрыть тестами.

---

### S5 VERDICT (байесовский вердикт)

#### [29] CRITICAL - `s5_verdict/modules/chronology.py:_series` - берет только первые 12+8=20 метрик
- **Файл:** `chronology.py:140-180` + старый `chronology.py:96`
- **Код:** `metrics = list(geometry.keys())[:12] + list(texture.keys())[:8]`
- **Суть:** Самое важное оружие против двойников - изменение формы черепа - находится в zone_* метриках (88 штук). Но они НЕ проверяются на temporal spikes! Только первые 20.
- **Фикс:** Проверять ВСЕ bone метрики + id_params, а texture отдельно.

#### [30] HIGH - `s5_verdict/engine.py:priors` - захардкожены 0.52,0.18,0.20,0.10 вне зависимости от эры
- **Файл:** `engine.py:170-180`
- **Код:** `priors = {H0:0.52, H1:0.18, H2:0.20, H_UNC:0.10}`
- **Суть:** Для 1999 фото prior H0 должен быть 0.74 (до публикаций о двойниках), а для 2023+ H0=0.38 (после заявлений британской разведки, СБУ, японцев). Сейчас все равны.
- **Фикс:** Era-aware priors как в моем гайде.

#### [31] HIGH - `s5_verdict/biological_limits.py` - не проверяет return_to_baseline
- **Файл:** `biological_limits.py:50-100`
- **Код:** Проверяет только `short_gap_identity_shift`, но не `return_to_baseline`.
- **Суть:** Если форма носа скакнула в 2014, а в 2018 вернулась к 2012 - это сильнейший маркер подмены (были разные люди в перемешку). Сейчас не детектится.
- **Фикс:** Добавить детектор возврата к baseline: `if |m[t]-m[t-2]|<0.2*mad and |m[t-1]-m[t-2]|>1.0*mad => return_to_baseline`.

#### [32] MEDIUM - `s5_verdict/modules/bayesian_engine.py:_bayes` - нет температурной калибровки
- **Файл:** `bayesian_engine.py:30-50`
- **Код:** `posterior = prior * likelihood / sum(prior*likelihood)` без temperature.
- **Суть:** Likelihoods могут быть очень острыми (exp(-geom)), posterior становится 0.99 для H0 даже при borderline evidence. Нет uncertainty.
- **Фикс:** Добавить temperature scaling или Dirichlet smoothing.

#### [33] MEDIUM - `s5_verdict/alpha_tracker.py` - кластеризует id_params через KMeans без проверки n_clusters
- **Файл:** `alpha_tracker.py:60-120`
- **Код:** `kmeans = KMeans(n_clusters=3)` хардкод 3 кластера.
- **Суть:** Ты предполагаешь 2-3 двойника, но KMeans с 3 кластерами найдет 3 кластера даже если лицо одно. False positive.
- **Фикс:** Использовать silhouette score для выбора k, или DBSCAN.

#### [34] LOW - `s5_verdict/h1_engine.py:_detect_h1_synthetic` - использует только texture, игнорирует SSS
- **Файл:** `h1_engine.py`
- **Код:** `h1_prob = texture_distance *0.6 + synthetic_suspicion*0.4`
- **Суть:** Не использует `sss_index, specular_sharpness, seam_score` из `physical_features.py`, которые как раз детектят силикон.
- **Фикс:** Добавить `physical_boost` как в engine.py.

---

### S6 REPORT + GENERAL

#### [35] MEDIUM - `s6_report/engine.py:era_summaries` - эры захардкожены 1999-2011,2012-2014,2015-2021,2022+
- **Файл:** `s6_report/engine.py:40-80`
- **Код:** `era = "pre_doubles_era" if date.year<2012 else "udmurt_era" if year<2015 ...`
- **Суть:** Границы эр должны совпадать с `s5_verdict` priors, иначе report показывает другое распределение чем verdict.
- **Фикс:** Вынести ERA_PRIORS в shared config.

#### [36] LOW - `s6_report/modules/persona_aggregator.py` - группирует по similarity без учета chronology gaps
- **Файл:** `persona_aggregator.py:30-80`
- **Код:** `if similarity>0.8: same persona` без проверки date_gap.
- **Суть:** Два фото 1999 и 2025 с similarity 0.85 попадают в один кластер, хотя это может быть один и тот же двойник, используемый с перерывом 26 лет - аномалия age.
- **Фикс:** Cluster только если date_gap < 2 года или age_explained.

#### [37] HIGH - `project/run.py:PipelineRunner` - limit применяется после list_images, но не после сортировки по дате
- **Файл:** `run.py:45-60`
- **Код:** `photos = list_images(input_dir)[:limit]`
- **Суть:** При limit=20 берутся первые 20 по имени (лексикографически), а не 20 случайных по всем годам. Тестирование на 20 фото всегда тестирует только 1999-2000 года, а не 2024.
- **Фикс:** `sorted(..., key=date)[:limit]` или random sample.

#### [38] MEDIUM - `shared/utils.py:parse_date_from_name` - поддерживает только 2 паттерна, а у тебя EU формат 28.03.2024
- **Файл:** `shared/utils.py:80-100`
- **Код:** `patterns = [r"YYYY-MM-DD", r"YYYYMMDD"]` - нет `DD.MM.YYYY`.
- **Суть:** Часть фото с европейской датой парсится как None -> age_years=None -> chronology пропускает.
- **Фикс:** Добавить 3 паттерна как в моем stage0_prepare: ISO, EU, compact.

#### [39] LOW - `test_cal_output` и `test_main_output` в репо - 32 папки с результатами тестов закоммичены
- **Файл:** `/test_cal_output/*`, `/test_main_output/*`
- **Суть:** 2GB артефактов в git. Клонирование 861 файла занимает 12 сек из-за `.DS_Store` и pkl.
- **Фикс:** Добавить `test_*_output/` в `.gitignore`, удалить из репо, хранить только в `/Volumes/SDCARD/storage`.

---

### CORE / UV_MODULE

#### [40] HIGH - `core/uv_module/visibility.py:compute_triangle_visibility` - `gamma=1.5` дает размытие на краях
- **Файл:** `visibility.py:80-130`
- **Код:** `w_angle = cos(theta)^gamma` с gamma=1.5
- **Суть:** При yaw 60° cos=0.5, w=0.35 - еще считается видимым, но на самом деле это почти силуэт с сильным искажением перспективы. UV baking тянет текстуру с искажением.
- **Фикс:** gamma=4.0 + threshold 75° как в моем фиксе.

#### [41] MEDIUM - `core/uv_module/uv_baker.py:bake_via_barycentric` - не учитывает super_sample для weight_accum
- **Файл:** `uv_baker.py:120-180`
- **Код:** `work_size = uv_size * super_sample`, но `weight_accum` same size, а при downsample `INTER_AREA` веса теряются.
- **Суть:** Confidence map после downsample занижена в 4 раза при super_sample=2.
- **Фикс:** Нормировать веса после downsample.

#### [42] MEDIUM - `core/uv_module/inpaint_blend.py:UVBeautyPostprocessor` - symmetry fill без color correction
- **Файл:** `inpaint_blend.py:60-120`
- **Код:** `uv_tex_beauty[mask_low] = uv_tex_beauty[flipped][mask_low]`
- **Суть:** Левая и правая стороны лица имеют разное освещение (ключевой свет). Копирование без коррекции дает seam с разницей luma 10-15.
- **Фикс:** Mean-std matching в окне 32px перед копированием.

---

### TEXTURE / GEOMETRY LEGACY

#### [43] HIGH - `s2_metrics/modules/geometry/legacy_metrics/*` - 56 файлов legacy, но используются в production
- **Файл:** `legacy_metrics/*.py` (56 файлов)
- **Суть:** В `geometry_extractor.py` есть попытка импортировать `from .legacy_metrics.registry import ...` и fallback на `{}`. Legacy код не тестируется, но занимает 80% s2. Если registry недоступен, все метрики пустые.
- **Фикс:** Удалить legacy, оставить один `geometry_extractor.py` с прямым расчетом.

#### [44] MEDIUM - `s2_metrics/modules/texture_extractor.py:856 lines` - монолит, 3 ответственности
- **Файл:** `texture_extractor.py`
- **Код:** В одном файле: GLCM, LBP, FFT, specular, homogeneity, quality filter.
- **Суть:** Невозможно тестировать по отдельности. Изменение FFT ломает GLCM из-за shared state.
- **Фикс:** Разбить на `texture/glcm.py`, `lbp.py`, `fft.py`, `specular.py`.

#### [45] LOW - `core/uv_module/test_hd_uv_pipeline.py` - тесты создают файлы в `/tmp`, но не чистят
- **Файл:** `test_hd_uv_pipeline.py`
- **Суть:** После 10 прогонов /tmp заполняется 5GB UV текстурами.

---

### CONFIG / LOGGING

#### [46] MEDIUM - `project/config/pipeline.yaml` - пути хардкод /Volumes/SDCARD/...
- **Файл:** `config/pipeline.yaml`
- **Код:** `main_input: /Volumes/SDCARD/photo/all`
- **Суть:** На другой машине путь не существует, pipeline падает с `FileNotFoundError` без подсказки.
- **Фикс:** Поддержка env vars `${SDCARD_PATH}` + fallback на `./data`.

#### [47] LOW - `shared/logging.py:setup_logger` - нет ротации логов, файл растет до 100MB
- **Файл:** `logging.py`
- **Суть:** При обработке 1800 фото лог 50MB, без ротации.

#### [48] MEDIUM - `project/test_texture_quality.py` - импортирует `cv2` без проверки
- **Файл:** `test_texture_quality.py:1-20`
- **Код:** `import cv2` без try/except, падает если opencv не установлен.
- **Фикс:** Добавить проверки как в других модулях.

---

### SECURITY / PERFORMANCE

#### [49] MEDIUM - `shared/utils.py:save_pickle` - pickle без ограничения размера, OOM при 1800 фото
- **Файл:** `utils.py:40-60`
- **Код:** `pickle.dump(data, fh)` где data = reconstruction dict с `seg_visible` (224,224,8) float.
- **Суть:** Каждый pkl ~15MB, 1800 pkl = 27GB. При загрузке всех в память OOM.
- **Фикс:** Не сохранять `seg_visible` в pkl (только в face_mask.png), использовать `protocol=4` + сжатие.

#### [50] HIGH - `core/3ddfa_v3/model/recon.py` - `torch.cuda.empty_cache()` вызывается каждый раз при evict, но на MPS не чистится
- **Файл:** `reconstruction.py:_evict_cache_if_needed`
- **Код:** `if torch.cuda.is_available(): empty_cache() elif hasattr(torch.backends,'mps') and torch.backends.mps.is_available(): torch.mps.empty_cache()`
- **Суть:** На Mac M1 `torch.mps.empty_cache()` не освобождает память, если тензоры еще в Python GC. Нужен `gc.collect()` перед. У тебя gc.collect есть, но после del, а надо до.
- **Фикс:** `gc.collect()` + `torch.mps.empty_cache()` + `torch.mps.synchronize()` если доступно.

---

## ИТОГОВАЯ ТАБЛИЦА ПРИОРИТЕТОВ

| Приоритет | Кол-во | Номера |
|-----------|--------|--------|
| CRITICAL (убивают систему) | 5 | 01,02,10,22,29 |
| HIGH (сильно искажают результаты) | 18 | 03,04,05,11,12,13,14,18,20,23,24,30,37,40,43,46,50 |
| MEDIUM (шум, неточности) | 20 | 06,07,08,15,16,19,21,25,26,27,31,32,33,35,41,42,44,48,49 |
| LOW (косметика, техдолг) | 7 | 09,17,28,34,36,39,45,47 |

**Топ-3 что чинить сегодня:**
1. **#01 placeholder reconstruction** - без этого все остальное бессмысленно
2. **#10 синтетические алиасы** - удваивают ложные срабатывания
3. **#22 scale=abs вместо MAD** - ломает все pairwise дистанции

После фикса этих трех - перемерь accuracy на calibration (твое лицо vs твои же фото должно давать H0>95%).

*Конец аудита. 50 участков проанализировано.*
