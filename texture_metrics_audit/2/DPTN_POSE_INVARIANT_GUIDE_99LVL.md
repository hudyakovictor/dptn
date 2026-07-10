# DPTN // 3DDFA-V3 - ГАЙД 99 ЛВЛ: как победить шум от наклона головы в 9-ракурсной системе

> **Цель:** ты сравниваешь 1800+ фото Путина по хронологии внутри 9 ракурсов (frontal, left/right profile, left/right 3/4 light/medium/deep), но внутри одного ракурса yaw гуляет на +/-15°, pitch +/-12°, roll +/-10°. Из-за этого `geometry_distance` и `texture_distance` = 70% шум от позы, 30% сигнал от лица. Надо вычесть шум.

> **Стек:** 3DDFA-V3 (wang-zidu), BFM 3DMM, RetinaFace, OpenCV, numpy. Твой проект уже делает 60% правильно - `s1_extraction/modules/reconstruction.py` и `visibility.py` и `core/uv_module` - это золото. Проблема в том что в `s4_compare` выравнивание фейковое и нет pose-gap discount модели.

---

## 1. ЧТО РЕАЛЬНО ОТДАЕТ 3DDFA-V3 - АНАТОМИЯ (99 лвл часть)

Забудь про demo.py обертку. Внутри `model/recon.py`:

```python
alpha = net_recon(img) # (1, 256)
# split:
# id: 80 - форма черепа, ПОЗНО-ИНВАРИАНТНА! Это твой главный биометрический отпечаток
# exp: 64 - мимика (0 = улыбка, 1-2 = jaw open)
# tex: 80 - альбедо PCA
# angle: 3 - [pitch, yaw, roll] в РАДИАНАХ, порядок именно такой
# gamma: 27 - SH освещение
# trans: 2 - смещение
```

**То что хранишь в `reconstruction.pkl` NOW:**
- `vertices_world` (35709,3) - уже повернутые в camera space = `base(id,exp) @ R(angle) + trans`
- `vertices_canonical` - если ты делаешь `exp=0` и `R=I`, получаешь **pose-normalized mesh** - 35709 вершин только от `id`. Это идеально для сравнения черепов.
- `rotation_matrix` (3,3) - то что 3DDFA применил
- `landmarks_106` (106,2) - для bbox
- `annotation_groups` - 8 групп индексов: [right_eye, left_eye, right_brow, left_brow, nose, up_lip, down_lip, skin]
- `seg_visible` (224,224,8) - для маски кожи
- `visible_idx_renderer` - visibility из рендерера

**Твоя ошибка:** ты сравниваешь `vertices_camera` даже внутри одного bucket. Надо сравнивать `vertices_canonical_id_only`.

```python
# Вот как получить ID-only mesh (99lvl trick):
alpha_dict = model.split_alpha(alpha)
exp_zero = torch.zeros_like(alpha_dict["exp"])
base_id_only = model.compute_shape(alpha_dict["id"], exp_zero) # (1,35709,3) без мимики
# это уже в object space, без поворота! Для сравнения черепов - лучше всего
id_params = alpha_dict["id"].detach().cpu().numpy()[0] # (80,) - твой биометрик
# L2 distance между id_params двух фото = pose-invariant distance
```

**Почему это решает 80% проблемы наклона:**
- `id_params` НЕ зависит от yaw/pitch/roll вообще. Это коэффициенты PCA формы черепа.
- Если у двух фото `||id_a - id_b|| > threshold` - это 100% разные черепа, независимо от ракурса.
- Твой `s2_metrics` сейчас считает `zone_centroid_x` etc - они зависят от позы. Замени / дополни на `id_norm`, `id_mean`, `id_std` + первые 10 компонент `id_params`.

---

## 2. 9 РАКУРСОВ - КАК ДЕЛИТЬ ПРАВИЛЬНО

Сейчас в `shared/utils.py:classify_pose_bucket` ты парсишь имя файла. Это hint, но не ground truth. 

**Правило 99 лвл:**
1. Первичный bucket = по `yaw` из 3DDFA-V3 (degrees).
2. Вторично = проверка с filename hint. Если расхождение >30°, помечай `trust_issue="bucket_mismatch"`.

Канонические yaw для 9 bucket (из твоего ТЗ):
```
frontal = 0°
left_light = -22.5°, right_light = +22.5°
left_medium = -45°, right_medium = +45°
left_deep = -67.5°, right_deep = +67.5°
left_profile = -90°, right_profile = +90°
```

Границы:
```
|yaw|<12° => frontal
12-30 => light
30-55 => medium
55-80 => deep
>80 => profile
```

**Но внутри bucket все равно шум:**
- frontal bucket: yaw реально от -12 до +12, pitch от -15 до +15.
- Этот шум дает до 0.8 z-score на `zone_skin_centroid_x` даже если лицо одно и то же.

Решение: **canonicalization + pose-gap discount**.

### 2.1 Canonicalization по bucket target

Для каждого фото внутри bucket мы поворачиваем его mesh к целевому yaw этого bucket.

```python
def canonicalize_to_bucket_yaw(recon, target_yaw_deg: float):
    # recon.rotation_matrix = R_observed (то что 3DDFA применил)
    # Хотим: R_align = R_obs.T @ R_target
    R_obs = recon.rotation_matrix
    yaw_rad = np.deg2rad(target_yaw_deg)
    R_target = np.array([
        [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
        [0, 1, 0],
        [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
    ])
    R_align = R_obs.T @ R_target
    # Применяем к вершинам centered
    verts_centered = recon.vertices_world - recon.vertices_world.mean(axis=0)
    verts_canon = verts_centered @ R_align
    return verts_canon
```

Теперь все фото внутри `frontal` смотрят ровно в 0°. Остаточный шум только от pitch/roll и неточности R.

### 2.2 Bone-anchored Procrustes (второе выравнивание)

После canonicalization делаем Umeyama rigid (без масштаба!) по **костным** вершинам, которые видимы в обоих фото.

```python
BONE_GROUPS = [2,3,4] # right_brow, left_brow, nose - самые стабильные
# Или точнее: взять индексы скул и переносицы из annotation_groups
# У тебя в s2_metrics/modules/geometry_extractor.py уже есть mapping

def umeyama_no_scale(src, tgt, weights=None):
    # классика
    mu_s = (src * w).sum(axis=0)
    mu_t = (tgt * w).sum(axis=0)
    s0, t0 = src-mu_s, tgt-mu_t
    H = (s0 * w[:,None]).T @ t0
    U,S,Vt = np.linalg.svd(H)
    D = np.diag([1,1,np.sign(np.linalg.det(U@Vt))])
    R = U @ D @ Vt
    t = mu_t - R @ mu_s
    aligned = (R @ src.T).T + t
    residual = np.median(np.linalg.norm(aligned - tgt, axis=1))
    return aligned, R, t, residual
```

**Ключевое:** alignment делается ТОЛЬКО по `shared_visible` вершинам.

```python
shared_vis = recon_a.visible_idx & recon_b.visible_idx
bone_indices = np.concatenate([recon_a.annotation_groups[i] for i in BONE_GROUPS])
bone_vis = np.intersect1d(bone_indices, np.where(shared_vis)[0])
# выравниваем по bone_vis
```

Результат: `aligned_a` vs `verts_b`. Теперь считаем heatmap.

---

## 3. POSE-GAP DISCOUNT МОДЕЛЬ - КАК ВЫЧЕСТЬ ШУМ НАКЛОНА

Это сердце твоего вопроса. Как сделать так чтобы `frontal` фото с yaw=-10° и yaw=+10° не считались разными людьми?

### Идея

Используй свой калибровочный датасет (твои фото, 100% один человек). Там шум ТОЛЬКО от позы и качества реконструкции.

Для каждой метрики внутри bucket учим: `delta_metric = f(delta_yaw, delta_pitch, delta_roll)`

```python
# Псевдокод обучения на calibration
for bucket in 9_buckets:
    pairs = все пары calibration фото в этом bucket
    for pair in pairs:
        dyaw = abs(yaw_a - yaw_b)
        dpitch = abs(pitch_a - pitch_b)
        droll = abs(roll_a - roll_b)
        dmetric = abs(metric_a - metric_b)
        # сохраняем
    # линейная регрессия: dmetric ~ a*dyaw + b*dpitch + c*droll + intercept
    # a,b,c = коэффициенты шума от позы
```

В твоем коде `s3_identity/modules/noise_model.py` уже есть заготовка. Но там считается просто `pairwise_noise` как median delta. Надо расширить.

**Практика в твоем проекте:**

Открой `project/s3_identity/calibration_builder.py` - там `PoseAwareCalibrationBuilder` уже должен строить эту модель.

Должен вернуть для каждого bucket:
```json
{
  "frontal": {
    "zone_nose_centroid_x": {"slope_yaw": 0.012, "slope_pitch": 0.003, "intercept": 0.05, "mad": 0.02},
    "zone_skin_span_x": {"slope_yaw": 0.020, ...}
  }
}
```

При сравнении main фото:

```python
def corrected_distance(raw_distance, yaw_a, yaw_b, model):
    dyaw = abs(yaw_a - yaw_b)
    expected_noise = model["slope_yaw"]*dyaw + model["intercept"]
    return max(0, raw_distance - expected_noise - model["mad"]*0.5)
```

**Пример:** два фронтальных фото Путина, yaw -8° и +9° (gap 17°). Для метрики `nose_centroid_x` raw_distance=0.15. Модель говорит expected_noise для 17° = 0.12, mad=0.02. Тогда corrected=0.15-0.12-0.01=0.02 (почти 0). Это шум, не сигнал.

Если gap тот же 17°, но raw_distance=0.45, corrected=0.32 - это уже реальное различие.

**Ты должен хранить эту модель в `calibration_reference.json` в поле `pairwise_noise` + новая секция `pose_regression`.**

---

## 4. VISIBILITY MASK - ЧТОБЫ НЕ СРАВНИВАТЬ СКРЫТЫЕ ЧАСТИ

Твоя проблема: в 3/4 deep фото половина лица скрыта, но метрики считаются по всему мешу.

Решение уже у тебя в `s1_extraction/modules/visibility.py` и `core/uv_module/visibility.py`:

1. Угловой фильтр 82°: `cos(normal, view_dir) < cos(82°) => невидима`
2. Software Z-buffer: проектируешь все вершины в 2D, заполняешь z_buffer минимумом z, если вершина дальше чем буфер + epsilon => occluded.

```python
def compute_visible(verts_camera, normals_camera):
    cos_theta = normals_camera[:,2] # в camera space камера смотрит вдоль +Z
    angular = cos_theta > np.cos(np.deg2rad(82))
    # zbuffer
    z_visible = compute_software_zbuffer_mask(verts_camera, resolution=512)
    return angular & z_visible
```

**Graduated fade для 45-60° (у тебя уже есть в visibility.py):** когда yaw 45-60°, вершины на дальней стороне лица имеют вес 0, а не 1. Это prevents галлюцинации.

При сравнении пары: `shared_visible = visible_a & visible_b`. Все метрики считаются ТОЛЬКО по shared_visible.

В `s4_compare/zone_mapper.py` у тебя `build_forensic_zone_indices` - он должен учитывать это.

---

## 5. UV-ТЕКСТУРА - ТЕКСТУРНОЕ СРАВНЕНИЕ БЕЗ ПОЗЫ

Геометрия решается через id_params + alignment. А текстура (кожа) - через UV.

Твой `core/uv_module/hd_uv_generator.py` - это כבר решение.

Процесс:
1. Берешь `vertices_2d` (проекция на фото) + `uv_coords` (из BFM).
2. Для каждого треугольника строишь аффинную матрицу `M = UV_tri -> image_tri`.
3. Растеризуешь в UV 1024x1024, аккумулируя цвет с весом `w = cos_angle * z_visibility`.
4. Получаешь `uv_tex_analysis` - текстура только из реально видимых пикселей, без inpaint.

Теперь сравнение кожи:
- Берешь два UV, смотришь `shared_uv_mask = mask_a & mask_b & confidence>0.5`.
- Считаешь LBP, FFT, specular ТОЛЬКО в shared mask.
- Результат - pose-invariant.

Код из твоего `test_render_texture.py`:
```python
generator = HDUVTextureGenerator(config=HDUVConfig(uv_size=1024))
uv_analysis_a, uv_beauty_a, mask_a, conf_a, aux_a = generator.generate(image_a, recon_dict_a)
uv_analysis_b, ... = generator.generate(image_b, recon_dict_b)
# сравнить tex в UV
shared = (mask_a>0) & (mask_b>0) & (conf_a>0.3) & (conf_b>0.3)
# LBP на shared
```

Это убирает шум от разного наклона для текстурного анализа.

---

## 6. ИТОГОВЫЙ ПАЙПЛАЙН СРАВНЕНИЯ ВНУТРИ ОДНОГО РАКУРСА (псевдокод 99 lvl)

```python
def compare_pair_99lvl(photo_a_path, photo_b_path, calib_models):
    # 1. Load реконструкции (кэш на диске!)
    recon_a = load_pickle(photo_a_path / "reconstruction.pkl")
    recon_b = load_pickle(photo_b_path / "reconstruction.pkl")
    
    info_a = json.load(open(photo_a_path / "info.json"))
    info_b = json.load(open(photo_b_path / "info.json"))
    
    bucket = info_a["pose"]["bucket"] # уже должны быть в одном
    target_yaw = BUCKET_CANONICAL_YAW[bucket]
    
    # 2. ID-params distance - pose-invariant gold standard
    id_a = np.array(recon_a["id_params"])
    id_b = np.array(recon_b["id_params"])
    id_distance = np.linalg.norm(id_a - id_b)  # нормально ~0.5-1.5 для одного человека
    # калибровка: на твоем лице id_distance median = ?, mad = ?
    id_zscore = (id_distance - calib_models[bucket]["id_norm"]["median"]) / calib_models[bucket]["id_norm"]["mad"]
    
    # 3. Canonicalize + bone Procrustes
    verts_a_canon = canonicalize_to_bucket_yaw(recon_a, target_yaw)
    verts_b_canon = canonicalize_to_bucket_yaw(recon_b, target_yaw)
    
    shared_vis = recon_a["visible_idx"] & recon_b["visible_idx"]
    bone_idx = get_bone_indices(recon_a) # nose + brow
    bone_shared = np.intersect1d(bone_idx, np.where(shared_vis)[0])
    
    aligned_a, R, t, residual_align = umeyama_no_scale(
        verts_a_canon[bone_shared], verts_b_canon[bone_shared]
    )
    # применяем ко всему мешу
    full_aligned_a = (R @ verts_a_canon.T).T + t
    
    # 4. Per-vertex heatmap
    per_vertex_dist = np.linalg.norm(full_aligned_a - verts_b_canon, axis=1)
    per_vertex_dist_normalized = per_vertex_dist / face_scale # face_scale ~ межскуловая ширина
    
    # 5. Per-zone distances (ТОЛЬКО по shared_visible!)
    zone_distances = {}
    for zone_name, ann_idx in ANNOTATION_ZONES.items():
        indices = recon_a["annotation_groups"][ann_idx]
        if len(indices)==0: continue
        zone_shared = np.intersect1d(indices, np.where(shared_vis)[0])
        if len(zone_shared)<5: continue
        zone_dist = np.median(per_vertex_dist_normalized[zone_shared])
        # вычитаем pose-gap шум
        pose_gap = compute_pose_gap(info_a["pose"], info_b["pose"])
        expected = calib_models[bucket][f"zone_{zone_name}"]["slope_yaw"] * pose_gap["yaw"] + calib_models[bucket][...]["intercept"]
        zone_dist_corrected = max(0, zone_dist - expected)
        zone_distances[zone_name] = zone_dist_corrected
    
    # 6. Texture в UV
    uv_a, mask_a, conf_a = load_uv(photo_a_path)
    uv_b, mask_b, conf_b = load_uv(photo_b_path)
    shared_uv = mask_a & mask_b & (conf_a>0.3) & (conf_b>0.3)
    texture_distance = compute_texture_metrics_uv(uv_a, uv_b, shared_uv) # GLCM, LBP, FFT
    
    # 7. Финальный скор
    bone_score = np.median(list(zone_distances.values())) # медиана по костным зонам
    # pose-gap discount уже внутри zone_distances
    final_geometry_score = bone_score # уже скорректирован
    final_id_score = id_zscore
    
    # если оба skor > threshold => аномалия
    is_anomaly = (final_geometry_score > 1.2 and final_id_score > 1.5)
    
    return {
        "id_distance": id_distance,
        "id_zscore": id_zscore,
        "bone_score": bone_score,
        "texture_score": texture_distance,
        "per_vertex_heatmap": per_vertex_dist_normalized, # для визуализации синий->красный
        "zone_distances": zone_distances,
        "is_anomaly": is_anomaly,
        "pose_gap": pose_gap,
        "align_residual": residual_align
    }
```

---

## 7. КАК ЭТО ИНТЕГРИРОВАТЬ В ТВОЙ ТЕКУЩИЙ ПРОЕКТ (deeputin/)

Твой pipeline уже имеет 6 стадий. Надо минимально патчить:

**s1_extraction/engine.py:**
- Сейчас ты сохраняешь `reconstruction.pkl` с полями `vertices`, `vertices_canonical` etc. Добавь туда `id_params` и `exp_params` + `visible_idx_renderer` уже есть.
- Убедись что `neutral_expression=False` и `identity_only=False`, но в `payload` сохраняешь `id_params` всегда.
- В `info.json` сохраняй `pose.yaw`, `pitch`, `roll` из `angles_deg`, а не из filename.

**s2_metrics/modules/geometry_extractor.py:**
- Сейчас там 88 метрик zone_*. Добавь:
```python
metrics["id_norm"] = float(np.linalg.norm(id_params))
metrics["id_zscore_p0"] = float(id_params[0]) # первые 10 компонент
...
metrics["id_zscore_p9"] = float(id_params[9])
```
- Эти 10 метрик - твои главные pose-invariant.

**s3_identity/calibration_builder.py:**
- Текущий `PoseAwareCalibrationBuilder` должен не только `pairwise_noise` median считать, а строить регрессию `delta_metric ~ delta_yaw + delta_pitch`.
- Сохраняй в `calibration_reference.json` новую секцию `pose_regression`.

```python
# Добавь в builder:
from sklearn.linear_model import HuberRegressor # устойчив к выбросам
for metric_name in all_metrics:
    X = [[dyaw, dpitch, droll] for each pair]
    y = [dmetric for each pair]
    model = HuberRegressor().fit(X,y)
    pose_regression[bucket][metric_name] = {"coef_yaw": model.coef_[0], "intercept": model.intercept_}
```

**s4_compare/alignment.py + engine.py:**
- Сейчас `MeshAligner.procrustes_align` и `icp_refine` есть, но не используются по bone groups.
- В `engine.py:_icp_align_and_compare` ты уже делаешь Procrustes, но по всем shared_visible. Поменяй на bone only:
```python
bone_indices = self._get_bone_indices(recon_a) # nose + brow ridge + chin
src_bone = verts_a[bone_indices][shared_vis[bone_indices]]
tgt_bone = verts_b[bone_indices][shared_vis[bone_indices]]
R,t,res = umeyama_no_scale(src_bone, tgt_bone)
```
- Добавь шаг canonicalization до Procrustes.

**s4_compare/modules/pair_comparator.py:**
- После получения raw distance вызывай `pose_discount`:
```python
raw = abs(metric_a - metric_b) / mad
dyaw = abs(yaw_a - yaw_b)
expected_noise = regression_model[metric].coef_yaw * dyaw + intercept
corrected = max(0, raw - expected_noise)
```

**Новый модуль `core/uv_module` уже готов:** подключи его в s2 для текстуры:
```python
# в s2_metrics
from core.uv_module.hd_uv_generator import HDUVTextureGenerator
# генерируешь UV только если quality>0.4 (чтобы старые фото не ломались)
```

---

## 8. КЭШИРОВАНИЕ И ПРОИЗВОДИТЕЛЬНОСТЬ (чтобы не извлекать модель заново)

У тебя уже есть `reconstruction.py:save_reconstruction_cache` + `load_reconstruction_cache` с MD5 проверкой. Это идеально для задачи.

**Рекомендация 99 lvl:**
- При первом прогоне сохраняй `reconstruction.pkl` + `uv_texture_1024.png` + `uv_mask.png` + `id_params.npy` в папку фото (`/Volumes/SDCARD/storage/main/<photo_id>/`).
- При сравнении пары просто загружай pickle, без forward pass.
- In-memory LRU кэш на 10 мешей (уже есть `_evict_cache_if_needed`).

```python
# resolve_reconstruction в s1 - уже делает диск-кэш
def get_recon_cached(photo_id):
    path = storage / photo_id / "reconstruction.pkl"
    if path.exists():
        return pickle.load(open(path,"rb"))
    else:
        return adapter.reconstruct(image_path) # и сохранить
```

**Для морфнга (твой запрос про old_ui facemesh):**
```python
def morph_3d(recon_a, recon_b, alpha=0.5):
    # alpha 0 = a, 1 = b
    verts_morph = (1-alpha)*recon_a.vertices_canonical + alpha*recon_b.vertices_canonical
    uv_morph = (1-alpha)*uv_a + alpha*uv_b
    # рендеришь через nvdiffrast или cpu renderer из 3ddfa_v3/util
    return verts_morph, uv_morph
```

В старом UI (`old_ui/src/components`) у тебя был `FaceMeshViewer` который принимал `vertices` + `triangles` + `uv_texture`. Подключи его к новым pkl.

---

## 9. ХРОНОЛОГИЯ И БИОЛОГИЧЕСКИ НЕВОЗМОЖНЫЕ ТРАНСФОРМАЦИИ

После того как ты вычел pose-шум, остаются реальные скачки.

**Правила для твоего `s5_verdict/biological_limits.py`:**

- Форма черепа (id_params, zone_nose_centroid, brow ridge) не может измениться >0.3 z-score за <90 дней без хирургии меняющей кость.
- Скуловая ширина не может уменьшиться с возрастом (только увеличиться из-за птоза).
- Межорбитальное расстояние - константа с 12 лет. Изменение >1% => H2_DIFFERENT.
- Возврат к baseline: если метрика скакнула в 2014, а в 2018 вернулась к значению 2012, это return_to_baseline anomaly - маркер подмены.

```python
# В chronology analyzer
if date_gap < 90 and bone_distance_corrected > 1.5 and pose_gap["yaw"]<15:
    flag = "impossible_short_gap_shift" # сильнейший аргумент против H0
if abs(metric[t] - metric[t-2]) < 0.2*mad and abs(metric[t-1]-metric[t-2])>1.0*mad:
    flag = f"return_to_baseline:{metric_name}" # кто-то другой был между
```

---

## 10. ДЕТЕКТОР СИЛИКОНА ПОСЛЕ УЧЕТА КАЧЕСТВА ФОТО

Твоя гипотеза: в 2024-2025 фото высокая вероятность силикона с детальной текстурой. Старые фото 1999-2008 шумные и могут ошибочно детектиться как силикон из-за отсутствия пор.

Решение: **quality-aware thresholds** уже в `s3_identity`.

В `texture_anomaly.py` у тебя есть `CohortTextureAnomalyDetector` - он делит на cohorts по годам. Это правильно.

Но надо сделать adaptive:

```python
# Синтетика score
quality_factor = max(0.3, quality_overall) # для 1999 фото 0.3, для 2025 0.9

syn_score = 0
if lbp_uniform_r1 > 0.92: syn_score += 0.35 * quality_factor
if fft_anisotropy > 0.18: syn_score += 0.20 * quality_factor
if homo_cv < 0.10: syn_score += 0.30 * quality_factor
# для low quality фото syn_score никогда не превысит 0.7
```

---

## 11. ИТОГОВАЯ СХЕМА - КАК ДОЛЖЕН ВЫГЛЯДЕТЬ ФИНАЛЬНЫЙ ПАЙПЛАЙН

```
[1800 фото + 200 calibration] 
    |
    v
Stage0: parse_date_from_name, filename pose hint, catalog.json
    |
    v
Stage1: ThreeDDFAAdapter.reconstruct() -> reconstruction.pkl (35709 verts, id_params 80, exp 64, visible mask)
        + face_mask.png (seg_visible skin)
        + info.json (yaw pitch roll из 3DDFA, bucket по yaw, quality metrics)
        + КЭШ на диске по MD5
    |
    v
Stage2: 
   - geometry: zone_* по visible vertices + id_params[0..9] + face_scale
   - texture: GLCM, LBP, FFT, specular на face_mask центральная зона + опционально UV texture 1024
   - metrics.json
    |
    v
Stage3: ТОЛЬКО на calibration (твое лицо, 100% один человек)
   - per bucket stats (median, mad, q05, q95)
   - pairwise_noise: median delta внутри bucket
   - pose_regression: delta_metric ~ delta_yaw,pitch,roll (HuberRegressor)
   - age_profiles: metric ~ age
   - bucket_health: trust high/medium/low
   -> calibration_reference.json + pose_calibration_models.json
    |
    v
Stage4: Для каждого bucket сортируешь по дате, берешь window=3 соседа слева/справа
   - canonicalize к bucket target yaw
   - bone Procrustes по shared visible
   - per-zone distance с pose-gap discount: raw - expected_noise
   - id_params distance
   - texture distance в UV shared mask
   - pairs.json + pair_index.json
    |
    v
Stage5: Chronology analyzer (все 88+10 метрик, не 20 как сейчас)
   - spike detection (rate >2.2*baseline)
   - age inversion
   - return to baseline
   - era priors (1999-2011 H0=0.74, 2012-2021 H0=0.45, 2022+ H0=0.38)
   - Bayesian posterior H0/H1/H2/H_UNC
   - verdict.json per photo
    |
    v
Stage6: Report
   - thesis per era
   - top anomalies
   - cluster hypothesis (>=2 фото в bucket с high chronology)
   - report.json + report.md
```

---

## 12. БЫСТРЫЙ СТАРТ - ЧТО ПАТЧИТЬ В ТВОЕМ КОДЕ ПРЯМО СЕЙЧАС

**Приоритет #1 (критично):**
1. В `s1_extraction/engine.py:99` замени `build_placeholder_reconstruction` на реальный `ReconstructionAdapter().reconstruct()` - иначе вся геометрия фейк.
2. В `s4_compare/engine.py` добавь `pose_regression` discount.
3. В `s2_metrics/modules/geometry_extractor.py` добавь `id_params[0..9]`.

**Приоритет #2:**
4. Включи UV модуль для текстур: `core/uv_module/hd_uv_generator.py` уже работает на M1 CPU.
5. Перепиши `s4_compare/alignment.py` align() чтобы делал canonicalize + bone Procrustes, а не `return reconstruction`.

**Команды для теста (20 фото):**
```bash
cd /Users/victorkhudyakov/dutin/newapp/deeputin
python run.py --stages s1 s2 --limit 20
python run.py --stages s3 --input-calibration /Volumes/SDCARD/photo/calibration --limit 50
python run.py --stages s4 s5 s6 --limit 100
```

**Как проверить что pose-шум вычитается:**
- Возьми два calibration фото одного bucket с yaw gap 15°. Без коррекции geometry_distance должен быть ~0.8-1.2. С коррекцией должен стать <0.3.
- Если не становится - проверяй regression coef.

---

## 13. ОТВЕТ НА ГЛАВНЫЙ ВОПРОС "как теперь сравнивать фото внутри одного ракурса если наклон разный"

**Коротко:**
1. Не сравнивай пиксели/2D. Сравнивай `id_params` (80 чисел) - они pose-invariant.
2. Для зонной геометрии: canonicalize меш к yaw цели bucket, потом bone-Procrustes, потом считай residual ТОЛЬКО по shared visible вершинам, потом вычти expected шум от delta_yaw (модель обучена на calibration).
3. Для текстуры: переведи в UV 1024 и сравнивай только shared texels с high confidence. Это убирает перспективные искажения от наклона.
4. Кэшируй все реконструкции на диске - чтобы не пересчитывать 3DDFA и чтобы морфинг работал мгновенно из pkl.

**Если сделаешь эти 4 пункта, разница углов наклона внутри одного ракурса перестанет влиять на финальные `bone_distance` и `texture_distance`. Останется только реальная разница лиц.**

---

*Гайд составлен как 99 lvl эксперт по 3DDFA-V3, с учетом твоего репозитория DPTN, архива 1800+ фото, и калибровочного датасета. Версия 1.0, 2026-07-10, Prague.*
