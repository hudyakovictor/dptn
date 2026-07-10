# TOP-20 новых текстурных метрик: эксперимент `simple-test`

## Что было проверено

Перебраны **37 новых дескрипторов**, которых нет в текущем `TextureExtractor` под этими именами: GLCM ASM/energy/correlation, multi-scale LBP histograms, local entropy, Gabor orientation, Sobel-orientation, structure tensor, HOG, black top-hat, LoG blobs, Harris corners, DoG zero-crossings и morphology gradient. 

Признаки извлекались из alpha-safe внутренних skin-патчей. На 296 изображениях выполнен stress-test: **72 режима JPEG/blur/downscale/noise/gamma × 12 якорей = 864 симуляции**. Ранжирование учитывает: абсолютный AUC метки папки, максимальную Spearman-связь с quality-прокси и нормированный сдвиг от симуляций.

**Ограничение:** AUC ниже является способностью различать временно разнесённые папки, а не доказанной точностью `real skin`/`silicone`. Все строки — кандидаты для controlled validation, а не готовые веса/пороги материала.

## Результирующий TOP-20

| # | Метрика для кода | Семейство | AUC папки | max abs(ρ) качества | median/p90 shift, IQR | Решение |
|---:|---|---|---:|---:|---:|---|
| 1 | `gabor_f16_anisotropy` | Gabor | 0.846 | 0.558 | 0.267 / 1.050 | добавить в experimental feature set |
| 2 | `gradient_orientation_entropy` | Gradient | 0.813 | 0.411 | 0.358 / 1.575 | добавить в experimental feature set |
| 3 | `glcm_correlation_d3` | GLCM | 0.815 | 0.654 | 0.306 / 1.159 | добавить в experimental feature set |
| 4 | `gabor_f08_anisotropy` | Gabor | 0.644 | 0.151 | 0.110 / 0.231 | добавить в experimental feature set |
| 5 | `gradient_orientation_anisotropy` | Gradient | 0.769 | 0.371 | 0.464 / 1.528 | добавить в experimental feature set |
| 6 | `local_entropy_r5` | Local entropy | 0.829 | 0.703 | 0.478 / 1.040 | добавить в experimental feature set |
| 7 | `glcm_energy_d3` | GLCM | 0.830 | 0.690 | 0.492 / 1.297 | добавить в experimental feature set |
| 8 | `glcm_energy_angle_iqr_d3` | GLCM | 0.839 | 0.630 | 0.534 / 1.710 | добавить в experimental feature set |
| 9 | `gabor_f24_anisotropy` | Gabor | 0.660 | 0.122 | 0.411 / 1.100 | добавить в experimental feature set |
| 10 | `glcm_asm_d3` | GLCM | 0.831 | 0.690 | 0.518 / 1.540 | добавить в experimental feature set |
| 11 | `black_tophat_density_r3` | Morphology | 0.771 | 0.446 | 0.626 / 1.626 | добавить в experimental feature set |
| 12 | `black_tophat_std_r3` | Morphology | 0.810 | 0.719 | 0.498 / 1.390 | добавить в experimental feature set |
| 13 | `local_entropy_r3` | Local entropy | 0.836 | 0.750 | 0.566 / 1.219 | добавить в experimental feature set |
| 14 | `shannon_entropy_q32` | Histogram | 0.620 | 0.305 | 0.165 / 0.472 | добавить в experimental feature set |
| 15 | `glcm_correlation_d1` | GLCM | 0.821 | 0.766 | 0.417 / 1.776 | добавить в experimental feature set |
| 16 | `hog_orientation_entropy` | HOG | 0.843 | 0.600 | 0.585 / 2.695 | только quality-conditional / пока не использовать в модели |
| 17 | `structure_tensor_coherence` | Structure tensor | 0.861 | 0.773 | 0.795 / 1.746 | только quality-conditional / пока не использовать в модели |
| 18 | `glcm_energy_d1` | GLCM | 0.834 | 0.799 | 0.705 / 1.557 | добавить в experimental feature set |
| 19 | `hog_mean_energy` | HOG | 0.885 | 0.703 | 0.830 / 2.421 | только quality-conditional / пока не использовать в модели |
| 20 | `dog_zero_cross_density` | DoG | 0.784 | 0.696 | 0.718 / 1.876 | добавить в experimental feature set |

## Что добавить первым — без дублирования семейства

Не следует подавать 20 сильно коррелированных признаков в один классификатор. Первый экспериментальный набор из 12 сравнительно разных дескрипторов:

```text
gabor_f16_anisotropy
gabor_f08_anisotropy
gradient_orientation_entropy
gradient_orientation_anisotropy
glcm_correlation_d3
glcm_energy_angle_iqr_d3
local_entropy_r5
black_tophat_density_r3
shannon_entropy_q32
dog_zero_cross_density
lbp_r3_hist_entropy
glcm_energy_d3
```

Из них наиболее устойчивы к проверенным искажениям `gabor_f08_anisotropy` (median/p90: 0.110/0.231 IQR) и `shannon_entropy_q32` (0.165/0.472 IQR). Высокие AUC HOG/structure-tensor признаков не делают их хорошими: они не прошли p90-устойчивость и остаются только quality-conditional диагностикой.

## Правило интеграции в проект

1. Добавить вычисление признаков в отдельный `texture_experimental` namespace, не смешивая с существующим `silicone_prob`.
2. Вычислять только по native RGB + eroded native skin mask; `face_mask.png` 424×500 оставить для UI.
3. Возвращать `feature_value`, `valid_patch_count`, `quality_weight`, `feature_available`; не подставлять отсутствующее значение нулём.
4. Для метрики с `p90 shift > 1 IQR` применять вес из quality gate или возвращать `not_assessable` на плохом снимке.
5. Когда появится ground truth, сделать split по человеку × сессии × камере, calibration curve, PR-AUC/ROC-AUC, FPR на старых сканах и ablation по каждому семейству. Только тогда отбирать окончательные веса/пороги.

## Файлы эксперимента

- `project/tools/discover_texture_metrics.py` — воспроизводимый feature discovery;
- `audit_artifacts/texture_metric_discovery/new_features_base.csv` — 37 новых признаков для 296 файлов;
- `audit_artifacts/texture_metric_discovery/new_features_simulations.csv` — 864 стресс-измерения;
- `audit_artifacts/texture_metric_discovery/top20_ranking.csv` — полный ranking;
- `audit_artifacts/texture_metric_discovery/REPORT.md` — машинный отчёт.
