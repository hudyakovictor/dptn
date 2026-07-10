# Feature discovery: новые scikit-image текстурные дескрипторы

Вычислено **37** новых дескрипторов на 296 файлах.
Стресс-тест: **864** измерений, 72 режима × 12 якорей.

> **Интерпретационное ограничение:** ранняя и поздняя папки разделены по годам. AUC ниже — ассоциация с folder/era label, не точность детекции силикона. `candidate_for_controlled_validation` означает только «приоритетно проверить на настоящем balanced ground truth». 

## TOP-20 по composite discovery score

| # | Метрика | Семейство | AUC папки | max abs(ρ) с quality | median / p90 shift (IQR) | Статус |
|---:|---|---|---:|---:|---:|---|
| 1 | `gabor_f16_anisotropy` | Gabor | 0.846 | 0.558 | 0.267 / 1.050 | candidate_for_controlled_validation |
| 2 | `gradient_orientation_entropy` | Gradient | 0.813 | 0.411 | 0.358 / 1.575 | candidate_for_controlled_validation |
| 3 | `glcm_correlation_d3` | GLCM | 0.815 | 0.654 | 0.306 / 1.159 | candidate_for_controlled_validation |
| 4 | `gabor_f08_anisotropy` | Gabor | 0.644 | 0.151 | 0.110 / 0.231 | candidate_for_controlled_validation |
| 5 | `gradient_orientation_anisotropy` | Gradient | 0.769 | 0.371 | 0.464 / 1.528 | candidate_for_controlled_validation |
| 6 | `local_entropy_r5` | Local entropy | 0.829 | 0.703 | 0.478 / 1.040 | candidate_for_controlled_validation |
| 7 | `glcm_energy_d3` | GLCM | 0.830 | 0.690 | 0.492 / 1.297 | candidate_for_controlled_validation |
| 8 | `glcm_energy_angle_iqr_d3` | GLCM | 0.839 | 0.630 | 0.534 / 1.710 | candidate_for_controlled_validation |
| 9 | `gabor_f24_anisotropy` | Gabor | 0.660 | 0.122 | 0.411 / 1.100 | candidate_for_controlled_validation |
| 10 | `glcm_asm_d3` | GLCM | 0.831 | 0.690 | 0.518 / 1.540 | candidate_for_controlled_validation |
| 11 | `black_tophat_density_r3` | Morphology | 0.771 | 0.446 | 0.626 / 1.626 | candidate_for_controlled_validation |
| 12 | `black_tophat_std_r3` | Morphology | 0.810 | 0.719 | 0.498 / 1.390 | candidate_for_controlled_validation |
| 13 | `local_entropy_r3` | Local entropy | 0.836 | 0.750 | 0.566 / 1.219 | candidate_for_controlled_validation |
| 14 | `shannon_entropy_q32` | Histogram | 0.620 | 0.305 | 0.165 / 0.472 | candidate_for_controlled_validation |
| 15 | `glcm_correlation_d1` | GLCM | 0.821 | 0.766 | 0.417 / 1.776 | candidate_for_controlled_validation |
| 16 | `hog_orientation_entropy` | HOG | 0.843 | 0.600 | 0.585 / 2.695 | reject_or_quality_condition_only |
| 17 | `structure_tensor_coherence` | Structure tensor | 0.861 | 0.773 | 0.795 / 1.746 | reject_or_quality_condition_only |
| 18 | `glcm_energy_d1` | GLCM | 0.834 | 0.799 | 0.705 / 1.557 | candidate_for_controlled_validation |
| 19 | `hog_mean_energy` | HOG | 0.885 | 0.703 | 0.830 / 2.421 | reject_or_quality_condition_only |
| 20 | `dog_zero_cross_density` | DoG | 0.784 | 0.696 | 0.718 / 1.876 | candidate_for_controlled_validation |

## Как использовать результат

1. Не добавлять все TOP-20 в одну логистическую модель: многие признаки коррелируют внутри семейства. Для controlled training оставить 1–2 представителя на семью после VIF/correlation pruning.
2. Признаки со сдвигом p90 > 1 IQR должны получать quality-dependent weight либо `not_assessable`; они не являются самостоятельным сигналом материала.
3. До появления ground truth нельзя преобразовать score/AUC в `silicone_prob` или публичный вывод об идентичности человека.
4. В production считать их на native skin ROI, не на 424×500 preview; иначе ресайз сам станет доминирующим «признаком». 
