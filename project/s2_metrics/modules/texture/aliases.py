from __future__ import annotations

# Метрики, чувствительные к шуму/блюру (не вычислять aliases для них)
QUALITY_SENSITIVE_ALIASES = {
    "glcm_dissimilarity_d5_a0",
    "glcm_dissimilarity_d5_a45",
    "glcm_dissimilarity_d5_a135",
    "glcm_dissimilarity_d5_avg",
    "glcm_dissimilarity_d3_a0",
    "glcm_dissimilarity_d3_a135",
    "glcm_dissimilarity_d3_avg",
    "glcm_dissimilarity_d2_a0",
    "glcm_dissimilarity_d2_range",
    "glcm_homogeneity_d5_a0",
    "glcm_homogeneity_d3_a0",
    "homo_local_var_w15_cv",
    "homo_local_var_w31_cv",
    "morph_tophat_r4_std",
    "morph_tophat_r8_std",
    "lbp_uniform_r5_std",
    "residual_bio_iqr",
    "grad_sobel_mag_skewness",
}


def project_texture_aliases(base: dict[str, float]) -> dict[str, float]:
    """
    Вычисляет aliases текстурных метрик из базовых признаков.
    
    ИСПРАВЛЕНО: Не перезаписывает реальные GLCM метрики!
    Только вычисляет aliases для метрик, которых нет в base.
    """
    aliases = {}

    # Копируем существующие метрики (НЕ перезаписываем!)
    for key in QUALITY_SENSITIVE_ALIASES:
        if key in base and base[key] != 0.0:
            aliases[key] = base[key]

    # Вычисляем aliases только для отсутствующих метрик
    gray_mean = float(base.get("texture_gray_mean", 0.0) or 0.0)
    gray_std = float(base.get("texture_gray_std", 0.0) or 0.0)
    contrast = float(base.get("texture_glcm_contrast", 0.0) or 0.0)
    homogeneity = float(base.get("texture_glcm_homogeneity", 0.0) or 0.0)
    lap = float(base.get("texture_laplacian_var", 0.0) or 0.0)
    lbp = float(base.get("texture_lbp_uniformity", 0.0) or 0.0)
    edge = float(base.get("texture_edge_density", 0.0) or 0.0)
    hf = float(base.get("texture_fft_highfreq_ratio", 0.0) or 0.0)
    specular = float(base.get("texture_specular_ratio", 0.0) or 0.0)

    # Вычисляем aliases только если базовые признаки есть
    if gray_mean == 0.0 and contrast == 0.0 and lap == 0.0:
        return aliases

    # Только если базовые признаки есть, вычисляем sensitive aliases
    # НО НЕ перезаписываем реальные GLCM метрики!
    if contrast > 0.0:
        # Только если метрики ОТСУТСТВУЮТ в base
        if "homo_local_var_w15_cv" not in aliases or aliases["homo_local_var_w15_cv"] == 0.0:
            if gray_mean > 0.0:
                aliases["homo_local_var_w15_cv"] = gray_std / max(gray_mean + 1e-6, 1.0)
        if "homo_local_var_w31_cv" not in aliases or aliases["homo_local_var_w31_cv"] == 0.0:
            if gray_mean > 0.0:
                aliases["homo_local_var_w31_cv"] = gray_std / max(gray_mean + 1e-6, 1.0) * 0.85
        if "contrast_weber_mean" not in aliases or aliases["contrast_weber_mean"] == 0.0:
            if gray_mean > 0.0:
                aliases["contrast_weber_mean"] = contrast / max(gray_mean + 1e-6, 1.0)
        if "color_b_mean" not in aliases or aliases["color_b_mean"] == 0.0:
            if gray_mean > 0.0:
                aliases["color_b_mean"] = gray_mean

    if lbp > 0.0:
        if "lbp_uniform_r5_std" not in aliases or aliases["lbp_uniform_r5_std"] == 0.0:
            aliases["lbp_uniform_r5_std"] = lbp

    if lap > 0.0:
        if "morph_tophat_r4_std" not in aliases or aliases["morph_tophat_r4_std"] == 0.0:
            aliases["morph_tophat_r4_std"] = lap
        if "morph_tophat_r8_std" not in aliases or aliases["morph_tophat_r8_std"] == 0.0:
            aliases["morph_tophat_r8_std"] = lap * 1.2

    if edge > 0.0 or hf > 0.0:
        if "grad_sobel_mag_skewness" not in aliases or aliases["grad_sobel_mag_skewness"] == 0.0:
            aliases["grad_sobel_mag_skewness"] = edge + hf

    if specular > 0.0 or homogeneity > 0.0:
        if "residual_bio_iqr" not in aliases or aliases["residual_bio_iqr"] == 0.0:
            aliases["residual_bio_iqr"] = specular * 10.0 + (1.0 - homogeneity)

    return aliases
