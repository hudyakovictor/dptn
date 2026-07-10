from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .visibility import compute_triangle_visibility
from .uv_baker import UVBaker
from .detail import UVDetailEnhancer
from .inpaint_blend import UVBeautyPostprocessor
from .delight import compute_shading_uv, albedo_from_texture

logger = logging.getLogger(__name__)


@dataclass
class HDUVConfig:
    uv_size: int = 1024  # Changed from 2048 to 1024 for smaller file size
    super_sample: int = 2
    enable_delighting: bool = False
    enable_symmetry_fill: bool = True
    enable_detail_boost: bool = True
    detail_strength: float = 1.05
    unsharp_amount: float = 0.2
    detail_base_sigma_s_ratio: float = 0.02
    detail_base_sigma_r: float = 0.15
    use_barycentric_bake: bool = True
    force_all_triangles_visible: bool = False  # Changed to False for production
    device: str = "mps"
    use_fast_path: bool = True
    verbose: bool = False


class HDUVTextureGenerator:
    """
    Генератор HD UV-текстур для 3DDFA_V3.

    Пайплайн:
      1. Visibility weights (нормали, угол к камере)
      2. UV baking (affine warp / barycentric, supersampled)
      3. De-lighting (опц., SH + нормали → albedo)
      4. Detail enhancement (base/detail split, selective gain)
      5. Beauty postprocessing (symmetry fill, inpaint, seam blend)

    Возвращает:
      - uv_tex_analysis  (uint8) — только реальные данные с фото
      - uv_tex_beauty    (uint8) — для рендера (+ symmetry/inpaint)
      - uv_mask_visible  (uint8) — маска видимых texel'ов
      - uv_confidence    (float32) — карта доверия
      - aux_data         (dict) — вспомогательные карты
    """

    def __init__(self, config: HDUVConfig | None = None):
        self.config = config or HDUVConfig()
        self.baker = UVBaker(
            uv_size=self.config.uv_size,
            super_sample=self.config.super_sample,
        )
        self.detail = UVDetailEnhancer(
            detail_strength=self.config.detail_strength,
            unsharp_amount=self.config.unsharp_amount,
            base_sigma_s_ratio=self.config.detail_base_sigma_s_ratio,
            base_sigma_r=self.config.detail_base_sigma_r,
        )
        self.beauty_post = UVBeautyPostprocessor()

    def generate(
        self,
        image: np.ndarray,
        recon_dict: Dict[str, Any],
        debug_output_dir: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:

        cfg = self.config

        # --- 1. Подготовка геометрии (ТЗ 5.1–5.2) ---
        vertices = np.asarray(recon_dict["vertices"], dtype=np.float32)
        tris_key = "triangles" if "triangles" in recon_dict else "tri"
        triangles = np.asarray(recon_dict[tris_key], dtype=np.int64)
        uv_coords = np.asarray(recon_dict["uv_coords"], dtype=np.float32)

        # 2D-координаты в пикселях изображения
        if "vertices_2d" in recon_dict:
            vertices_2d = np.asarray(recon_dict["vertices_2d"], dtype=np.float32)
        else:
            raise ValueError("vertices_2d is mandatory but not found in recon_dict")

        # 3D для расчёта нормалей / видимости
        vertices_3d = np.asarray(
            recon_dict.get("vertices_3d", vertices), dtype=np.float32
        )
        if vertices_3d.ndim == 3:
            vertices_3d = vertices_3d[0]

        # --- 2. Видимость и веса (ТЗ 5.3) ---
        if cfg.force_all_triangles_visible:
            tri_weights = np.ones(triangles.shape[0], dtype=np.float32)
        else:
            tri_weights = compute_triangle_visibility(
                vertices_3d=vertices_3d,
                triangles=triangles,
                use_zbuffer=True,
                vertices_2d=vertices_2d,
                image_size=(image.shape[0], image.shape[1]),
                z_tolerance=5.0,  # Reduced from 25.0 for tighter occlusion
                angle_threshold_deg=75.0, # Reduced from 89.0 to fix side smearing
                gamma=4.0, # Increased from 1.5 to fix edge smearing
                occlusion_falloff=0.001,
            )

        if cfg.verbose:
            logger.info("[VIS] total triangles:   %d", triangles.shape[0])
            logger.info("[VIS] tri_weights > 0:   %d", (tri_weights > 0).sum())
            logger.info("[VIS] tri_weights > 0.5: %d", (tri_weights > 0.5).sum())
            logger.info("[VIS] tri_weights max:   %.4f", float(tri_weights.max()))
            logger.info("[VIS] tri_weights mean:  %.6f", float(tri_weights.mean()))

        # --- 3. UV Baking (ТЗ 5.4) ---
        if cfg.use_barycentric_bake:
            uv_texture_raw, uv_mask_visible, uv_confidence, uv_is_original = (
                self.baker.bake_via_barycentric(
                    image=image,
                    vertices_2d=vertices_2d,
                    uv_coords=uv_coords,
                    triangles=triangles,
                    tri_visibility_weights=tri_weights,
                )
            )
        else:
            uv_texture_raw, uv_mask_visible, uv_confidence, uv_is_original = (
                self.baker.bake(
                    image=image,
                    vertices_2d=vertices_2d,
                    uv_coords=uv_coords,
                    triangles=triangles,
                    tri_visibility_weights=tri_weights,
                    debug_output_dir=debug_output_dir,
                )
            )

        aux: Dict[str, Any] = {
            "uv_is_original": uv_is_original,
            "tri_visibility": tri_weights,
            "uv_texture_raw": uv_texture_raw.copy(),
        }

        # --- 4. De-lighting (ТЗ 5.6, опционально) ---
        if (
            cfg.enable_delighting
            and "normals_3d" in recon_dict
            and "alpha_sh" in recon_dict
        ):
            work_size = self.baker.uv_size * self.baker.super_sample
            normals_3d = np.asarray(recon_dict["normals_3d"], dtype=np.float32)
            normals_01 = np.clip((normals_3d + 1.0) * 0.5, 0.0, 1.0)
            normals_uv_01 = self.baker.bake_vertex_colors(
                uv_coords, triangles, normals_01, size=work_size
            )
            normals_uv = 2.0 * normals_uv_01 - 1.0
            nnorm = np.linalg.norm(normals_uv, axis=-1, keepdims=True)
            normals_uv = np.where(nnorm > 1e-8, normals_uv / nnorm, 0.0)

            shading_uv = compute_shading_uv(normals_uv, recon_dict["alpha_sh"])
            uv_texture_raw = albedo_from_texture(
                uv_texture_raw, shading_uv, eps=1e-4, clamp_max=255.0
            )
            aux["shading_uv"] = shading_uv
            aux["albedo_uv"] = uv_texture_raw.copy()

        # --- 5. Detail enhancement (ТЗ 5.5) ---
        detail_strength_map = None
        if cfg.enable_detail_boost:
            uv_tex_analysis, detail_strength_map = self.detail.enhance(
                uv_texture_raw=uv_texture_raw,
                uv_mask_visible=uv_mask_visible,
                uv_confidence=uv_confidence,
            )
        else:
            uv_tex_analysis = uv_texture_raw.copy()
            detail_strength_map = np.zeros_like(uv_confidence, dtype=np.float32)

        # --- 6. Beauty: symmetry + inpaint + seam blend (ТЗ 5.7–5.9) ---
        uv_tex_analysis = np.clip(uv_tex_analysis, 0, 255).astype(np.uint8)

        uv_tex_beauty = self.beauty_post.process(
            uv_tex_analysis=uv_tex_analysis,
            uv_mask_visible=uv_mask_visible,
            uv_confidence=uv_confidence,
            enable_symmetry=cfg.enable_symmetry_fill,
        )

        uv_tex_beauty = np.clip(uv_tex_beauty, 0, 255).astype(np.uint8)

        if detail_strength_map is not None:
            aux["uv_detail_strength_map"] = detail_strength_map

        return uv_tex_analysis, uv_tex_beauty, uv_mask_visible, uv_confidence, aux