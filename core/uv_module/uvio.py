"""
UVIO — экспорт UV-текстур, мешей и вспомогательных карт.

Форматы вывода полностью совместимы с 3DDFA_V3:
  - {base}.obj  + {base}.mtl  + {base}_texture.png
  - {base}_analysis.png       — текстура для анализа (только реальные данные)
  - {base}_beauty.png         — текстура для рендера
  - {base}_confidence.png     — карта доверия (grayscale)
  - {base}_confidence.exr     — карта доверия (float32, если поддерживается)
  - {base}_mask_visible.png   — маска видимых texel'ов
  - {base}_is_original.png    — маска оригинальных данных (без симметрии/inpaint)
  - {base}_uv_wire.png        — визуализация UV-сетки (debug)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Структура данных для передачи во внешние рендеры / пайплайны
# ---------------------------------------------------------------------------

@dataclass
class ObjData:
    """
    Всё необходимое для рендера меша с текстурой.
    Совместимо с ожиданиями 3DDFA_V3.
    """
    vertices: np.ndarray          # (N, 3) float32
    triangles: np.ndarray         # (T, 3) int64, 0-indexed
    uv_coords: np.ndarray        # (N, 2) float32, [0, 1]
    texture_path: str = ""        # абсолютный путь к текстуре
    obj_path: str = ""
    mtl_path: str = ""

    # Опционально — для расширенного анализа
    texture_analysis_path: str = ""
    texture_beauty_path: str = ""
    confidence_path: str = ""
    mask_visible_path: str = ""


# ---------------------------------------------------------------------------
# Главный класс экспорта
# ---------------------------------------------------------------------------

class UVIOExporter:
    """
    Экспорт мешей, текстур и вспомогательных карт на диск.

    Пример использования::

        exporter = UVIOExporter()

        obj_data = exporter.save_all(
            output_dir="renders/face_01",
            base_name="face_01",
            vertices=recon_dict["vertices"],
            triangles=recon_dict["triangles"],
            uv_coords=recon_dict["uv_coords"],
            uv_tex_analysis=uv_tex_analysis,
            uv_tex_beauty=uv_tex_beauty,
            uv_mask_visible=uv_mask_visible,
            uv_confidence=uv_confidence,
            aux_data=aux_data,
        )
    """

    def __init__(
        self,
        save_exr: bool = False,
        save_wireframe: bool = False,
        texture_suffix: str = "_texture",
        jpeg_quality: int = 95,
        save_auxiliary: bool = False,  # Set to False to skip intermediate PNGs
        texture_format: str = "png",  # "png" or "jpg"
    ):
        """
        Args:
            save_exr: сохранять ли float32-карты в EXR (требует OpenCV с поддержкой)
            save_wireframe: сохранять ли визуализацию UV-сетки
            texture_suffix: суффикс для основной текстуры (default: '_texture')
            jpeg_quality: качество JPEG (1-100, выше = лучше качество)
            save_auxiliary: сохранять ли вспомогательные карты (analysis, confidence, mask, etc)
            texture_format: формат сохранения текстуры - "png" или "jpg"
        """
        self.save_exr = save_exr
        self.save_wireframe = save_wireframe
        self.texture_suffix = texture_suffix
        self.jpeg_quality = jpeg_quality
        self.save_auxiliary = save_auxiliary
        self.texture_format = texture_format.lower()

    # -----------------------------------------------------------------------
    # Основной метод — сохранить всё
    # -----------------------------------------------------------------------

    def save_all(
        self,
        output_dir: str,
        base_name: str,
        vertices: np.ndarray,
        triangles: np.ndarray,
        uv_coords: np.ndarray,
        uv_tex_analysis: np.ndarray,
        uv_tex_beauty: np.ndarray,
        uv_mask_visible: np.ndarray,
        uv_confidence: np.ndarray,
        aux_data: Optional[Dict[str, Any]] = None,
    ) -> ObjData:
        """
        Сохраняет полный набор файлов и возвращает ObjData.

        Структура файлов::

            output_dir/
              base_name.obj
              base_name.mtl
              base_name_texture.png      ← beauty-текстура (для рендера)
              base_name_analysis.png     ← analysis-текстура
              base_name_confidence.png
              base_name_confidence.exr   ← (опционально)
              base_name_mask_visible.png
              base_name_is_original.png
              base_name_uv_wire.png      ← (опционально)
        """
        os.makedirs(output_dir, exist_ok=True)
        aux = aux_data or {}

        # --- Пути ---
        def p(suffix: str, ext: str = ".png") -> str:
            return os.path.join(output_dir, f"{base_name}{suffix}{ext}")

        # Main texture - use configured format
        tex_ext = ".jpg" if self.texture_format == "jpg" else ".png"
        texture_path = p(self.texture_suffix, tex_ext)
        obj_path = p("", ".obj")
        mtl_path = p("", ".mtl")

        # --- 1. Main texture only ---
        self.save_image(texture_path, uv_tex_beauty, use_jpeg=(self.texture_format == "jpg"))
        logger.info("Saved texture: %s", texture_path)

        # --- 2. Auxiliary maps (only if save_auxiliary=True) ---
        if self.save_auxiliary:
            analysis_path = p("_analysis")
            confidence_path = p("_confidence")
            mask_path = p("_mask_visible")
            is_orig_path = p("_is_original")
            
            self.save_image(analysis_path, uv_tex_analysis)
            self.save_mask(mask_path, uv_mask_visible)
            self.save_confidence(confidence_path, uv_confidence)

            if "uv_is_original" in aux:
                self.save_mask(is_orig_path, aux["uv_is_original"])

            # EXR для float-карт
            if self.save_exr:
                self.save_float_map(p("_confidence", ".exr"), uv_confidence)
                if "shading_uv" in aux:
                    self.save_float_map(p("_shading", ".exr"), aux["shading_uv"])
                if "albedo_uv" in aux:
                    self.save_image(p("_albedo"), aux["albedo_uv"])

            # Дополнительные карты из aux
            if "uv_detail_strength_map" in aux:
                dsm = aux["uv_detail_strength_map"]
                self.save_confidence(p("_detail_strength"), dsm / max(dsm.max(), 1e-6))

            if "uv_visibility_map" in aux:
                self.save_confidence(p("_visibility"), aux["uv_visibility_map"])

        # --- 3. UV wireframe (debug, only if save_wireframe=True) ---
        if self.save_wireframe:
            wire_path = p("_uv_wire")
            self.save_uv_wireframe(wire_path, uv_coords, triangles,
                                    uv_tex_beauty.shape[0])

        # --- 4. OBJ + MTL ---
        texture_filename = os.path.basename(texture_path)
        self.save_obj(
            obj_path=obj_path,
            mtl_path=mtl_path,
            vertices=vertices,
            triangles=triangles,
            uv_coords=uv_coords,
            texture_filename=texture_filename,
            base_name=base_name,
        )

        # --- ObjData ---
        obj_data = ObjData(
            vertices=np.asarray(vertices, dtype=np.float32),
            triangles=np.asarray(triangles, dtype=np.int64),
            uv_coords=np.asarray(uv_coords, dtype=np.float32),
            texture_path=os.path.abspath(texture_path),
            obj_path=os.path.abspath(obj_path),
            mtl_path=os.path.abspath(mtl_path),
        )

        logger.info("Export complete → %s  (%d verts, %d tris)",
                     output_dir, vertices.shape[0], triangles.shape[0])

        return obj_data

    # -----------------------------------------------------------------------
    # OBJ / MTL
    # -----------------------------------------------------------------------

    def save_obj(
        self,
        obj_path: str,
        mtl_path: str,
        vertices: np.ndarray,
        triangles: np.ndarray,
        uv_coords: np.ndarray,
        texture_filename: str,
        base_name: str = "face",
        material_name: str = "face_material",
    ) -> None:
        """
        Сохраняет OBJ + MTL, совместимые с 3DDFA_V3.

        OBJ формат::

            mtllib base_name.mtl
            usemtl face_material
            v x y z
            vt u v
            f v1/vt1 v2/vt2 v3/vt3

        Индексы — 1-based (стандарт OBJ).
        """
        verts = np.asarray(vertices, dtype=np.float64)
        tris = np.asarray(triangles, dtype=np.int64)
        uv = np.asarray(uv_coords, dtype=np.float64)

        # Убедиться что vertices (N, 3)
        if verts.ndim == 3:
            verts = verts[0]
        if verts.shape[1] == 2:
            # Если только 2D — добавить z=0
            verts = np.hstack([verts, np.zeros((verts.shape[0], 1), dtype=np.float64)])

        # UV: убедиться (N, 2)
        if uv.shape[1] > 2:
            uv = uv[:, :2]

        # Нормализация UV в [0, 1]
        uv_max = uv.max()
        if uv_max > 1.5:
            uv = uv / max(uv_max, 1e-6)

        n_verts = verts.shape[0]
        n_uv = uv.shape[0]
        n_tris = tris.shape[0]

        mtl_filename = os.path.basename(mtl_path)

        # --- Write OBJ ---
        with open(obj_path, "w") as f:
            f.write(f"# HD UV Texture Generator (3DDFA_V3 compatible)\n")
            f.write(f"# Vertices: {n_verts}  Triangles: {n_tris}\n")
            f.write(f"mtllib {mtl_filename}\n")
            f.write(f"usemtl {material_name}\n\n")

            # Vertices
            for i in range(n_verts):
                f.write(f"v {verts[i, 0]:.6f} {verts[i, 1]:.6f} {verts[i, 2]:.6f}\n")

            f.write("\n")

            # Texture coordinates
            for i in range(n_uv):
                f.write(f"vt {uv[i, 0]:.6f} {uv[i, 1]:.6f}\n")

            f.write("\n")

            # Faces (1-indexed, v/vt format)
            # Если количество UV == количество вершин, используем одинаковые индексы
            # (стандартная ситуация для BFM/FFHQ)
            for i in range(n_tris):
                i0 = tris[i, 0] + 1  # OBJ 1-indexed
                i1 = tris[i, 1] + 1
                i2 = tris[i, 2] + 1
                f.write(f"f {i0}/{i0} {i1}/{i1} {i2}/{i2}\n")

        logger.debug("Saved OBJ: %s (%d verts, %d tris)", obj_path, n_verts, n_tris)

        # --- Write MTL ---
        self._write_mtl(mtl_path, material_name, texture_filename)

    @staticmethod
    def _write_mtl(
        mtl_path: str,
        material_name: str,
        texture_filename: str,
    ) -> None:
        """
        Минимальный MTL, совместимый с 3DDFA_V3 и большинством 3D-вьюеров.

        Содержимое::

            newmtl face_material
            Ka 0.2 0.2 0.2
            Kd 1.0 1.0 1.0
            Ks 0.0 0.0 0.0
            d 1.0
            illum 1
            map_Kd texture.png
        """
        with open(mtl_path, "w") as f:
            f.write(f"# HD UV Texture Generator\n")
            f.write(f"newmtl {material_name}\n")
            f.write("Ka 0.200 0.200 0.200\n")
            f.write("Kd 1.000 1.000 1.000\n")
            f.write("Ks 0.000 0.000 0.000\n")
            f.write("d 1.0\n")
            f.write("illum 1\n")
            f.write(f"map_Kd {texture_filename}\n")

        logger.debug("Saved MTL: %s (map_Kd=%s)", mtl_path, texture_filename)

    # -----------------------------------------------------------------------
    # Изображения и карты
    # -----------------------------------------------------------------------

    @staticmethod
    def save_image(path: str, image: np.ndarray, use_jpeg: bool = False) -> None:
        """
        Сохраняет изображение как PNG или JPEG.
        Принимает uint8 или float32 (0–255).
        Input is expected in BGR order (standard OpenCV convention).
        
        Args:
            path: путь к файлу (с расширением .png или .jpg)
            image: numpy array с изображением (BGR)
            use_jpeg: если True и путь заканчивается на .jpg, использовать JPEG с jpeg_quality
        """
        img = np.asarray(image)
        if img.dtype in (np.float32, np.float64):
            img = np.clip(img, 0, 255).astype(np.uint8)

        # Save with appropriate format
        # Data is in BGR (OpenCV convention) — cv2.imwrite expects BGR
        ext = os.path.splitext(path)[1].lower()
        if use_jpeg and ext in (".jpg", ".jpeg"):
            # Use high quality JPEG compression
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
            cv2.imwrite(path, img, encode_params)
        else:
            # PNG for lossless quality
            cv2.imwrite(path, img)

        logger.debug("Saved image: %s (%s)", path, img.shape)

    @staticmethod
    def save_mask(path: str, mask: np.ndarray) -> None:
        """Сохраняет бинарную маску как grayscale PNG (0 / 255)."""
        m = np.asarray(mask)
        if m.dtype == bool:
            m = m.astype(np.uint8) * 255
        elif m.max() <= 1:
            m = (m * 255).astype(np.uint8)
        else:
            m = m.astype(np.uint8)
        cv2.imwrite(path, m)
        logger.debug("Saved mask:  %s (nonzero=%.1f%%)",
                     path, 100.0 * (m > 0).sum() / max(m.size, 1))

    @staticmethod
    def save_confidence(path: str, confidence: np.ndarray) -> None:
        """
        Сохраняет карту доверия как grayscale PNG.
        Вход: float32 [0, 1] → выход: uint8 [0, 255].
        """
        conf = np.asarray(confidence, dtype=np.float32)
        # Нормализация в [0, 1] если не в диапазоне
        c_max = conf.max()
        if c_max > 1.0 and c_max > 0:
            conf = conf / c_max
        conf_u8 = np.clip(conf * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(path, conf_u8)
        logger.debug("Saved confidence: %s (mean=%.3f, coverage=%.1f%%)",
                     path, conf.mean(),
                     100.0 * (conf > 0.01).sum() / max(conf.size, 1))

    @staticmethod
    def save_float_map(path: str, data: np.ndarray) -> None:
        """
        Сохраняет float32-карту в EXR (или TIFF fallback).
        Требует OpenCV, скомпилированного с поддержкой OpenEXR.
        """
        arr = np.asarray(data, dtype=np.float32)
        ext = os.path.splitext(path)[1].lower()

        try:
            if ext == ".exr":
                # OpenCV EXR write
                success = cv2.imwrite(path, arr)
                if not success:
                    raise IOError("cv2.imwrite returned False for EXR")
            elif ext in (".tiff", ".tif"):
                success = cv2.imwrite(path, arr)
                if not success:
                    raise IOError("cv2.imwrite returned False for TIFF")
            else:
                # Fallback: save as .npy
                npy_path = os.path.splitext(path)[0] + ".npy"
                np.save(npy_path, arr)
                logger.warning("EXR/TIFF not supported, saved as %s", npy_path)
                return

            logger.debug("Saved float map: %s (%s, range=[%.3f, %.3f])",
                         path, arr.shape, arr.min(), arr.max())

        except Exception as e:
            # Fallback: сохранить как .npy
            npy_path = os.path.splitext(path)[0] + ".npy"
            np.save(npy_path, arr)
            logger.warning("Failed to save %s (%s), saved as %s", path, e, npy_path)

    # -----------------------------------------------------------------------
    # UV wireframe визуализация
    # -----------------------------------------------------------------------

    @staticmethod
    def save_uv_wireframe(
        path: str,
        uv_coords: np.ndarray,
        triangles: np.ndarray,
        uv_size: int = 1024,
        bg_color: Tuple[int, int, int] = (32, 32, 32),
        wire_color: Tuple[int, int, int] = (0, 200, 0),
        fill_color: Optional[Tuple[int, int, int]] = (0, 60, 0),
    ) -> None:
        """Визуализация UV-сетки: заливка треугольников + wireframe."""
        uv = np.asarray(uv_coords, dtype=np.float32)
        if uv.shape[1] > 2:
            uv = uv[:, :2]
        uv_max = uv.max()
        if uv_max > 1.5:
            uv = uv / max(uv_max, 1e-6)

        uv_px = np.empty_like(uv)
        uv_px[:, 0] = uv[:, 0] * (uv_size - 1)
        uv_px[:, 1] = (1.0 - uv[:, 1]) * (uv_size - 1)

        canvas = np.full((uv_size, uv_size, 3), bg_color, dtype=np.uint8)
        tris = np.asarray(triangles, dtype=np.int64)

        for i in range(tris.shape[0]):
            pts = uv_px[tris[i]].astype(np.int32).reshape(-1, 1, 2)
            if fill_color is not None:
                cv2.fillConvexPoly(canvas, pts, fill_color)

        for i in range(tris.shape[0]):
            pts = uv_px[tris[i]].astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], isClosed=True,
                         color=wire_color, thickness=1, lineType=cv2.LINE_AA)

        cv2.imwrite(path, canvas)
        logger.debug("Saved UV wireframe: %s (%d triangles)", path, tris.shape[0])

    # -----------------------------------------------------------------------
    # Утилиты для загрузки (для тестов / пайплайна)
    # -----------------------------------------------------------------------

    @staticmethod
    def load_texture(path: str) -> np.ndarray:
        """Загружает текстуру как BGR uint8."""
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot load texture: {path}")
        return img

    @staticmethod
    def load_confidence(path: str) -> np.ndarray:
        """Загружает карту доверия как float32 [0, 1]."""
        ext = os.path.splitext(path)[1].lower()
        if ext == ".npy":
            return np.load(path).astype(np.float32)
        elif ext == ".exr":
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(f"Cannot load EXR: {path}")
            return img.astype(np.float32)
        else:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Cannot load confidence: {path}")
            return img.astype(np.float32) / 255.0

    @staticmethod
    def load_mask(path: str) -> np.ndarray:
        """Загружает маску как bool."""
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot load mask: {path}")
        return img > 127