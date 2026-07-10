#!/usr/bin/env python3
"""
Интеграционный тест: 3DDFA_V3 → HDUVTextureGenerator → экспорт.

Берёт фотографии из указанной папки, прогоняет реконструкцию 3DDFA_V3,
генерирует HD UV-текстуру и сохраняет результаты.

Использование:
    python test_hd_uv_pipeline.py
    python test_hd_uv_pipeline.py --input_dir face --output_dir renders_hd_uv
    python test_hd_uv_pipeline.py --input_dir face --uv_size 2048 --super_sample 2
    python test_hd_uv_pipeline.py --input_dir face --single photo_01.jpg --verbose
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Пути — настроить под вашу структуру проекта
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
CORE_3DDFA_DIR = WORKSPACE_ROOT / "core" / "3ddfa_v3"

# Добавляем пути для импорта
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CORE_3DDFA_DIR))
sys.path.insert(0, str(CORE_3DDFA_DIR.parent))

# ---------------------------------------------------------------------------
# Логирование
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_hd_uv")


# ---------------------------------------------------------------------------
# Вспомогательные утилиты
# ---------------------------------------------------------------------------


@contextmanager
def temporary_chdir(path: Path):
    prev = Path(os.getcwd())
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _resolve_torch_device(device: str) -> str:
    device = device.lower()
    if device == "cuda" and torch.cuda.is_available():
        return "cuda"
    if device == "mps" and torch.backends.mps.is_available():
        return "mps"
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    return "cpu"


def _resolve_detector_device(device: str) -> str:
    return "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"


# ---------------------------------------------------------------------------
# Адаптер 3DDFA_V3 → recon_dict
# ---------------------------------------------------------------------------


@dataclass
class _PipelineArgs:
    device: str
    detector_device: str
    detector: str = "retinaface"
    iscrop: bool = True
    backbone: str = "resnet50"
    ldm68: bool = False
    ldm106: bool = False
    ldm106_2d: bool = False
    ldm134: bool = False
    seg: bool = False
    seg_visible: bool = False
    useTex: bool = False
    extractTex: bool = False
    extractTexNew: bool = False
    extractTexNew_symmetry: bool = True
    extractTexNew_detail: bool = True
    extractTexNew_delight: bool = False
    uv_res: int = 1024
    detail_strength: float = 1.0
    inputpath: str = "examples/"
    savepath: str = "examples/results"


class FaceReconstructionPipeline:
    """Минимальная интеграция 3DDFA_V3 для получения recon_dict."""

    def __init__(self, device: str):
        self.device = _resolve_torch_device(device)
        detector_device = _resolve_detector_device(self.device)
        self.args = _PipelineArgs(
            device=self.device,
            detector_device=detector_device,
        )
        with temporary_chdir(CORE_3DDFA_DIR):
            from face_box import face_box  # type: ignore
            from model.recon import face_model  # type: ignore

            self.detector = face_box(self.args).detector
            self.model = face_model(self.args)

    def __call__(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Empty image passed to reconstruction pipeline")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        trans_params, crop_tensor = self.detector(pil_image)
        if crop_tensor is None:
            raise RuntimeError("Face detector did not return a crop tensor")

        crop_tensor = crop_tensor.to(self.device)
        self.model.input_img = crop_tensor
        self.model.trans_params = trans_params
        image_rgb_norm = (image_rgb.astype(np.float32) / 255.0).clip(0.0, 1.0)
        orig_tensor = torch.tensor(
            image_rgb_norm,
            dtype=torch.float32,
            device=self.device,
        ).permute(2, 0, 1).unsqueeze(0)
        self.model.orig_img_tensor = orig_tensor

        with torch.no_grad():
            results = self.model.forward()

        recon = self._build_recon_dict(
            results=results,
            trans_params=trans_params,
        )
        return recon

    @staticmethod
    def _back_resize(points: np.ndarray, trans_params: Optional[np.ndarray]) -> np.ndarray:
        if trans_params is None:
            return points
        w0, h0, s, t0, t1 = float(trans_params[0]), float(trans_params[1]), float(
            trans_params[2]
        ), float(trans_params[3]), float(trans_params[4])
        target_size = 224.0
        w = w0 * s
        h = h0 * s
        left = w / 2.0 - target_size / 2.0 + (t0 - w0 / 2.0) * s
        up = h / 2.0 - target_size / 2.0 + (h0 / 2.0 - t1) * s
        pts = points.copy()
        pts[:, 0] = (pts[:, 0] + left) / w * w0
        pts[:, 1] = (pts[:, 1] + up) / h * h0
        return pts

    def _build_recon_dict(
        self,
        results: Dict[str, Any],
        trans_params: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        v3d = np.asarray(results["v3d"], dtype=np.float32)[0]
        v2d = np.asarray(results["v2d"], dtype=np.float32)[0]

        v2d_img = v2d.copy()
        v2d_img[:, 1] = 224.0 - 1.0 - v2d_img[:, 1]
        v2d_img = self._back_resize(v2d_img, trans_params)

        vertices_for_uv = np.concatenate([v2d_img, v3d[:, 2:3]], axis=1).astype(
            np.float32
        )
        vertices_camera = v3d.astype(np.float32)
        vertices_obj = vertices_camera.copy()
        camera_distance = float(getattr(self.model, "camera_distance", 10.0))
        vertices_obj[:, 2] = camera_distance - vertices_obj[:, 2]
        triangles = np.asarray(results["tri"], dtype=np.int64)
        uv_coords = np.asarray(results["uv_coords"], dtype=np.float32)

        recon = {
            "vertices": vertices_for_uv,
            "vertices_2d": v2d_img.astype(np.float32),
            "vertices_3d": vertices_camera,
            "vertices_obj": vertices_obj,
            "triangles": triangles,
            "uv_coords": uv_coords,
        }

        if "alpha_sh" in results:
            recon["alpha_sh"] = np.asarray(results["alpha_sh"], dtype=np.float32).flatten()
        if "normals_3d" in results:
            recon["normals_3d"] = np.asarray(results["normals_3d"], dtype=np.float32)

        return recon


class TDDFA_V3_Adapter:
    """
    Обёртка над 3DDFA_V3 для получения recon_dict,
    совместимого с HDUVTextureGenerator.

    Адаптирует вызовы конкретной версии 3DDFA_V3,
    установленной в core/3ddfa_v3.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.pipeline = None
        self._load_pipeline()

    def _load_pipeline(self):
        """
        Загрузка пайплайна 3DDFA_V3.

        ВАЖНО: адаптируйте этот метод под вашу версию 3DDFA_V3.
        Ниже — типичные варианты API.
        """
        try:
            # --- Вариант A: если есть Pipeline / TDDFA класс ---
            # Попробовать несколько типичных путей импорта
            pipeline = None

            # Попытка 1: прямой импорт из пакета
            try:
                from tddfa_v3 import TDDFA_V3
                pipeline = TDDFA_V3(device=self.device)
                logger.info("Loaded 3DDFA_V3 via 'from tddfa_v3 import TDDFA_V3'")
            except ImportError:
                pass

            # Попытка 2: из core.3ddfa_v3
            if pipeline is None:
                try:
                    from core.tddfa_v3 import TDDFA_V3
                    pipeline = TDDFA_V3(device=self.device)
                    logger.info("Loaded 3DDFA_V3 via 'from core.tddfa_v3'")
                except ImportError:
                    pass

            # Попытка 3: через run_pipeline / pipeline module
            if pipeline is None:
                try:
                    # Некоторые версии 3DDFA_V3 используют функциональный API
                    import importlib
                    spec = importlib.util.spec_from_file_location(
                        "pipeline_3ddfa",
                        str(CORE_3DDFA_DIR / "pipeline.py"),
                    )
                    if spec and spec.loader:
                        mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(mod)
                        if hasattr(mod, "Pipeline"):
                            pipeline = mod.Pipeline(device=self.device)
                            logger.info("Loaded 3DDFA_V3 via pipeline.py")
                except Exception as e:
                    logger.debug("Pipeline.py attempt failed: %s", e)

            # Попытка 4: через demo/run файл
            if pipeline is None:
                try:
                    from core.tddfa_v3.demo import reconstruct_face
                    self._reconstruct_fn = reconstruct_face
                    logger.info("Loaded 3DDFA_V3 via demo.reconstruct_face")
                    return
                except ImportError:
                    pass

            if pipeline is not None:
                self.pipeline = pipeline
                return

            # Попытка 5: локальная интеграция (face_box + face_model)
            try:
                self.pipeline = FaceReconstructionPipeline(device=self.device)
                logger.info("Loaded 3DDFA_V3 via FaceReconstructionPipeline")
                return
            except Exception as adapter_exc:
                logger.error(
                    "Internal FaceReconstructionPipeline init failed: %s",
                    adapter_exc,
                )
            else:
                logger.error(
                    "Не удалось загрузить 3DDFA_V3. Проверьте установку в %s",
                    CORE_3DDFA_DIR,
                )
                logger.error(
                    "Будет использован fallback-режим с mock-данными для отладки."
                )

        except Exception as e:
            logger.error("Ошибка загрузки 3DDFA_V3: %s", e)
            traceback.print_exc()

    def reconstruct(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Запускает реконструкцию лица и возвращает recon_dict.

        Returns:
            dict с ключами:
                vertices    — (N, 3) float32, спроецированные вершины (x, y, z)
                triangles   — (T, 3) int64
                uv_coords   — (N, 2) float32, UV в [0, 1]
                vertices_3d — (N, 3) float32, вершины в camera space (опц.)
                coeffs      — dict с shape/exp/tex/illum (опц.)
            или None при ошибке.
        """
        if self.pipeline is not None:
            return self._reconstruct_via_pipeline(image)
        elif hasattr(self, "_reconstruct_fn"):
            return self._reconstruct_via_function(image)
        else:
            logger.warning("3DDFA_V3 не загружен — используем mock-данные")
            return self._mock_reconstruct(image)

    def _reconstruct_via_pipeline(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Адаптируйте под API вашей версии 3DDFA_V3.
        Типичные методы: pipeline(image), pipeline.reconstruct(image), и т.п.
        """
        try:
            # === АДАПТИРУЙТЕ ПОД ВАШЕ API ===
            #
            # Типичный вариант 1:
            #   result = self.pipeline(image)
            #   vertices = result['vertices']       # (N, 3)
            #   triangles = result['triangles']     # (T, 3)
            #   uv_coords = result['uv_coords']    # (N, 2)
            #
            # Типичный вариант 2:
            #   coeffs = self.pipeline.get_coeffs(image)
            #   vertices = self.pipeline.recon_vers(coeffs)
            #   triangles = self.pipeline.tri
            #   uv_coords = self.pipeline.uv_coords
            #
            # Типичный вариант 3:
            #   result = self.pipeline.run(image)
            #   vertices = result.vertices
            #   triangles = result.tri
            #   uv_coords = result.uv

            result = self.pipeline(image)

            # Извлечение данных (адаптировать под формат вывода)
            recon_dict = self._extract_recon_dict(result)
            return recon_dict

        except Exception as e:
            logger.error("Ошибка реконструкции: %s", e)
            traceback.print_exc()
            return None

    def _extract_recon_dict(self, result: Any) -> Dict[str, Any]:
        """
        Извлечь vertices, triangles, uv_coords из результата 3DDFA_V3.

        АДАПТИРУЙТЕ под формат вывода вашей версии.
        """
        recon_dict = {}

        # --- Vertices ---
        if isinstance(result, dict):
            # Словарь
            for key in ["vertices", "vertex", "verts", "ver"]:
                if key in result:
                    recon_dict["vertices"] = np.asarray(result[key], dtype=np.float32)
                    break

            for key in ["triangles", "tri", "faces"]:
                if key in result:
                    recon_dict["triangles"] = np.asarray(result[key], dtype=np.int64)
                    break

            for key in ["uv_coords", "uv", "tex_coords"]:
                if key in result:
                    recon_dict["uv_coords"] = np.asarray(result[key], dtype=np.float32)
                    break

            for key in ["vertices_2d", "v2d", "verts_2d"]:
                if key in result:
                    recon_dict["vertices_2d"] = np.asarray(result[key], dtype=np.float32)
                    break

            # Дополнительные данные
            for key in ["vertices_3d", "verts_3d", "v3d"]:
                if key in result:
                    recon_dict["vertices_3d"] = np.asarray(result[key], dtype=np.float32)
                    break

            for key in ["vertices_obj", "verts_obj", "v_obj"]:
                if key in result:
                    recon_dict["vertices_obj"] = np.asarray(result[key], dtype=np.float32)
                    break
            
            # Если 3D вершины были, но obj нет, сдублируем в object space
            if "vertices_obj" not in recon_dict and "vertices_3d" in recon_dict:
                v3 = recon_dict["vertices_3d"].copy()
                if v3.shape[1] == 3:
                    v3[:, 2] = 10.0 - v3[:, 2] # 3DDFA default camera distance
                recon_dict["vertices_obj"] = v3

            if "alpha_sh" in result:
                recon_dict["alpha_sh"] = np.asarray(result["alpha_sh"], dtype=np.float32)
            if "normals_3d" in result:
                recon_dict["normals_3d"] = np.asarray(result["normals_3d"], dtype=np.float32)

        elif hasattr(result, "vertices"):
            # Объект с атрибутами
            recon_dict["vertices"] = np.asarray(result.vertices, dtype=np.float32)
            recon_dict["triangles"] = np.asarray(
                getattr(result, "triangles", getattr(result, "tri", None)),
                dtype=np.int64,
            )
            recon_dict["uv_coords"] = np.asarray(
                getattr(result, "uv_coords", getattr(result, "uv", None)),
                dtype=np.float32,
            )
            for key in ["vertices_2d", "v2d", "verts_2d"]:
                if hasattr(result, key):
                    recon_dict["vertices_2d"] = np.asarray(getattr(result, key), dtype=np.float32)
                    break
            for key in ["vertices_3d", "verts_3d", "v3d"]:
                if hasattr(result, key):
                    recon_dict["vertices_3d"] = np.asarray(getattr(result, key), dtype=np.float32)
                    break
            for key in ["vertices_obj", "verts_obj", "v_obj"]:
                if hasattr(result, key):
                    recon_dict["vertices_obj"] = np.asarray(getattr(result, key), dtype=np.float32)
                    break
                    
            if "vertices_obj" not in recon_dict and "vertices_3d" in recon_dict:
                v3 = recon_dict["vertices_3d"].copy()
                if v3.ndim == 2 and v3.shape[1] == 3:
                    v3[:, 2] = 10.0 - v3[:, 2]
                recon_dict["vertices_obj"] = v3

        # Снять лишнее измерение batch, если есть
        for key in ["vertices", "vertices_2d", "vertices_3d", "vertices_obj", "uv_coords"]:
            if key in recon_dict and recon_dict[key].ndim == 3:
                recon_dict[key] = recon_dict[key][0]

        # Валидация
        self._validate_recon_dict(recon_dict)

        return recon_dict

    @staticmethod
    def _validate_recon_dict(recon_dict: Dict[str, Any]) -> None:
        """Проверка обязательных полей."""
        required = ["vertices", "triangles", "uv_coords", "vertices_2d"]
        missing = [k for k in required if k not in recon_dict]
        if missing:
            raise ValueError(
                f"recon_dict отсутствуют обязательные поля: {missing}. "
                f"Адаптируйте _extract_recon_dict() под ваш формат 3DDFA_V3."
            )

        v = recon_dict["vertices"]
        t = recon_dict["triangles"]
        uv = recon_dict["uv_coords"]

        assert v.ndim == 2 and v.shape[1] >= 2, f"vertices shape: {v.shape}"
        assert t.ndim == 2 and t.shape[1] == 3, f"triangles shape: {t.shape}"
        assert uv.ndim == 2 and uv.shape[1] >= 2, f"uv_coords shape: {uv.shape}"
        assert t.max() < v.shape[0], (
            f"triangle index {t.max()} >= num vertices {v.shape[0]}"
        )

    def _reconstruct_via_function(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """Реконструкция через функциональный API."""
        try:
            result = self._reconstruct_fn(image)
            return self._extract_recon_dict(result)
        except Exception as e:
            logger.error("Ошибка реконструкции (function): %s", e)
            return None

    @staticmethod
    def _mock_reconstruct(image: np.ndarray) -> Dict[str, Any]:
        """
        Mock-реконструкция для отладки UV-пайплайна без 3DDFA_V3.
        Генерирует плоский «меш» поверх лица.
        """
        h, w = image.shape[:2]
        cx, cy = w // 2, h // 2
        face_w, face_h = int(w * 0.5), int(h * 0.6)

        # Простая сетка 20×25 вершин
        nx, ny = 20, 25
        n_verts = nx * ny

        xs = np.linspace(cx - face_w // 2, cx + face_w // 2, nx, dtype=np.float32)
        ys = np.linspace(cy - face_h // 2, cy + face_h // 2, ny, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)

        vertices = np.stack([
            gx.ravel(),
            gy.ravel(),
            np.zeros(n_verts, dtype=np.float32),
        ], axis=1)

        uv_u = np.linspace(0, 1, nx, dtype=np.float32)
        uv_v = np.linspace(0, 1, ny, dtype=np.float32)
        gu, gv = np.meshgrid(uv_u, uv_v)
        uv_coords = np.stack([gu.ravel(), gv.ravel()], axis=1)

        # Треугольники
        triangles = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                idx = j * nx + i
                triangles.append([idx, idx + 1, idx + nx])
                triangles.append([idx + 1, idx + nx + 1, idx + nx])
        triangles = np.array(triangles, dtype=np.int64)

        logger.warning(
            "MOCK recon: %d verts, %d tris (НЕ настоящая реконструкция!)",
            n_verts, triangles.shape[0],
        )

        return {
            "vertices": vertices,
            "triangles": triangles,
            "uv_coords": uv_coords,
        }


# ---------------------------------------------------------------------------
# Основной тест
# ---------------------------------------------------------------------------

def find_images(input_dir: str, extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp")) -> List[str]:
    """Найти все изображения в директории."""
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
        images.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))
    images = sorted(set(images))
    return images


def process_single_image(
    image_path: str,
    output_dir: str,
    adapter: TDDFA_V3_Adapter,
    uv_config: Any,
    save_wireframe: bool = False,
    save_exr: bool = False,
) -> bool:
    """
    Обработка одного изображения: реконструкция → UV генерация → экспорт.
    Возвращает True при успехе.
    """
    from uv_module import HDUVTextureGenerator, HDUVConfig
    from uv_module.uvio import UVIOExporter

    base_name = Path(image_path).stem
    face_output_dir = os.path.join(output_dir, base_name)

    logger.info("=" * 60)
    logger.info("Обработка: %s", image_path)
    logger.info("=" * 60)

    # --- 1. Загрузка изображения ---
    t0 = time.time()
    image = cv2.imread(image_path)
    if image is None:
        logger.error("Не удалось загрузить: %s", image_path)
        return False
    logger.info("Изображение: %dx%d", image.shape[1], image.shape[0])

    # --- 2. Реконструкция 3DDFA_V3 ---
    t1 = time.time()
    recon_dict = adapter.reconstruct(image)
    if recon_dict is None:
        logger.error("Реконструкция не удалась: %s", image_path)
        return False
    t_recon = time.time() - t1

    n_verts = recon_dict["vertices"].shape[0]
    n_tris = recon_dict["triangles"].shape[0]
    logger.info(
        "Реконструкция: %d вершин, %d треугольников (%.2f сек)",
        n_verts, n_tris, t_recon,
    )

    # --- 3. Генерация HD UV-текстуры ---
    t2 = time.time()
    generator = HDUVTextureGenerator(config=uv_config)
    uv_tex_analysis, uv_tex_beauty, uv_mask_visible, uv_confidence, aux_data = (
        generator.generate(image, recon_dict)
    )
    t_uv = time.time() - t2

    visible_ratio = 100.0 * uv_mask_visible.sum() / max(uv_mask_visible.size, 1)
    logger.info(
        "UV генерация: %dx%d, видимость %.1f%% (%.2f сек)",
        uv_tex_analysis.shape[1], uv_tex_analysis.shape[0],
        visible_ratio, t_uv,
    )

    # --- 4. Экспорт ---
    t3 = time.time()
    exporter = UVIOExporter(
        save_exr=save_exr,
        save_wireframe=save_wireframe,
        texture_format="jpg",  # Use JPEG for smaller file size
        jpeg_quality=95,
    )
    export_vertices = recon_dict.get("vertices_obj", recon_dict["vertices"])
    obj_data = exporter.save_all(
        output_dir=face_output_dir,
        base_name=base_name,
        vertices=export_vertices,
        triangles=recon_dict["triangles"],
        uv_coords=recon_dict["uv_coords"],
        uv_tex_analysis=uv_tex_analysis,
        uv_tex_beauty=uv_tex_beauty,
        uv_mask_visible=uv_mask_visible,
        uv_confidence=uv_confidence,
        aux_data=aux_data,
    )
    t_export = time.time() - t3

    t_total = time.time() - t0

    logger.info("Экспорт: %s (%.2f сек)", face_output_dir, t_export)
    logger.info(
        "ИТОГО: %.2f сек (recon=%.2f, uv=%.2f, export=%.2f)",
        t_total, t_recon, t_uv, t_export,
    )
    logger.info("Файлы:")
    logger.info("  OBJ:        %s", obj_data.obj_path)
    logger.info("  Текстура:   %s", obj_data.texture_path)

    return True


# save_comparison function removed - no longer generates comparison.png


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Тест HD UV-текстур: 3DDFA_V3 → HDUVTextureGenerator → экспорт",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  # Обработать все фото из face/
  python test_hd_uv_pipeline.py

  # Только одно фото
  python test_hd_uv_pipeline.py --single face/photo_01.jpg

  # Высокое разрешение с суперсэмплингом
  python test_hd_uv_pipeline.py --uv_size 2048 --super_sample 2

  # С wireframe и verbose
  python test_hd_uv_pipeline.py --verbose --wireframe

  # Mock-режим (без 3DDFA_V3, для отладки UV-пайплайна)
  python test_hd_uv_pipeline.py --mock
        """,
    )

    parser.add_argument(
        "--input_dir", type=str, default="face",
        help="Папка с входными фотографиями (default: face)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="renders_hd_uv",
        help="Папка для результатов (default: renders_hd_uv)",
    )
    parser.add_argument(
        "--single", type=str, default=None,
        help="Обработать только один файл (путь к изображению)",
    )
    parser.add_argument(
        "--uv_size", type=int, default=1024,
        help="Размер UV-карты: 512, 1024, 2048, 4096 (default: 1024)",
    )
    parser.add_argument(
        "--super_sample", type=int, default=1,
        help="Суперсэмплинг: 1, 2, 4 (default: 1)",
    )
    parser.add_argument(
        "--detail_strength", type=float, default=1.2,
        help="Сила усиления деталей (default: 1.2)",
    )
    parser.add_argument(
        "--no_detail", action="store_true",
        help="Отключить усиление деталей",
    )
    parser.add_argument(
        "--no_symmetry", action="store_true",
        help="Отключить симметрийное заполнение",
    )
    parser.add_argument(
        "--delighting", action="store_true",
        help="Включить de-lighting (требует SH-коэффициенты)",
    )
    parser.add_argument(
        "--barycentric", action="store_true", default=True,
        help="Использовать barycentric bake (default: True)",
    )
    parser.add_argument(
        "--affine", action="store_true",
        help="Использовать affine bake вместо barycentric",
    )
    parser.add_argument(
        "--wireframe", action="store_true",
        help="Сохранять визуализацию UV-сетки",
    )
    parser.add_argument(
        "--exr", action="store_true",
        help="Сохранять float-карты в EXR",
    )
    parser.add_argument(
        "--force_visible", action="store_true", default=False,
        help="Все треугольники видимы (default: False, для отладки можно включить)",
    )
    parser.add_argument(
        "--no_force_visible", action="store_true",
        help="Использовать visibility по нормалям (теперь по умолчанию включено)",
    )
    parser.add_argument(
        "--device", type=str, default="mps",
        choices=["cpu", "mps", "cuda"],
        help="Устройство для 3DDFA_V3 (default: mps)",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Mock-режим: не использовать 3DDFA_V3 (для отладки UV-пайплайна)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Подробный вывод",
    )
    parser.add_argument(
        "--max_images", type=int, default=0,
        help="Максимум изображений для обработки (0 = все)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("uv_module").setLevel(logging.DEBUG)

    # --- Конфигурация UV ---
    from uv_module import HDUVConfig

    uv_config = HDUVConfig(
        uv_size=args.uv_size,
        super_sample=args.super_sample,
        enable_delighting=args.delighting,
        enable_symmetry_fill=not args.no_symmetry,
        enable_detail_boost=not args.no_detail,
        detail_strength=args.detail_strength,
        unsharp_amount=0.4,
        use_barycentric_bake=not args.affine,
        force_all_triangles_visible=(
            args.force_visible and not args.no_force_visible
        ),
        verbose=args.verbose,
    )

    logger.info("UV Config:")
    logger.info("  uv_size:        %d", uv_config.uv_size)
    logger.info("  super_sample:   %d", uv_config.super_sample)
    logger.info("  work_size:      %d", uv_config.uv_size * uv_config.super_sample)
    logger.info("  detail_boost:   %s (strength=%.2f)",
                uv_config.enable_detail_boost, uv_config.detail_strength)
    logger.info("  symmetry_fill:  %s", uv_config.enable_symmetry_fill)
    logger.info("  delighting:     %s", uv_config.enable_delighting)
    logger.info("  bake_method:    %s",
                "barycentric" if uv_config.use_barycentric_bake else "affine")
    logger.info("  force_visible:  %s", uv_config.force_all_triangles_visible)

    # --- Список изображений ---
    if args.single:
        if not os.path.isfile(args.single):
            logger.error("Файл не найден: %s", args.single)
            sys.exit(1)
        image_paths = [args.single]
    else:
        image_paths = find_images(args.input_dir)
        if not image_paths:
            logger.error("Не найдено изображений в %s", args.input_dir)
            sys.exit(1)

    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]

    logger.info("Найдено изображений: %d", len(image_paths))

    # --- Адаптер 3DDFA_V3 ---
    if args.mock:
        logger.warning("MOCK-режим: 3DDFA_V3 НЕ используется")

    adapter = TDDFA_V3_Adapter(device=args.device)

    if args.mock:
        # Принудительно использовать mock
        adapter.pipeline = None
        if hasattr(adapter, "_reconstruct_fn"):
            delattr(adapter, "_reconstruct_fn")

    # --- Обработка ---
    os.makedirs(args.output_dir, exist_ok=True)

    results = {"success": 0, "failed": 0, "times": []}

    for i, image_path in enumerate(image_paths, 1):
        logger.info("")
        logger.info("[%d/%d] %s", i, len(image_paths), os.path.basename(image_path))

        t0 = time.time()
        try:
            ok = process_single_image(
                image_path=image_path,
                output_dir=args.output_dir,
                adapter=adapter,
                uv_config=uv_config,
                save_wireframe=args.wireframe,
                save_exr=args.exr,
            )
            elapsed = time.time() - t0

            if ok:
                results["success"] += 1
                results["times"].append(elapsed)
            else:
                results["failed"] += 1

        except Exception as e:
            logger.error("ОШИБКА при обработке %s: %s", image_path, e)
            if args.verbose:
                traceback.print_exc()
            results["failed"] += 1

    # --- Итоги ---
    logger.info("")
    logger.info("=" * 60)
    logger.info("ИТОГИ")
    logger.info("=" * 60)
    logger.info("Успешно:  %d", results["success"])
    logger.info("Ошибок:   %d", results["failed"])

    if results["times"]:
        times = results["times"]
        logger.info("Время:")
        logger.info("  Среднее:    %.2f сек", np.mean(times))
        logger.info("  Медиана:    %.2f сек", np.median(times))
        logger.info("  Мин:        %.2f сек", np.min(times))
        logger.info("  Макс:       %.2f сек", np.max(times))
        logger.info("  Суммарно:   %.2f сек", np.sum(times))

    logger.info("Результаты: %s", os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()