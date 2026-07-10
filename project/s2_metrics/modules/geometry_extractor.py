"""Geometry extractor - обёртка над backend/metrics/registry для извлечения геометрических метрик."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Добавляем путь к бэкенду для импорта модулей метрик
BACKEND_ROOT = Path(__file__).resolve().parents[4] / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

try:
    from metrics.registry import load_modules, all_specs, specs_for_bucket
    from metrics.types import MetricContext, MetricValue, MetricSpec
    from metrics.policy import spec_production_ready
    _HAS_BACKEND = True
except ImportError:
    _HAS_BACKEND = False


class GeometryExtractor:
    """Извлечение геометрических метрик из 3D реконструкции лица."""

    def __init__(self):
        self._modules = None
        if _HAS_BACKEND:
            try:
                self._modules = {m.__name__.split(".")[-1] + ".py": m for m in load_modules()}
            except Exception:
                self._modules = None

    def extract(self, ctx: Any) -> dict[str, float]:
        """
        Извлекает все геометрические метрики для данного контекста.

        Args:
            ctx: MetricContext с данными реконструкции

        Returns:
            Словарь {имя_метрики: значение}
        """
        if not _HAS_BACKEND or self._modules is None:
            return self._extract_stub(ctx)

        from collections import defaultdict

        specs = [
            s for s in specs_for_bucket(ctx.pose_bucket, scope="single")
            if spec_production_ready(s)
        ]
        by_impl: dict[str, list] = defaultdict(list)
        for spec in specs:
            by_impl[spec.implementation].append(spec)

        values: dict[str, float] = {}
        for impl, impl_specs in by_impl.items():
            module = self._modules.get(impl)
            if module is None or not hasattr(module, "compute"):
                continue
            try:
                computed = module.compute(ctx, impl_specs)
                for mv in computed:
                    if mv.value is not None and isinstance(mv.value, (int, float)):
                        values[mv.spec.name] = float(mv.value)
            except Exception:
                continue

        return values

    def extract_landmarks(self, ctx: Any) -> dict[str, float]:
        """Извлечение метрик на основе landmarks (106-точечная модель)."""
        if not _HAS_BACKEND or not hasattr(ctx, 'landmarks_106') or ctx.landmarks_106 is None:
            return {}

        landmarks = ctx.landmarks_106
        if landmarks.size == 0:
            return {}

        result = {}
        # Базовые расстояния между ключевыми точками
        try:
            # Nose tip to chin
            nose_tip = landmarks[76]  # pronasale
            chin = landmarks[8]  # pogonion
            result["landmark_nose_chin_distance"] = float(np.linalg.norm(nose_tip - chin))

            # Eye width (left)
            left_eye_inner = landmarks[35]
            left_eye_outer = landmarks[33]
            result["landmark_eye_width_L"] = float(np.linalg.norm(left_eye_outer - left_eye_inner))

            # Eye width (right)
            right_eye_inner = landmarks[89]
            right_eye_outer = landmarks[87]
            result["landmark_eye_width_R"] = float(np.linalg.norm(right_eye_outer - right_eye_inner))

            # Jaw width
            left_jaw = landmarks[2]
            right_jaw = landmarks[14]
            result["landmark_jaw_width"] = float(np.linalg.norm(right_jaw - left_jaw))
        except (IndexError, KeyError):
            pass

        return result

    def extract_mesh_stats(self, ctx: Any) -> dict[str, float]:
        """Извлечение статистик меша (面積, объём, симметрия)."""
        if not hasattr(ctx, 'vertices_canon') or ctx.vertices_canon is None:
            return {}

        vertices = ctx.vertices_canon
        if vertices.size == 0:
            return {}

        result = {}
        try:
            # bbox stats
            v_min = vertices.min(axis=0)
            v_max = vertices.max(axis=0)
            bbox = v_max - v_min
            result["mesh_bbox_width"] = float(bbox[0])
            result["mesh_bbox_height"] = float(bbox[1])
            result["mesh_bbox_depth"] = float(bbox[2])
            result["mesh_bbox_volume"] = float(bbox[0] * bbox[1] * bbox[2])

            # Face scale (расстояние между скулами)
            if hasattr(ctx, 'macro_indices') and 'cheekbone_L' in ctx.macro_indices and 'cheekbone_R' in ctx.macro_indices:
                idx_l = ctx.macro_indices['cheekbone_L']
                idx_r = ctx.macro_indices['cheekbone_R']
                if isinstance(idx_l, (list, np.ndarray)) and isinstance(idx_r, (list, np.ndarray)):
                    if len(idx_l) > 0 and len(idx_r) > 0:
                        cl = vertices[idx_l[0]] if isinstance(idx_l, np.ndarray) and idx_l.ndim > 0 else vertices[idx_l]
                        cr = vertices[idx_r[0]] if isinstance(idx_r, np.ndarray) and idx_r.ndim > 0 else vertices[idx_r]
                        result["face_scale"] = float(np.linalg.norm(cl - cr))

            # Symmetry (зеркальная асимметрия по X)
            if len(vertices) > 0:
                x_coords = vertices[:, 0]
                result["mesh_symmetry_x"] = float(abs(x_coords.mean()))
        except Exception:
            pass

        return result

    def extract_zone_stats(self, ctx: Any) -> dict[str, float]:
        """Извлечение зональных метрик по анатомическим зонам."""
        if not _HAS_BACKEND or not hasattr(ctx, 'annotation_groups') or not ctx.annotation_groups:
            return {}

        if not hasattr(ctx, 'vertices_canon') or ctx.vertices_canon is None:
            return {}

        vertices = ctx.vertices_canon
        result = {}

        try:
            for zone_idx, zone_vertices in enumerate(ctx.annotation_groups):
                if zone_vertices is None or len(zone_vertices) == 0:
                    continue

                zone_verts = vertices[zone_vertices]
                if zone_verts.size == 0:
                    continue

                prefix = f"zone_{zone_idx}"
                # bbox
                v_min = zone_verts.min(axis=0)
                v_max = zone_verts.max(axis=0)
                bbox = v_max - v_min

                result[f"{prefix}_bbox_volume_ratio"] = float(np.prod(bbox) / max(np.prod(vertices.max(axis=0) - vertices.min(axis=0)), 1e-10))
                result[f"{prefix}_centroid_x"] = float(zone_verts[:, 0].mean())
                result[f"{prefix}_centroid_y"] = float(zone_verts[:, 1].mean())
                result[f"{prefix}_centroid_z"] = float(zone_verts[:, 2].mean())
                result[f"{prefix}_span_x"] = float(bbox[0])
                result[f"{prefix}_span_y"] = float(bbox[1])
                result[f"{prefix}_span_z"] = float(bbox[2])
        except Exception:
            pass

        return result

    def _extract_stub(self, ctx: Any) -> dict[str, float]:
        """Заглушка когда backend недоступен."""
        return {}
