import logging
import os
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class UVBakerConfig:
    interpolation: int = cv2.INTER_LANCZOS4
    border_mode: int = cv2.BORDER_REPLICATE


logger = logging.getLogger(__name__)


class UVBaker:
    """
    High-res UV baker: per-triangle affine warp image->UV.
    На выходе даёт:
        - uv_texture_raw      (work_size x work_size x 3, float32)
        - uv_mask_visible     (work_size x work_size, bool)
        - uv_confidence_map   (work_size x work_size, float32)
        - uv_is_original      (work_size x work_size, bool)
    """

    def __init__(
        self,
        uv_size: int = 2048,
        super_sample: int = 2,
        config: UVBakerConfig | None = None,
    ) -> None:
        self.uv_size = int(uv_size)
        self.super_sample = int(super_sample)
        self.work_size = self.uv_size * self.super_sample
        self.config = config or UVBakerConfig()

    def bake(
        self,
        image: np.ndarray,
        vertices_2d: np.ndarray,
        uv_coords: np.ndarray,
        triangles: np.ndarray,
        tri_visibility_weights: np.ndarray,
        debug_output_dir: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        h, w = image.shape[:2]
        work = self.work_size

        uv_px = self._prepare_uv_px(uv_coords, work)
        tris = np.asarray(triangles, dtype=np.int64)
        verts2d = np.asarray(vertices_2d, dtype=np.float32)
        w_tri = np.asarray(tri_visibility_weights, dtype=np.float32).reshape(-1)

        if debug_output_dir:
            os.makedirs(debug_output_dir, exist_ok=True)
            debug_uv_mesh = np.zeros((work, work, 3), dtype=np.uint8)
            for i in range(tris.shape[0]):
                idx0, idx1, idx2 = tris[i]
                pts = uv_px[[idx0, idx1, idx2]].astype(np.int32).reshape(-1, 1, 2)
                cv2.fillConvexPoly(debug_uv_mesh, pts, (0, 100, 0))
                cv2.polylines(debug_uv_mesh, [pts], isClosed=True, color=(0, 255, 0), thickness=1)
            mesh_path = os.path.join(debug_output_dir, "debug_uv_mesh.png")
            cv2.imwrite(mesh_path, debug_uv_mesh)
            filled_mesh = int((debug_uv_mesh[:, :, 1] > 0).sum())
            logger.debug(
                "[BAKER] debug_uv_mesh saved -> %s  (filled %d / %d = %.1f%%)",
                mesh_path,
                filled_mesh,
                work * work,
                100.0 * filled_mesh / (work * work),
            )

        color_accum = np.zeros((work, work, 3), dtype=np.float64)
        weight_accum = np.zeros((work, work), dtype=np.float64)

        for i in range(tris.shape[0]):
            w_i = float(w_tri[i])
            if w_i <= 0.0:
                continue

            idx0, idx1, idx2 = tris[i]
            src_pts = verts2d[[idx0, idx1, idx2]]  # image space
            dst_pts = uv_px[[idx0, idx1, idx2]]  # UV pixel space

            if self._tri_area(dst_pts) < 0.01 or self._tri_area(src_pts) < 0.01:
                continue

            M = np.asarray(
                cv2.getAffineTransform(dst_pts.astype(np.float32), src_pts.astype(np.float32)),
                dtype=np.float64,
            )

            bb_min = np.floor(dst_pts.min(axis=0)).astype(int)
            bb_max = np.ceil(dst_pts.max(axis=0)).astype(int)
            bb_min = np.clip(bb_min, 0, work - 1)
            bb_max = np.clip(bb_max, 0, work - 1)
            if bb_min[0] >= bb_max[0] or bb_min[1] >= bb_max[1]:
                continue

            bw = bb_max[0] - bb_min[0] + 1
            bh = bb_max[1] - bb_min[1] + 1
            local_pts = dst_pts - bb_min.astype(np.float32)

            mask = np.zeros((bh, bw), dtype=np.uint8)
            pts_fixed = np.round(local_pts * 16).astype(np.int32).reshape(-1, 1, 2)
            cv2.fillConvexPoly(mask, pts_fixed, (1,), lineType=cv2.LINE_8, shift=4)

            ys, xs = np.where(mask > 0)
            if xs.size == 0:
                continue

            uv_pixels = np.stack(
                [
                    xs + bb_min[0],
                    ys + bb_min[1],
                    np.ones_like(xs, dtype=np.float64),
                ],
                axis=0,
            )
            img_coords = M @ uv_pixels
            ix = img_coords[0]
            iy = img_coords[1]

            sample_x = ix.astype(np.float32).reshape(1, -1)
            sample_y = iy.astype(np.float32).reshape(1, -1)
            sampled = cv2.remap(
                image,
                sample_x,
                sample_y,
                self.config.interpolation,
                borderMode=self.config.border_mode,
            )
            colors = sampled.reshape(-1, 3).astype(np.float64)

            write_y = ys + bb_min[1]
            write_x = xs + bb_min[0]

            in_bounds = (ix >= 0.0) & (ix < w) & (iy >= 0.0) & (iy < h)
            effective_weight = w_i * in_bounds.astype(np.float64)
            color_accum[write_y, write_x] += colors * effective_weight[:, None]
            weight_accum[write_y, write_x] += effective_weight

        mask_visible = weight_accum > 0.0
        uv_texture_raw = np.zeros((work, work, 3), dtype=np.float32)
        uv_texture_raw[mask_visible] = (
            color_accum[mask_visible] / weight_accum[mask_visible, None]
        ).astype(np.float32)

        uv_confidence = np.zeros((work, work), dtype=np.float32)
        if np.any(mask_visible):
            max_w = float(weight_accum[mask_visible].max())
            if max_w > 0.0:
                uv_confidence[mask_visible] = (
                    weight_accum[mask_visible] / max_w
                ).astype(np.float32)

        uv_is_original = mask_visible.copy()

        if self.super_sample > 1:
            uv_texture_raw, mask_visible, uv_confidence, uv_is_original = self._downsample_results(
                uv_texture_raw, mask_visible, uv_confidence, uv_is_original
            )

        total_uv_pixels = work * work
        filled_pixels = int(mask_visible.sum())
        logger.debug("[BAKER] work_size=%d", work)
        logger.debug("[BAKER] total triangles:     %d", tris.shape[0])
        logger.debug("[BAKER] triangles with w>0:  %d", int((w_tri > 0).sum()))
        logger.debug(
            "[BAKER] UV pixels filled:    %d / %d  (%.1f%%)",
            filled_pixels,
            total_uv_pixels,
            100.0 * filled_pixels / max(total_uv_pixels, 1),
        )
        logger.debug("[BAKER] weight_accum max:    %.4f", float(weight_accum.max()))
        logger.debug("[BAKER] color_accum max:     %.1f", float(color_accum.max()))

        return uv_texture_raw, mask_visible, uv_confidence, uv_is_original

    def bake_via_barycentric(
        self,
        image: np.ndarray,
        vertices_2d: np.ndarray,
        uv_coords: np.ndarray,
        triangles: np.ndarray,
        tri_visibility_weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Бейкинг через растеризацию fillConvexPoly + барицентрическая интерполяция.
        Надёжнее per-triangle affine: без щелей между треугольниками, один remap в конце.
        Возвращает (uv_texture_raw, mask_visible, uv_confidence, uv_is_original) как bake().
        """
        h, w = image.shape[:2]
        work = self.work_size
        uv_px = self._prepare_uv_px(uv_coords, work)
        tris = np.asarray(triangles, dtype=np.int64)
        verts2d = np.asarray(vertices_2d, dtype=np.float32)
        w_tri = np.asarray(tri_visibility_weights, dtype=np.float32).reshape(-1)

        map_x_accum = np.zeros((work, work), dtype=np.float64)
        map_y_accum = np.zeros((work, work), dtype=np.float64)
        weight_accum = np.zeros((work, work), dtype=np.float64)

        for i in range(tris.shape[0]):
            w_i = float(w_tri[i])
            if w_i <= 0.0:
                continue

            idx0, idx1, idx2 = tris[i]
            uv_tri = uv_px[[idx0, idx1, idx2]].astype(np.float64)
            img_tri = verts2d[[idx0, idx1, idx2]].astype(np.float64)

            bb_min = np.floor(uv_tri.min(axis=0)).astype(int)
            bb_max = np.ceil(uv_tri.max(axis=0)).astype(int)
            bb_min = np.clip(bb_min, 0, work - 1)
            bb_max = np.clip(bb_max, 0, work - 1)
            if bb_min[0] > bb_max[0] or bb_min[1] > bb_max[1]:
                continue

            bw = bb_max[0] - bb_min[0] + 1
            bh = bb_max[1] - bb_min[1] + 1
            
            v0 = uv_tri[2] - uv_tri[0]
            v1 = uv_tri[1] - uv_tri[0]
            d00 = float(v0 @ v0)
            d01 = float(v0 @ v1)
            d11 = float(v1 @ v1)
            denom = d00 * d11 - d01 * d01
            if abs(denom) < 1e-10:
                continue

            px_local = np.arange(bb_min[0], bb_max[0] + 1, dtype=np.float64)
            py_local = np.arange(bb_min[1], bb_max[1] + 1, dtype=np.float64)
            gx, gy = np.meshgrid(px_local, py_local)
            xs_all = gx.ravel()
            ys_all = gy.ravel()

            v2_x = xs_all - uv_tri[0, 0]
            v2_y = ys_all - uv_tri[0, 1]
            
            d20 = v2_x * v0[0] + v2_y * v0[1]
            d21 = v2_x * v1[0] + v2_y * v1[1]
            
            bary_v = (d11 * d20 - d01 * d21) / denom
            bary_w = (d00 * d21 - d01 * d20) / denom
            bary_u = 1.0 - bary_v - bary_w

            # Eps buffer avoids seam gaps
            eps = 1e-4
            inside = (bary_u >= -eps) & (bary_v >= -eps) & (bary_w >= -eps)
            if not inside.any():
                continue
                
            xs = xs_all[inside].astype(np.int32)
            ys = ys_all[inside].astype(np.int32)
            bary_u = bary_u[inside]
            bary_v = bary_v[inside]
            bary_w = bary_w[inside]

            img_x = bary_u * img_tri[0, 0] + bary_w * img_tri[1, 0] + bary_v * img_tri[2, 0]
            img_y = bary_u * img_tri[0, 1] + bary_w * img_tri[1, 1] + bary_v * img_tri[2, 1]

            in_bounds = (img_x >= 0.0) & (img_x <= w - 1) & (img_y >= 0.0) & (img_y <= h - 1)
            
            ys_in = ys[in_bounds]
            xs_in = xs[in_bounds]
            
            map_x_accum[ys_in, xs_in] += img_x[in_bounds] * w_i
            map_y_accum[ys_in, xs_in] += img_y[in_bounds] * w_i
            weight_accum[ys_in, xs_in] += w_i

        valid = weight_accum > 0.0
        filled = int(valid.sum())
        total = work * work
        logger.debug(
            "[BAKER] bake_via_barycentric: filled %d / %d (%.1f%%)",
            filled,
            total,
            100.0 * filled / max(total, 1),
        )

        map_x_final = np.zeros((work, work), dtype=np.float32)
        map_y_final = np.zeros((work, work), dtype=np.float32)
        if filled > 0:
            map_x_final[valid] = np.clip(
                (map_x_accum[valid] / weight_accum[valid]).astype(np.float32),
                0.0,
                float(w - 1),
            )
            map_y_final[valid] = np.clip(
                (map_y_accum[valid] / weight_accum[valid]).astype(np.float32),
                0.0,
                float(h - 1),
            )

        uv_texture_raw = cv2.remap(
            image,
            map_x_final,
            map_y_final,
            self.config.interpolation,
            borderMode=self.config.border_mode,
        ).astype(np.float32)
        uv_texture_raw[~valid] = 0.0

        uv_confidence = np.zeros((work, work), dtype=np.float32)
        if filled > 0:
            max_w = weight_accum[valid].max()
            if max_w > 0:
                uv_confidence[valid] = (weight_accum[valid] / max_w).astype(np.float32)

        uv_is_original = valid.copy()

        if self.super_sample > 1:
            uv_texture_raw, valid, uv_confidence, uv_is_original = self._downsample_results(
                uv_texture_raw, valid, uv_confidence, uv_is_original
            )

        return uv_texture_raw, valid, uv_confidence, uv_is_original

    def bake_vertex_colors(
        self,
        uv_coords: np.ndarray,
        triangles: np.ndarray,
        vertex_colors: np.ndarray,
        size: int | None = None,
    ) -> np.ndarray:
        """
        Растеризация вершинных цветов в UV без сэмплинга из изображения.
        Возвращает стандартную UV-карту (H, W, 3) в [0, 1], float32.
        size: разрешение карты (по умолчанию uv_size, без super_sample).
        """
        work = int(size) if size is not None else self.uv_size
        uv_px = self._prepare_uv_px(uv_coords, work)

        color_accum = np.zeros((work, work, 3), dtype=np.float64)
        weight_accum = np.zeros((work, work), dtype=np.float64)
        tris = np.asarray(triangles, dtype=np.int64)
        vcolors = np.asarray(vertex_colors, dtype=np.float64)
        if vcolors.ndim == 3:
            vcolors = vcolors[0]
        vcolors = np.clip(vcolors, 0.0, 1.0)

        for i in range(tris.shape[0]):
            idx0, idx1, idx2 = tris[i]
            dst_pts = uv_px[[idx0, idx1, idx2]].astype(np.float64)
            if self._tri_area(dst_pts) < 0.01:
                continue

            bb_min = np.floor(dst_pts.min(axis=0)).astype(int)
            bb_max = np.ceil(dst_pts.max(axis=0)).astype(int)
            bb_min = np.clip(bb_min, 0, work - 1)
            bb_max = np.clip(bb_max, 0, work - 1)
            if bb_min[0] > bb_max[0] or bb_min[1] > bb_max[1]:
                continue

            v0 = dst_pts[2] - dst_pts[0]
            v1 = dst_pts[1] - dst_pts[0]
            d00 = float(v0 @ v0)
            d01 = float(v0 @ v1)
            d11 = float(v1 @ v1)
            denom = d00 * d11 - d01 * d01
            if abs(denom) < 1e-10:
                continue

            px_local = np.arange(bb_min[0], bb_max[0] + 1, dtype=np.float64)
            py_local = np.arange(bb_min[1], bb_max[1] + 1, dtype=np.float64)
            gx, gy = np.meshgrid(px_local, py_local)
            xs_all = gx.ravel()
            ys_all = gy.ravel()

            v2_x = xs_all - dst_pts[0, 0]
            v2_y = ys_all - dst_pts[0, 1]
            
            d20 = v2_x * v0[0] + v2_y * v0[1]
            d21 = v2_x * v1[0] + v2_y * v1[1]
            
            bary_v = (d11 * d20 - d01 * d21) / denom
            bary_w = (d00 * d21 - d01 * d20) / denom
            bary_u = 1.0 - bary_v - bary_w

            # Eps buffer avoids seam gaps
            eps = 1e-4
            inside = (bary_u >= -eps) & (bary_v >= -eps) & (bary_w >= -eps)
            if not inside.any():
                continue
                
            xs = xs_all[inside].astype(np.int32)
            ys = ys_all[inside].astype(np.int32)
            bary_u = bary_u[inside]
            bary_v = bary_v[inside]
            bary_w = bary_w[inside]

            c0, c1, c2 = vcolors[idx0], vcolors[idx1], vcolors[idx2]
            colors = (
                bary_u[:, None] * c0[None, :]
                + bary_w[:, None] * c1[None, :]
                + bary_v[:, None] * c2[None, :]
            )

            wy = ys
            wx = xs
            color_accum[wy, wx] += colors
            weight_accum[wy, wx] += 1.0

        out = np.zeros((work, work, 3), dtype=np.float32)
        mask_visible = weight_accum > 0.0
        out[mask_visible] = (
            color_accum[mask_visible] / weight_accum[mask_visible, None]
        ).astype(np.float32)
        return np.clip(out, 0.0, 1.0)

    @staticmethod
    def _tri_area(pts: np.ndarray) -> float:
        d1 = pts[1] - pts[0]
        d2 = pts[2] - pts[0]
        return float(abs(d1[0] * d2[1] - d1[1] * d2[0]) * 0.5)

    @staticmethod
    def _tri_areas_batch(pts: np.ndarray) -> np.ndarray:
        d1 = pts[:, 1] - pts[:, 0]
        d2 = pts[:, 2] - pts[:, 0]
        return np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) * 0.5

    def _prepare_uv_px(self, uv_coords: np.ndarray, size: int) -> np.ndarray:
        uv = np.asarray(uv_coords, dtype=np.float32)
        if uv.ndim != 2 or uv.shape[1] < 2:
            raise ValueError("uv_coords must be (N, 2+) array")
        if uv.shape[1] > 2:
            uv = uv[:, :2]
        else:
            uv = uv.copy()

        uv_max = float(uv.max(initial=0.0))
        if uv_max > 1.5:
            denom = max(uv_max, 1e-6)
            uv = uv / denom

        uv_px = np.empty((uv.shape[0], 2), dtype=np.float32)
        uv_px[:, 0] = uv[:, 0] * (size - 1)
        uv_px[:, 1] = (1.0 - uv[:, 1]) * (size - 1)
        return uv_px

    def _downsample_results(
        self,
        uv_texture_raw: np.ndarray,
        mask_visible: np.ndarray,
        uv_confidence: np.ndarray,
        uv_is_original: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        target = self.uv_size
        texture = cv2.resize(uv_texture_raw, (target, target), interpolation=cv2.INTER_AREA)
        mask = (
            cv2.resize(mask_visible.astype(np.uint8), (target, target), interpolation=cv2.INTER_NEAREST) > 0
        )
        confidence = cv2.resize(uv_confidence, (target, target), interpolation=cv2.INTER_AREA)
        is_original = (
            cv2.resize(uv_is_original.astype(np.uint8), (target, target), interpolation=cv2.INTER_NEAREST) > 0
        )
        return texture, mask, confidence, is_original

