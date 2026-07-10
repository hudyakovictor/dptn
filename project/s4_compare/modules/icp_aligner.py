from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class ICPAligner:
    """Robust rigid alignment for two meshes or point clouds."""

    @dataclass
    class Result:
        rotation: np.ndarray
        translation: np.ndarray
        scale: float
        residual: float
        residual_median: float
        inliers: int
        iterations: int
        aligned_vertices: np.ndarray
        status: str

    def align(self, a_mesh, b_mesh, shared_indices: np.ndarray | None = None) -> dict[str, Any]:
        a = self._extract_vertices(a_mesh)
        b = self._extract_vertices(b_mesh)
        if a.size == 0 or b.size == 0:
            return self._empty_result("insufficient_vertices")

        if shared_indices is not None:
            idx = np.asarray(shared_indices, dtype=int).reshape(-1)
            if idx.size >= 4:
                valid = idx[(idx >= 0) & (idx < len(a)) & (idx < len(b))]
                if valid.size >= 4:
                    a = a[valid]
                    b = b[valid]

        n = min(len(a), len(b))
        if n < 4:
            return self._empty_result("insufficient_points")

        a = a[:n]
        b = b[:n]
        rot = np.eye(3, dtype=float)
        scale = 1.0
        residuals = np.zeros(len(a), dtype=float)
        aligned = a.copy()
        iterations = 0

        for _ in range(3):
            iterations += 1
            a_centroid = np.mean(a, axis=0)
            b_centroid = np.mean(b, axis=0)
            a0 = a - a_centroid
            b0 = b - b_centroid
            cov = a0.T @ b0
            u, s, vh = np.linalg.svd(cov)
            det = np.sign(np.linalg.det(u @ vh)) or 1.0
            rot = u @ np.diag([1.0, 1.0, det]) @ vh
            scale = float(np.sum(s) / max(np.sum(a0**2), 1e-6))
            scale = float(np.clip(scale, 0.75, 1.35))
            aligned = (a0 * scale) @ rot + b_centroid
            residuals = np.linalg.norm(aligned - b, axis=1)

            if residuals.size < 6:
                break
            med = float(np.median(residuals))
            mad = float(np.median(np.abs(residuals - med))) + 1e-6
            keep = residuals <= (med + 1.75 * mad)
            if keep.sum() < 4 or keep.all():
                break
            a = a[keep]
            b = b[keep]
            if len(a) < 4:
                break

        med = float(np.median(residuals)) if residuals.size else 0.0
        mad = float(np.median(np.abs(residuals - med))) if residuals.size else 0.0
        threshold = med + 1.75 * (mad + 1e-6)
        inliers = int(np.count_nonzero(residuals <= threshold))
        return {
            "rotation": rot,
            "translation": b_centroid - (a_centroid * scale) @ rot,
            "scale": scale,
            "residual": float(np.mean(residuals)),
            "residual_median": med,
            "inliers": inliers,
            "iterations": iterations,
            "aligned_vertices": aligned,
            "status": "ok",
        }

    def _extract_vertices(self, mesh: Any) -> np.ndarray:
        if mesh is None:
            return np.empty((0, 3), dtype=float)
        if isinstance(mesh, dict):
            verts = mesh.get("vertices")
            if verts is None:
                verts = mesh.get("verts")
            if verts is None:
                verts = mesh.get("points")
            if verts is None:
                verts = mesh.get("aligned_vertices")
        else:
            verts = mesh
        arr = np.asarray(verts, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 3:
            return np.empty((0, 3), dtype=float)
        return arr[:, :3]

    def _empty_result(self, status: str) -> dict[str, Any]:
        return {
            "rotation": np.eye(3, dtype=float),
            "translation": np.zeros(3, dtype=float),
            "scale": 1.0,
            "residual": 0.0,
            "residual_median": 0.0,
            "inliers": 0,
            "iterations": 0,
            "aligned_vertices": np.empty((0, 3), dtype=float),
            "status": status,
        }
