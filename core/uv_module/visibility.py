"""Utilities for estimating visibility weights for 3DDFA_V3 triangles."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

__all__ = ["compute_triangle_visibility", "compute_vertex_visibility"]

logger = logging.getLogger(__name__)


def compute_triangle_visibility(
    vertices_3d: np.ndarray,
    triangles: np.ndarray,
    view_dir: Optional[np.ndarray] = None,
    angle_threshold_deg: float = 85.0,
    gamma: float = 1.5,
    use_zbuffer: bool = False,
    vertices_2d: Optional[np.ndarray] = None,
    image_size: Optional[tuple[int, int]] = None,
    z_tolerance: float = 1e-3,
    occlusion_falloff: float = 0.1,
) -> np.ndarray:
    """Return per-triangle visibility weights in the [0, 1] range.

    Args:
        vertices_3d: (N, 3) vertex positions in camera space.
        triangles: (T, 3) triangle indices.
        view_dir: Optional (3,) view direction. Defaults to +Z.
        angle_threshold_deg: Maximum allowed angle between normal and view dir.
        gamma: Falloff exponent applied to cos(angle).
        use_zbuffer: Enable lightweight centroid z-buffer test.
        vertices_2d: (N, 2) projected vertices (required for z-buffer mode).
        image_size: (H, W) image resolution for z-buffer grid.
        z_tolerance: Allowed depth slack before marking as occluded.
        occlusion_falloff: Multiplier applied to occluded triangles.

    Returns:
        (T,) float32 weights.
    """

    verts = np.asarray(vertices_3d, dtype=np.float32)
    tris = np.asarray(triangles, dtype=np.int64)
    if tris.ndim != 2 or tris.shape[1] != 3:
        raise ValueError("triangles must have shape (T, 3)")

    if view_dir is None:
        view_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        view_dir = np.asarray(view_dir, dtype=np.float32)

    view_norm = np.linalg.norm(view_dir)
    if view_norm < 1e-8:
        raise ValueError("view_dir magnitude is too small")
    view = view_dir / view_norm

    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]

    normals = np.cross(v1 - v0, v2 - v0)
    norm_len = np.linalg.norm(normals, axis=1, keepdims=True)
    valid_normals = norm_len.squeeze(-1) > 1e-8
    normals = np.divide(
        normals,
        np.maximum(norm_len, 1e-8),
        out=np.zeros_like(normals),
        where=norm_len > 1e-8,
    )

    cos_angle = np.clip((normals @ view).astype(np.float32), -1.0, 1.0)
    mask = valid_normals

    weights = np.zeros(tris.shape[0], dtype=np.float32)
    # Даем базовый минимальный вес 0.001 даже почти перпендикулярным или отвернутым участкам (чтобы избежать дыр в сетке)
    weights[mask] = np.maximum(np.power(np.maximum(cos_angle[mask], 0.0), gamma), 0.001).astype(
        np.float32
    )

    if use_zbuffer:
        if vertices_2d is None or image_size is None:
            logger.warning("Z-buffer requested but vertices_2d/image_size missing")
        else:
            _apply_centroid_zbuffer(
                weights=weights,
                mask=mask,
                verts=verts,
                tris=tris,
                vertices_2d=vertices_2d,
                image_size=image_size,
                z_tolerance=z_tolerance,
                occlusion_falloff=occlusion_falloff,
            )

    logger.debug(
        "Visibility: %d/%d total valid normal tris processed for occlusion",
        int(mask.sum()),
        tris.shape[0],
    )

    return weights


def _apply_centroid_zbuffer(
    weights: np.ndarray,
    mask: np.ndarray,
    verts: np.ndarray,
    tris: np.ndarray,
    vertices_2d: np.ndarray,
    image_size: tuple[int, int],
    z_tolerance: float,
    occlusion_falloff: float,
) -> None:
    """Lightweight centroid-based occlusion test."""

    h, w = int(image_size[0]), int(image_size[1])
    if h <= 0 or w <= 0:
        return

    verts_2d = np.asarray(vertices_2d, dtype=np.float32)
    centroids_2d = verts_2d[tris].mean(axis=1)
    centroids_z = verts[tris].mean(axis=1)[:, 2]

    cx = np.clip(np.rint(centroids_2d[:, 0]).astype(int), 0, w - 1)
    cy = np.clip(np.rint(centroids_2d[:, 1]).astype(int), 0, h - 1)

    zbuffer = np.full((h, w), np.inf, dtype=np.float32)

    for idx in np.where(mask)[0]:
        z = centroids_z[idx]
        x, y = cx[idx], cy[idx]
        if z < zbuffer[y, x]:
            zbuffer[y, x] = z

    for idx in np.where(mask)[0]:
        z = centroids_z[idx]
        x, y = cx[idx], cy[idx]
        if z > zbuffer[y, x] + z_tolerance:
            weights[idx] *= occlusion_falloff


def compute_vertex_visibility(
    triangles: np.ndarray,
    tri_weights: np.ndarray,
    num_vertices: Optional[int] = None,
) -> np.ndarray:
    """Map per-triangle weights to per-vertex visibility via averaging."""

    tris = np.asarray(triangles, dtype=np.int64)
    weights = np.asarray(tri_weights, dtype=np.float32).reshape(-1)
    if tris.shape[0] != weights.shape[0]:
        raise ValueError("triangles and tri_weights must have the same length")

    if num_vertices is None:
        num_vertices = int(tris.max()) + 1 if tris.size > 0 else 0

    vert_sum = np.zeros(num_vertices, dtype=np.float64)
    vert_count = np.zeros(num_vertices, dtype=np.float64)

    for j in range(3):
        np.add.at(vert_sum, tris[:, j], weights)
        np.add.at(vert_count, tris[:, j], 1.0)

    vert_count = np.maximum(vert_count, 1e-6)
    return (vert_sum / vert_count).astype(np.float32)
