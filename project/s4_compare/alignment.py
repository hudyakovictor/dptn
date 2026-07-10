from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree
from typing import Dict, List, Optional


class MeshAligner:
    """Выравнивает меш B к мешу A по костным вершинам."""
    
    BONE_VERTEX_INDICES = []  # Индексы костных вершин из Basel Face Model
    
    def __init__(self, bone_indices: Optional[List[int]] = None):
        if bone_indices:
            self.BONE_VERTEX_INDICES = bone_indices
    
    def procrustes_align(self, verts_b: np.ndarray, verts_a: np.ndarray,
                         use_bones_only: bool = True) -> np.ndarray:
        """
        Returns: verts_b_aligned
        """
        if use_bones_only and self.BONE_VERTEX_INDICES:
            idx = self.BONE_VERTEX_INDICES
            src = verts_b[idx]
            tgt = verts_a[idx]
        else:
            src = verts_b
            tgt = verts_a
        
        # Центрирование
        src_mean = src.mean(axis=0)
        tgt_mean = tgt.mean(axis=0)
        src_c = src - src_mean
        tgt_c = tgt - tgt_mean
        
        # Масштаб
        src_scale = np.linalg.norm(src_c)
        tgt_scale = np.linalg.norm(tgt_c)
        scale = tgt_scale / src_scale if src_scale > 0 else 1.0
        
        # Поворот (SVD)
        H = src_c.T @ tgt_c
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Коррекция отражения
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Применяем ко всему мешу
        aligned = (verts_b - src_mean) @ R * scale + tgt_mean
        return aligned
    
    def icp_refine(self, verts_b: np.ndarray, verts_a: np.ndarray,
                   max_iter: int = 10, threshold: float = 0.001) -> np.ndarray:
        """Уточняет alignment итеративно."""
        aligned = verts_b.copy()
        tree = KDTree(verts_a)
        
        for _ in range(max_iter):
            dists, indices = tree.query(aligned)
            # Фильтруем outliers (расстояние > 3*median)
            median_dist = np.median(dists)
            mask = dists < 3 * median_dist
            
            if mask.sum() < 100:
                break
            
            src = aligned[mask]
            tgt = verts_a[indices[mask]]
            
            # Procrustes на inliers
            src_mean = src.mean(axis=0)
            tgt_mean = tgt.mean(axis=0)
            src_c = src - src_mean
            tgt_c = tgt - tgt_mean
            
            H = src_c.T @ tgt_c
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            
            scale = np.linalg.norm(tgt_c) / (np.linalg.norm(src_c) + 1e-8)
            
            new_aligned = (aligned - src_mean) @ R * scale + tgt_mean
            shift = np.linalg.norm(new_aligned - aligned)
            aligned = new_aligned
            
            if shift < threshold:
                break
        
        return aligned
    
    def align_and_compare(self, verts_a: np.ndarray, verts_b: np.ndarray,
                          bone_indices: Optional[List[int]] = None) -> Dict:
        """
        Полный пайплайн: Procrustes -> ICP -> per-vertex difference.
        Returns dict with aligned vertices, distances, and stats.
        """
        if bone_indices:
            self.BONE_VERTEX_INDICES = bone_indices
        
        # 1. Procrustes on bone vertices
        aligned = self.procrustes_align(verts_b, verts_a, use_bones_only=True)
        
        # 2. ICP refinement
        aligned = self.icp_refine(aligned, verts_a)
        
        # 3. Per-vertex difference
        diff_per_vertex = np.linalg.norm(aligned - verts_a, axis=1)
        
        return {
            "aligned_vertices": aligned,
            "diff_per_vertex": diff_per_vertex,
            "mean_distance": float(diff_per_vertex.mean()),
            "max_distance": float(diff_per_vertex.max()),
            "median_distance": float(np.median(diff_per_vertex)),
        }