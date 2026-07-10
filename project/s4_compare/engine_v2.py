from __future__ import annotations

from typing import List, Dict, Optional, Any
from pathlib import Path
import numpy as np

from ..shared.utils import load_json
from .alignment import MeshAligner
from .heatmap import DifferenceHeatmap
from .zone_mapper import build_forensic_zone_indices, get_zone_types, get_bone_vertex_indices


# Zone types for bone vs soft tissue classification
ZONE_TYPES = get_zone_types()


class AnchorBasedCompareEngine:
    """Сравнение всех фото в bucket с anchor + all-pairs в пределах 3 лет."""
    
    def __init__(self, config: dict, calibration_discount=None):
        self.config = config
        self.discount = calibration_discount
        self.aligner = MeshAligner()
        self.heatmap = DifferenceHeatmap()
        self.max_date_gap_days = config.get("max_date_gap_days", 365 * 3)  # 3 года
    
    def build_pairwise_evidence(self, main_root: Path, 
                                 reference_path: Optional[Path] = None) -> List[dict]:
        records = self._load_stage2_records(main_root)
        stage1 = self._load_stage1_records(main_root)
        reconstructions = self._load_reconstructions(main_root)
        
        grouped = self._group_by_bucket(records)
        evidence = []
        
        for bucket, bucket_records in grouped.items():
            # 1. Выбираем anchor
            anchor = self._select_anchor(bucket_records, stage1)
            if not anchor:
                continue
            
            anchor_recon = reconstructions.get(anchor.photo_id)
            if not anchor_recon:
                continue
            
            anchor_verts = anchor_recon.get("vertices_canon")
            if anchor_verts is None:
                continue
            
            # 2. Сравниваем каждое фото с anchor
            for record in bucket_records:
                if record.photo_id == anchor.photo_id:
                    continue
                
                pair = self._compare_with_anchor(record, anchor, stage1, bucket, anchor_recon, reconstructions)
                if pair:
                    evidence.append(pair)
            
            # 3. All-pairs для близких по дате (не только соседи)
            ordered = sorted(bucket_records, 
                           key=lambda r: stage1.get(r.photo_id, {}).get("date", "9999") if stage1.get(r.photo_id) else "9999")
            
            for i, a in enumerate(ordered):
                for b in ordered[i+1:]:
                    gap_days = self._date_gap(a, b, stage1)
                    if gap_days < self.max_date_gap_days:
                        pair = self._compare_pair(a, b, stage1, bucket, reconstructions)
                        if pair:
                            evidence.append(pair)
        
        return evidence
    
    def _select_anchor(self, records, stage1) -> Optional[Any]:
        """Выбираем anchor: раннее фото + лучшее качество + нейтральное выражение."""
        scored = []
        for r in records:
            s1 = stage1.get(r.photo_id)
            if not s1:
                continue
            score = 0.0
            # Ранние фото предпочтительнее (ближе к "оригиналу")
            if hasattr(s1, 'date') and s1.date:
                score += max(0, 2020 - s1.date.year) * 0.1
            # Качество
            if hasattr(s1, 'quality') and hasattr(s1.quality, 'overall_quality'):
                score += s1.quality.overall_quality * 2.0
            # Нейтральное выражение
            if hasattr(s1, 'expression_flags') and not s1.expression_flags.get("neutralized", False):
                score += 1.0
            scored.append((score, r))
        
        return max(scored, key=lambda x: x[0])[1] if scored else records[0]
    
    def _compare_with_anchor(self, record, anchor, stage1, bucket, anchor_recon, reconstructions) -> Optional[dict]:
        """Сравнение фото с anchor."""
        rec_a = reconstructions.get(record.photo_id)
        if not rec_a:
            return None
        
        verts_a = rec_a.get("vertices_canon")
        verts_anchor = anchor_recon.get("vertices_canon")
        if verts_a is None or verts_anchor is None:
            return None
        
        # Build forensic zone indices from anchor's annotation_groups
        anchor_groups = anchor_recon.get("annotation_groups", [])
        if anchor_groups:
            zone_indices = build_forensic_zone_indices(verts_anchor, anchor_groups)
        else:
            zone_indices = {}
        
        # ICP alignment
        verts_aligned = self.aligner.icp_refine(verts_a, verts_anchor)
        
        # Per-vertex difference
        diff_per_vertex = np.linalg.norm(verts_aligned - verts_anchor, axis=1)
        raw_distance = float(np.mean(diff_per_vertex))
        
        # Калибровочная скидка
        s1_a = stage1.get(record.photo_id)
        s1_anchor = stage1.get(anchor.photo_id)
        if not s1_a or not s1_anchor:
            return None
        
        pose_gap = self._pose_gap(s1_a, s1_anchor)
        quality = min(
            getattr(s1_a.quality, 'overall_quality', 0.5) if hasattr(s1_a, 'quality') else 0.5,
            getattr(s1_anchor.quality, 'overall_quality', 0.5) if hasattr(s1_anchor, 'quality') else 0.5
        )
        
        discount_result = None
        if self.discount:
            discount_result = self.discount.discount(bucket, raw_distance, pose_gap, quality)
        
        # Heatmap with dynamic zone indices
        heat = self.heatmap.compute(verts_aligned, verts_anchor, zone_indices)
        bone_violations = self.heatmap.count_zone_violations(heat, zone_indices, ZONE_TYPES)
        
        return {
            "pair_id": f"{record.photo_id}__ANCHOR__{anchor.photo_id}",
            "photo_a": record.photo_id,
            "photo_b": anchor.photo_id,
            "anchor_comparison": True,
            "bucket": bucket,
            "raw_distance": raw_distance,
            "excess_distance": discount_result.excess_distance if discount_result else raw_distance,
            "expected_noise": discount_result.expected_noise if discount_result else 0.0,
            "is_significant": discount_result.is_significant if discount_result else raw_distance > 1.0,
            "confidence": discount_result.confidence if discount_result else 0.5,
            "heatmap_mean": float(heat.mean()),
            "heatmap_max": float(heat.max()),
            "bone_zone_violations": bone_violations,
        }
    
    def _compare_pair(self, a, b, stage1, bucket, reconstructions) -> Optional[dict]:
        """Сравнение двух фото (не anchor)."""
        rec_a = reconstructions.get(a.photo_id)
        rec_b = reconstructions.get(b.photo_id)
        if not rec_a or not rec_b:
            return None
        
        verts_a = rec_a.get("vertices_canon")
        verts_b = rec_b.get("vertices_canon")
        if verts_a is None or verts_b is None:
            return None
        
        # Build forensic zone indices from first photo's annotation_groups
        groups_a = rec_a.get("annotation_groups", [])
        if groups_a:
            zone_indices = build_forensic_zone_indices(verts_a, groups_a)
        else:
            zone_indices = {}
        
        # ICP alignment
        verts_aligned = self.aligner.icp_refine(verts_a, verts_b)
        
        # Per-vertex difference
        diff_per_vertex = np.linalg.norm(verts_aligned - verts_b, axis=1)
        raw_distance = float(np.mean(diff_per_vertex))
        
        s1_a = stage1.get(a.photo_id)
        s1_b = stage1.get(b.photo_id)
        if not s1_a or not s1_b:
            return None
        
        pose_gap = self._pose_gap(s1_a, s1_b)
        quality = min(
            getattr(s1_a.quality, 'overall_quality', 0.5) if hasattr(s1_a, 'quality') else 0.5,
            getattr(s1_b.quality, 'overall_quality', 0.5) if hasattr(s1_b, 'quality') else 0.5
        )
        
        discount_result = None
        if self.discount:
            discount_result = self.discount.discount(bucket, raw_distance, pose_gap, quality)
        
        # Heatmap with dynamic zone indices
        heat = self.heatmap.compute(verts_aligned, verts_b, zone_indices)
        bone_violations = self.heatmap.count_zone_violations(heat, zone_indices, ZONE_TYPES)
        
        return {
            "pair_id": f"{a.photo_id}__{b.photo_id}",
            "photo_a": a.photo_id,
            "photo_b": b.photo_id,
            "anchor_comparison": False,
            "bucket": bucket,
            "raw_distance": raw_distance,
            "excess_distance": discount_result.excess_distance if discount_result else raw_distance,
            "expected_noise": discount_result.expected_noise if discount_result else 0.0,
            "is_significant": discount_result.is_significant if discount_result else raw_distance > 1.0,
            "confidence": discount_result.confidence if discount_result else 0.5,
            "heatmap_mean": float(heat.mean()),
            "heatmap_max": float(heat.max()),
            "bone_zone_violations": bone_violations,
        }
    
    def _load_stage2_records(self, root: Path) -> List:
        records = []
        for photo_dir in sorted(root.iterdir()):
            if not photo_dir.is_dir():
                continue
            info = load_json(photo_dir / "info.json")
            geo = load_json(photo_dir / "geometry_metrics.json")
            tex = load_json(photo_dir / "texture_metrics.json")
            if info and isinstance(geo, dict):
                records.append({
                    "photo_id": info.get("photo_id", photo_dir.name),
                    "dataset": info.get("dataset", "main"),
                    "bucket": info.get("pose", {}).get("bucket", "unknown"),
                    "quality": info.get("quality", {}),
                    "geometry": geo,
                    "texture": tex or {},
                })
        return records
    
    def _load_stage1_records(self, root: Path) -> Dict:
        records = {}
        for path in sorted(root.glob("*/info.json")):
            payload = load_json(path)
            if payload:
                records[payload.get("photo_id", path.parent.name)] = payload
        return records
    
    def _load_reconstructions(self, root: Path) -> Dict:
        reconstructions = {}
        for path in sorted(root.glob("*/reconstruction.pkl")):
            try:
                photo_id = path.parent.name
                import pickle
                with open(path, 'rb') as f:
                    recon = pickle.load(f)
                reconstructions[photo_id] = recon
            except Exception as e:
                pass
        return reconstructions
    
    def _group_by_bucket(self, records) -> Dict[str, List]:
        from collections import defaultdict
        grouped = defaultdict(list)
        for record in records:
            bucket = record.get("bucket", "unknown")
            grouped[bucket].append(record)
        return grouped
    
    def _date_gap(self, a, b, stage1) -> int:
        s1_a = stage1.get(a.photo_id)
        s1_b = stage1.get(b.photo_id)
        if not s1_a or not s1_b:
            return 9999
        date_a = s1_a.get("date") if isinstance(s1_a, dict) else getattr(s1_a, 'date', None)
        date_b = s1_b.get("date") if isinstance(s1_b, dict) else getattr(s1_b, 'date', None)
        if not date_a or not date_b:
            return 9999
        from datetime import date
        if isinstance(date_a, str):
            date_a = date.fromisoformat(date_a)
        if isinstance(date_b, str):
            date_b = date.fromisoformat(date_b)
        return abs((date_a - date_b).days)
    
    def _pose_gap(self, a, b) -> float:
        pose_a = a.get("pose", {}) if isinstance(a, dict) else getattr(a, 'pose', None)
        pose_b = b.get("pose", {}) if isinstance(b, dict) else getattr(b, 'pose', None)
        if not pose_a or not pose_b:
            return 0.0
        dy = abs(pose_a.get("yaw", 0) - pose_b.get("yaw", 0))
        dp = abs(pose_a.get("pitch", 0) - pose_b.get("pitch", 0))
        dr = abs(pose_a.get("roll", 0) - pose_b.get("roll", 0))
        return float(np.sqrt((1.4 * dy) ** 2 + dp ** 2 + (0.6 * dr) ** 2))