from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..shared.logging import setup_logger
from ..shared.schemas import CalibrationReference, PairEvidence, PipelineDataset, Stage1Record, Stage2Record
from ..shared.utils import load_json, load_pickle, save_json
from ..s1_extraction.expression_analyzer import ExpressionAnalyzer3D, ExpressionNormalizedComparator
from .zone_mapper import build_forensic_zone_indices, get_zone_types

logger = setup_logger("deeputin.s4")


def procrustes_align(source: np.ndarray, target: np.ndarray, weights: Optional[np.ndarray] = None, 
                     allow_scale: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Weighted Procrustes alignment of source to target.
    Returns: (aligned_source, R, t, scale)
    """
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must be (N, 3) arrays of same shape")
    
    if weights is None:
        weights = np.ones(len(source), dtype=np.float32)
    
    # Weighted centroids
    w_sum = weights.sum()
    if w_sum <= 1e-8:
        raise ValueError("Weight sum too small")
    
    w_norm = weights / w_sum
    w_norm = w_norm[:, np.newaxis]
    
    source_centroid = np.sum(source * w_norm, axis=0)
    target_centroid = np.sum(target * w_norm, axis=0)
    
    source_centered = source - source_centroid
    target_centered = target - target_centroid
    
    # Weighted covariance
    H = (source_centered * w_norm).T @ target_centered
    
    # SVD
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Fix reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    if allow_scale:
        var_source = np.sum(weights * np.sum(source_centered**2, axis=1))
        if var_source > 1e-8:
            scale = float(np.sum(S) / var_source)
        else:
            scale = 1.0
    else:
        scale = 1.0
    
    t = target_centroid - scale * (source_centroid @ R)
    aligned = scale * (source @ R) + t
    
    return aligned, R, t, scale


def compute_heatmap(source_aligned: np.ndarray, target: np.ndarray, 
                    zone_indices: Dict[str, np.ndarray],
                    face_scale: float) -> Dict[str, float]:
    """Compute per-zone heatmap distances."""
    if source_aligned.shape != target.shape:
        return {}
    
    diff = np.linalg.norm(source_aligned - target, axis=1) / face_scale
    
    heat = {}
    for zone_name, indices in zone_indices.items():
        if len(indices) == 0:
            continue
        # Bone zones: strict threshold (2mm), soft zones: loose (5mm)
        is_bone = any(b in zone_name for b in ["bone", "nasion", "orbit", "zygomatic", "gonial", "chin"])
        threshold = 0.002 if is_bone else 0.005  # normalized by face_scale
        zone_diff = diff[indices]
        heat[zone_name] = float(np.clip(zone_diff.mean() / threshold, 0.0, 1.0))
    
    return heat


class CompareEngine:
    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}
        self.anchor_every_n = self.config.get("anchor_every_n", 10)  # compare with anchors
        self.max_anchors = self.config.get("max_anchors", 5)
        self.expr_analyzer = ExpressionAnalyzer3D()
        self.expr_comparator = ExpressionNormalizedComparator()

    def build_pairwise_evidence(self, main_root: str | Path, reference_path: str | Path | None = None) -> list[PairEvidence]:
        root = Path(main_root)
        records = self._load_stage2_records(root)
        if not records:
            logger.warning("Нет stage2 записей для сравнения в %s", root)
            return []

        reference = CalibrationReference.model_validate(load_json(reference_path)) if reference_path and Path(reference_path).exists() else None
        stage1_records = self._load_stage1_records(root)
        
        # Load all 3D reconstructions for ICP
        reconstructions = self._load_all_reconstructions(root)
        
        grouped: dict[str, list[Stage2Record]] = defaultdict(list)
        for record in records:
            grouped[record.bucket.value].append(record)

        evidence: list[PairEvidence] = []
        pair_index: dict[str, list[dict[str, object]]] = defaultdict(list)
        
        for bucket, bucket_records in grouped.items():
            ordered = sorted(bucket_records, key=lambda rec: self._sort_key(rec, stage1_records))
            
            # Select anchor photos (earliest, best quality, calibration)
            anchors = self._select_anchors(ordered, stage1_records)
            
            for idx, current in enumerate(ordered):
                # Compare with adjacent (window)
                window = max(1, int(self.config.get("comparison_window", 2)))
                for offset in range(1, min(window, idx) + 1):
                    a = ordered[idx - offset]
                    b = current
                    pair = self._compare_pair(a, b, reference, stage1_records, reconstructions)
                    evidence.append(pair)
                    pair_index[a.photo_id].append(pair.model_dump())
                    pair_index[b.photo_id].append(pair.model_dump())
                
                # Compare with anchors
                for anchor in anchors:
                    if anchor.photo_id == current.photo_id:
                        continue
                    pair = self._compare_pair(anchor, current, reference, stage1_records, reconstructions)
                    evidence.append(pair)
                    pair_index[anchor.photo_id].append(pair.model_dump())
                    pair_index[current.photo_id].append(pair.model_dump())

        save_json([p.model_dump() for p in evidence], root / "pairs.json")
        save_json(pair_index, root / "pair_index.json")
        return evidence

    def _load_all_reconstructions(self, root: Path) -> Dict[str, dict]:
        """Load all 3DDFA-V3 reconstructions for ICP."""
        reconstructions = {}
        for path in sorted(root.glob("*/reconstruction.pkl")):
            try:
                photo_id = path.parent.name
                recon = load_pickle(path)
                reconstructions[photo_id] = recon
            except Exception as e:
                logger.warning("Failed to load reconstruction for %s: %s", path, e)
        logger.info("Loaded %d reconstructions for ICP", len(reconstructions))
        return reconstructions

    def _select_anchors(self, ordered_records: list[Stage2Record], 
                        stage1_records: dict[str, Stage1Record]) -> list[Stage2Record]:
        """Select anchor photos for comparison."""
        if not ordered_records:
            return []
        
        anchors = []
        # Earliest photo
        anchors.append(ordered_records[0])
        # Best quality photo
        best_q = max(ordered_records, key=lambda r: r.quality.overall_quality)
        if best_q not in anchors:
            anchors.append(best_q)
        # Calibration-matched (if any)
        for rec in ordered_records:
            s1 = stage1_records.get(rec.photo_id)
            if s1 and s1.dataset.value == "calibration":
                if rec not in anchors:
                    anchors.append(rec)
        # Evenly spaced
        n = len(ordered_records)
        if n > 10:
            for i in [n // 4, n // 2, 3 * n // 4]:
                if ordered_records[i] not in anchors:
                    anchors.append(ordered_records[i])
        
        return anchors[:self.max_anchors]

    def _load_stage2_records(self, root: Path) -> list[Stage2Record]:
        records: list[Stage2Record] = []
        for photo_dir in sorted(root.iterdir()):
            if not photo_dir.is_dir():
                continue
            info = load_json(photo_dir / "info.json")
            geo = load_json(photo_dir / "geometry_metrics.json")
            tex = load_json(photo_dir / "texture_metrics.json")
            if info and isinstance(geo, dict):
                records.append(Stage2Record(
                    photo_id=info.get("photo_id", photo_dir.name),
                    dataset=info.get("dataset", "main"),
                    bucket=info.get("pose", {}).get("bucket", "unknown"),
                    quality=info.get("quality", {}),
                    geometry=geo,
                    texture=tex or {},
                ))
        return records

    def _load_stage1_records(self, root: Path) -> dict[str, Stage1Record]:
        records: dict[str, Stage1Record] = {}
        for path in sorted(root.glob("*/info.json")):
            payload = load_json(path)
            if not payload:
                continue
            record = Stage1Record.model_validate(payload)
            records[record.photo_id] = record
        return records

    def _sort_key(self, record: Stage2Record, stage1_records: dict[str, Stage1Record]) -> tuple[str, str]:
        stage1 = stage1_records.get(record.photo_id)
        date_value = stage1.date.isoformat() if stage1 and stage1.date else "9999-99-99"
        return (date_value, record.photo_id)

    def _compare_pair(
        self,
        a: Stage2Record,
        b: Stage2Record,
        reference: CalibrationReference | None,
        stage1_records: dict[str, Stage1Record],
        reconstructions: Dict[str, dict],
    ) -> PairEvidence:
        stage1_a = stage1_records.get(a.photo_id)
        stage1_b = stage1_records.get(b.photo_id)
        age_gap_years = abs(float(stage1_a.age_years) - float(stage1_b.age_years)) if stage1_a and stage1_b and stage1_a.age_years is not None and stage1_b.age_years is not None else 0.0
        
        # Scalar metrics distance
        geometry_distance, geometry_noise_discount, geometry_overlap = self._normalized_distance(
            a.geometry, b.geometry, reference, a.bucket.value, channel="geometry", age_gap_years=age_gap_years
        )
        texture_distance, texture_noise_discount, texture_overlap = self._normalized_distance(
            a.texture, b.texture, reference, a.bucket.value, channel="texture", age_gap_years=age_gap_years
        )
        
        # ICP alignment if both reconstructions available
        icp_distance = 0.0
        heatmap = {}
        recon_a = reconstructions.get(a.photo_id)
        recon_b = reconstructions.get(b.photo_id)
        
        if recon_a and recon_b:
            icp_result = self._icp_align_and_compare(recon_a, recon_b, reference, a.bucket.value)
            if icp_result:
                icp_distance = icp_result["distance"]
                heatmap = icp_result["heatmap"]
        
        qa = float(a.quality.overall_quality)
        qb = float(b.quality.overall_quality)
        quality_penalty = float(np.clip(1.15 - ((qa + qb) / 2.0), 0.55, 1.35))
        pose_gap_deg = self._pose_gap_deg(stage1_a, stage1_b)
        date_gap_days = abs(self._date_to_ord(stage1_a) - self._date_to_ord(stage1_b))
        chronology_penalty = float(np.clip(1.0 + min(date_gap_days / 180.0, 2.0) * 0.18 + min(pose_gap_deg / 45.0, 1.0) * 0.12, 1.0, 1.65))
        
        synthetic_suspicion = float(np.clip((texture_distance * 0.85 + texture_noise_discount * 0.2) / 3.0, 0.0, 1.0))
        different_suspicion = float(np.clip((geometry_distance * 0.9 + geometry_noise_discount * 0.15) / 3.0, 0.0, 1.0))
        same_raw = (geometry_distance * 0.62 + texture_distance * 0.28 + pose_gap_deg / 120.0 + quality_penalty * 0.15)
        same_suspicion = float(np.clip(1.0 - same_raw / 3.6, 0.0, 1.0))
        
        age_explained_distance = float(
            self._age_explained_distance(
                a.geometry, b.geometry, a.texture, b.texture, reference, a.bucket.value, age_gap_years
            )
        )
        if age_explained_distance > 0:
            geometry_distance = float(max(0.0, geometry_distance - min(geometry_distance * 0.55, age_explained_distance)))
            texture_distance = float(max(0.0, texture_distance - min(texture_distance * 0.45, age_explained_distance * 0.75)))
        
        anomaly_flags: list[str] = []
        if synthetic_suspicion > 0.7 and geometry_distance < 1.0:
            anomaly_flags.append("geometry_stable_texture_break")
        if date_gap_days < 90 and geometry_distance > 1.5 and pose_gap_deg < 22.5:
            anomaly_flags.append("short_gap_identity_shift")
        if chronology_penalty > 1.15 and same_suspicion < 0.4:
            anomaly_flags.append("chrono_pressure")
        if geometry_distance > 1.3 and texture_distance > 1.0 and abs(geometry_distance - texture_distance) > 0.8:
            anomaly_flags.append("cross_modal_disagreement")
        if pose_gap_deg > 35.0 and date_gap_days <= 90:
            anomaly_flags.append("pose_inconsistent_neighbor")
        if geometry_noise_discount > 0.2 or texture_noise_discount > 0.2:
            anomaly_flags.append("calibration_discounted")
        
        # Add ICP evidence if available
        if icp_distance > 0:
            anomaly_flags.append(f"icp_dist={icp_distance:.3f}")
        
        # Expression-aware zone weighting
        expr_flags_a = self._get_expression_flags(stage1_a)
        expr_flags_b = self._get_expression_flags(stage1_b)
        if expr_flags_a or expr_flags_b:
            # Merge exclusion zones from both photos
            excluded_zones = set()
            for flags in [expr_flags_a, expr_flags_b]:
                if flags:
                    excluded_zones.update(flags.get("excluded_zones", []))
            if excluded_zones:
                anomaly_flags.append(f"expr_excluded_zones={','.join(sorted(excluded_zones))}")
        
        pair_id = f"{a.photo_id}__{b.photo_id}"
        return PairEvidence(
            pair_id=pair_id,
            photo_a=a.photo_id,
            photo_b=b.photo_id,
            bucket=a.bucket.value,
            date_gap_days=int(date_gap_days),
            age_gap_years=float(age_gap_years),
            pose_gap_deg=float(pose_gap_deg),
            geometry_distance=float(geometry_distance),
            texture_distance=float(texture_distance),
            age_explained_distance=float(age_explained_distance),
            quality_penalty=quality_penalty,
            chronology_penalty=chronology_penalty,
            noise_discount=float(max(geometry_noise_discount, texture_noise_discount)),
            metric_overlap=int(max(geometry_overlap, texture_overlap)),
            synthetic_suspicion=synthetic_suspicion,
            different_suspicion=different_suspicion,
            same_suspicion=same_suspicion,
            anomaly_flags=anomaly_flags,
            notes=[
                f"bucket={a.bucket.value}",
                "pairwise evidence: adjacent + anchor comparisons with ICP alignment",
            ],
        )

    def _icp_align_and_compare(self, recon_a: dict, recon_b: dict, 
                               reference: CalibrationReference | None,
                               bucket: str) -> Optional[Dict]:
        """ICP alignment of two 3DDFA-V3 meshes."""
        try:
            # Get canonical vertices
            verts_a = np.asarray(recon_a.get("vertices_canonical", recon_a.get("vertices", [])), dtype=np.float32)
            verts_b = np.asarray(recon_b.get("vertices_canonical", recon_b.get("vertices", [])), dtype=np.float32)
            
            if verts_a.size == 0 or verts_b.size == 0:
                return None
            
            # Use bone anchor indices for alignment (stable zones)
            # For now use all visible vertices
            vis_a = np.asarray(recon_a.get("visible_idx_renderer", []), dtype=bool)
            vis_b = np.asarray(recon_b.get("visible_idx_renderer", []), dtype=bool)
            
            if vis_a.size == len(verts_a) and vis_b.size == len(verts_b):
                shared_vis = vis_a & vis_b
            else:
                shared_vis = np.ones(len(verts_a), dtype=bool)
            
            if shared_vis.sum() < 100:
                return None
            
            # Align using shared visible vertices
            src = verts_a[shared_vis]
            tgt = verts_b[shared_vis]
            
            # Procrustes alignment (no scale for forensic)
            aligned_src, R, t, scale = procrustes_align(src, tgt, allow_scale=False)
            
            # Apply to full mesh
            full_aligned = procrustes_align(verts_a, verts_b, allow_scale=False)[0]
            
            # Distance
            face_scale = max(verts_a[:, 0].max() - verts_a[:, 0].min(), 1.0)
            diff = np.linalg.norm(full_aligned - verts_b, axis=1)
            distance = float(diff.mean() / face_scale)
            
            # Zone heatmap using forensic zones
            annotation_groups = recon_a.get("annotation_groups", [])
            if annotation_groups:
                zone_indices = build_forensic_zone_indices(verts_b, annotation_groups)
            else:
                zone_indices = {}
            
            heatmap = compute_heatmap(full_aligned, verts_b, zone_indices, face_scale)
            
            return {"distance": distance, "heatmap": heatmap}
        except Exception as e:
            logger.debug("ICP failed for pair: %s", e)
            return None

    def _normalized_distance(
        self,
        a_metrics: dict[str, float],
        b_metrics: dict[str, float],
        reference: CalibrationReference | None,
        bucket: str,
        *,
        channel: str,
        age_gap_years: float = 0.0,
    ) -> tuple[float, float, int]:
        keys = sorted(set(a_metrics) & set(b_metrics))
        if not keys:
            return 0.0, 0.0, 0
        noise_bucket = reference.pairwise_noise.get(bucket, {}) if reference is not None else {}
        ref_stats = reference.global_stats if reference is not None else {}
        weighted = []
        discounts = []
        for key in keys:
            va = float(a_metrics[key])
            vb = float(b_metrics[key])
            ref = ref_stats.get(key, {})
            scale = max(ref.get("mad", 0.0) or ref.get("std", 0.0) or 1.0, 1e-6)
            base = abs(va - vb) / scale
            if key.startswith("texture_"):
                key_weight = 0.85
            elif key.startswith("mesh_") or key.endswith("_span") or key.startswith("face_") or key.startswith("zone_") or key.startswith("bone_"):
                key_weight = 1.15
            else:
                key_weight = 1.0
            weighted.append(base * key_weight)
            noise_entry = noise_bucket.get(key, {})
            noise_level = float(noise_entry.get("mad", 0.0) or noise_entry.get("std", 0.0) or 0.0)
            if noise_level > 0:
                discounts.append(min(noise_level / max(scale, 1e-6), 0.8))
        raw_distance = float(np.median(weighted) if weighted else 0.0)
        noise_discount = float(np.mean(discounts) if discounts else 0.0)
        age_shift = self._expected_age_shift(reference, bucket, channel, age_gap_years, keys)
        if age_shift > 0:
            raw_distance = max(0.0, raw_distance - min(raw_distance * 0.6, age_shift))
        if channel == "geometry":
            raw_distance *= 1.05
        else:
            raw_distance *= 0.95
        return float(max(0.0, raw_distance - min(raw_distance * 0.7, noise_discount))), noise_discount, len(keys)

    def _weighted_distance(self, a: dict[str, float], b: dict[str, float]) -> float:
        keys = sorted(set(a) & set(b))
        if not keys:
            return 0.0
        vals = []
        for key in keys:
            va = float(a[key])
            vb = float(b[key])
            scale = max(abs(va), abs(vb), 1.0)
            vals.append(abs(va - vb) / scale)
        return float(np.mean(vals) * 3.0)

    def _date_to_ord(self, record: Stage1Record | None) -> int:
        if record is None or record.date is None:
            return 0
        try:
            return record.date.toordinal()
        except Exception:
            return 0

    def _pose_gap_deg(self, a: Stage1Record | None, b: Stage1Record | None) -> float:
        if a is None or b is None:
            return 0.0
        pose_a = a.pose
        pose_b = b.pose
        dy = abs(float(pose_a.yaw) - float(pose_b.yaw))
        dp = abs(float(pose_a.pitch) - float(pose_b.pitch))
        dr = abs(float(pose_a.roll) - float(pose_b.roll))
        return float(np.sqrt((1.4 * dy) ** 2 + dp ** 2 + (0.6 * dr) ** 2))

    def _expected_age_shift(
        self,
        reference: CalibrationReference | None,
        bucket: str,
        channel: str,
        age_gap_years: float,
        keys: list[str],
    ) -> float:
        if reference is None or age_gap_years <= 0:
            return 0.0
        age_profiles = reference.age_profiles.get(bucket, {})
        shifts = []
        for key in keys:
            if channel == "texture" and not key.startswith("texture_"):
                continue
            profile = age_profiles.get(key)
            if not profile:
                continue
            slope = float(profile.get("slope", 0.0))
            corr = abs(float(profile.get("corr", 0.0)))
            weight = 1.0 + min(corr, 1.0)
            shifts.append(abs(slope) * age_gap_years * weight)
        if not shifts:
            return 0.0
        return float(np.median(shifts))

    def _age_explained_distance(
        self,
        a_geometry: dict[str, float],
        b_geometry: dict[str, float],
        a_texture: dict[str, float],
        b_texture: dict[str, float],
        reference: CalibrationReference | None,
        bucket: str,
        age_gap_years: float,
    ) -> float:
        if reference is None or age_gap_years <= 0:
            return 0.0
        geom_shift = self._expected_age_shift(reference, bucket, "geometry", age_gap_years, list(a_geometry.keys()))
        tex_shift = self._expected_age_shift(reference, bucket, "texture", age_gap_years, list(a_texture.keys()))
        return float(max(geom_shift, tex_shift))

    def _get_expression_flags(self, stage1: Stage1Record | None) -> dict | None:
        """Извлекает expression flags из Stage1Record через ExpressionAnalyzer3D."""
        if stage1 is None:
            return None
        # Проверяем, есть ли exp данные в реконструкции
        exp_data = getattr(stage1, 'expression_flags', None)
        if exp_data and isinstance(exp_data, dict) and "intensities" in exp_data:
            # Уже проанализировано
            return exp_data
        # Пытаемся извлечь из exp_vector если есть
        exp_vector = getattr(stage1, 'exp_vector', None)
        if exp_vector is not None:
            try:
                import numpy as np
                exp_arr = np.asarray(exp_vector, dtype=np.float64)
                face_scale = getattr(stage1, 'face_scale', 1.0) or 1.0
                analysis = self.expr_analyzer.analyze(exp_arr, face_scale)
                return {
                    "flags": analysis.flags,
                    "intensities": analysis.intensities,
                    "excluded_zones": analysis.excluded_zones,
                    "expression_label": analysis.expression_label,
                }
            except Exception:
                pass
        return None