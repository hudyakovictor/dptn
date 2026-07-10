from __future__ import annotations

from typing import Any

import numpy as np

from .icp_aligner import ICPAligner
from .mesh_evidence import MeshEvidence


class PairComparator:
    """Pairwise compare orchestration: align + evidence + flags."""

    def __init__(self) -> None:
        self.aligner = ICPAligner()
        self.evidence = MeshEvidence()

    def compare(self, a: dict[str, Any] | Any, b: dict[str, Any] | Any, reference: dict | None = None) -> dict[str, Any]:
        aligned = self.aligner.align(self._mesh(a), self._mesh(b))
        evidence = self.evidence.compare(a, b, reference=reference)
        geometry_distance = float(evidence.get("geometry_distance", 0.0))
        texture_distance = float(evidence.get("texture_distance", 0.0))
        same = float(evidence.get("same_suspicion", 0.0))
        diff = float(evidence.get("different_suspicion", 0.0))
        syn = float(evidence.get("synthetic_suspicion", 0.0))
        quality = float(np.clip(((self._quality(a) + self._quality(b)) / 2.0), 0.0, 1.0))
        alignment_residual = float(aligned.get("residual", 0.0) or 0.0)
        inlier_ratio = float(aligned.get("inliers", 0) / max(len(aligned.get("aligned_vertices", [])), 1))
        penalty = float(np.clip(1.0 - quality + min(alignment_residual / 3.0, 0.35), 0.0, 1.0))
        score = {
            "H0_SAME": float(np.clip(same * (0.84 + 0.16 * quality) * (1.0 - min(alignment_residual / 4.0, 0.35)), 0.0, 1.0)),
            "H1_SYNTHETIC": float(np.clip(syn * (0.78 + 0.22 * penalty), 0.0, 1.0)),
            "H2_DIFFERENT": float(np.clip(diff * (0.86 + 0.14 * penalty) * (1.0 + min(alignment_residual / 5.0, 0.2)), 0.0, 1.0)),
            "H_UNCERTAIN": float(np.clip(max(0.0, 1.0 - max(same, diff, syn)) * (0.85 + 0.15 * penalty), 0.0, 1.0)),
        }
        best = max(score, key=score.get)
        flags = list(evidence.get("notes", []))
        if aligned.get("status") != "ok":
            flags.append(f"alignment_{aligned.get('status')}")
        if evidence.get("pose_gap_deg", 0.0) > 35.0:
            flags.append("pose_gap_high")
        if evidence.get("age_explained_distance", 0.0) > 0:
            flags.append("age_explained")
        if alignment_residual > 1.2:
            flags.append("alignment_residual_high")
        if inlier_ratio < 0.6:
            flags.append("low_inlier_ratio")
        return {
            "pair_id": self._pair_id(a, b),
            "alignment": aligned,
            "evidence": evidence,
            "posterior": score,
            "decision": best,
            "confidence": float(score[best]),
            "flags": flags,
            "quality": quality,
            "geometry_distance": geometry_distance,
            "texture_distance": texture_distance,
            "alignment_residual": alignment_residual,
            "alignment_inlier_ratio": inlier_ratio,
        }

    def _quality(self, record: dict[str, Any] | Any) -> float:
        if hasattr(record, "quality"):
            quality = getattr(record, "quality")
            if quality is not None and hasattr(quality, "overall_quality"):
                return float(getattr(quality, "overall_quality", 0.0) or 0.0)
        if isinstance(record, dict):
            quality = record.get("quality")
        else:
            quality = None
        if isinstance(quality, dict):
            value = quality.get("overall_quality", 0.0)
            if isinstance(value, (int, float)):
                return float(value)
        return 0.0

    def _mesh(self, record: dict[str, Any] | Any) -> Any:
        if hasattr(record, "mesh"):
            mesh = getattr(record, "mesh")
            if mesh is not None:
                return mesh
        if isinstance(record, dict):
            return record.get("mesh")
        return None

    def _pair_id(self, a: dict[str, Any] | Any, b: dict[str, Any] | Any) -> str:
        def _photo_id(item: Any, fallback: str) -> str:
            if hasattr(item, "photo_id"):
                value = getattr(item, "photo_id")
                if value:
                    return str(value)
            if isinstance(item, dict):
                value = item.get("photo_id")
                if value:
                    return str(value)
            return fallback

        return f"{_photo_id(a, 'a')}__{_photo_id(b, 'b')}"
