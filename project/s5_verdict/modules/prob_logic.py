from __future__ import annotations

from typing import Any


class ProbLogic:
    """Evidence-to-likelihood transform."""

    def evaluate(self, evidence: dict[str, Any]) -> dict[str, float]:
        geometry = float(evidence.get("geometry_distance", 0.0) or 0.0)
        texture = float(evidence.get("texture_distance", 0.0) or 0.0)
        pose_gap = float(evidence.get("pose_gap_deg", 0.0) or 0.0)
        chronology = float(evidence.get("chronology_score", 0.0) or 0.0)
        age_gap = float(evidence.get("age_gap_years", 0.0) or 0.0)
        age_explained = float(evidence.get("age_explained_distance", 0.0) or 0.0)
        synthetic = float(evidence.get("synthetic_suspicion", 0.0) or 0.0)
        different = float(evidence.get("different_suspicion", 0.0) or 0.0)
        same = float(evidence.get("same_suspicion", 0.0) or 0.0)
        quality_penalty = float(evidence.get("quality_penalty", 0.0) or 0.0)
        alignment_residual = float(evidence.get("alignment_residual", 0.0) or 0.0)
        noise_discount = float(evidence.get("noise_discount", 0.0) or 0.0)
        inlier_value = evidence.get("alignment_inlier_ratio", None)
        inlier_ratio = float(1.0 if inlier_value is None else inlier_value)

        same_like = max(
            1e-6,
            (
                same + 0.18
                + max(0.0, 0.3 - geometry) * 0.08
                + max(0.0, 0.3 - texture) * 0.05
                + max(0.0, 0.25 - pose_gap / 120.0) * 0.08
            )
            * (1.0 - min(0.45, chronology * 0.18))
            * (1.0 + min(0.18, age_explained * 0.04))
            * (1.0 + min(0.12, noise_discount * 0.12)),
        )
        syn_like = max(
            1e-6,
            (
                synthetic
                + texture * 0.18
                + max(0.0, quality_penalty - 0.25) * 0.12
                + max(0.0, alignment_residual - 0.8) * 0.10
            )
            * (1.0 + min(0.25, max(0.0, 0.7 - inlier_ratio) * 0.4))
            * (1.0 + min(0.15, noise_discount * 0.2)),
        )
        diff_like = max(
            1e-6,
            (
                different
                + geometry * 0.14
                + max(0.0, age_gap - age_explained) * 0.035
                + max(0.0, pose_gap / 90.0 - 0.2) * 0.08
            )
            * (1.0 + min(0.25, chronology * 0.12))
            * (1.0 + min(0.12, alignment_residual * 0.08)),
        )
        uncertain_like = max(
            1e-6,
            1.0
            + chronology * 0.22
            + max(0.0, age_gap - age_explained) * 0.02
            + max(0.0, quality_penalty - 0.4) * 0.25
            + max(0.0, alignment_residual - 1.1) * 0.15
            - same * 0.08,
        )

        return {
            "H0_SAME": float(same_like),
            "H1_SYNTHETIC": float(syn_like),
            "H2_DIFFERENT": float(diff_like),
            "H_UNCERTAIN": float(uncertain_like),
        }
