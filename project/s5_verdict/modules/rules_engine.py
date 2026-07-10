from __future__ import annotations

from typing import Any


class RulesEngine:
    """Human-readable forensic rules."""

    def apply(self, evidence: dict[str, Any]) -> dict[str, Any]:
        rules: list[dict[str, Any]] = []
        geometry = float(evidence.get("geometry_distance", 0.0) or 0.0)
        texture = float(evidence.get("texture_distance", 0.0) or 0.0)
        age_gap = float(evidence.get("age_gap_years", 0.0) or 0.0)
        age_explained = float(evidence.get("age_explained_distance", 0.0) or 0.0)
        chronology = float(evidence.get("chronology_score", 0.0) or 0.0)
        pose_gap = float(evidence.get("pose_gap_deg", 0.0) or 0.0)
        synthetic = float(evidence.get("synthetic_suspicion", 0.0) or 0.0)
        same = float(evidence.get("same_suspicion", 0.0) or 0.0)
        diff = float(evidence.get("different_suspicion", 0.0) or 0.0)
        quality_penalty = float(evidence.get("quality_penalty", 0.0) or 0.0)
        alignment_residual = float(evidence.get("alignment_residual", 0.0) or 0.0)
        flags = list(evidence.get("anomaly_flags", []) or [])
        if not flags:
            flags = list(evidence.get("flags", []) or [])

        if synthetic > 0.72 and geometry < 1.0:
            rules.append(self._rule("R_SYNTHETIC_GEOMETRY_STABLE", 0.9, "H1_SYNTHETIC", evidence="texture-heavy synthetic suspicion"))
        if texture > 1.1 and synthetic > 0.45:
            rules.append(self._rule("R_SYNTHETIC_TEXTURE_BREAK", 0.8, "H1_SYNTHETIC", evidence="texture breaks without matching geometry shift"))
        if geometry > 1.35 and age_gap > age_explained + 1.0:
            rules.append(self._rule("R_GEOMETRY_AGE_MISMATCH", 0.85, "H2_DIFFERENT", evidence="geometry drift exceeds age-explained movement"))
        if chronology > 0.85 or "strong_temporal_break" in flags:
            rules.append(self._rule("R_TEMPORAL_BREAK", 0.75, "H_UNCERTAIN", evidence="temporal ordering is unstable"))
        if pose_gap > 35.0 and geometry > 1.0:
            rules.append(self._rule("R_POSE_INCONSISTENT", 0.7, "H2_DIFFERENT", evidence="pose gap is too large for a stable pair"))
        if same > 0.72 and diff < 0.35 and synthetic < 0.35 and chronology < 0.4:
            rules.append(self._rule("R_SAME_STABLE", 0.8, "H0_SAME", evidence="pair is geometrically and temporally stable"))
        if quality_penalty > 0.45 and alignment_residual > 1.0:
            rules.append(self._rule("R_LOW_QUALITY_AMBIGUOUS", 0.55, "H_UNCERTAIN", evidence="quality and alignment are both weak"))
        if texture > geometry + 0.6 and synthetic > 0.35:
            rules.append(self._rule("R_TEXTURE_DOMINANT", 0.6, "H1_SYNTHETIC", evidence="texture dominates over geometry"))
        if diff > 0.7 and geometry > 0.9 and chronology < 0.55:
            rules.append(self._rule("R_DIFFERENCE_CONSISTENT", 0.65, "H2_DIFFERENT", evidence="difference is stable across evidence channels"))

        summary = self._summarize(rules)
        return {
            "rules": rules,
            "summary": summary,
            "dominant_rule": summary["dominant_rule"],
        }

    def _rule(self, rule_id: str, weight: float, hypothesis: str, *, evidence: str = "") -> dict[str, Any]:
        return {
            "rule_id": rule_id,
            "weight": float(weight),
            "hypothesis": hypothesis,
            "evidence": evidence,
        }

    def _summarize(self, rules: list[dict[str, Any]]) -> dict[str, Any]:
        counts = {"H0_SAME": 0, "H1_SYNTHETIC": 0, "H2_DIFFERENT": 0, "H_UNCERTAIN": 0}
        weights = {"H0_SAME": 0.0, "H1_SYNTHETIC": 0.0, "H2_DIFFERENT": 0.0, "H_UNCERTAIN": 0.0}
        for rule in rules:
            hyp = rule.get("hypothesis", "H_UNCERTAIN")
            counts[hyp] = counts.get(hyp, 0) + 1
            weights[hyp] = weights.get(hyp, 0.0) + float(rule.get("weight", 0.0) or 0.0)
        dominant = max(counts, key=lambda key: (counts.get(key, 0), weights.get(key, 0.0))) if rules else "H_UNCERTAIN"
        dominant_rule_id = max(rules, key=lambda rule: float(rule.get("weight", 0.0) or 0.0)).get("rule_id") if rules else None
        warning_level = "low"
        total_weight = sum(weights.values())
        if total_weight > 2.2 or counts.get("H_UNCERTAIN", 0) >= 2:
            warning_level = "high"
        elif total_weight > 1.2 or len(rules) >= 2:
            warning_level = "medium"
        return {
            "counts": counts,
            "weights": weights,
            "dominant_rule": dominant,
            "dominant_rule_id": dominant_rule_id,
            "warning_level": warning_level,
            "rule_count": len(rules),
            "active_rule_ids": [rule.get("rule_id") for rule in rules],
        }
