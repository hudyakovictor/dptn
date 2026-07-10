from __future__ import annotations

import math
from typing import Any

from .prob_logic import ProbLogic
from .rules_engine import RulesEngine


class BayesianEngine:
    """Bayesian synthesis over pairwise + chronology evidence."""

    def __init__(self) -> None:
        self.prob = ProbLogic()
        self.rules = RulesEngine()

    def infer(self, evidence: dict[str, Any]) -> dict[str, Any]:
        rule_output = self.rules.apply(evidence)
        enriched = {**evidence, **rule_output}
        likelihoods = self.prob.evaluate(enriched)
        priors = self._priors(evidence)
        posterior = self._bayes(priors, likelihoods, rule_output)
        hypothesis = max(posterior, key=posterior.get)
        sorted_scores = sorted(posterior.values(), reverse=True)
        confidence = float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else float(sorted_scores[0])
        if rule_output.get("summary", {}).get("warning_level") == "high":
            confidence *= 0.9
        if confidence < 0.08 or sorted_scores[0] < 0.35:
            hypothesis = "H_UNCERTAIN"
        return {
            "hypothesis": hypothesis,
            "posterior": posterior,
            "confidence": float(confidence),
            "rules": rule_output,
            "likelihoods": likelihoods,
            "priors": priors,
        }

    def _priors(self, evidence: dict[str, Any]) -> dict[str, float]:
        age_gap = float(evidence.get("age_gap_years", 0.0) or 0.0)
        chronology = float(evidence.get("chronology_score", 0.0) or 0.0)
        syn = float(evidence.get("synthetic_suspicion", 0.0) or 0.0)
        diff = float(evidence.get("different_suspicion", 0.0) or 0.0)
        quality_penalty = float(evidence.get("quality_penalty", 0.0) or 0.0)
        same = max(0.16, 0.54 - diff * 0.12 - syn * 0.08 - quality_penalty * 0.08)
        synthetic = max(0.11, 0.18 + syn * 0.18 + quality_penalty * 0.04)
        different = max(0.13, 0.20 + diff * 0.16 + max(0.0, age_gap - 2.0) * 0.01)
        uncertainty = 0.10 + min(0.28, chronology * 0.08 + age_gap * 0.01 + quality_penalty * 0.06)
        total = same + synthetic + different + uncertainty
        return {
            "H0_SAME": same / total,
            "H1_SYNTHETIC": synthetic / total,
            "H2_DIFFERENT": different / total,
            "H_UNCERTAIN": uncertainty / total,
        }

    def _bayes(self, priors: dict[str, float], likelihoods: dict[str, float], rule_output: dict[str, Any]) -> dict[str, float]:
        rule_bias = self._rule_bias(rule_output)
        log_scores: dict[str, float] = {}
        for key in priors:
            prior = max(priors[key], 1e-9)
            like = max(likelihoods.get(key, 1e-9), 1e-9)
            bias = max(rule_bias.get(key, 1.0), 1e-6)
            log_scores[key] = math.log(prior) + math.log(like) + math.log(bias)
        max_log = max(log_scores.values())
        scores = {key: math.exp(value - max_log) for key, value in log_scores.items()}
        total = sum(scores.values()) or 1.0
        return {key: value / total for key, value in scores.items()}

    def _rule_bias(self, rule_output: dict[str, Any]) -> dict[str, float]:
        bias = {"H0_SAME": 1.0, "H1_SYNTHETIC": 1.0, "H2_DIFFERENT": 1.0, "H_UNCERTAIN": 1.0}
        for rule in rule_output.get("rules", []) or []:
            hypothesis = rule.get("hypothesis", "H_UNCERTAIN")
            weight = float(rule.get("weight", 0.0) or 0.0)
            if hypothesis in bias:
                bias[hypothesis] += weight * 0.75
        summary = rule_output.get("summary", {}) if isinstance(rule_output, dict) else {}
        warning = summary.get("warning_level")
        if warning == "high":
            bias["H_UNCERTAIN"] += 0.45
        elif warning == "medium":
            bias["H_UNCERTAIN"] += 0.2
        return bias
