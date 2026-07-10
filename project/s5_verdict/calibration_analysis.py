"""Calibration analysis for verdict posteriors.
Provides reliability diagram data, ECE/MCE computation, and fitted Platt calibrator.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CalibrationBin:
    """One bin in the reliability diagram."""
    bin_lower: float
    bin_upper: float
    mean_predicted: float
    mean_observed: float
    count: int


@dataclass
class CalibrationMetrics:
    """Aggregated calibration metrics."""
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    n_bins: int
    total_samples: int
    bins: List[CalibrationBin]


def calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute calibration curve (reliability diagram data).
    
    Args:
        y_true: binary ground truth labels (0 or 1)
        y_prob: predicted probabilities for the positive class
        n_bins: number of bins
    
    Returns:
        bin_true_probs: mean observed frequency per bin
        bin_pred_probs: mean predicted probability per bin
        bin_counts: number of samples per bin
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_true_probs = np.zeros(n_bins)
    bin_pred_probs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)
    
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:  # include right edge in last bin
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        
        bin_counts[i] = mask.sum()
        if bin_counts[i] > 0:
            bin_pred_probs[i] = y_prob[mask].mean()
            bin_true_probs[i] = y_true[mask].mean()
    
    return bin_true_probs, bin_pred_probs, bin_counts


def compute_ece_mce(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> CalibrationMetrics:
    """
    Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    
    ECE = sum(|observed_i - predicted_i| * count_i / total)
    MCE = max(|observed_i - predicted_i|)
    """
    bin_true, bin_pred, bin_counts = calibration_curve(y_true, y_prob, n_bins)
    
    total = bin_counts.sum()
    if total == 0:
        return CalibrationMetrics(ece=0.0, mce=0.0, n_bins=n_bins, total_samples=0, bins=[])
    
    abs_errors = np.abs(bin_true - bin_pred)
    ece = float(np.sum(abs_errors * bin_counts) / total)
    mce = float(np.max(abs_errors))
    
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for i in range(n_bins):
        if bin_counts[i] > 0:
            bins.append(CalibrationBin(
                bin_lower=float(bin_edges[i]),
                bin_upper=float(bin_edges[i + 1]),
                mean_predicted=float(bin_pred[i]),
                mean_observed=float(bin_true[i]),
                count=int(bin_counts[i]),
            ))
    
    return CalibrationMetrics(
        ece=ece,
        mce=mce,
        n_bins=n_bins,
        total_samples=int(total),
        bins=bins,
    )


class FittedPlattCalibrator:
    """
    Platt calibrator with learned parameters.
    Fits sigmoid: calibrated = 1 / (1 + exp(a * logit(raw) + b))
    on calibration data using scipy.optimize.minimize.
    """

    def __init__(self):
        self.a: float = -1.0
        self.b: float = 1.0
        self._fitted = False

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> None:
        """Fit Platt scaling parameters on calibration data."""
        from scipy.optimize import minimize

        y_true = np.asarray(y_true, dtype=float)
        y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
        logit = np.log(y_prob / (1 - y_prob))

        def objective(params):
            a, b = params
            calibrated = 1.0 / (1.0 + np.exp(-(a * logit + b)))
            # Negative log-likelihood
            eps = 1e-6
            nll = -np.mean(
                y_true * np.log(calibrated + eps) +
                (1 - y_true) * np.log(1 - calibrated + eps)
            )
            return nll

        result = minimize(objective, x0=[-1.0, 1.0], method="Nelder-Mead")
        self.a, self.b = result.x
        self._fitted = True

    def calibrate(self, posterior: Dict[str, float], quality: float = 0.5) -> Dict[str, float]:
        """Calibrate a single posterior dict using fitted parameters."""
        calibrated = {}
        for key, prob in posterior.items():
            prob_clipped = np.clip(prob, 1e-6, 1 - 1e-6)
            logit = np.log(prob_clipped / (1 - prob_clipped))
            
            if self._fitted:
                # Use fitted Platt parameters
                calibrated_logit = self.a * logit + self.b
            else:
                # Fallback: heuristic quality-dependent scaling
                scale = 0.5 + 0.5 * quality
                calibrated_logit = logit * scale
            
            calibrated_prob = 1.0 / (1.0 + np.exp(-calibrated_logit))
            calibrated[key] = float(calibrated_prob)
        
        # Normalize
        total = sum(calibrated.values()) or 1.0
        return {k: v / total for k, v in calibrated.items()}

    def is_fitted(self) -> bool:
        return self._fitted

    def get_params(self) -> Dict[str, float]:
        return {"a": self.a, "b": self.b, "fitted": self._fitted}

    def set_params(self, a: float, b: float) -> None:
        self.a = a
        self.b = b
        self._fitted = True


def generate_reliability_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    hypothesis_name: str = "",
) -> str:
    """Generate a text report of calibration quality."""
    metrics = compute_ece_mce(y_true, y_prob)
    
    lines = [
        f"Calibration Report: {hypothesis_name}" if hypothesis_name else "Calibration Report",
        f"  Samples: {metrics.total_samples}",
        f"  ECE: {metrics.ece:.4f}",
        f"  MCE: {metrics.mce:.4f}",
        "",
        "  Reliability Diagram:",
        "  Predicted | Observed | Count",
    ]
    
    for b in metrics.bins:
        bar_len = int(b.mean_observed * 20)
        bar = "#" * bar_len
        lines.append(
            f"  {b.bin_lower:.1f}-{b.bin_upper:.1f}   | "
            f"{b.mean_observed:.3f}    | {b.count:4d}  {bar}"
        )
    
    # Interpretation
    if metrics.ece < 0.05:
        lines.append("\n  Status: WELL CALIBRATED (ECE < 5%)")
    elif metrics.ece < 0.10:
        lines.append("\n  Status: MODERATELY CALIBRATED (ECE < 10%)")
    else:
        lines.append("\n  Status: POORLY CALIBRATED (ECE >= 10%)")
    
    return "\n".join(lines)
