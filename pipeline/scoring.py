"""Composite recipe score: weighted combination of CV mean, CV std penalty, robustness."""

from __future__ import annotations

from typing import Any, Dict, List


def compute_composite_score(
    cv_scores: List[float],
    robustness_score: float,
    cfg: Dict[str, Any],
) -> float:
    """
    score = cv_mean_weight * cv_mean
          - cv_std_weight  * cv_std
          + robustness_weight * robustness_score

    All weights are read from cfg["scoring"].
    """
    if not cv_scores:
        return 0.0

    import statistics

    scoring = cfg["scoring"]
    w_mean = float(scoring.get("cv_mean_weight", 0.5))
    w_std = float(scoring.get("cv_std_weight", 0.2))
    w_rob = float(scoring.get("robustness_weight", 0.2))

    cv_mean = statistics.mean(cv_scores)
    cv_std = statistics.stdev(cv_scores) if len(cv_scores) > 1 else 0.0

    return w_mean * cv_mean - w_std * cv_std + w_rob * robustness_score
