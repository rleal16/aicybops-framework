"""
Objective metrics for hyperparameter optimization (score extraction and min/max).
"""

import logging
from typing import Dict, Any, Optional

_logger = logging.getLogger(__name__)

# Objectives to minimize.
OBJECTIVE_MINIMIZE = frozenset({"val_loss", "test_anomaly_scores_mean"})

# Objectives to maximize.
OBJECTIVE_MAXIMIZE = frozenset({"f1_score"})


def get_score(
    dam,
    objective: str,
    results: Dict[str, Any],
    trial_data: Dict[str, Any],
    params: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Extract the scalar score for the given objective from training/eval results.

    For minimization objectives (val_loss, test_anomaly_scores_mean) we return the
    raw value (lower is better). For f1_score we return the positive F1 value so
    OBJECTIVE_MAXIMIZE / is_better correctly selects the highest F1.

    Args:
        dam: DAMAnomalyDetector instance (used for evaluate() when objective is f1_score).
        objective: One of 'val_loss', 'f1_score', 'test_anomaly_scores_mean'.
        results: Dict from train_test_validate (val_loss, test_anomaly_scores_mean, etc.).
        trial_data: Data dict for this trial (passed to evaluate for f1_score).
        params: Full merged trial params (used to build the EVT eval config for f1_score).
                If None, a default config with risk_level=1e-3 is used.

    Returns:
        Scalar score in its natural direction:
          - val_loss / test_anomaly_scores_mean: raw value (lower=better, MINIMIZE)
          - f1_score: positive F1 in [0, 1] (higher=better, MAXIMIZE).
            Returns 0.0 on failure so the trial is treated as worst-case.
    """
    if objective == "val_loss":
        return results.get("val_loss", float("inf"))
    if objective == "test_anomaly_scores_mean":
        return results.get("test_anomaly_scores_mean", float("inf"))
    if objective == "f1_score":
        try:
            p = params or {}
            # Build eval config from trial EVT parameters.
            eval_config = {
                "evt_parameters": {
                    "risk_level": float(p.get("risk_level", 1e-3)),
                    "spot_type": p.get("spot_type", "dSPOT"),
                    "depth": int(p.get("depth", 10)),
                    "init_quantile": float(p.get("init_quantile", 0.95)),
                },
                "max_memory_gb": 2.0,
            }
            eval_results = dam.evaluate(
                data=trial_data,
                config=eval_config,
                dataset_type="test",
                require_labels=True,
            )
            f1 = eval_results.get("metrics", {}).get("f1_score", 0.0)
            return f1  # Positive value; maximize chooses best F1.
        except Exception as e:
            _logger.warning(
                "f1_score eval failed: %s; returning 0.0 (worst F1) for this trial",
                e,
                exc_info=True,
            )
            return 0.0  # worst-case F1 so the trial is not selected as best
    raise ValueError(f"Unknown objective: {objective}")


def is_better(objective: str, score: float, best_score: float) -> bool:
    """Return True if score is better than best_score for the given objective."""
    if objective in OBJECTIVE_MINIMIZE:
        return score < best_score
    if objective in OBJECTIVE_MAXIMIZE:
        return score > best_score
    raise ValueError(f"Unknown objective: {objective}")


def initial_best_score(objective: str) -> float:
    """Return the initial 'worst' value so any first result is better."""
    if objective in OBJECTIVE_MINIMIZE:
        return float("inf")
    if objective in OBJECTIVE_MAXIMIZE:
        return float("-inf")
    raise ValueError(f"Unknown objective: {objective}")


def display_score(objective: str, score: float) -> float:
    """Return the score for display. All objectives are stored in their natural direction
    (f1_score is positive [0,1]; val_loss is raw MSE). No negation needed."""
    return score
