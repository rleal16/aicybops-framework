"""
Parameter combination generation for hyperparameter search (random and grid).
"""

from typing import Dict, List, Any, Optional
import random
import itertools


def generate_random_combinations(param_space: Dict[str, Any], n_combinations: int, seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """Generate random parameter combinations from param_space."""
    if seed is not None:
        random.seed(seed)
    combinations = []
    for _ in range(n_combinations):
        params = {}
        for key, values in param_space.items():
            if isinstance(values, list):
                params[key] = random.choice(values)
            else:
                params[key] = values
        combinations.append(params)
    return combinations


def generate_grid_combinations(param_space: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate all parameter combinations from param_space (grid search)."""
    keys = list(param_space.keys())
    values = [
        param_space[key] if isinstance(param_space[key], list) else [param_space[key]]
        for key in keys
    ]

    combinations = []
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        # Handle conditional parameters (e.g., depth only if spot_type='dSPOT')
        if "spot_type" in params and "depth" in params:
            if params["spot_type"] == "SPOT":
                params.pop("depth", None)
        combinations.append(params)

    return combinations


def get_param_combinations(
    param_space: Dict[str, Any],
    method: str,
    max_evals: int,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Return list of parameter combinations for the given method.

    Args:
        param_space: Dict of hyperparameter names to lists of values (or single value).
        method: 'random' or 'grid'.
        max_evals: Maximum number of evaluations. For grid, combinations are capped/sampled to this.
        seed: Optional random seed for reproducibility.

    Returns:
        List of parameter dicts to evaluate.
    """
    if method == "grid":
        if seed is not None:
            random.seed(seed)
        combinations = generate_grid_combinations(param_space)
        if len(combinations) > max_evals:
            combinations = random.sample(combinations, max_evals)
        return combinations
    if method == "random":
        return generate_random_combinations(param_space, max_evals, seed=seed)
    raise ValueError(f"Unknown method: {method}. Use 'random' or 'grid'.")
