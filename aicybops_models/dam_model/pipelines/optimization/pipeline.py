"""Hyperparameter optimization pipeline."""

from typing import Dict, List, Tuple, Any, Optional
import traceback

from .param_sampling import get_param_combinations
from . import objectives as obj


class OptimizationPipeline:
    """
    Pipeline for hyperparameter optimization of DAM models.

    Handles trial execution and best model tracking; parameter generation and
    objective scoring are delegated to the optimization subpackage.
    """

    def __init__(
        self,
        dam,
        param_space: Dict[str, Any],
        max_evals: int,
        epochs: int,
        objective: str,
        method: str,
        **kwargs: Any,
    ):
        """
        Initialize the optimization pipeline.

        Args:
            dam: DAMAnomalyDetector instance
            param_space: Dictionary of hyperparameter ranges
            max_evals: Maximum number of evaluations
            epochs: Number of training epochs per evaluation
            objective: Metric to optimize ('val_loss', 'f1_score', 'test_anomaly_scores_mean')
            method: 'random' or 'grid' search
            **kwargs: Passed to train_test_validate (e.g., data, early_stopping)
        """
        self.dam = dam
        self.param_space = param_space
        self.max_evals = max_evals
        self.epochs = epochs
        self.objective = objective
        self.method = method
        self.kwargs = kwargs

        self.best_score: Optional[float] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_model_state: Optional[Dict] = None
        self.trial_results: List[Dict[str, Any]] = []

    def _resolve_data(self) -> Dict[str, Any]:
        """Get data from kwargs or dam._last_data."""
        data = self.kwargs.get("data")
        if data is None:
            if self.dam._last_data is None:
                raise ValueError(
                    "No data available for optimization. Provide data via kwargs or call train() first "
                    "(which sets _last_data)."
                )
            data = self.dam._last_data
        return data

    def _run_one_trial(
        self,
        trial_idx: int,
        total: int,
        params: Dict[str, Any],
        data: Dict[str, Any],
        original_lstm_hidden_dim: int,
        original_model_state: Optional[Dict],
    ) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
        """Run one trial and return (score, results) or (None, None) on failure."""
        print(f"\n[Trial {trial_idx}/{total}] Testing parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")

        if "lstm_hidden_dim" in params and params["lstm_hidden_dim"] != original_lstm_hidden_dim:
            self.dam.lstm_hidden_dim = params["lstm_hidden_dim"]
            if self.dam.dimensions is not None:
                self.dam.build_model()
        else:
            if self.dam.lstm_hidden_dim != original_lstm_hidden_dim:
                self.dam.lstm_hidden_dim = original_lstm_hidden_dim
                if self.dam.dimensions is not None:
                    self.dam.build_model()
            if self.dam.model is not None and original_model_state is not None:
                self.dam.model.load_state_dict(original_model_state)

        base = self.kwargs.get("base_params", {})
        merged_params = {**base, **params}

        trial_data = data.copy()
        if "window_size" in merged_params:
            trial_data["window_size"] = merged_params["window_size"]

        try:
            results = self.dam.train_test_validate(
                params=merged_params,
                epochs=self.epochs,
                data=trial_data,
                **{k: v for k, v in self.kwargs.items() if k not in ("data", "base_params")},
            )
            score = obj.get_score(self.dam, self.objective, results, trial_data, params=merged_params)
            return score, results
        except Exception as e:
            print(f"  X Trial failed: {e}")
            traceback.print_exc()
            return None, None

    def run(self) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
        """
        Run the optimization pipeline.

        Returns:
            Tuple of (best_params, best_score, trial_results)
        """
        data = self._resolve_data()

        param_combinations = get_param_combinations(
            self.param_space, self.method, self.max_evals
        )

        print(f"\n[Hyperparameter Optimization] Starting {self.method} search with {len(param_combinations)} trials")
        print(f"  Objective: {self.objective}")
        print(f"  Epochs per trial: {self.epochs}")

        self.best_score = obj.initial_best_score(self.objective)
        self.best_params = None
        self.best_model_state = None
        self.trial_results = []

        original_lstm_hidden_dim = self.dam.lstm_hidden_dim
        original_model_state = (
            self.dam.model.state_dict().copy() if self.dam.model is not None else None
        )

        for trial_idx, params in enumerate(param_combinations, 1):
            score, results = self._run_one_trial(
                trial_idx,
                len(param_combinations),
                params,
                data,
                original_lstm_hidden_dim,
                original_model_state,
            )
            if score is None or results is None:
                continue

            if obj.is_better(self.objective, score, self.best_score):
                self.best_score = score
                self.best_params = params.copy()
                if self.dam.model is not None:
                    self.best_model_state = self.dam.model.state_dict().copy()
                print(f"  New best score: {obj.display_score(self.objective, self.best_score):.4f}")
            else:
                print(f"  Score: {obj.display_score(self.objective, score):.4f} (best: {obj.display_score(self.objective, self.best_score):.4f})")

            self.trial_results.append({
                "trial": trial_idx,
                "params": params,
                "score": score,
                "results": results,
            })

        if self.best_model_state is not None and self.best_params is not None and self.dam.model is not None:
            if "lstm_hidden_dim" in self.best_params and self.dam.lstm_hidden_dim != self.best_params["lstm_hidden_dim"]:
                self.dam.lstm_hidden_dim = self.best_params["lstm_hidden_dim"]
                if self.dam.dimensions is not None:
                    self.dam.build_model()
            if "window_size" in self.best_params and self.dam.window_size != self.best_params["window_size"]:
                self.dam.window_size = self.best_params["window_size"]
            self.dam.model.load_state_dict(self.best_model_state)
            print(f"\n[Optimization Complete] Restored best model with score: {obj.display_score(self.objective, self.best_score):.4f}")

        print(f"\n[Optimization Summary]")
        print(f"  Best parameters: {self.best_params}")
        if self.best_score != float("inf") and self.best_score != float("-inf"):
            print(f"  Best {self.objective}: {obj.display_score(self.objective, self.best_score):.4f}")
        else:
            print(f"  Best {self.objective}: {self.best_score} (no successful trials)")
        print(f"  Total trials: {len(param_combinations)}")
        print(f"  Successful trials: {len(self.trial_results)}")

        if self.best_params is None:
            raise ValueError(
                "No successful trials completed. Check MLflow connection and data availability."
            )

        return self.best_params, self.best_score, self.trial_results
