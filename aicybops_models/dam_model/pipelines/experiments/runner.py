"""
Experiment runner: sweep over config paths and seeds, one MLflow run per (config, seed).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SweepSpec:
    """Specification for an experiment sweep: configs x seeds."""

    config_paths: List[str]
    seeds: List[int]
    experiment_name: str = "DAM_experiments"
    tracking_uri: Optional[str] = None
    overrides: Optional[Dict[str, Any]] = None
    run_evaluation: bool = False
    eval_config: Optional[Dict[str, Any]] = None
    output_dir: Optional[str] = None
    fail_fast: bool = False
    quick_test: bool = False

    def __post_init__(self) -> None:
        if self.overrides is None:
            self.overrides = {}
        if self.run_evaluation and self.eval_config is None:
            raise ValueError("eval_config is required when run_evaluation is True")


class ExperimentRunner:
    """
    Runs a sweep of (config_path, seed) combinations.
    One MLflow run per combination; optional evaluation and summary CSV.
    """

    def __init__(self, spec: SweepSpec) -> None:
        self.spec = spec
        self.failed_runs: List[Dict[str, Any]] = []

    def _run_name(self, config_index: int, seed: int) -> str:
        return f"config_{config_index:03d}_seed_{seed}"

    def _build_data_dict(self, config_path: str, seed: int) -> Dict[str, Any]:
        from utils.config import build_data_config

        data_dict = build_data_config(
            config_path,
            overrides=self.spec.overrides,
            quick_test=self.spec.quick_test,
        )
        data_dict["random_state"] = seed
        return data_dict

    def _get_epochs(self) -> int:
        if self.spec.overrides and "epochs" in self.spec.overrides:
            return int(self.spec.overrides["epochs"])
        return 10

    def run(self) -> Dict[str, Any]:
        """
        Execute the sweep. Returns a summary dict with 'failed_runs' and optionally
        'summary_csv_path' if output_dir was set.
        """
        from core.dam_anomaly_detector import DAMAnomalyDetector

        self.failed_runs = []
        epochs = self._get_epochs()
        experiment_name = self.spec.experiment_name
        tracking_uri = self.spec.tracking_uri
        experiment_id = None  # Captured from first successful run.

        for config_index, config_path in enumerate(self.spec.config_paths):
            dam = DAMAnomalyDetector(
                experiment_name=experiment_name,
                config_path=config_path,
                tracking_uri=tracking_uri,
            )

            for seed in self.spec.seeds:
                run_name = self._run_name(config_index, seed)
                data_dict = self._build_data_dict(config_path, seed)

                try:
                    with dam.run(run_name) as run:
                        if experiment_id is None and run is not None and getattr(run, "info", None) is not None:
                            experiment_id = run.info.experiment_id
                        if self.spec.run_evaluation and self.spec.eval_config:
                            # Train and evaluate.
                            dam.train(
                                epochs=epochs,
                                data=data_dict,
                                params=self.spec.overrides or {},
                                run_name=run_name,
                                config_path=config_path,
                                seed=seed,
                            )
                            dam.evaluate(
                                data=data_dict,
                                config=self.spec.eval_config,
                                output_dir=None,
                                dataset_type="test",
                                device="cpu",
                            )
                        else:
                            # Run train/test/validate.
                            dam.train_test_validate(
                                epochs=epochs,
                                data=data_dict,
                                params=self.spec.overrides or {},
                                config_path=config_path,
                                seed=seed,
                            )
                except Exception as e:
                    logger.exception("Run %s failed: %s", run_name, e)
                    self.failed_runs.append(
                        {"run_name": run_name, "config_path": config_path, "seed": seed, "error": str(e)}
                    )
                    if self.spec.fail_fast:
                        raise

                # Clear cached data between seeds.
                if hasattr(dam, "clear_data_cache"):
                    dam.clear_data_cache()

        result: Dict[str, Any] = {"failed_runs": self.failed_runs}

        if self.spec.output_dir and experiment_id is not None:
            summary_path = self._write_summary_csv(experiment_id)
            if summary_path:
                result["summary_csv_path"] = str(summary_path)

        return result

    def _write_summary_csv(self, experiment_id: str) -> Optional[Path]:
        import mlflow

        try:
            runs_df = mlflow.search_runs(experiment_ids=[experiment_id])
            if runs_df.empty:
                return None
            out_path = Path(self.spec.output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            csv_path = out_path / "experiment_summary.csv"
            runs_df.to_csv(csv_path, index=False)
            logger.info("Summary written to %s", csv_path)
            return csv_path
        except Exception as e:
            logger.warning("Could not write summary CSV: %s", e)
            return None
