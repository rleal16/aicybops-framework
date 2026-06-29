"""BaseModel wrapper for modelos_uc XGBoost demo script."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import classification_report

from aicybops_lib.base_model import BaseModel, with_mlflow_logging
from aicybops_models.modelos_uc import supervised_xgboost

_UC_XGB_IN_MEMORY_CACHE: Dict[str, Dict[str, List[int]]] = {}
_UC_XGB_CACHE_LOCK = threading.Lock()


class ModelosUCXGBDetector(BaseModel):
    """Supervised XGBoost detector backed by run_pipeline()."""

    def __init__(self, experiment_name: str, tracking_uri: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(experiment_name, tracking_uri, **kwargs)
        self.model: Any = None
        self._pipeline_outputs: Optional[Dict[str, Any]] = None
        self._report: Optional[Dict[str, Any]] = None

    def _cache_key(self) -> str:
        return "modelos_uc_xgb"

    def _cache_file_path(self) -> Path:
        cache_dir = Path(os.getenv("MODELOS_UC_CACHE_DIR", "/app/data"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{self._cache_key()}.json"

    def _load_persistent_cache(self) -> Optional[Dict[str, List[int]]]:
        cache_file = self._cache_file_path()
        if not cache_file.exists():
            return None
        with cache_file.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        if isinstance(data, dict) and "y_test" in data and "y_pred" in data:
            return data
        return None

    def _save_persistent_cache(self, outputs: Dict[str, List[int]]) -> None:
        cache_file = self._cache_file_path()
        with cache_file.open("w", encoding="utf-8") as fp:
            json.dump(outputs, fp)

    def _to_cache_payload(self, outputs: Dict[str, Any]) -> Dict[str, List[int]]:
        return {
            "y_test": [int(x) for x in np.asarray(outputs["y_test"]).tolist()],
            "y_pred": [int(x) for x in np.asarray(outputs["y_pred"]).tolist()],
        }

    def _run_pipeline(self) -> Dict[str, Any]:
        if self._pipeline_outputs is not None:
            return self._pipeline_outputs

        key = self._cache_key()
        with _UC_XGB_CACHE_LOCK:
            cached = _UC_XGB_IN_MEMORY_CACHE.get(key)
            if cached is None:
                cached = self._load_persistent_cache()
            if cached is None:
                raw_outputs = supervised_xgboost.run_pipeline()
                if not isinstance(raw_outputs, dict):
                    raise TypeError("run_pipeline() must return a dictionary")
                cached = self._to_cache_payload(raw_outputs)
                _UC_XGB_IN_MEMORY_CACHE[key] = cached
                self._save_persistent_cache(cached)
            else:
                _UC_XGB_IN_MEMORY_CACHE[key] = cached

        self._pipeline_outputs = cached
        self.model = None
        self._report = classification_report(
            cached["y_test"],
            cached["y_pred"],
            target_names=["normal", "attack"],
            output_dict=True,
            zero_division=0,
        )
        return cached

    def _summary_metrics(self) -> Dict[str, float]:
        outputs = self._run_pipeline()
        report = self._report or {}
        weighted = report.get("weighted avg", {})
        return {
            "accuracy": float(report.get("accuracy", 0.0)),
            "f1_weighted": float(weighted.get("f1-score", 0.0)),
            "precision_weighted": float(weighted.get("precision", 0.0)),
            "recall_weighted": float(weighted.get("recall", 0.0)),
            "anomalies_detected": int(np.asarray(outputs["y_pred"]).sum()),
            "samples": int(len(outputs["y_pred"])),
        }

    def get_training_data(self, **kwargs: Any) -> None:
        return None

    def get_test_data(self, **kwargs: Any) -> None:
        return None

    def get_validation_data(self, **kwargs: Any) -> None:
        return None

    def get_prediction_data(self, **kwargs: Any) -> None:
        return None

    def build_model(self) -> None:
        return None

    def get_example_input(self) -> None:
        return None

    def get_model_metrics(self) -> Dict[str, Any]:
        return {
            "prediction": {"metric": "f1_weighted", "mode": "max"},
            "training": ["accuracy", "f1_weighted"],
            "evaluation": ["accuracy", "f1_weighted", "anomalies_detected"],
        }

    @with_mlflow_logging
    def train(self, **kwargs: Any) -> Dict[str, float]:
        self._run_pipeline()
        return self._summary_metrics()

    def validate(self, **kwargs: Any) -> Dict[str, float]:
        run_name = kwargs.pop("run_name", "validate")
        with self.run(run_name):
            metrics = self._summary_metrics()
            self._logger.log_metrics(metrics)
        return metrics

    def test(self, **kwargs: Any) -> Dict[str, float]:
        run_name = kwargs.pop("run_name", "test")
        with self.run(run_name):
            metrics = self._summary_metrics()
            self._logger.log_metrics(metrics)
        return metrics

    def predict(self, model: Any = None, model_info: Any = None, data: Any = None) -> List[int]:
        outputs = self._run_pipeline()
        return [int(x) for x in np.asarray(outputs["y_pred"]).tolist()]

    def get_prediction(
        self,
        run_name: Optional[str] = None,
        nested_run_name: Optional[str] = None,
        metric_name: Optional[str] = None,
        mode: Optional[str] = None,
        registered_model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        data: Any = None,
    ) -> List[int]:
        return self.predict(data=data)

    def evaluate(
        self,
        data: Any = None,
        config: Any = None,
        output_dir: Optional[str] = None,
        dataset_type: str = "test",
        device: str = "cpu",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        run_name = kwargs.pop("run_name", "evaluate")
        with self.run(run_name):
            metrics = self._summary_metrics()
            self._logger.log_metrics(metrics)
        return {
            **metrics,
            "classification_report": self._report,
            "dataset_type": dataset_type,
        }

    def optimize(
        self,
        param_space: Any = None,
        max_evals: Any = None,
        epochs: Any = None,
        objective: Any = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self.train_test_validate(**kwargs)
