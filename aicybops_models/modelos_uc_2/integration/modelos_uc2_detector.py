"""BaseModel wrapper for modelos_uc_2 run_pipeline()."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from aicybops_lib.base_model import BaseModel, with_mlflow_logging
from aicybops_models.modelos_uc_2 import models as uc2_models

_UC2_IN_MEMORY_CACHE: Dict[str, Dict[str, Dict[str, Any]]] = {}
_UC2_CACHE_LOCK = threading.Lock()


class ModelosUC2Detector(BaseModel):
    """Single wrapper that delegates execution to modelos_uc_2.run_pipeline()."""

    def __init__(self, experiment_name: str, tracking_uri: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(experiment_name, tracking_uri, **kwargs)
        self.model: Any = None
        self._pipeline_outputs: Optional[Dict[str, Dict[str, Any]]] = None
        self._model_names: Optional[List[str]] = kwargs.get("model_names")

    def _cache_key(self) -> str:
        names = self._model_names or list(uc2_models.MODELS.keys())
        canonical = ",".join(sorted(names))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _cache_file_path(self) -> Path:
        cache_dir = Path(os.getenv("MODELOS_UC2_CACHE_DIR", "/app/data"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"modelos_uc2_cache_{self._cache_key()}.json"

    def _load_persistent_cache(self) -> Optional[Dict[str, Dict[str, Any]]]:
        cache_file = self._cache_file_path()
        if not cache_file.exists():
            return None
        with cache_file.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        return data if isinstance(data, dict) else None

    def _save_persistent_cache(self, outputs: Dict[str, Dict[str, Any]]) -> None:
        cache_file = self._cache_file_path()
        with cache_file.open("w", encoding="utf-8") as fp:
            json.dump(outputs, fp)

    def _run_pipeline(self) -> Dict[str, Dict[str, Any]]:
        if self._pipeline_outputs is None:
            key = self._cache_key()
            with _UC2_CACHE_LOCK:
                cached = _UC2_IN_MEMORY_CACHE.get(key)
                if cached is not None:
                    self._pipeline_outputs = cached
                else:
                    persistent = self._load_persistent_cache()
                    if persistent is not None:
                        _UC2_IN_MEMORY_CACHE[key] = persistent
                        self._pipeline_outputs = persistent
                    else:
                        outputs = uc2_models.run_pipeline(
                            model_names=self._model_names,
                            output_format="normalized",
                        )
                        _UC2_IN_MEMORY_CACHE[key] = outputs
                        self._pipeline_outputs = outputs
                        self._save_persistent_cache(outputs)
        return self._pipeline_outputs

    def _summary_metrics(self) -> Dict[str, float]:
        outputs = self._run_pipeline()
        accuracies = [
            float(metrics.get("accuracy"))
            for metrics in outputs.values()
            if isinstance(metrics, dict) and metrics.get("accuracy") is not None
        ]
        return {
            "models_run": float(len(outputs)),
            "models_with_accuracy": float(len(accuracies)),
            "avg_accuracy": float(sum(accuracies) / len(accuracies)) if accuracies else 0.0,
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
            "prediction": {"metric": "avg_accuracy", "mode": "max"},
            "training": ["avg_accuracy"],
            "evaluation": ["avg_accuracy"],
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
        # modelos_uc_2 is an offline evaluation workflow; no live prediction vector.
        self._run_pipeline()
        return []

    def get_prediction(
        self,
        run_name: Optional[str] = None,
        nested_run_name: Optional[str] = None,
        metric_name: Optional[str] = None,
        mode: Optional[str] = None,
        registered_model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        data: Any = None,
    ) -> Dict[str, Any]:
        outputs = self._run_pipeline()
        return {
            "predictions": [],
            "prediction_diagnostics": outputs,
        }

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
        outputs = self._run_pipeline()
        with self.run(run_name):
            summary = self._summary_metrics()
            self._logger.log_metrics(summary)
        return {
            "summary": summary,
            "models": outputs,
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

