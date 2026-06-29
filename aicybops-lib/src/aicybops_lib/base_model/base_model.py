from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import wraps
import os
import time
import logging
from ..tracking.mlflow_logging import Logger

from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)


def with_mlflow_logging(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        run_name = kwargs.pop('run_name', func.__name__)
        
        with self.run(run_name) as run:
            if func.__name__ == 'train':
                self._logger.log_params(kwargs)
            
            result = func(self, *args, **kwargs)

            if isinstance(result, dict):
                logger.info("Logging metrics to run %s: %s", run.info.run_id, result)
                step = int(time.time() * 1000) if func.__name__ == "evaluate" else None
                self._logger.log_metrics(result, step=step)

            if func.__name__ in ['validate', 'test', 'evaluate']:
                example_input = self.get_example_input()
                signature = self._logger.get_model_signature(self.model, example_input)
                reg_name = getattr(self, "registered_model_name", None)
                self._logger.log_model(self.model, "model", signature, example_input, registered_model_name=reg_name)
            
            return result
    return wrapper


class BaseModel(ABC):
    def __init__(self, experiment_name, tracking_uri=None, **kwargs):
        self.experiment_name = experiment_name
        self._logger = Logger(tracking_uri=tracking_uri)
        self.registered_model_name = kwargs.get("registered_model_name")
    
    @contextmanager
    def run(self, run_name=None):
        with self._logger.run(self.experiment_name, run_name) as run:
            yield run

    @abstractmethod
    def get_training_data(self, **kwargs) -> Any:
        pass

    @abstractmethod
    def get_test_data(self, **kwargs) -> Any:
        pass

    @abstractmethod
    def get_validation_data(self, **kwargs) -> Any:
        pass

    @abstractmethod
    def get_prediction_data(self, **kwargs) -> Any:
        pass

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    @with_mlflow_logging
    def train(self, **kwargs) -> dict:
        pass

    @abstractmethod
    @with_mlflow_logging
    def validate(self, **kwargs) -> dict:
        pass

    @abstractmethod
    @with_mlflow_logging
    def test(self, **kwargs) -> dict:
        pass
    
    @abstractmethod
    def predict(self, model, model_info = None, data = None):
        pass

    @abstractmethod
    def get_example_input(self):
        pass
    
    @abstractmethod
    def get_model_metrics(self) -> dict:
        """
        Returns a dictionary of metrics used by the model.
        Format: {
            'prediction': {'metric': 'metric_name', 'mode': 'max/min'},
            'training': ['metric1', 'metric2'],
            'evaluation': ['metric1', 'metric2']
        }
        """
        pass


    def train_test_validate(self, **kwargs):
        """
        Train, test, and validate the model in sequence.
        Returns dict including registered_model_name and model_version when model was logged to registry.
        """
        with self.run("train_test_validate"):
            train_result = self.train(**kwargs)
            test_result = self.test(**kwargs)
            val_result = self.validate(**kwargs)
            out = {**train_result, **test_result, **val_result}
        name = getattr(self, "registered_model_name", None) or os.environ.get("MLFLOW_REGISTERED_MODEL_NAME", "aicybops_model")
        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            versions = client.search_model_versions(
                filter_string=f"name='{name}'",
                order_by=["version_number DESC"],
                max_results=1,
            )
            if versions:
                out["registered_model_name"] = name
                out["model_version"] = versions[0].version
        except Exception as e:
            logger.debug("Could not get model version from registry: %s", e)
        return out


    def optimize(self, param_space, max_evals, epochs):
        best_params = param_space
        best_loss = float('inf')
        for params in param_space:
            result = self.train_test_validate(params=params, epochs=epochs)
            val_accuracy = result.get("validation_accuracy")
            if val_accuracy is not None and val_accuracy < best_loss:
                best_loss = val_accuracy
                best_params = params
        return best_params, best_loss
    
    def _get_best_model(
        self, 
        run_name: Optional[str] = None, 
        nested_run_name: Optional[str] = None, 
        metric_name: str = "accuracy", 
        mode: str = "max"
    ) -> Tuple[Optional[Any], Optional[Dict]]:
        """Get the best model based on the specified metric (runs/search). Deprecated in favor of _get_model (registry)."""
        logger.info("Getting best model for run: %s, nested run: %s", run_name, nested_run_name)
        try:
            best_model, run_info = self._logger.load_best_model(
                self.experiment_name,
                run_name,
                nested_run_name,
                metric_name,
                mode
            )
            return best_model, run_info
        except Exception as e:
            logger.error("Error: %s", e)
            return None, None

    def _get_model(
        self,
        registered_model_name: Optional[str] = None,
        model_version: Optional[str] = None
    ) -> Tuple[Optional[Any], Optional[Dict]]:
        """Load model from MLflow Model Registry. Uses default registry name when not provided."""
        name = registered_model_name or getattr(self, "registered_model_name", None) or os.environ.get("MLFLOW_REGISTERED_MODEL_NAME", "aicybops_model")
        ver = model_version or "latest"
        model, info = self._logger.load_model_from_registry(name, ver)
        return model, info

    def get_prediction(self, run_name=None, nested_run_name=None, metric_name=None, mode=None,
                      registered_model_name=None, model_version=None, data=None):
        """Get prediction using Model Registry by default; falls back to run-based load when registry params not used.
        data: optional dict passed to predict() (e.g. {'use_api': True} for live API data).
        """
        logger.info("Getting prediction for experiment: %s", self.experiment_name)
        model, run_info = self._get_model(registered_model_name, model_version or "latest")
        if model is None:
            pred_metrics = self.get_model_metrics()["prediction"]
            metric_name = metric_name or pred_metrics["metric"]
            mode = mode or pred_metrics["mode"]
            model, run_info = self._get_best_model(run_name, nested_run_name, metric_name, mode)
        if model is None:
            if not hasattr(self, "model") or self.model is None:
                raise ValueError("No model available for prediction")
            model = self.model
            run_info = None
        else:
            self.model = model
        return self.predict(model, run_info, data=data)

    def evaluate(self, data=None, config=None, output_dir=None, dataset_type="test", device="cpu", **kwargs):
        """Optional evaluation. Default: not supported. Override in models that support evaluation (e.g. DAM)."""
        raise NotImplementedError("This model does not support evaluation.")
