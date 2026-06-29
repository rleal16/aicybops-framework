import os
import logging
from contextlib import contextmanager
from typing import Optional

import numpy as np
import torch
import mlflow
import mlflow.pytorch
from mlflow.models import infer_signature

logger = logging.getLogger(__name__)

class Logger:
    def __init__(self, tracking_uri=None):
        if tracking_uri is None:
            tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5001')
        mlflow.set_tracking_uri(tracking_uri)

    def has_active_run(self):
        return mlflow.active_run() is not None

    @contextmanager
    def run(self, experiment_name, run_name=None):
        mlflow.set_experiment(experiment_name)
        should_nest = self.has_active_run()
        with mlflow.start_run(run_name=run_name, nested=should_nest) as run:
            logger.info(f"{'Nested' if should_nest else 'Parent'} run: {run_name}")
            yield run

    def log_metrics(self, metrics, step: Optional[int] = None):
        """Log metrics to MLflow, filtering non-numeric values. Flattens nested dicts with dot notation."""
        def extract_numeric_metrics(obj, prefix=""):
            numeric_metrics = {}
            if isinstance(obj, dict):
                for key, value in obj.items():
                    full_key = f"{prefix}.{key}" if prefix else key
                    if isinstance(value, (int, float, np.integer, np.floating)):
                        v = float(value)
                        if not (v != v):  # skip NaN
                            numeric_metrics[full_key] = v
                    elif isinstance(value, dict):
                        numeric_metrics.update(extract_numeric_metrics(value, full_key))
            elif isinstance(obj, (int, float, np.integer, np.floating)):
                v = float(obj)
                if prefix and not (v != v):
                    numeric_metrics[prefix] = v
            return numeric_metrics

        numeric_metrics = extract_numeric_metrics(metrics)
        if numeric_metrics:
            logger.debug(f"Logging metrics: {numeric_metrics}")
            kwargs = {} if step is None else {"step": step}
            mlflow.log_metrics(numeric_metrics, **kwargs)
        else:
            logger.warning(f"No numeric metrics found in {metrics}")

    def log_params(self, params):
        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_artifacts(self, artifact_path):
        mlflow.log_artifacts(artifact_path)

    def get_model_signature(self, model, example_input):
        model.eval()
        is_multi_input = isinstance(example_input, (tuple, list))
        with torch.no_grad():
            if is_multi_input:
                example_output = model(*example_input)
                input_numpy = {
                    f"input_{i}": inp.numpy() if isinstance(inp, torch.Tensor) else inp
                    for i, inp in enumerate(example_input)
                }
                output_numpy = {
                    f"output_{i}": out.numpy() if isinstance(out, torch.Tensor) else out
                    for i, out in enumerate(example_output)
                }
            else:
                example_output = model(example_input)
                input_numpy = example_input.numpy() if isinstance(example_input, torch.Tensor) else example_input
                output_numpy = example_output.numpy() if isinstance(example_output, torch.Tensor) else example_output
            return infer_signature(input_numpy, output_numpy)

    def log_model(self, model, artifact_path, signature, input_example, registered_model_name: Optional[str] = None):
        name = registered_model_name or os.environ.get('MLFLOW_REGISTERED_MODEL_NAME', 'aicybops_model')
        if isinstance(input_example, (tuple, list)):
            input_example_numpy = None
        else:
            input_example_numpy = input_example.numpy() if isinstance(input_example, torch.Tensor) else input_example
        mlflow.pytorch.log_model(
            pytorch_model=model,
            name=artifact_path,
            signature=signature,
            input_example=input_example_numpy,
            pip_requirements=None,
            registered_model_name=name,
        )


    def load_best_model(self, experiment_name, run_name=None, nested_run_name=None, metric_name="accuracy", mode="max"):
        logger.info(f"Loading best model: experiment={experiment_name}, run={run_name}, metric={metric_name} ({mode})")
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.warning(f"Experiment '{experiment_name}' not found")
            return None, None

        filter_string = f"tags.mlflow.runName = '{run_name}'" if run_name else ""
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], filter_string=filter_string)
        if runs.empty:
            logger.warning(f"No runs found for experiment '{experiment_name}'")
            return None, None

        if nested_run_name:
            parent_run_id = runs.iloc[0].run_id
            nested_filter = f"tags.mlflow.parentRunId = '{parent_run_id}' and tags.mlflow.runName = '{nested_run_name}'"
            nested_runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], filter_string=nested_filter)
            if not nested_runs.empty:
                runs = nested_runs
            else:
                logger.warning(f"No nested run '{nested_run_name}' found for parent run '{run_name}'")
                return None, None

        runs = runs[runs[f"metrics.{metric_name}"].notna()]
        if runs.empty:
            logger.warning(f"No runs found with metric '{metric_name}'")
            return None, None

        ascending = mode == "min"
        best_run = runs.sort_values(f"metrics.{metric_name}", ascending=ascending).iloc[0]
        try:
            model = mlflow.pytorch.load_model(f"runs:/{best_run.run_id}/model")
            return model, best_run
        except Exception as e:
            logger.error(f"Error loading model from run {best_run.run_id}: {e}")
            return None, best_run

    def load_model_from_registry(self, registered_model_name: str, version: Optional[str] = None):
        """
        Load model from MLflow Model Registry.
        models:/{registered_model_name}/{version}; when version is None or 'latest', use latest.
        Returns (model, info_dict with version/run_id if available).
        """
        ver = version if version else "latest"
        model_uri = f"models:/{registered_model_name}/{ver}"
        logger.info(f"Loading model from registry: {model_uri}")
        try:
            model = mlflow.pytorch.load_model(model_uri)
            info = {"registered_model_name": registered_model_name, "version": ver}
            return model, info
        except Exception as e:
            logger.error(f"Error loading model from registry {model_uri}: {e}")
            return None, None
    
    def get_available_metrics(self, experiment_name, run_name=None):
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_name}' not found")

        filter_string = f"tags.mlflow.runName = '{run_name}'" if run_name else ""
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], filter_string=filter_string)

        if runs.empty:
            return []

        metrics_columns = [col for col in runs.columns if col.startswith('metrics.')]
        return [col.split('.', 1)[1] for col in metrics_columns]
    
    def get_run_data(self, experiment_name, run_name=None):
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_name}' not found")

        filter_string = f"tags.mlflow.runName = '{run_name}'" if run_name else ""
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], filter_string=filter_string)

        if runs.empty:
            return {}

        return runs.to_dict('records')[0]
