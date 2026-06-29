import requests
from typing import Dict, Any, Optional, Union, List, Tuple


def get_data_collection_counts(response: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Return (metrics_count, logs_count) from a train or job-status response, or (None, None)."""
    return (
        response.get("data_collection_metrics_count"),
        response.get("data_collection_logs_count"),
    )


class AICybOpsClient:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def train_model(
        self,
        params: Dict[str, Any],
        epochs: int,
        experiment_name: str,
        model_type: str = "pytorch",
        model_params: Optional[Dict[str, Any]] = None,
        run_optimization: bool = True,
        param_space: Optional[List[Dict[str, Any]]] = None,
        max_evals: Optional[int] = None,
        objective: str = "val_loss",
        wait: bool = False,
        file_name: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Train a model. When wait=True, returns dict with 'result', 'model_reference',
        and when data was collected from API: 'data_collection_metrics_count', 'data_collection_logs_count'.
        Use get_data_collection_counts(response) to extract the counts.
        """
        if model_params is None:
            model_params = {}
        if file_name:
            model_params["file_name"] = file_name
        payload = {
            "params": params,
            "epochs": epochs,
            "experiment_name": experiment_name,
            "model_type": model_type,
            "model_params": model_params,
            "run_optimization": run_optimization,
            "objective": objective,
        }
        if param_space is not None:
            payload["param_space"] = param_space
        if max_evals is not None:
            payload["max_evals"] = max_evals
        payload.update(kwargs)
        url = f"{self.base_url}/train/"
        if wait:
            url += "?wait=true"
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Poll job status. When status is 'completed', response may include
        data_collection_metrics_count and data_collection_logs_count. Use get_data_collection_counts(response)."""
        response = requests.get(f"{self.base_url}/jobs/{job_id}")
        response.raise_for_status()
        return response.json()

    def predict(
        self,
        experiment_name: str,
        model_type: str = "pytorch",
        registered_model_name: Optional[str] = None,
        model_version: Optional[Union[str, int]] = None,
        run_name: Optional[str] = None,
        nested_run_name: Optional[str] = None,
        metric_name: Optional[str] = None,
        mode: Optional[str] = None,
        model_params: Optional[Dict[str, Any]] = None,
        file_name: Optional[str] = None,
    ) -> Dict[str, Union[bool, List[int]]]:
        if model_params is None:
            model_params = {}
        if file_name:
            model_params["file_name"] = file_name
        payload = {
            "experiment_name": experiment_name,
            "model_type": model_type,
            "model_params": model_params,
        }
        if registered_model_name is not None:
            payload["registered_model_name"] = registered_model_name
        if model_version is not None:
            payload["model_version"] = str(model_version)
        if run_name is not None:
            payload["run_name"] = run_name
        if nested_run_name is not None:
            payload["nested_run_name"] = nested_run_name
        if metric_name is not None:
            payload["metric_name"] = metric_name
        if mode is not None:
            payload["mode"] = mode
        response = requests.post(f"{self.base_url}/predict/", json=payload)
        response.raise_for_status()
        return response.json()

    def health_check(self) -> Dict[str, Any]:
        response = requests.get(f"{self.base_url}/")
        response.raise_for_status()
        return response.json()

    def evaluate(
        self,
        experiment_name: str,
        model_type: str,
        evaluation_config: Dict[str, Any],
        registered_model_name: Optional[str] = None,
        model_version: Optional[Union[str, int]] = None,
        dataset_type: str = "test",
        model_params: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None,
        wait: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate a model. When wait=False (default), returns {job_id, status_url} (202).
        When wait=True, blocks and returns {result: ...} (200)."""
        payload = {
            "experiment_name": experiment_name,
            "model_type": model_type,
            "evaluation_config": evaluation_config,
            "dataset_type": dataset_type,
        }
        if model_params:
            payload["model_params"] = model_params
        if registered_model_name is not None:
            payload["registered_model_name"] = registered_model_name
        if model_version is not None:
            payload["model_version"] = str(model_version)
        if output_dir is not None:
            payload["output_dir"] = output_dir
        url = f"{self.base_url}/evaluate/"
        if wait:
            url += "?wait=true"
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    def get_eval_job_status(self, job_id: str) -> Dict[str, Any]:
        """Poll evaluation job status. When status is 'completed', response includes result."""
        response = requests.get(f"{self.base_url}/eval-jobs/{job_id}")
        response.raise_for_status()
        return response.json()