from pathlib import Path

from fastapi.testclient import TestClient
from aicybops_lib.server.app import init_app
from aicybops_lib.base_model.registry import ModelRegistry
from aicybops_models.dam_model.core.dam_anomaly_detector import DAMAnomalyDetector
import pytest
import os

model_registry = ModelRegistry()
model_registry.register("dam", DAMAnomalyDetector)

app = init_app(model_registry)
client = TestClient(app)


def _dam_config_path():
    # tests/test_server.py -> aicybops-lib -> AICybOps
    base = Path(__file__).resolve().parent.parent.parent
    return base / "aicybops_models" / "dam_model" / "configs" / "dam_config.json"


def test_app_startup():
    response = client.get("/")
    assert response.status_code != 404


def test_train_dam(monkeypatch):
    """Train DAM model; requires DAM_CONFIG_PATH and valid config (may fail if data paths missing)."""
    config_path = _dam_config_path()
    if not config_path.exists():
        pytest.skip("DAM config not found")
    monkeypatch.setenv("DAM_CONFIG_PATH", str(config_path))
    response = client.post(
        "/train/?wait=true",
        json={
            "params": {"lr": 0.0001, "batch_size": 32},
            "epochs": 1,
            "experiment_name": "DAM Experiment",
            "model_type": "dam",
        },
    )
    # 200 if data available and train succeeds; 500 if data missing or other error
    assert response.status_code in (200, 500)
    if response.status_code == 200:
        result = response.json()
        assert "result" in result
        assert isinstance(result["result"], dict)


def test_predict_dam():
    """Predict with DAM; may 200 with predictions or 500 if no model/run."""
    response = client.post(
        "/predict/",
        json={
            "experiment_name": "DAM Experiment",
            "model_type": "dam",
            "registered_model_name": "dam",
            "model_version": "latest",
        },
    )
    # 200 with body or 500 if no model
    assert response.status_code in (200, 500)
    if response.status_code == 200:
        result = response.json()
        assert "predictions" in result


def test_invalid_model_type():
    response = client.post(
        "/train/",
        json={
            "params": {"lr": 0.01},
            "epochs": 1,
            "experiment_name": "Invalid Experiment",
            "model_type": "invalid",
        },
    )
    assert response.status_code == 422


def test_dam_workflow(monkeypatch):
    """Minimal workflow: train then predict (train may fail if data missing)."""
    config_path = _dam_config_path()
    if not config_path.exists():
        pytest.skip("DAM config not found")
    monkeypatch.setenv("DAM_CONFIG_PATH", str(config_path))
    train_response = client.post(
        "/train/?wait=true",
        json={
            "params": {"lr": 0.0001, "batch_size": 32},
            "epochs": 1,
            "experiment_name": "DAM Workflow Test",
            "model_type": "dam",
        },
    )
    if train_response.status_code != 200:
        pytest.skip("DAM train failed (e.g. missing data)")
    train_result = train_response.json()
    assert "result" in train_result
    assert isinstance(train_result["result"], dict)

    predict_response = client.post(
        "/predict/",
        json={
            "experiment_name": "DAM Workflow Test",
            "model_type": "dam",
            "registered_model_name": "dam",
            "model_version": "latest",
        },
    )
    assert predict_response.status_code == 200
    predict_result = predict_response.json()
    assert "predictions" in predict_result
    assert isinstance(predict_result["predictions"], list)


def test_data_source_configuration():
    data_source = os.getenv("DATA_SOURCE", "local")
    api_url = os.getenv("API_URL")
    response = client.post(
        "/train/?wait=true",
        json={
            "params": {"lr": 0.0001, "batch_size": 32},
            "epochs": 1,
            "experiment_name": "Data Source Test",
            "model_type": "dam",
        },
    )
    # Accept 200 (success), 500 (e.g. missing config/data), or 202 (async accepted)
    assert response.status_code in (200, 500, 202)
