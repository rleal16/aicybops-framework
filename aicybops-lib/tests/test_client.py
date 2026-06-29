import pytest
from aicybops_lib.client.client import AICybOpsClient
from unittest.mock import patch


@pytest.fixture
def client():
    return AICybOpsClient(base_url="http://127.0.0.1:8000")


@patch("aicybops_lib.client.client.requests.post")
def test_train_model_dam(mock_post, client):
    mock_post.return_value.json.return_value = {
        "result": {"train_loss": 0.5, "val_loss": 0.6, "registered_model_name": "dam", "model_version": "1"},
    }
    result = client.train_model(
        params={"lr": 0.0001, "batch_size": 32},
        epochs=1,
        experiment_name="test",
        model_type="dam",
    )
    assert "result" in result
    assert "train_loss" in result["result"] or "registered_model_name" in result["result"]


@patch("aicybops_lib.client.client.requests.post")
def test_predict_dam(mock_post, client):
    mock_predictions = [0.1, 0.2, 0.9]
    mock_post.return_value.json.return_value = {"predictions": mock_predictions}
    result = client.predict(
        experiment_name="test",
        model_type="dam",
    )
    assert "predictions" in result
    assert isinstance(result["predictions"], list)
    assert len(result["predictions"]) == 3
