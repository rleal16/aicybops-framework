"""
Tests for DAMAnomalyDetector contract: validate/test require data; load_model requires dimensions and unified_config.
"""

import json
import pytest
import torch
from pathlib import Path

import sys
_dam_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_dam_root))

from core.dam_anomaly_detector import DAMAnomalyDetector


@pytest.fixture
def file_tracking_uri(tmp_path):
    """File-based MLflow tracking URI for tests."""
    return f"file://{tmp_path}/mlruns"


def test_validate_raises_when_no_data_and_no_train(minimal_config_path, file_tracking_uri):
    """validate() raises when data is not provided and train() was not called."""
    dam = DAMAnomalyDetector(
        experiment_name="test_contract",
        config_path=minimal_config_path,
        tracking_uri=file_tracking_uri,
    )
    with pytest.raises(ValueError) as exc_info:
        dam.validate()
    assert "No data available" in str(exc_info.value) or "data" in str(exc_info.value).lower()


def test_test_raises_when_no_data_and_no_train(minimal_config_path, file_tracking_uri):
    """test() raises when data is not provided and train() was not called."""
    dam = DAMAnomalyDetector(
        experiment_name="test_contract",
        config_path=minimal_config_path,
        tracking_uri=file_tracking_uri,
    )
    with pytest.raises(ValueError) as exc_info:
        dam.test()
    assert "No data available" in str(exc_info.value) or "data" in str(exc_info.value).lower()


def test_validate_raises_when_no_pipeline(minimal_config_path, file_tracking_uri):
    """validate() raises when pipeline is None (model not trained)."""
    dam = DAMAnomalyDetector(
        experiment_name="test_contract",
        config_path=minimal_config_path,
        tracking_uri=file_tracking_uri,
    )
    data = {"config_path": minimal_config_path, "window_size": 10, "stride": 1, "align_freq": "1s", "batch_size": 32}
    with pytest.raises(ValueError) as exc_info:
        dam.validate(data=data)
    assert "trained" in str(exc_info.value).lower() or "pipeline" in str(exc_info.value).lower()


def test_load_model_raises_when_dimensions_missing(tmp_path, file_tracking_uri):
    """load_model raises when checkpoint has no 'dimensions' key."""
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({
        "data_paths": {"metrics_csv": "m.csv", "log_file": "l.txt"},
        "model_architecture": {"lstm_hidden_dim": 32, "window_size": 10, "stride": 1, "align_freq": "1s"},
        "training": {"learning_rate": 0.0001, "batch_size": 32, "num_epochs": 10, "train_ratio": 0.8, "val_ratio": 0.2, "early_stopping": {"enabled": True, "patience": 5, "min_delta": 0.001, "mode": "min"}},
        "anomaly_detection": {"spot_type": "dSPOT", "risk_level": 0.001, "depth": 10, "init_quantile": 0.95},
    }))
    bad_path = tmp_path / "bad.pth"
    torch.save({"model_state_dict": {}}, bad_path)
    dam = DAMAnomalyDetector(experiment_name="x", config_path=str(config_file), tracking_uri=file_tracking_uri)
    with pytest.raises(ValueError) as exc_info:
        dam.load_model(str(bad_path))
    assert "dimensions" in str(exc_info.value).lower()


def test_load_model_raises_when_unified_config_missing(tmp_path, minimal_config_path, file_tracking_uri):
    """load_model raises when checkpoint has dimensions but no unified_config."""
    from core.dam import DAMModel
    checkpoint_path = tmp_path / "no_unified.pth"
    model = DAMModel(load_metrics_dim=3, traffic_metrics_dim=2, log_seq_dim=1, lstm_hidden_dim=32)
    state = {
        "model_state_dict": model.state_dict(),
        "dimensions": {"load_metrics_dim": 3, "traffic_metrics_dim": 2, "log_seq_dim": 1, "lstm_hidden_dim": 32},
        "config": {"window_size": 10, "stride": 1, "align_freq": "1s", "lstm_hidden_dim": 32},
    }
    torch.save(state, checkpoint_path)
    dam = DAMAnomalyDetector(experiment_name="x", config_path=minimal_config_path, tracking_uri=file_tracking_uri)
    with pytest.raises(ValueError) as exc_info:
        dam.load_model(str(checkpoint_path))
    assert "unified_config" in str(exc_info.value)
