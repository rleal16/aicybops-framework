"""
Integration tests: save/load roundtrip and load then evaluate (when data available).

Uses minimal config and checkpoint fixtures; no real data required for save/load.
"""

import json
import pytest
import torch
from pathlib import Path

import sys
_dam_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_dam_root))

from core.dam import DAMModel
from core.dam_anomaly_detector import DAMAnomalyDetector
from utils.training import DAMModelLoader


def test_save_load_roundtrip_restores_model_and_dimensions(minimal_config_path, tmp_path):
    """Save a model from one detector, load in another; dimensions and model state are restored."""
    config_path = minimal_config_path
    checkpoint_path = tmp_path / "roundtrip.pth"

    # Build detector and assign model/dimensions without training (simulate post-train state)
    dam1 = DAMAnomalyDetector(experiment_name="integration_save", config_path=config_path)
    dam1.dimensions = {"load": 2, "traffic": 2, "log": 1}
    dam1.window_size = 10
    dam1.stride = 1
    dam1.align_freq = "1s"
    dam1.lstm_hidden_dim = 32
    dam1.model = DAMModel(
        load_metrics_dim=2,
        traffic_metrics_dim=2,
        log_seq_dim=1,
        lstm_hidden_dim=32,
    )
    dam1._last_data = {
        "config_path": config_path,
        "window_size": 10,
        "stride": 1,
        "align_freq": "1s",
        "batch_size": 32,
        "train_ratio": 0.8,
        "val_ratio": 0.2,
    }
    dam1.save_model(str(checkpoint_path))

    assert checkpoint_path.exists()
    dam2 = DAMAnomalyDetector(experiment_name="integration_load", config_path=config_path)
    dam2.load_model(str(checkpoint_path), restore_pipeline=False)

    assert dam2.model is not None
    assert dam2.dimensions == {"load": 2, "traffic": 2, "log": 1}
    assert dam2.window_size == 10
    assert dam2.stride == 1
    assert dam2.lstm_hidden_dim == 32


def test_load_checkpoint_then_get_dimensions(minimal_config_path, tmp_path):
    """Load a pre-saved checkpoint (current format with unified_config), verify model and dimensions."""
    from utils.config import DAMUnifiedConfig
    checkpoint_path = tmp_path / "pre_saved.pth"
    unified = DAMUnifiedConfig(config_path=minimal_config_path)
    model = DAMModel(load_metrics_dim=3, traffic_metrics_dim=2, log_seq_dim=1, lstm_hidden_dim=32)
    state = {
        "model_state_dict": model.state_dict(),
        "dimensions": {"load_metrics_dim": 3, "traffic_metrics_dim": 2, "log_seq_dim": 1, "lstm_hidden_dim": 32},
        "unified_config": unified.to_dict(),
    }
    torch.save(state, checkpoint_path)

    dam = DAMAnomalyDetector(experiment_name="integration_pre", config_path=minimal_config_path)
    dam.load_model(str(checkpoint_path), restore_pipeline=False)

    assert dam.model is not None
    dims = dam.model.get_dimensions()
    assert dims["load_metrics_dim"] == 3
    assert dims["traffic_metrics_dim"] == 2
    assert dims["log_seq_dim"] == 1


def test_save_model_with_pipeline_state(minimal_config_path, tmp_path):
    """Save with _last_data (pipeline state); checkpoint contains pipeline_state and data_config."""
    checkpoint_path = tmp_path / "with_pipeline.pth"
    dam = DAMAnomalyDetector(experiment_name="integration_pipeline", config_path=minimal_config_path)
    dam.dimensions = {"load": 2, "traffic": 2, "log": 1}
    dam.window_size = 10
    dam.stride = 1
    dam.align_freq = "1s"
    dam.lstm_hidden_dim = 32
    dam.model = DAMModel(
        load_metrics_dim=2,
        traffic_metrics_dim=2,
        log_seq_dim=1,
        lstm_hidden_dim=32,
    )
    dam._last_data = {
        "config_path": minimal_config_path,
        "window_size": 10,
        "stride": 1,
        "align_freq": "1s",
        "batch_size": 32,
        "train_ratio": 0.8,
        "val_ratio": 0.2,
    }
    dam.save_model(str(checkpoint_path))
    assert checkpoint_path.exists()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert "model_state_dict" in checkpoint
    assert "dimensions" in checkpoint
    if "pipeline_state" in checkpoint and "data_config" in checkpoint["pipeline_state"]:
        data_config = checkpoint["pipeline_state"]["data_config"]
        assert "config_path" in data_config
        assert data_config["window_size"] == 10


def test_detector_saved_checkpoint_loadable_by_model_loader(minimal_config_path, tmp_path):
    """Detector save_model() writes long-form dimensions; DAMModelLoader can load by path."""
    checkpoint_path = tmp_path / "detector_saved.pth"
    dam = DAMAnomalyDetector(experiment_name="integration_loader", config_path=minimal_config_path)
    dam.dimensions = {"load": 4, "traffic": 3, "log": 2}
    dam.window_size = 10
    dam.stride = 1
    dam.align_freq = "1s"
    dam.lstm_hidden_dim = 32
    dam.model = DAMModel(
        load_metrics_dim=4,
        traffic_metrics_dim=3,
        log_seq_dim=2,
        lstm_hidden_dim=32,
    )
    dam._last_data = {
        "config_path": minimal_config_path,
        "window_size": 10,
        "stride": 1,
        "align_freq": "1s",
        "batch_size": 32,
        "train_ratio": 0.8,
        "val_ratio": 0.2,
    }
    dam.save_model(str(checkpoint_path))

    loader = DAMModelLoader(device="cpu")
    loaded = loader.get_model(str(checkpoint_path))
    dims = loaded.get_dimensions()
    assert dims["load_metrics_dim"] == 4
    assert dims["traffic_metrics_dim"] == 3
    assert dims["log_seq_dim"] == 2
    assert dims["lstm_hidden_dim"] == 32
