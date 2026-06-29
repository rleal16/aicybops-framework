"""
Tests for DAMModelLoader: load_checkpoint and get_model.
"""

import pytest
import tempfile
import shutil
import torch
from pathlib import Path
import sys
_dam_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_dam_root))

from core.dam import DAMModel
from utils.training import DAMModelLoader, LoadedCheckpoint


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def valid_checkpoint_path(temp_dir):
    """Create a minimal valid .pth checkpoint with model_state_dict and dimensions."""
    path = temp_dir / "valid.pth"
    torch.save(
        {
            "model_state_dict": {
                "lstm_load_metrics.weight_ih_l0": torch.randn(128, 3),
                "lstm_traffic_metrics.weight_ih_l0": torch.randn(128, 2),
                "lstm_log_seq.weight_ih_l0": torch.randn(128, 1),
                "lstm_load_metrics.weight_hh_l0": torch.randn(128, 32),
                "lstm_traffic_metrics.weight_hh_l0": torch.randn(128, 32),
                "lstm_log_seq.weight_hh_l0": torch.randn(128, 32),
                "lstm_load_metrics.bias_ih_l0": torch.randn(128),
                "lstm_load_metrics.bias_hh_l0": torch.randn(128),
                "lstm_traffic_metrics.bias_ih_l0": torch.randn(128),
                "lstm_traffic_metrics.bias_hh_l0": torch.randn(128),
                "lstm_log_seq.bias_ih_l0": torch.randn(128),
                "lstm_log_seq.bias_hh_l0": torch.randn(128),
                "fc_load.0.weight": torch.randn(32, 96),
                "fc_load.0.bias": torch.randn(32),
                "fc_load.2.weight": torch.randn(8, 32),
                "fc_load.2.bias": torch.randn(8),
                "fc_load.4.weight": torch.randn(3, 8),
                "fc_load.4.bias": torch.randn(3),
                "fc_traffic.0.weight": torch.randn(32, 96),
                "fc_traffic.0.bias": torch.randn(32),
                "fc_traffic.2.weight": torch.randn(8, 32),
                "fc_traffic.2.bias": torch.randn(8),
                "fc_traffic.4.weight": torch.randn(2, 8),
                "fc_traffic.4.bias": torch.randn(2),
                "fc_log.0.weight": torch.randn(32, 96),
                "fc_log.0.bias": torch.randn(32),
                "fc_log.2.weight": torch.randn(8, 32),
                "fc_log.2.bias": torch.randn(8),
                "fc_log.4.weight": torch.randn(1, 8),
                "fc_log.4.bias": torch.randn(1),
            },
            "dimensions": {
                "load_metrics_dim": 3,
                "traffic_metrics_dim": 2,
                "log_seq_dim": 1,
                "lstm_hidden_dim": 32,
            },
        },
        path,
    )
    return path


def test_load_checkpoint_returns_loaded_checkpoint(valid_checkpoint_path):
    """load_checkpoint(valid path) returns LoadedCheckpoint with state_dict and dimensions."""
    loader = DAMModelLoader(device="cpu")
    checkpoint = loader.load_checkpoint(valid_checkpoint_path)
    assert isinstance(checkpoint, LoadedCheckpoint)
    assert "lstm_load_metrics.weight_ih_l0" in checkpoint.state_dict
    assert checkpoint.dimensions.load_metrics_dim == 3
    assert checkpoint.dimensions.traffic_metrics_dim == 2
    assert checkpoint.dimensions.log_seq_dim == 1
    assert checkpoint.dimensions.lstm_hidden_dim == 32


def test_load_checkpoint_missing_model_state_dict_raises(temp_dir):
    """load_checkpoint with no model_state_dict raises ValueError."""
    path = temp_dir / "bad.pth"
    torch.save(
        {"dimensions": {"load_metrics_dim": 3, "traffic_metrics_dim": 2, "log_seq_dim": 1, "lstm_hidden_dim": 32}},
        path,
    )
    loader = DAMModelLoader(device="cpu")
    with pytest.raises(ValueError, match="missing 'model_state_dict'"):
        loader.load_checkpoint(path)


def test_load_checkpoint_missing_dimensions_raises(temp_dir):
    """load_checkpoint with no dimensions raises ValueError."""
    path = temp_dir / "bad.pth"
    torch.save({"model_state_dict": {"lstm_load_metrics.weight_ih_l0": torch.randn(128, 3)}}, path)
    loader = DAMModelLoader(device="cpu")
    with pytest.raises(ValueError, match="missing 'dimensions'"):
        loader.load_checkpoint(path)


def test_get_model_from_path_returns_dam_model(valid_checkpoint_path):
    """get_model(path) returns DAMModel with correct dimensions."""
    loader = DAMModelLoader(device="cpu")
    model = loader.get_model(valid_checkpoint_path)
    assert isinstance(model, DAMModel)
    assert model.load_metrics_dim == 3
    assert model.traffic_metrics_dim == 2
    assert model.log_seq_dim == 1
    assert model.lstm_hidden_dim == 32


def test_get_model_from_dam_model_returns_same_instance():
    """get_model(DAMModel) returns the same instance."""
    loader = DAMModelLoader(device="cpu")
    model_in = DAMModel(load_metrics_dim=4, traffic_metrics_dim=3, log_seq_dim=2, lstm_hidden_dim=64)
    model_out = loader.get_model(model_in)
    assert model_out is model_in
    assert model_out.load_metrics_dim == 4
