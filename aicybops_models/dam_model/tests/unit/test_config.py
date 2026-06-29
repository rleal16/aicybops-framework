"""
Minimal tests for config: unified config and build_data_config (utils.config),
plus one Pydantic validation test (processing.config).
"""

import json
import pytest
from pathlib import Path
from pydantic import ValidationError

import sys
_dam_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_dam_root))
sys.path.insert(0, str(_dam_root / "processing"))

from utils.config import DAMUnifiedConfig, build_data_config


def test_unified_config_load_from_path(minimal_config_path):
    """DAMUnifiedConfig loads from JSON and exposes required sections."""
    cfg = DAMUnifiedConfig(config_path=minimal_config_path)
    arch = cfg.get_model_architecture()
    assert arch["window_size"] == 10
    assert arch["lstm_hidden_dim"] == 32
    training = cfg.get_training()
    assert training["batch_size"] == 32
    assert training["train_ratio"] == 0.8


def test_unified_config_missing_file_raises():
    """DAMUnifiedConfig raises FileNotFoundError when path does not exist."""
    with pytest.raises(FileNotFoundError):
        DAMUnifiedConfig(config_path="/nonexistent/dam_config.json")


def test_build_data_config_returns_expected_keys(minimal_config_path):
    """build_data_config returns dict with config_path, batch_size, train_ratio, etc."""
    config_dict = build_data_config(minimal_config_path)
    assert "config_path" in config_dict
    assert config_dict["use_api"] is False
    assert config_dict["batch_size"] == 32
    assert config_dict["train_ratio"] == 0.8
    assert config_dict["val_ratio"] == 0.2
    assert config_dict["window_size"] == 10


def test_pydantic_invalid_config_raises():
    """Pydantic DAMDataConfig catches invalid config (e.g. window_size < 1)."""
    from processing.config import DAMDataConfig
    with pytest.raises(ValidationError):
        DAMDataConfig(**{"window_size": -1})
