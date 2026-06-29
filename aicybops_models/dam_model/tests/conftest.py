"""
Shared pytest configuration and fixtures for DAM tests.

Provides common fixtures: temp_dir, complete_model_file, valid_config,
mock_pipeline, minimal_config_path.
"""

import json
import pytest
import tempfile
import shutil
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def minimal_config_path(tmp_path):
    """Minimal dam_config.json with required sections (for detector, config, integration)."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    path = config_dir / "dam_config.json"
    config = {
        "data_paths": {"metrics_csv": "m.csv", "log_file": "l.txt"},
        "model_architecture": {"lstm_hidden_dim": 32, "window_size": 10, "stride": 1, "align_freq": "1s"},
        "training": {
            "learning_rate": 0.0001,
            "batch_size": 32,
            "num_epochs": 10,
            "train_ratio": 0.8,
            "val_ratio": 0.2,
            "early_stopping": {"enabled": True, "patience": 5, "min_delta": 0.001, "mode": "min"},
        },
        "anomaly_detection": {"spot_type": "dSPOT", "risk_level": 0.001, "depth": 10, "init_quantile": 0.95},
    }
    path.write_text(json.dumps(config))
    return str(path)


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def complete_model_file(temp_dir):
    """Create complete model file with full DAM architecture"""
    model_path = temp_dir / "complete_model.pth"
    mock_state = {
        'model_state_dict': {
            # LSTM weights - format: (4*hidden_size, input_size)
            'lstm_load_metrics.weight_ih_l0': torch.randn(128, 3),      # 4*32=128, load_dim=3
            'lstm_traffic_metrics.weight_ih_l0': torch.randn(128, 2),   # 4*32=128, traffic_dim=2
            'lstm_log_seq.weight_ih_l0': torch.randn(128, 1),           # 4*32=128, log_dim=1
            
            # LSTM hidden-to-hidden weights
            'lstm_load_metrics.weight_hh_l0': torch.randn(128, 32),     # 4*32, 32
            'lstm_traffic_metrics.weight_hh_l0': torch.randn(128, 32),
            'lstm_log_seq.weight_hh_l0': torch.randn(128, 32),
            
            # LSTM biases
            'lstm_load_metrics.bias_ih_l0': torch.randn(128),
            'lstm_load_metrics.bias_hh_l0': torch.randn(128),
            'lstm_traffic_metrics.bias_ih_l0': torch.randn(128),
            'lstm_traffic_metrics.bias_hh_l0': torch.randn(128),
            'lstm_log_seq.bias_ih_l0': torch.randn(128),
            'lstm_log_seq.bias_hh_l0': torch.randn(128),
            
            # Dense layer weights - input is 3*32=96 (concatenated hidden states)
            'fc_load.0.weight': torch.randn(32, 96),       # First dense layer
            'fc_load.0.bias': torch.randn(32),
            'fc_load.2.weight': torch.randn(8, 32),        # Second dense layer
            'fc_load.2.bias': torch.randn(8),
            'fc_load.4.weight': torch.randn(3, 8),         # Output layer (load_dim=3)
            'fc_load.4.bias': torch.randn(3),
            
            'fc_traffic.0.weight': torch.randn(32, 96),
            'fc_traffic.0.bias': torch.randn(32),
            'fc_traffic.2.weight': torch.randn(8, 32),
            'fc_traffic.2.bias': torch.randn(8),
            'fc_traffic.4.weight': torch.randn(2, 8),      # Output layer (traffic_dim=2)
            'fc_traffic.4.bias': torch.randn(2),
            
            'fc_log.0.weight': torch.randn(32, 96),
            'fc_log.0.bias': torch.randn(32),
            'fc_log.2.weight': torch.randn(8, 32),
            'fc_log.2.bias': torch.randn(8),
            'fc_log.4.weight': torch.randn(1, 8),          # Output layer (log_dim=1)
            'fc_log.4.bias': torch.randn(1),
        },
        'dimensions': {
            'load_metrics_dim': 3,
            'traffic_metrics_dim': 2,
            'log_seq_dim': 1,
            'lstm_hidden_dim': 32
        }
    }
    torch.save(mock_state, model_path)
    return model_path


@pytest.fixture
def sample_processed_data():
    """Create sample processed data for testing"""
    return {
        'load_windows': np.random.rand(10, 10, 3).astype(np.float32),
        'traffic_windows': np.random.rand(10, 10, 2).astype(np.float32),
        'log_windows': np.random.rand(10, 10, 1).astype(np.float32),
    }


@pytest.fixture
def sample_formatted_data():
    """Create sample formatted data for testing model inference"""
    return {
        'load_windows': np.random.rand(5, 10, 3).astype(np.float32),
        'traffic_windows': np.random.rand(5, 10, 2).astype(np.float32),
        'log_windows': np.random.rand(5, 10, 1).astype(np.float32),
        'target_load': np.random.rand(5, 3).astype(np.float32),
        'target_traffic': np.random.rand(5, 2).astype(np.float32),
        'target_log': np.random.rand(5, 1).astype(np.float32),
        'timestamps': [f"2024-07-{i+1:02d}" for i in range(5)]
    }


@pytest.fixture
def valid_config():
    """Create valid evaluation configuration (required by DAMEvaluationPipeline)"""
    return {
        "test_scenarios": ["baseline"],
        "evt_parameters": {
            "initial_threshold_quantile": 0.95,
            "min_peaks_for_fitting": 10,
            "q_values": [1e-3, 5e-3],
        },
        "max_memory_gb": 8.0
    }


@pytest.fixture
def mock_pipeline():
    """Create a mock DAMPipeline for testing"""
    from unittest.mock import MagicMock
    from utils import SPOT
    
    mock_pipe = MagicMock()
    mock_pipe.anomaly_detector = SPOT(risk_level=0.01)
    mock_pipe.run_anomaly_detection_on_stream = MagicMock(return_value={
        'alarms': [],
        'thresholds': [0.5] * 10,
        'anomaly_scores': [0.1] * 10,
        'predictions': []
    })
    return mock_pipe 