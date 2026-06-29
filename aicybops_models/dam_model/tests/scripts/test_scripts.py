"""
Tests for scripts: evaluate_dam.py and full_pipeline.py.
Missing model path or config leads to raise or non-zero exit.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
_dam_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_dam_root))


def test_evaluate_dam_main_missing_model_path_raises(tmp_path):
    """When --model-path is missing and not --train-first, find_model_path raises and propagates."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_file = config_dir / "dam_config.json"
    (config_dir / "m.csv").write_text("a,b\n1,2\n")
    (config_dir / "logs.txt").write_text("log")
    config_file.write_text(json.dumps({
        "data_paths": {"metrics_csv": "m.csv", "log_file": "logs.txt"},
        "model_architecture": {"lstm_hidden_dim": 32, "window_size": 10, "stride": 1, "align_freq": "1s"},
        "training": {"learning_rate": 0.0001, "batch_size": 32, "num_epochs": 10, "train_ratio": 0.8, "val_ratio": 0.2, "early_stopping": {"enabled": True, "patience": 5, "min_delta": 0.001, "mode": "min"}},
        "anomaly_detection": {"spot_type": "dSPOT", "risk_level": 0.001, "depth": 10, "init_quantile": 0.95},
    }))
    import scripts.evaluate_dam as ev
    with patch("scripts.evaluate_dam.find_data_files") as mock_find:
        with patch("scripts.evaluate_dam.find_model_path") as mock_model:
            mock_find.return_value = (config_dir / "m.csv", config_dir / "logs.txt", config_file)
            mock_model.side_effect = FileNotFoundError("Model path not found")
            with patch("sys.argv", ["evaluate_dam.py"]):
                with pytest.raises(FileNotFoundError, match="Model path not found"):
                    ev.main()


def test_evaluate_dam_main_missing_config_raises():
    """When config/data files are missing, find_data_files raises."""
    import scripts.evaluate_dam as ev
    with patch("scripts.evaluate_dam.find_data_files") as mock_find:
        mock_find.side_effect = FileNotFoundError("Config file not found")
        with patch("sys.argv", ["evaluate_dam.py"]):
            with pytest.raises(FileNotFoundError, match="Config file not found"):
                ev.main()


def test_train_model_raises_when_no_model_and_no_mlflow_artifact(tmp_path):
    """train_model raises RuntimeError when dam.model is None and no MLflow artifact found."""
    from scripts.evaluate_dam import train_model
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "data_paths": {"metrics_csv": "m.csv", "log_file": "l.txt"},
        "model_architecture": {"lstm_hidden_dim": 32, "window_size": 10, "stride": 1, "align_freq": "1s"},
        "training": {"learning_rate": 0.0001, "batch_size": 32, "num_epochs": 10, "train_ratio": 0.8, "val_ratio": 0.2, "early_stopping": {"enabled": True, "patience": 5, "min_delta": 0.001, "mode": "min"}},
        "anomaly_detection": {"spot_type": "dSPOT", "risk_level": 0.001, "depth": 10, "init_quantile": 0.95},
    }))
    dam = MagicMock()
    dam.model = None
    dam.tracking_uri = f"file://{tmp_path}"
    dam.train.return_value = {"train_loss": 0.1, "val_loss": 0.2}
    args = MagicMock()
    args.train_first = True
    args.batch_size = 32
    args.train_ratio = 0.8
    args.val_ratio = 0.2
    args.window_size = 10
    args.stride = 1
    args.learning_rate = 0.0001
    args.spot_type = "dSPOT"
    args.risk_level = 1e-3
    args.depth = 10
    args.epochs = 1
    args.quick_test = False
    with pytest.raises(RuntimeError) as exc_info:
        train_model(dam, args, config_path)
    assert "Could not save or find model" in str(exc_info.value) or "model" in str(exc_info.value).lower()


def test_full_pipeline_missing_required_arg_exits_nonzero():
    """full_pipeline with missing required --config-path exits non-zero (SystemExit 2)."""
    import scripts.full_pipeline as fp
    with patch("sys.argv", ["full_pipeline.py"]):
        with pytest.raises(SystemExit) as exc_info:
            fp.main()
    assert exc_info.value.code == 2
