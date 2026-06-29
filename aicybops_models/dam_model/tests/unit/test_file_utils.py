"""
Tests for utils.file_utils: find_data_files, find_model_path, make_json_serializable.
"""

import json
import pytest
from pathlib import Path
import sys
_dam_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_dam_root))

from utils.file_utils import find_data_files, find_model_path, make_json_serializable


def test_find_data_files_success(tmp_path):
    """find_data_files returns (metrics_csv, log_file, config_path) when all exist."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_file = config_dir / "dam_config.json"
    metrics_csv = config_dir / "metrics.csv"
    log_file = config_dir / "logs.txt"
    metrics_csv.write_text("a,b\n1,2\n")
    log_file.write_text("log line")
    config_file.write_text(json.dumps({
        "data_paths": {"metrics_csv": "metrics.csv", "log_file": "logs.txt"},
    }))
    result = find_data_files(tmp_path)
    assert len(result) == 3
    metrics_path, log_path, cfg_path = result
    assert metrics_path == metrics_csv.resolve()
    assert log_path == log_file.resolve()
    assert cfg_path == config_file
    assert metrics_path.exists() and log_path.exists() and cfg_path.exists()


def test_find_data_files_missing_config_raises(tmp_path):
    """find_data_files raises FileNotFoundError when config file does not exist."""
    with pytest.raises(FileNotFoundError) as exc_info:
        find_data_files(tmp_path)
    assert "dam_config.json" in str(exc_info.value)


def test_find_data_files_missing_metrics_csv_raises(tmp_path):
    """find_data_files raises FileNotFoundError when metrics_csv path does not exist."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_file = config_dir / "dam_config.json"
    (config_dir / "logs.txt").write_text("")
    config_file.write_text(json.dumps({
        "data_paths": {"metrics_csv": "nonexistent_metrics.csv", "log_file": "logs.txt"},
    }))
    with pytest.raises(FileNotFoundError):
        find_data_files(tmp_path)


def test_find_model_path_explicit_exists_and_not_found(tmp_path):
    """find_model_path: explicit path exists returns Path; nonexistent path raises."""
    model_file = tmp_path / "model.pth"
    model_file.write_bytes(b"x")
    assert find_model_path(tmp_path, str(model_file)) == model_file
    with pytest.raises(FileNotFoundError):
        find_model_path(tmp_path, "nonexistent.pth")


def test_find_model_path_no_arg_found_in_models(tmp_path):
    """find_model_path returns Path when model exists in base_dir/models/dam_model.pth."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    model_file = models_dir / "dam_model.pth"
    model_file.write_bytes(b"x")
    assert find_model_path(tmp_path) == model_file


def test_find_data_files_custom_config_path_and_name(tmp_path):
    """find_data_files supports alternate config path/name via config_path_arg."""
    config_dir = tmp_path / "some_other_configs"
    config_dir.mkdir()
    config_file = config_dir / "custom_dam_config.json"
    metrics_csv = config_dir / "metrics.csv"
    log_file = config_dir / "logs.txt"

    metrics_csv.write_text("a,b\n1,2\n")
    log_file.write_text("log line")
    config_file.write_text(
        json.dumps(
            {
                "data_paths": {"metrics_csv": "metrics.csv", "log_file": "logs.txt"}
            }
        )
    )

    result_metrics, result_log, result_config = find_data_files(
        tmp_path, config_path_arg=str(config_file)
    )
    assert result_config == config_file
    assert result_metrics == metrics_csv.resolve()
    assert result_log == log_file.resolve()


def test_make_json_serializable_ndarray():
    """make_json_serializable converts numpy arrays and nested structures."""
    import numpy as np
    assert make_json_serializable(np.array([1, 2, 3])) == [1, 2, 3]
    obj = {"a": np.array([1.0, 2.0]), "b": [np.int64(3)]}
    assert make_json_serializable(obj) == {"a": [1.0, 2.0], "b": [3]}
