"""
Unit tests for experiment runner: run_name, data dict, and runner with mocked DAM.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_dam_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_dam_root))
sys.path.insert(0, str(_dam_root / "processing"))

from pipelines.experiments import ExperimentRunner, SweepSpec


def test_run_name_unique_and_deterministic():
    """_run_name produces deterministic names; different (config_index, seed) => different name."""
    spec = SweepSpec(config_paths=["a.json"], seeds=[0, 1])
    runner = ExperimentRunner(spec)
    names = {runner._run_name(i, s) for i in range(2) for s in [0, 1]}
    assert len(names) == 4
    assert runner._run_name(0, 0) == "config_000_seed_0"
    assert runner._run_name(1, 42) == "config_001_seed_42"


def test_build_data_dict_has_random_state(minimal_config_path):
    """_build_data_dict includes random_state from seed."""
    spec = SweepSpec(config_paths=[minimal_config_path], seeds=[99])
    runner = ExperimentRunner(spec)
    data = runner._build_data_dict(minimal_config_path, 99)
    assert data["random_state"] == 99
    assert "config_path" in data
    assert data["batch_size"] == 32


def test_get_epochs_from_overrides():
    """_get_epochs returns epochs from spec.overrides when set."""
    spec = SweepSpec(config_paths=["a.json"], seeds=[0], overrides={"epochs": 3})
    runner = ExperimentRunner(spec)
    assert runner._get_epochs() == 3


def test_get_epochs_default():
    """_get_epochs returns 10 when overrides has no epochs."""
    spec = SweepSpec(config_paths=["a.json"], seeds=[0])
    runner = ExperimentRunner(spec)
    assert runner._get_epochs() == 10


def test_sweep_spec_requires_eval_config_when_run_evaluation():
    """SweepSpec raises ValueError when run_evaluation=True and eval_config is None."""
    with pytest.raises(ValueError, match="eval_config is required when run_evaluation is True"):
        SweepSpec(config_paths=["a.json"], seeds=[0], run_evaluation=True)


def test_runner_run_with_mocked_dam(minimal_config_path):
    """Runner with run_evaluation=False uses train_test_validate (BaseModel); once per (config, seed)."""
    spec = SweepSpec(
        config_paths=[minimal_config_path],
        seeds=[7],
        run_evaluation=False,
        overrides={"epochs": 1},
    )
    with patch("core.dam_anomaly_detector.DAMAnomalyDetector") as MockDAM:
        mock_dam = MagicMock()
        MockDAM.return_value = mock_dam
        mock_dam.run.return_value.__enter__ = lambda s: None
        mock_dam.run.return_value.__exit__ = lambda s, *a: None
        mock_dam.train_test_validate.return_value = {"train_loss": 0.1, "val_loss": 0.2}

        runner = ExperimentRunner(spec)
        result = runner.run()

    assert result["failed_runs"] == []
    assert mock_dam.train_test_validate.call_count == 1
    call_kwargs = mock_dam.train_test_validate.call_args[1]
    assert call_kwargs["data"]["random_state"] == 7
    assert call_kwargs["epochs"] == 1
    assert call_kwargs["config_path"] == minimal_config_path
    assert call_kwargs["seed"] == 7


def test_runner_run_failure_recorded(minimal_config_path):
    """On train_test_validate failure, run is recorded in failed_runs and runner continues (unless fail_fast)."""
    spec = SweepSpec(
        config_paths=[minimal_config_path],
        seeds=[0, 1],
        run_evaluation=False,
        fail_fast=False,
    )
    with patch("core.dam_anomaly_detector.DAMAnomalyDetector") as MockDAM:
        mock_dam = MagicMock()
        MockDAM.return_value = mock_dam
        mock_dam.run.return_value.__enter__ = lambda s: None
        mock_dam.run.return_value.__exit__ = lambda s, *a: None
        mock_dam.train_test_validate.side_effect = [RuntimeError("fail"), {"train_loss": 0.1}]

        runner = ExperimentRunner(spec)
        result = runner.run()

    assert len(result["failed_runs"]) == 1
    assert result["failed_runs"][0]["run_name"] == "config_000_seed_0"
    assert "fail" in result["failed_runs"][0]["error"]


def test_runner_fail_fast_raises(minimal_config_path):
    """With fail_fast=True, first failure is re-raised."""
    spec = SweepSpec(
        config_paths=[minimal_config_path],
        seeds=[0],
        run_evaluation=False,
        fail_fast=True,
    )
    with patch("core.dam_anomaly_detector.DAMAnomalyDetector") as MockDAM:
        mock_dam = MagicMock()
        MockDAM.return_value = mock_dam
        mock_dam.run.return_value.__enter__ = lambda s: None
        mock_dam.run.return_value.__exit__ = lambda s, *a: None
        mock_dam.train_test_validate.side_effect = RuntimeError("fail")

        runner = ExperimentRunner(spec)
        with pytest.raises(RuntimeError, match="fail"):
            runner.run()
