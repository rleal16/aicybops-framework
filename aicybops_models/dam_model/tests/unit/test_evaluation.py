"""
Merged evaluation tests: model input types, config, run_evaluation, error handling, and data helper.
Uses conftest fixtures: complete_model_file, valid_config, mock_pipeline, temp_dir.
"""

import logging
import pytest
import torch
import numpy as np
from pathlib import Path

import sys
_dam_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_dam_root))

from pipelines.evaluation import DAMEvaluationPipeline
from core.dam import DAMModel
from utils.evaluation import EvaluationConfig, EvaluationDataHelper


@pytest.fixture
def evaluator(complete_model_file, mock_pipeline, temp_dir, valid_config):
    """DAMEvaluationPipeline built from conftest fixtures."""
    return DAMEvaluationPipeline(
        model=str(complete_model_file),
        pipeline=mock_pipeline,
        config=valid_config,
        output_dir=str(temp_dir / "test_output"),
    )


# --- Model input types (from test_model_input_types) ---
def test_evaluator_accepts_string_path(complete_model_file, mock_pipeline, temp_dir, valid_config):
    """DAMEvaluationPipeline accepts model as str path; model_metadata has dimension keys."""
    evaluator = DAMEvaluationPipeline(
        model=str(complete_model_file),
        pipeline=mock_pipeline,
        config=valid_config,
        output_dir=str(temp_dir / "test_output"),
    )
    assert evaluator.model is not None
    assert isinstance(evaluator.model, DAMModel)
    assert "load_metrics_dim" in evaluator.model_metadata
    assert "traffic_metrics_dim" in evaluator.model_metadata
    assert "log_seq_dim" in evaluator.model_metadata
    assert "lstm_hidden_dim" in evaluator.model_metadata


def test_evaluator_accepts_path_object(complete_model_file, mock_pipeline, temp_dir, valid_config):
    """DAMEvaluationPipeline accepts model as Path; model_metadata has dimension keys."""
    evaluator = DAMEvaluationPipeline(
        model=Path(complete_model_file),
        pipeline=mock_pipeline,
        config=valid_config,
        output_dir=str(temp_dir / "test_output"),
    )
    assert evaluator.model is not None
    assert isinstance(evaluator.model, DAMModel)
    assert "load_metrics_dim" in evaluator.model_metadata


def test_evaluator_accepts_dam_model_instance(complete_model_file, mock_pipeline, temp_dir, valid_config):
    """DAMEvaluationPipeline accepts DAMModel instance; same instance and metadata."""
    model_state = torch.load(complete_model_file, map_location="cpu")
    dimensions = model_state["dimensions"]
    state_dict = model_state["model_state_dict"]
    model_instance = DAMModel(
        load_metrics_dim=dimensions["load_metrics_dim"],
        traffic_metrics_dim=dimensions["traffic_metrics_dim"],
        log_seq_dim=dimensions["log_seq_dim"],
        lstm_hidden_dim=dimensions["lstm_hidden_dim"],
    )
    model_instance.load_state_dict(state_dict)
    evaluator = DAMEvaluationPipeline(
        model=model_instance,
        pipeline=mock_pipeline,
        config=valid_config,
        output_dir=str(temp_dir / "test_output"),
    )
    assert evaluator.model is model_instance
    assert "load_metrics_dim" in evaluator.model_metadata


# --- Config (from test_dam_evaluation_methods) ---
def test_evaluation_config_load_valid(valid_config):
    """EvaluationConfig.load(valid_config) succeeds."""
    EvaluationConfig.load(valid_config)


def test_evaluation_config_invalid_raises():
    """Invalid config (e.g. invalid q_value or max_memory_gb) raises."""
    with pytest.raises(ValueError):
        EvaluationConfig.load({
            "evt_parameters": {
                "initial_threshold_quantile": 0.95,
                "min_peaks_for_fitting": 10,
                "q_values": [1.5],
            },
            "max_memory_gb": 8.0,
        })
    with pytest.raises(ValueError):
        EvaluationConfig.load({
            "test_scenarios": ["baseline"],
            "evt_parameters": {"initial_threshold_quantile": 0.95, "min_peaks_for_fitting": 10},
            "max_memory_gb": -1.0,
        })


# --- Run (one test: run_evaluation with minimal data_dict) ---
def test_run_evaluation_completes(evaluator, temp_dir):
    """run_evaluation with minimal data_dict (evaluation_loader + evaluation_labels) completes."""
    from torch.utils.data import DataLoader, TensorDataset
    # One batch of 10 samples so mock returns 10 anomaly_scores
    load_w = torch.randn(10, 10, 3)
    traffic_w = torch.randn(10, 10, 2)
    log_w = torch.randn(10, 10, 1)
    ds = TensorDataset(load_w, traffic_w, log_w)
    loader = DataLoader(ds, batch_size=5)
    data_dict = {
        "evaluation_loader": loader,
        "evaluation_labels": np.zeros(10, dtype=np.int64),
    }
    result = evaluator.run_evaluation(data_dict, output_dir=str(temp_dir / "eval_out"))
    assert "metrics" in result
    assert "anomaly_scores" in result or "num_samples" in result


# --- Error handling (from test_dam_evaluation_edge_cases) ---
def test_evaluator_nonexistent_model_raises(temp_dir, mock_pipeline, valid_config):
    """DAMEvaluationPipeline(model=non_existent_path, ...) raises FileNotFoundError."""
    non_existent = temp_dir / "missing_model.pth"
    with pytest.raises(FileNotFoundError):
        DAMEvaluationPipeline(
            model=str(non_existent),
            pipeline=mock_pipeline,
            config=valid_config,
            output_dir=str(temp_dir / "output"),
        )


# --- Helper (from test_evaluation_data_helper) ---
def test_helper_require_labels_true_success():
    """When require_labels=True and labels match length, returns labels."""
    logger = logging.getLogger("test")
    labels = np.array([0, 1, 0, 1, 0], dtype=int)
    helper = EvaluationDataHelper(logger, data_dict={"evaluation_labels": labels})
    result = helper.get_aligned_evaluation_labels(None, 5, require_labels=True)
    np.testing.assert_array_equal(result, labels)


def test_helper_labels_to_segments_one_segment():
    """labels_to_segments extracts one contiguous segment."""
    logger = logging.getLogger("test")
    helper = EvaluationDataHelper(logger)
    segs = helper.labels_to_segments(np.array([0, 1, 1, 1, 0]))
    assert segs == [(1, 3)]
