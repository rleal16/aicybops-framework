import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .evaluation_data_helper import EvaluationDataHelper
from .evaluation_metrics import EvaluationMetrics
from utils.file_utils import make_json_serializable


@dataclass
class EvaluationResultData:
    """Result of a single evaluation run (stream detection + metrics)."""

    anomaly_scores: np.ndarray
    thresholds: np.ndarray
    alarms: List[int]
    metrics: Dict[str, Any]
    num_samples: int
    anomaly_rate: float
    actual_anomaly_rate: float

    def to_dict(
        self,
        evaluation_timestamp: Optional[str] = None,
        model_dimensions: Optional[Dict[str, Any]] = None,
        dam_f1_target: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Return the evaluation result as a serializable dict. Optionally include run metadata."""
        thresholds = np.asarray(self.thresholds)
        scores = np.asarray(self.anomaly_scores)
        thresholds_list = thresholds.tolist()
        scores_list = scores.tolist()
        n = len(thresholds)
        d = {
            "anomaly_scores": scores_list,
            "thresholds": thresholds_list,
            "threshold_initial": float(thresholds[0]) if n > 0 else None,
            "threshold_final": float(thresholds[-1]) if n > 0 else None,
            "threshold_mean": float(np.mean(thresholds)) if n > 0 else None,
            "alarms": self.alarms,
            "metrics": self.metrics,
            "num_samples": self.num_samples,
            "anomaly_rate": self.anomaly_rate,
            "actual_anomaly_rate": self.actual_anomaly_rate,
        }
        if (
            evaluation_timestamp is not None
            and model_dimensions is not None
            and dam_f1_target is not None
        ):
            d["evaluation_timestamp"] = evaluation_timestamp
            d["model_dimensions"] = model_dimensions
            d["dam_target_achieved"] = self.metrics.get("f1_score", 0.0) >= dam_f1_target
        return d


class EvaluationResult:
    """
    Builds the full evaluation results dict from stream results and data.
    All logic (labels→segments, result data, metrics, serialization, save) lives here.
    """

    def __init__(
        self,
        logger: logging.Logger,
        get_model_metadata: Callable[[], Dict[str, Any]],
        dam_f1_target: float,
    ) -> None:
        self.logger = logger
        self._data_helper = EvaluationDataHelper(logger)
        self._get_model_metadata = get_model_metadata
        self._dam_f1_target = dam_f1_target

    def save_results(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Save evaluation results to evaluation_results.json under output_dir."""
        self.logger.info("Saving evaluation results...")
        try:
            results_to_save = results.copy()
            results_file = output_dir / "evaluation_results.json"
            with open(results_file, "w") as f:
                serializable = make_json_serializable(results_to_save)
                json.dump(serializable, f, indent=2, default=str)
            self.logger.info(f"Results saved to {results_file}")
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            raise

    def _build_result_data(
        self,
        stream_results: Dict[str, Any],
        y_true: np.ndarray,
    ) -> EvaluationResultData:
        """Build EvaluationResultData from stream detection output and aligned labels."""
        alarms = stream_results["alarms"]
        thresholds = stream_results["thresholds"]
        anomaly_scores = stream_results["anomaly_scores"]

        thresholds = np.asarray(thresholds)
        anomaly_scores = np.asarray(anomaly_scores)

        n = len(anomaly_scores)
        classifications = np.zeros(n, dtype=int)
        classifications[alarms] = 1

        ground_truth_segments = self._data_helper.labels_to_segments(y_true)
        eval_metrics = EvaluationMetrics.from_segment_data(
            y_true=y_true,
            y_pred=classifications,
            y_scores=anomaly_scores,
            ground_truth_segments=ground_truth_segments,
        )
        metrics = eval_metrics.to_dict()

        return EvaluationResultData(
            anomaly_scores=anomaly_scores,
            thresholds=thresholds,
            alarms=alarms,
            metrics=metrics,
            num_samples=n,
            anomaly_rate=float(np.mean(classifications)),
            actual_anomaly_rate=float(np.mean(y_true)),
        )

    def build_result(
        self,
        stream_results: Dict[str, Any],
        data: Dict[str, Any],
        evaluation_timestamp: str,
        output_dir: Optional[Path] = None,
        require_labels: bool = True,
    ) -> Dict[str, Any]:
        """
        From stream results and data, build the full evaluation results dict.
        Resolves labels, builds result data, logs metrics, adds metadata, optionally saves.
        """
        self._data_helper.data_dict = data
        num_samples = len(stream_results["anomaly_scores"])
        y_true = self._data_helper.get_aligned_evaluation_labels(
            data, num_samples, require_labels=require_labels
        )
        result = self._build_result_data(stream_results, y_true)

        metrics = result.metrics
        self.logger.info(
            f"Evaluation completed. F1: {metrics['f1_score']:.3f}, "
            f"Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}"
        )

        evaluation_results = result.to_dict(
            evaluation_timestamp,
            self._get_model_metadata(),
            self._dam_f1_target,
        )

        if output_dir is not None:
            self.save_results(evaluation_results, output_dir)

        return evaluation_results
