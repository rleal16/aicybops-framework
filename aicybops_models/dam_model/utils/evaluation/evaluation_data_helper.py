import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class EvaluationDataHelper:
    def __init__(
        self,
        logger: logging.Logger,
        data_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.logger = logger
        self.data_dict = data_dict

    def labels_to_segments(self, y_true: np.ndarray) -> List[Tuple[int, int]]:
        """Extract anomaly segments from sequence-level labels (consecutive 1s)."""
        segments = []
        in_segment = False
        segment_start = None
        for i, label in enumerate(y_true):
            if label == 1 and not in_segment:
                segment_start = i
                in_segment = True
            elif label == 0 and in_segment:
                segments.append((segment_start, i - 1))
                in_segment = False
                segment_start = None
        if in_segment:
            segments.append((segment_start, len(y_true) - 1))
        return segments

    def _count_samples_from_data(self, data: Dict[str, Any]) -> int:
        """Count samples from data dictionary (handles DataLoader or timestamps)."""
        if "evaluation_loader" in data:
            return sum(len(batch[0]) for batch in data["evaluation_loader"])
        if "timestamps" in data:
            return len(data["timestamps"])
        return 0

    def get_aligned_evaluation_labels(
        self,
        data: Optional[Dict[str, Any]],
        num_samples_from_stream: int,
        require_labels: bool = True,
    ) -> np.ndarray:
        """
        Return length-aligned sequence-level labels for evaluation.

        When require_labels=True (default): raises ValueError if labels are
        missing or if label length does not match num_samples_from_stream.
        When require_labels=False: returns all-normal (zeros) when labels are
        missing or length mismatch (opt-in for label-free evaluation).
        """
        if self.data_dict is not None and "evaluation_labels" in self.data_dict:
            labels = self.data_dict["evaluation_labels"]
            if labels is not None:
                self.logger.info(
                    f"Using labels from data_dict: {len(labels)} labels, "
                    f"{np.sum(labels)} anomalies ({100 * np.mean(labels):.2f}%)"
                )
                if len(labels) == num_samples_from_stream:
                    return np.asarray(labels, dtype=int)
                if require_labels:
                    raise ValueError(
                        f"Label length ({len(labels)}) does not match anomaly scores length "
                        f"({num_samples_from_stream}). Evaluation requires aligned labels."
                    )
                self.logger.warning(
                    f"Label length ({len(labels)}) doesn't match anomaly scores length "
                    f"({num_samples_from_stream}). Using all-normal fallback (require_labels=False)."
                )
                return np.zeros(num_samples_from_stream, dtype=int)

        n_samples = self._count_samples_from_data(data) if data is not None else 0
        if n_samples > 0:
            if require_labels:
                raise ValueError(
                    "No labels in data_dict. Evaluation requires 'evaluation_labels' from "
                    "the same source as training (DAMDataProcessor). Set require_labels=False "
                    "to allow evaluation without labels (all-normal assumption)."
                )
            self.logger.warning(
                f"No labels in data_dict, using all-normal fallback for {n_samples} samples."
            )
            return np.zeros(n_samples, dtype=int)

        if require_labels:
            raise ValueError(
                "No labels available for evaluation. data_dict must contain 'evaluation_labels' "
                "or provide an evaluation_loader. Set require_labels=False to use all-normal."
            )
        self.logger.warning("No labels available, using all-normal assumption for evaluation")
        return np.zeros(num_samples_from_stream, dtype=int)

    def validate_processed_data_structure(
        self,
        data: Dict[str, Any],
        expected_window_length: Optional[int],
        model_dims: Dict[str, int],
    ) -> None:
        """Validate that processed data has the correct structure and dimensions."""
        required_keys = ["load_windows", "traffic_windows", "log_windows"]
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required data key: {key}")
            if not isinstance(data[key], np.ndarray):
                raise ValueError(
                    f"Data key '{key}' must be numpy array, got {type(data[key])}"
                )

        load_windows = data["load_windows"]
        traffic_windows = data["traffic_windows"]
        log_windows = data["log_windows"]

        n_load = load_windows.shape[0]
        n_traffic = traffic_windows.shape[0]
        n_log = log_windows.shape[0]
        if not (n_load == n_traffic == n_log):
            raise ValueError(
                f"Mismatched sample counts: load={n_load}, "
                f"traffic={n_traffic}, log={n_log}"
            )
        if n_load == 0:
            raise ValueError("Cannot process empty dataset: no samples provided")

        window_default = (
            10 if expected_window_length is None else expected_window_length
        )
        w_load, w_traffic, w_log = (
            load_windows.shape[1],
            traffic_windows.shape[1],
            log_windows.shape[1],
        )
        if not (w_load == w_traffic == w_log == window_default):
            self.logger.warning(
                f"Window length mismatch. Expected: {window_default}, "
                f"Got: load={w_load}, traffic={w_traffic}, log={w_log}"
            )

        load_dim = load_windows.shape[2]
        traffic_dim = traffic_windows.shape[2]
        log_dim = log_windows.shape[2]
        expected_load = model_dims["load_metrics_dim"]
        expected_traffic = model_dims["traffic_metrics_dim"]
        expected_log = model_dims["log_seq_dim"]
        if load_dim != expected_load:
            raise ValueError(
                f"Load metrics dimension mismatch: expected {expected_load}, "
                f"got {load_dim}"
            )
        if traffic_dim != expected_traffic:
            raise ValueError(
                f"Traffic metrics dimension mismatch: expected {expected_traffic}, "
                f"got {traffic_dim}"
            )
        if log_dim != expected_log:
            raise ValueError(
                f"Log sequences dimension mismatch: expected {expected_log}, "
                f"got {log_dim}"
            )
        self.logger.info("Processed data structure validation passed")

    def format_processed_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format processed data for evaluation (float32 windows, timestamps, targets)."""
        formatted = {
            "load_windows": data["load_windows"].astype(np.float32),
            "traffic_windows": data["traffic_windows"].astype(np.float32),
            "log_windows": data["log_windows"].astype(np.float32),
        }
        n_samples = len(formatted["load_windows"])

        if "timestamps" in data:
            formatted["timestamps"] = data["timestamps"]
        else:
            formatted["timestamps"] = np.arange(n_samples)
            self.logger.warning(
                "No timestamps provided, generated sequential indices"
            )

        formatted["target_load"] = (
            data["target_load"].astype(np.float32)
            if "target_load" in data
            else data["load_windows"][:, -1, :].astype(np.float32)
        )
        formatted["target_traffic"] = (
            data["target_traffic"].astype(np.float32)
            if "target_traffic" in data
            else data["traffic_windows"][:, -1, :].astype(np.float32)
        )
        formatted["target_log"] = (
            data["target_log"].astype(np.float32)
            if "target_log" in data
            else data["log_windows"][:, -1, :].astype(np.float32)
        )
        formatted["metadata"] = data.get("metadata", {})
        formatted["original_length"] = n_samples
        self.logger.info("Data formatting completed")
        return formatted
