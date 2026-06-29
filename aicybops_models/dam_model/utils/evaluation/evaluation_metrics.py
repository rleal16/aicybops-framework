import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    roc_curve,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    roc_auc: float
    average_precision: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int

    def to_dict(self) -> Dict[str, Union[float, int]]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "accuracy": self.accuracy,
            "roc_auc": self.roc_auc,
            "average_precision": self.average_precision,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
        }

    @staticmethod
    def _segment_precision_recall_f1(
        detected_indices: List[int],
        ground_truth_segments: List[Tuple[int, int]],
    ) -> Tuple[float, float, float]:
        """Segment-based precision, recall, and F1. Returns (precision, recall, f1)."""
        detected_segments = sum(
            1
            for start_idx, end_idx in ground_truth_segments
            if any(start_idx <= idx <= end_idx for idx in detected_indices)
        )
        total_segments = len(ground_truth_segments)
        recall = detected_segments / total_segments if total_segments > 0 else 0.0

        segment_indices = set()
        for start_idx, end_idx in ground_truth_segments:
            segment_indices.update(range(start_idx, end_idx + 1))
        valid_detections = sum(1 for idx in detected_indices if idx in segment_indices)
        precision = (
            valid_detections / len(detected_indices) if detected_indices else 0.0
        )

        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        return precision, recall, f1

    @staticmethod
    def _confusion_and_accuracy(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Tuple[int, int, int, int, float]:
        """Confusion matrix counts and accuracy. Returns (tn, fp, fn, tp, accuracy)."""
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (1, 1):
            if y_true[0] == 0:
                tn, fp, fn, tp = int(cm[0, 0]), 0, 0, 0
            else:
                tn, fp, fn, tp = 0, 0, 0, int(cm[0, 0])
        else:
            tn, fp, fn, tp = (int(x) for x in cm.ravel())
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        return tn, fp, fn, tp, accuracy

    @staticmethod
    def _roc_auc_and_ap(
        y_true: np.ndarray, y_scores: np.ndarray
    ) -> Tuple[float, float]:
        """ROC AUC and average precision from scores. Returns (roc_auc, average_precision)."""
        try:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = float(auc(fpr, tpr))
        except ValueError:
            logger.warning("ROC AUC undefined (single class in y_true); returning 0.0")
            roc_auc = 0.0
        try:
            avg_precision = float(average_precision_score(y_true, y_scores))
        except ValueError:
            logger.warning("Average precision undefined (single class in y_true); returning 0.0")
            avg_precision = 0.0
        return roc_auc, avg_precision

    @classmethod
    def from_segment_data(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: np.ndarray,
        ground_truth_segments: List[Tuple[int, int]],
    ) -> "EvaluationMetrics":
        """
        Compute segment-based and sequence-level metrics from labels and scores.

        Precision/recall/F1 are segment-based; accuracy and confusion matrix
        are at sequence level; ROC AUC and average precision from scores.
        """
        detected_indices = np.where(y_pred == 1)[0].tolist()
        precision, recall, f1_score = cls._segment_precision_recall_f1(
            detected_indices, ground_truth_segments
        )
        tn, fp, fn, tp, accuracy = cls._confusion_and_accuracy(y_true, y_pred)
        roc_auc, average_precision = cls._roc_auc_and_ap(y_true, y_scores)

        return cls(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            accuracy=accuracy,
            roc_auc=roc_auc,
            average_precision=average_precision,
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
        )
