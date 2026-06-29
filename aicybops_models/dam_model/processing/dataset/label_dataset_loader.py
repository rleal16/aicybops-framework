import json
import logging
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class LabelDatasetLoader:
    def __init__(self, config_path: str, use_pydantic: bool = True):
        self.config_path = Path(config_path)
        # Environment variable USE_PYDANTIC_CONFIG overrides this toggle.
        env_value = os.getenv('USE_PYDANTIC_CONFIG')
        self.use_pydantic = (env_value.lower() == 'true') if env_value is not None else use_pydantic
        self.labels_df = None
        self.sequence_labels = None

        if self.use_pydantic:
            try:
                from ..config import DAMConfigLoader
                self.config_loader = DAMConfigLoader(str(config_path))
                self.config = self.config_loader.get_full_config()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load Pydantic config with DAMConfigLoader for config file '{config_path}': {type(e).__name__}: {e}"
                ) from e
        else:
            self.config = None
            self._load_config()

    def _load_config(self):
        """Load config if not already loaded (legacy fallback)."""
        if self.config is None:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        return self.config

    def _convert_timestamp(self, df: pd.DataFrame, timestamp_col: str, timestamp_format: str) -> pd.Series:
        if timestamp_format == 'unix_ms':
            return pd.to_datetime(df[timestamp_col], unit='ms')
        elif timestamp_format == 'unix_s':
            return pd.to_datetime(df[timestamp_col], unit='s')
        else:
            return pd.to_datetime(df[timestamp_col])

    def _get_label_path(self, label_config: dict, label_path: Optional[str] = None) -> Path:
        if label_path is None:
            if 'path' not in label_config:
                raise ValueError(
                    f"Label dataset path not found in config. "
                    f"Please add 'path' to 'label_dataset' section in {self.config_path}"
                )
            # Resolve relative to config directory (same convention as metrics/log paths).
            config_dir = self.config_path.parent
            label_path = (config_dir / label_config['path']).resolve()
        else:
            label_path = Path(label_path)

        if not label_path.exists():
            raise FileNotFoundError(
                f"Label file not found at {label_path}. "
                f"Please ensure the file exists or update the 'path' in config."
            )

        return label_path

    def _normalize_timezone(self, labels_df: pd.DataFrame, target_index: pd.DatetimeIndex) -> pd.DataFrame:
        labels_df = labels_df.copy()
        labels_tz = labels_df.index.tz
        target_tz = target_index.tz

        # Make label index match target timezone style.
        if target_tz is None:
            if labels_tz is not None:
                labels_df.index = labels_df.index.tz_localize(None)
        else:
            if labels_tz is None:
                labels_df.index = labels_df.index.tz_localize("UTC").tz_convert(target_tz)
            elif labels_tz != target_tz:
                labels_df.index = labels_df.index.tz_convert(target_tz)

        return labels_df

    def load_labels(self, label_path: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Load label dataset from CSV. Returns DataFrame with timestamp index and 'anomaly_label' column, or None if disabled."""
        config = self._load_config()
        # Support top-level, nested, and legacy label config layouts.
        label_config = (
            config.get('label_dataset')
            or config.get('label_generation')
            or (config.get('data_processing') or {}).get('label_dataset')
            or {}
        )
        if not isinstance(label_config, dict):
            label_config = label_config.model_dump() if hasattr(label_config, 'model_dump') else {}

        if not label_config.get('enabled', True):
            logger.info("Labels disabled in config")
            return None

        label_path = self._get_label_path(label_config, label_path)

        logger.info("Loading labels from %s", label_path)
        labels_df = pd.read_csv(label_path)

        # Support both new structure (data_processing) and old structure (model_training) for sequence_index/timestamp_format
        data_processing = config.get('data_processing', {})
        model_training = config.get('model_training', {})
        sequence_index = data_processing.get('sequence_index', model_training.get('sequence_index', config.get('sequence_index', '_time')))
        timestamp_format = data_processing.get('timestamp_format', model_training.get('timestamp_format', config.get('timestamp_format', 'unix_ms')))

        timestamp_col = label_config.get('timestamp_column', sequence_index)
        label_col = label_config.get('label_column', 'anomaly_label')
        timestamp_format = label_config.get('timestamp_format', timestamp_format)

        timestamps = self._convert_timestamp(labels_df, timestamp_col, timestamp_format)

        result_df = pd.DataFrame({
            'anomaly_label': labels_df[label_col].values
        }, index=timestamps)
        result_df.index.name = 'timestamp'

        # Defensive: label files may contain duplicate timestamps (e.g. when
        # generated from long-format metrics with many rows per _time).
        # Collapse to one label per timestamp so reindex/align downstream stays
        # valid. Prefer anomaly (1) over normal (0) over unlabellable (-1).
        if not result_df.index.is_unique:
            n_dupes = int(result_df.index.duplicated().sum())
            result_df = (
                result_df
                .assign(_p=result_df['anomaly_label'].map({1: 2, 0: 1, -1: 0}).fillna(0))
                .sort_values('_p', ascending=False)
                .loc[lambda d: ~d.index.duplicated(keep='first')]
                .drop(columns='_p')
                .sort_index()
            )
            result_df.index.name = 'timestamp'
            logger.warning(
                "Labels had %d duplicate timestamp(s); collapsed to one row per timestamp.",
                n_dupes,
            )

        self.labels_df = result_df

        # Compute anomaly ratio only over labelled points (0/1).
        labelled = result_df[result_df['anomaly_label'].isin([0, 1])]
        n_labelled = len(labelled)
        n_unlabellable = int((result_df['anomaly_label'] == -1).sum())
        logger.info("Loaded %d labels", len(result_df))
        if n_labelled > 0:
            logger.info("Anomaly ratio (excluding -1): %.2f%%", np.mean(labelled['anomaly_label']) * 100)
        if n_unlabellable > 0:
            logger.info("Unlabellable (-1): %d points", n_unlabellable)
        logger.info("Timestamp range: %s to %s", result_df.index.min(), result_df.index.max())

        return result_df

    def map_labels_to_sequences(self, timestamp_index: pd.DatetimeIndex, window_size: int, stride: int) -> np.ndarray:
        """
        Map point-level anomaly labels to sequence-level labels using timestamps.

        For each sequence: if any point is 1 (anomaly) -> sequence = 1;
        else if any point is -1 (unlabellable) -> sequence = -1; else -> 0 (normal).
        """
        if self.labels_df is None:
            raise ValueError("Labels must be loaded before mapping to sequences. Call load_labels() first.")

        # Same sequence-count logic as sequence generation.
        num_sequences = len(range(0, len(timestamp_index) - window_size + 1, stride))
        if num_sequences <= 0:
            raise ValueError(f"Not enough timestamps ({len(timestamp_index)}) for window size {window_size}")

        labels_df = self._normalize_timezone(self.labels_df, timestamp_index)

        # Align point labels to the metric timeline once (avoids per-window pandas slices).
        point_labels = (
            labels_df["anomaly_label"]
            .reindex(timestamp_index)
            .fillna(0)
            .astype(int)
            .to_numpy()
        )
        sequence_labels = np.zeros(num_sequences, dtype=int)
        for i in range(num_sequences):
            start = i * stride
            window = point_labels[start : start + window_size]
            if np.any(window == 1):
                sequence_labels[i] = 1
            elif np.any(window == -1):
                sequence_labels[i] = -1

        self.sequence_labels = sequence_labels

        num_normal = int(np.sum(self.sequence_labels == 0))
        num_anomalous = int(np.sum(self.sequence_labels == 1))
        num_unlabellable = int(np.sum(self.sequence_labels == -1))
        logger.info(
            "Sequence labels: %d total, %d normal (%.1f%%), %d anomalous (%.1f%%)",
            len(self.sequence_labels),
            num_normal, 100 * num_normal / len(self.sequence_labels),
            num_anomalous, 100 * num_anomalous / len(self.sequence_labels),
        )
        if num_unlabellable > 0:
            logger.info(
                "Unlabellable (-1): %d (%.1f%%)",
                num_unlabellable, 100 * num_unlabellable / len(self.sequence_labels),
            )

        return self.sequence_labels

    def get_labels(self) -> Optional[pd.DataFrame]:
        return self.labels_df

    def get_sequence_labels(self) -> Optional[np.ndarray]:
        return self.sequence_labels

    def verify_label_metrics_alignment(self, labels_df: pd.DataFrame, metrics_df: pd.DataFrame,
                                       metrics_csv_path: str) -> bool:
        """
        Verify that every metric timestamp has a corresponding label.

        Labels may have timestamps with no metric — that is allowed.
        Returns True if every metric timestamp has a label; False otherwise.
        """
        logger.info("Verifying label-metrics alignment...")

        config = self._load_config()
        label_config = (
            config.get('label_dataset')
            or config.get('label_generation')
            or (config.get('data_processing') or {}).get('label_dataset')
            or {}
        )
        if not isinstance(label_config, dict):
            label_config = label_config.model_dump() if hasattr(label_config, 'model_dump') else {}
        label_column = label_config.get('label_column', 'anomaly_label')
        timestamp_column = label_config.get('timestamp_column', config.get('sequence_index', '_time'))
        timestamp_format = label_config.get('timestamp_format', config.get('timestamp_format', 'unix_ms'))

        if label_column not in labels_df.columns:
            logger.warning("Labels DataFrame missing '%s' column (from config)", label_column)
            return False

        if not isinstance(labels_df.index, pd.DatetimeIndex):
            logger.warning("Labels DataFrame index is not a DatetimeIndex")
            return False

        logger.info("Loading raw metrics from %s...", metrics_csv_path)
        try:
            # Load only timestamp column for alignment checks.
            raw_metrics_df = pd.read_csv(
                metrics_csv_path,
                usecols=[timestamp_column],
                low_memory=False,
            )
        except Exception as e:
            logger.error("Could not load raw metrics CSV: %s", e)
            return False

        if timestamp_column not in raw_metrics_df.columns:
            logger.warning("Raw metrics CSV missing '%s' column", timestamp_column)
            return False

        if timestamp_format == 'unix_ms':
            raw_metrics_timestamps = pd.to_datetime(raw_metrics_df[timestamp_column], unit='ms')
        elif timestamp_format == 'unix_s':
            raw_metrics_timestamps = pd.to_datetime(raw_metrics_df[timestamp_column], unit='s')
        else:
            raw_metrics_timestamps = pd.to_datetime(raw_metrics_df[timestamp_column])

        # Normalize timezones before comparison.
        labels_timestamps = labels_df.index.to_series().copy()
        raw_metrics_timestamps_series = raw_metrics_timestamps.copy()

        if labels_timestamps.dt.tz is None:
            labels_timestamps = labels_timestamps.dt.tz_localize('UTC')
        if raw_metrics_timestamps_series.dt.tz is None:
            raw_metrics_timestamps_series = raw_metrics_timestamps_series.dt.tz_localize('UTC')
        if labels_timestamps.dt.tz != raw_metrics_timestamps_series.dt.tz:
            raw_metrics_timestamps_series = raw_metrics_timestamps_series.dt.tz_convert(labels_timestamps.dt.tz)

        num_labels = len(labels_df)
        num_metrics = len(raw_metrics_df)
        raw_metrics_ts_set = set(raw_metrics_timestamps_series)

        logger.info("Labels: %d rows, raw metrics: %d rows (%d unique timestamps)", num_labels, num_metrics, len(raw_metrics_ts_set))

        # Use processed metrics index (real timestamps only).
        metrics_index = metrics_df.index.to_series()
        if metrics_index.dt.tz is None:
            metrics_index = metrics_index.dt.tz_localize('UTC')
        if metrics_index.dt.tz != labels_timestamps.dt.tz:
            metrics_index = metrics_index.dt.tz_convert(labels_timestamps.dt.tz)
        metrics_ts_set = set(metrics_index)

        common_min = max(labels_df.index.min(), metrics_df.index.min())
        common_max = min(labels_df.index.max(), metrics_df.index.max())
        target_tz = labels_timestamps.dt.tz
        if common_min.tzinfo is None and target_tz is not None:
            common_min = pd.Timestamp(common_min).tz_localize('UTC')
        if common_max.tzinfo is None and target_tz is not None:
            common_max = pd.Timestamp(common_max).tz_localize('UTC')
        if common_min.tzinfo is not None and target_tz is not None and common_min.tz != target_tz:
            common_min = common_min.tz_convert(target_tz)
            common_max = common_max.tz_convert(target_tz)

        def _floor_s(ts):
            t = pd.Timestamp(ts)
            if t.tzinfo is None and target_tz is not None:
                t = t.tz_localize("UTC")
            if t.tzinfo is not None and hasattr(t, "floor"):
                return t.floor("s")
            return t

        # Compare floored-second timestamp sets in the overlapping range.
        labels_in_range = {_floor_s(t) for t in labels_timestamps if common_min <= t <= common_max}
        metrics_in_range = {_floor_s(t) for t in metrics_ts_set if common_min <= t <= common_max}

        if not metrics_in_range:
            logger.warning("No metrics in overlap with labels (labels [%s to %s])", common_min, common_max)
            return False

        # Every metric timestamp must have a label; extra label timestamps are allowed.
        missing_in_labels = metrics_in_range - labels_in_range
        if missing_in_labels:
            logger.warning("%d metric timestamps have no corresponding label", len(missing_in_labels))
            return False

        labels_only = labels_in_range - metrics_in_range
        if labels_only:
            logger.info(
                "Every metric row has a label. Overlap: %d metric rows, %d label-only timestamps (allowed).",
                len(metrics_in_range), len(labels_only),
            )
        else:
            logger.info("Every metric row has a label. Overlap: %d timestamps.", len(metrics_in_range))
        return True
