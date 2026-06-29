import gc
import logging
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from .analyzers.metrics_analyser import MetricsAnalyser
from .analyzers.log_analyser import LogAnalyzer
from .dataset.dataset_utils import DataCleaner, DatasetBuilder, GroupExtractor, get_dataloader_kwargs
from .dataset.label_dataset_loader import LabelDatasetLoader
from .collector_utils import collect_data_from_api

logger = logging.getLogger(__name__)


def _log_processing_stage(message: str) -> None:
    """Visible in docker logs even when logging is buffered."""
    logger.info(message)
    print(f"[DAMDataProcessor] {message}", flush=True)


def _max_timesteps_cap() -> int:
    """Safety cap on aligned timesteps (1 row per second). 0 disables."""
    raw = os.environ.get("DAM_MAX_TIMESTEPS", "900")
    try:
        return max(0, int(raw))
    except ValueError:
        return 900


def _collapse_duplicate_index(df: pd.DataFrame, name: str, reducer: str) -> pd.DataFrame:
    """Ensure a unique DatetimeIndex before pandas reindex calls."""
    if df.index.is_unique:
        return df.sort_index()

    duplicate_count = int(df.index.duplicated().sum())
    _log_processing_stage(
        f"Collapsing {duplicate_count} duplicate timestamp row(s) in {name} before reindex"
    )
    if reducer == "mean":
        return df.groupby(level=0).mean().sort_index()
    if reducer == "last":
        return df.groupby(level=0).last().sort_index()
    raise ValueError(f"Unsupported duplicate-index reducer: {reducer}")


def _create_sequences_from_group(
    group_df: pd.DataFrame, window_size: int, stride: int
) -> np.ndarray:
    """Build sliding windows with one pre-allocated array (avoids list + np.array copy)."""
    data = np.ascontiguousarray(group_df.to_numpy(dtype=np.float32, copy=False))
    n_rows, n_features = data.shape
    if n_rows < window_size:
        return np.empty((0, window_size, n_features), dtype=np.float32)

    n_seqs = (n_rows - window_size) // stride + 1
    if n_seqs <= 0:
        return np.empty((0, window_size, n_features), dtype=np.float32)

    try:
        from numpy.lib.stride_tricks import sliding_window_view

        # sliding_window_view(data, W, axis=0) returns shape
        # (n_seqs, n_features, W). DAM expects (n_seqs, W, n_features),
        # so swap the last two axes before returning.
        windows = sliding_window_view(data, window_shape=window_size, axis=0)
        windows = windows[::stride]
        windows = np.swapaxes(windows, 1, 2)
        return np.ascontiguousarray(windows, dtype=np.float32)
    except Exception:
        out = np.empty((n_seqs, window_size, n_features), dtype=np.float32)
        for i, start in enumerate(range(0, n_rows - window_size + 1, stride)):
            out[i] = data[start : start + window_size]
        return out


class DAMDataProcessor:
    def __init__(self, metrics_csv_path: str, log_file_path: str, config_path: str, window_size: int = 10, stride: int = 1, align_freq: str = '1s', use_api: bool = False, labels_csv_path: Optional[str] = None, anomaly_labels: Optional[np.ndarray] = None, use_pydantic: bool = True, scaler_stats: Optional[Dict[str, Dict[str, float]]] = None, use_session_time_range: bool = True, start: Optional[str] = None, training_window_minutes: int = 0):
        self.config_path = config_path
        self.use_session_time_range = use_session_time_range
        self.start = start
        self.training_window_minutes = training_window_minutes
        self.window_size = window_size
        self.stride = stride
        self.align_freq = align_freq
        self.use_api = use_api
        self.scaler_stats = scaler_stats
        env_value = os.getenv('USE_PYDANTIC_CONFIG')
        self.use_pydantic = (env_value.lower() == 'true') if env_value is not None else use_pydantic
        self.validation_results = {}
        self.groups = {}
        self.sequences = {}
        self.combined_df = None
        self.metrics_csv_path = metrics_csv_path
        self.log_file_path = log_file_path
        self._labels_csv_path_override = labels_csv_path
        self.metrics_analyser = None
        self.log_analyser = None
        self.label_loader = None
        self.anomaly_labels = anomaly_labels
        self.sequence_anomaly_labels = None
        self.train_anomaly_labels = None
        self.val_anomaly_labels = None
        self.test_anomaly_labels = None
        # When use_api=True: counts from DataCollector.get_data() (for downstream / client)
        self.data_collection_metrics_count = None
        self.data_collection_logs_count = None

    def _process_data(self):
        # When use_api=True, collect metrics/logs via API and generate labels for alignment.
        labels_path_override = None
        if self.use_api:
            data_paths = collect_data_from_api(self.config_path, use_session_time_range=self.use_session_time_range, start=self.start, training_window_minutes=self.training_window_minutes)
            self.metrics_csv_path = data_paths['metrics_csv_path']   # from API
            self.log_file_path = data_paths['log_file_path']        # from API
            labels_path_override = data_paths.get('labels_csv_path') # generated from metrics for alignment
            self.labels_csv_path = data_paths.get('labels_csv_path')  # for scripts (e.g. report_unlabellable_periods)
            self.data_collection_metrics_count = data_paths.get('metrics_count')
            self.data_collection_logs_count = data_paths.get('logs_count')
            gc.collect()
            _log_processing_stage("API data collection complete; starting metrics/log processing")
        elif self._labels_csv_path_override is not None:
            labels_path_override = self._labels_csv_path_override
            self.labels_csv_path = self._labels_csv_path_override

        _log_processing_stage("Initializing metrics analyser")
        self.metrics_analyser = MetricsAnalyser(self.metrics_csv_path, self.config_path, use_pydantic=self.use_pydantic)

        drain3_path = os.environ.get("DAM_DRAIN3_STATE_PATH")
        _log_processing_stage("Initializing log analyser (Drain3)")
        self.log_analyser = LogAnalyzer(self.log_file_path, drain3_state_path=drain3_path)
        gc.collect()

        try:
            self.label_loader = LabelDatasetLoader(self.config_path, use_pydantic=self.use_pydantic)
            self.label_loader.load_labels(label_path=labels_path_override)
        except (ValueError, FileNotFoundError) as e:
            logger.warning("Labels not available: %s", e)
            self.label_loader = None

        _log_processing_stage("Processing metrics (load, align, normalize)")
        metrics_df = self.metrics_analyser.get_processed_data(align_freq=self.align_freq, scaler_stats=self.scaler_stats)
        gc.collect()

        _log_processing_stage("Processing logs (aligned features)")
        log_df = self.log_analyser.get_processed_data()
        gc.collect()

        if self.label_loader is not None and self.label_loader.labels_df is not None:
            alignment_valid = self.label_loader.verify_label_metrics_alignment(
                self.label_loader.labels_df, metrics_df, self.metrics_csv_path
            )
            if not alignment_valid:
                raise ValueError(
                    "Label-metrics alignment verification failed! "
                    "Every metric row must have a corresponding label. "
                    "Ensure anomaly_labels.csv covers all timestamps present in metrics.csv."
                )

        if metrics_df.index.tz is None:
            metrics_df.index = metrics_df.index.tz_localize('UTC')
        if log_df.index.tz is None:
            log_df.index = log_df.index.tz_localize('UTC')

        min_time = max(metrics_df.index.min(), log_df.index.min())
        max_time = min(metrics_df.index.max(), log_df.index.max())
        if min_time > max_time:
            raise ValueError(
                "Metrics and logs have no overlapping time range. "
                "Metrics: [%s, %s]; Logs: [%s, %s]. "
                "Ensure the API returns metrics and logs for the same session window (check collect_metrics API and Influx query range)."
                % (metrics_df.index.min(), metrics_df.index.max(), log_df.index.min(), log_df.index.max())
            )

        if self.training_window_minutes > 0:
            window_end = min_time + pd.Timedelta(minutes=self.training_window_minutes)
            if max_time > window_end:
                max_time = window_end
                _log_processing_stage(
                    f"Clipping to first {self.training_window_minutes} min of overlap "
                    f"({min_time} .. {max_time})"
                )

        metrics_df = metrics_df[(metrics_df.index >= min_time) & (metrics_df.index <= max_time)]
        log_df = log_df[(log_df.index >= min_time) & (log_df.index <= max_time)]
        metrics_dupes = int(metrics_df.index.duplicated().sum())
        log_dupes = int(log_df.index.duplicated().sum())
        if metrics_dupes or log_dupes:
            _log_processing_stage(
                "Pre-reindex duplicate timestamps: "
                f"metrics={metrics_dupes}/{len(metrics_df)}, logs={log_dupes}/{len(log_df)}"
            )
        metrics_df = _collapse_duplicate_index(metrics_df, "metrics", reducer="mean")
        log_df = _collapse_duplicate_index(log_df, "logs", reducer="last")

        max_steps = _max_timesteps_cap()
        if max_steps > 0 and len(metrics_df) > max_steps:
            _log_processing_stage(
                f"Clipping timesteps from {len(metrics_df)} to DAM_MAX_TIMESTEPS={max_steps}"
            )
            metrics_df = metrics_df.iloc[:max_steps]
            log_df = log_df.reindex(metrics_df.index)

        _log_processing_stage(
            f"Aligned frames: metrics={metrics_df.shape}, logs={log_df.shape}"
        )

        common_index = metrics_df.index

        def _reindex(df):
            df = df.reindex(common_index)
            df = df.ffill(limit=3)
            df = df.interpolate(method='linear', limit=10, limit_direction='both')
            df = df.bfill(limit=3)
            return df
        metrics_df = _reindex(metrics_df)
        log_df = _reindex(log_df)

        logger.debug("After alignment: metrics %s (missing=%d), log %s (missing=%d)",
            metrics_df.shape, metrics_df.isna().sum().sum(),
            log_df.shape, log_df.isna().sum().sum())

        metrics_df = metrics_df.ffill().bfill()

        _log_processing_stage("Extracting metric/log groups from config")
        self.groups = GroupExtractor.extract_groups_from_config(
            self.config_path, metrics_df, log_df, use_pydantic=self.use_pydantic
        )
        _log_processing_stage(
            f"Data processing complete ({len(self.groups)} group(s))"
        )

    def _create_sequences(self):
        """
        Create sliding windows for each group from the separated groups.
        Returns sequences for all groups (dynamically from config) separately,
        ready for use with the DAM model.
        Label mapping is handled separately by _map_labels_to_sequences().
        """
        _log_processing_stage("Creating sequences per group")
        for group_name, group_df in self.groups.items():
            _log_processing_stage(
                f"Group '{group_name}' input shape={group_df.shape}"
            )
            self.sequences[group_name] = _create_sequences_from_group(
                group_df, self.window_size, self.stride
            )
            _log_processing_stage(
                f"Sequences for '{group_name}': shape={self.sequences[group_name].shape}"
            )

        self._map_labels_to_sequences()
        if self.sequence_anomaly_labels is not None:
            _log_processing_stage(
                f"Label mapping complete ({len(self.sequence_anomaly_labels)} sequence labels)"
            )
        else:
            _log_processing_stage("Label mapping skipped (no labels)")

    def _map_labels_to_sequences(self):
        if self.label_loader is None or self.label_loader.labels_df is None:
            self.sequence_anomaly_labels = None
            logger.info("No anomaly labels provided — sequence-level labels not computed")
            return

        if len(self.groups) == 0:
            raise ValueError("No groups available. Call _process_data() first.")
        
        timestamp_index = list(self.groups.values())[0].index

        self.sequence_anomaly_labels = self.label_loader.map_labels_to_sequences(
            timestamp_index=timestamp_index,
            window_size=self.window_size,
            stride=self.stride
        )

    def get_labels(self) -> Dict[str, Optional[np.ndarray]]:
        return {
            'train_labels': self.train_anomaly_labels,
            'val_labels': self.val_anomaly_labels,
            'test_labels': self.test_anomaly_labels
        }


    def _split_data(self, train_ratio: float = 0.8, val_ratio: float = 0.2, random_state: int = 42, use_random_state: bool = True, training_mode: str = 'supervised'):
        """
        Split data into train, validation, and test sets.

        supervised mode (requires labels):
        - Training set: Contains ONLY normal data (train_ratio of normal sequences)
        - Validation set: Mix of normal and anomalous (stratified from remaining normal + all anomalous)
        - Test set: Mix of normal and anomalous (stratified from remaining normal + all anomalous)

        unsupervised mode (no labels):
        - Shuffled split across all sequences (unsupervised training split)

        Args:
            train_ratio: Ratio of NORMAL sequences to use for training (supervised), or of all sequences (unsupervised)
            val_ratio: Ratio of val/test pool for validation (supervised), or of all sequences (unsupervised)
            random_state: Random seed for reproducibility
            training_mode: 'supervised' or 'unsupervised'

        Returns:
            Tuple of (train_data, val_data, test_data) dictionaries
        """
        if training_mode == 'unsupervised':
            return self._split_data_shuffled(train_ratio, val_ratio, random_state)

        # Supervised mode requires labels.
        if self.sequence_anomaly_labels is None:
            raise ValueError(
                "supervised training_mode requires anomaly labels. "
                "Configure 'label_dataset' in your config, or set training_mode to 'unsupervised'."
            )

        # Stratified splitting with normal-only training (exclude -1).
        num_sequences = len(self.sequence_anomaly_labels)
        usable_mask = self.sequence_anomaly_labels != -1
        normal_indices = np.where((self.sequence_anomaly_labels == 0) & usable_mask)[0]
        anomalous_indices = np.where((self.sequence_anomaly_labels == 1) & usable_mask)[0]
        n_unlabellable = int(np.sum(~usable_mask))
        if n_unlabellable > 0:
            logger.info("Excluded %d unlabellable (-1) sequences from train/val/test", n_unlabellable)

        logger.info("Normal sequences: %d, anomalous: %d", len(normal_indices), len(anomalous_indices))

        if len(normal_indices) == 0:
            raise ValueError(
                "No normal sequences found in the dataset. "
                "supervised training requires normal sequences to learn from."
            )
        
        if use_random_state:
            np.random.seed(random_state)
        shuffled_normal_indices = np.random.permutation(normal_indices)

        train_normal_size = int(len(normal_indices) * train_ratio)
        train_normal_indices = shuffled_normal_indices[:train_normal_size]
        remaining_normal_indices = shuffled_normal_indices[train_normal_size:]

        logger.debug("Training: %d normal sequences, remaining for val/test: %d",
            len(train_normal_indices), len(remaining_normal_indices))

        val_test_pool_indices = np.concatenate([remaining_normal_indices, anomalous_indices])
        val_test_labels = self.sequence_anomaly_labels[val_test_pool_indices]
        
        if len(val_test_pool_indices) == 0:
            raise ValueError("No sequences left for val/test after reserving train set.")
        
        # Stratified val/test split with small-pool fallback.
        n_pool = len(val_test_pool_indices)
        if n_pool == 1:
            val_n, test_n = 0, 1
        else:
            val_n = max(1, int(n_pool * val_ratio))
            test_n = n_pool - val_n
            if test_n < 1:
                test_n = 1
                val_n = n_pool - 1
        # Stratify needs at least 2 samples per class.
        unique_labels, counts = np.unique(val_test_labels, return_counts=True)
        min_class_count = int(counts.min()) if len(counts) else 0
        use_stratify = (
            n_pool >= 5 and val_n >= 1 and test_n >= 1 and min_class_count >= 2
        )
        try:
            if use_stratify:
                val_indices, test_indices = train_test_split(
                    val_test_pool_indices,
                    test_size=(1 - val_ratio),  # e.g., if val_ratio=0.2, test_size=0.8 means 20% val, 80% test
                    stratify=val_test_labels,
                    random_state=random_state
                )
            else:
                shuffled = np.random.RandomState(random_state).permutation(val_test_pool_indices)
                val_indices = shuffled[:val_n]
                test_indices = shuffled[val_n:]
        except ValueError as e:
            raise ValueError(f"Stratified split failed ({e}).")
        
        self.train_anomaly_labels = self.sequence_anomaly_labels[train_normal_indices]
        self.val_anomaly_labels = self.sequence_anomaly_labels[val_indices]
        self.test_anomaly_labels = self.sequence_anomaly_labels[test_indices]

        logger.info("Split: train=%d (%.1f%% anomaly), val=%d (%.1f%%), test=%d (%.1f%%)",
            len(train_normal_indices), np.mean(self.train_anomaly_labels) * 100,
            len(val_indices), np.mean(self.val_anomaly_labels) * 100,
            len(test_indices), np.mean(self.test_anomaly_labels) * 100)

        self.train_data = {}
        self.val_data = {}
        self.test_data = {}
        
        for group_name, seqs in self.sequences.items():
            self.train_data[group_name] = seqs[train_normal_indices]
            self.val_data[group_name] = seqs[val_indices]
            self.test_data[group_name] = seqs[test_indices]
        
        return self.train_data, self.val_data, self.test_data
    
    def _split_data_shuffled(self, train_ratio: float, val_ratio: float, random_state: int):
        """Shuffled split for unsupervised training mode (no labels). Creates train/val/test."""
        if not self.sequences:
            raise ValueError("No sequences available for splitting")
        num_sequences = len(list(self.sequences.values())[0])

        all_indices = np.arange(num_sequences)
        np.random.seed(random_state)
        shuffled_indices = np.random.permutation(all_indices)

        train_size = int(num_sequences * train_ratio)
        val_size = int(num_sequences * val_ratio)

        train_indices = shuffled_indices[:train_size]
        val_indices = shuffled_indices[train_size:train_size + val_size]
        test_indices = shuffled_indices[train_size + val_size:]

        self.train_data = {}
        self.val_data = {}
        self.test_data = {}

        for group_name, seqs in self.sequences.items():
            self.train_data[group_name] = seqs[train_indices]
            self.val_data[group_name] = seqs[val_indices]
            self.test_data[group_name] = seqs[test_indices]

        self.train_anomaly_labels = None
        self.val_anomaly_labels = None
        self.test_anomaly_labels = None

        logger.info("Shuffled split: train=%d, val=%d, test=%d",
            len(train_indices), len(val_indices), len(test_indices))

        return self.train_data, self.val_data, self.test_data
    
    def _create_targets(self, sequences: dict):
        targets = {}
        for group_name, sequence_array in sequences.items():
            targets[group_name] = sequence_array[:, -1, :]
        return targets
    
    def prepare_for_training(self, train_ratio: float = 0.8, val_ratio: float = 0.2,
                        batch_size: int = 32, max_train_samples: int = None,
                        max_test_samples: int = None, random_state: int = None,
                        training_mode: str = 'supervised'):
        """
        Prepare data for training with train/val/test splits.
        Args:
            train_ratio: Ratio for train split (default 0.8)
            val_ratio: Ratio of validation data (default 0.2)
            batch_size: Batch size for data loaders (default 32)
            max_train_samples: Optional limit on number of training samples
            max_test_samples: Optional limit on number of test samples
            random_state: Random seed for reproducibility (default None, uses 42)
            training_mode: 'supervised' (labels required) or 'unsupervised' (shuffled split)
        Returns:
            Tuple of (train_loader, val_loader, test_dataset, dimensions)
            - train_loader: DataLoader for training
            - val_loader: DataLoader for validation
            - test_dataset: TensorDataset for testing
            - dimensions: Dictionary with group names as keys and feature dimensions as values
        """
        random_state = 42 if random_state is None else random_state
        self._process_data()
        self._create_sequences()
        _log_processing_stage(f"Splitting data (mode={training_mode})")
        train_sequences, val_sequences, test_sequences = self._split_data(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            random_state=random_state,
            training_mode=training_mode
        )
        _log_processing_stage("Building targets and tensors for training")

        train_targets = self._create_targets(train_sequences)
        val_targets = self._create_targets(val_sequences)
        test_targets = self._create_targets(test_sequences)

        train_tensors, val_tensors, test_tensors = self._clean_and_convert_to_tensors(
            train_sequences, train_targets,
            val_sequences, val_targets,
            test_sequences, test_targets
        )
        _log_processing_stage("Tensors ready; creating DataLoaders")
        gc.collect()

        train_loader, val_loader, test_dataset = self._create_datasets_and_loaders(
            train_tensors, val_tensors, test_tensors,
            batch_size, max_train_samples, max_test_samples
        )

        dimensions = self._get_model_dimensions()

        return train_loader, val_loader, test_dataset, dimensions
    
    def prepare_for_validation(self, train_ratio: float = 0.8, val_ratio: float = 0.2,
                              batch_size: int = 32, max_train_samples: int = None,
                              max_test_samples: int = None, random_state: int = None):
        """
        Note: Creates a NEW split each time. For consistent splits with training,
        use prepare_for_training() once and extract val_loader from it, OR pass the same random_state.
        """
        _, val_loader, _, dimensions = self.prepare_for_training(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            batch_size=batch_size,
            max_train_samples=max_train_samples,
            max_test_samples=max_test_samples,
            random_state=random_state
        )
        
        return val_loader, dimensions
        
    def prepare_for_test(self, train_ratio: float = 0.8, val_ratio: float = 0.2,
                        batch_size: int = 32, max_train_samples: int = None,
                        max_test_samples: int = None, random_state: int = None):
        """
        Note: Creates a NEW split each time. For consistent splits with training,
        use prepare_for_training() once and extract test_dataset from it, OR pass the same random_state.
        """
        _, _, test_dataset, dimensions = self.prepare_for_training(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            batch_size=batch_size,
            max_train_samples=max_train_samples,
            max_test_samples=max_test_samples,
            random_state=random_state
        )
        
        return test_dataset, dimensions
    
    def prepare_for_prediction(self, batch_size: int = 32, include_targets: bool = False):
        """
        Prepare data for prediction without train/test split.
        
        Args:
            batch_size: Batch size for data loader (default 32)
            include_targets: If True, creates targets (for evaluation/testing scenarios where targets exist).
                           If False, does NOT create targets (for production prediction where targets are unavailable).
        
        Returns:
            Tuple of (prediction_loader, dimensions)
            - prediction_loader: DataLoader for prediction
            - dimensions: Dictionary with group names as keys and feature dimensions as values
        
        Note:
        - If include_targets=True: Returns loader with (load_seq, traffic_seq, log_seq, target_load, target_traffic, target_log) - 6 items
        - If include_targets=False: Returns loader with (load_seq, traffic_seq, log_seq) only - 3 items (targets not included)
        """
        self._process_data()
        self._create_sequences()

        prediction_sequences = self.sequences

        if include_targets:
            prediction_targets = self._create_targets(prediction_sequences)
        else:
            prediction_targets = None

        prediction_sequence_tensors = DataCleaner.clean_and_convert_dict_to_tensors(
            prediction_sequences, data_name="prediction sequences"
        )

        if include_targets:
            prediction_target_tensors = DataCleaner.clean_and_convert_dict_to_tensors(
                prediction_targets, data_name="prediction targets"
            )
            prediction_dataset = DatasetBuilder.create_dataset(
                prediction_sequence_tensors,
                prediction_target_tensors
            )
        else:
            prediction_dataset = DatasetBuilder.create_dataset(
                prediction_sequence_tensors,
                target_tensors=None
            )

        prediction_loader = DataLoader(
            prediction_dataset,
            batch_size=batch_size,
            shuffle=False,
            **get_dataloader_kwargs(),
        )
        dimensions = self._get_model_dimensions()

        return prediction_loader, dimensions
    
    def _clean_and_convert_to_tensors(self, train_sequences: dict, train_targets: dict,
                                     val_sequences: dict, val_targets: dict,
                                     test_sequences: dict, test_targets: dict) -> tuple:
        train_sequence_tensors = DataCleaner.clean_and_convert_dict_to_tensors(
            train_sequences, data_name="train sequences"
        )
        train_target_tensors = DataCleaner.clean_and_convert_dict_to_tensors(
            train_targets, data_name="train targets"
        )
        val_sequence_tensors = DataCleaner.clean_and_convert_dict_to_tensors(
            val_sequences, data_name="validation sequences"
        )
        val_target_tensors = DataCleaner.clean_and_convert_dict_to_tensors(
            val_targets, data_name="validation targets"
        )
        test_sequence_tensors = DataCleaner.clean_and_convert_dict_to_tensors(
            test_sequences, data_name="test sequences"
        )
        test_target_tensors = DataCleaner.clean_and_convert_dict_to_tensors(
            test_targets, data_name="test targets"
        )

        train_tensors = (train_sequence_tensors, train_target_tensors)
        val_tensors = (val_sequence_tensors, val_target_tensors)
        test_tensors = (test_sequence_tensors, test_target_tensors)

        return train_tensors, val_tensors, test_tensors
    
    def _create_datasets_and_loaders(self, train_tensors: tuple, val_tensors: tuple, test_tensors: tuple,
                                    batch_size: int, max_train_samples: int = None,
                                    max_test_samples: int = None):
        train_sequence_tensors, train_target_tensors = train_tensors
        val_sequence_tensors, val_target_tensors = val_tensors
        test_sequence_tensors, test_target_tensors = test_tensors

        train_dataset = DatasetBuilder.create_dataset(
            train_sequence_tensors,
            train_target_tensors,
            max_samples=max_train_samples
        )
        val_dataset = DatasetBuilder.create_dataset(
            val_sequence_tensors,
            val_target_tensors
        )
        train_loader, val_loader = DatasetBuilder.create_data_loaders(
            train_dataset,
            val_dataset,
            batch_size=batch_size
        )
        test_dataset = DatasetBuilder.create_dataset(
            test_sequence_tensors,
            test_target_tensors,
            max_samples=max_test_samples
        )

        return train_loader, val_loader, test_dataset
    
    def _get_model_dimensions(self) -> dict:
        """Get feature dimensions for each group (group_name values as keys)."""
        return {group_name: group_df.shape[1] for group_name, group_df in self.groups.items()}
            
    def diagnostics(self):
        print("\n=== Diagnostics: Data Alignment and Content ===")
        for group_name, group_df in self.groups.items():
            print(f"{group_name}: shape={group_df.shape}, index range=({group_df.index.min()}, {group_df.index.max()})")
            print(f"  Missing values: {group_df.isna().sum().sum()} (per column: {group_df.isna().sum().to_dict()})")
            print(f"  Head:\n{group_df.head(3)}\n  Tail:\n{group_df.tail(3)}\n")

        print("\n=== Diagnostics: Sequence Creation ===")
        for group_name, seqs in self.sequences.items():
            print(f"{group_name}: sequences shape={seqs.shape}")
            if seqs.size > 0:
                print(f"  First sequence (shape {seqs[0].shape}):\n{seqs[0]}")
                print(f"  Any NaNs in sequences: {np.isnan(seqs).sum()}\n")
            else:
                print("  WARNING: No sequences created!\n")

        print("\n=== Diagnostics: Train/Test Split ===")
        for group_name in self.train_data:
            print(f"{group_name}: train shape={self.train_data[group_name].shape}, test shape={self.test_data[group_name].shape}")
            if self.train_data[group_name].size > 0:
                print(f"  Train sample (first window):\n{self.train_data[group_name][0]}")
            if self.test_data[group_name].size > 0:
                print(f"  Test sample (first window):\n{self.test_data[group_name][0]}")

        print("\n=== Diagnostics: Data Leakage Check ===")
        for group_name in self.train_data:
            if self.train_data[group_name].size > 0 and self.test_data[group_name].size > 0:
                overlap = np.intersect1d(self.train_data[group_name][-1].flatten(), self.test_data[group_name][0].flatten())
                print(f"{group_name}: overlap between last train and first test window: {overlap}")


