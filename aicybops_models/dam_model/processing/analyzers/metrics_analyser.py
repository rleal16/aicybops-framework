import logging
from typing import Dict, Any, Optional
from collections import defaultdict
import json, os
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class MetricsAnalyser:

    def __init__(self, metrics_csv_path: str, config_path: str, use_pydantic: bool = True):
        self.metrics_data = {}
        self.validation_results = {}
        self.metrics_csv_path = Path(metrics_csv_path)
        self.config_path = Path(config_path)

        # Toggle Pydantic config loading; env var takes precedence.
        env_value = os.getenv('USE_PYDANTIC_CONFIG')
        self.use_pydantic = (env_value.lower() == 'true') if env_value is not None else use_pydantic

        if self.use_pydantic:
            try:
                from ..config import DAMConfigLoader
                self.config_loader = DAMConfigLoader(str(config_path))
                self.core_metrics_config = self.config_loader.get_core_metrics_config()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load Pydantic config with DAMConfigLoader for config file '{config_path}': {type(e).__name__}: {e}"
                ) from e
        else:
            # Legacy fallback.
            self.core_metrics_config = self._build_core_metrics_from_config()

        # Backward-compatible mapping: metric_name -> measurement_name.
        self.core_metrics = {name: config['_measurement'] for name, config in self.core_metrics_config.items()}

        # Populated by normalize_metrics() for inference-time normalization.
        self.scaler_stats = {}

        if not self.metrics_csv_path.exists():
            raise FileNotFoundError(f"Metrics CSV file not found: {self.metrics_csv_path}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

    def _build_core_metrics_from_config(self) -> Dict[str, Dict[str, Any]]:
        with open(self.config_path, 'r') as f:
            config = json.load(f)

        core_metrics_config = {}

        if 'data_processing' not in config:
            raise ValueError("Config file must contain 'data_processing' section")

        if 'metric_groups' not in config['data_processing']:
            raise ValueError("Config file must contain 'data_processing.metric_groups' section")

        metric_groups = config['data_processing']['metric_groups']

        if 'load_metrics' in metric_groups:
            load_mapping = metric_groups['load_metrics'].get('csv_mapping', {})
            for metric_name, metric_config in load_mapping.items():
                core_metrics_config[metric_name] = {
                    '_measurement': metric_config['_measurement'],
                    'value_column': metric_config['value_column'],
                    'filter': metric_config.get('filter', {})
                }

        if 'traffic_metrics' in metric_groups:
            traffic_mapping = metric_groups['traffic_metrics'].get('csv_mapping', {})
            for metric_name, metric_config in traffic_mapping.items():
                core_metrics_config[metric_name] = {
                    '_measurement': metric_config['_measurement'],
                    'value_column': metric_config['value_column'],
                    'filter': metric_config.get('filter', {})
                }

        return core_metrics_config

    def _convert_to_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df['value'] = pd.to_numeric(df['value'], errors='coerce')

            if df['value'].isna().any():
                logger.warning("Found %d non-numeric values, attempting to clean", df['value'].isna().sum())
                df['value'] = df['value'].astype(str).str.replace(',', '').str.replace('"', '')
                df['value'] = pd.to_numeric(df['value'], errors='coerce')

                if df['value'].isna().any():
                    logger.warning("%d values could not be converted to numeric", df['value'].isna().sum())
                    df['value'] = df['value'].ffill()

            return df
        except (TypeError, ValueError) as e:
            logger.error("Error converting values to numeric: %s", e)
            return df

    def load_metrics(self):
        metric_stats = defaultdict(lambda: {
            'count': 0,
            'min': float('inf'),
            'max': float('-inf')
        })

        logger.info("Loading metrics from CSV: %s", self.metrics_csv_path)

        try:
            # Load only columns needed for extraction.
            required_columns = {"_time", "_measurement", "_field", "_value"}
            for metric_config in self.core_metrics_config.values():
                required_columns.add(metric_config["value_column"])
                for filter_key in metric_config.get("filter", {}).keys():
                    required_columns.add(filter_key)

            df = pd.read_csv(
                self.metrics_csv_path,
                usecols=lambda col: col in required_columns,
                low_memory=False,
            )
            logger.info("Loaded CSV with %d rows and %d columns", len(df), len(df.columns))
        except (OSError, pd.errors.ParserError, ValueError) as e:
            raise FileNotFoundError(f"Error reading CSV file {self.metrics_csv_path}: {e}")

        if '_time' not in df.columns:
            raise ValueError(f"CSV file must contain '_time' column. Found columns: {list(df.columns)}")
        if '_measurement' not in df.columns:
            raise ValueError(
                f"CSV file must contain '_measurement' column (required for metric filtering). "
                f"Found columns: {list(df.columns)}. "
                f"Ensure collect_metrics API returns long-format data with _time, _measurement, _field, _value."
            )
        df['timestamp'] = pd.to_datetime(df['_time'], unit='ms')

        for metric_name, metric_config in self.core_metrics_config.items():
            measurement = metric_config['_measurement']
            value_column = metric_config['value_column']
            filter_dict = metric_config.get('filter', {})

            logger.debug("Processing %s (measurement: %s)", metric_name, measurement)

            metric_df = df[df['_measurement'] == measurement].copy()

            if metric_df.empty:
                logger.warning("No rows found for measurement '%s'", measurement)
                continue

            for filter_key, filter_value in filter_dict.items():
                if filter_key in metric_df.columns:
                    metric_df = metric_df[metric_df[filter_key] == filter_value]
                else:
                    logger.warning("Filter column '%s' not found in CSV", filter_key)

            if metric_df.empty:
                logger.warning("No rows remaining after filters for %s", metric_name)
                continue

            # Extract from a real column or from _field/_value.
            if value_column in metric_df.columns:
                raw_values = metric_df[value_column]
            elif '_field' in metric_df.columns and '_value' in metric_df.columns:
                metric_df = metric_df[metric_df['_field'] == value_column]
                if metric_df.empty:
                    logger.warning("No rows with _field='%s' for %s", value_column, metric_name)
                    continue
                raw_values = metric_df['_value']
            else:
                logger.warning("Value column '%s' not found in CSV", value_column)
                continue

            metric_data = pd.DataFrame({
                'timestamp': metric_df['timestamp'],
                'value': pd.to_numeric(raw_values, errors='coerce')
            })

            metric_data = metric_data.dropna()

            if metric_data.empty:
                logger.warning("No valid values found for %s", metric_name)
                continue

            metric_data = metric_data.groupby('timestamp')['value'].mean().reset_index()
            metric_data = metric_data.set_index('timestamp')
            metric_data = metric_data.sort_index()
            metric_data = self._convert_to_numeric(metric_data)

            self.metrics_data[metric_name] = metric_data

            if not metric_data.empty:
                metric_stats[metric_name]['count'] += len(metric_data)
                metric_stats[metric_name]['min'] = min(metric_stats[metric_name]['min'], metric_data['value'].min())
                metric_stats[metric_name]['max'] = max(metric_stats[metric_name]['max'], metric_data['value'].max())

            logger.info("%s: %d records, range [%.2f, %.2f]",
                metric_name, len(metric_data), metric_data['value'].min(), metric_data['value'].max())

        return self.metrics_data

    def _align_metrics(self, metrics: Dict[str, pd.DataFrame], freq: str = '1s') -> Dict[str, pd.DataFrame]:
        """Align metric DataFrames to a common timeline with specified intervals."""
        aligned_metrics = {}

        for metric_type, df in metrics.items():
            if df.empty:
                logger.debug("Skipping empty dataframe for %s", metric_type)
                aligned_metrics[metric_type] = df
                continue

            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')

            df = df.sort_index()

            resampled_df = df.resample(freq).mean()

            # Keep only real observed seconds (no synthetic grid rows).
            resampled_df = resampled_df.dropna(how="all")
            resampled_df = resampled_df.ffill().bfill()

            aligned_metrics[metric_type] = resampled_df
            logger.debug("Aligned %s to %d records", metric_type, resampled_df.shape[0])

        return aligned_metrics

    def align_metrics(self, freq: str = '1s'):
        logger.info("Aligning metrics to %s intervals...", freq)
        if not self.metrics_data:
            raise ValueError("No metrics data loaded. Please load metrics first.")

        aligned_metrics = self._align_metrics(self.metrics_data, freq=freq)

        self.metrics_data = aligned_metrics

        if 'resampling' not in self.validation_results:
            self.validation_results['resampling'] = {}

        self.validation_results['resampling']['completed'] = True

        if 'alignment' not in self.validation_results:
            self.validation_results['alignment'] = {}
        self.validation_results['alignment']['completed'] = True

        return self.metrics_data

    def normalize_metrics(self):
        """Normalize metrics using z-score normalization."""
        logger.info("Normalizing metrics...")
        if not self.metrics_data:
            raise ValueError("No metrics data loaded.")

        for metric_type, df in self.metrics_data.items():
            if df.empty:
                logger.debug("Skipping empty dataframe for %s", metric_type)
                continue

            mean = df['value'].mean()
            std = df['value'].std()

            if pd.notnull(mean) and pd.notnull(std) and std > 0:
                df['normalized'] = (df['value'] - mean) / std
                self.scaler_stats[metric_type] = {'mean': float(mean), 'std': float(std)}
                logger.debug("%s: mean=%.4f, std=%.4f, normalized range [%.4f, %.4f]",
                    metric_type, mean, std, df['normalized'].min(), df['normalized'].max())
            else:
                reason = "mean" if pd.isnull(mean) else "std"
                logger.warning("Skipping normalization for %s due to invalid %s", metric_type, reason)
                df['normalized'] = df['value']

            self.metrics_data[metric_type] = df

        if 'normalization' not in self.validation_results:
            self.validation_results['normalization'] = {}
        self.validation_results['normalization']['completed'] = True
        return self.metrics_data

    def normalize_with_saved_stats(self, scaler_stats: Dict[str, Dict[str, float]]) -> None:
        """
        Apply precomputed mean/std per metric (e.g. from training) for inference.
        For each metric in self.metrics_data, if present in scaler_stats, apply z-score; else use raw value.
        """
        for metric_type, df in self.metrics_data.items():
            if df.empty:
                continue
            stats = scaler_stats.get(metric_type)
            if stats is not None and isinstance(stats, dict):
                mean = stats.get('mean')
                std = stats.get('std')
                if pd.notnull(mean) and pd.notnull(std) and std > 0:
                    df['normalized'] = (df['value'] - mean) / std
                else:
                    df['normalized'] = df['value']
            else:
                df['normalized'] = df['value']
            self.metrics_data[metric_type] = df
        if 'normalization' not in self.validation_results:
            self.validation_results['normalization'] = {}
        self.validation_results['normalization']['completed'] = True

    def get_processed_data(self, align_freq: str = '1s', scaler_stats: Optional[Dict[str, Dict[str, float]]] = None) -> pd.DataFrame:
        """
        Runs the full metrics processing pipeline and returns the aligned, normalized DataFrame.
        If scaler_stats is provided (e.g. from a saved model), uses those for normalization instead of recomputing.
        """
        self.load_metrics()
        self.align_metrics(freq=align_freq)
        if scaler_stats:
            self.normalize_with_saved_stats(scaler_stats)
        else:
            self.normalize_metrics()
        metrics_df = pd.concat(
            [df[['normalized']] for df in self.metrics_data.values()],
            axis=1
        )
        metrics_df.columns = list(self.metrics_data.keys())
        return metrics_df

    def validate_metrics(self) -> Dict[str, Any]:
        """Validate data quality: completeness, value ranges, time continuity, outliers."""
        logger.info("Validating metrics data quality...")

        if not self.metrics_data:
            raise ValueError("No metrics data loaded. Please load metrics first.")

        if 'data_quality' not in self.validation_results:
            self.validation_results['data_quality'] = {}

        missing_metrics = set(self.core_metrics.keys()) - set(self.metrics_data.keys())
        if missing_metrics:
            logger.warning("Missing metrics for: %s", ', '.join(missing_metrics))

        self.validation_results['data_quality']['missing_metrics'] = list(missing_metrics)
        self.validation_results['data_quality']['completeness'] = len(missing_metrics) == 0

        for metric_type, df in self.metrics_data.items():
            if df.empty:
                self.validation_results['data_quality'][metric_type] = {
                    'valid': False,
                    'error': 'Empty dataset'
                }
                continue

            metric_validation = {
                'valid': True,
                'point_count': len(df),
                'nan_count': 0,
                'time_continuity': True,
                'value_range': {
                    'min': float(df['value'].min()),
                    'max': float(df['value'].max()),
                    'mean': float(df['value'].mean()),
                    'std': float(df['value'].std())
                }
            }

            nan_count = df['value'].isna().sum()
            metric_validation['nan_count'] = int(nan_count)
            if nan_count > 0:
                logger.warning("Found %d NaN values in %s", nan_count, metric_type)
                metric_validation['valid'] = False

            if hasattr(df.index, 'is_monotonic_increasing'):
                if not df.index.is_monotonic_increasing:
                    logger.warning("Timeline is not monotonically increasing for %s", metric_type)
                    metric_validation['time_continuity'] = False
                    metric_validation['valid'] = False

            if 'alignment' in self.validation_results and self.validation_results['alignment'].get('completed', False):
                if hasattr(df.index, 'to_series'):
                    time_diffs = df.index.to_series().diff().dropna()
                    one_second = pd.Timedelta(seconds=1)
                    gaps = time_diffs[time_diffs > one_second]

                    if not gaps.empty:
                        logger.warning("Found %d time gaps in %s", len(gaps), metric_type)
                        metric_validation['time_gaps'] = {
                            'count': int(len(gaps)),
                            'max_gap': float(gaps.max().total_seconds()),
                            'total_gap_time': float(gaps.sum().total_seconds())
                        }
                        metric_validation['valid'] = False

            # Detect outliers (values more than 3 standard deviations from mean)
            if metric_validation['value_range']['std'] > 0:
                z_scores = np.abs((df['value'] - metric_validation['value_range']['mean']) /
                                 metric_validation['value_range']['std'])
                outliers = z_scores > 3
                outlier_count = outliers.sum()

                if outlier_count > 0:
                    logger.debug("Found %d outliers in %s (>3 std dev)", outlier_count, metric_type)
                    metric_validation['outlier_count'] = int(outlier_count)
                    metric_validation['outlier_percentage'] = float(outlier_count / len(df) * 100)

            metric_validation['value_range']['skewness'] = float(df['value'].skew())
            metric_validation['value_range']['kurtosis'] = float(df['value'].kurtosis())

            self.validation_results['data_quality'][metric_type] = metric_validation

        self.validation_results['data_quality']['valid'] = all(
            self.validation_results['data_quality'].get(m, {}).get('valid', False)
            for m in self.metrics_data.keys()
        )

        return self.validation_results

    def create_sequences(self, window_size: int = 10, stride: int = 1, align_freq: str = '1s'):
        """
        Create sliding windows (sequences) from the aligned, normalized metrics DataFrame.
        Each window contains window_size consecutive 1s-timestep rows, with stride between windows.
        Returns a numpy array of shape (n_windows, window_size, n_features).
        """
        metrics_df = self.get_processed_data(align_freq=align_freq)
        data = metrics_df.values
        seqs = []
        for i in range(0, len(data) - window_size + 1, stride):
            seqs.append(data[i:i+window_size])
        return np.array(seqs)
