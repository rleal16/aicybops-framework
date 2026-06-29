import json
import logging
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

logger = logging.getLogger(__name__)

def get_dataloader_kwargs() -> dict:
    """
    Shared DataLoader kwargs for DAM training/inference.

    Controlled by env var `AICYBOPS_DATALOADER_NUM_WORKERS` to optionally enable
    parallel batch preparation via DataLoader worker processes.
    """
    raw = os.getenv("AICYBOPS_DATALOADER_NUM_WORKERS", "0").strip()
    try:
        num_workers = int(raw)
    except ValueError:
        num_workers = 0

    if num_workers <= 0:
        return {}

    # Keep defaults conservative to avoid oversubscribing CPU.
    return {
        "num_workers": num_workers,
        "prefetch_factor": 2,
        "persistent_workers": True,
    }


class DataCleaner:
    """Cleans data by handling NaN and infinite values."""

    @staticmethod
    def check_nan_values(sequences: dict, targets: dict):
        nan_counts = {
            'load_windows': np.isnan(sequences['load']).sum(),
            'traffic_windows': np.isnan(sequences['traffic']).sum(),
            'log_windows': np.isnan(sequences['log']).sum(),
            'target_load': np.isnan(targets['load']).sum(),
            'target_traffic': np.isnan(targets['traffic']).sum(),
            'target_log': np.isnan(targets['log']).sum()
        }

        for key, count in nan_counts.items():
            logger.debug("%s NaN count: %d", key, count)

        return nan_counts

    @staticmethod
    def clean_data(sequences: dict, targets: dict):
        cleaned_sequences = {
            'load': np.nan_to_num(sequences['load'], nan=0.0, posinf=0.0, neginf=0.0),
            'traffic': np.nan_to_num(sequences['traffic'], nan=0.0, posinf=0.0, neginf=0.0),
            'log': np.nan_to_num(sequences['log'], nan=0.0, posinf=0.0, neginf=0.0)
        }

        cleaned_targets = {
            'load': np.nan_to_num(targets['load'], nan=0.0, posinf=0.0, neginf=0.0),
            'traffic': np.nan_to_num(targets['traffic'], nan=0.0, posinf=0.0, neginf=0.0),
            'log': np.nan_to_num(targets['log'], nan=0.0, posinf=0.0, neginf=0.0)
        }

        logger.debug("Cleaned NaN/inf. Remaining inf: load=%d, traffic=%d, log=%d",
            np.isinf(cleaned_sequences['load']).sum(),
            np.isinf(cleaned_sequences['traffic']).sum(),
            np.isinf(cleaned_sequences['log']).sum(),
        )

        return cleaned_sequences, cleaned_targets

    @staticmethod
    def convert_to_tensors(sequences: dict, targets: dict):
        sequence_tensors = {
            'load': torch.tensor(sequences['load'], dtype=torch.float32),
            'traffic': torch.tensor(sequences['traffic'], dtype=torch.float32),
            'log': torch.tensor(sequences['log'], dtype=torch.float32)
        }

        target_tensors = {
            'load': torch.tensor(targets['load'], dtype=torch.float32),
            'traffic': torch.tensor(targets['traffic'], dtype=torch.float32),
            'log': torch.tensor(targets['log'], dtype=torch.float32)
        }

        logger.debug("Tensor NaN: load=%d, traffic=%d, log=%d",
            torch.isnan(sequence_tensors['load']).sum().item(),
            torch.isnan(sequence_tensors['traffic']).sum().item(),
            torch.isnan(sequence_tensors['log']).sum().item(),
        )

        return sequence_tensors, target_tensors

    @staticmethod
    def clean_and_convert_dict_to_tensors(data_dict: dict, data_name: str = "data"):
        nan_count = sum(np.isnan(data_dict[key]).sum() for key in data_dict.keys())
        if nan_count > 0:
            logger.debug("%s NaN count: %d", data_name, nan_count)

        cleaned_dict = {
            key: np.nan_to_num(data_dict[key], nan=0.0, posinf=0.0, neginf=0.0)
            for key in data_dict.keys()
        }

        tensor_dict = {
            key: torch.tensor(cleaned_dict[key], dtype=torch.float32)
            for key in cleaned_dict.keys()
        }

        nan_in_tensors = sum(torch.isnan(tensor_dict[key]).sum().item() for key in tensor_dict.keys())
        if nan_in_tensors > 0:
            logger.warning("%d NaNs found in %s tensors after conversion", nan_in_tensors, data_name)

        return tensor_dict


class DatasetBuilder:
    """Creates PyTorch datasets and data loaders for training."""

    @staticmethod
    def create_dataset(sequence_tensors: dict, target_tensors: dict = None, max_samples: int = None):
        keys = sorted(sequence_tensors.keys())

        if target_tensors is not None:
            if set(keys) != set(target_tensors.keys()):
                raise ValueError("Sequence and target tensors must have the same keys")
            # Combine tensors: sequences first, then targets, maintaining sorted key order
            tensors = [sequence_tensors[key] for key in keys] + [target_tensors[key] for key in keys]
        else:
            tensors = [sequence_tensors[key] for key in keys]

        dataset = TensorDataset(*tensors)

        if max_samples and len(dataset) > max_samples:
            dataset = TensorDataset(*[t[:max_samples] for t in dataset.tensors])
            logger.debug("Limited to %d samples", max_samples)

        return dataset

    @staticmethod
    def split_train_for_validation(train_dataset: TensorDataset, val_ratio: float = 0.2):
        """Split training dataset into train/validation. val_ratio is the fraction taken from train."""
        val_size = int(val_ratio * len(train_dataset))
        train_size = len(train_dataset) - val_size
        train_subset, val_dataset = random_split(train_dataset, [train_size, val_size])

        logger.debug("Train size: %d, Val size: %d", train_size, val_size)
        return train_subset, val_dataset

    @staticmethod
    def create_data_loaders(train_dataset, val_dataset, batch_size: int = 32):
        dl_kwargs = get_dataloader_kwargs()
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **dl_kwargs)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **dl_kwargs)

        logger.debug("Created data loaders, batch size: %d", batch_size)
        return train_loader, val_loader


class GroupExtractor:
    """Extracts metric and log groups from config."""

    @staticmethod
    def extract_groups_from_config(config_path: str, metrics_df: pd.DataFrame, log_df: pd.DataFrame, use_pydantic: bool = True) -> dict:
        """Extract metric groups from config and return group DataFrames keyed by group_name."""
        if use_pydantic:
            try:
                from ..config import DAMDataConfig
                config_obj = DAMDataConfig.from_json(config_path)

                # Convert Pydantic models to the existing dict format.
                metric_groups = {
                    key: {
                        'group_name': group.group_name,
                        'csv_mapping': {
                            metric_name: {
                                '_measurement': metric.measurement,
                                'value_column': metric.value_column,
                                'filter': metric.filter
                            }
                            for metric_name, metric in group.csv_mapping.items()
                        },
                        'description': group.description
                    }
                    for key, group in config_obj.metric_groups.items()
                }

                log_groups = {
                    key: {
                        'group_name': group.group_name,
                        'columns': group.columns,
                        'description': group.description
                    }
                    for key, group in config_obj.log_groups.items()
                }
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load Pydantic config for group extraction from '{config_path}': {type(e).__name__}: {e}"
                ) from e
        else:
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Support both nested and top-level config layouts.
            if 'data_processing' in config:
                metric_groups = config['data_processing'].get('metric_groups', {})
                log_groups = config['data_processing'].get('log_groups', {})
            else:
                metric_groups = config.get('metric_groups', {})
                log_groups = config.get('log_groups', {})

        groups = {}
        GroupExtractor._extract_metric_groups(metric_groups, metrics_df, groups)
        GroupExtractor._extract_log_groups(log_groups, log_df, groups)

        return groups

    @staticmethod
    def _extract_metric_groups(metric_groups: dict, metrics_df: pd.DataFrame, groups: dict):
        for config_key, group_config in metric_groups.items():
            csv_mapping = group_config.get('csv_mapping')

            if csv_mapping is None:
                continue

            if 'group_name' not in group_config:
                raise ValueError(f"Missing 'group_name' for metric group '{config_key}'. "
                               f"Please add 'group_name' field to the group configuration.")
            group_name = group_config['group_name']

            processed_columns = list(csv_mapping.keys())

            if not processed_columns:
                continue

            available_cols = [col for col in processed_columns if col in metrics_df.columns]

            if available_cols:
                groups[group_name] = metrics_df[available_cols]
                if len(available_cols) < len(processed_columns):
                    missing_cols = set(processed_columns) - set(available_cols)
                    logger.warning("%s (group_name: %s) missing columns: %s", config_key, group_name, missing_cols)

    @staticmethod
    def _extract_log_groups(log_groups: dict, log_df: pd.DataFrame, groups: dict):
        for config_key, group_config in log_groups.items():
            columns = group_config.get('columns')

            if columns is None or group_config.get('csv_mapping') is not None:
                continue

            if 'group_name' not in group_config:
                raise ValueError(f"Missing 'group_name' for log group '{config_key}'. "
                               f"Please add 'group_name' field to the group configuration.")
            group_name = group_config['group_name']

            if not columns:
                continue

            available_cols = [col for col in columns if col in log_df.columns]

            if available_cols:
                groups[group_name] = log_df[available_cols]
                if len(available_cols) < len(columns):
                    missing_cols = set(columns) - set(available_cols)
                    logger.warning("%s (group_name: %s) missing columns: %s", config_key, group_name, missing_cols)
