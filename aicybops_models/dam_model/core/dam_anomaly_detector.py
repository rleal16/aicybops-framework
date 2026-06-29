import logging
import os
import sys
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

try:
    import aicybops_lib.torch_bootstrap  # noqa: F401
except ImportError:
    pass

import torch
import torch.nn as nn
import torch.optim as optim
from aicybops_lib.base_model import BaseModel, with_mlflow_logging
import numpy as np
from torch.utils.data import DataLoader

# Add project paths for local imports.
base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(base_dir / "processing"))

from processing.data_analysis import DAMDataProcessor
from processing.dataset.dataset_utils import get_dataloader_kwargs
from core.dam import DAMModel
from utils.config import DAMUnifiedConfig, INIT_QUANTILE

logger = logging.getLogger(__name__)


def _resolve_device() -> str:
    """Resolve the computation device from the DAM_DEVICE environment variable.

    Returns 'cpu' when DAM_DEVICE is unset (default, backward-compatible).
    Returns the requested value when DAM_DEVICE is set and CUDA is available.
    Falls back to 'cpu' with a warning when DAM_DEVICE requests CUDA but the
    runtime does not expose it (e.g. CPU-only container).
    """
    requested = os.getenv("DAM_DEVICE", "cpu").strip().lower()
    if requested.startswith("cuda"):
        if torch.cuda.is_available():
            return requested
        logger.warning(
            "DAM_DEVICE=%r requested but torch.cuda.is_available() is False; falling back to cpu",
            requested,
        )
        return "cpu"
    try:
        torch.device(requested)
    except Exception:
        logger.warning("DAM_DEVICE=%r is invalid; falling back to cpu", requested)
        return "cpu"
    return requested


class DAMAnomalyDetector(BaseModel):
    
    def __init__(self, 
                 experiment_name: str,
                 config_path: str = None,
                 tracking_uri: str = None,
                 **kwargs):
        """
        Initialize DAMAnomalyDetector.
        
        Args:
            experiment_name: Name of the MLflow experiment
            config_path: Path to JSON config file (optional; defaults to DAM_CONFIG_PATH env or package-relative configs/dam_config.json so it works regardless of cwd)
            tracking_uri: MLflow tracking URI (optional)
            **kwargs: Passed to BaseModel (e.g. registered_model_name for Model Registry)
        """
        tracking_uri = tracking_uri or os.getenv('MLFLOW_TRACKING_URI')
        super().__init__(experiment_name, tracking_uri, **kwargs)
        
        # Use env/default config path when not provided.
        if config_path is None:
            default_config = base_dir / "configs" / "dam_config.json"
            config_path = os.getenv('DAM_CONFIG_PATH', str(default_config))
        
        self._config_path = config_path

        self._unified_config = self._load_unified_config(config_path)
        
        arch = self._unified_config.get_model_architecture()
        self.window_size = arch.get('window_size')
        self.stride = arch.get('stride')
        self.align_freq = arch.get('align_freq')
        self.lstm_hidden_dim = arch.get('lstm_hidden_dim')

        self.model = None
        self.pipeline = None
        self.dimensions = None
        
        # BaseModel compatibility state.
        self._last_data = None  # Last training data dict.
        self._last_processor = None  # Last processor (for labels).
        
        self._cached_train_loader = None
        self._cached_val_loader = None
        self._cached_test_dataset = None
        self._cached_dimensions = None
        self._cached_processor = None  # Cached processor for labels.
        self._cached_api_data_paths = None  # Cached API-fetched file paths.
        self._last_cached_window_size = None  # Window size for cached loaders.
        self._cached_data_collection_metrics_count = None
        self._cached_data_collection_logs_count = None
        # Training scaler stats reused at inference.
        self._scaler_stats = None

    def _resolve_data_source(self, data: dict, config_path: str) -> tuple:
        """
        Resolve data source (API vs file-based) and return paths.
        
        Args:
            data: Data dictionary
            config_path: Path to config file
        
        Returns:
            Tuple of (metrics_csv_path, log_file_path, use_api)
        """
        use_api = data.get('use_api', False)

        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Select API or file source.
        if use_api:
            # DataCollector sets these paths.
            metrics_csv_path = None
            log_file_path = None

            api_url = os.getenv('API_URL')
            if not api_url:
                raise ValueError(
                    "API_URL environment variable is required when use_api=True. "
                    "Set it with: export API_URL=http://api-url"
                )
        else:
            if 'data_paths' not in config:
                raise ValueError(
                    "Config file must contain 'data_paths' section with 'metrics_csv' and 'log_file'."
                )
            
            data_paths = config['data_paths']
            if 'metrics_csv' not in data_paths or 'log_file' not in data_paths:
                raise ValueError(
                    "Config file 'data_paths' must contain both 'metrics_csv' and 'log_file'."
                )
            
            # Resolve paths relative to the config directory.
            config_dir = config_file.parent
            metrics_csv_path = str((config_dir / data_paths['metrics_csv']).resolve())
            log_file_path = str((config_dir / data_paths['log_file']).resolve())

            if not Path(metrics_csv_path).exists():
                raise FileNotFoundError(f"Metrics CSV file not found: {metrics_csv_path}")
            if not Path(log_file_path).exists():
                raise FileNotFoundError(f"Log file not found: {log_file_path}")
        
        return metrics_csv_path, log_file_path, use_api
    
    def _load_unified_config(self, config_path: str) -> DAMUnifiedConfig:
        """
        Load unified configuration from JSON file.
        
        Args:
            config_path: Path to JSON config file
        
        Returns:
            DAMUnifiedConfig instance
        """
        return DAMUnifiedConfig(config_path=config_path)
    
    def _extract_data_params(self, data: dict) -> dict:
        """
        Extract data processing parameters from data dictionary, with config as fallback.
        
        This method implements the priority order:
        1. Data dict values (highest priority - runtime overrides)
        2. Config file values (fallback - persistent defaults)
        3. Pydantic model defaults (if field not in config)
        
        Args:
            data: Data dictionary containing runtime overrides and runtime-only parameters
        
        Returns:
            Dictionary with extracted parameters
            
        Note:
            - Config file provides persistent, version-controlled defaults
            - Data dict allows runtime overrides for flexibility (e.g., command-line args)
            - Runtime-only parameters (max_train_samples, use_api, etc.) only exist in data dict
        """
        arch = self._unified_config.get_model_architecture()
        training = self._unified_config.get_training()
        
        return {
            # Model params
            'window_size': data.get('window_size', arch.get('window_size')),
            'stride': data.get('stride', arch.get('stride')),
            'align_freq': data.get('align_freq', arch.get('align_freq')),
            
            # Training params
            'batch_size': data.get('batch_size', training.get('batch_size')),
            'train_ratio': data.get('train_ratio', training.get('train_ratio')),
            'val_ratio': data.get('val_ratio', training.get('val_ratio')),
            'training_mode': data.get('training_mode', training.get('training_mode', 'supervised')),
            
            # Runtime-only params
            'max_train_samples': data.get('max_train_samples', None),
            'max_test_samples': data.get('max_test_samples', None),
            'random_state': data.get('random_state', None),
            'include_targets': data.get('include_targets', False),
            'force_caching': data.get('force_caching', False),
            'use_session_time_range': data.get('use_session_time_range', True),
            'start': data.get('start', None),  # e.g. "-600s"
            'training_window_minutes': data.get('training_window_minutes', 0),
        }
    
    def _prepare_data(self, mode='training', data=None, **kwargs):
        """
        Internal method to prepare data using data dictionary.
        
        Uses both config file (for defaults) and data dict (for overrides):
        - Config file: Provides data paths, model architecture defaults, training defaults
        - Data dict: Allows runtime overrides (window_size, batch_size, etc.) and runtime-only params
        
        Args:
            mode: Mode of operation ('training', 'validation', 'test', 'prediction')
            data: Data dictionary (required) containing:
                - Optional overrides: window_size, batch_size, train_ratio, etc.
                - Runtime-only params: max_train_samples, use_api, force_caching, etc.
                - config_path: Optional (redundant if provided in constructor)
            **kwargs: Additional parameters
        
        Returns:
            Tuple of prepared data based on mode:
            - 'training' → (train_loader, val_loader, test_dataset, dimensions)
            - 'validation' → (val_loader, dimensions)
            - 'test' → (test_dataset, dimensions)
            - 'prediction' → (prediction_loader, dimensions)
        """
        if data is None:
            raise ValueError("data parameter is required. Provide a data dictionary.")

        config_path = None
        if self._unified_config and self._unified_config.config_path:
            config_path = str(self._unified_config.config_path)
        else:
            config_path = data.get('config_path')
            if not config_path:
                raise ValueError(
                    "'config_path' is required. Either provide it in constructor or in data dictionary."
                )
        
        # Warn if constructor/data config paths differ.
        data_config_path = data.get('config_path')
        if data_config_path and config_path != data_config_path:
            logger.warning(
                f"Config path mismatch: constructor loaded '{config_path}', "
                f"but data dict provides '{data_config_path}'. Using config from constructor."
            )

        if 'metrics_csv_path' in data or 'log_file_path' in data:
            raise ValueError(
                "metrics_csv_path and log_file_path should NOT be provided in data dictionary. "
                "They must be defined in the config file under 'data_paths'."
            )
        
        # Resolve data source.
        metrics_csv_path, log_file_path, use_api = self._resolve_data_source(data, config_path)
        
        # Extract data params.
        params = self._extract_data_params(data)
        
        # Reuse API-fetched files when available.
        labels_csv_path = None
        if use_api and self._cached_api_data_paths is not None:
            metrics_csv_path = self._cached_api_data_paths.get('metrics_csv_path')
            log_file_path = self._cached_api_data_paths.get('log_file_path')
            labels_csv_path = self._cached_api_data_paths.get('labels_csv_path')
            use_api = False

        # Build data processor.
        processor = DAMDataProcessor(
            metrics_csv_path=metrics_csv_path,
            log_file_path=log_file_path,
            config_path=config_path,
            window_size=params['window_size'],
            stride=params['stride'],
            align_freq=params['align_freq'],
            use_api=use_api,
            labels_csv_path=labels_csv_path,
            scaler_stats=self._scaler_stats,
            use_session_time_range=params['use_session_time_range'],
            start=params.get('start'),
            training_window_minutes=params.get('training_window_minutes', 0),
        )

        if mode == 'prediction':
            result = processor.prepare_for_prediction(
                batch_size=params['batch_size'],
                include_targets=params['include_targets']
            )
            self._last_processor = processor
            return result  # (prediction_loader, dimensions)
        
        # For non-prediction modes, cache train/val/test splits.
        self._cache_data(
            processor,
            params['train_ratio'],
            params['val_ratio'],
            params['batch_size'],
            params['max_train_samples'],
            params['max_test_samples'],
            params['random_state'],
            params['training_mode'],
            params['force_caching']
        )
        
        if mode == 'training':
            return (self._cached_train_loader, self._cached_val_loader, 
                    self._cached_test_dataset, self._cached_dimensions)
        elif mode == 'validation':
            return (self._cached_val_loader, self._cached_dimensions)
        elif mode == 'test':
            return (self._cached_test_dataset, self._cached_dimensions)
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be one of: 'training', 'validation', 'test', 'prediction'")
    
    def _cache_data(self, processor, train_ratio, val_ratio, batch_size, max_train_samples, max_test_samples, random_state, training_mode='supervised', force_caching=False) -> None:

        if force_caching:
            self.clear_data_cache()
        current_window_size = getattr(processor, "window_size", None)
        if (
            self._last_cached_window_size is not None
            and current_window_size is not None
            and current_window_size != self._last_cached_window_size
            and not force_caching
        ):
            self._cached_train_loader = None
            self._cached_val_loader = None
            self._cached_test_dataset = None
            self._cached_dimensions = None
            self._cached_processor = None
        if (self._cached_train_loader is None or
            self._cached_val_loader is None or
            self._cached_test_dataset is None):
            train_loader, val_loader, test_dataset, dimensions = processor.prepare_for_training(
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                batch_size=batch_size,
                max_train_samples=max_train_samples,
                max_test_samples=max_test_samples,
                random_state=random_state,
                training_mode=training_mode
            )
            self._cached_train_loader = train_loader
            self._cached_val_loader = val_loader
            self._cached_test_dataset = test_dataset
            self._cached_dimensions = dimensions
            self._cached_processor = processor
            self._last_cached_window_size = current_window_size
            if getattr(processor, "data_collection_metrics_count", None) is not None:
                self._cached_data_collection_metrics_count = processor.data_collection_metrics_count
            if getattr(processor, "data_collection_logs_count", None) is not None:
                self._cached_data_collection_logs_count = processor.data_collection_logs_count
            if (
                self._cached_api_data_paths is None
                and getattr(processor, "metrics_csv_path", None)
                and getattr(processor, "log_file_path", None)
            ):
                self._cached_api_data_paths = {
                    "metrics_csv_path": processor.metrics_csv_path,
                    "log_file_path": processor.log_file_path,
                    "labels_csv_path": getattr(processor, "labels_csv_path", None),
                }
        
        self._last_processor = self._cached_processor
        

    def clear_data_cache(self):
        """
        Clear cached data splits. Forces re-preparation on next _prepare_data call.
        
        This is useful when you need to:
        - Change data source or configuration
        - Force a new split with different parameters
        - Reset the cache for debugging
        """
        self._cached_train_loader = None
        self._cached_val_loader = None
        self._cached_test_dataset = None
        self._cached_dimensions = None
        self._cached_processor = None
        self._cached_api_data_paths = None
        self._last_cached_window_size = None
        self._cached_data_collection_metrics_count = None
        self._cached_data_collection_logs_count = None
    
    def get_training_data(self, **kwargs):
        """
        Get training data loader.
        
        Args:
            **kwargs: Optional parameters including:
                - data: Data dictionary. If not provided, falls back to self._last_data from train().
        
        Returns:
            Training data loader
        """
        data = kwargs.get('data')
        if data is None:
            if self._last_data is None:
                raise ValueError(
                    "No data available. Provide data via kwargs or call train() first (which sets _last_data)."
                )
            data = self._last_data
        train_loader, _, _, _ = self._prepare_data(mode='training', data=data)
        return train_loader
    
    def get_test_data(self, **kwargs):
        """
        Get test dataset.
        
        Args:
            **kwargs: Optional parameters including:
                - data: Data dictionary. If not provided, falls back to self._last_data from train().
        
        Returns:
            Test dataset
        """
        data = kwargs.get('data')
        if data is None:
            if self._last_data is None:
                raise ValueError(
                    "No data available. Provide data via kwargs or call train() first (which sets _last_data)."
                )
            data = self._last_data
        test_dataset, _ = self._prepare_data(mode='test', data=data)
        return test_dataset
    
    def get_validation_data(self, **kwargs):
        """
        Get validation data loader.
        
        Args:
            **kwargs: Optional parameters including:
                - data: Data dictionary. If not provided, falls back to self._last_data from train().
        
        Returns:
            Validation data loader
        """
        data = kwargs.get('data')
        if data is None:
            if self._last_data is None:
                raise ValueError(
                    "No data available. Provide data via kwargs or call train() first (which sets _last_data)."
                )
            data = self._last_data
        val_loader, _ = self._prepare_data(mode='validation', data=data)
        return val_loader
    
    def get_prediction_data(self, **kwargs):
        """
        Get prediction data loader.
        
        Args:
            **kwargs: Optional parameters including:
                - data: Data dictionary. If not provided, falls back to self._last_data from train().
        
        Returns:
            Prediction data loader
        """
        data = kwargs.get('data')
        if data is None:
            if self._last_data is None:
                raise ValueError(
                    "No data available for prediction. Provide data via kwargs or call train() first (which sets _last_data)."
                )
            data = self._last_data
        prediction_loader, _ = self._prepare_data(mode='prediction', data=data)
        return prediction_loader
    
    def get_evaluation_data_dict(self, data, dataset_type: str = "test", batch_size: int = 32) -> Dict:
        """
        Get data dictionary for evaluation pipeline with both data loader and labels.
        
        Not part of BaseModel abstract interface, so can have explicit data parameter.
        
        Args:
            data: Data dictionary (required) containing config_path and other parameters
            dataset_type: Type of dataset ("test" or "val")
            batch_size: Batch size for DataLoader (if test dataset needs to be converted)
            
        Returns:
            Dictionary containing:
            - 'evaluation_loader': DataLoader (generic name, regardless of dataset_type)
            - 'evaluation_labels': numpy array of labels (or None, generic name)
        """
        if data is None:
            raise ValueError("data parameter is required.")
        
        # Load dataset by type.
        if dataset_type == "test":
            evaluation_dataset, dimensions = self._prepare_data(mode='test', data=data)
            evaluation_loader = DataLoader(
                evaluation_dataset,
                batch_size=batch_size,
                shuffle=False,
                **get_dataloader_kwargs(),
            )

            evaluation_labels = None
            if self._last_processor is not None:
                labels_dict = self._last_processor.get_labels()
                evaluation_labels = labels_dict.get('test_labels')
        elif dataset_type == "val":
            evaluation_dataset, dimensions = self._prepare_data(mode='validation', data=data)
            evaluation_loader = DataLoader(
                evaluation_dataset,
                batch_size=batch_size,
                shuffle=False,
                **get_dataloader_kwargs(),
            )

            evaluation_labels = None
            if self._last_processor is not None:
                labels_dict = self._last_processor.get_labels()
                evaluation_labels = labels_dict.get('val_labels')
        else:
            raise ValueError(f"dataset_type must be 'test' or 'val', got '{dataset_type}'")
        
        # Align labels with the evaluation dataset length.
        if evaluation_labels is not None and len(evaluation_labels) != len(evaluation_dataset):
            evaluation_labels = evaluation_labels[:len(evaluation_dataset)].copy()
        
        # Return normalized keys for both dataset types.
        data_dict = {
            'evaluation_loader': evaluation_loader,
            'evaluation_labels': evaluation_labels
        }
        
        return data_dict
    
    def build_model(self):
        """
        Build the DAM model using dimensions.
        
        Matches BaseModel signature: build_model() (no parameters)
        Uses self.dimensions that should be set during train(). No-op when dimensions is None
        (train() sets dimensions and calls build_model() when needed).
        """
        if self.dimensions is None:
            return
        self.model = DAMModel(
            load_metrics_dim=self.dimensions['load'],
            traffic_metrics_dim=self.dimensions['traffic'],
            log_seq_dim=self.dimensions['log'],
            lstm_hidden_dim=self.lstm_hidden_dim
        )
    
    @with_mlflow_logging
    def train(self, **kwargs) -> dict:
        """
        Train the DAM model using DAMPipeline.
        
        Args:
            **kwargs: Training parameters including:
                - params: Dictionary with training parameters:
                - learning_rate: Learning rate for optimizer (default 0.0001)
                - spot_type: Type of SPOT detector ("SPOT" or "dSPOT", default "dSPOT")
                - risk_level: Risk level for EVT (default 1e-3)
                - depth: Window size for dSPOT (default 10)
                - epochs: Number of training epochs (required)
                - data: Data dictionary (optional). If not provided, auto-generated from stored config_path
                  and config file defaults. Should contain config_path and other parameters if provided.
                - early_stopping: Optional EarlyStopping instance. If None, created from config file
                  (config file: training.early_stopping section with enabled, patience, min_delta, mode)
        
        Returns:
            Dictionary with training metrics
        """
        params = kwargs.get('params', {})
        epochs = kwargs.get('epochs')
        if epochs is None:
            raise ValueError("epochs parameter required. Pass epochs=<number> in kwargs.")
        data = kwargs.get('data')
        if data is None:
            raise ValueError(
                "data parameter is required. Pass a data dict with at least config_path and use_api (e.g. data={'config_path': ..., 'use_api': True, 'batch_size': 32, ...})."
            )
        train_loader, val_loader, test_dataset, dimensions = self._prepare_data(mode='training', data=data)
        
        self.dimensions = dimensions
        
        self.window_size = data.get('window_size', self.window_size)
        
        # Build model after dimensions are set.
        if self.model is not None:
            if (self.model.load_metrics_dim != dimensions['load'] or
                self.model.traffic_metrics_dim != dimensions['traffic'] or
                self.model.log_seq_dim != dimensions['log']):
                raise ValueError(
                    f"Model dimensions ({self.model.load_metrics_dim}, {self.model.traffic_metrics_dim}, {self.model.log_seq_dim}) "
                    f"do not match data dimensions ({dimensions['load']}, {dimensions['traffic']}, {dimensions['log']}). "
                    "Please rebuild the model or use data with matching dimensions."
                )
        else:
            # Build model from current dimensions.
            self.build_model()
        
        self._last_data = data
        
        # Load training/anomaly defaults from config.
        if self._unified_config:
            training = self._unified_config.get_training()
            anomaly = self._unified_config.get_anomaly_detection()
        else:
            training = {}
            anomaly = {}
        
        # Runtime params override config defaults.
        learning_rate = params.get('learning_rate', training.get('learning_rate'))
        spot_type = params.get('spot_type', anomaly.get('spot_type'))
        risk_level = params.get('risk_level', anomaly.get('risk_level'))
        depth = params.get('depth', anomaly.get('depth'))
        init_quantile = params.get('init_quantile', anomaly.get('init_quantile', INIT_QUANTILE))
        
        # Runtime early_stopping overrides config.
        early_stopping = kwargs.get('early_stopping')
        if early_stopping is None:
            # Build early stopping from config.
            from utils.training import EarlyStopping
            if self._unified_config:
                es_config = training.get('early_stopping', {})
                if es_config.get('enabled', True):
                    early_stopping = EarlyStopping(
                        patience=es_config.get('patience'),
                        min_delta=es_config.get('min_delta'),
                        mode=es_config.get('mode', 'min')
                    )
                else:
                    early_stopping = None
        
        # Optimizer/loss.
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Build training pipeline.
        from pipelines.training import DAMPipeline
        resolved_device = _resolve_device()
        self.pipeline = DAMPipeline(
            model=self.model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=epochs,
            spot_type=spot_type,
            risk_level=risk_level,
            depth=depth,
            init_quantile=init_quantile,
            early_stopping=early_stopping,
            device=resolved_device,
        )
        
        # Train model.
        self.pipeline.train()

        val_loss = self.pipeline.validate()
        train_loss = self.pipeline.get_final_train_loss()
        # Keep scaler stats for inference/save.
        if self._cached_processor is not None and getattr(self._cached_processor, 'metrics_analyser', None) is not None:
            stats = getattr(self._cached_processor.metrics_analyser, 'scaler_stats', None)
            if stats:
                self._scaler_stats = stats
        
        return {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'data_collection_metrics_count': (
                getattr(self._last_processor, 'data_collection_metrics_count', None)
                if self._last_processor is not None else None
            ) or self._cached_data_collection_metrics_count,
            'data_collection_logs_count': (
                getattr(self._last_processor, 'data_collection_logs_count', None)
                if self._last_processor is not None else None
            ) or self._cached_data_collection_logs_count,
        }
    
    @with_mlflow_logging
    def validate(self, **kwargs) -> dict:
        """
        Validate the model using DAMPipeline.

        Requires `data` via kwargs or self._last_data from a prior train() call.
        """
        data = kwargs.get('data')
        if data is None:
            if self._last_data is None:
                raise ValueError(
                    "No data available for validation. Provide data via kwargs or call train() first (which sets _last_data)."
                )
            data = self._last_data

        if self.pipeline is None:
            raise ValueError("Model must be trained first. Call train() before validate().")
        
        val_loader, _ = self._prepare_data(mode='validation', data=data)
        
        val_loss = self.pipeline.validate(data_loader=val_loader)

        # Log dSPOT baseline with this run for later restore.
        self._log_pipeline_state_artifact(self.pipeline)

        return {'validation_loss': val_loss}
    
    @with_mlflow_logging
    def test(self, **kwargs) -> dict:
        """
        Test the model using DAMPipeline.

        Requires `data` via kwargs or self._last_data from a prior train() call.
        """
        data = kwargs.get('data')
        if data is None:
            if self._last_data is None:
                raise ValueError(
                    "No data available for test. Provide data via kwargs or call train() first (which sets _last_data)."
                )
            data = self._last_data

        if self.pipeline is None:
            raise ValueError("Model must be trained first. Call train() before test().")
        
        test_dataset, _ = self._prepare_data(mode='test', data=data)
        
        batch_size = data.get('batch_size', 32)
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            **get_dataloader_kwargs(),
        )

        result = self.pipeline.predict(
            data_loader=test_loader,
            return_anomaly_scores=True,
            plot_anomaly_scores=False,
            stream_mode=False
        )
        
        if isinstance(result, tuple) and len(result) == 3:
            predictions, anomaly_scores, classifications = result
        else:
            raise ValueError(f"Unexpected return format from pipeline.predict(): {type(result)} with length {len(result) if isinstance(result, tuple) else 'N/A'}")
        
        n = len(anomaly_scores)
        if n == 0:
            raise ValueError(
                "Insufficient test data: no test sequences. "
                "Metrics must have enough timestamps to form at least one full window after train/val split. "
                "Check that API metrics have multiple scrapes in the requested range and that train_ratio/val_ratio leave test data."
            )
        # Keep stats well-defined for small n.
        mean_val = float(anomaly_scores.mean())

        if n == 1:
            return {
                'test_anomaly_scores_mean': mean_val,
                'test_anomaly_scores_std': 0.0,
                'test_anomaly_scores_min': mean_val,
                'test_anomaly_scores_max': mean_val,
            }
        return {
            'test_anomaly_scores_mean': mean_val,
            'test_anomaly_scores_std': float(anomaly_scores.std()),
            'test_anomaly_scores_min': float(anomaly_scores.min()),
            'test_anomaly_scores_max': float(anomaly_scores.max()),
        }

    def _resolve_prediction_data(self, data=None, reuse_training_loaders=False):
        """Resolve prediction DataLoader and optional data dict for pipeline building. Returns (prediction_loader, data_for_pipeline).
        When reuse_training_loaders=True (used by predict() for single-load flow), assumes _prepare_data(mode='training') was already
        called so _cached_processor and train/val loaders exist; builds prediction_loader from cached processor (no second API fetch)."""
        if isinstance(data, dict):
            data_with_targets = data.copy()
            if 'include_targets' not in data_with_targets:
                data_with_targets['include_targets'] = True
            if reuse_training_loaders and getattr(self, "_cached_processor", None) is not None:
                params = self._extract_data_params(data_with_targets)
                prediction_loader, _ = self._cached_processor.prepare_for_prediction(
                    batch_size=params["batch_size"],
                    include_targets=params["include_targets"],
                )
                return prediction_loader, data_with_targets
            prediction_loader, _ = self._prepare_data(mode='prediction', data=data_with_targets)
            return prediction_loader, data_with_targets
        if isinstance(data, DataLoader):
            return data, None
        if data is None:
            if hasattr(self, '_last_data') and self._last_data is not None:
                data_with_targets = self._last_data.copy()
                if 'include_targets' not in data_with_targets:
                    data_with_targets['include_targets'] = True
                prediction_loader, _ = self._prepare_data(mode='prediction', data=data_with_targets)
                return prediction_loader, data_with_targets
            if hasattr(self, '_config_path') and self._config_path:
                data_with_targets = {
                    'config_path': self._config_path,
                    'window_size': self.window_size,
                    'stride': self.stride,
                    'align_freq': self.align_freq,
                    'batch_size': 32,
                    'include_targets': True,
                }
                prediction_loader, _ = self._prepare_data(mode='prediction', data=data_with_targets)
                return prediction_loader, data_with_targets
            raise ValueError("No data available for prediction. Provide data parameter or call train() first.")
        raise ValueError(f"Unsupported data type: {type(data)}. Expected dict, DataLoader, or None.")

    def _ensure_pipeline(self, data_for_pipeline=None, train_loader=None, val_loader=None):
        """Build pipeline for prediction when loaded from registry (model set but pipeline None). Raises if pipeline still None.
        When train_loader/val_loader are provided, they are passed to _pipeline_for_prediction to avoid a second data load."""
        if self.pipeline is None and self.model is not None and data_for_pipeline is not None:
            evt = self._unified_config.get_anomaly_detection() if self._unified_config else {}
            config = {
                "spot_type": evt.get("spot_type", "dSPOT"),
                "risk_level": evt.get("risk_level", 1e-3),
                "depth": evt.get("depth", 10),
            }
            self.pipeline = self._pipeline_for_prediction(
                data_for_pipeline, config, _resolve_device(),
                train_loader=train_loader, val_loader=val_loader,
            )
        if self.pipeline is None:
            raise ValueError("Model must be trained first. Call train() before predict().")

    def predict(self, model, model_info=None, data=None, stream_mode: bool = False, return_only_alarms: bool = True):
        """
        Make predictions using the model.
        
        Args:
            model: DAMModel instance (can be None to use self.model) - kept for compatibility but not used
            model_info: Optional model metadata - kept for compatibility but not used
            data: Optional data for prediction. Can be:
                - dict: Data dictionary (will be prepared via _prepare_data with include_targets=True)
                - DataLoader: Use directly (assumes it has targets if needed for anomaly scores)
                - None: Try self._last_data (for compatibility), otherwise raise error
            stream_mode: If True, uses streaming anomaly detection; if False, uses batch mode with static threshold
            return_only_alarms: If True (default), return only the alarms (single value for API). If False, return full tuple.
            
        Returns:
            - If return_only_alarms=True: alarms only (5th element in stream mode; classifications in batch)
            - If return_only_alarms=False and stream_mode=False: (predictions, anomaly_scores, classifications) - 3-tuple
            - If return_only_alarms=False and stream_mode=True: (predictions, anomaly_scores, classifications, thresholds, alarms) - 5-tuple
        """
        # Require restored dSPOT baseline.
        if self.pipeline is None:
            raise RuntimeError(
                "dSPOT baseline not restored for the registered model version. "
                "The MLflow run behind this version is missing the 'pipeline_state/' artifact "
                "(pipeline_state/<scores>.npy + pipeline_state/config.json). "
                "Re-register a version whose run contains the artifact, or pin --model-version to one that does."
            )
        prediction_loader, data_for_pipeline = self._resolve_prediction_data(data, reuse_training_loaders=False)
        if data_for_pipeline is None and isinstance(data, dict):
            data_for_pipeline = data.copy()
            if "include_targets" not in data_for_pipeline:
                data_for_pipeline["include_targets"] = True
        setattr(
            self,
            "_last_prediction_num_sequences",
            len(prediction_loader.dataset) if getattr(prediction_loader, "dataset", None) is not None else None,
        )
        self._ensure_pipeline(data_for_pipeline)

        # Always request anomaly scores/classifications.
        result = self.pipeline.predict(
            data_loader=prediction_loader,
            return_anomaly_scores=True,
            plot_anomaly_scores=False,
            stream_mode=stream_mode
        )
        
        # Batch: (predictions, scores, classes); stream adds (thresholds, alarms).
        if return_only_alarms:
            if stream_mode and isinstance(result, tuple) and len(result) >= 5:
                alarms = result[4]
                return alarms.tolist() if hasattr(alarms, 'tolist') else list(alarms)
            if not stream_mode and isinstance(result, tuple) and len(result) >= 3:
                cls = result[2]
                return cls.tolist() if hasattr(cls, 'tolist') else list(cls)
            return result
        return result
    
    def get_example_input(self):
        """
        Get example input for model signature.
        
        Uses stored dimensions and window_size.
        """
        if self.dimensions is None:
            raise ValueError(
                "Dimensions not set. Call train() first or set self.dimensions manually."
            )
        
        if self.window_size is None:
            raise ValueError(
                "window_size not set. Pass window_size to constructor or set self.window_size manually."
            )
        
        return (
            torch.randn(1, self.window_size, self.dimensions['load']),
            torch.randn(1, self.window_size, self.dimensions['traffic']),
            torch.randn(1, self.window_size, self.dimensions['log'])
        )
    
    @with_mlflow_logging
    def evaluate(self, data=None,
                 config: Dict = None,
                 output_dir: Optional[str] = None,
                 dataset_type: str = "test",
                 device: Optional[str] = None,
                 require_labels: bool = True) -> dict:
        """
        Evaluate the model using DAMEvaluationPipeline with stream mode.
        Supports evaluation when model was loaded from registry (self.pipeline is None)
        by building a pipeline for prediction from config and fitting the baseline.
        
        Args:
            data: Data dictionary (optional). When None, uses self._last_data or builds from _config_path.
            config: Evaluation configuration dictionary (required). Must include
                    evt_parameters and max_memory_gb.
            output_dir: Optional output directory for results
            dataset_type: Dataset type for evaluation ("test" or "val", default "test")
            device: Device to run evaluation on. If omitted, uses DAM_DEVICE
                    with safe fallback to "cpu".
            
        Returns:
            Dictionary with evaluation metrics including thresholds array and alarms
        """
        if device is None:
            device = _resolve_device()
        if not config:
            raise ValueError("config is required and must be a non-empty dictionary.")
        if self.model is None:
            raise ValueError("Model must be trained first or load from registry before evaluate().")
        # Backfill dimensions when loading from registry.
        if self.dimensions is None and hasattr(self.model, "get_dimensions"):
            dims = self.model.get_dimensions()
            self.dimensions = {
                "load": dims["load_metrics_dim"],
                "traffic": dims["traffic_metrics_dim"],
                "log": dims["log_seq_dim"],
            }
        if data is None:
            data = getattr(self, "_last_data", None) or {
                "config_path": self._config_path,
                "window_size": self.window_size,
                "stride": self.stride,
                "align_freq": self.align_freq,
                "batch_size": 32,
                "train_ratio": 0.8,
                "val_ratio": 0.2,
                "random_state": 42,
            }
        pipeline = self.pipeline
        if pipeline is None or not hasattr(pipeline, "anomaly_detector"):
            pipeline = self._pipeline_for_prediction(data, config, device)
        data_dict = self.get_evaluation_data_dict(data=data, dataset_type=dataset_type)
        from pipelines.evaluation import DAMEvaluationPipeline
        evaluator = DAMEvaluationPipeline(
            model=self.model,
            config=config,
            output_dir=output_dir,
            device=device,
            pipeline=pipeline
        )
        metrics = evaluator.run_evaluation(data_dict=data_dict, output_dir=output_dir, require_labels=require_labels)
        # Log dSPOT baseline for prediction restore.
        self._log_pipeline_state_artifact(pipeline)
        return metrics

    def _pipeline_for_prediction(
        self,
        data: Dict,
        config: Dict,
        device: str,
        train_loader=None,
        val_loader=None,
    ):
        """Return a DAMPipeline ready for prediction (no training), for use in evaluate() or predict().
        When train_loader/val_loader are provided (e.g. from a previous _prepare_data in the same request),
        they are reused to avoid a second data load."""
        from pipelines.training import DAMPipeline
        if train_loader is None or val_loader is None:
            train_loader, val_loader, _, _ = self._prepare_data(mode="training", data=data)
        anomaly = self._unified_config.get_anomaly_detection() if self._unified_config else {}
        evt_params = config.get("evt_parameters", {})
        spot_type = evt_params.get("spot_type") or config.get("spot_type") or anomaly.get("spot_type", "dSPOT")
        risk_level = float(evt_params.get("risk_level") or config.get("risk_level") or anomaly.get("risk_level", 1e-3))
        depth = int(evt_params.get("depth") or config.get("depth") or anomaly.get("depth", 10))
        init_quantile = float(evt_params.get("init_quantile") or config.get("init_quantile") or anomaly.get("init_quantile", 0.95))
        device_obj = torch.device(device) if isinstance(device, str) else device
        return DAMPipeline.for_prediction(
            self.model, train_loader, val_loader,
            spot_type=spot_type, risk_level=risk_level, depth=depth,
            init_quantile=init_quantile, device=device_obj,
        )
    
    def get_model_metrics(self) -> dict:
        """Return metrics configuration for the model."""
        return {
            'prediction': {'metric': 'validation_loss', 'mode': 'min'},
            'training': ['train_loss', 'val_loss'],
            'evaluation': ['test_anomaly_scores_mean', 'test_anomaly_scores_std', 
                          'test_anomaly_scores_min', 'test_anomaly_scores_max']
        }

    def _try_restore_pipeline_from_artifact(self, model_name: str, model_version: str) -> bool:
        """Download train_anomaly_scores from the MLflow run that produced this model version
        and restore dSPOT baseline without fetching any training data from the API.

        Returns True if restoration succeeded, False otherwise (caller falls back to re-fit).
        """
        try:
            from mlflow.tracking import MlflowClient
            import json, glob, tempfile, os
            from pipelines.training import DAMPipeline
            import torch.nn as nn
            import torch.optim as optim

            client = MlflowClient()

            # Resolve run_id for target model version.
            if model_version == "latest":
                versions = client.search_model_versions(
                    filter_string=f"name='{model_name}'",
                    order_by=["version_number DESC"],
                    max_results=1,
                )
                if not versions:
                    return False
                run_id = versions[0].run_id
            else:
                mv = client.get_model_version(model_name, str(model_version))
                run_id = mv.run_id

            # Download pipeline_state artifacts.
            with tempfile.TemporaryDirectory() as tmp_dir:
                try:
                    artifact_dir = client.download_artifacts(run_id, "pipeline_state", tmp_dir)
                except Exception:
                    return False  # Artifact missing.

                config_path = os.path.join(artifact_dir, "config.json")
                if not os.path.exists(config_path):
                    return False
                with open(config_path) as f:
                    config = json.load(f)
                if not config.get("baseline_fitted"):
                    return False

                npy_files = glob.glob(os.path.join(artifact_dir, "*.npy"))
                if not npy_files:
                    return False
                train_scores = np.load(npy_files[0])

            spot_type = config.get("spot_type", "dSPOT")
            risk_level = float(config.get("risk_level", 1e-3))
            depth = int(config.get("depth", 10))
            init_quantile = float(config.get("init_quantile", 0.95))

            optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
            criterion = nn.MSELoss()
            resolved_device = _resolve_device()
            self.pipeline = DAMPipeline(
                model=self.model,
                train_loader=None,
                val_loader=None,
                optimizer=optimizer,
                criterion=criterion,
                num_epochs=0,
                spot_type=spot_type,
                risk_level=risk_level,
                depth=depth,
                device=resolved_device,
            )
            self.pipeline.train_anomaly_scores = train_scores
            self.pipeline.anomaly_detector.fit_baseline(
                baseline_scores=train_scores,
                stream_scores=np.array([]),
                init_quantile=init_quantile,
            )
            logger.info(
                "Restored dSPOT pipeline from MLflow artifact (model=%s version=%s run_id=%s).",
                model_name, model_version, run_id,
            )
            return True
        except Exception as e:
            logger.warning(
                "Could not restore dSPOT from MLflow artifact: %s. Will re-fit on training data.", e
            )
            return False

    def get_prediction(self, run_name=None, nested_run_name=None, metric_name=None, mode=None,
                      registered_model_name=None, model_version=None, data=None):
        """Override so API path gets only alarms via stream mode. Uses _get_model when registry params given, else _get_best_model.
        data: optional dict passed to predict() (e.g. {'use_api': True} for live API data).
        """
        pred_metrics = self.get_model_metrics()['prediction']
        metric_name = metric_name or pred_metrics['metric']
        mode = mode or pred_metrics['mode']
        if hasattr(self, '_get_model'):
            model, run_info = self._get_model(registered_model_name, model_version or "latest")
        else:
            model, run_info = self._get_best_model(run_name, nested_run_name, metric_name, mode)
        if model is None:
            if not hasattr(self, 'model') or self.model is None:
                raise ValueError("No model available for prediction")
            model = self.model
        else:
            self.model = model

        _name = registered_model_name or getattr(self, "registered_model_name", None) or "dam"
        self._try_restore_pipeline_from_artifact(_name, model_version or "latest")

        result = self.predict(model, run_info, data=data, stream_mode=True, return_only_alarms=True)
        num_seq = getattr(self, "_last_prediction_num_sequences", None)
        diag = {"num_sequences": num_seq}
        if isinstance(result, list):
            diag["num_alarms"] = len(result)
        return {"predictions": result, "prediction_diagnostics": diag}

    def optimize(self, param_space: dict, max_evals: int = 50, epochs: int = 10,
                 objective: str = 'val_loss', method: str = 'random', **kwargs) -> dict:
        """
        Optimize hyperparameters for DAM model.
        Accepts param_space as either:
        - List of param dicts (service/BaseModel style): run one trial per dict, pick best.
        - Dict of hyperparameter ranges: use OptimizationPipeline (random/grid over combinations).

        Returns:
            Dict with best_params, best_loss, and optionally registered_model_name, model_version.
        """
        from pipelines.optimization import objectives as obj

        # Expand a single config into a search space.
        if isinstance(param_space, list) and len(param_space) == 1:
            base_params = param_space[0]
            param_space = {
                'learning_rate':    [1e-5, 3e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3],
                'lstm_hidden_dim':  [16, 24, 32, 48, 64, 96, 128],
                'window_size':      [5, 8, 10, 12, 15, 20, 30],
                'risk_level':       [round(i * 1e-3, 3) for i in range(1, 91)],
                'spot_type':        ['SPOT', 'dSPOT'],
                'depth':            [3, 5, 7, 10, 12, 15, 20],
                'init_quantile':    [0.75, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98],
            }
            kwargs["base_params"] = base_params
            _DEFAULT_MAX_EVALS = 20
            if max_evals <= 1:
                logger.info(
                    "max_evals=%d from single-item param_space; using %d.",
                    max_evals, _DEFAULT_MAX_EVALS,
                )
                max_evals = _DEFAULT_MAX_EVALS
            logger.info(
                "Expanded single-item param_space to search space (%d params), max_evals=%d.",
                len(param_space), max_evals,
            )

        # List-of-dicts mode: evaluate each config.
        if isinstance(param_space, list):
            if not param_space:
                raise ValueError("param_space must not be empty when provided as a list.")
            data = kwargs.get("data")
            if data is None:
                raise ValueError(
                    "data parameter is required for optimize(). Pass data=... (e.g. from the service)."
                )
            best_score = obj.initial_best_score(objective)
            best_params = None
            best_run_result = None
            with self.run("optimize"):
                for params in param_space:
                    res = self.train_test_validate(params=params, epochs=epochs, data=data)
                    score = obj.get_score(self, objective, res, data, params=params)
                    if obj.is_better(objective, score, best_score):
                        best_score = score
                        best_params = params.copy()
                if best_params is None:
                    raise ValueError("No successful trials in list-mode optimization.")
                # Re-run best config to keep model/run state aligned.
                best_run_result = self.train_test_validate(params=best_params, epochs=epochs, data=data)
            name = getattr(self, "registered_model_name", None) or os.environ.get("MLFLOW_REGISTERED_MODEL_NAME", "aicybops_model")
            out = {"best_params": best_params, "best_loss": obj.display_score(objective, best_score)}
            if isinstance(best_run_result, dict):
                if best_run_result.get("data_collection_metrics_count") is not None:
                    out["data_collection_metrics_count"] = best_run_result["data_collection_metrics_count"]
                if best_run_result.get("data_collection_logs_count") is not None:
                    out["data_collection_logs_count"] = best_run_result["data_collection_logs_count"]
            if out.get("data_collection_metrics_count") is None and self._cached_data_collection_metrics_count is not None:
                out["data_collection_metrics_count"] = self._cached_data_collection_metrics_count
            if out.get("data_collection_logs_count") is None and self._cached_data_collection_logs_count is not None:
                out["data_collection_logs_count"] = self._cached_data_collection_logs_count
            try:
                from mlflow.tracking import MlflowClient
                client = MlflowClient()
                versions = client.search_model_versions(
                    filter_string=f"name='{name}'",
                    order_by=["version_number DESC"],
                    max_results=1,
                )
                if versions:
                    out["registered_model_name"] = name
                    out["model_version"] = versions[0].version
            except Exception as e:
                logger.debug("Could not get model version from registry after optimize: %s", e)
            return out

        # Dict-of-ranges mode.
        from pipelines.optimization import OptimizationPipeline
        pipeline = OptimizationPipeline(
            dam=self,
            param_space=param_space,
            max_evals=max_evals,
            epochs=epochs,
            objective=objective,
            method=method,
            **kwargs
        )
        retrain_result = None
        with self.run("optimize"):
            best_params, best_score, trial_results = pipeline.run()

            # Re-train best params so registry version matches best trial.
            if best_params is not None:
                base_params = kwargs.get("base_params", {})
                retrain_params = {**base_params, **best_params}
                data = kwargs.get("data")
                if data is None and self._last_data is not None:
                    data = self._last_data
                if data is not None:
                    retrain_data = data.copy()
                    if "window_size" in retrain_params:
                        retrain_data["window_size"] = retrain_params["window_size"]
                    logger.info(
                        "Re-training with best params to register correct model in MLflow: %s",
                        best_params,
                    )
                    extra_kw = {k: v for k, v in kwargs.items() if k not in ("data", "base_params")}
                    retrain_result = self.train_test_validate(
                        params=retrain_params, epochs=epochs, data=retrain_data, **extra_kw
                    )

        name = getattr(self, "registered_model_name", None) or os.environ.get("MLFLOW_REGISTERED_MODEL_NAME", "aicybops_model")
        out = {"best_params": best_params, "best_loss": best_score}
        if isinstance(retrain_result, dict):
            if retrain_result.get("data_collection_metrics_count") is not None:
                out["data_collection_metrics_count"] = retrain_result["data_collection_metrics_count"]
            if retrain_result.get("data_collection_logs_count") is not None:
                out["data_collection_logs_count"] = retrain_result["data_collection_logs_count"]
        elif self._last_processor is not None:
            metrics_count = getattr(self._last_processor, "data_collection_metrics_count", None)
            logs_count = getattr(self._last_processor, "data_collection_logs_count", None)
            if metrics_count is not None:
                out["data_collection_metrics_count"] = metrics_count
            if logs_count is not None:
                out["data_collection_logs_count"] = logs_count
        if out.get("data_collection_metrics_count") is None and self._cached_data_collection_metrics_count is not None:
            out["data_collection_metrics_count"] = self._cached_data_collection_metrics_count
        if out.get("data_collection_logs_count") is None and self._cached_data_collection_logs_count is not None:
            out["data_collection_logs_count"] = self._cached_data_collection_logs_count
        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            versions = client.search_model_versions(
                filter_string=f"name='{name}'",
                order_by=["version_number DESC"],
                max_results=1,
            )
            if versions:
                out["registered_model_name"] = name
                out["model_version"] = versions[0].version
        except Exception as e:
            logger.debug("Could not get model version from registry after optimize: %s", e)
        return out

    def save_model(self, path: str):
        """
        Save the DAM model with full state including dimensions, configuration, and pipeline state.
        
        Args:
            path: Path where to save the model file
        
        Raises:
            ValueError: If model is not trained or dimensions are not set
        """
        if self.model is None:
            raise ValueError("Cannot save model: model has not been built. Call train() first.")
        
        if self.dimensions is None:
            raise ValueError("Cannot save model: dimensions are not set. Call train() first.")
        
        # Ensure output directory exists.
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save dimensions in long-form for loader compatibility.
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'dimensions': {
                'load_metrics_dim': self.dimensions['load'],
                'traffic_metrics_dim': self.dimensions['traffic'],
                'log_seq_dim': self.dimensions['log'],
                'lstm_hidden_dim': self.lstm_hidden_dim,
            },
            'config': {
                'window_size': self.window_size,
                'stride': self.stride,
                'align_freq': self.align_freq,
                'lstm_hidden_dim': self.lstm_hidden_dim
            }
        }
        
        # Save unified config when available.
        if self._unified_config:
            save_dict['unified_config'] = self._unified_config.to_dict()
        
        # Save pipeline state when available.
        if self.pipeline is not None:
            pipeline_state = {
                'spot_type': self.pipeline.spot_type,
                'risk_level': self.pipeline.get_risk_level(),
                'depth': self.pipeline.get_depth(),
            }
            
            # Save baseline inputs/state when available.
            train_scores = self.pipeline.get_train_anomaly_scores()
            if train_scores is not None:
                pipeline_state['train_anomaly_scores'] = train_scores
            
            baseline_fitted = self.pipeline.get_baseline_fitted()
            if baseline_fitted:
                pipeline_state['baseline_fitted'] = baseline_fitted
            
            save_dict['pipeline_state'] = pipeline_state
        
        # Save scaler stats for inference normalization.
        if self._cached_processor is not None and getattr(self._cached_processor, 'metrics_analyser', None) is not None:
            stats = getattr(self._cached_processor.metrics_analyser, 'scaler_stats', None)
            if stats:
                save_dict['scaler_stats'] = stats
        
        # Save data config for pipeline restoration.
        if self._last_data is not None:
            training = self._unified_config.get_training()
            arch = self._unified_config.get_model_architecture()
            data_config = {
                'config_path': self._last_data.get('config_path') or (str(self._unified_config.config_path) if self._unified_config.config_path else None),
                'window_size': self._last_data.get('window_size', arch.get('window_size')),
                'stride': self._last_data.get('stride', arch.get('stride')),
                'align_freq': self._last_data.get('align_freq', arch.get('align_freq')),
                'batch_size': self._last_data.get('batch_size', training.get('batch_size')),
                'train_ratio': self._last_data.get('train_ratio', training.get('train_ratio')),
                'val_ratio': self._last_data.get('val_ratio', training.get('val_ratio')),
                'use_api': self._last_data.get('use_api', False),
            }
            # Persist data_config only when config_path exists.
            if data_config.get('config_path'):
                if 'pipeline_state' not in save_dict:
                    save_dict['pipeline_state'] = {}
                save_dict['pipeline_state']['data_config'] = data_config
        
        # Save checkpoint.
        torch.save(save_dict, model_path)
        logger.info("Model saved to: %s", model_path)

    def _apply_checkpoint(self, checkpoint: dict) -> None:
        """Apply a checkpoint dict (e.g. from MLflow registry) to this instance: set dimensions, load model, restore pipeline."""
        if 'unified_config' not in checkpoint:
            raise ValueError("Invalid checkpoint: 'unified_config' not found.")
        self._unified_config = DAMUnifiedConfig(config_dict=checkpoint['unified_config'])
        dimensions, _ = self._extract_model_metadata(checkpoint)
        arch = self._unified_config.get_model_architecture()
        self.window_size = arch.get('window_size', self.window_size)
        self.stride = arch.get('stride', self.stride)
        self.align_freq = arch.get('align_freq', self.align_freq)
        self.lstm_hidden_dim = arch.get('lstm_hidden_dim', self.lstm_hidden_dim)
        if 'load_metrics_dim' not in dimensions:
            raise ValueError("Invalid checkpoint: 'dimensions' must use long form (load_metrics_dim, etc.).")
        self.dimensions = {
            'load': dimensions['load_metrics_dim'],
            'traffic': dimensions['traffic_metrics_dim'],
            'log': dimensions['log_seq_dim'],
        }
        self.lstm_hidden_dim = dimensions.get('lstm_hidden_dim') or arch.get('lstm_hidden_dim', self.lstm_hidden_dim)
        self.build_model()
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        self.model.load_state_dict(state_dict)
        self._scaler_stats = checkpoint.get('scaler_stats')
        if 'pipeline_state' in checkpoint:
            self._restore_pipeline_state(checkpoint['pipeline_state'])
    
    def _extract_model_metadata(self, checkpoint: dict) -> tuple:
        """
        Extract dimensions and config from checkpoint.
        
        Returns:
            Tuple of (dimensions dict, config dict)
        
        Raises:
            ValueError: If checkpoint doesn't contain required 'dimensions' key
        """
        if 'dimensions' not in checkpoint:
            raise ValueError(
                "Invalid checkpoint: 'dimensions' key not found. "
                "Retrain and save the model with the current code to produce a loadable checkpoint."
            )
        
        dimensions = checkpoint['dimensions']
        config = checkpoint.get('config', {})
        return dimensions, config

    def _log_pipeline_state_artifact(self, pipeline) -> bool:
        """Log dSPOT baseline scores/config to `pipeline_state/` in MLflow."""
        import mlflow
        import tempfile

        scores = getattr(pipeline, "train_anomaly_scores", None) if pipeline is not None else None
        if scores is None or not mlflow.active_run():
            reason = "pipeline/scores missing" if scores is None else "no active MLflow run"
            logger.warning("pipeline_state/: skipped - %s.", reason)
            return False

        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as file_handle:
            np.save(file_handle, scores)
            tmp_file = file_handle.name
        try:
            mlflow.log_artifact(tmp_file, artifact_path="pipeline_state")
            from utils.config import INIT_QUANTILE
            risk = pipeline.get_risk_level()
            depth = pipeline.get_depth()
            mlflow.log_dict(
                {
                    "baseline_fitted": bool(pipeline.get_baseline_fitted()),
                    "init_quantile": float(INIT_QUANTILE),
                    "spot_type": pipeline.spot_type,
                    "risk_level": float(risk) if risk is not None else None,
                    "depth": int(depth) if depth is not None else None,
                },
                "pipeline_state/config.json",
            )
            logger.info("pipeline_state/: logged %d scores to active MLflow run.", len(scores))
            return True
        finally:
            os.unlink(tmp_file)

    def _restore_pipeline_state(self, pipeline_state: dict):
        """Restore pipeline state including anomaly detector configuration.

        When saved train_anomaly_scores are present, restore directly from those
        scores; otherwise fetch data and refit as a fallback.
        """
        from pipelines.training import DAMPipeline
        import torch.nn as nn
        import torch.optim as optim

        anomaly = self._unified_config.get_anomaly_detection() if self._unified_config else {}
        training = self._unified_config.get_training() if self._unified_config else {}

        spot_type = pipeline_state.get('spot_type', anomaly.get('spot_type', 'dSPOT'))
        risk_level = pipeline_state.get('risk_level', anomaly.get('risk_level', 1e-3))
        depth = pipeline_state.get('depth', anomaly.get('depth', 10))
        learning_rate = training.get('learning_rate') if training else 1e-4
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # Fast path: rebuild from saved anomaly scores.
        if 'train_anomaly_scores' in pipeline_state and pipeline_state.get('baseline_fitted', False):
            resolved_device = _resolve_device()
            self.pipeline = DAMPipeline(
                model=self.model,
                train_loader=None,
                val_loader=None,
                optimizer=optimizer,
                criterion=criterion,
                num_epochs=0,
                spot_type=spot_type,
                risk_level=risk_level,
                depth=depth,
                device=resolved_device,
            )
            self.pipeline.train_anomaly_scores = pipeline_state['train_anomaly_scores']
            init_quantile = pipeline_state.get('init_quantile', INIT_QUANTILE)
            self.pipeline.anomaly_detector.fit_baseline(
                baseline_scores=self.pipeline.train_anomaly_scores,
                stream_scores=np.array([]),
                init_quantile=init_quantile,
            )
            return

        # Fallback: fetch data and refit baseline.
        data = self._last_data
        if data is None and 'data_config' in pipeline_state:
            data_config = pipeline_state['data_config']
            data = {
                'config_path': data_config.get('config_path'),
                'window_size': data_config.get('window_size', self.window_size),
                'stride': data_config.get('stride', self.stride),
                'align_freq': data_config.get('align_freq', self.align_freq),
                'batch_size': data_config.get('batch_size'),
                'train_ratio': data_config.get('train_ratio'),
                'val_ratio': data_config.get('val_ratio'),
                'use_api': data_config.get('use_api', False),
            }
            if not data.get('config_path'):
                logger.warning(
                    "Cannot restore pipeline state: saved data_config missing config_path. "
                    "Model loaded for inference only. Pipeline state will not be restored."
                )
                return

        if data is None:
            logger.warning(
                "Cannot restore pipeline state: no data available (_last_data or data_config). "
                "Model loaded for inference only. Pipeline state will not be restored."
            )
            return

        train_loader, val_loader, _, _ = self._prepare_data(mode='training', data=data)
        resolved_device = _resolve_device()
        self.pipeline = DAMPipeline(
            model=self.model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=1,
            spot_type=spot_type,
            risk_level=risk_level,
            depth=depth,
            device=resolved_device,
        )
        if 'train_anomaly_scores' in pipeline_state:
            self.pipeline.train_anomaly_scores = pipeline_state['train_anomaly_scores']
            if pipeline_state.get('baseline_fitted', False):
                self.pipeline.anomaly_detector.fit_baseline(
                    baseline_scores=self.pipeline.train_anomaly_scores,
                    stream_scores=np.array([]),
                    init_quantile=INIT_QUANTILE,
                )
    
    def load_model(self, path: str, restore_pipeline: bool = True):
        """
        Load a saved DAM model and restore its state.
        
        Args:
            path: Path to the saved model file
            restore_pipeline: If True, restores pipeline state (anomaly detector) if available
        
        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model file format is invalid or dimensions don't match
        """
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Validate required checkpoint fields.
        dimensions, _ = self._extract_model_metadata(checkpoint)
        if 'unified_config' not in checkpoint:
            raise ValueError(
                "Invalid checkpoint: 'unified_config' not found. "
                "Retrain and save the model with the current code to produce a loadable checkpoint."
            )
        self._unified_config = DAMUnifiedConfig(config_dict=checkpoint['unified_config'])
        arch = self._unified_config.get_model_architecture()
        self.window_size = arch.get('window_size', self.window_size)
        self.stride = arch.get('stride', self.stride)
        self.align_freq = arch.get('align_freq', self.align_freq)
        self.lstm_hidden_dim = arch.get('lstm_hidden_dim', self.lstm_hidden_dim)

        # Require long-form dimension keys.
        if 'load_metrics_dim' not in dimensions:
            raise ValueError(
                "Invalid checkpoint: 'dimensions' must use long form (load_metrics_dim, traffic_metrics_dim, "
                "log_seq_dim, lstm_hidden_dim). Resave the model with the current code."
            )
        self.dimensions = {
            'load': dimensions['load_metrics_dim'],
            'traffic': dimensions['traffic_metrics_dim'],
            'log': dimensions['log_seq_dim'],
        }
        self.lstm_hidden_dim = dimensions.get('lstm_hidden_dim') or arch.get('lstm_hidden_dim', self.lstm_hidden_dim)

        # Build and load model.
        self.build_model()
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        self.model.load_state_dict(state_dict)
        self._scaler_stats = checkpoint.get('scaler_stats')
        
        # Restore pipeline if requested.
        if restore_pipeline and 'pipeline_state' in checkpoint:
            self._restore_pipeline_state(checkpoint['pipeline_state'])
        
        logger.info(
            "Model loaded from: %s; dimensions: load=%s, traffic=%s, log=%s; "
            "config: window_size=%s, lstm_hidden_dim=%s",
            model_path, self.dimensions["load"], self.dimensions["traffic"], self.dimensions["log"],
            self.window_size, self.lstm_hidden_dim,
        )
        if self._unified_config:
            logger.info("Unified config: restored from checkpoint")

