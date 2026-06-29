from pathlib import Path
from typing import Dict, Any, Optional
import sys

# Ensure `processing` imports resolve.
_dam_model_dir = Path(__file__).resolve().parent.parent.parent
if str(_dam_model_dir) not in sys.path:
    sys.path.insert(0, str(_dam_model_dir))
_processing_dir = _dam_model_dir / "processing"
if str(_processing_dir) not in sys.path:
    sys.path.insert(0, str(_processing_dir))

try:
    from processing.config.config_models import DAMUnifiedConfigModel  # type: ignore
except ImportError:
    from processing.config import DAMUnifiedConfigModel  # type: ignore


class DAMUnifiedConfig:
    """Load and expose DAM unified config."""

    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict[str, Any]] = None):
        if config_path:
            self.config_path = Path(config_path)
            if not self.config_path.exists():
                raise FileNotFoundError(
                    f"Config file not found: {config_path}"
                )
            try:
                self._pydantic_config = DAMUnifiedConfigModel.from_json(str(config_path))
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Failed to load or validate config file '{config_path}': {e}\n"
                    "Required sections: model_architecture, training, anomaly_detection, data_paths"
                ) from e
        elif config_dict:
            # Build from in-memory config.
            try:
                self._pydantic_config = DAMUnifiedConfigModel(**config_dict)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Failed to load config from dictionary: {e}\n"
                    "Required sections: model_architecture, training, anomaly_detection, data_paths"
                ) from e
            self.config_path = None
        else:
            raise ValueError(
                "config_path is required. Example: DAMUnifiedConfig(config_path='configs/dam_config.json')"
            )
    
    def get_model_architecture(self) -> Dict[str, Any]:
        arch = self._pydantic_config.model_architecture
        return {
            'lstm_hidden_dim': arch.lstm_hidden_dim,
            'window_size': arch.window_size,
            'stride': arch.stride,
            'align_freq': arch.align_freq,
        }
    
    def get_training(self) -> Dict[str, Any]:
        training = self._pydantic_config.training
        es = training.early_stopping
        return {
            'learning_rate': training.learning_rate,
            'batch_size': training.batch_size,
            'num_epochs': training.num_epochs,
            'train_ratio': training.train_ratio,
            'val_ratio': training.val_ratio,
            'early_stopping': {
                'enabled': es.enabled,
                'patience': es.patience,
                'min_delta': es.min_delta,
                'mode': es.mode,
            }
        }
    
    def get_anomaly_detection(self) -> Dict[str, Any]:
        anomaly = self._pydantic_config.anomaly_detection
        return {
            'spot_type': anomaly.spot_type,
            'risk_level': anomaly.risk_level,
            'depth': anomaly.depth,
            'init_quantile': anomaly.init_quantile,
        }
    
    def get_data_paths(self) -> Dict[str, Any]:
        return self._pydantic_config.data_paths.model_dump()

    def get_model_paths(self) -> Dict[str, Any]:
        if self._pydantic_config.model_paths:
            return self._pydantic_config.model_paths.model_dump()
        return {}

    def get_data_processing(self) -> Dict[str, Any]:
        if self._pydantic_config.data_processing:
            return self._pydantic_config.data_processing.model_dump()
        # Fall back to top-level metric/log groups for legacy configs.
        result = {}
        if self._pydantic_config.metric_groups:
            result['metric_groups'] = {k: v.model_dump() for k, v in self._pydantic_config.metric_groups.items()}
        if self._pydantic_config.log_groups:
            result['log_groups'] = {k: v.model_dump() for k, v in self._pydantic_config.log_groups.items()}
        return result
    
    def get_label_dataset(self) -> Dict[str, Any]:
        if self._pydantic_config.label_dataset:
            return self._pydantic_config.label_dataset.model_dump()
        return {}

    def get_full_config(self) -> Dict[str, Any]:
        return self._pydantic_config.model_dump(exclude_none=True)

    def merge_with_overrides(self, overrides: Dict[str, Any]) -> 'DAMUnifiedConfig':
        current_dict = self._pydantic_config.model_dump(exclude_none=True)
        merged = self._deep_merge(current_dict, overrides)
        return DAMUnifiedConfig(config_dict=merged)

    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = DAMUnifiedConfig._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_architecture': self.get_model_architecture(),
            'training': self.get_training(),
            'anomaly_detection': self.get_anomaly_detection(),
            'data_paths': self.get_data_paths(),
            'model_paths': self.get_model_paths(),
            'data_processing': self.get_data_processing(),
            'label_dataset': self.get_label_dataset(),
        }


def build_data_config(
    config_path: str,
    overrides: Optional[Dict[str, Any]] = None,
    quick_test: bool = False,
) -> Dict[str, Any]:
    """Build dataset config for DAM training/evaluation."""
    unified = DAMUnifiedConfig(config_path=config_path)
    arch = unified.get_model_architecture()
    training = unified.get_training()
    config_dict = {
        "config_path": str(unified.config_path) if unified.config_path else config_path,
        "use_api": False,
        "batch_size": training.get("batch_size", 32),
        "train_ratio": training.get("train_ratio", 0.8),
        "val_ratio": training.get("val_ratio", 0.2),
        "window_size": arch.get("window_size", 10),
        "stride": arch.get("stride", 1),
        "align_freq": arch.get("align_freq", "1s"),
    }
    if quick_test:
        config_dict["max_train_samples"] = 1000
        config_dict["max_test_samples"] = 500
    if overrides:
        config_dict.update(overrides)
    return config_dict
