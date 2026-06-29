from .config_models import DAMDataConfig
from typing import Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DAMConfigLoader:
    """Backward-compatible wrapper around DAMDataConfig."""

    def __init__(self, config_path: str):
        logger.debug("Loading config with Pydantic")
        self.config_path = Path(config_path)
        self.config = DAMDataConfig.from_json(str(config_path))
    
    def get_core_metrics_config(self) -> Dict[str, Dict[str, Any]]:
        return self.config.get_core_metrics()

    def get_window_size(self) -> int:
        return self.config.window_size

    def get_stride(self) -> int:
        return self.config.stride

    def get_align_freq(self) -> str:
        return self.config.align_freq

    def get_label_config(self) -> Dict[str, Any]:
        if self.config.label_dataset is None:
            return {}
        label_config = self.config.label_dataset
        return {
            'enabled': label_config.enabled,
            'path': label_config.path,
            'timestamp_column': label_config.timestamp_column,
            'label_column': label_config.label_column,
            'timestamp_format': label_config.timestamp_format
        }
    
    def get_full_config(self) -> Dict[str, Any]:
        config_dict = self.config.model_dump(exclude_none=False)
        if config_dict.get('label_dataset') is None:
            # Match legacy behavior: missing key means labels disabled.
            config_dict.pop('label_dataset', None)
        return config_dict

