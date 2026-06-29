from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from typing import Dict, List, Literal, Optional, Any
import json


class MetricConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    
    measurement: str = Field(..., alias='_measurement', description="Measurement type identifier")
    value_column: str = Field(..., description="Column name for metric values")
    filter: Dict[str, Any] = Field(default_factory=dict, description="Optional filters")


class MetricGroupConfig(BaseModel):
    """Configuration for a metric group (load, traffic, etc.)."""
    group_name: str = Field(..., description="Name of the group")
    csv_mapping: Dict[str, MetricConfig] = Field(default_factory=dict)
    description: Optional[str] = None
    
    @field_validator('group_name')
    @classmethod
    def validate_group_name(cls, v):
        if not v or not v.strip():
            raise ValueError("group_name cannot be empty")
        return v.strip()


class LogGroupConfig(BaseModel):
    """Configuration for log groups."""
    group_name: str = Field(..., description="Name of the log group")
    columns: List[str] = Field(default_factory=list, description="Column names for log data")
    description: Optional[str] = None


class LabelDatasetConfig(BaseModel):
    enabled: bool = Field(default=False)
    path: Optional[str] = Field(None, description="Path to label CSV file")
    timestamp_column: str = Field(default="_time")
    label_column: str = Field(default="anomaly_label")
    timestamp_format: str = Field(default="unix_ms")
    
    @field_validator('timestamp_format')
    @classmethod
    def validate_timestamp_format(cls, v):
        valid_formats = ['unix_ms', 'unix_s', 'iso']
        if v not in valid_formats:
            raise ValueError(f"timestamp_format must be one of {valid_formats}")
        return v


class OutputPathsConfig(BaseModel):
    generated_metrics: str = Field(default="data/generated/metrics.csv")
    generated_logs: str = Field(default="logs/logs.txt")
    model: str = Field(default="models/generator_model.pth")


class EarlyStoppingConfig(BaseModel):
    enabled: bool = Field(default=True)
    patience: int = Field(default=5, ge=1)
    min_delta: float = Field(default=0.001, ge=0.0)
    mode: str = Field(default='min')
    
    @field_validator('mode')
    @classmethod
    def validate_mode(cls, v):
        valid_modes = ['min', 'max']
        if v not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}")
        return v


class ModelArchitectureConfig(BaseModel):
    lstm_hidden_dim: int = Field(default=32, ge=1, le=512)
    window_size: int = Field(default=10, ge=1, le=100)
    stride: int = Field(default=1, ge=1)
    align_freq: str = Field(default='1s')


class TrainingConfig(BaseModel):
    learning_rate: float = Field(default=0.0001, gt=0.0, le=1.0)
    batch_size: int = Field(default=32, ge=1)
    num_epochs: int = Field(default=10, ge=1)
    train_ratio: float = Field(default=0.8, gt=0.0, lt=1.0)
    val_ratio: float = Field(default=0.2, gt=0.0, lt=1.0)
    training_mode: Literal['supervised', 'unsupervised'] = Field(
        default='supervised',
        description=(
            "supervised: labels required; stratified normal-only training split. "
            "unsupervised: no labels needed; shuffled split (train_ratio + val_ratio must be < 1.0)."
        )
    )
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)

    @model_validator(mode='after')
    def validate_ratios(self):
        if self.training_mode == 'unsupervised':
            if self.train_ratio + self.val_ratio >= 1.0:
                raise ValueError(
                    f"In unsupervised mode, train_ratio + val_ratio must be < 1.0 to leave data for the test set "
                    f"(got {self.train_ratio} + {self.val_ratio} = {self.train_ratio + self.val_ratio:.4f})."
                )
        return self


class AnomalyDetectionConfig(BaseModel):
    """Configuration for anomaly detection (EVT/SPOT) parameters."""
    spot_type: str = Field(..., description="SPOT type: 'SPOT' or 'dSPOT'")
    risk_level: float = Field(..., gt=0.0, le=1.0, description="Risk level (q parameter) for EVT")
    depth: int = Field(..., ge=1, description="Window size for dSPOT moving average")
    init_quantile: float = Field(..., gt=0.0, lt=1.0, description="Initial quantile for threshold")
    
    @field_validator('spot_type')
    @classmethod
    def validate_spot_type(cls, v):
        valid_types = ['SPOT', 'dSPOT']
        if v not in valid_types:
            raise ValueError(f"spot_type must be one of {valid_types}")
        return v


class DataPathsConfig(BaseModel):
    metrics_csv: str = Field(..., description="Path to metrics CSV file")
    log_file: str = Field(..., description="Path to log file")
    labels_csv: Optional[str] = Field(None, description="Path to labels CSV file")


class ModelPathsConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_dir: str = Field(default="models")
    default_model: str = Field(default="models/dam_model.pth")


class DAMDataConfig(BaseModel):
    metric_groups: Dict[str, MetricGroupConfig] = Field(default_factory=dict)
    log_groups: Dict[str, LogGroupConfig] = Field(default_factory=dict)
    label_dataset: Optional[LabelDatasetConfig] = None
    output_paths: Optional[OutputPathsConfig] = None
    
    # Optional fields from config.
    window_size: int = Field(default=10, ge=1, le=100)
    stride: int = Field(default=1, ge=1)
    align_freq: str = Field(default='1s')
    
    # Additional fields kept for compatibility.
    entity_column: Optional[str] = None
    sequence_index: Optional[str] = None
    timestamp_format: Optional[str] = None
    all_columns: Optional[List[str]] = None
    column_descriptions: Optional[Dict[str, str]] = None
    entity_columns: Optional[List[str]] = None
    segment_size: Optional[int] = None
    epochs: Optional[int] = None
    num_entities: Optional[int] = None
    start_date: Optional[str] = None
    data_types: Optional[Dict[str, str]] = None
    auto_detect_data_types: Optional[bool] = None
    
    @classmethod
    def from_json(cls, config_path: str):
        """Load data processing config from JSON file."""
        with open(config_path, 'r') as f:
            data = json.load(f)
        
        # Extract data_processing section if it exists, otherwise use top-level fields
        if 'data_processing' in data:
            data_processing = data['data_processing'].copy()
        else:
            # Use top-level fields for backward compatibility
            data_processing = data.copy()
        
        # Also include label_dataset from top level if it exists (for backward compatibility)
        if 'label_dataset' in data and 'label_dataset' not in data_processing:
            data_processing['label_dataset'] = data['label_dataset']
        
        return cls(**data_processing)
    
    def get_core_metrics(self) -> Dict[str, Dict[str, Any]]:
        core_metrics = {}
        for group_config in self.metric_groups.values():
            if group_config.csv_mapping:
                for metric_name, metric_config in group_config.csv_mapping.items():
                    core_metrics[metric_name] = {
                        '_measurement': metric_config.measurement,
                        'value_column': metric_config.value_column,
                        'filter': metric_config.filter,
                    }
        return core_metrics


class DAMUnifiedConfigModel(BaseModel):
    model_config = ConfigDict(
        extra='allow',
        protected_namespaces=(),
    )

    data_processing: Optional[DAMDataConfig] = None
    metric_groups: Dict[str, MetricGroupConfig] = Field(default_factory=dict)
    log_groups: Dict[str, LogGroupConfig] = Field(default_factory=dict)
    label_dataset: Optional[LabelDatasetConfig] = None
    
    model_architecture: ModelArchitectureConfig = Field(...)
    training: TrainingConfig = Field(...)
    anomaly_detection: AnomalyDetectionConfig = Field(...)
    data_paths: DataPathsConfig = Field(...)
    model_paths: Optional[ModelPathsConfig] = None
    
    @classmethod
    def from_json(cls, config_path: str):
        """Load from JSON; supports both nested (data_processing) and flat structures."""
        with open(config_path, 'r') as f:
            data = json.load(f)
        
        if 'data_processing' in data:
            data_processing = data['data_processing'].copy()
            if 'metric_groups' in data_processing:
                data['metric_groups'] = data_processing['metric_groups']
            if 'log_groups' in data_processing:
                data['log_groups'] = data_processing['log_groups']
            if 'sequence_index' in data_processing:
                data['sequence_index'] = data_processing['sequence_index']
            if 'timestamp_format' in data_processing:
                data['timestamp_format'] = data_processing['timestamp_format']
            data['data_processing'] = DAMDataConfig(**data_processing)
        
        if 'label_generation' in data and 'label_dataset' not in data:
            data['label_dataset'] = data['label_generation']
        return cls(**data)
    
    def get_core_metrics(self) -> Dict[str, Dict[str, Any]]:
        core_metrics = {}
        for group_config in self.metric_groups.values():
            if group_config.csv_mapping:
                for metric_name, metric_config in group_config.csv_mapping.items():
                    core_metrics[metric_name] = {
                        '_measurement': metric_config.measurement,
                        'value_column': metric_config.value_column,
                        'filter': metric_config.filter,
                    }
        return core_metrics
