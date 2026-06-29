"""
Configuration loading and validation utilities.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from log_generator import LogGenerator
from utils.path_utils import resolve_path_from_config, resolve_path_from_script_dir


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load and validate config file exists."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    print(f"Loading config from: {config_path}")
    with open(config_path, "r") as f:
        return json.load(f)


def validate_training_config(config: Dict[str, Any]) -> None:
    """Validate config fields required for training."""
    if 'model_training' not in config:
        raise ValueError("Config file must contain 'model_training' section")
    model_config = config['model_training']
    required_fields = ['epochs', 'entity_columns', 'sequence_index', 'segment_size', 
                      'timestamp_format', 'auto_detect_data_types']
    for field in required_fields:
        if field not in model_config:
            raise ValueError(f"Config file must contain 'model_training.{field}' field")


def validate_generation_config(config: Dict[str, Any], script_file: str) -> Tuple[int, int, Optional[str], List[str], str, str, Optional[List[str]], Path, Path]:
    """Validate and extract config fields required for generation."""
    # Get nested config sections
    if 'model_training' not in config:
        raise ValueError("Config file must contain 'model_training' section")
    if 'metrics_generation' not in config:
        raise ValueError("Config file must contain 'metrics_generation' section")
    if 'paths' not in config:
        raise ValueError("Config file must contain 'paths' section")
    
    model_config = config['model_training']
    metrics_config = config['metrics_generation']
    paths_config = config['paths']
    
    # Required fields from metrics_generation
    if 'num_entities' not in metrics_config:
        raise ValueError("Config file must contain 'metrics_generation.num_entities' field")
    num_entities = metrics_config['num_entities']
    
    # Required fields from model_training
    if 'segment_size' not in model_config:
        raise ValueError("Config file must contain 'model_training.segment_size' field")
    segment_size = model_config['segment_size']
    
    if 'sequence_index' not in model_config:
        raise ValueError("Config file must contain 'model_training.sequence_index' field")
    sequence_index = model_config['sequence_index']
    
    if 'timestamp_format' not in model_config:
        raise ValueError("Config file must contain 'model_training.timestamp_format' field")
    timestamp_format = model_config['timestamp_format']
    
    # Entity columns (support both singular and plural)
    if 'entity_columns' in model_config:
        entity_columns = model_config['entity_columns']
    elif 'entity_column' in model_config:
        entity_columns = [model_config['entity_column']]
    else:
        raise ValueError("Config file must contain 'model_training.entity_columns' or 'model_training.entity_column' field")
    
    # Optional fields from metrics_generation
    start_date = metrics_config.get('start_date', None)
    all_columns = metrics_config.get('all_columns', None)
    
    # Output paths
    if 'output_paths' not in paths_config:
        raise ValueError("Config file must contain 'paths.output_paths' field")
    output_paths = paths_config['output_paths']
    
    if 'generated_logs' not in output_paths:
        raise ValueError("Config file must contain 'output_paths.generated_logs' field")
    log_file_path = resolve_path_from_script_dir(output_paths['generated_logs'], script_file)
    
    if 'generated_metrics' not in output_paths:
        raise ValueError("Config file must contain 'output_paths.generated_metrics' field")
    metrics_file_path = resolve_path_from_script_dir(output_paths['generated_metrics'], script_file)
    
    return (num_entities, segment_size, start_date, entity_columns, sequence_index,
            timestamp_format, all_columns, log_file_path, metrics_file_path)


def resolve_paths_from_config(
    config: Dict[str, Any], 
    config_file_path: Path, 
    mode: str,
    script_file: str
) -> Tuple[Optional[Path], Path, Optional[LogGenerator]]:
    """Resolve all paths from config based on mode."""
    if 'paths' not in config:
        raise ValueError("Config file must contain 'paths' section")
    paths_config = config['paths']
    
    data_path = None
    if mode in ['train', 'train-and-generate']:
        if 'data_path' not in paths_config:
            raise ValueError("Config file must contain 'paths.data_path' field for training")
        data_path = resolve_path_from_script_dir(paths_config['data_path'], script_file, must_exist=True)
        print(f"Using training data from: {data_path}")
    
    if 'output_paths' not in paths_config or 'model' not in paths_config['output_paths']:
        raise ValueError("Config file must contain 'paths.output_paths.model' field")
    model_path = resolve_path_from_script_dir(paths_config['output_paths']['model'], script_file)
    print(f"Using model path: {model_path}")
    
    log_gen = None
    if mode in ['generate', 'train-and-generate']:
        if 'log_generation' not in config:
            raise ValueError("Config file must contain 'log_generation' section for generation")
        log_config = config['log_generation']
        if 'log_generator_config_path' not in log_config:
            raise ValueError("Config file must contain 'log_generation.log_generator_config_path' field for generation")
        log_generator_config_path = resolve_path_from_config(
            config_file_path, 'log_generation.log_generator_config_path', config, must_exist=True
        )
        print(f"Initializing log generator from: {log_generator_config_path}")
        log_gen = LogGenerator(root_config_path=log_generator_config_path)
    
    return data_path, model_path, log_gen
