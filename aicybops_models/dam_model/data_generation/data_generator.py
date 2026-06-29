import argparse
import json
import datetime
from deepecho import PARModel
import pandas as pd
import os
import torch
from pathlib import Path
from log_generator import LogGenerator
from typing import Any, Dict, List, Optional, Tuple
from utils.path_utils import resolve_path_from_config, resolve_path_from_script_dir
from utils.config_utils import (
    load_config,
    validate_training_config,
    validate_generation_config,
    resolve_paths_from_config
)
from utils.timestamp_utils import (
    format_datetime_to_timestamp,
    convert_to_datetime_for_logs,
    generate_timestamp_for_row
)
from utils.postprocessing_utils import (
    clip_negative_values,
    enforce_counter_gauge_exclusivity
)
from utils.label_utils import generate_labels_from_metrics


# ============================================================================
# Data Type Detection
# ============================================================================

def _auto_detect_data_types(df: pd.DataFrame, entity_cols: list, sequence_ix: str) -> Dict[str, str]:
    """Auto-detect data types for columns (continuous or categorical)."""
    data_types = {}

    for col in df.columns:
        # skip entity columns, sequence index, and columns that are all NaN/empty
        if col in entity_cols or col == sequence_ix or df[col].isna().all():
            continue
    
        if pd.api.types.is_numeric_dtype(df[col]):
            data_types[col] = "continuous"
        else:
            data_types[col] = "categorical"
    
    return data_types
            

# ============================================================================
# Timestamp Conversion Utilities
# ============================================================================

def convert_timestamp_format(df: pd.DataFrame, timestamp_format: str, sequence_index: str) -> pd.DataFrame:
    """Converts the timestamp column FROM the specified format TO datetime (for training)."""
    if timestamp_format == 'unix_ms':
        df[sequence_index] = pd.to_datetime(df[sequence_index], unit='ms')
        print(f"Converted {sequence_index} from unix_ms to datetime")
    elif timestamp_format == 'unix_s':
        df[sequence_index] = pd.to_datetime(df[sequence_index], unit='s')
        print(f"Converted {sequence_index} from unix_s to datetime")
    elif timestamp_format == 'unix_ns':
        df[sequence_index] = pd.to_datetime(df[sequence_index], unit='ns')
        print(f"Converted {sequence_index} from unix_ns to datetime")
    else:
        df[sequence_index] = pd.to_datetime(df[sequence_index])
        print(f"Parsed {sequence_index} as datetime from string format")
    return df




# ============================================================================
# Model Training
# ============================================================================

def train_model(data_path: Path, configs: Dict[str, Any], model_output_path: Path) -> PARModel:
    """ Loads the data, trains the model, and saves it"""
    
    print(f"Loading data from {data_path}...")
    # Extract model configuration from nested structure
    try:
        model_config = configs['model_training']
        metrics_config = configs.get('metrics_generation', {})
        epochs = model_config['epochs']
        entity_columns = model_config['entity_columns']
        data_types = None
        sequence_index = model_config['sequence_index']
        segment_size = model_config['segment_size']
        timestamp_format = model_config['timestamp_format']
        auto_detect_data_types = model_config['auto_detect_data_types']
        all_columns = metrics_config.get('all_columns', None)  # Optional: for column filtering
    except (KeyError, ValueError) as e:
        raise ValueError(f"Error loading config: {e}") from e
    
    # Build list of columns to load if all_columns is specified
    usecols = None
    if all_columns is not None:
        # Create set of required columns (entity_columns + sequence_index)
        required_columns = set(entity_columns) | {sequence_index}
        # Union with all_columns to ensure we load everything needed
        columns_to_load = set(all_columns) | required_columns
        usecols = list(columns_to_load)
        print(f"Filtering to load only {len(usecols)} columns: {sorted(usecols)}")

    try:
        df = pd.read_csv(data_path, usecols=usecols, low_memory=False)
    except (FileNotFoundError, pd.errors.ParserError) as e:
        raise ValueError(f"Error loading data: {e}") from e

    required_columns = set(entity_columns) | {sequence_index}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Required columns are missing from the loaded data: {sorted(missing_columns)}. "
            f"Available columns: {sorted(df.columns)}"
        )
    
    print(f"Dataframe columns: {list(df.columns)}")
    print(f"Dataframe shape: {df.shape}")
    print(f"Sequence index: {sequence_index}")
    print(f"Timestamp format: {timestamp_format}")
    print(f"Auto detect data types: {auto_detect_data_types}")

    df = convert_timestamp_format(df, timestamp_format, sequence_index)
    
    if auto_detect_data_types:
        data_types = _auto_detect_data_types(df, entity_columns, sequence_index)
    else:
        data_types = model_config.get('data_types', {})
    
    if not data_types:
        raise ValueError("Data types are required for training. Either provide data_types in config or set auto_detect_data_types to True")

    if not entity_columns:
        raise ValueError("Entity columns are required for training. Either provide entity_columns in config or set auto_detect_entity_columns to True")
    
    print(f"Training the model for {epochs} epochs")
    model = PARModel(
        epochs=epochs,
    )
    
    
    model.fit(df,
              entity_columns=entity_columns,
              data_types=data_types,
              sequence_index=sequence_index,
              segment_size=segment_size
              )
    torch.save(model, model_output_path)
    print(f"Model trained and saved to {model_output_path}")
    return model


# ============================================================================
# Entity Column Formatting
# ============================================================================

def _format_entity_column(entity_column: str, entity_num: int) -> str:
    """Format entity column value to string format (e.g., '/entity_0.scope')."""
    if not isinstance(entity_column, str):
        raise ValueError(f"Entity column name must be a string, got {type(entity_column)}")
    return f"/entity_{entity_num}.scope"


def _format_all_entity_columns(
    new_metric_dict: Dict[str, Any], 
    primary_entity_col: str, 
    primary_entity_num: int, 
    entity_columns: List[str]
) -> Dict[str, Any]:
    """Format all entity columns from numeric IDs to string format (e.g., '/entity_0.scope')"""
    missing_entities = [col for col in entity_columns if col not in new_metric_dict]
    if missing_entities:
        raise ValueError(f"The following entity columns are missing in new_metric_dict: {missing_entities}")

    ent_cols = entity_columns.copy()             # Copy to avoid mutating input list
    ent_cols.remove(primary_entity_col)          # Remove the primary one for special handling

    # Format primary entity column using its designated entity number
    new_metric_dict[primary_entity_col] = _format_entity_column(primary_entity_col, primary_entity_num)

    # Format the rest of the entity columns (if any)
    while len(ent_cols) > 0:
        entity_col = ent_cols.pop(0)          # Get and remove first entity column from list
        entity_col_num = int(new_metric_dict[entity_col])  # Convert value to int (from generated numeric id)
        new_metric_dict[entity_col] = _format_entity_column(entity_col, entity_col_num)
    
    # Return the updated dict (mutated in place, but return for clarity)
    return new_metric_dict


# ============================================================================
# Measurement Mapping
# ============================================================================

def _build_measurement_value_column_mapping(data_path: Path) -> Dict[str, str]:
    """Build a mapping from measurement type to value column (counter or gauge) from training data"""
    try:
        # Read only the measurement and value columns for mapping; set low_memory=False for consistency
        df = pd.read_csv(data_path, usecols=['_measurement', 'counter', 'gauge'], low_memory=False)
        mapping = {}
        for measurement in df['_measurement'].unique():
            
            subset = df[df['_measurement'] == measurement]
            has_counter = subset['counter'].notna().any()
            has_gauge = subset['gauge'].notna().any()
            
            if has_counter and not has_gauge:
                mapping[measurement] = 'counter'
            elif has_gauge and not has_counter: 
                mapping[measurement] = 'gauge'
            elif has_counter:  # If both, prefer counter
                mapping[measurement] = 'counter'
            else:
                mapping[measurement] = 'gauge'
        return mapping
    except (OSError, pd.errors.ParserError, ValueError) as e:
        print(f"Warning: Could not build measurement mapping: {e}")
        return {}

# ============================================================================
# Row Processing
# ============================================================================

def _process_single_row(
    row: pd.Series,
    ix: int,
    primary_entity_col: str,
    entity_columns: List[str],
    segment_size: int,
    start_date: Optional[str],
    sequence_index: str,
    timestamp_format: str,
    all_columns: Optional[List[str]],
    log_generator: LogGenerator
) -> Dict[str, Any]:
    """Process a single generated row into a metric dictionary with logs."""
    new_metric_dict = row.to_dict()

    if primary_entity_col not in new_metric_dict:
        raise ValueError(
            f"Primary entity column '{primary_entity_col}' is missing from generated data. "
            f"This is required for timestamp calculation."
        )
    entity_num = int(new_metric_dict[primary_entity_col])
    
    # Format entity columns
    new_metric_dict = _format_all_entity_columns(
        new_metric_dict, primary_entity_col, entity_num, entity_columns
    )
    
    # Handle timestamp
    if sequence_index not in new_metric_dict:
        new_metric_dict[sequence_index] = generate_timestamp_for_row(
            ix, entity_num, segment_size, start_date, timestamp_format, sequence_index
        )
    elif hasattr(new_metric_dict[sequence_index], 'strftime'):
        # Convert datetime to target format
        new_metric_dict[sequence_index] = format_datetime_to_timestamp(
            new_metric_dict[sequence_index], timestamp_format
        )

    if all_columns:
        for col in all_columns:
            if col not in new_metric_dict:
                new_metric_dict[col] = None
    
    # Add timestamp for log generator (must be datetime object)
    if sequence_index in new_metric_dict and new_metric_dict[sequence_index] is not None:
        new_metric_dict['timestamp'] = convert_to_datetime_for_logs(
            new_metric_dict[sequence_index], timestamp_format
        )
    
    # Map entity column to service_name for log generator
    if primary_entity_col in new_metric_dict and new_metric_dict[primary_entity_col] is not None:
        new_metric_dict['service_name'] = new_metric_dict[primary_entity_col]
    else:
        new_metric_dict['service_name'] = 'unknown-service'
    
    return new_metric_dict


# ============================================================================
# Data Generation
# ============================================================================

def generate_data(
    model: PARModel, 
    num_entities: int, 
    log_generator: LogGenerator, 
    log_file_path: Path, 
    metrics_file_path: Path,
    segment_size: int, 
    start_date: Optional[str],
    entity_columns: List[str], 
    sequence_index: str,
    timestamp_format: str, 
    all_columns: Optional[List[str]],
    training_data_path: Optional[Path]
) -> None:
    """Generate metrics and logs using the trained model and log generator."""
    if not entity_columns:
        raise ValueError("entity_columns list cannot be empty.")
    
    print(f"Generating {num_entities} entities...")
    data = model.sample(num_entities=num_entities)
    
    primary_entity_col = entity_columns[0]
    generate_metrics_list = []
    
    # Process each row and generate logs
    with open(log_file_path, "w") as f:
        for ix, row in data.iterrows():
            new_metric_dict = _process_single_row(
                row, ix, primary_entity_col, entity_columns, segment_size,
                start_date, sequence_index, timestamp_format, all_columns, log_generator
            )
            
            # Generate and write logs
            log_lines = log_generator.generate_log(new_metric_dict)
            if log_lines:
                for line in log_lines:
                    f.write(line + "\n")
            
            generate_metrics_list.append(new_metric_dict)
    
    # Build final dataframe
    final_dataframe = pd.DataFrame(generate_metrics_list)
    
    # Reorder columns if specified
    if all_columns:
        for col in all_columns:
            if col not in final_dataframe.columns:
                final_dataframe[col] = None
        final_dataframe = final_dataframe[all_columns]
    
    # Post-process metrics
    clip_negative_values(final_dataframe)
    
    measurement_mapping = {}
    if training_data_path:
        measurement_mapping = _build_measurement_value_column_mapping(training_data_path)
    enforce_counter_gauge_exclusivity(final_dataframe, measurement_mapping)
    
    # Save results
    final_dataframe.to_csv(metrics_file_path, index=False)
    print("Data generation complete")
    print(f"Data generated and saved to {metrics_file_path}")
    print(f"Log generated and saved to {log_file_path}")
    
# ============================================================================
# Output Display
# ============================================================================

def print_data_generated(metrics_file_path: Path, log_file_path: Path) -> None:
    """Print sample of generated metrics and logs."""
    print("\n=== Sample of Generated Metrics ===")
    try:
        df = pd.read_csv(metrics_file_path)
        print(df.head(5).to_string(index=False))
        print(f"\nTotal metrics rows: {len(df)}")
    except (OSError, pd.errors.ParserError, ValueError) as e:
        print(f"Could not read metrics file: {e}")

    print("\n=== Sample of Generated Log ===")
    try:
        with open(log_file_path, "r") as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                print(line.strip())
        with open(log_file_path, "r") as f:
            total_lines = sum(1 for _ in f)
        print(f"\nTotal log lines: {total_lines}")
    except OSError as e:
        print(f"Could not read log file: {e}")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data generation and training")
    parser.add_argument('mode', choices=['train', 'generate', 'train-and-generate'], 
                       help='Mode to run the script')
    parser.add_argument('--config-path', type=str, required=True, 
                       help='Path to the configuration JSON file')
    parser.add_argument('--generate-labels', action='store_true',
                       help='Generate anomaly labels after metrics and logs (requires label_dataset in config)')
    args = parser.parse_args()

    config_file_path = Path(args.config_path)
    config = load_config(config_file_path)
    
    # Resolve paths and initialize components
    data_path, model_path, log_gen = resolve_paths_from_config(config, config_file_path, args.mode, __file__)
    
    # Train or load model
    trained_model = None
    if args.mode in ['train', 'train-and-generate']:
        validate_training_config(config)
        trained_model = train_model(data_path, config, model_path)
    elif args.mode == 'generate':
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        trained_model = torch.load(model_path, weights_only=False)
    
    # Generate data if needed
    if args.mode in ['generate', 'train-and-generate']:
        (num_entities, segment_size, start_date, entity_columns, sequence_index,
         timestamp_format, all_columns, log_file_path, metrics_file_path) = validate_generation_config(config, __file__)
        
        # Create output directories
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine training data path for measurement mapping
        training_data_for_mapping = data_path if args.mode == 'train-and-generate' else None
        
        # Generate data
        generate_data(
            trained_model, num_entities, log_gen, log_file_path, metrics_file_path,
            segment_size, start_date, entity_columns, sequence_index,
            timestamp_format, all_columns, training_data_for_mapping
        )
        print_data_generated(metrics_file_path, log_file_path)
        
        # Generate labels if requested
        if args.generate_labels:
            label_config = config.get('label_generation', {})
            if not label_config.get('enabled', False):
                print("\nWarning: Label generation requested but not enabled in config (label_generation.enabled = false)")
            else:
                print("\n=== Generating Anomaly Labels ===")
                label_output_path = resolve_path_from_script_dir(label_config['path'], __file__)
                timestamp_column = label_config.get('timestamp_column', sequence_index)
                label_column = label_config.get('label_column', 'anomaly_label')
                anomaly_ratio = label_config.get('anomaly_ratio', 0.3)
                min_cluster_size = label_config.get('min_cluster_size', 5)
                max_cluster_size = label_config.get('max_cluster_size', 20)
                random_seed = label_config.get('random_seed', 42)
                
                success = generate_labels_from_metrics(
                    metrics_path=metrics_file_path,
                    output_path=label_output_path,
                    timestamp_column=timestamp_column,
                    label_column=label_column,
                    anomaly_ratio=anomaly_ratio,
                    min_cluster_size=min_cluster_size,
                    max_cluster_size=max_cluster_size,
                    random_seed=random_seed
                )
                if not success:
                    print("Warning: Label generation failed")