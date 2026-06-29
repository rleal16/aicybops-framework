#!/usr/bin/env python3
"""
Script to generate logs from metrics.csv using LogGenerator.

This script reads metrics from a CSV file and generates corresponding logs
using the LogGenerator class with proper configuration.
"""

import csv
import json
import datetime
from pathlib import Path
from typing import Dict, Any, List
import argparse

from log_generator.log_generator import LogGenerator  # Generate logs from metric data using config rules
# Path utilities: resolve relative paths from config files or script directory
from utils.path_utils import (
    resolve_path_from_config,  # Resolve path from config file (relative to config file's directory)
    resolve_path_from_script_dir  # Resolve path relative to this script's directory
)


def read_metrics_csv(csv_path: Path, config: Dict[str, Any], 
                     start_time: datetime.datetime = None, 
                     time_interval: int = 1) -> List[Dict[str, Any]]:
    """
    Read metrics from CSV file and return as list of dictionaries.
    
    Args:
        csv_path: Path to the metrics CSV file
        config: Configuration dictionary containing sequence_index, timestamp_format, and entity_columns
        start_time: Starting timestamp (default: current time)
        time_interval: Seconds between each metric row (default: 1)
        
    Returns:
        List of metric rows as dictionaries with timestamps
    """
    model_config = config.get('model_training', {})
    if 'sequence_index' not in model_config:
        raise ValueError("Config must contain 'model_training.sequence_index' field")
    sequence_index = model_config['sequence_index']
    
    if 'timestamp_format' not in model_config:
        raise ValueError("Config must contain 'model_training.timestamp_format' field")
    timestamp_format = model_config['timestamp_format']
    
    if 'entity_columns' not in model_config and 'entity_column' not in model_config:
        raise ValueError("Config must contain 'model_training.entity_columns' or 'model_training.entity_column' field")
    entity_columns = model_config.get('entity_columns', [model_config.get('entity_column')])
    primary_entity_col = entity_columns[0] if entity_columns else None
    
    metrics = []
    
    if start_time is None:
        start_time = datetime.datetime.now()
    
    with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            # Convert numeric values
            processed_row = {}
            for key, value in row.items():
                if key == 'service_name':
                    processed_row[key] = value
                else:
                    # Try to convert to float for numeric metrics
                    try:
                        processed_row[key] = float(value)
                    except ValueError:
                        processed_row[key] = value
            
            # Handle timestamp using config values
            if sequence_index in processed_row and processed_row[sequence_index]:
                try:
                    # Convert timestamp based on format from config
                    ts_value = processed_row[sequence_index]
                    if timestamp_format == 'unix_ms':
                        timestamp_ms = int(float(ts_value))
                        timestamp_dt = datetime.datetime.fromtimestamp(timestamp_ms / 1000.0)
                    elif timestamp_format in ['unix', 'unix_s']:
                        timestamp_s = int(float(ts_value))
                        timestamp_dt = datetime.datetime.fromtimestamp(timestamp_s)
                    elif timestamp_format == 'unix_ns':
                        timestamp_ns = int(float(ts_value))
                        timestamp_dt = datetime.datetime.fromtimestamp(timestamp_ns / 1e9)
                    else:
                        # Try to parse as ISO format string
                        timestamp_dt = datetime.datetime.fromisoformat(str(ts_value).replace('Z', '+00:00'))
                    
                    processed_row['timestamp'] = timestamp_dt.isoformat() + 'Z'
                except (ValueError, TypeError, OSError) as e:
                    # Fallback to generated timestamp if conversion fails
                    current_time = start_time + datetime.timedelta(seconds=i * time_interval)
                    processed_row['timestamp'] = current_time.isoformat() + 'Z'
            else:
                # Add timestamp (increment by time_interval seconds for each row)
                current_time = start_time + datetime.timedelta(seconds=i * time_interval)
                processed_row['timestamp'] = current_time.isoformat() + 'Z'
            
            # Map entity column to 'service_name' using config
            # The log generator expects 'service_name' but CSV may have different entity column names
            if 'service_name' not in processed_row or not processed_row.get('service_name'):
                if primary_entity_col and primary_entity_col in processed_row and processed_row[primary_entity_col]:
                    processed_row['service_name'] = str(processed_row[primary_entity_col])
                else:
                    raise ValueError(f"Could not determine service_name: entity column '{primary_entity_col}' not found or empty in row {i+1}")
            
            metrics.append(processed_row)
    
    return metrics


def generate_logs_for_metrics(metrics: List[Dict[str, Any]], 
                            log_generator: LogGenerator,
                            output_path: Path) -> None:
    """
    Generate logs for all metrics and save to output file.
    
    Args:
        metrics: List of metric dictionaries
        log_generator: Configured LogGenerator instance
        output_path: Path to save generated logs
    """
    all_logs = []
    
    print(f"Processing {len(metrics)} metric rows...")
    
    for i, metric_row in enumerate(metrics):
        try:
            # Generate logs for this metric row
            logs = log_generator.generate_log(metric_row)
            
            if logs:
                all_logs.extend(logs)
                print(f"Generated {len(logs)} logs for row {i+1}")
            else:
                print(f"No logs generated for row {i+1}")
                
        except Exception as e:
            print(f"Error processing row {i+1}: {e}")
            continue
    
    # Save all logs to file
    with open(output_path, 'w', encoding='utf-8') as f:
        for log in all_logs:
            f.write(log + '\n')
    
    print(f"Generated {len(all_logs)} total logs")
    print(f"Logs saved to: {output_path}")


def main():
    """Main function to run the log generation process."""
    parser = argparse.ArgumentParser(description='Generate logs from metrics CSV')
    parser.add_argument('--metrics-csv', required=True, 
                       help='Path to metrics CSV file')
    parser.add_argument('--config', required=True,
                       help='Path to main config JSON file (e.g., config_main_test.json). Must contain root_config_path and output_paths.generated_logs')
    parser.add_argument('--start-time', 
                       help='Start time for timestamps (ISO format, e.g., "2024-01-01T00:00:00Z")')
    parser.add_argument('--time-interval', type=int, default=1,
                       help='Seconds between metric rows (default: 1)')
    parser.add_argument('--max-rows', type=int,
                       help='Maximum number of rows to process (for testing)')
    
    args = parser.parse_args()

    metrics_path = Path(args.metrics_csv)
    if not metrics_path.exists():
        print(f"Error: Metrics CSV file not found: {metrics_path}")
        return 1

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1
    
    with open(config_path, 'r') as f:
        config = json.load(f)

    try:
        if 'log_generation' not in config:
            print(f"Error: Config file must contain 'log_generation' section")
            return 1
        log_generator_config_path = resolve_path_from_config(
            config_path, 'log_generation.log_generator_config_path', config, must_exist=True
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        return 1

    try:
        if 'paths' not in config or 'output_paths' not in config['paths']:
            print(f"Error: Config file must contain 'paths.output_paths' section")
            return 1
        output_path = resolve_path_from_script_dir(config['paths']['output_paths']['generated_logs'], __file__)
    except (ValueError, KeyError) as e:
        print(f"Error: Config file must contain 'paths.output_paths.generated_logs' field")
        return 1
    
    try:
        # Parse start time if provided
        start_time = None
        if args.start_time:
            try:
                start_time = datetime.datetime.fromisoformat(args.start_time.replace('Z', '+00:00'))
            except ValueError:
                print(f"Error: Invalid start time format: {args.start_time}")
                return 1

        print(f"Loading log generator configuration from: {log_generator_config_path}")
        log_generator = LogGenerator(root_config_path=log_generator_config_path)
        print("LogGenerator initialized successfully")
        
        # Read metrics from CSV
        print(f"Reading metrics from: {metrics_path}")
        metrics = read_metrics_csv(metrics_path, config, start_time, args.time_interval)
        
        # Limit rows if specified
        if args.max_rows and args.max_rows < len(metrics):
            metrics = metrics[:args.max_rows]
            print(f"Limited to {args.max_rows} rows for processing")
        
        print(f"Loaded {len(metrics)} metric rows")
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate logs
        generate_logs_for_metrics(metrics, log_generator, output_path)
        
        print("Log generation completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Error during log generation: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
