#!/usr/bin/env python3
"""
Generate anomaly labels for DAM model testing.

This script reads metrics.csv and generates point-level anomaly labels
with a clustered distribution (anomalies occur in bursts).
"""

import json
import argparse
from pathlib import Path
from utils.path_utils import resolve_path_from_script_dir
from utils.label_utils import generate_labels_from_metrics
from utils.config_utils import load_config


def main():
    """Main function to generate labels from config."""
    parser = argparse.ArgumentParser(description='Generate anomaly labels from metrics CSV')
    parser.add_argument('--config', required=True,
                       help='Path to main config JSON file (e.g., config_main_test.json)')
    parser.add_argument('--metrics-csv',
                       help='Path to metrics CSV file (overrides config if provided)')
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        return False
    
    config = load_config(config_path)

    label_config = config.get('label_generation', {})
    if not label_config.get('enabled', False):
        print("Label generation is not enabled in config (label_generation.enabled = false)")
        return False

    if args.metrics_csv:
        metrics_path = Path(args.metrics_csv)
    else:
        if 'paths' not in config or 'output_paths' not in config['paths']:
            print("ERROR: Config must contain 'paths.output_paths' section")
            return False
        output_paths = config['paths']['output_paths']
        if 'generated_metrics' not in output_paths:
            print("ERROR: Config must contain 'paths.output_paths.generated_metrics' or provide --metrics-csv")
            return False
        metrics_path = resolve_path_from_script_dir(
            output_paths['generated_metrics'], __file__, must_exist=True
        )

    if 'path' not in label_config:
        print("ERROR: Config must contain 'label_generation.path' field")
        return False
    output_path = resolve_path_from_script_dir(label_config['path'], __file__)

    model_config = config.get('model_training', {})
    timestamp_column = label_config.get('timestamp_column', model_config.get('sequence_index', '_time'))
    label_column = label_config.get('label_column', 'anomaly_label')
    anomaly_ratio = label_config.get('anomaly_ratio', 0.3)
    min_cluster_size = label_config.get('min_cluster_size', 5)
    max_cluster_size = label_config.get('max_cluster_size', 20)
    random_seed = label_config.get('random_seed', 42)
    
    # Generate labels
    return generate_labels_from_metrics(
        metrics_path=metrics_path,
        output_path=output_path,
        timestamp_column=timestamp_column,
        label_column=label_column,
        anomaly_ratio=anomaly_ratio,
        min_cluster_size=min_cluster_size,
        max_cluster_size=max_cluster_size,
        random_seed=random_seed
    )


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

