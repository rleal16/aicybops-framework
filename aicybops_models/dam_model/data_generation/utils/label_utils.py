"""
Utilities for generating anomaly labels for DAM model testing.
"""
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any


def generate_clustered_labels(
    total_points: int, 
    anomaly_ratio: float, 
    min_cluster_size: int, 
    max_cluster_size: int,
    random_seed: int
) -> np.ndarray:
    """
    Generate point-level anomaly labels with clustered distribution.
    
    Args:
        total_points: Total number of data points
        anomaly_ratio: Percentage of data points that should be anomalous
        min_cluster_size: Minimum size of anomaly clusters
        max_cluster_size: Maximum size of anomaly clusters
        random_seed: Random seed for reproducibility
    
    Returns:
        numpy array of labels (0=normal, 1=anomaly)
    """
    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Calculate number of anomalous points needed
    num_anomalies = int(total_points * anomaly_ratio)
    
    # Generate clusters until we have enough anomalies
    anomaly_indices = set()
    while len(anomaly_indices) < num_anomalies:
        # Random cluster start position
        cluster_start = random.randint(0, max(0, total_points - max_cluster_size))
        # Random cluster size
        cluster_size = random.randint(min_cluster_size, max_cluster_size)
        # Add cluster indices
        for i in range(cluster_size):
            if cluster_start + i < total_points:
                anomaly_indices.add(cluster_start + i)
            if len(anomaly_indices) >= num_anomalies:
                break
    
    # Create label array (all zeros, then set anomalies to 1)
    labels = np.zeros(total_points, dtype=int)
    labels[list(anomaly_indices)] = 1
    
    return labels


def generate_labels_from_metrics(
    metrics_path: Path,
    output_path: Path,
    timestamp_column: str,
    label_column: str,
    anomaly_ratio: float = 0.3,
    min_cluster_size: int = 5,
    max_cluster_size: int = 20,
    random_seed: int = 42
) -> bool:
    """
    Generate anomaly labels from metrics CSV file.
    
    Args:
        metrics_path: Path to metrics CSV file
        output_path: Path to save anomaly labels CSV
        timestamp_column: Name of timestamp column in metrics CSV
        label_column: Name of label column in output CSV
        anomaly_ratio: Percentage of data points that should be anomalous
        min_cluster_size: Minimum size of anomaly clusters
        max_cluster_size: Maximum size of anomaly clusters
        random_seed: Random seed for reproducibility
    
    Returns:
        True if successful, False otherwise
    """
    if not metrics_path.exists():
        print(f"ERROR: Metrics file not found: {metrics_path}")
        return False
    
    # Read metrics.csv to get row count
    print(f"Reading {metrics_path}...")
    df = pd.read_csv(metrics_path)
    total_points = len(df)
    print(f"  Found {total_points} data points")

    if timestamp_column not in df.columns:
        print(f"ERROR: Timestamp column '{timestamp_column}' not found in CSV. Found columns: {list(df.columns)}")
        return False
    
    # Generate labels with clustered distribution
    print(f"\nGenerating labels with clustered anomalies...")
    print(f"  Anomaly ratio: {anomaly_ratio*100:.1f}%")
    print(f"  Cluster size: {min_cluster_size}-{max_cluster_size} points")
    print(f"  Random seed: {random_seed}")
    
    labels = generate_clustered_labels(
        total_points=total_points,
        anomaly_ratio=anomaly_ratio,
        min_cluster_size=min_cluster_size,
        max_cluster_size=max_cluster_size,
        random_seed=random_seed
    )
    
    # Print summary statistics
    num_normal = np.sum(labels == 0)
    num_anomalous = np.sum(labels == 1)
    anomaly_percentage = (num_anomalous / total_points) * 100
    
    print(f"\nLabel Summary:")
    print(f"  Total points: {total_points}")
    print(f"  Normal: {num_normal} ({100*num_normal/total_points:.1f}%)")
    print(f"  Anomalous: {num_anomalous} ({anomaly_percentage:.1f}%)")
    
    # Save labels to CSV with timestamps
    print(f"\nSaving labels to {output_path}...")
    labels_df = pd.DataFrame({
        timestamp_column: df[timestamp_column].values,
        label_column: labels
    })
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels_df.to_csv(output_path, index=False)
    print(f"  ✓ Saved {len(labels)} labels with timestamps to {output_path}")
    
    return True


def generate_labels_from_metrics_unique_timestamps(
    metrics_path: Path,
    output_path: Path,
    timestamp_column: str = "_time",
    label_column: str = "anomaly_label",
    anomaly_ratio: float = 0.3,
    min_cluster_size: int = 5,
    max_cluster_size: int = 20,
    random_seed: int = 42,
) -> bool:
    """
    Generate anomaly labels from metrics CSV with one label per unique timestamp.

    Use this when metrics are in long format (multiple rows per timestamp, e.g. from API).
    Alignment verification requires every metric timestamp to have a corresponding label.

    Args:
        metrics_path: Path to metrics CSV file
        output_path: Path to save anomaly labels CSV
        timestamp_column: Name of timestamp column in metrics CSV
        label_column: Name of label column in output CSV
        anomaly_ratio: Fraction of time points that should be anomalous
        min_cluster_size: Minimum size of anomaly clusters
        max_cluster_size: Maximum size of anomaly clusters
        random_seed: Random seed for reproducibility

    Returns:
        True if successful, False otherwise
    """
    if not metrics_path.exists():
        print(f"ERROR: Metrics file not found: {metrics_path}")
        return False

    print(f"Reading {metrics_path} (unique timestamps only)...")
    df = pd.read_csv(metrics_path)
    if timestamp_column not in df.columns:
        print(f"ERROR: Timestamp column '{timestamp_column}' not found. Columns: {list(df.columns)}")
        return False

    # One row per unique timestamp (preserve order)
    unique_times = df[timestamp_column].drop_duplicates()
    total_points = len(unique_times)
    print(f"  Found {total_points} unique timestamps (from {len(df)} rows)")

    print(f"\nGenerating labels with clustered anomalies...")
    print(f"  Anomaly ratio: {anomaly_ratio*100:.1f}%")
    print(f"  Cluster size: {min_cluster_size}-{max_cluster_size} points")
    print(f"  Random seed: {random_seed}")

    labels = generate_clustered_labels(
        total_points=total_points,
        anomaly_ratio=anomaly_ratio,
        min_cluster_size=min_cluster_size,
        max_cluster_size=max_cluster_size,
        random_seed=random_seed,
    )

    num_normal = np.sum(labels == 0)
    num_anomalous = np.sum(labels == 1)
    print(f"\nLabel Summary:")
    print(f"  Total points: {total_points}")
    print(f"  Normal: {num_normal} ({100*num_normal/total_points:.1f}%)")
    print(f"  Anomalous: {num_anomalous} ({100*num_anomalous/total_points:.1f}%)")

    print(f"\nSaving labels to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels_df = pd.DataFrame({
        timestamp_column: unique_times.values,
        label_column: labels,
    })
    labels_df.to_csv(output_path, index=False)
    print(f"  ✓ Saved {len(labels)} labels (one per unique timestamp) to {output_path}")
    return True
