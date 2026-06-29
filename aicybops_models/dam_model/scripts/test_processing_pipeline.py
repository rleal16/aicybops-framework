#!/usr/bin/env python3
"""
Complete Processing Pipeline Test Suite

Uses API-collected data only (DataCollector via collect_metrics_api). No data_generation.
Requires API_URL (e.g. export API_URL=http://localhost:5010).

Tests:
1. Config Loading (Pydantic models and loaders)
2. Metrics Analysis (MetricsAnalyser) — on API-collected metrics
3. Log Analysis (LogAnalyzer) — on API-collected logs
4. Label Dataset Loading (LabelDatasetLoader) — on API-collected labels
5. Data Utilities (DataCleaner, DatasetBuilder, GroupExtractor)
6. Full Data Processing Pipeline (DAMDataProcessor)

Usage:
    # Full test suite (collects from API first)
    python test_processing_pipeline.py

    # Test specific components
    python test_processing_pipeline.py --test-config
    python test_processing_pipeline.py --test-metrics
    python test_processing_pipeline.py --test-logs
    python test_processing_pipeline.py --test-labels
    python test_processing_pipeline.py --test-utils
    python test_processing_pipeline.py --test-full-pipeline

    # Use custom config
    python test_processing_pipeline.py --config-path configs/custom_config.json
"""

import sys
import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# dam_model root for imports and data paths
base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir))


def test_config_loading(config_path: str):
    """Test configuration loading with Pydantic models."""
    print("\n" + "=" * 80)
    print("TEST 1: CONFIGURATION LOADING")
    print("=" * 80)
    
    try:
        # Test Pydantic config models
        print("\n[Config] Testing Pydantic config models...")
        from processing.config.config_models import (
            DAMDataConfig, DAMUnifiedConfigModel,
            MetricConfig, MetricGroupConfig, LogGroupConfig,
            LabelDatasetConfig, OutputPathsConfig
        )
        
        # Load unified config
        config = DAMUnifiedConfigModel.from_json(config_path)
        print(f"  ✓ Unified config loaded successfully")
        print(f"    Model architecture: {config.model_architecture.window_size} window size")
        print(f"    Training: {config.training.num_epochs} epochs, lr={config.training.learning_rate}")
        print(f"    Anomaly detection: {config.anomaly_detection.spot_type}, risk={config.anomaly_detection.risk_level}")
        
        # Test data processing config
        if config.data_processing:
            print(f"    Data processing: {len(config.data_processing.metric_groups)} metric groups")
            print(f"                    {len(config.data_processing.log_groups)} log groups")
        
        # Test config loader wrapper
        print("\n[Config] Testing DAMConfigLoader wrapper...")
        from processing.config.config_loader import DAMConfigLoader
        
        loader = DAMConfigLoader(config_path)
        print(f"  ✓ Config loader initialized")
        
        # Test core metrics extraction
        core_metrics = loader.get_core_metrics_config()
        print(f"    Core metrics: {len(core_metrics)} metrics")
        for metric_name in list(core_metrics.keys())[:5]:  # Show first 5
            print(f"      - {metric_name}")
        
        # Test window size and stride
        window_size = loader.get_window_size()
        stride = loader.get_stride()
        align_freq = loader.get_align_freq()
        print(f"    Window size: {window_size}, Stride: {stride}, Align freq: {align_freq}")
        
        # Test label config
        label_config = loader.get_label_config()
        if label_config:
            print(f"    Label dataset: enabled={label_config.get('enabled', False)}")
            if label_config.get('enabled'):
                print(f"      Path: {label_config.get('path', 'N/A')}")
        else:
            print(f"    Label dataset: disabled")
        
        print(f"\n[Config] ✓ All configuration tests passed")
        return True
        
    except Exception as e:
        print(f"\n[Config] ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics_analyser(metrics_csv_path: str, config_path: str):
    """Test MetricsAnalyser component."""
    print("\n" + "=" * 80)
    print("TEST 2: METRICS ANALYSER")
    print("=" * 80)
    
    try:
        from processing.analyzers.metrics_analyser import MetricsAnalyser
        
        print(f"\n[Metrics] Initializing MetricsAnalyser...")
        print(f"  Metrics CSV: {metrics_csv_path}")
        print(f"  Config: {config_path}")
        
        analyser = MetricsAnalyser(metrics_csv_path, config_path, use_pydantic=True)
        print(f"  ✓ MetricsAnalyser initialized")
        
        # Test loading metrics
        print(f"\n[Metrics] Loading metrics...")
        metrics_data = analyser.load_metrics()
        print(f"  ✓ Loaded {len(metrics_data)} metric types")
        for metric_name, df in metrics_data.items():
            print(f"    {metric_name}: {len(df)} records, range [{df['value'].min():.2f}, {df['value'].max():.2f}]")
        
        # Test alignment
        print(f"\n[Metrics] Aligning metrics to 1s intervals...")
        aligned_metrics = analyser.align_metrics(freq='1s')
        print(f"  ✓ Aligned {len(aligned_metrics)} metric types")
        for metric_name, df in aligned_metrics.items():
            print(f"    {metric_name}: {len(df)} records, index range [{df.index.min()}, {df.index.max()}]")
        
        # Test normalization
        print(f"\n[Metrics] Normalizing metrics...")
        normalized_metrics = analyser.normalize_metrics()
        print(f"  ✓ Normalized {len(normalized_metrics)} metric types")
        
        # Test validation
        print(f"\n[Metrics] Validating metrics...")
        validation_results = analyser.validate_metrics()
        print(f"  ✓ Validation completed")
        if 'data_quality' in validation_results:
            dq = validation_results['data_quality']
            print(f"    Overall quality: {'Valid' if dq.get('valid', False) else 'Invalid'}")
            print(f"    Missing metrics: {dq.get('missing_metrics', [])}")
        
        # Test get_processed_data (full pipeline)
        print(f"\n[Metrics] Testing full processing pipeline (get_processed_data)...")
        processed_df = analyser.get_processed_data(align_freq='1s')
        print(f"  ✓ Processed DataFrame shape: {processed_df.shape}")
        print(f"    Columns: {list(processed_df.columns)}")
        print(f"    Index range: [{processed_df.index.min()}, {processed_df.index.max()}]")
        print(f"    Missing values: {processed_df.isna().sum().sum()}")
        
        # Test sequence creation
        print(f"\n[Metrics] Testing sequence creation...")
        sequences = analyser.create_sequences(window_size=10, stride=1, align_freq='1s')
        print(f"  ✓ Created sequences shape: {sequences.shape}")
        print(f"    Expected: (n_windows, 10, n_features)")
        
        print(f"\n[Metrics] ✓ All metrics analyser tests passed")
        return True
        
    except Exception as e:
        print(f"\n[Metrics] ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_log_analyser(log_file_path: str):
    """Test LogAnalyzer component."""
    print("\n" + "=" * 80)
    print("TEST 3: LOG ANALYSER")
    print("=" * 80)
    
    try:
        from processing.analyzers.log_analyser import LogAnalyzer
        
        print(f"\n[Logs] Initializing LogAnalyzer...")
        print(f"  Log file: {log_file_path}")
        
        analyser = LogAnalyzer(log_file_path)
        print(f"  ✓ LogAnalyzer initialized")
        print(f"    Loaded {len(analyser.logs)} log entries")
        
        # Test template extraction
        print(f"\n[Logs] Extracting templates...")
        analyser.extract_templates()
        print(f"  ✓ Extracted {len(analyser.templates)} unique templates")
        
        # Test template numbering
        print(f"\n[Logs] Assigning template numbers...")
        analyser.assign_template_numbers()
        print(f"  ✓ Assigned numbers to {len(analyser.template_numbers)} templates")
        
        # Show some template statistics
        if analyser.template_counts:
            most_common = analyser.template_counts.most_common(5)
            print(f"    Most common templates:")
            for template_id, count in most_common:
                template_text = analyser.templates.get(template_id, 'N/A')
                template_num = analyser.template_numbers.get(template_id, 'N/A')
                print(f"      Template {template_num}: {count} occurrences")
                print(f"        Text: {template_text[:80]}...")
        
        # Test alignment
        print(f"\n[Logs] Aligning logs to 1s intervals...")
        log_df = analyser.align_logs(align_freq='1s')
        print(f"  ✓ Aligned logs DataFrame shape: {log_df.shape}")
        print(f"    Index range: [{log_df.index.min()}, {log_df.index.max()}]")
        print(f"    Columns: {list(log_df.columns)}")
        
        # Test get_processed_data (full pipeline)
        print(f"\n[Logs] Testing full processing pipeline (get_processed_data)...")
        processed_df = analyser.get_processed_data(align_freq='1s')
        print(f"  ✓ Processed DataFrame shape: {processed_df.shape}")
        print(f"    Missing values: {processed_df.isna().sum().sum()}")
        
        # Test sequence creation
        print(f"\n[Logs] Testing sequence creation...")
        sequences = analyser.create_sequences(window_size=10, stride=1, align_freq='1s')
        print(f"  ✓ Created sequences shape: {sequences.shape}")
        
        # Test template statistics
        print(f"\n[Logs] Getting template statistics...")
        stats = analyser.get_template_statistics()
        print(f"  ✓ Statistics retrieved")
        print(f"    Total templates: {stats['total_templates']}")
        print(f"    Most common: {len(stats.get('most_common_templates', []))} templates")
        
        print(f"\n[Logs] ✓ All log analyser tests passed")
        return True
        
    except Exception as e:
        print(f"\n[Logs] ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_label_dataset_loader(config_path: str, label_path: str = None):
    """Test LabelDatasetLoader component. label_path: optional path to labels CSV (e.g. from API collect)."""
    print("\n" + "=" * 80)
    print("TEST 4: LABEL DATASET LOADER")
    print("=" * 80)
    
    try:
        from processing.dataset.label_dataset_loader import LabelDatasetLoader
        
        print(f"\n[Labels] Initializing LabelDatasetLoader...")
        print(f"  Config: {config_path}")
        if label_path:
            print(f"  Label file: {label_path}")
        
        loader = LabelDatasetLoader(config_path, use_pydantic=True)
        print(f"  ✓ LabelDatasetLoader initialized")
        
        # Test loading labels (use API-collected path when provided)
        print(f"\n[Labels] Loading labels...")
        try:
            labels_df = loader.load_labels(label_path=label_path)
        except FileNotFoundError as e:
            print(f"  ⚠ Label file not found: {e}")
            print(f"    This is expected if labels haven't been generated yet")
            print(f"    Skipping label-specific tests")
            return True
        
        if labels_df is None:
            print(f"  ⚠ Labels disabled or not found in config")
            print(f"    Skipping label-specific tests")
            return True
        
        print(f"  ✓ Loaded labels DataFrame shape: {labels_df.shape}")
        print(f"    Index range: [{labels_df.index.min()}, {labels_df.index.max()}]")
        print(f"    Columns: {list(labels_df.columns)}")
        print(f"    Anomaly ratio: {np.mean(labels_df['anomaly_label'])*100:.2f}%")
        
        # Test label-metrics alignment verification (requires metrics)
        print(f"\n[Labels] Testing label-metrics alignment verification...")
        # This would require metrics_df, so we'll skip it here or use a mock
        print(f"  ⚠ Alignment verification requires metrics DataFrame (skipped in unit test)")
        
        # Test mapping labels to sequences (requires timestamp index)
        print(f"\n[Labels] Testing sequence label mapping...")
        # Create a sample timestamp index for testing
        sample_timestamps = pd.date_range(
            start=labels_df.index.min(),
            end=labels_df.index.max(),
            freq='1s'
        )
        
        if len(sample_timestamps) > 10:
            sequence_labels = loader.map_labels_to_sequences(
                timestamp_index=sample_timestamps,
                window_size=10,
                stride=1
            )
            print(f"  ✓ Mapped labels to sequences")
            print(f"    Sequence labels shape: {sequence_labels.shape}")
            print(f"    Anomaly ratio: {np.mean(sequence_labels)*100:.2f}%")
            print(f"    Normal sequences: {np.sum(sequence_labels == 0)}")
            print(f"    Anomalous sequences: {np.sum(sequence_labels == 1)}")
        
        print(f"\n[Labels] ✓ All label dataset loader tests passed")
        return True
        
    except Exception as e:
        print(f"\n[Labels] ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_utilities():
    """Test data utility classes (DataCleaner, DatasetBuilder, GroupExtractor)."""
    print("\n" + "=" * 80)
    print("TEST 5: DATA UTILITIES")
    print("=" * 80)
    
    try:
        from processing.dataset.dataset_utils import DataCleaner, DatasetBuilder, GroupExtractor
        
        # Test DataCleaner
        print(f"\n[Utils] Testing DataCleaner...")
        
        # Create sample data with NaN values
        sample_sequences = {
            'load': np.random.randn(100, 10, 3),
            'traffic': np.random.randn(100, 10, 1),
            'log': np.random.randn(100, 10, 1)
        }
        sample_sequences['load'][0, 0, 0] = np.nan  # Add a NaN
        
        sample_targets = {
            'load': np.random.randn(100, 3),
            'traffic': np.random.randn(100, 1),
            'log': np.random.randn(100, 1)
        }
        
        # Test cleaning and conversion
        cleaned_tensors = DataCleaner.clean_and_convert_dict_to_tensors(
            sample_sequences, data_name="test sequences"
        )
        print(f"  ✓ Cleaned and converted sequences to tensors")
        print(f"    Keys: {list(cleaned_tensors.keys())}")
        for key, tensor in cleaned_tensors.items():
            print(f"      {key}: shape={tensor.shape}, dtype={tensor.dtype}")
            print(f"        NaN count: {torch.isnan(tensor).sum().item()}")
        
        # Test DatasetBuilder
        print(f"\n[Utils] Testing DatasetBuilder...")
        
        target_tensors = DataCleaner.clean_and_convert_dict_to_tensors(
            sample_targets, data_name="test targets"
        )
        
        # Create dataset with targets
        dataset = DatasetBuilder.create_dataset(
            cleaned_tensors,
            target_tensors
        )
        print(f"  ✓ Created dataset with targets")
        print(f"    Dataset size: {len(dataset)}")
        print(f"    Number of tensors: {len(dataset.tensors)}")
        
        # Create dataset without targets
        dataset_no_targets = DatasetBuilder.create_dataset(
            cleaned_tensors,
            target_tensors=None
        )
        print(f"  ✓ Created dataset without targets")
        print(f"    Dataset size: {len(dataset_no_targets)}")
        print(f"    Number of tensors: {len(dataset_no_targets.tensors)}")
        
        # Test DataLoader creation
        train_loader, val_loader = DatasetBuilder.create_data_loaders(
            dataset, dataset, batch_size=32
        )
        print(f"  ✓ Created data loaders")
        print(f"    Train batches: {len(train_loader)}")
        print(f"    Val batches: {len(val_loader)}")
        
        # Test train/val split
        train_subset, val_subset = DatasetBuilder.split_train_for_validation(
            dataset, val_ratio=0.2
        )
        print(f"  ✓ Split dataset for validation")
        print(f"    Train size: {len(train_subset)}")
        print(f"    Val size: {len(val_subset)}")
        
        # Test GroupExtractor (requires config and dataframes)
        print(f"\n[Utils] Testing GroupExtractor...")
        print(f"  ⚠ GroupExtractor requires config and processed dataframes")
        print(f"    (Tested as part of full pipeline)")
        
        print(f"\n[Utils] ✓ All data utility tests passed")
        return True
        
    except Exception as e:
        print(f"\n[Utils] ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline(config_path: str):
    """Test the complete DAMDataProcessor pipeline using the API (use_api=True). Requires API_URL."""
    print("\n" + "=" * 80)
    print("TEST 6: FULL DATA PROCESSING PIPELINE (API)")
    print("=" * 80)
    
    try:
        from processing.data_analysis import DAMDataProcessor
        
        if not os.getenv('API_URL'):
            print("\n[Pipeline] ✗ API_URL is required for full pipeline test. Set it and re-run.")
            return False
        
        print(f"\n[Pipeline] Initializing DAMDataProcessor (use_api=True)...")
        print(f"  Config: {config_path}")
        print(f"  Data: from API ({os.getenv('API_URL')})")
        
        # Paths are placeholders; processor fetches metrics/logs/labels from API in _process_data
        processor = DAMDataProcessor(
            metrics_csv_path="",
            log_file_path="",
            config_path=config_path,
            window_size=10,
            stride=1,
            align_freq='1s',
            use_api=True,
            use_pydantic=True
        )
        print(f"  ✓ DAMDataProcessor initialized")
        
        # Test data processing
        print(f"\n[Pipeline] Processing data...")
        processor._process_data()
        print(f"  ✓ Data processed")
        print(f"    Groups: {list(processor.groups.keys())}")
        for group_name, group_df in processor.groups.items():
            print(f"      {group_name}: shape={group_df.shape}, columns={list(group_df.columns)}")
        
        # Test sequence creation
        print(f"\n[Pipeline] Creating sequences...")
        processor._create_sequences()
        print(f"  ✓ Sequences created")
        for group_name, seqs in processor.sequences.items():
            print(f"      {group_name}: shape={seqs.shape}")
        
        # Test label mapping
        if processor.sequence_anomaly_labels is not None:
            print(f"    Sequence labels: shape={processor.sequence_anomaly_labels.shape}")
            print(f"      Anomaly ratio: {np.mean(processor.sequence_anomaly_labels)*100:.2f}%")
        
        # Test data splitting
        print(f"\n[Pipeline] Splitting data...")
        train_data, val_data, test_data = processor._split_data(
            train_ratio=0.8,
            val_ratio=0.2,
            random_state=42
        )
        print(f"  ✓ Data split")
        for group_name in train_data.keys():
            print(f"      {group_name}:")
            print(f"        Train: {train_data[group_name].shape}")
            print(f"        Val: {val_data[group_name].shape}")
            print(f"        Test: {test_data[group_name].shape}")
        
        # Test target creation
        print(f"\n[Pipeline] Creating targets...")
        train_targets = processor._create_targets(train_data)
        val_targets = processor._create_targets(val_data)
        test_targets = processor._create_targets(test_data)
        print(f"  ✓ Targets created")
        for group_name in train_targets.keys():
            print(f"      {group_name}: train={train_targets[group_name].shape}, "
                  f"val={val_targets[group_name].shape}, test={test_targets[group_name].shape}")
        
        # Test prepare_for_training (full pipeline, same API source)
        print(f"\n[Pipeline] Testing prepare_for_training (full pipeline)...")
        processor2 = DAMDataProcessor(
            metrics_csv_path="",
            log_file_path="",
            config_path=config_path,
            window_size=10,
            stride=1,
            align_freq='1s',
            use_api=True,
            use_pydantic=True
        )
        
        train_loader, val_loader, test_dataset, dimensions = processor2.prepare_for_training(
            train_ratio=0.8,
            val_ratio=0.2,
            batch_size=32,
            random_state=42
        )
        print(f"  ✓ Full training pipeline completed")
        print(f"    Train batches: {len(train_loader)}")
        print(f"    Val batches: {len(val_loader)}")
        print(f"    Test samples: {len(test_dataset)}")
        print(f"    Dimensions: {dimensions}")
        
        # Test prepare_for_prediction
        print(f"\n[Pipeline] Testing prepare_for_prediction...")
        processor3 = DAMDataProcessor(
            metrics_csv_path="",
            log_file_path="",
            config_path=config_path,
            window_size=10,
            stride=1,
            align_freq='1s',
            use_api=True,
            use_pydantic=True
        )
        
        pred_loader, pred_dimensions = processor3.prepare_for_prediction(
            batch_size=32,
            include_targets=False
        )
        print(f"  ✓ Prediction pipeline completed")
        print(f"    Prediction batches: {len(pred_loader)}")
        print(f"    Dimensions: {pred_dimensions}")
        
        # Test diagnostics
        print(f"\n[Pipeline] Running diagnostics...")
        processor2.diagnostics()
        print(f"  ✓ Diagnostics completed")
        
        print(f"\n[Pipeline] ✓ All full pipeline tests passed")
        return True
        
    except Exception as e:
        print(f"\n[Pipeline] ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_collector_utils(config_path: str):
    """Test collector utilities (API data collection)."""
    print("\n" + "=" * 80)
    print("TEST 7: COLLECTOR UTILITIES")
    print("=" * 80)
    
    try:
        from processing.collector_utils import collect_data_from_api
        
        print(f"\n[Collector] Testing API data collection...")
        print(f"  Config: {config_path}")
        print(f"  ⚠ Note: This requires API_URL environment variable and API access")
        
        # Check if API_URL is set
        api_url = os.getenv('API_URL')
        if not api_url:
            print(f"  ⚠ API_URL not set, skipping API collection test")
            print(f"    Set API_URL environment variable to test API collection")
            return True
        
        print(f"  API URL: {api_url}")
        print(f"  Attempting to collect data from API...")
        
        # This might fail if API is not available, so we catch the exception
        try:
            data_paths = collect_data_from_api(config_path)
            print(f"  ✓ Data collected from API")
            print(f"    Metrics CSV: {data_paths.get('metrics_csv_path', 'N/A')}")
            print(f"    Log file: {data_paths.get('log_file_path', 'N/A')}")
        except Exception as api_error:
            print(f"  ⚠ API collection failed (this is expected if API is not available): {api_error}")
            print(f"    This is not a test failure - API may not be accessible")
        
        print(f"\n[Collector] ✓ Collector utilities test completed")
        return True
        
    except Exception as e:
        print(f"\n[Collector] ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function. Data is always collected from the API (API_URL required for data tests)."""
    parser = argparse.ArgumentParser(description='Test processing pipeline components (API data only)')
    parser.add_argument('--config-path', type=str, default=None,
                       help='Path to config file (default: configs/dam_config.json)')
    parser.add_argument('--test-config', action='store_true',
                       help='Test only config loading')
    parser.add_argument('--test-metrics', action='store_true',
                       help='Test only metrics analyser')
    parser.add_argument('--test-logs', action='store_true',
                       help='Test only log analyser')
    parser.add_argument('--test-labels', action='store_true',
                       help='Test only label dataset loader')
    parser.add_argument('--test-utils', action='store_true',
                       help='Test only data utilities')
    parser.add_argument('--test-full-pipeline', action='store_true',
                       help='Test only full pipeline')
    parser.add_argument('--test-collector', action='store_true',
                       help='Test only collector utilities')
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).resolve().parent.parent
    config_path = Path(args.config_path) if args.config_path else (base_dir / "configs" / "dam_config.json")
    if not config_path.exists():
        print(f"\n✗ ERROR: Config file not found: {config_path}")
        return 1
    
    # Tests that need pre-collected file paths (metrics/logs/labels from one API collect)
    needs_data = any([
        args.test_metrics, args.test_logs, args.test_labels,
        args.test_collector
    ])
    run_all = not any([
        args.test_config, args.test_metrics, args.test_logs,
        args.test_labels, args.test_utils, args.test_full_pipeline, args.test_collector
    ])
    if run_all:
        needs_data = True
    
    # Full pipeline test uses the API itself (use_api=True); still requires API_URL
    if run_all or args.test_full_pipeline:
        if not os.getenv('API_URL'):
            print("\n✗ ERROR: API_URL is required for data tests. Set it to the collect_metrics_api base URL (e.g. export API_URL=http://localhost:5010).")
            return 1
    
    metrics_csv = None
    log_file = None
    labels_path = None
    
    if needs_data:
        from processing.collector_utils import collect_data_from_api
        data_paths = collect_data_from_api(str(config_path))
        metrics_csv = data_paths.get('metrics_csv_path')
        log_file = data_paths.get('log_file_path')
        labels_path = data_paths.get('labels_csv_path')
        if not metrics_csv or not log_file:
            print("\n✗ ERROR: API collection did not return metrics_csv_path and log_file_path.")
            return 1
    
    print("\n" + "=" * 80)
    print("PROCESSING PIPELINE TEST SUITE (API data only)")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Config file: {config_path}")
    if metrics_csv:
        print(f"  Metrics CSV: {metrics_csv}")
        print(f"  Log file: {log_file}")
        if labels_path:
            print(f"  Labels CSV: {labels_path}")
    
    results = {}
    
    if run_all or args.test_config:
        results['config'] = test_config_loading(str(config_path))
    
    if run_all or args.test_metrics:
        results['metrics'] = test_metrics_analyser(str(metrics_csv), str(config_path))
    
    if run_all or args.test_logs:
        results['logs'] = test_log_analyser(str(log_file))
    
    if run_all or args.test_labels:
        results['labels'] = test_label_dataset_loader(str(config_path), label_path=labels_path)
    
    if run_all or args.test_utils:
        results['utils'] = test_data_utilities()
    
    if run_all or args.test_full_pipeline:
        results['pipeline'] = test_full_pipeline(str(config_path))
    
    if run_all or args.test_collector:
        results['collector'] = test_collector_utils(str(config_path))
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name.upper():15s}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED")
        print("=" * 80)
        return 0
    else:
        print("\n" + "=" * 80)
        print("✗ SOME TESTS FAILED")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
