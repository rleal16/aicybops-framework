#!/usr/bin/env python3
"""
DAM Model Full Pipeline Test Script

This script tests the complete DAM model pipeline from training to evaluation:
1. Data preparation and loading (with labels from config)
2. Model training (optional, with --train-first)
3. Model evaluation with comprehensive metrics
4. Results visualization and export

The script can run in two modes:
- EVALUATE ONLY: Evaluate a pre-trained model (requires --model-path)
- TRAIN + EVALUATE: Train a model from scratch, then evaluate it (--train-first)

Evaluation Features:
- Baseline evaluation with precision/recall/F1 metrics
- Threshold sensitivity analysis (testing different EVT q-values)
- Evaluation on test or validation sets
- Visualization generation (can be skipped for faster runs)
- Results export to structured JSON and markdown reports

This script properly tests the evaluation pipeline by:
- Loading models from saved checkpoints
- Passing data_dict with DataLoaders and labels
- Testing all evaluation scenarios (baseline, sensitivity, etc.)
- Verifying label handling from multiple sources
- Testing the complete evaluation pipeline end-to-end

Usage:
    # Full pipeline: Train and then evaluate
    python evaluate_dam.py --train-first --epochs 10

    # Quick full pipeline test (limited samples, faster)
    python evaluate_dam.py --train-first --epochs 2 --quick-test

    # Evaluate a pre-trained model only
    python evaluate_dam.py --model-path <path_to_model.pth>

    # Evaluate on validation set instead of test
    python evaluate_dam.py --model-path <path> --dataset-type val

    # Evaluate a pre-trained model
    python evaluate_dam.py --model-path <path>

    # Run only baseline evaluation
    python evaluate_dam.py --model-path <path> --scenarios baseline
"""

import sys
import os
import argparse
from pathlib import Path
import numpy as np
import torch

# dam_model root for imports and data paths
base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(base_dir / "processing"))

from core.dam_anomaly_detector import DAMAnomalyDetector
from utils.file_utils import find_data_files, find_model_path
from utils.config import build_data_config


def train_model(dam, args, config_path):
    """
    Train model if --train-first is specified or model doesn't exist.
    
    Args:
        dam: DAMAnomalyDetector instance
        args: Command-line arguments
        config_path: Path to config file
    
    Returns:
        Path to trained model or None
    """
    if not args.train_first:
        return None
    
    print("\n" + "=" * 80)
    print("TRAINING MODEL (--train-first specified)")
    print("=" * 80)
    
    # Prepare data config from config file with optional overrides
    print("\n[Setup] Preparing data config...")
    config_dict = build_data_config(
        str(config_path),
        overrides={
            "batch_size": args.batch_size,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "window_size": args.window_size,
            "stride": args.stride,
            "use_api": getattr(args, "use_api", False),
        },
        quick_test=args.quick_test,
    )
    if args.quick_test:
        print("  Quick test mode: Limiting samples for faster iteration")
    else:
        print("  Full training mode: Using all available data")
    
    # Training parameters
    params = {
        'learning_rate': args.learning_rate,
        'spot_type': args.spot_type,
        'risk_level': args.risk_level,
        'depth': args.depth
    }
    
    print(f"\n[Training] Starting training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Batch size: {args.batch_size}")
    
    # Train
    results = dam.train(params=params, epochs=args.epochs, data=config_dict)
    
    print("\n[Training] Training completed!")
    print(f"  Final training loss: {results.get('train_loss', 'N/A'):.4f}")
    print(f"  Final validation loss: {results.get('val_loss', 'N/A'):.4f}")
    
    # Save model for evaluation (use save_model so pipeline_state is included for evaluate-only)
    model_path = None
    
    if dam.model is not None:
        models_dir = Path(__file__).parent / "models"
        models_dir.mkdir(exist_ok=True)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = models_dir / f"dam_model_{timestamp}.pth"
        dam.save_model(str(model_path))
        print(f"  ✓ Model saved to: {model_path}")
    else:
        # Try to find the most recent model from MLflow
        mlruns_dir = Path(dam.tracking_uri.replace('file://', '')) if hasattr(dam, 'tracking_uri') else None
        if mlruns_dir and mlruns_dir.exists():
            # Find most recent run with model artifacts
            runs = sorted(mlruns_dir.glob('*/artifacts/model/*.pth'), key=lambda p: p.stat().st_mtime, reverse=True)
            if runs:
                model_path = runs[0]
                print(f"  ✓ Found model from MLflow: {model_path}")
    
    if model_path is None:
        raise RuntimeError(
            "Could not save or find model after training. Cannot proceed with evaluation. "
            "Please ensure training completes with a valid model, or provide --model-path."
        )

    return model_path


def run_evaluation(dam, args, model_path: str, config_path, data_already_prepared: bool = False):
    """
    Run the evaluation pipeline.
    
    Args:
        dam: DAMAnomalyDetector instance (for data preparation)
        args: Command-line arguments
        model_path: Path to trained model
        config_path: Path to config file
        data_already_prepared: If True, skip data preparation (data was already prepared for training)
    """
    print("\n" + "=" * 80)
    print("EVALUATION MODE")
    print("=" * 80)
    
    # Prepare data for evaluation (only if not already prepared)
    if not data_already_prepared:
        print("\n[Setup] Preparing evaluation data...")
        config_dict = build_data_config(
            str(config_path),
            overrides={
                "batch_size": args.batch_size,
                "train_ratio": args.train_ratio,
                "val_ratio": args.val_ratio,
                "window_size": args.window_size,
                "stride": args.stride,
                "use_api": getattr(args, "use_api", False),
            },
            quick_test=args.quick_test,
        )
        if args.quick_test:
            print("  Quick test mode: Limiting samples for faster iteration")
        else:
            print("  Full evaluation mode: Using all available data")
        dam._prepare_data(mode='training', data=config_dict)
    else:
        print("\n[Setup] Using data already prepared for training (consistent splits)")
    
    # Verify and report on labels
    print("\n[Labels] Checking label availability for evaluation...")
    labels = dam._last_processor.get_labels() if dam._last_processor else None
    dataset_key = f"{args.dataset_type}_labels"
    has_labels = labels is not None and dataset_key in labels and labels[dataset_key] is not None
    if has_labels:
        print(f"  ✓ {args.dataset_type.capitalize()} labels available for evaluation")
        label_array = labels[dataset_key]
        print(f"  {args.dataset_type.capitalize()} labels: {len(label_array)} sequences")
        print(f"    - Normal: {np.sum(label_array == 0)} ({100*np.mean(label_array == 0):.1f}%)")
        print(f"    - Anomalous: {np.sum(label_array == 1)} ({100*np.mean(label_array == 1):.1f}%)")
        print("  Note: Labels will be used for evaluation metrics (precision, recall, F1, etc.)")
    else:
        print(f"  ⚠ No {args.dataset_type} labels available - evaluation will use anomaly scores only")
    
    # Ensure we have data prepared - use _last_data if available, otherwise prepare
    if not hasattr(dam, '_last_data') or dam._last_data is None:
        config_dict = build_data_config(
            str(config_path),
            overrides={
                "batch_size": args.batch_size,
                "train_ratio": args.train_ratio,
                "val_ratio": args.val_ratio,
                "window_size": args.window_size,
                "stride": args.stride,
                "use_api": getattr(args, "use_api", False),
            },
            quick_test=args.quick_test,
        )
        dam._prepare_data(mode='training', data=config_dict)

    config_dict = build_data_config(
        str(config_path),
        overrides={
            "batch_size": args.batch_size,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "window_size": args.window_size,
            "stride": args.stride,
            "use_api": getattr(args, "use_api", False),
        },
        quick_test=args.quick_test,
    )
    
    # Create evaluation config (required by DAMEvaluationPipeline)
    eval_config = {
        "window_length": args.window_size,
        "stride": args.stride,
        "evt_parameters": {
            "q_values": args.q_values if args.q_values else [1e-3, 5e-3, 1e-2, 5e-2],
            "initial_threshold_quantile": 0.95,
            "min_peaks_for_fitting": 10
        },
        "max_memory_gb": 8.0
    }
    
    # Use dam.evaluate() instead of creating DAMEvaluationPipeline directly
    # This ensures threshold is extracted from training pipeline
    print(f"\n[Setup] Running evaluation via DAMAnomalyDetector.evaluate()...")
    print(f"  Using model from training (self.model)")
    print(f"  Dataset type: {args.dataset_type}")
    print("  Note: EVT threshold will be extracted from training pipeline to prevent data leakage")
    
    try:
        # Run evaluation using dam.evaluate() which handles threshold extraction
        # Note: model_path is not needed - dam.evaluate() uses self.model from training
        print("\n[Evaluation] Running evaluation pipeline...")
        results = dam.evaluate(
            data=config_dict,
            config=eval_config,
            output_dir=args.output_dir,
            dataset_type=args.dataset_type,
            device=args.device,
            require_labels=has_labels,
        )
        
        # Print summary
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS SUMMARY")
        print("=" * 80)
        
        # Print evaluation results (main metrics)
        if 'metrics' in results:
            metrics = results['metrics']
            print("\n[Evaluation Metrics]")
            print(f"  Precision: {metrics.get('precision', 'N/A'):.4f}" if isinstance(metrics.get('precision'), (int, float)) else f"  Precision: {metrics.get('precision', 'N/A')}")
            print(f"  Recall:    {metrics.get('recall', 'N/A'):.4f}" if isinstance(metrics.get('recall'), (int, float)) else f"  Recall:    {metrics.get('recall', 'N/A')}")
            print(f"  F1 Score:  {metrics.get('f1_score', 'N/A'):.4f}" if isinstance(metrics.get('f1_score'), (int, float)) else f"  F1 Score:  {metrics.get('f1_score', 'N/A')}")
            print(f"  Accuracy:  {metrics.get('accuracy', 'N/A'):.4f}" if isinstance(metrics.get('accuracy'), (int, float)) else f"  Accuracy:  {metrics.get('accuracy', 'N/A')}")
            if 'roc_auc' in metrics:
                print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
            if 'average_precision' in metrics:
                print(f"  Avg Prec:  {metrics['average_precision']:.4f}")
            if 'thresholds' in results:
                thresholds = results['thresholds']
                if len(thresholds) > 0:
                    print(f"  Threshold (Initial): {results.get('threshold_initial', 'N/A')}")
                    print(f"  Threshold (Final):   {results.get('threshold_final', 'N/A')}")
                    print(f"  Threshold (Mean):    {results.get('threshold_mean', 'N/A')}")
            if 'alarms' in results:
                print(f"  Alarms:     {len(results['alarms'])}")
            if 'num_samples' in results:
                print(f"  Samples:   {results['num_samples']}")
        
        
        if args.output_dir:
            print(f"\n[Output] Results saved to: {args.output_dir}")
        
        print("\n" + "=" * 80)
        print("✓ EVALUATION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
        return results
        
    except Exception as e:
        print(f"\n✗ ERROR during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description='DAM Model Evaluation Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate a trained model
  python evaluate_dam.py --model-path models/dam_model.pth

  # Train and then evaluate
  python evaluate_dam.py --train-first --epochs 10

  # Quick evaluation (limited samples)
  python evaluate_dam.py --model-path models/dam_model.pth --quick-test

  # Evaluate on validation set
  python evaluate_dam.py --model-path models/dam_model.pth --dataset-type val

  # Skip visualizations (faster)
  python evaluate_dam.py --model-path models/dam_model.pth

  # Run only baseline evaluation
  python evaluate_dam.py --model-path models/dam_model.pth --scenarios baseline

  # Custom threshold sensitivity analysis
  python evaluate_dam.py --model-path models/dam_model.pth --q-values 0.95 0.98 0.99
        """
    )
    
    # Model parameters
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to trained model file (required unless --train-first)')
    parser.add_argument('--train-first', action='store_true',
                       help='Train model before evaluation (default: False)')
    
    # Data parameters
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Ratio of data for training (default: 0.8)')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                       help='Ratio of training data for validation (default: 0.2)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for evaluation (default: 32)')
    parser.add_argument('--dataset-type', type=str, default='test',
                       choices=['test', 'val'],
                       help='Dataset type to evaluate (default: test)')
    
    # Model architecture parameters
    parser.add_argument('--window-size', type=int, default=10,
                       help='Window size for sequences (default: 10)')
    parser.add_argument('--stride', type=int, default=1,
                       help='Stride for sequence creation (default: 1)')
    parser.add_argument('--lstm-hidden-dim', type=int, default=32,
                       help='LSTM hidden dimension (default: 32)')
    
    # Training parameters (if --train-first)
    parser.add_argument('--epochs', type=int, default=2,
                       help='Number of training epochs (default: 2, only used with --train-first)')
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                       help='Learning rate (default: 0.0001, only used with --train-first)')
    parser.add_argument('--spot-type', type=str, default='dSPOT',
                       choices=['SPOT', 'dSPOT'],
                       help='SPOT detector type (default: dSPOT)')
    parser.add_argument('--risk-level', type=float, default=1e-3,
                       help='Risk level for EVT (default: 1e-3)')
    parser.add_argument('--depth', type=int, default=10,
                       help='Window size for dSPOT (default: 10)')
    
    # Evaluation parameters (scenarios removed - dataset contains scenarios)
    parser.add_argument('--scenarios', type=str, nargs='+',
                       default=[],
                       help='Deprecated: scenarios are embedded in the dataset (ignored)')
    parser.add_argument('--q-values', type=float, nargs='+', default=None,
                       help='Q-values for threshold sensitivity analysis (default: [0.95, 0.98, 0.99, 0.995, 0.999])')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for evaluation results (default: auto-generated)')
    
    # Quick test flag
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test mode: limit samples for faster iteration (default: False)')
    
    # MLflow parameters
    parser.add_argument('--experiment-name', type=str, default='dam_experiment',
                       help='MLflow experiment name (default: dam_experiment)')
    parser.add_argument('--mlflow-uri', type=str, default=None,
                       help='MLflow tracking URI (default: file://<script_dir>/mlruns)')
    
    # Technical parameters
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use for evaluation (default: cpu)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    parser.add_argument('--use-api', action='store_true',
                       help='Fetch data from API via DataCollector (requires API_URL)')
    parser.add_argument(
        '--config-path',
        type=str,
        default=None,
        help='Path to dam_config.json (default: configs/dam_config.json)',
    )
    
    args = parser.parse_args()
    if args.use_api and not os.getenv('API_URL'):
        print("ERROR: --use-api requires API_URL environment variable (e.g. export API_URL=http://213.30.51.238:5010)")
        sys.exit(1)
    
    print("=" * 80)
    print("DAM Model Full Pipeline Test")
    print("=" * 80)
    if args.train_first:
        print("Mode: TRAIN + EVALUATE (Full Pipeline)")
        print("  This will test the complete pipeline:")
        print("    1. Data preparation and loading")
        print("    2. Model training")
        print("    3. Model saving")
        print("    4. Model evaluation with metrics")
    else:
        print("Mode: EVALUATE ONLY")
        print("  This will test the evaluation pipeline:")
        print("    1. Model loading from checkpoint")
        print("    2. Data preparation")
        print("    3. Model evaluation with metrics")
    if args.quick_test:
        print("Quick test mode: ENABLED (limited samples)")
    if args.use_api:
        print("Data source: API (DataCollector)")
        print(f"  API_URL: {os.getenv('API_URL')}")
    print("=" * 80)
    
    base_dir = Path(__file__).resolve().parent.parent
    config_path = (
        Path(args.config_path)
        if args.config_path
        else base_dir / "configs" / "dam_config.json"
    )
    if (
        args.config_path
        and not config_path.is_absolute()
        and not config_path.exists()
    ):
        config_path = base_dir / args.config_path

    if args.use_api:
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        metrics_csv, log_file = None, None
    else:
        metrics_csv, log_file, config_path = find_data_files(
            base_dir, config_path_arg=args.config_path
        )

    # Find or train model
    model_path = None
    if args.train_first:
        pass  # Will train and get model path below
    else:
        model_path = find_model_path(base_dir, args.model_path)

    try:
        # Initialize DAMAnomalyDetector (needed for data preparation)
        print("\n[Setup] Initializing DAMAnomalyDetector...")
        tracking_uri = args.mlflow_uri or os.getenv('MLFLOW_TRACKING_URI', 
                                                     'file://' + str(base_dir / 'mlruns'))
        dam = DAMAnomalyDetector(
            experiment_name=args.experiment_name,
            config_path=str(config_path),
            tracking_uri=tracking_uri
        )
        print("  ✓ DAMAnomalyDetector initialized")
        if getattr(args, "use_api", False):
            print(f"    Note: Data will be fetched from API (DataCollector)")
        else:
            print(f"    Note: Data files will be loaded from config: {config_path}")
        
        # Train if needed, or load model for evaluate-only mode
        data_already_prepared = False
        if not args.train_first:
            # Evaluate-only: set _last_data so pipeline restoration uses current config
            config_dict = build_data_config(
                str(config_path),
                overrides={
                    "batch_size": args.batch_size,
                    "train_ratio": args.train_ratio,
                    "val_ratio": args.val_ratio,
                    "window_size": args.window_size,
                    "stride": args.stride,
                    "use_api": getattr(args, "use_api", False),
                },
                quick_test=args.quick_test,
            )
            dam._last_data = config_dict
            print(f"\n[Setup] Loading model from {model_path}...")
            dam.load_model(str(model_path), restore_pipeline=True)
            print("  ✓ Model and pipeline restored from checkpoint")
        elif args.train_first:
            # Pass config_path to train_model_if_needed
            model_path = train_model(dam, args, config_path)
            if model_path is None:
                print("\n✗ ERROR: Could not determine model path after training")
                return False
            # Data was already prepared during training, so we can reuse it
            data_already_prepared = True
        
        # Run evaluation (pass config_path for data preparation if needed)
        results = run_evaluation(dam, args, str(model_path), config_path, data_already_prepared=data_already_prepared)
        
        if results is None:
            return False
        
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

