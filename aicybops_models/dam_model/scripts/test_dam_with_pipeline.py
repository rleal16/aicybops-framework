#!/usr/bin/env python3
"""
DAM Model Development and Evaluation Script

Uses API-collected data only (DataCollector via collect_metrics_api). No data_generation.
Requires API_URL (e.g. export API_URL=http://localhost:5010).

Usage:
    # Quick test (limited samples)
    python test_dam_with_pipeline.py --quick-test

    # Full training
    python test_dam_with_pipeline.py --mode train --epochs 10

    # Evaluation only (requires trained model)
    python test_dam_with_pipeline.py --mode evaluate --model-path <path>

    # Custom parameters
    python test_dam_with_pipeline.py --epochs 5 --learning-rate 0.001 --batch-size 64
"""

import sys
import os
import argparse
from pathlib import Path
import numpy as np

# dam_model root for imports and data paths
base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(base_dir / "processing"))

from core.dam_anomaly_detector import DAMAnomalyDetector


def train_model(dam, args, config_path):
    """Train the model with specified parameters."""
    print("\n" + "=" * 80)
    print("TRAINING MODE")
    print("=" * 80)
    
    # Prepare data dictionary: always use API (no data_generation).
    print("\n[Setup] Preparing data dictionary (data from API)...")
    data = {
        'config_path': str(config_path),
        'use_api': True,
        'batch_size': args.batch_size,
        'train_ratio': args.train_ratio,
        'val_ratio': args.val_ratio,
        'window_size': args.window_size,
        'stride': args.stride,
        'align_freq': '1s'
    }
    
    if args.quick_test:
        # Quick test mode - limit samples
        print("  Quick test mode: Limiting samples for faster iteration")
        data['max_train_samples'] = 1000
        data['max_test_samples'] = 500
    else:
        # Full training mode
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
    print(f"  SPOT type: {args.spot_type}")
    print(f"  Risk level: {args.risk_level}")
    print(f"  Depth: {args.depth}")
    
    # Train (which will prepare data internally)
    results = dam.train(params=params, epochs=args.epochs, data=data)
    
    print("\n[Training] Training completed!")
    print(f"  Final training loss: {results.get('train_loss', 'N/A'):.4f}")
    print(f"  Final validation loss: {results.get('val_loss', 'N/A'):.4f}")
    
    # Run test and validate using stored data
    print("\n[Training] Running test and validate...")
    test_results = dam.test()
    val_results = dam.validate()
    
    if 'test_anomaly_scores_mean' in test_results:
        print(f"  Test anomaly score mean: {test_results['test_anomaly_scores_mean']:.4f}")
        print(f"  Test anomaly score std: {test_results['test_anomaly_scores_std']:.4f}")
    
    # Combine results
    results = {**results, **test_results, **val_results}
    
    return results


def evaluate_model(dam, args, config_path):
    """Evaluate a trained model."""
    print("\n" + "=" * 80)
    print("EVALUATION MODE")
    print("=" * 80)
    
    if args.model_path and os.path.exists(args.model_path):
        print(f"\n[Evaluation] Loading model from {args.model_path}")
        # TODO: Implement model loading if needed
        # For now, model should be trained first
        pass
    
    # Prepare data dictionary: always use API (no data_generation).
    data = {
        'config_path': str(config_path),
        'use_api': True,
        'batch_size': args.batch_size,
        'train_ratio': args.train_ratio,
        'val_ratio': args.val_ratio,
        'window_size': args.window_size,
        'stride': args.stride,
        'align_freq': '1s'
    }
    
    if dam.pipeline is None:
        print("\n[Evaluation] Model not trained. Training first...")
        params = {
            'learning_rate': args.learning_rate,
            'spot_type': args.spot_type,
            'risk_level': args.risk_level,
            'depth': args.depth
        }
        dam.train(params=params, epochs=args.epochs, data=data)
    
    # Run evaluation (uses stored data from train)
    print("\n[Evaluation] Running evaluation...")
    val_results = dam.validate()
    test_results = dam.test()
    
    print("\n[Evaluation] Evaluation completed!")
    print(f"  Validation loss: {val_results.get('validation_loss', 'N/A'):.4f}")
    print(f"  Test anomaly score mean: {test_results.get('test_anomaly_scores_mean', 'N/A'):.4f}")
    print(f"  Test anomaly score std: {test_results.get('test_anomaly_scores_std', 'N/A'):.4f}")
    print(f"  Test anomaly score min: {test_results.get('test_anomaly_scores_min', 'N/A'):.4f}")
    print(f"  Test anomaly score max: {test_results.get('test_anomaly_scores_max', 'N/A'):.4f}")
    
    return {**val_results, **test_results}


def main():
    """Main function with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description='DAM Model Development and Evaluation Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (fast, limited data)
  python test_dam_with_pipeline.py --quick-test

  # Full training
  python test_dam_with_pipeline.py --mode train --epochs 10

  # Evaluation only
  python test_dam_with_pipeline.py --mode evaluate

  # Custom hyperparameters
  python test_dam_with_pipeline.py --epochs 5 --learning-rate 0.001 --batch-size 64
        """
    )
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'evaluate', 'test'],
                       help='Operation mode: train, evaluate, or test (default: train)')
    
    # Quick test flag
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test mode: limit samples for faster iteration (default: False)')
    
    # Data parameters
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Ratio of data for training (default: 0.8)')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                       help='Ratio of training data for validation (default: 0.2)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training (default: 32)')
    
    # Model parameters
    parser.add_argument('--window-size', type=int, default=10,
                       help='Window size for sequences (default: 10)')
    parser.add_argument('--stride', type=int, default=1,
                       help='Stride for sequence creation (default: 1)')
    parser.add_argument('--lstm-hidden-dim', type=int, default=32,
                       help='LSTM hidden dimension (default: 32)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=2,
                       help='Number of training epochs (default: 2)')
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                       help='Learning rate (default: 0.0001)')
    parser.add_argument('--spot-type', type=str, default='dSPOT',
                       choices=['SPOT', 'dSPOT'],
                       help='SPOT detector type (default: dSPOT)')
    parser.add_argument('--risk-level', type=float, default=1e-3,
                       help='Risk level for EVT (default: 1e-3)')
    parser.add_argument('--depth', type=int, default=10,
                       help='Window size for dSPOT (default: 10)')
    
    # MLflow parameters
    parser.add_argument('--experiment-name', type=str, default='dam_experiment',
                       help='MLflow experiment name (default: dam_experiment)')
    parser.add_argument('--mlflow-uri', type=str, default=None,
                       help='MLflow tracking URI (default: file://<script_dir>/mlruns)')
    
    # Model loading
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to saved model for evaluation (optional)')
    
    args = parser.parse_args()
    
    # Set quick test if epochs are low (common during development)
    if args.epochs <= 2 and not args.quick_test and args.mode == 'train':
        args.quick_test = True
        print("[Info] Low epoch count detected. Enabling quick test mode for faster iteration.")
    
    print("=" * 80)
    print("DAM Model Development and Evaluation")
    print("=" * 80)
    print(f"Mode: {args.mode.upper()}")
    if args.quick_test:
        print("Quick test mode: ENABLED (limited samples)")
    print("Data source: API (collect_metrics_api); no data_generation.")
    print("=" * 80)

    base_dir = Path(__file__).resolve().parent.parent
    config_path = base_dir / "configs" / "dam_config.json"
    if not config_path.exists():
        print(f"\n✗ ERROR: Config not found: {config_path}")
        return False
    if not os.getenv("API_URL"):
        print("\n✗ ERROR: API_URL is required. Set it to the collect_metrics_api URL (e.g. export API_URL=http://localhost:5010)")
        return False

    try:
        # Initialize DAMAnomalyDetector
        print("\n[Setup] Initializing DAMAnomalyDetector...")
        tracking_uri = args.mlflow_uri or os.getenv('MLFLOW_TRACKING_URI',
                                                     'file://' + str(base_dir / 'mlruns'))
        dam = DAMAnomalyDetector(
            experiment_name=args.experiment_name,
            config_path=str(config_path),
            tracking_uri=tracking_uri
        )
        print("  ✓ DAMAnomalyDetector initialized")
        print(f"    Config: {config_path}")
        print(f"    Data: from API ({os.getenv('API_URL')})")
        
        # Run based on mode
        if args.mode == 'train':
            results = train_model(dam, args, config_path)
        elif args.mode == 'evaluate':
            results = evaluate_model(dam, args, config_path)
        elif args.mode == 'test':
            # Quick test mode (original behavior)
            results = train_model(dam, args, config_path)
            print("\n[Test] Running additional test...")
            test_results = dam.test()
            print(f"  Test results: {test_results}")
        
        print("\n" + "=" * 80)
        print("✓ OPERATION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dam_with_pipeline():
    """Legacy function for backward compatibility - calls main() with default args."""
    return main()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
