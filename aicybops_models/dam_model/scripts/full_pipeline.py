#!/usr/bin/env python3
"""
Complete DAM Model Pipeline: Optimization → Training → Evaluation

This script provides a complete end-to-end pipeline:
1. Hyperparameter optimization (including EVT q parameter)
2. Training with best parameters
3. Comprehensive evaluation

Usage:
    # Full pipeline with default settings
    python full_pipeline.py

    # Custom optimization settings
    python full_pipeline.py --optimization-trials 20 --optimization-epochs 5

    # Skip optimization (use provided parameters)
    python full_pipeline.py --skip-optimization --learning-rate 0.0001 --risk-level 1e-3

    # Quick test mode (limited data)
    python full_pipeline.py --quick-test
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch

# dam_model root for imports and data paths
base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(base_dir / "processing"))

from core.dam_anomaly_detector import DAMAnomalyDetector
from utils.training import EarlyStopping
from utils.file_utils import find_data_files
from utils.config import PipelineConfig


def run_optimization(dam : 'DAMAnomalyDetector', data, config: PipelineConfig):
    """Run hyperparameter optimization."""
    print("\n" + "=" * 80)
    print("STEP 1: HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)
    
    # Get parameter space from config
    param_space = config.get_param_space()
    
    print("\n[Optimization] Parameter space:")
    for key, values in param_space.items():
        print(f"  {key}: {values}")
    
    print(f"\n[Optimization] Running {config.optimization_trials} trials with {config.optimization_epochs} epochs each...")
    print(f"  Method: {config.optimization_method}")
    print(f"  Objective: {config.optimization_objective}")
    
    try:
        result = dam.optimize(
            param_space=param_space,
            max_evals=config.optimization_trials,
            epochs=config.optimization_epochs,
            objective=config.optimization_objective,
            method=config.optimization_method,
            data=data
        )
        best_params = result["best_params"]
        best_score = result["best_loss"]
        print(f"\n[Optimization] ✓ Completed successfully")
        print(f"  Best parameters:")
        for key, value in best_params.items():
            print(f"    {key}: {value}")
        print(f"  Best {config.optimization_objective}: {best_score:.4f}")
        
        return best_params
        
    except Exception as e:
        print(f"\n[Optimization] ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_training(dam, data, params, config: PipelineConfig):
    """Train model with given parameters."""
    print("\n" + "=" * 80)
    print("STEP 2: MODEL TRAINING")
    print("=" * 80)
    
    print(f"\n[Training] Parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # Setup early stopping
    early_stopping = None
    if config.use_early_stopping:
        early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
            mode='min'
        )
        print(f"\n[Training] Early stopping enabled:")
        print(f"  Patience: {config.early_stopping_patience}")
        print(f"  Min delta: {config.early_stopping_min_delta}")
    
    print(f"\n[Training] Starting training with {config.training_epochs} epochs...")
    
    try:
        results = dam.train(
            params=params,
            epochs=config.training_epochs,
            data=data,
            early_stopping=early_stopping
        )
        
        print(f"\n[Training] ✓ Completed successfully")
        print(f"  Final train_loss: {results.get('train_loss', 'N/A'):.4f}")
        print(f"  Final val_loss: {results.get('val_loss', 'N/A'):.4f}")
        
        return results
        
    except Exception as e:
        print(f"\n[Training] ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_evaluation(dam, data, config: PipelineConfig):
    """Run comprehensive evaluation."""
    print("\n" + "=" * 80)
    print("STEP 3: MODEL EVALUATION")
    print("=" * 80)
    
    # Get evaluation config from PipelineConfig
    eval_config = config.get_eval_config(
        window_size=data.get('window_size', 10),
        stride=data.get('stride', 1)
    )
    
    print(f"\n[Evaluation] Configuration:")
    print(f"  Dataset type: {config.eval_dataset_type}")
    print(f"  Q-values for sensitivity: {eval_config['evt_parameters']['q_values']}")
    
    try:
        # When labels are disabled (e.g. quick-test) or eval_require_labels is False, allow evaluation without labels
        require_labels = not getattr(config, "quick_test", False) and getattr(config, "eval_require_labels", True)
        results = dam.evaluate(
            data=data,
            config=eval_config,
            output_dir=config.eval_output_dir,
            dataset_type=config.eval_dataset_type,
            device=config.device,
            require_labels=require_labels,
        )
        
        print(f"\n[Evaluation] ✓ Completed successfully")
        
        # Print key metrics
        if 'metrics' in results:
            metrics = results['metrics']
            print(f"\n  Performance Metrics:")
            print(f"    Precision: {metrics.get('precision', 'N/A'):.4f}" if isinstance(metrics.get('precision'), (int, float)) else f"    Precision: {metrics.get('precision', 'N/A')}")
            print(f"    Recall:    {metrics.get('recall', 'N/A'):.4f}" if isinstance(metrics.get('recall'), (int, float)) else f"    Recall:    {metrics.get('recall', 'N/A')}")
            print(f"    F1 Score:  {metrics.get('f1_score', 'N/A'):.4f}" if isinstance(metrics.get('f1_score'), (int, float)) else f"    F1 Score:  {metrics.get('f1_score', 'N/A')}")
            print(f"    Accuracy:  {metrics.get('accuracy', 'N/A'):.4f}" if isinstance(metrics.get('accuracy'), (int, float)) else f"    Accuracy:  {metrics.get('accuracy', 'N/A')}")
            if 'roc_auc' in metrics:
                print(f"    ROC AUC:   {metrics['roc_auc']:.4f}")
            if 'average_precision' in metrics:
                print(f"    Avg Prec:  {metrics['average_precision']:.4f}")
        
        if 'thresholds' in results and len(results['thresholds']) > 0:
            print(f"\n  Threshold Information:")
            print(f"    Initial: {results.get('threshold_initial', 'N/A')}")
            print(f"    Final:   {results.get('threshold_final', 'N/A')}")
            print(f"    Mean:    {results.get('threshold_mean', 'N/A')}")
        
        if 'alarms' in results:
            print(f"\n  Anomaly Detection:")
            print(f"    Alarms detected: {len(results['alarms'])}")
            print(f"    Total samples:   {results.get('num_samples', 'N/A')}")
        
        if config.eval_output_dir:
            print(f"\n  Results saved to: {config.eval_output_dir}")
        
        return results
        
    except Exception as e:
        print(f"\n[Evaluation] ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_final_model(dam, config: PipelineConfig, best_params):
    """Save the final trained model."""
    if config.save_model:
        print("\n" + "=" * 80)
        print("SAVING FINAL MODEL")
        print("=" * 80)
        
        # Get save path from config
        base_dir = Path(__file__).resolve().parent.parent
        save_path = config.get_model_save_path(base_dir)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            dam.save_model(str(save_path))
            print(f"\n[Save] ✓ Model saved to: {save_path}")
            
            # Save optimization summary
            if best_params:
                summary_path = Path(save_path).with_suffix('.json')
                import json
                summary = {
                    'model_path': str(save_path),
                    'optimized_parameters': best_params,
                    'timestamp': timestamp
                }
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)
                print(f"  Optimization summary saved to: {summary_path}")
            
            return str(save_path)
            
        except Exception as e:
            print(f"\n[Save] ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Main pipeline function."""
    # Parse configuration
    config = PipelineConfig()
    config.print_summary()
    
    # Find data files (raises FileNotFoundError or ValueError if missing/invalid)
    base_dir = Path(__file__).resolve().parent.parent
    # Use the config path provided to PipelineConfig (so metrics/log paths match).
    metrics_csv, log_file, config_path = find_data_files(
        base_dir, config_path_arg=str(config.config_path)
    )
    
    # Initialize DAMAnomalyDetector
    print("\n[Setup] Initializing DAMAnomalyDetector...")
    tracking_uri = config.get_mlflow_uri(base_dir)
    dam = DAMAnomalyDetector(
        experiment_name=config.experiment_name,
        config_path=str(config_path),
        tracking_uri=tracking_uri
    )
    print(f"  ✓ DAMAnomalyDetector initialized")
    print(f"    Config: {config_path}")
    print(f"    MLflow: {tracking_uri}")
    
    # Prepare data dictionary using config
    print("\n[Setup] Preparing data configuration...")
    data = config.get_data_dict(str(config_path))
    if config.quick_test:
        print("  Quick test mode: Limiting samples")
    else:
        print("  Full mode: Using all available data")
    
    # STEP 1: Optimization (if not skipped)
    best_params = None
    if not config.skip_optimization:
        best_params = run_optimization(dam, data, config)
        if best_params is None:
            print("\n✗ ERROR: Optimization failed. Cannot proceed with training.")
            return 1
    else:
        # Use provided parameters from config
        print("\n[Setup] Skipping optimization, using provided parameters")
        best_params = config.get_training_params()
        print("  Parameters:")
        for key, value in best_params.items():
            print(f"    {key}: {value}")
    
    # Update window_size if it was optimized
    if 'window_size' in best_params:
        data['window_size'] = best_params['window_size']
        dam.window_size = best_params['window_size']
    
    # Update lstm_hidden_dim if it was optimized
    if 'lstm_hidden_dim' in best_params and best_params['lstm_hidden_dim'] != config.lstm_hidden_dim:
        dam.lstm_hidden_dim = best_params['lstm_hidden_dim']
        # Model will be rebuilt during training
    
    # STEP 2: Training
    train_results = run_training(dam, data, best_params, config)
    if train_results is None:
        print("\n✗ ERROR: Training failed. Cannot proceed with evaluation.")
        return 1
    
    # STEP 3: Evaluation
    eval_results = run_evaluation(dam, data, config)
    if eval_results is None:
        print("\n⚠ WARNING: Evaluation failed, but model is trained.")
        print("  You can still use the trained model for predictions.")
    
    # STEP 4: Save model
    model_path = save_final_model(dam, config, best_params)
    
    # Final summary
    print("\n" + "=" * 80)
    print("PIPELINE SUMMARY")
    print("=" * 80)
    
    if not config.skip_optimization:
        print(f"\n[Optimization]")
        print(f"  Best parameters found:")
        for key, value in best_params.items():
            print(f"    {key}: {value}")
    
    print(f"\n[Training]")
    print(f"  Final train_loss: {train_results.get('train_loss', 'N/A'):.4f}")
    print(f"  Final val_loss: {train_results.get('val_loss', 'N/A'):.4f}")
    
    if eval_results:
        print(f"\n[Evaluation]")
        if 'metrics' in eval_results:
            metrics = eval_results['metrics']
            print(f"  Precision: {metrics.get('precision', 'N/A'):.4f}" if isinstance(metrics.get('precision'), (int, float)) else f"  Precision: {metrics.get('precision', 'N/A')}")
            print(f"  Recall:    {metrics.get('recall', 'N/A'):.4f}" if isinstance(metrics.get('recall'), (int, float)) else f"  Recall:    {metrics.get('recall', 'N/A')}")
            print(f"  F1 Score:  {metrics.get('f1_score', 'N/A'):.4f}" if isinstance(metrics.get('f1_score', 'N/A'), (int, float)) else f"  F1 Score:  {metrics.get('f1_score', 'N/A')}")
    
    if model_path:
        print(f"\n[Model]")
        print(f"  Saved to: {model_path}")
    
    print("\n" + "=" * 80)
    print("✓ FULL PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
