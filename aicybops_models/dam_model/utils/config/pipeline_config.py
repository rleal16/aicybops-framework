import argparse
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from .config_loader import DAMUnifiedConfig


class PipelineConfig:
    def __init__(self, args: Optional[argparse.Namespace] = None):
        if args is None:
            args = self._parse_arguments()

        self.args = args
        self.config_path = args.config_path
        if not self.config_path:
            raise ValueError(
                "config_path is required. Provide path to dam_config.json file.\n"
                "Example: --config-path configs/dam_config.json"
            )
        
        try:
            self._unified_config = DAMUnifiedConfig(config_path=self.config_path)
        except (FileNotFoundError, ValueError) as e:
            raise ValueError(
                f"Failed to load required config file '{self.config_path}': {e}\n"
                "Please ensure dam_config.json exists and contains all required sections."
            ) from e
        
        training = self._unified_config.get_training()
        arch = self._unified_config.get_model_architecture()
        self.train_ratio = args.train_ratio if args.train_ratio is not None else training.get('train_ratio')
        self.val_ratio = args.val_ratio if args.val_ratio is not None else training.get('val_ratio')
        self.batch_size = args.batch_size if args.batch_size is not None else training.get('batch_size')
        self.window_size = args.window_size if args.window_size is not None else arch.get('window_size')
        self.stride = args.stride if args.stride is not None else arch.get('stride')
        self.quick_test = args.quick_test
        self.skip_optimization = args.skip_optimization
        self.optimization_trials = args.optimization_trials
        self.optimization_epochs = args.optimization_epochs
        self.optimization_method = args.optimization_method
        self.optimization_objective = args.optimization_objective
        self.learning_rate_space = args.learning_rate_space
        self.risk_level_space = args.risk_level_space
        self.lstm_hidden_dim_space = args.lstm_hidden_dim_space
        self.window_size_space = args.window_size_space
        self.spot_type_space = args.spot_type_space
        self.depth_space = args.depth_space
        training = self._unified_config.get_training()
        anomaly = self._unified_config.get_anomaly_detection()
        arch = self._unified_config.get_model_architecture()
        self.learning_rate = args.learning_rate if args.learning_rate is not None else training.get('learning_rate')
        self.lstm_hidden_dim = args.lstm_hidden_dim if args.lstm_hidden_dim is not None else arch.get('lstm_hidden_dim')
        self.risk_level = args.risk_level if args.risk_level is not None else anomaly.get('risk_level')
        self.spot_type = args.spot_type if args.spot_type is not None else anomaly.get('spot_type')
        self.depth = args.depth if args.depth is not None else anomaly.get('depth')
        self.training_epochs = args.training_epochs if args.training_epochs is not None else training.get('num_epochs')
        es_config = training.get('early_stopping', {})
        self.use_early_stopping = args.use_early_stopping if args.use_early_stopping is not None else es_config.get('enabled', True)
        self.early_stopping_patience = args.early_stopping_patience if args.early_stopping_patience is not None else es_config.get('patience')
        self.early_stopping_min_delta = args.early_stopping_min_delta if args.early_stopping_min_delta is not None else es_config.get('min_delta')
        self.eval_dataset_type = args.eval_dataset_type
        self.eval_q_values = args.eval_q_values
        self.eval_output_dir = args.eval_output_dir
        self.save_model = args.save_model
        self.model_save_path = args.model_save_path
        self.experiment_name = args.experiment_name
        self.mlflow_uri = args.mlflow_uri
        self.device = args.device
    
    @staticmethod
    def _parse_arguments() -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description='Complete DAM Model Pipeline: Optimization → Training → Evaluation',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Full pipeline with defaults
  python full_pipeline.py

  # Custom optimization
  python full_pipeline.py --optimization-trials 30 --optimization-epochs 3

  # Skip optimization, use provided parameters
  python full_pipeline.py --skip-optimization --learning-rate 0.0001 --risk-level 1e-3

  # Quick test (limited data)
  python full_pipeline.py --quick-test --optimization-trials 5

  # Full pipeline with custom paths
  python full_pipeline.py --save-model --model-save-path models/my_model.pth
            """
        )
        
        # Data parameters
        parser.add_argument('--config-path', type=str, required=True,
                           help='Path to config file (required: e.g., configs/dam_config.json)')
        parser.add_argument('--train-ratio', type=float, default=0.8,
                           help='Ratio of data for training (default: 0.8)')
        parser.add_argument('--val-ratio', type=float, default=0.2,
                           help='Ratio of training data for validation (default: 0.2)')
        parser.add_argument('--batch-size', type=int, default=32,
                           help='Batch size (default: 32)')
        parser.add_argument('--window-size', type=int, default=10,
                           help='Window size for sequences (default: 10)')
        parser.add_argument('--stride', type=int, default=1,
                           help='Stride for sequence creation (default: 1)')
        parser.add_argument('--quick-test', action='store_true',
                           help='Quick test mode: limit samples for faster iteration')
        
        # Optimization parameters
        parser.add_argument('--skip-optimization', action='store_true',
                           help='Skip hyperparameter optimization, use provided parameters')
        parser.add_argument('--optimization-trials', type=int, default=20,
                           help='Number of optimization trials (default: 20)')
        parser.add_argument('--optimization-epochs', type=int, default=3,
                           help='Epochs per optimization trial (default: 3)')
        parser.add_argument('--optimization-method', type=str, default='random',
                           choices=['random', 'grid'],
                           help='Optimization method (default: random)')
        parser.add_argument('--optimization-objective', type=str, default='val_loss',
                           choices=['val_loss', 'f1_score', 'test_anomaly_scores_mean'],
                           help='Objective metric for optimization (default: val_loss)')
        
        # Parameter spaces (for optimization)
        parser.add_argument('--learning-rate-space', type=float, nargs='+', default=None,
                           help='Learning rate search space (default: [1e-5, 5e-5, 1e-4, 5e-4, 1e-3])')
        parser.add_argument('--risk-level-space', type=float, nargs='+', default=None,
                           help='Risk level (EVT q) search space (default: [1e-4, 5e-4, 1e-3, 5e-3, 1e-2])')
        parser.add_argument('--lstm-hidden-dim-space', type=int, nargs='+', default=None,
                           help='LSTM hidden dim search space (default: [32, 64, 128])')
        parser.add_argument('--window-size-space', type=int, nargs='+', default=None,
                           help='Window size search space (default: [5, 10, 15, 20])')
        parser.add_argument('--spot-type-space', type=str, nargs='+', default=None,
                           help='SPOT type search space (default: [SPOT, dSPOT])')
        parser.add_argument('--depth-space', type=int, nargs='+', default=None,
                           help='Depth search space (default: [5, 10, 15, 20])')
        
        # Training parameters (used if skip-optimization)
        parser.add_argument('--learning-rate', type=float, default=0.0001,
                           help='Learning rate (default: 0.0001, used if skip-optimization)')
        parser.add_argument('--lstm-hidden-dim', type=int, default=32,
                           help='LSTM hidden dimension (default: 32)')
        parser.add_argument('--risk-level', type=float, default=1e-3,
                           help='Risk level for EVT (default: 1e-3, used if skip-optimization)')
        parser.add_argument('--spot-type', type=str, default='dSPOT',
                           choices=['SPOT', 'dSPOT'],
                           help='SPOT detector type (default: dSPOT)')
        parser.add_argument('--depth', type=int, default=10,
                           help='Window size for dSPOT (default: 10)')
        parser.add_argument('--training-epochs', type=int, default=10,
                           help='Number of training epochs (default: 10)')
        parser.add_argument('--use-early-stopping', action='store_true', default=True,
                           help='Use early stopping during training (default: True)')
        parser.add_argument('--early-stopping-patience', type=int, default=5,
                           help='Early stopping patience (default: 5)')
        parser.add_argument('--early-stopping-min-delta', type=float, default=0.001,
                           help='Early stopping min delta (default: 0.001)')
        
        # Evaluation parameters
        parser.add_argument('--eval-dataset-type', type=str, default='test',
                           choices=['test', 'val'],
                           help='Dataset type for evaluation (default: test)')
        parser.add_argument('--eval-q-values', type=float, nargs='+', default=None,
                           help='Q-values for threshold sensitivity analysis')
        parser.add_argument('--eval-output-dir', type=str, default=None,
                           help='Output directory for evaluation results (default: auto-generated)')
        
        # Model saving
        parser.add_argument('--save-model', action='store_true', default=True,
                           help='Save final trained model (default: True)')
        parser.add_argument('--model-save-path', type=str, default=None,
                           help='Path to save final model (default: auto-generated with timestamp)')
        
        # MLflow parameters
        parser.add_argument('--experiment-name', type=str, default='dam_full_pipeline',
                           help='MLflow experiment name (default: dam_full_pipeline)')
        parser.add_argument('--mlflow-uri', type=str, default=None,
                           help='MLflow tracking URI (default: file://<script_dir>/mlruns)')
        
        # Technical parameters
        parser.add_argument('--device', type=str, default='cpu',
                           choices=['cpu', 'cuda'],
                           help='Device to use (default: cpu)')
        
        return parser.parse_args()
    
    def get_param_space(self) -> Dict[str, List[Any]]:
        return {
            'learning_rate': self.learning_rate_space if self.learning_rate_space else [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
            'lstm_hidden_dim': self.lstm_hidden_dim_space if self.lstm_hidden_dim_space else [32, 64, 128],
            'window_size': self.window_size_space if self.window_size_space else [5, 10, 15, 20],
            'risk_level': self.risk_level_space if self.risk_level_space else [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
            'spot_type': self.spot_type_space if self.spot_type_space else ['SPOT', 'dSPOT'],
            'depth': self.depth_space if self.depth_space else [5, 10, 15, 20],
            'init_quantile': getattr(self, 'init_quantile_space', None) or [0.80, 0.85, 0.90, 0.95],
        }
    
    def get_training_params(self) -> Dict[str, Any]:
        return {
            'learning_rate': self.learning_rate,
            'lstm_hidden_dim': self.lstm_hidden_dim,
            'risk_level': self.risk_level,
            'spot_type': self.spot_type,
            'depth': self.depth
        }
    
    def get_data_dict(self, config_path: str) -> Dict[str, Any]:
        data = {
            'config_path': config_path,
            'use_api': False,
            'batch_size': self.batch_size,
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'window_size': self.window_size,
            'stride': self.stride,
            'align_freq': '1s'
        }
        
        if self.quick_test:
            data['max_train_samples'] = 1000
            data['max_test_samples'] = 500
        
        return data
    
    def get_eval_config(self, window_size: int, stride: int) -> Dict[str, Any]:
        return {
            "window_length": window_size,
            "stride": stride,
            "evt_parameters": {
                "q_values": self.eval_q_values if self.eval_q_values else [1e-3, 5e-3, 1e-2, 5e-2],
                "initial_threshold_quantile": 0.95,
                "min_peaks_for_fitting": 10
            },
            "max_memory_gb": 8.0
        }
    
    def get_mlflow_uri(self, base_dir: Path) -> str:
        return self.mlflow_uri or os.getenv('MLFLOW_TRACKING_URI', 
                                             'file://' + str(base_dir / 'mlruns'))
    
    def get_model_save_path(self, base_dir: Path) -> Path:
        if self.model_save_path:
            return Path(self.model_save_path)
        
        models_dir = base_dir / "models"
        models_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return models_dir / f"dam_model_optimized_{timestamp}.pth"
    
    def print_summary(self):
        print("=" * 80)
        print("PIPELINE CONFIGURATION")
        print("=" * 80)
        
        if self.skip_optimization:
            print("Mode: Training + Evaluation (optimization skipped)")
        else:
            print("Mode: Full pipeline (optimization enabled)")
            print(f"  Optimization trials: {self.optimization_trials}")
            print(f"  Optimization epochs per trial: {self.optimization_epochs}")
            print(f"  Optimization method: {self.optimization_method}")
            print(f"  Optimization objective: {self.optimization_objective}")
        
        print(f"\nTraining:")
        print(f"  Epochs: {self.training_epochs}")
        print(f"  Early stopping: {self.use_early_stopping}")
        if self.use_early_stopping:
            print(f"    Patience: {self.early_stopping_patience}")
            print(f"    Min delta: {self.early_stopping_min_delta}")
        
        print(f"\nEvaluation:")
        print(f"  Dataset type: {self.eval_dataset_type}")
        
        print(f"\nModel saving: {self.save_model}")
        if self.save_model and self.model_save_path:
            print(f"  Save path: {self.model_save_path}")
        
        if self.quick_test:
            print("\nQuick test mode: ENABLED (limited samples)")
        
        print("=" * 80)
