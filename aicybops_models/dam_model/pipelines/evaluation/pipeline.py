from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union, TYPE_CHECKING

import copy
import logging

from core.dam import DAMModel
from utils.evaluation import EvaluationConfig, EvaluationResult
from utils.training import DAMModelLoader

if TYPE_CHECKING:
    from pipelines.training import DAMPipeline


class DAMEvaluationPipeline:

    def __init__(self,
                 model: Union[str, Path, DAMModel],
                 pipeline: "DAMPipeline",
                 config: Dict,
                 output_dir: Optional[str] = None,
                 device: str = "cpu",
                 log_level: str = "INFO") -> None:
        """
        Initialize the DAM evaluation pipeline.

        Args:
            model: Trained model path or DAMModel instance.
            pipeline: Training pipeline with stream anomaly detection support.
            config: Evaluation configuration dictionary.
            output_dir: Optional output directory for evaluation results.
            device: Evaluation device.
            log_level: Logging level.
        """
        self.pipeline = pipeline
        self.evaluation_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.device = device

        self._setup_logging(log_level)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing DAM Evaluation Pipeline - Run ID: {self.evaluation_timestamp}")

        self._loader = DAMModelLoader(device=self.device, logger=self.logger)
        self.logger.info("Setting up the model")
        self.model = self._setup_model(model)

        self.config = EvaluationConfig.load(config)
        self.logger.info("Configuration loaded and validated successfully")

        self.output_dir = self._setup_output_directory(output_dir)
        self._result_builder = EvaluationResult(
            self.logger,
            self._model_metadata_dict,
            self.config.dam_f1_target,
        )
        self._initialize_evaluation_components()

        self.logger.info("DAM Evaluation Pipeline initialized successfully")

    def _setup_model(self, model: Union[str, Path, DAMModel]) -> DAMModel:
        """Normalize model input (str → Path) and return DAMModel."""
        if isinstance(model, str):
            model = Path(model)
        return self._loader.get_model(model)

    def _model_metadata_dict(self) -> Dict:
        """Return model dimensions as a dict."""
        return self.model.get_dimensions()

    @property
    def model_metadata(self) -> Dict:
        """Model dimensions as dict."""
        return self._model_metadata_dict()

    def _setup_logging(self, log_level: str) -> None:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(
                    log_dir / f"dam_evaluation_{self.evaluation_timestamp}.log"
                ),
                logging.StreamHandler(),
            ],
        )

    def _setup_output_directory(self, output_dir: Optional[str]) -> Path:
        """Setup output directory for evaluation results."""
        if output_dir is None:
            base_results_dir = Path("evaluation_results")
            base_results_dir.mkdir(exist_ok=True)
            output_dir = base_results_dir / f"evaluation_results_{self.evaluation_timestamp}"
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Output directory setup: {output_path}")
        return output_path

    def _initialize_evaluation_components(self) -> None:
        """Initialize evaluation state."""
        self._detailed_results = {}
        self.logger.info("Evaluation components initialized")

    def get_model_info(self) -> Dict:
        """Get model architecture and metadata information."""
        window_val = self.config.window_length if self.config.window_length is not None else "N/A"
        return {
            "model_metadata": self._model_metadata_dict(),
            "config_summary": {
                "evt_q_values": self.config.evt_parameters.q_values,
                "window_length": window_val,
            },
        }

    def _handle_evaluation_error(self, error: Exception) -> None:
        """Handle and log evaluation errors."""
        self.logger.error(f"Error during evaluation: {error}")

    def run_evaluation(self, data_dict: Dict, output_dir: str = None, require_labels: bool = True) -> Dict:
        """
        Run complete DAM model evaluation pipeline using stream mode.

        Args:
            data_dict: Dictionary containing preprocessed data from DAMDataProcessor (REQUIRED).
                      Must contain 'evaluation_loader' key with a DataLoader.
                      Labels can be included via 'evaluation_labels' key.
            output_dir: Directory to save results (optional)

        Returns:
            Dictionary containing evaluation results:
            {
                'evaluation_timestamp': str,
                'metrics': {...},
                'thresholds': [...],
                'alarms': [...],
                'num_samples': int,
                'model_dimensions': {...},
                ...
            }
        """
        self.logger.info("Starting evaluation pipeline...")
        try:
            self.logger.info("Using provided data_dict from DAMDataProcessor")

            if "evaluation_loader" not in data_dict:
                raise ValueError(
                    "data_dict must contain 'evaluation_loader' key with a DataLoader"
                )
            stream_loader = data_dict["evaluation_loader"]

            # Run evaluation on a copy of the detector so the pipeline's detector state is unchanged.
            evaluation_detector = copy.deepcopy(self.pipeline.anomaly_detector)
            original_detector = self.pipeline.anomaly_detector
            self.pipeline.anomaly_detector = evaluation_detector
            try:
                stream_results = self.pipeline.run_anomaly_detection_on_stream(
                    stream_loader=stream_loader,
                    plot_results=False,
                    update_with_alarm=False,
                )
            finally:
                self.pipeline.anomaly_detector = original_detector

            evaluation_results = self._result_builder.build_result(
                stream_results,
                data_dict,
                self.evaluation_timestamp,
                self.output_dir if output_dir is not None else None,
                require_labels=require_labels,
            )
            self._detailed_results = evaluation_results

            f1_score = evaluation_results["metrics"].get("f1_score", 0.0)
            self.logger.info("DAM evaluation pipeline completed successfully!")
            self.logger.info(f"Overall F1 Score: {f1_score:.3f}")
            self.logger.info(
                f"DAM Target Achieved: {evaluation_results['dam_target_achieved']}"
            )

            return evaluation_results

        except Exception as e:
            self._handle_evaluation_error(e)
            raise
