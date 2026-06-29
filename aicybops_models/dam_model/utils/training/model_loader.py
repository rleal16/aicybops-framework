from dataclasses import dataclass
from pathlib import Path
import logging
from typing import Any, Dict, Optional, Union

import torch
from pydantic import BaseModel, Field

from core.dam import DAMModel


class DAMCheckpointDimensions(BaseModel):
    """Validated dimensions from a DAM checkpoint."""

    load_metrics_dim: int = Field(..., gt=0, description="Load metrics feature dimension")
    traffic_metrics_dim: int = Field(..., gt=0, description="Traffic metrics feature dimension")
    log_seq_dim: int = Field(..., gt=0, description="Log sequence feature dimension")
    lstm_hidden_dim: int = Field(..., gt=0, description="LSTM hidden size")


@dataclass
class LoadedCheckpoint:
    """Typed loaded checkpoint: state_dict and validated dimensions."""

    state_dict: Dict[str, Any]
    dimensions: DAMCheckpointDimensions


class DAMModelLoader:
    """Load DAM checkpoints and build DAMModel instances with validated dimensions."""

    def __init__(self, device: str = "cpu", logger: Optional[logging.Logger] = None) -> None:
        self.device = device
        self.logger = logger if logger is not None else logging.getLogger(__name__)

    def load_checkpoint(self, model_path: Union[str, Path]) -> LoadedCheckpoint:
        """Load and validate a .pth checkpoint. Raises FileNotFoundError/ValueError on bad input."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path {model_path} does not exist")
        try:
            model_state = torch.load(model_path, map_location=self.device)
        except Exception as e:
            self.logger.error("Error loading DAM model: %s", e)
            raise
        if "model_state_dict" not in model_state:
            raise ValueError(
                "Invalid model file: missing 'model_state_dict'. "
                "The model file must contain 'model_state_dict' and 'dimensions'."
            )
        if "dimensions" not in model_state:
            raise ValueError(
                "Invalid model file: missing 'dimensions'. "
                "The model file must contain 'dimensions' with load_metrics_dim, traffic_metrics_dim, "
                "log_seq_dim, lstm_hidden_dim."
            )
        try:
            dimensions = DAMCheckpointDimensions(**model_state["dimensions"])
        except Exception as e:
            raise ValueError(
                f"Invalid model file: 'dimensions' validation failed: {e}. "
                "Required keys with positive integers: load_metrics_dim, traffic_metrics_dim, "
                "log_seq_dim, lstm_hidden_dim."
            ) from e
        state_dict = model_state["model_state_dict"]
        return LoadedCheckpoint(state_dict=state_dict, dimensions=dimensions)

    def build_model(
        self,
        dimensions: DAMCheckpointDimensions,
        state_dict: Dict[str, Any],
    ) -> DAMModel:
        """Build DAMModel from dimensions and state_dict, load weights, move to device."""
        model = DAMModel(
            load_metrics_dim=dimensions.load_metrics_dim,
            traffic_metrics_dim=dimensions.traffic_metrics_dim,
            log_seq_dim=dimensions.log_seq_dim,
            lstm_hidden_dim=dimensions.lstm_hidden_dim,
        )
        model.load_state_dict(state_dict)
        model.to(self.device)
        return model

    def get_model(self, model: Union[str, Path, DAMModel]) -> DAMModel:
        """
        Return DAMModel from a path or an existing DAMModel.

        If model is a path: load checkpoint, build DAMModel, load state, move to device.
        If model is DAMModel: return it as-is. Dimensions live on the model (single source of truth).
        """
        if isinstance(model, (str, Path)):
            checkpoint = self.load_checkpoint(model)
            dam_model = self.build_model(checkpoint.dimensions, checkpoint.state_dict)
            dims = dam_model.get_dimensions()
            self.logger.info(
                "Model loaded successfully - Architecture: "
                "Load(%s) + Traffic(%s) + Log(%s) -> LSTM(%s)",
                dims["load_metrics_dim"],
                dims["traffic_metrics_dim"],
                dims["log_seq_dim"],
                dims["lstm_hidden_dim"],
            )
            return dam_model
        # model is DAMModel
        dims = model.get_dimensions()
        self.logger.info(
            "Model metadata from instance - Architecture: "
            "Load(%s) + Traffic(%s) + Log(%s) -> LSTM(%s)",
            dims["load_metrics_dim"],
            dims["traffic_metrics_dim"],
            dims["log_seq_dim"],
            dims["lstm_hidden_dim"],
        )
        return model
