from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator
from pydantic import ValidationError as PydanticValidationError


class EVTParameters(BaseModel):
    """EVT (Extreme Value Theory) parameters for threshold fitting."""

    initial_threshold_quantile: float = Field(0.95, gt=0.0, lt=1.0)
    min_peaks_for_fitting: int = Field(10, ge=5)
    q_values: List[float] = Field(default_factory=list)
    risk_level: Optional[float] = Field(None, gt=0.0, le=1.0)

    @field_validator("q_values")
    @classmethod
    def q_values_in_range(cls, v: List[float]) -> List[float]:
        for q in v:
            if not (0 < q < 1):
                raise ValueError(f"Each q_value must be a float between 0 and 1, got {q}")
        return v


class EvaluationConfig(BaseModel):

    evt_parameters: EVTParameters
    max_memory_gb: float = Field(..., gt=0.0)
    window_length: Optional[int] = Field(None, ge=1)
    batch_size: Optional[int] = Field(None, ge=1)
    dam_f1_target: float = Field(default=0.8046, gt=0.0, le=1.0)
    test_scenarios: Optional[List[str]] = None

    @classmethod
    def load(cls, config: Dict) -> "EvaluationConfig":
        """Validate and return EvaluationConfig from dict. Raises ValueError on bad input."""
        if not config:
            raise ValueError("config is required and must be a non-empty dictionary")
        try:
            return cls.model_validate(config)
        except PydanticValidationError as e:
            err = e.errors()[0]
            loc = err.get("loc", ())
            loc_str = ".".join(str(x) for x in loc)
            msg = err.get("msg", "")
            if "initial_threshold_quantile" in loc_str:
                raise ValueError("initial_threshold_quantile must be between 0 and 1") from e
            if "max_memory_gb" in loc_str:
                raise ValueError("max_memory_gb must be positive") from e
            if "q_value" in loc_str or "q_values" in loc_str:
                raise ValueError("Each q_value must be a float between 0 and 1") from e
            raise ValueError(msg or str(e)) from e
