import json
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np


def make_json_serializable(obj: Any) -> Any:
    """Recursively convert numpy and other non-JSON types to JSON-compatible values."""
    if isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    return obj


def find_data_files(
    base_dir: Path, config_path_arg: Optional[str] = None
) -> Tuple[Path, Path, Path]:
    """Locate and validate required data files referenced by DAM config.

    If `config_path_arg` is omitted, it defaults to `base_dir / "configs" / "dam_config.json"`.
    If provided, it's tried relative to CWD first; if it doesn't exist there, it's tried
    relative to `base_dir`.
    """
    if config_path_arg:
        candidate = Path(config_path_arg)
        # First try as provided (relative to CWD); if it doesn't exist, try
        # resolving relative to base_dir.
        if not candidate.is_absolute() and not candidate.exists():
            candidate = base_dir / candidate
        config_path = candidate
    else:
        config_path = base_dir / "configs" / "dam_config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    if "data_paths" not in config:
        raise ValueError(
            "Config file must contain 'data_paths' section with 'metrics_csv' and 'log_file'."
        )

    data_paths = config["data_paths"]
    metrics_path = data_paths.get("metrics_csv")
    log_path = data_paths.get("log_file")

    if not metrics_path or not log_path:
        raise ValueError(
            "Config file 'data_paths' must contain both 'metrics_csv' and 'log_file'."
        )

    config_dir = config_path.parent
    metrics_csv = (config_dir / metrics_path).resolve()
    log_file = (config_dir / log_path).resolve()

    if not metrics_csv.exists():
        raise FileNotFoundError(
            f"Metrics CSV file not found: {metrics_csv} (from config data_paths.metrics_csv)."
        )
    if not log_file.exists():
        raise FileNotFoundError(
            f"Log file not found: {log_file} (from config data_paths.log_file)."
        )

    return metrics_csv, log_file, config_path


def find_model_path(base_dir: Path, model_path_arg: Optional[str] = None) -> Path:
    """Find model path from argument or search common locations (models/, mlruns/, checkpoints/)."""
    if model_path_arg:
        model_path = Path(model_path_arg)
        if model_path.exists():
            return model_path
        model_path = base_dir / model_path_arg
        if model_path.exists():
            return model_path
        raise FileNotFoundError(
            f"Model path not found: {model_path_arg} (tried as absolute and relative to {base_dir})."
        )

    search_paths = [
        base_dir / "models" / "dam_model.pth",
        base_dir / "mlruns" / "**" / "artifacts" / "model" / "dam_model.pth",
        base_dir / "checkpoints" / "dam_model.pth",
    ]

    for pattern in search_paths:
        if "**" in str(pattern):
            matches = list(base_dir.glob(str(pattern.relative_to(base_dir))))
            if matches:
                return matches[0]
        elif pattern.is_file():
            return pattern

    raise FileNotFoundError(
        "Model path not found. Please provide --model-path or train a model first. "
        f"Searched: models/, mlruns/.../artifacts/model/, checkpoints/ under {base_dir}."
    )
