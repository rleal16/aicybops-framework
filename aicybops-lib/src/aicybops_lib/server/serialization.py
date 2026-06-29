"""JSON-serialization helpers for numpy and related types."""
import math
from typing import Any, Optional

import numpy as np


def _sanitize_float(x: float) -> Optional[float]:
    """Return a JSON-serializable float; replace inf/nan with ``None``."""
    if math.isfinite(x):
        return x
    return None


def np_encoder(obj: Any) -> Any:
    """Convert numpy types and arrays to JSON-serializable Python types."""
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return _sanitize_float(float(obj))
    elif isinstance(obj, float):
        return _sanitize_float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.ndarray, list)):
        items = obj.tolist() if isinstance(obj, np.ndarray) else obj
        return [np_encoder(item) for item in items]
    elif isinstance(obj, dict):
        return {k: np_encoder(v) for k, v in obj.items()}
    return obj
