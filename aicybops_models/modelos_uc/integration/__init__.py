"""Integration wrappers that adapt the modelos_uc demo scripts to BaseModel.

The original scripts in the parent directory (``anomalydetection_ae.py`` and
``supervised_xgboost.py``) are kept untouched; these wrappers execute them
via ``runpy`` and expose the standard AICybOps lifecycle (train/test/validate/
predict/evaluate) so they can be served through the existing endpoints.
"""

from .modelos_uc_xgb_detector import ModelosUCXGBDetector
from .modelos_uc_ae_detector import ModelosUCAEDetector

__all__ = ["ModelosUCXGBDetector", "ModelosUCAEDetector"]
