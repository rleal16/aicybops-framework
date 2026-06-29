import importlib
import logging
import os

import aicybops_lib.torch_bootstrap  # noqa: F401 — before torch

from aicybops_lib.base_model.registry import ModelRegistry
from aicybops_lib.server.app import init_app
from aicybops_lib.server.worker import run_worker_loop
from aicybops_models.dam_model.core.dam_anomaly_detector import DAMAnomalyDetector

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _register_uc_models(model_registry: ModelRegistry) -> None:
    """Additive registration of modelos_uc demo wrappers."""
    try:
        module = importlib.import_module("aicybops_models.modelos_uc.integration")
        model_registry.register(
            os.getenv("NEXUS_XGB_MODEL_NAME", "nexus_xgb"),
            getattr(module, "ModelosUCXGBDetector"),
        )
        model_registry.register(
            os.getenv("NEXUS_AE_MODEL_NAME", "nexus_ae"),
            getattr(module, "ModelosUCAEDetector"),
        )
    except Exception as exc:
        logger.warning("Skipping modelos_uc registration: %s", exc)


def _register_modelos_uc2(model_registry: ModelRegistry) -> None:
    """Additive registration of modelos_uc_2 wrapper."""
    try:
        module = importlib.import_module("aicybops_models.modelos_uc_2.integration")
        model_registry.register(
            os.getenv("MODELOS_UC2_MODEL_NAME", "modelos_uc2"),
            getattr(module, "ModelosUC2Detector"),
        )
    except Exception as exc:
        logger.warning("Skipping modelos_uc_2 registration: %s", exc)


def register_models() -> ModelRegistry:
    model_registry = ModelRegistry()
    dam_model_name = os.getenv("DAM_MODEL_NAME", "dam")
    model_registry.register(dam_model_name, DAMAnomalyDetector)
    _register_uc_models(model_registry)
    _register_modelos_uc2(model_registry)
    return model_registry


if __name__ == "__main__":
    logger.info("Starting AICybOps worker...")
    init_app(register_models())
    run_worker_loop()
