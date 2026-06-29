import aicybops_lib.torch_bootstrap  # noqa: F401 — before torch

import uvicorn
from aicybops_lib.server.app import init_app
from aicybops_lib.base_model.registry import ModelRegistry
from aicybops_models.dam_model.core.dam_anomaly_detector import DAMAnomalyDetector
import importlib
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def configure_cpu_parallelism() -> None:
    """Apply optional PyTorch thread caps from environment for inference service."""
    if torch is None:
        return

    num_threads = os.getenv("TORCH_NUM_THREADS")
    if num_threads:
        try:
            torch.set_num_threads(int(num_threads))
        except ValueError:
            logger.warning("Invalid TORCH_NUM_THREADS=%s", num_threads)

    num_interop_threads = os.getenv("TORCH_NUM_INTEROP_THREADS")
    if num_interop_threads:
        try:
            torch.set_num_interop_threads(int(num_interop_threads))
        except ValueError:
            logger.warning("Invalid TORCH_NUM_INTEROP_THREADS=%s", num_interop_threads)

    logger.info(
        "Torch CPU threads configured for service: intraop=%s interop=%s",
        torch.get_num_threads(),
        torch.get_num_interop_threads(),
    )


def _register_modelos_uc(model_registry: ModelRegistry) -> None:
    """Additive registration of modelos_uc demo wrappers.

    Failure to import does not affect DAM registration; we log and continue.
    """
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
    logger.info("Registering models...")
    logger.debug("Available models before registration: %s", model_registry.list())
    
    dam_model_name = os.getenv('DAM_MODEL_NAME', 'dam')
    model_registry.register(dam_model_name, DAMAnomalyDetector)
    _register_modelos_uc(model_registry)
    _register_modelos_uc2(model_registry)
    
    logger.info("Available models after registration: %s", model_registry.list())
    return model_registry

configure_cpu_parallelism()
model_registry = register_models()

@asynccontextmanager
async def lifespan(app: Any):
    logger.info("Server starting up...")
    logger.info("Available models: %s", app.state.model_registry.list())
    yield
    logger.info("Server shutting down...")

app = init_app(model_registry, lifespan=lifespan)

if __name__ == "__main__":
    logger.info("Starting server...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug"
    ) 