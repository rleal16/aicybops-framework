import json
import logging
import os
import sys
from pathlib import Path
from time import sleep
from typing import Any, Dict, Optional

import requests

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from service_url import resolve_aicybops_service_url

from aicybops_lib.client import AICybOpsClient, get_data_collection_counts

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('client.log', mode='w')
    ]
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def wait_for_service(base_url: str, max_retries: int = 5, retry_delay: int = 2) -> bool:
    """Wait for the service to become available."""
    logger.info(f"Attempting to connect to service at {base_url}")
    for i in range(max_retries):
        try:
            logger.debug(f"Attempt {i+1}/{max_retries} to connect to {base_url}/docs")
            response = requests.get(f"{base_url}/docs")
            if response.status_code == 200:
                logger.info("Service is available")
                return True
        except requests.ConnectionError as e:
            logger.warning(f"Connection error on attempt {i+1}: {str(e)}")
            logger.warning(f"Service not available, retrying in {retry_delay} seconds...")
            sleep(retry_delay)
    return False

def main() -> None:
    data_source = os.getenv('DATA_SOURCE', 'api')
    api_url = os.getenv('API_URL', 'http://localhost:5010')
    os.environ['DATA_SOURCE'] = data_source
    os.environ['API_URL'] = api_url
    service_url = resolve_aicybops_service_url()
    logger.info(f"Connecting to AICybOps service at: {service_url}")
    logger.info(f"Data source: {data_source}")
    if data_source == 'api':
        logger.info(f"API URL: {api_url}")
    
    client = AICybOpsClient(base_url=service_url)

    if not wait_for_service(client.base_url, max_retries=5, retry_delay=2):
        logger.error("Service not available after maximum retries")
        sys.exit(1)

    def run_model(
        name: str,
        experiment_name: str,
        model_type: str,
        train_kwargs: Dict[str, Any],
        predict_model_type: Optional[str] = None,
    ) -> bool:
        """Run train + predict for one model. Any 4xx/5xx or exception = failure."""
        predict_model_type = predict_model_type or model_type
        try:
            logger.info("=== Starting %s Model Training ===", name)
            train_result = client.train_model(
                experiment_name=experiment_name,
                model_type=model_type,
                wait=True,
                **train_kwargs
            )
            logger.info(f"{name} Training result: {json.dumps(train_result, indent=2)}")
            metrics_count, logs_count = get_data_collection_counts(train_result)
            if metrics_count is not None or logs_count is not None:
                logger.info("Data collected: %s metrics, %s logs", metrics_count or "?", logs_count or "?")
            model_ref = train_result.get("model_reference") or {}
            logger.info("Waiting for MLflow to process training results...")
            sleep(2)
            logger.info("=== Starting %s Model Prediction ===", name)
            predictions = client.predict(
                experiment_name=experiment_name,
                model_type=predict_model_type,
                registered_model_name=model_ref.get("registered_model_name") or model_type,
                model_version=model_ref.get("model_version") or "latest",
                model_params={"use_api": True},
            )
            logger.info(f"{name} Predictions: {json.dumps(predictions, indent=2)}")
            return True
        except (requests.RequestException, ValueError, RuntimeError) as e:
            logger.error("%s failed (any 4xx/5xx is a test failure): %s", name, e)
            return False

    dam_model_name = os.getenv("DAM_MODEL_NAME", "dam")

    ok = run_model(
        "DAM (no optimization)",
        "DAMClientTest",
        dam_model_name,
        {"params": {"use_api": True}, "epochs": 1, "run_optimization": False, "model_params": {"use_api": True}},
    )
    if not ok:
        logger.error("DAM without optimization failed (4xx/5xx). Test failure.")
        sys.exit(1)

    ok = run_model(
        "DAM (with optimization)",
        "DAMClientTestOpt",
        dam_model_name,
        {"params": {"use_api": True}, "epochs": 1, "run_optimization": True, "model_params": {"use_api": True}},
    )
    if not ok:
        logger.error("DAM with optimization failed (4xx/5xx). Test failure.")
        sys.exit(1)

if __name__ == "__main__":
    main()
