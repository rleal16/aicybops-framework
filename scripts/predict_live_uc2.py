#!/usr/bin/env python3
"""AICybOps — modelos_uc_2 Demo Prediction."""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from service_url import resolve_aicybops_service_url

from aicybops_lib.client import AICybOpsClient
from aicybops_models.modelos_uc_2.models import MODELS

SUPPORTED_MODEL_TYPE = "modelos_uc2"


def _section(title: str) -> None:
    width = 72
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def _model_params_from_keys(model_keys: Optional[List[str]]) -> Dict[str, List[str]]:
    if not model_keys:
        return {}
    return {"model_names": model_keys}


def main() -> int:
    parser = argparse.ArgumentParser(description="Call /predict/ for modelos_uc_2 and print diagnostics summary.")
    parser.add_argument("--model-type", default=SUPPORTED_MODEL_TYPE, choices=(SUPPORTED_MODEL_TYPE,), help="Model type key (fixed: modelos_uc2).")
    parser.add_argument("--model-keys", default=None, help=f"Comma-separated modelos_uc_2 keys subset to run. Available: {', '.join(MODELS.keys())}")
    parser.add_argument("--experiment", default="ModelosUC2Demo", help="MLflow experiment name (default: ModelosUC2Demo).")
    parser.add_argument("--model-version", default="latest", help="Registered model version (default: latest).")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print full response payload.")
    args = parser.parse_args()

    model_keys = None
    if args.model_keys:
        model_keys = [k.strip() for k in args.model_keys.split(",") if k.strip()]
        unknown = [k for k in model_keys if k not in MODELS]
        if unknown:
            print(f"[ERROR] Unknown --model-keys: {unknown}. Available: {list(MODELS.keys())}")
            return 2

    model_params = _model_params_from_keys(model_keys)
    base_url = resolve_aicybops_service_url()

    _section("AICybOps — modelos_uc_2 Demo Prediction")
    print(f"  Service URL  : {base_url}")
    print(f"  Model type   : {args.model_type}")
    print(f"  Model keys   : {model_keys if model_keys else 'all'}")
    print(f"  Experiment   : {args.experiment}")

    client = AICybOpsClient(base_url=base_url)
    try:
        health = client.health_check()
        print(f"  Health       : {health.get('status', '?')}")
    except Exception as exc:
        print(f"  [ERROR] Service unreachable at {base_url}: {exc}")
        return 2

    _section("Prediction")
    t0 = time.monotonic()
    try:
        result = client.predict(
            experiment_name=args.experiment,
            model_type=args.model_type,
            registered_model_name=args.model_type,
            model_version=args.model_version,
            model_params=model_params,
        )
    except Exception as exc:
        print(f"  [ERROR] Predict call failed: {exc}")
        if hasattr(exc, "response") and exc.response is not None:
            try:
                print(f"    Response: {exc.response.text[:500]}")
            except (AttributeError, TypeError):
                pass
        traceback.print_exc()
        return 2

    elapsed = time.monotonic() - t0
    preds = result.get("predictions") or []
    diagnostics = result.get("prediction_diagnostics") or {}

    print(f"  Duration           : {elapsed:.1f}s")
    print(f"  Predictions count  : {len(preds) if hasattr(preds, '__len__') else 0}")
    print(f"  Diagnostic models  : {len(diagnostics)}")
    if diagnostics:
        print(f"  Model names        : {', '.join(diagnostics.keys())}")

    if args.verbose:
        print()
        print("  Full response:")
        print(json.dumps(result, indent=2, default=str))

    return 0


if __name__ == "__main__":
    sys.exit(main())
