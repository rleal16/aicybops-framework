#!/usr/bin/env python3
"""Run demo prediction for `modelos_uc` models via the AICybOps API."""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from service_url import resolve_aicybops_service_url

from aicybops_lib.client import AICybOpsClient

SUPPORTED_MODEL_TYPES = ("nexus_xgb", "nexus_ae")


def _section(title: str) -> None:
    width = 72
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def _summarise_predictions(preds: List[Any]) -> Dict[str, Any]:
    if not isinstance(preds, list):
        return {"total": 0, "anomalies": 0, "normal": 0}

    total = len(preds)
    anomalies = 0
    for p in preds:
        try:
            anomalies += int(p)
        except (TypeError, ValueError):
            continue
    return {
        "total": total,
        "anomalies": anomalies,
        "normal": max(total - anomalies, 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Call /predict/ for a modelos_uc demo model and print a summary.",
    )
    parser.add_argument(
        "--model-type",
        choices=SUPPORTED_MODEL_TYPES,
        default="nexus_xgb",
        help="Demo model type to query (default: nexus_xgb).",
    )
    parser.add_argument(
        "--experiment",
        default="ModelosUCDemo",
        help="MLflow experiment name (default: ModelosUCDemo).",
    )
    parser.add_argument(
        "--model-version",
        default="latest",
        help="Registered model version (default: latest). Demo models ignore this but the field is required by the API.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print the full response payload.",
    )
    args = parser.parse_args()

    base_url = resolve_aicybops_service_url()

    _section("AICybOps — modelos_uc Demo Prediction")
    print(f"  Service URL  : {base_url}")
    print(f"  Model type   : {args.model_type}")
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
            model_params={},
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
    summary = _summarise_predictions(preds)

    print(f"  Duration     : {elapsed:.1f}s")
    print(f"  Samples      : {summary['total']}")
    print(f"  Normal       : {summary['normal']}")
    print(f"  Anomalies    : {summary['anomalies']}")

    if args.verbose:
        print()
        print("  Full response:")
        print(json.dumps(result, indent=2, default=str))

    return 1 if summary["anomalies"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
