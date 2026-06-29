#!/usr/bin/env python3
"""Run a simple requests-based smoke test for health, train, and predict."""
import argparse
import os
import sys
import time
from pathlib import Path

import requests

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from service_url import resolve_aicybops_service_url


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test for AICybOps service (health, train, predict).")
    parser.add_argument("--timeout-health", type=int, default=10, help="Timeout for health check (seconds, default 10)")
    parser.add_argument("--timeout-train", type=int, default=180, help="Timeout for train request (seconds, default 180)")
    parser.add_argument("--timeout-predict", type=int, default=300, help="Timeout for predict request (seconds, default 300)")
    args = parser.parse_args()

    base_url = resolve_aicybops_service_url()
    model_type = os.environ.get("DAM_MODEL_NAME", "dam")
    print(f"Smoke test against {base_url} (DAM)")
    print(f"  Timeouts: health={args.timeout_health}s, train={args.timeout_train}s, predict={args.timeout_predict}s")

    # 1) Health
    try:
        r = requests.get(f"{base_url}/", timeout=args.timeout_health)
        r.raise_for_status()
        assert r.json().get("status") == "ok"
        print("  Health: OK")
    except Exception as e:
        print(f"  Health: FAIL - {e}")
        return 1

    # 2) Train (wait=true, no optimization). Longer window for more sequences.
    train_data_params = {"use_api": True, "use_session_time_range": False, "start": "-600s"}
    try:
        r = requests.post(
            f"{base_url}/train/?wait=true",
            json={
                "experiment_name": "SmokeTestDAM",
                "model_type": model_type,
                "params": train_data_params,
                "epochs": 1,
                "model_params": train_data_params,
                "run_optimization": False,
            },
            timeout=args.timeout_train,
        )
        r.raise_for_status()
        data = r.json()
        assert "result" in data
        print("  Train (no opt): OK")
        ref = data.get("model_reference") or {}
        reg_name = ref.get("registered_model_name") or model_type
        version = ref.get("model_version") or "latest"
    except Exception as e:
        print(f"  Train (no opt): FAIL - {e}")
        if hasattr(e, "response") and e.response is not None:
            try:
                body = e.response.text
                if body:
                    print(f"  Response body: {body[:500]}{'...' if len(body) > 500 else ''}")
            except Exception:
                pass
        return 1

    time.sleep(2)

    # 3) Predict (full API collection + pipeline; often >60s). Same window as train.
    try:
        r = requests.post(
            f"{base_url}/predict/",
            json={
                "experiment_name": "SmokeTestDAM",
                "model_type": model_type,
                "registered_model_name": reg_name,
                "model_version": version,
                "model_params": {"use_api": True, "use_session_time_range": False, "start": "-600s"},
            },
            timeout=args.timeout_predict,
        )
        r.raise_for_status()
        pred = r.json()
        assert "predictions" in pred
        print("  Predict: OK")
    except Exception as e:
        print(f"  Predict: FAIL - {e}")
        if hasattr(e, "response") and e.response is not None:
            try:
                body = e.response.text
                if body:
                    print(f"  Response body: {body[:500]}{'...' if len(body) > 500 else ''}")
            except Exception:
                pass
        return 1

    print("All smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
