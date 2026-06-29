#!/usr/bin/env python3
"""Control server-side DAM monitoring through `/monitor` endpoints."""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from service_url import resolve_aicybops_service_url


def _print_json(label: str, payload: Dict[str, Any]) -> None:
    print(f"{label}:")
    print(json.dumps(payload, indent=2, default=str))


def _post(base_url: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    response = requests.post(f"{base_url}{path}", json=payload or {})
    response.raise_for_status()
    return response.json()


def _get(base_url: str, path: str) -> Dict[str, Any]:
    response = requests.get(f"{base_url}{path}")
    response.raise_for_status()
    return response.json()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Control server-side continuous DAM monitoring via /monitor endpoints."
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="LiveMonitoring",
        help="Experiment name passed to monitor start (default: LiveMonitoring).",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=os.environ.get("DAM_MODEL_NAME", "dam"),
        help="Model type key (default: DAM_MODEL_NAME env or 'dam').",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="latest",
        help="Model Registry version to monitor with (default: latest).",
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=60,
        help="Server-side monitor loop interval in seconds (default: 60).",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=10.0,
        help="Client-side polling interval for /monitor/status and /monitor/alarms (default: 10).",
    )
    parser.add_argument(
        "--no-stop-on-exit",
        action="store_true",
        help="Do not call /monitor/stop when this script exits.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full JSON payloads.",
    )
    args = parser.parse_args()

    base_url = resolve_aicybops_service_url().rstrip("/")

    print("AICybOps — Server-side Live Monitoring")
    print(f"Service        : {base_url}")
    print(f"Experiment     : {args.experiment_name}")
    print(f"Model          : {args.model_type}  version={args.model_version}")
    print(f"Server interval: {args.interval_seconds}s")
    print(f"Poll interval  : {args.poll_interval:.1f}s")

    start_payload = {
        "experiment_name": args.experiment_name,
        "model_type": args.model_type,
        "model_params": {},
        "registered_model_name": args.model_type,
        "model_version": args.model_version,
        "interval_seconds": args.interval_seconds,
    }

    started_here = False
    try:
        try:
            start_resp = _post(base_url, "/monitor/start", start_payload)
            started_here = True
            print("\n[OK] Monitor started")
            if args.verbose:
                _print_json("Start response", start_resp)
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 409:
                print("\n[INFO] Monitor already running; attaching to existing loop")
            else:
                raise

        print("\nPolling monitor status/alarms (Ctrl+C to stop)...")
        while True:
            status = _get(base_url, "/monitor/status")
            alarms = _get(base_url, "/monitor/alarms")

            running = status.get("running")
            last_alarm_at = status.get("last_alarm_at")
            err = status.get("error")
            alarm_items = alarms.get("alarms")
            alarm_count = len(alarm_items) if isinstance(alarm_items, list) else None

            print(
                f"- running={running} "
                f"last_alarm_at={last_alarm_at} "
                f"alarms={alarm_count if alarm_count is not None else 'n/a'}"
            )
            if err:
                print(f"  error={err}")

            if args.verbose:
                _print_json("Status", status)
                _print_json("Alarms", alarms)

            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("\nStopping monitor session...")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        return 1
    finally:
        if not args.no_stop_on_exit and started_here:
            try:
                stop_resp = _post(base_url, "/monitor/stop")
                if args.verbose:
                    _print_json("Stop response", stop_resp)
                else:
                    print("[OK] Monitor stopped")
            except Exception as stop_err:
                print(f"[WARN] Could not stop monitor: {stop_err}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
