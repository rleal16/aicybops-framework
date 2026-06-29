#!/usr/bin/env python3
"""Run live DAM prediction and optional session-log validation."""

import argparse
import json
import os
import sys
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from service_url import resolve_aicybops_service_url

from aicybops_lib.client import AICybOpsClient

try:
    import requests as requests_lib
except ImportError:
    requests_lib = None

DEFAULT_PREDICT_DATA_PARAMS: Dict[str, Any] = {
    "use_api": True,
    "use_session_time_range": False,  # live data — no testbed session required
    "start": "-600s",                 # overridden by --start at runtime
}

def _section(title: str) -> None:
    width = 72
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _utcnow_str() -> str:
    return _utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_start_offset(start: str) -> timedelta:
    """Parse a start offset like '-600s' into a timedelta."""
    s = start.strip()
    if s.endswith("s"):
        s = s[:-1]
    return timedelta(seconds=int(s))


def _parse_iso_utc(ts: str) -> Optional[datetime]:
    """Parse an ISO timestamp string to a UTC-aware datetime, or None on failure."""
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _fetch_session_log_api(api_url: str) -> Optional[Dict]:
    """Fetch session log from collect_metrics API (/session_log). Returns None on failure."""
    if requests_lib is None:
        print("  [WARN] requests package not installed")
        return None
    base = api_url.rstrip("/")
    try:
        # Authenticate
        r = requests_lib.post(
            f"{base}/login",
            json={
                "username": os.environ.get("API_USER", "test_user_api"),
                "password": os.environ.get("API_PASS", "test_password_api"),
            },
            timeout=10,
        )
        r.raise_for_status()
        token = r.json().get("access_token")
        if not token:
            print(f"  [WARN] API login returned no access_token")
            return None
        r2 = requests_lib.get(
            f"{base}/session_log",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        r2.raise_for_status()
        data = r2.json()
        if isinstance(data, dict) and "normal_windows" in data:
            return data
        print(f"  [WARN] API session_log missing 'normal_windows' key")
        return None
    except Exception as e:
        print(f"  [WARN] Could not fetch session log from API ({api_url}): {e}")
        return None


def _get_session_log(collect_api_url: Optional[str]) -> Optional[Dict]:
    """Fetch session log from collect_metrics API. Returns session log dict or None."""
    if collect_api_url:
        log = _fetch_session_log_api(collect_api_url)
        if log is not None:
            return log
    return None


def _ground_truth_for_window(
    session_log: Dict,
    window_start: datetime,
    window_end: datetime,
) -> Dict[str, Any]:
    """
    Check what the session log says happened during [window_start, window_end].

    Returns a dict with:
      covered          bool  — whether the session log covers this window at all
      has_anomaly      bool  — at least one fault_event overlaps the window
      is_normal        bool  — window is fully within normal_windows (no faults)
      fault_events     list  — fault events that overlap the window
      normal_overlap   bool  — any normal_window overlaps the window
      session_range    tuple — (session_start, session_end) from the log
    """
    result: Dict[str, Any] = {
        "covered": False,
        "has_anomaly": False,
        "is_normal": False,
        "fault_events": [],
        "normal_overlap": False,
        "session_range": (None, None),
    }

    sess_start = _parse_iso_utc(session_log.get("session_start", ""))
    sess_end = _parse_iso_utc(session_log.get("session_end", ""))
    result["session_range"] = (sess_start, sess_end)

    # Check if the session covers any part of our prediction window
    if sess_start and sess_end:
        if sess_end < window_start or sess_start > window_end:
            # Session is entirely outside the prediction window
            return result
        result["covered"] = True
    elif sess_start:
        # Session still running (no end yet) — assume it covers up to now
        if sess_start <= window_end:
            result["covered"] = True

    # Check fault events that overlap the prediction window
    for ev in session_log.get("fault_events") or []:
        ev_start = _parse_iso_utc(ev.get("start_ts", ""))
        ev_end = _parse_iso_utc(ev.get("end_ts", ""))
        if ev_start is None or ev_end is None:
            continue
        # Overlaps if ev_start <= window_end AND ev_end >= window_start
        if ev_start <= window_end and ev_end >= window_start:
            result["has_anomaly"] = True
            result["fault_events"].append({
                "fault_type": ev.get("fault_type", "?"),
                "target":     ev.get("target_service", "?"),
                "start_ts":   ev.get("start_ts"),
                "end_ts":     ev.get("end_ts"),
                "params":     ev.get("params", {}),
            })

    # Check normal windows that overlap the prediction window
    for nw in session_log.get("normal_windows") or []:
        nw_start = _parse_iso_utc(nw.get("start_ts", ""))
        nw_end = _parse_iso_utc(nw.get("end_ts", ""))
        if nw_start is None or nw_end is None:
            continue
        if nw_start <= window_end and nw_end >= window_start:
            result["normal_overlap"] = True
            break

    result["is_normal"] = result["normal_overlap"] and not result["has_anomaly"]
    return result


def _print_validation(
    predictions: List,
    ground_truth: Dict[str, Any],
    window_start: datetime,
    window_end: datetime,
) -> None:
    """Print ground truth comparison between model predictions and session log."""
    _section("Ground Truth Validation")

    model_anomaly = len(predictions) > 0
    gt = ground_truth

    sess_start, sess_end = gt["session_range"]
    print(f"  Prediction window : {window_start.strftime('%Y-%m-%dT%H:%M:%SZ')} → "
          f"{window_end.strftime('%Y-%m-%dT%H:%M:%SZ')}")
    if sess_start:
        print(f"  Session range     : {sess_start.strftime('%Y-%m-%dT%H:%M:%SZ')} → "
              f"{sess_end.strftime('%Y-%m-%dT%H:%M:%SZ') if sess_end else 'running'}")

    if not gt["covered"]:
        print()
        print("  ⚠️  Session log does not cover this prediction window.")
        print("      The testbed session ran at a different time — no comparison possible.")
        return

    print()
    print(f"  Model predicted   : {'🔴 ANOMALY' if model_anomaly else '✅ NORMAL'}"
          f"  ({len(predictions)} sequence(s) flagged)")
    print(f"  Ground truth      : {'🔴 ANOMALY' if gt['has_anomaly'] else '✅ NORMAL'}"
          f"  ({len(gt['fault_events'])} fault event(s) in window)")
    print()

    # Verdict
    if gt["has_anomaly"] and model_anomaly:
        print("  Verdict : ✅ TRUE POSITIVE  — anomaly present and detected")
    elif not gt["has_anomaly"] and not model_anomaly:
        print("  Verdict : ✅ TRUE NEGATIVE  — no anomaly present, none predicted")
    elif gt["has_anomaly"] and not model_anomaly:
        print("  Verdict : ❌ FALSE NEGATIVE — anomaly present but NOT detected (missed)")
    else:
        print("  Verdict : ⚠️  FALSE POSITIVE — no anomaly present but model raised alarm")

    # Show what was actually injected (if any)
    if gt["fault_events"]:
        print()
        print(f"  Faults injected during window:")
        for ev in gt["fault_events"]:
            print(f"    • {ev['fault_type']} → {ev['target']}"
                  f"  [{ev['start_ts']} → {ev['end_ts']}]"
                  f"  params={ev['params']}")


def run_predict(
    client: AICybOpsClient,
    model_type: str,
    model_version: str,
    data_params: Dict[str, Any],
    verbose: bool,
    validate: bool,
    collect_api_url: Optional[str],
) -> Tuple[int, Dict[str, Any]]:
    """
    Call POST /predict/, print results, and optionally validate against session log.

    Returns:
        Tuple of (exit_code, run_report):
        - exit_code: 0 = no anomalies, 1 = anomalies detected, 2 = error
        - run_report: dict with prediction details for JSON report output
    """
    # Capture the window timestamps before the call so they're as accurate as possible
    window_end = _utcnow()
    window_start = window_end + _parse_start_offset(data_params.get("start", "-600s"))

    t0 = time.monotonic()
    try:
        result = client.predict(
            experiment_name="LiveMonitoring",
            model_type=model_type,
            registered_model_name=model_type,
            model_version=model_version,
            model_params=data_params,
        )
    except Exception as e:
        print(f"  [ERROR] Predict call failed: {e}")
        if hasattr(e, "response") and e.response is not None:
            try:
                print(f"    Response: {e.response.text[:500]}")
            except (AttributeError, TypeError):
                pass
        return 2, {"timestamp": _utcnow_str(), "error": str(e)}

    elapsed = time.monotonic() - t0
    predictions = result.get("predictions") or []
    diag = result.get("prediction_diagnostics") or {}
    num_seq = diag.get("num_sequences")

    print(f"  Timestamp  : {_utcnow_str()}")
    print(f"  Duration   : {elapsed:.1f}s")
    print(f"  Window     : {data_params.get('start', '?')} → now")
    if num_seq is not None:
        print(f"  Sequences  : {num_seq}")

    if not predictions:
        if num_seq == 0:
            print(
                "  Result     : ⚠️  No sequences formed — "
                "window may be too short or metrics API returned no data"
            )
            rc = 2
        else:
            total = f"  (0/{num_seq} sequences flagged)" if num_seq else ""
            print(f"  Result     : ✅ No anomalies detected{total}")
            rc = 0
    else:
        if num_seq:
            pct = 100 * len(predictions) / num_seq
            print(f"  Result     : 🔴 {len(predictions)}/{num_seq} sequences flagged ({pct:.0f}%)")
        else:
            print(f"  Result     : 🔴 {len(predictions)} sequences flagged")
        if verbose:
            print()
            print("  Predictions (full):")
            print(json.dumps(predictions, indent=4, default=str))
        else:
            print()
            for i, pred in enumerate(predictions[:5]):
                if isinstance(pred, dict):
                    parts = []
                    for key in ("timestamp", "time", "start_ts"):
                        if pred.get(key):
                            parts.append(f"t={pred[key]}")
                            break
                    score = pred.get("anomaly_score") or pred.get("score")
                    if score is not None:
                        parts.append(
                            f"score={score:.4f}" if isinstance(score, float) else f"score={score}"
                        )
                    label = pred.get("label") or pred.get("predicted_label")
                    if label is not None:
                        parts.append(f"label={label}")
                    print(f"    [{i + 1}] {' | '.join(parts) or str(pred)[:100]}")
                else:
                    print(f"    [{i + 1}] {str(pred)[:100]}")
            if len(predictions) > 5:
                print(f"    ... and {len(predictions) - 5} more  (use --verbose to see all)")
        rc = 1

    if validate:
        session_log = _get_session_log(collect_api_url)
        if session_log is None:
            _section("Ground Truth Validation")
            print("  ⚠️  No session log available — cannot validate.")
            print("      Make sure the testbed is running (or has run) and collect_metrics API is reachable.")
            print(f"      API URL    : {collect_api_url or '(not set)'}")
        else:
            ground_truth = _ground_truth_for_window(session_log, window_start, window_end)
            _print_validation(predictions, ground_truth, window_start, window_end)

    exit_labels = {0: "0 — clear", 1: "1 — ALERT (anomalies detected)", 2: "2 — error"}
    print(f"  Exit code  : {exit_labels.get(rc, str(rc))}")

    # Build run report
    run_report: Dict[str, Any] = {
        "timestamp": _utcnow_str(),
        "window": {
            "start": window_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": window_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "model_prediction": "anomaly" if predictions else "normal",
        "sequences_flagged": len(predictions),
        "sequences_total": num_seq,
        "exit_code": rc,
    }
    if validate and session_log is not None:
        gt = ground_truth
        verdict = "UNCOVERED"
        if gt["covered"]:
            if gt["has_anomaly"] and predictions:
                verdict = "TRUE_POSITIVE"
            elif not gt["has_anomaly"] and not predictions:
                verdict = "TRUE_NEGATIVE"
            elif gt["has_anomaly"] and not predictions:
                verdict = "FALSE_NEGATIVE"
            else:
                verdict = "FALSE_POSITIVE"
        run_report["ground_truth"] = {
            "has_anomaly": gt["has_anomaly"],
            "fault_events": gt["fault_events"],
            "covered": gt["covered"],
        }
        run_report["verdict"] = verdict

    return rc, run_report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score live data against the trained DAM model (production prediction).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--start",
        type=str,
        default="-600s",
        help="Monitoring window start offset (default: -600s = last 10 minutes). "
             "Examples: -300s (5 min), -1800s (30 min).",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="latest",
        help="Registered model version to use (default: latest).",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Run continuously, scoring a fresh window every --interval seconds.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=60.0,
        help="Seconds between predictions in --watch mode (default: 60).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print full prediction payloads.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="After predicting, fetch the session log for the same time window and compare "
             "model predictions against ground truth (what the testbed actually injected). "
             "Requires a running or recently completed testbed session.",
    )
    parser.add_argument(
        "--collect-api-url",
        type=str,
        default=os.environ.get("COLLECT_API_URL", "http://kube-worker1.lis.ipn.pt:5010"),
        help="collect_metrics API URL to fetch session log from for --validate "
             "(default: http://kube-worker1.lis.ipn.pt:5010 or COLLECT_API_URL env var).",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Write a JSON report with prediction results to this path. "
             "In --watch mode, accumulates all runs and rewrites after each cycle.",
    )
    parser.add_argument(
        "--strict-exit-codes",
        action="store_true",
        help="Return non-zero exit codes for anomaly/error outcomes "
             "(legacy behavior). By default, CLI is informational and exits 0 "
             "unless there is an error.",
    )
    args = parser.parse_args()

    base_url = resolve_aicybops_service_url()
    model_type = os.getenv("DAM_MODEL_NAME", "dam")

    data_params = {**DEFAULT_PREDICT_DATA_PARAMS, "start": args.start}
    collect_api_url = args.collect_api_url.strip() if args.collect_api_url else None

    _section("AICybOps — Live Anomaly Detection")
    print(f"  Service      : {base_url}")
    print(f"  Model        : {model_type}  version={args.model_version}")
    print(f"  Window       : {args.start} → now")
    if args.validate:
        print(f"  Validate     : yes  (session-log source: collect_metrics API)")
        print(f"  API URL      : {collect_api_url or '(not set)'}")
    else:
        print(f"  Session log  : off  (pass --validate to compare against testbed ground truth)")
    if args.watch:
        print(f"  Mode         : continuous (every {args.interval:.0f}s — Ctrl+C to stop)")
    else:
        print(f"  Mode         : single prediction")
    print(f"  Exit mode    : {'strict' if args.strict_exit_codes else 'informational'}")

    client = AICybOpsClient(base_url=base_url)

    # Quick health check before committing to a (potentially long) predict call
    try:
        health = client.health_check()
        status = health.get("status", "?")
        if status != "ok":
            print(f"\n  [WARN] Service health status: {status}  —  {health}")
        else:
            print(f"\n  Service health: ok")
    except Exception as e:
        print(f"\n  [ERROR] Service unreachable at {base_url}: {e}")
        return 2

    predict_kwargs = dict(
        client=client,
        model_type=model_type,
        model_version=args.model_version,
        data_params=data_params,
        verbose=args.verbose,
        validate=args.validate,
        collect_api_url=collect_api_url,
    )

    def _write_report(report_data: Dict[str, Any]) -> None:
        if not args.report:
            return
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=2, default=str)
        print(f"  Report saved to: {report_path}")

    if not args.watch:
        _section("Prediction")
        rc, run_report = run_predict(**predict_kwargs)
        if args.report:
            report = {
                "model_type": model_type,
                "model_version": args.model_version,
                "mode": "single",
                "runs": [run_report],
            }
            _write_report(report)
        if args.strict_exit_codes:
            return rc
        # Default CLI behavior: print outcomes; only hard-fail on actual errors.
        return 2 if rc == 2 else 0

    _section(f"Continuous monitoring  (interval={args.interval:.0f}s — Ctrl+C to stop)")
    last_rc = 0
    run_count = 0
    all_runs: deque = deque(maxlen=1000)

    def _build_watch_report() -> Dict[str, Any]:
        verdicts = [r.get("verdict") for r in all_runs if "verdict" in r]
        tp = verdicts.count("TRUE_POSITIVE")
        tn = verdicts.count("TRUE_NEGATIVE")
        fp = verdicts.count("FALSE_POSITIVE")
        fn = verdicts.count("FALSE_NEGATIVE")
        total_validated = tp + tn + fp + fn
        return {
            "model_type": model_type,
            "model_version": args.model_version,
            "mode": "watch",
            "runs": list(all_runs),
            "summary": {
                "total_runs": len(all_runs),
                "true_positives": tp,
                "true_negatives": tn,
                "false_positives": fp,
                "false_negatives": fn,
                "accuracy": (tp + tn) / total_validated if total_validated > 0 else None,
            },
        }

    try:
        while True:
            run_count += 1
            print(f"\n  --- Run #{run_count} ---")
            last_rc, run_report = run_predict(**predict_kwargs)
            run_report["run_number"] = run_count
            all_runs.append(run_report)
            if args.report:
                _write_report(_build_watch_report())
            print(f"\n  Next check in {args.interval:.0f}s  (Ctrl+C to stop)")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n\n  Monitoring stopped.")
        if args.report:
            _write_report(_build_watch_report())
        return 0

    if args.strict_exit_codes:
        return last_rc
    return 2 if last_rc == 2 else 0


if __name__ == "__main__":
    sys.exit(main())
