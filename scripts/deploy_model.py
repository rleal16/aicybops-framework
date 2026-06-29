#!/usr/bin/env python3
"""Train, predict, and evaluate DAM models through the AICybOps service.

Training and evaluation run asynchronously; use `--resume` with the saved run JSON
to continue interrupted runs.
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from service_url import resolve_aicybops_service_url

from aicybops_lib.client import AICybOpsClient, get_data_collection_counts

DEFAULT_TRAIN_DATA_PARAMS = {
    "use_api": True,
    "use_session_time_range": True,
    "start": "-600s",
    "risk_level": 1e-3,
    "depth": 10,
    "init_quantile": 0.80,
}

def _section(title: str, char: str = "=") -> None:
    width = 72
    print()
    print(char * width)
    print(f"  {title}")
    print(char * width)


def _step(
    name: str,
    status: str,
    duration_sec: Optional[float] = None,
    detail: Optional[str] = None,
) -> None:
    line = f"  [{status}] {name}"
    if duration_sec is not None:
        line += f"  ({duration_sec:.1f}s)"
    print(line)
    if detail:
        for d in detail.split("\n"):
            print(f"      {d}")


def _log_response(label: str, data: dict, verbose: bool) -> None:
    if verbose:
        print(f"  {label} (full):")
        print(json.dumps(data, indent=2, default=str))
    else:
        keys = list(data.keys())[:12]
        print(f"  {label} keys: {keys}")
        if "model_reference" in data and data["model_reference"]:
            print(f"      model_reference: {data['model_reference']}")
        if "result" in data and isinstance(data["result"], dict):
            rk = list(data["result"].keys())[:8]
            print(f"      result keys: {rk}")


def _log_request(label: str, payload: dict, verbose: bool) -> None:
    if verbose:
        print(f"  {label}:")
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(f"  {label}: {list(payload.keys())}")


def _save_run_file(run_state: Dict[str, Any], path: Path) -> None:
    """Atomically save run state to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(run_state, f, indent=2, default=str)
    tmp.rename(path)


def _load_run_file(path: Path) -> Dict[str, Any]:
    """Load run state from JSON."""
    with open(path) as f:
        return json.load(f)


def _default_run_file_path() -> Path:
    """Generate default resume JSON path in cwd."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(f"deploy_run_{ts}.json")


def run_health(client: AICybOpsClient, verbose: bool) -> bool:
    _section("1. Health check")
    t0 = time.monotonic()
    try:
        _log_request("Request GET /", {}, verbose)
        response = client.health_check()
        elapsed = time.monotonic() - t0
        _log_response("Response", response, verbose)
        _step("Health", "OK", duration_sec=elapsed, detail=f"status={response.get('status', '?')}")
        return True
    except Exception as e:
        elapsed = time.monotonic() - t0
        _step("Health", "FAIL", duration_sec=elapsed, detail=str(e))
        if hasattr(e, "response") and e.response is not None:
            try:
                print(f"      Response body: {e.response.text[:500]}")
            except (AttributeError, TypeError):
                pass
        traceback.print_exc()
        return False


def refresh_resume_job_states(
    client: AICybOpsClient,
    run_state: Dict[str, Any],
    run_file_path: Path,
    verbose: bool,
) -> None:
    """
    Refresh stale resume statuses from server-side async job state.

    This keeps local run files in sync when a job finishes while the script is not running.
    """
    updated = False
    training = run_state.get("training", {})
    training_job_id = training.get("job_id")
    if training_job_id and training.get("status") in {"pending", "running"}:
        try:
            status_resp = client.get_job_status(training_job_id)
            status = status_resp.get("status", "?")
            print(f"  Refresh training job {training_job_id}: status={status}")
            if status == "completed":
                run_state["training"] = {
                    "job_id": training_job_id,
                    "status": "completed",
                    "result": status_resp,
                    "model_reference": status_resp.get("model_reference"),
                }
                updated = True
            elif status == "failed":
                run_state["training"]["status"] = "failed"
                run_state["training"]["result"] = status_resp
                updated = True
            elif status in {"pending", "running"}:
                run_state["training"]["status"] = status
            if verbose:
                _log_response("Refreshed training status", status_resp, verbose)
        except Exception as e:
            print(f"  [WARN] Could not refresh training job {training_job_id}: {e}")

    evaluation = run_state.get("evaluation", {})
    eval_job_id = evaluation.get("job_id")
    if eval_job_id and evaluation.get("status") in {"pending", "running"}:
        try:
            status_resp = client.get_eval_job_status(eval_job_id)
            status = status_resp.get("status", "?")
            print(f"  Refresh evaluation job {eval_job_id}: status={status}")
            if status == "completed":
                run_state["evaluation"] = {
                    "job_id": eval_job_id,
                    "status": "completed",
                    "result": status_resp,
                }
                updated = True
            elif status == "failed":
                run_state["evaluation"]["status"] = "failed"
                run_state["evaluation"]["result"] = status_resp
                updated = True
            elif status in {"pending", "running"}:
                run_state["evaluation"]["status"] = status
            if verbose:
                _log_response("Refreshed evaluation status", status_resp, verbose)
        except Exception as e:
            print(f"  [WARN] Could not refresh evaluation job {eval_job_id}: {e}")

    if updated:
        _save_run_file(run_state, run_file_path)
        print("  Resume file updated with latest server job state.")


def run_train_async(
    client: AICybOpsClient,
    experiment_name: str,
    model_type: str,
    data_params: dict,
    epochs: int,
    run_optimization: bool,
    poll_interval_sec: float,
    verbose: bool,
    run_state: Dict[str, Any],
    run_file_path: Path,
    objective: str = "f1_score",
    max_evals: Optional[int] = None,
    max_poll_seconds: float = 3600,
) -> Tuple[bool, Dict[str, Any]]:
    """Start or resume async training. Updates run_state in place and saves to disk."""
    label = "Train (with optimisation)" if run_optimization else "Train"
    _section(f"2. {label}")
    t0 = time.monotonic()

    training = run_state.get("training", {})
    job_id = training.get("job_id")

    if training.get("status") == "completed" and training.get("result") is not None:
        _step(label, "OK (cached)", detail="Using result from previous run")
        _log_response("Cached result", training.get("result", {}), verbose)
        return True, training.get("result", {})

    try:
        if not job_id:
            payload = {
                "experiment_name": experiment_name,
                "model_type": model_type,
                "params": data_params,
                "epochs": epochs,
                "model_params": data_params,
                "run_optimization": run_optimization,
                "objective": objective,
            }
            if max_evals is not None:
                payload["max_evals"] = max_evals
            _log_request("Request POST /train/ (async)", payload, verbose)
            result = client.train_model(
                experiment_name=experiment_name,
                model_type=model_type,
                params=data_params,
                epochs=epochs,
                model_params=data_params,
                run_optimization=run_optimization,
                objective=objective,
                max_evals=max_evals,
                wait=False,
            )
            job_id = result.get("job_id")
            if not job_id:
                _step(label, "FAIL", duration_sec=time.monotonic() - t0, detail="No job_id in response")
                return False, {}
            print(f"  Job accepted: job_id={job_id}")
            run_state["training"] = {"job_id": job_id, "status": "pending", "result": None, "model_reference": None}
            _save_run_file(run_state, run_file_path)
        else:
            print(f"  Resuming poll for job_id={job_id}")

        poll_deadline = time.monotonic() + max_poll_seconds
        while time.monotonic() < poll_deadline:
            time.sleep(poll_interval_sec)
            try:
                status_resp = client.get_job_status(job_id)
            except Exception as poll_err:
                if hasattr(poll_err, "response") and poll_err.response is not None and poll_err.response.status_code == 404:
                    _step(label, "FAIL", duration_sec=time.monotonic() - t0,
                          detail=f"Job {job_id} not found — service may have restarted. Start a new run.")
                    return False, {}
                raise
            status = status_resp.get("status", "?")
            print(f"  Poll job {job_id}: status={status}")
            if status == "completed":
                elapsed = time.monotonic() - t0
                _log_response("Job response", status_resp, verbose)
                metrics_count, logs_count = get_data_collection_counts(status_resp)
                detail = f"result present={('result' in status_resp)}"
                if metrics_count is not None or logs_count is not None:
                    detail += f" | metrics={metrics_count or '?'} logs={logs_count or '?'}"
                _step(label, "OK", duration_sec=elapsed, detail=detail)
                run_state["training"] = {
                    "job_id": job_id,
                    "status": "completed",
                    "result": status_resp,
                    "model_reference": status_resp.get("model_reference"),
                }
                _save_run_file(run_state, run_file_path)
                return True, status_resp
            if status == "failed":
                elapsed = time.monotonic() - t0
                err = status_resp.get("error", "unknown")
                _step(label, "FAIL", duration_sec=elapsed, detail=f"error={err}")
                run_state["training"]["status"] = "failed"
                _save_run_file(run_state, run_file_path)
                return False, status_resp
        else:
            _step(label, "FAIL", duration_sec=time.monotonic() - t0,
                  detail=f"Polling timed out after {max_poll_seconds}s")
            return False, {}
    except Exception as e:
        elapsed = time.monotonic() - t0
        _step(label, "FAIL", duration_sec=elapsed, detail=str(e))
        traceback.print_exc()
        return False, {}


def run_predict(
    client: AICybOpsClient,
    experiment_name: str,
    model_type: str,
    model_reference: Optional[Dict[str, Any]],
    model_params: Dict[str, Any],
    verbose: bool,
) -> Tuple[bool, Dict[str, Any]]:
    _section("3. Predict")
    t0 = time.monotonic()
    reg_name = (model_reference or {}).get("registered_model_name") or model_type
    version = (model_reference or {}).get("model_version") or "latest"
    try:
        payload = {
            "experiment_name": experiment_name,
            "model_type": model_type,
            "registered_model_name": reg_name,
            "model_version": version,
            "model_params": model_params,
        }
        _log_request("Request POST /predict/", payload, verbose)
        result = client.predict(
            experiment_name=experiment_name,
            model_type=model_type,
            registered_model_name=reg_name,
            model_version=version,
            model_params=model_params,
        )
        elapsed = time.monotonic() - t0
        _log_response("Response", result, verbose)
        pred = result.get("predictions")
        diag = result.get("prediction_diagnostics") or {}
        num_seq = diag.get("num_sequences")
        if pred is not None:
            if isinstance(pred, list):
                detail = f"predictions length={len(pred)}"
            else:
                detail = f"predictions type={type(pred).__name__}"
        else:
            detail = "predictions key missing"
        if diag:
            print(f"      prediction_diagnostics: {diag}")
        if (pred is not None and isinstance(pred, list) and len(pred) == 0) and diag:
            if num_seq is not None and num_seq == 0:
                print(
                    "      -> No sequences in prediction data: API/metrics returned no data "
                    "for the window, or not enough points to form one sequence. "
                    "Check start/window and that the service can reach the metrics API."
                )
            elif num_seq is not None and num_seq > 0:
                print("      -> All sequences scored normal (no alarms).")
        _step("Predict", "OK", duration_sec=elapsed, detail=detail)
        return True, result
    except Exception as e:
        elapsed = time.monotonic() - t0
        _step("Predict", "FAIL", duration_sec=elapsed, detail=str(e))
        if hasattr(e, "response") and e.response is not None:
            try:
                print(f"      Response body: {e.response.text[:800]}")
            except (AttributeError, TypeError):
                pass
        traceback.print_exc()
        return False, {}


def _print_eval_metrics(res: dict) -> None:
    """Pretty-print evaluation metrics."""
    if not isinstance(res, dict):
        return
    m = res.get("metrics") or {}
    anomaly_rate = res.get("anomaly_rate")
    actual_anomaly_rate = res.get("actual_anomaly_rate")
    target_achieved = res.get("dam_target_achieved")
    if m:
        print()
        print("  Metrics:")
        if "precision" in m:
            print(f"    Precision        : {m['precision']:.3f}")
        if "recall" in m:
            print(f"    Recall           : {m['recall']:.3f}")
        if "f1_score" in m:
            print(f"    F1 Score         : {m['f1_score']:.3f}")
        if "roc_auc" in m:
            print(f"    ROC-AUC          : {m['roc_auc']:.3f}")
        if "average_precision" in m:
            print(f"    Avg Precision    : {m['average_precision']:.3f}")
        tp = m.get("true_positives")
        fp = m.get("false_positives")
        tn = m.get("true_negatives")
        fn = m.get("false_negatives")
        if all(v is not None for v in (tp, fp, tn, fn)):
            print(f"    TP/FP/TN/FN      : {tp} / {fp} / {tn} / {fn}")
        if anomaly_rate is not None:
            print(f"    Anomaly rate     : {anomaly_rate * 100:.1f}%", end="")
            if actual_anomaly_rate is not None:
                print(f"  (actual: {actual_anomaly_rate * 100:.1f}%)", end="")
            print()
        if target_achieved is not None:
            print(f"    Target achieved  : {'yes' if target_achieved else 'no'}")


def _collect_eval_report(res: dict) -> Dict[str, Any]:
    """Extract evaluation metrics for the report JSON."""
    eval_report: Dict[str, Any] = {}
    if isinstance(res, dict):
        m = res.get("metrics") or {}
        eval_report["metrics"] = m
        if res.get("anomaly_rate") is not None:
            eval_report["anomaly_rate"] = res["anomaly_rate"]
        if res.get("actual_anomaly_rate") is not None:
            eval_report["actual_anomaly_rate"] = res["actual_anomaly_rate"]
        if res.get("dam_target_achieved") is not None:
            eval_report["target_achieved"] = res["dam_target_achieved"]
    return eval_report


def run_evaluate_async(
    client: AICybOpsClient,
    experiment_name: str,
    model_type: str,
    model_reference: Optional[Dict[str, Any]],
    poll_interval_sec: float,
    verbose: bool,
    run_state: Dict[str, Any],
    run_file_path: Path,
    eval_data_params: Optional[Dict[str, Any]] = None,
    best_params: Optional[Dict[str, Any]] = None,
    max_poll_seconds: float = 3600,
) -> Tuple[bool, Dict[str, Any]]:
    """Start or resume async evaluation. Updates run_state in place and saves to disk."""
    _section("4. Evaluate")
    t0 = time.monotonic()
    reg_name = (model_reference or {}).get("registered_model_name") or model_type
    version = (model_reference or {}).get("model_version") or "latest"
    eval_risk_level = float((best_params or {}).get("risk_level", 1e-3))
    eval_init_quantile = float((best_params or {}).get("init_quantile", 0.95))
    evaluation_config = {
        "evt_parameters": {
            "risk_level": eval_risk_level,
            "init_quantile": eval_init_quantile,
        },
        "max_memory_gb": 2.0,
    }
    evaluation_model_params: Dict[str, Any] = {"use_api": True}
    if eval_data_params:
        evaluation_model_params.update(eval_data_params)

    evaluation = run_state.get("evaluation", {})
    job_id = evaluation.get("job_id")

    if evaluation.get("status") == "completed" and evaluation.get("result") is not None:
        _step("Evaluate", "OK (cached)", detail="Using result from previous run")
        res = evaluation["result"].get("result", evaluation["result"])
        _print_eval_metrics(res)
        return True, _collect_eval_report(res)

    try:
        if not job_id:
            payload = {
                "experiment_name": experiment_name,
                "model_type": model_type,
                "registered_model_name": reg_name,
                "model_version": version,
                "model_params": evaluation_model_params,
                "evaluation_config": evaluation_config,
                "dataset_type": "test",
            }
            _log_request("Request POST /evaluate/ (async)", payload, verbose)
            result = client.evaluate(
                experiment_name=experiment_name,
                model_type=model_type,
                model_params=evaluation_model_params,
                evaluation_config=evaluation_config,
                registered_model_name=reg_name,
                model_version=version,
                dataset_type="test",
                wait=False,
            )
            job_id = result.get("job_id")
            if not job_id:
                _step("Evaluate", "FAIL", duration_sec=time.monotonic() - t0, detail="No job_id in response")
                return False, {}
            print(f"  Eval job accepted: job_id={job_id}")
            run_state["evaluation"] = {"job_id": job_id, "status": "pending", "result": None}
            _save_run_file(run_state, run_file_path)
        else:
            print(f"  Resuming poll for eval job_id={job_id}")

        poll_deadline = time.monotonic() + max_poll_seconds
        while time.monotonic() < poll_deadline:
            time.sleep(poll_interval_sec)
            try:
                status_resp = client.get_eval_job_status(job_id)
            except Exception as poll_err:
                if hasattr(poll_err, "response") and poll_err.response is not None and poll_err.response.status_code == 404:
                    _step("Evaluate", "FAIL", duration_sec=time.monotonic() - t0,
                          detail=f"Eval job {job_id} not found — service may have restarted. Start a new run.")
                    return False, {}
                raise
            status = status_resp.get("status", "?")
            print(f"  Poll eval job {job_id}: status={status}")
            if status == "completed":
                elapsed = time.monotonic() - t0
                _log_response("Eval job response", status_resp, verbose)
                res = status_resp.get("result", {})
                _print_eval_metrics(res)
                eval_report = _collect_eval_report(res)
                detail = f"model={reg_name} v{version}"
                _step("Evaluate", "OK", duration_sec=elapsed, detail=detail)
                run_state["evaluation"] = {
                    "job_id": job_id,
                    "status": "completed",
                    "result": status_resp,
                }
                _save_run_file(run_state, run_file_path)
                return True, eval_report
            if status == "failed":
                elapsed = time.monotonic() - t0
                err = status_resp.get("error", "unknown")
                if "not supported" in err.lower() or "501" in err:
                    _step("Evaluate", "SKIP", duration_sec=elapsed, detail="Evaluation not supported")
                    run_state["evaluation"]["status"] = "skipped"
                    _save_run_file(run_state, run_file_path)
                    return True, {}
                if "No metrics data loaded" in err or "load metrics first" in err:
                    _step("Evaluate", "SKIP", duration_sec=elapsed,
                          detail="Evaluate requires loaded metrics (not yet supported via API)")
                    run_state["evaluation"]["status"] = "skipped"
                    _save_run_file(run_state, run_file_path)
                    return True, {}
                _step("Evaluate", "FAIL", duration_sec=elapsed, detail=f"error={err}")
                run_state["evaluation"]["status"] = "failed"
                _save_run_file(run_state, run_file_path)
                return False, {}
        else:
            _step("Evaluate", "FAIL", duration_sec=time.monotonic() - t0,
                  detail=f"Polling timed out after {max_poll_seconds}s")
            return False, {}
    except Exception as e:
        elapsed = time.monotonic() - t0
        _step("Evaluate", "FAIL", duration_sec=elapsed, detail=str(e))
        traceback.print_exc()
        return False, {}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a DAM model, then predict and evaluate on the test set. "
                    "All runs are async — you can safely Ctrl+C and resume with --resume.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Log full request/response bodies",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Skip hyperparameter optimisation (default: optimise)",
    )
    parser.add_argument(
        "--no-evaluate",
        action="store_true",
        help="Skip the evaluate step",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        default=False,
        help="Enable the predict step (disabled by default).",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Seconds between job status polls (default 5)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="AICybOps",
        help="MLflow experiment name (default: AICybOps)",
    )
    parser.add_argument(
        "--training-window",
        type=int,
        default=0,
        help="Limit training data to the first N minutes of the session. "
             "Default: 0 (use all data from session_start to now).",
    )
    parser.add_argument(
        "--eval-start",
        type=str,
        default=None,
        help="Evaluation start offset when not using session range (e.g. -1800s, -45m).",
    )
    parser.add_argument(
        "--eval-window",
        type=int,
        default=None,
        help="Limit evaluation to first N session minutes (requires session range).",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Write a JSON report with metrics to this path (for stakeholder evidence).",
    )
    parser.add_argument(
        "--run-file",
        type=str,
        default=None,
        help="Path for the resume JSON file. Default: deploy_run_<timestamp>.json in cwd.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume a previous run from a saved JSON file.",
    )
    parser.add_argument(
        "--resume-run",
        action="store_true",
        help="When used with --resume, continue remaining steps (train/predict/evaluate). "
             "By default, --resume is read-only and performs no API actions.",
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="f1_score",
        choices=["val_loss", "f1_score", "test_anomaly_scores_mean"],
        help="Optimization objective (default: f1_score). f1_score optimises for "
             "detection performance; val_loss for reconstruction quality.",
    )
    parser.add_argument(
        "--max-evals",
        type=int,
        default=None,
        help="Max number of hyperparameter trials (default: auto). "
             "Higher values explore more of the search space.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    base_url = resolve_aicybops_service_url()
    model_type = os.getenv("DAM_MODEL_NAME", "dam")

    run_optimization = not args.no_optimize

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            print(f"  ERROR: Resume file not found: {resume_path}")
            return 1
        run_state = _load_run_file(resume_path)
        run_file_path = resume_path
        cli_args = run_state.get("cli_args", {})
        data_params = run_state.get("data_params", DEFAULT_TRAIN_DATA_PARAMS.copy())
        experiment_name = cli_args.get("experiment", args.experiment)
        epochs = cli_args.get("epochs", args.epochs)
        run_optimization = cli_args.get("optimize", run_optimization)
        no_evaluate = cli_args.get("no_evaluate", args.no_evaluate)
        verbose = args.verbose or cli_args.get("verbose", False)
        eval_data_params = run_state.get("eval_data_params", {})
        if not isinstance(eval_data_params, dict):
            eval_data_params = {}
        # Backward compatibility with older run files that don't contain prediction state.
        run_state.setdefault("prediction", {"status": "pending", "result": None})
        print(f"  Resuming from: {resume_path}")
    else:
        data_params = DEFAULT_TRAIN_DATA_PARAMS.copy()
        if args.training_window > 0:
            data_params["training_window_minutes"] = args.training_window
        experiment_name = args.experiment
        epochs = args.epochs
        no_evaluate = args.no_evaluate
        verbose = args.verbose
        eval_data_params: Dict[str, Any] = {}

        if args.run_file:
            run_file_path = Path(args.run_file)
        else:
            run_file_path = _default_run_file_path()

        run_state = {
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "cli_args": {
                "experiment": experiment_name,
                "epochs": epochs,
                "optimize": run_optimization,
                "objective": args.objective,
                "max_evals": args.max_evals,
                "no_evaluate": no_evaluate,
                "training_window": args.training_window,
                "eval_start": args.eval_start,
                "eval_window": args.eval_window,
                "verbose": verbose,
                "report": args.report,
                "poll_interval": args.poll_interval,
            },
            "model_type": model_type,
            "service_url": base_url,
            "data_params": data_params,
            "eval_data_params": {},
            "training": {"job_id": None, "status": "pending", "result": None, "model_reference": None},
            "prediction": {"status": "pending", "result": None},
            "evaluation": {"job_id": None, "status": "pending", "result": None},
        }

    if args.eval_start is not None:
        eval_data_params["use_session_time_range"] = False
        eval_data_params["start"] = args.eval_start
    if args.eval_window is not None:
        eval_data_params["use_session_time_range"] = True
        eval_data_params["training_window_minutes"] = args.eval_window
        eval_data_params.pop("start", None)

    if not eval_data_params:
        use_session_time_range = bool(data_params.get("use_session_time_range", True))
        train_window_minutes = int(data_params.get("training_window_minutes", 0) or 0)
        if use_session_time_range and train_window_minutes > 0:
            eval_data_params = {
                "use_session_time_range": True,
                "training_window_minutes": train_window_minutes,
            }
        else:
            eval_data_params = {
                "use_session_time_range": False,
                "start": data_params.get("start", "-600s"),
            }

    eval_data_params["use_api"] = True
    run_state["eval_data_params"] = eval_data_params

    _section("AICybOps — Model Training & Evaluation", "=")
    print(f"  Service URL  : {base_url}")
    print(f"  Model type   : {model_type}")
    print(f"  Experiment   : {experiment_name}")
    print(f"  Epochs       : {epochs}")
    print(f"  Optimise     : {'yes' if run_optimization else 'no  (--no-optimize)'}")
    if run_optimization:
        print(f"  Objective    : {args.objective}")
        if args.max_evals:
            print(f"  Max evals    : {args.max_evals}")
    print(f"  Evaluate     : {'no  (--no-evaluate)' if no_evaluate else 'yes'}")
    if data_params.get("training_window_minutes"):
        print(f"  Train window : first {data_params['training_window_minutes']} minutes of session")
    if eval_data_params.get("use_session_time_range"):
        eval_window = eval_data_params.get("training_window_minutes")
        detail = f"first {eval_window} minutes of session" if eval_window else "full session range"
        print(f"  Eval window  : {detail}")
    else:
        print(f"  Eval window  : start={eval_data_params.get('start', '-600s')} to now")
    print(f"  Data params  : {data_params}")
    print(f"  Run file     : {run_file_path}")
    if args.report:
        print(f"  Report       : {args.report}")
    if args.resume:
        train_status = run_state.get("training", {}).get("status", "?")
        pred_status = run_state.get("prediction", {}).get("status", "?")
        eval_status = run_state.get("evaluation", {}).get("status", "?")
        print(f"  Resume       : training={train_status}, predict={pred_status}, evaluation={eval_status}")
        print(f"  Resume mode  : {'continue remaining steps (--resume-run)' if args.resume_run else 'read-only (no API actions)'}")

    if args.resume:
        try:
            refresh_client = AICybOpsClient(base_url=base_url)
            refresh_resume_job_states(refresh_client, run_state, run_file_path, verbose)
        except Exception as e:
            print(f"  [WARN] Could not initialize client for resume refresh: {e}")

    if args.resume and not args.resume_run:
        _section("Resume Snapshot", "=")
        training = run_state.get("training", {})
        prediction = run_state.get("prediction", {})
        evaluation = run_state.get("evaluation", {})
        print(f"  Training status  : {training.get('status', '?')}")
        print(f"  Training job_id  : {training.get('job_id')}")
        print(f"  Predict status   : {prediction.get('status', '?')}")
        print(f"  Evaluation status: {evaluation.get('status', '?')}")
        print(f"  Evaluation job_id: {evaluation.get('job_id')}")
        if training.get("status") == "completed" and training.get("result") is not None:
            _log_response("Training result", training.get("result", {}), verbose)
        if prediction.get("status") == "completed" and prediction.get("result") is not None:
            _log_response("Prediction result", prediction.get("result", {}), verbose)
        if evaluation.get("status") == "completed" and evaluation.get("result") is not None:
            _log_response("Evaluation result", evaluation.get("result", {}), verbose)
        print()
        print("  Read-only resume completed. No new requests were made.")
        return 0

    client = AICybOpsClient(base_url=base_url)
    report: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "experiment": experiment_name,
        "model_type": model_type,
        "service_url": base_url,
    }
    run_start = time.monotonic()

    if not run_health(client, verbose):
        return 1

    ok, train_result = run_train_async(
        client, experiment_name, model_type, data_params,
        epochs, run_optimization, args.poll_interval, verbose,
        run_state, run_file_path,
        objective=args.objective,
        max_evals=args.max_evals,
    )
    if not ok:
        return 1
    model_reference = run_state.get("training", {}).get("model_reference") or train_result.get("model_reference") or {}
    metrics_count, logs_count = get_data_collection_counts(train_result)
    report["training"] = {
        "epochs": epochs,
        "optimize": run_optimization,
        "objective": args.objective,
        "training_window_minutes": data_params.get("training_window_minutes"),
        "model_reference": model_reference,
        "data_collection": {"metrics_count": metrics_count, "logs_count": logs_count},
    }
    best_params = None
    result_dict = train_result.get("result") if isinstance(train_result.get("result"), dict) else train_result
    if isinstance(result_dict, dict):
        best_params = result_dict.get("best_params")

    if args.predict:
        prediction = run_state.get("prediction", {})
        if prediction.get("status") == "completed" and prediction.get("result") is not None:
            _step("Predict", "OK (cached)", detail="Using result from previous run")
            _log_response("Cached prediction", prediction.get("result", {}), verbose)
        else:
            predict_params = data_params.copy()
            if isinstance(best_params, dict):
                predict_params.update(best_params)
            ok, predict_result = run_predict(
                client, experiment_name, model_type, model_reference, predict_params, verbose
            )
            if not ok:
                run_state["prediction"] = {"status": "failed", "result": None}
                _save_run_file(run_state, run_file_path)
                return 1
            run_state["prediction"] = {"status": "completed", "result": predict_result}
            _save_run_file(run_state, run_file_path)

    eval_report: Dict[str, Any] = {}
    if not no_evaluate:
        ok, eval_report = run_evaluate_async(
            client, experiment_name, model_type, model_reference,
            args.poll_interval, verbose, run_state, run_file_path,
            eval_data_params=eval_data_params,
            best_params=best_params,
        )
        if not ok:
            return 1
    report["evaluation"] = eval_report

    report["total_duration_s"] = round(time.monotonic() - run_start, 1)

    _section("Summary", "=")
    print("  All steps completed successfully.")
    print()

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"  Report saved to: {report_path}")

    print(f"  Run file: {run_file_path}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
