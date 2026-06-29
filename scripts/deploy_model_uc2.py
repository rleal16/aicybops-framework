#!/usr/bin/env python3
"""AICybOps — modelos_uc_2 Demo Model Training & Evaluation."""

import argparse
import json
import logging
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from service_url import resolve_aicybops_service_url

from aicybops_lib.client import AICybOpsClient
from aicybops_models.modelos_uc_2.models import MODELS

SUPPORTED_MODEL_TYPE = "modelos_uc2"


def _section(title: str, char: str = "=") -> None:
    width = 72
    print()
    print(char * width)
    print(f"  {title}")
    print(char * width)


def _step(name: str, status: str, duration_sec: Optional[float] = None, detail: Optional[str] = None) -> None:
    line = f"  [{status}] {name}"
    if duration_sec is not None:
        line += f"  ({duration_sec:.1f}s)"
    print(line)
    if detail:
        for d in detail.split("\n"):
            print(f"      {d}")


def _log(label: str, data: Any, verbose: bool) -> None:
    if verbose:
        print(f"  {label}:")
        print(json.dumps(data, indent=2, default=str))
    elif isinstance(data, dict):
        print(f"  {label} keys: {list(data.keys())[:12]}")


def _save_run_file(run_state: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(run_state, f, indent=2, default=str)
    tmp.rename(path)


def _load_run_file(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _default_run_file_path() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(f"deploy_uc2_run_{ts}.json")


def refresh_resume_job_states(
    client: AICybOpsClient,
    run_state: Dict[str, Any],
    run_file_path: Path,
    verbose: bool,
) -> None:
    updated = False
    training = run_state.get("training", {})
    training_job_id = training.get("job_id")
    if training_job_id and training.get("status") in {"pending", "running"}:
        try:
            status_resp = client.get_job_status(training_job_id)
            status = status_resp.get("status", "?")
            print(f"  Refresh training job {training_job_id}: status={status}")
            if status in {"completed", "failed"}:
                run_state["training"]["status"] = status
                run_state["training"]["result"] = status_resp
                updated = True
            if verbose:
                _log("Refreshed training status", status_resp, verbose)
        except Exception as exc:
            print(f"  [WARN] Could not refresh training job {training_job_id}: {exc}")

    evaluation = run_state.get("evaluation", {})
    eval_job_id = evaluation.get("job_id")
    if eval_job_id and evaluation.get("status") in {"pending", "running"}:
        try:
            status_resp = client.get_eval_job_status(eval_job_id)
            status = status_resp.get("status", "?")
            print(f"  Refresh evaluation job {eval_job_id}: status={status}")
            if status in {"completed", "failed"}:
                run_state["evaluation"]["status"] = status
                run_state["evaluation"]["result"] = status_resp
                updated = True
            if verbose:
                _log("Refreshed evaluation status", status_resp, verbose)
        except Exception as exc:
            print(f"  [WARN] Could not refresh eval job {eval_job_id}: {exc}")

    if updated:
        _save_run_file(run_state, run_file_path)
        print("  Resume file updated with latest server job state.")


def _poll_until_done(fetch_status, job_id: str, poll_interval_sec: float, max_poll_seconds: float, label: str) -> Tuple[Optional[str], Dict[str, Any]]:
    deadline = time.monotonic() + max_poll_seconds
    while time.monotonic() < deadline:
        time.sleep(poll_interval_sec)
        try:
            status_resp = fetch_status(job_id)
        except Exception as exc:
            print(f"  [WARN] {label} poll error: {exc}")
            continue
        status = status_resp.get("status", "?")
        print(f"  Poll {label} {job_id}: status={status}")
        if status in {"completed", "failed"}:
            return status, status_resp
    return None, {}


def _model_params_from_keys(model_keys: Optional[List[str]]) -> Dict[str, Any]:
    if not model_keys:
        return {}
    return {"model_names": model_keys}


def run_health(client: AICybOpsClient, verbose: bool) -> bool:
    _section("1. Health check")
    t0 = time.monotonic()
    try:
        resp = client.health_check()
        _log("Response", resp, verbose)
        _step("Health", "OK", duration_sec=time.monotonic() - t0, detail=f"status={resp.get('status', '?')}")
        return True
    except Exception as exc:
        _step("Health", "FAIL", duration_sec=time.monotonic() - t0, detail=str(exc))
        traceback.print_exc()
        return False


def run_train(
    client: AICybOpsClient,
    experiment_name: str,
    model_params: Dict[str, Any],
    epochs: int,
    poll_interval_sec: float,
    max_poll_seconds: float,
    verbose: bool,
    run_state: Dict[str, Any],
    run_file_path: Path,
) -> Tuple[bool, Dict[str, Any]]:
    _section("2. Train")
    t0 = time.monotonic()
    training = run_state.get("training", {})
    job_id = training.get("job_id")
    if training.get("status") == "completed" and training.get("result") is not None:
        _step("Train", "OK (cached)", detail="Using result from previous run")
        _log("Cached train result", training.get("result"), verbose)
        return True, training["result"]
    try:
        if not job_id:
            result = client.train_model(
                experiment_name=experiment_name,
                model_type=SUPPORTED_MODEL_TYPE,
                params={},
                epochs=epochs,
                model_params=model_params,
                run_optimization=False,
                wait=False,
            )
            job_id = result.get("job_id")
            if not job_id:
                _step("Train", "FAIL", detail="No job_id in response")
                return False, {}
            print(f"  Job accepted: job_id={job_id}")
            run_state["training"] = {"job_id": job_id, "status": "pending", "result": None}
            _save_run_file(run_state, run_file_path)
        else:
            print(f"  Resuming poll for train job_id={job_id}")
        status, status_resp = _poll_until_done(client.get_job_status, job_id, poll_interval_sec, max_poll_seconds, "train")
        elapsed = time.monotonic() - t0
        if status == "completed":
            _log("Job response", status_resp, verbose)
            res = status_resp.get("result") or {}
            detail = (
                f"models_run={res.get('models_run', '?')} "
                f"models_with_accuracy={res.get('models_with_accuracy', '?')} "
                f"avg_accuracy={res.get('avg_accuracy', '?')}"
            )
            _step("Train", "OK", duration_sec=elapsed, detail=detail)
            run_state["training"] = {"job_id": job_id, "status": "completed", "result": status_resp}
            _save_run_file(run_state, run_file_path)
            return True, status_resp
        if status == "failed":
            _step("Train", "FAIL", duration_sec=elapsed, detail=f"error={status_resp.get('error')}")
            run_state["training"] = {"job_id": job_id, "status": "failed", "result": status_resp}
            _save_run_file(run_state, run_file_path)
            return False, status_resp
        _step("Train", "FAIL", duration_sec=elapsed, detail=f"Polling timed out after {max_poll_seconds}s")
        return False, {}
    except Exception as exc:
        _step("Train", "FAIL", duration_sec=time.monotonic() - t0, detail=str(exc))
        traceback.print_exc()
        return False, {}


def run_predict(client: AICybOpsClient, experiment_name: str, model_params: Dict[str, Any], verbose: bool) -> Tuple[bool, Dict[str, Any]]:
    _section("3. Predict")
    t0 = time.monotonic()
    try:
        result = client.predict(
            experiment_name=experiment_name,
            model_type=SUPPORTED_MODEL_TYPE,
            registered_model_name=SUPPORTED_MODEL_TYPE,
            model_version="latest",
            model_params=model_params,
        )
        elapsed = time.monotonic() - t0
        diagnostics = result.get("prediction_diagnostics") or {}
        detail = f"diagnostic_models={len(diagnostics)}"
        _log("Response", result, verbose)
        _step("Predict", "OK", duration_sec=elapsed, detail=detail)
        return True, result
    except Exception as exc:
        _step("Predict", "FAIL", duration_sec=time.monotonic() - t0, detail=str(exc))
        traceback.print_exc()
        return False, {}


def run_evaluate(
    client: AICybOpsClient,
    experiment_name: str,
    model_params: Dict[str, Any],
    poll_interval_sec: float,
    max_poll_seconds: float,
    verbose: bool,
    run_state: Dict[str, Any],
    run_file_path: Path,
) -> Tuple[bool, Dict[str, Any]]:
    _section("4. Evaluate")
    t0 = time.monotonic()
    evaluation = run_state.get("evaluation", {})
    job_id = evaluation.get("job_id")
    if evaluation.get("status") == "completed" and evaluation.get("result") is not None:
        _step("Evaluate", "OK (cached)", detail="Using result from previous run")
        _log("Cached evaluation result", evaluation.get("result"), verbose)
        return True, evaluation["result"]
    try:
        if not job_id:
            result = client.evaluate(
                experiment_name=experiment_name,
                model_type=SUPPORTED_MODEL_TYPE,
                model_params=model_params,
                evaluation_config={},
                registered_model_name=SUPPORTED_MODEL_TYPE,
                model_version="latest",
                dataset_type="test",
                wait=False,
            )
            job_id = result.get("job_id")
            if not job_id:
                _step("Evaluate", "FAIL", detail="No job_id in response")
                return False, {}
            print(f"  Eval job accepted: job_id={job_id}")
            run_state["evaluation"] = {"job_id": job_id, "status": "pending", "result": None}
            _save_run_file(run_state, run_file_path)
        else:
            print(f"  Resuming poll for eval job_id={job_id}")
        status, status_resp = _poll_until_done(client.get_eval_job_status, job_id, poll_interval_sec, max_poll_seconds, "eval")
        elapsed = time.monotonic() - t0
        if status == "completed":
            _log("Eval job response", status_resp, verbose)
            res = status_resp.get("result") or {}
            summary = res.get("summary") or {}
            detail = (
                f"models_run={summary.get('models_run', '?')} "
                f"models_with_accuracy={summary.get('models_with_accuracy', '?')} "
                f"avg_accuracy={summary.get('avg_accuracy', '?')}"
            )
            _step("Evaluate", "OK", duration_sec=elapsed, detail=detail)
            run_state["evaluation"] = {"job_id": job_id, "status": "completed", "result": status_resp}
            _save_run_file(run_state, run_file_path)
            return True, status_resp
        if status == "failed":
            _step("Evaluate", "FAIL", duration_sec=elapsed, detail=f"error={status_resp.get('error')}")
            run_state["evaluation"] = {"job_id": job_id, "status": "failed", "result": status_resp}
            _save_run_file(run_state, run_file_path)
            return False, status_resp
        _step("Evaluate", "FAIL", duration_sec=elapsed, detail=f"Polling timed out after {max_poll_seconds}s")
        return False, {}
    except Exception as exc:
        _step("Evaluate", "FAIL", duration_sec=time.monotonic() - t0, detail=str(exc))
        traceback.print_exc()
        return False, {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train, predict and evaluate modelos_uc_2 through existing AICybOps endpoints.")
    parser.add_argument("--model-type", default=SUPPORTED_MODEL_TYPE, choices=(SUPPORTED_MODEL_TYPE,), help="Model type key (fixed: modelos_uc2).")
    parser.add_argument("--model-keys", default=None, help=f"Comma-separated modelos_uc_2 keys subset to run. Available: {', '.join(MODELS.keys())}")
    parser.add_argument("--experiment", default="ModelosUC2Demo", help="MLflow experiment name (default: ModelosUC2Demo).")
    parser.add_argument("--epochs", type=int, default=1, help="Nominal epochs passed to /train/.")
    parser.add_argument("--no-evaluate", action="store_true", help="Skip /evaluate/ step.")
    parser.add_argument("--skip-predict", action="store_true", help="Skip /predict/ step.")
    parser.add_argument("--poll-interval", type=float, default=5.0, help="Seconds between job polls.")
    parser.add_argument("--max-poll-seconds", type=float, default=1800.0, help="Max seconds to poll each async job.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Log full request/response payloads.")
    parser.add_argument("--run-file", type=str, default=None, help="Path for resume JSON file.")
    parser.add_argument("--resume", type=str, default=None, help="Resume from a previous run JSON file.")
    parser.add_argument("--resume-run", action="store_true", help="With --resume, continue pending steps (otherwise read-only snapshot).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S", stream=sys.stdout)

    model_keys = None
    if args.model_keys:
        model_keys = [k.strip() for k in args.model_keys.split(",") if k.strip()]
        unknown = [k for k in model_keys if k not in MODELS]
        if unknown:
            print(f"[ERROR] Unknown --model-keys: {unknown}. Available: {list(MODELS.keys())}")
            return 2

    model_params = _model_params_from_keys(model_keys)
    base_url = resolve_aicybops_service_url()

    if args.resume:
        run_file_path = Path(args.resume)
        if not run_file_path.exists():
            print(f"  ERROR: Resume file not found: {run_file_path}")
            return 1
        run_state = _load_run_file(run_file_path)
        experiment_name = run_state.get("experiment", args.experiment)
        epochs = int(run_state.get("epochs", args.epochs))
        skip_predict = bool(run_state.get("skip_predict", args.skip_predict))
        no_evaluate = bool(run_state.get("no_evaluate", args.no_evaluate))
        if run_state.get("model_keys") is not None:
            model_params = _model_params_from_keys(run_state.get("model_keys"))
            model_keys = run_state.get("model_keys")
    else:
        run_file_path = Path(args.run_file) if args.run_file else _default_run_file_path()
        experiment_name = args.experiment
        epochs = args.epochs
        skip_predict = args.skip_predict
        no_evaluate = args.no_evaluate
        run_state = {
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model_type": SUPPORTED_MODEL_TYPE,
            "model_keys": model_keys,
            "experiment": experiment_name,
            "epochs": epochs,
            "skip_predict": skip_predict,
            "no_evaluate": no_evaluate,
            "training": {"job_id": None, "status": "pending", "result": None},
            "prediction": {"status": "pending", "result": None},
            "evaluation": {"job_id": None, "status": "pending", "result": None},
        }
        _save_run_file(run_state, run_file_path)

    _section("AICybOps — modelos_uc_2 Demo Deploy", "=")
    print(f"  Service URL  : {base_url}")
    print(f"  Model type   : {args.model_type}")
    print(f"  Model keys   : {model_keys if model_keys else 'all'}")
    print(f"  Experiment   : {experiment_name}")
    print(f"  Epochs       : {epochs}")
    print(f"  Predict      : {'no  (--skip-predict)' if skip_predict else 'yes'}")
    print(f"  Evaluate     : {'no  (--no-evaluate)' if no_evaluate else 'yes'}")
    print(f"  Run file     : {run_file_path}")
    if args.resume:
        print(f"  Resume mode  : {'continue (--resume-run)' if args.resume_run else 'read-only snapshot'}")

    client = AICybOpsClient(base_url=base_url)
    if args.resume:
        refresh_resume_job_states(client, run_state, run_file_path, args.verbose)
        if not args.resume_run:
            _section("Resume Snapshot", "=")
            print(f"  Training status  : {run_state.get('training', {}).get('status', '?')}")
            print(f"  Training job_id  : {run_state.get('training', {}).get('job_id')}")
            print(f"  Predict status   : {run_state.get('prediction', {}).get('status', '?')}")
            print(f"  Evaluation status: {run_state.get('evaluation', {}).get('status', '?')}")
            print(f"  Evaluation job_id: {run_state.get('evaluation', {}).get('job_id')}")
            print()
            print("  Read-only resume completed. No new requests were made.")
            return 0

    if not run_health(client, args.verbose):
        return 1

    ok, _ = run_train(client, experiment_name, model_params, epochs, args.poll_interval, args.max_poll_seconds, args.verbose, run_state, run_file_path)
    if not ok:
        return 1

    if not skip_predict:
        prediction = run_state.get("prediction", {})
        if prediction.get("status") == "completed" and prediction.get("result") is not None:
            _step("Predict", "OK (cached)", detail="Using result from previous run")
            _log("Cached prediction result", prediction.get("result"), args.verbose)
            ok = True
        else:
            ok, predict_result = run_predict(client, experiment_name, model_params, args.verbose)
            if ok:
                run_state["prediction"] = {"status": "completed", "result": predict_result}
                _save_run_file(run_state, run_file_path)
            else:
                run_state["prediction"] = {"status": "failed", "result": None}
                _save_run_file(run_state, run_file_path)
        if not ok:
            return 1

    if not no_evaluate:
        ok, _ = run_evaluate(client, experiment_name, model_params, args.poll_interval, args.max_poll_seconds, args.verbose, run_state, run_file_path)
        if not ok:
            return 1

    _section("Summary", "=")
    print("  All steps completed successfully.")
    print(f"  Run file: {run_file_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
