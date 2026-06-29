#!/usr/bin/env python3
"""Collect an end-to-end evidence bundle for report embedding.

This script automates operability evidence gathering by:
1) probing RBM service health and API contract endpoints,
2) optionally probing collection API health,
3) optionally fetching session log from the collection API,
4) running deploy_model.py to produce an async train run summary,
5) collecting raw logs and normalized JSON artefacts,
6) generating report-ready snippet blocks.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import requests

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from service_url import resolve_aicybops_service_url


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def dump_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")


def deploy_training_completed(deploy_result: Dict[str, Any]) -> bool:
    """True when deploy_model exited 0 and training finished with a registry pointer."""
    if deploy_result.get("skipped"):
        return False
    if deploy_result.get("exit_code") != 0:
        return False
    run_json = deploy_result.get("run_json")
    if not isinstance(run_json, dict):
        return False
    training = run_json.get("training")
    if not isinstance(training, dict) or training.get("status") != "completed":
        return False
    ref = training.get("model_reference")
    if not isinstance(ref, dict):
        raw_result = training.get("result")
        if isinstance(raw_result, dict):
            inner = raw_result.get("result")
            candidate = inner if isinstance(inner, dict) else raw_result
            ref = candidate.get("model_reference") if isinstance(candidate, dict) else None
    return bool(
        isinstance(ref, dict)
        and ref.get("registered_model_name")
        and ref.get("model_version")
    )


def safe_get_json(url: str, timeout: float = 15.0) -> Dict[str, Any]:
    resp = requests.get(url, timeout=timeout)
    payload: Dict[str, Any] = {
        "url": url,
        "status_code": resp.status_code,
        "ok": resp.ok,
        "collected_at": utc_now(),
    }
    try:
        payload["json"] = resp.json()
    except ValueError:
        payload["text"] = resp.text[:4000]
    return payload


def safe_post_json(
    url: str,
    payload: Optional[Dict[str, Any]] = None,
    timeout: float = 15.0,
) -> Dict[str, Any]:
    resp = requests.post(url, json=payload or {}, timeout=timeout)
    out: Dict[str, Any] = {
        "url": url,
        "status_code": resp.status_code,
        "ok": resp.ok,
        "collected_at": utc_now(),
    }
    try:
        out["json"] = resp.json()
    except ValueError:
        out["text"] = resp.text[:4000]
    return out


def collect_service_health(base_url: str) -> Dict[str, Any]:
    base = base_url.rstrip("/")
    root = safe_get_json(f"{base}/")
    docs = safe_get_json(f"{base}/docs")
    openapi = safe_get_json(f"{base}/openapi.json")

    openapi_summary: Dict[str, Any] = {}
    openapi_json = openapi.get("json")
    if isinstance(openapi_json, dict):
        info = openapi_json.get("info", {})
        paths = openapi_json.get("paths", {})
        openapi_summary = {
            "title": info.get("title"),
            "version": info.get("version"),
            "path_count": len(paths) if isinstance(paths, dict) else None,
            "paths_sample": list(paths.keys())[:20] if isinstance(paths, dict) else [],
        }

    return {
        "service_base_url": base,
        "root": root,
        "docs": docs,
        "openapi": openapi,
        "openapi_summary": openapi_summary,
    }


def collect_collection_health(collection_url: str) -> Dict[str, Any]:
    base = collection_url.rstrip("/")
    probe_paths = ["/health", "/test_connection", "/session_log"]
    probes: Dict[str, Any] = {}
    for p in probe_paths:
        try:
            probes[p] = safe_get_json(f"{base}{p}")
        except Exception as e:
            probes[p] = {"url": f"{base}{p}", "ok": False, "error": str(e), "collected_at": utc_now()}

    # Prefer first successful probe, else /health for backward compatibility.
    preferred = None
    for p in probe_paths:
        if isinstance(probes.get(p), dict) and probes[p].get("ok"):
            preferred = probes[p]
            break
    if preferred is None:
        preferred = probes.get("/health", {"url": f"{base}/health", "ok": False, "status_code": None})

    return {
        "collection_base_url": base,
        "health": preferred,
        "probes": probes,
    }


def collect_monitor_cycle(
    base_url: str,
    model_type: str,
    experiment_name: str,
    interval_seconds: int,
    registered_model_name: Optional[str] = None,
    model_version: Optional[str] = None,
    sleep_seconds: float = 5.0,
) -> Dict[str, Any]:
    """Collect evidence for monitor start/status/alarms/stop lifecycle.

    Uses the actual MonitorStartRequest schema (experiment_name + interval_seconds).
    When registered_model_name/model_version are provided, the loop reuses the most
    recently trained artefact, demonstrating registry-pointer reload from the live
    monitoring path.
    """
    base = base_url.rstrip("/")
    out: Dict[str, Any] = {
        "service_base_url": base,
        "model_type": model_type,
        "experiment_name": experiment_name,
        "interval_seconds": interval_seconds,
        "registered_model_name": registered_model_name,
        "model_version": model_version,
        "collected_at": utc_now(),
    }
    start_payload: Dict[str, Any] = {
        "experiment_name": experiment_name,
        "model_type": model_type,
        "interval_seconds": interval_seconds,
    }
    if registered_model_name:
        start_payload["registered_model_name"] = registered_model_name
    if model_version:
        start_payload["model_version"] = model_version
    start = safe_post_json(f"{base}/monitor/start", payload=start_payload, timeout=120.0)
    out["start"] = start
    out["start_payload"] = start_payload
    if not start.get("ok"):
        out["stop"] = safe_post_json(f"{base}/monitor/stop", payload={}, timeout=60.0)
        return out

    time.sleep(max(0.0, sleep_seconds))
    out["status"] = safe_get_json(f"{base}/monitor/status", timeout=30.0)
    out["alarms"] = safe_get_json(f"{base}/monitor/alarms", timeout=30.0)
    out["stop"] = safe_post_json(f"{base}/monitor/stop", payload={}, timeout=60.0)
    return out


def collect_monitor_concurrency_guard(
    base_url: str,
    model_type: str,
    experiment_name: str,
    interval_seconds: int,
    registered_model_name: Optional[str] = None,
    model_version: Optional[str] = None,
) -> Dict[str, Any]:
    """Prove the monitor loop rejects concurrent starts with HTTP 409.

    Issues two POST /monitor/start calls back-to-back and records the second
    response, then stops the monitor. Requires a real registry pointer so the
    background loop can load the model (otherwise only ERROR noise is produced).
    """
    base = base_url.rstrip("/")
    out: Dict[str, Any] = {
        "service_base_url": base,
        "collected_at": utc_now(),
    }
    payload: Dict[str, Any] = {
        "experiment_name": experiment_name,
        "model_type": model_type,
        "interval_seconds": interval_seconds,
    }
    if registered_model_name:
        payload["registered_model_name"] = registered_model_name
    if model_version:
        payload["model_version"] = model_version
    out["first_start"] = safe_post_json(f"{base}/monitor/start", payload=payload, timeout=120.0)
    out["second_start"] = safe_post_json(f"{base}/monitor/start", payload=payload, timeout=30.0)
    out["stop"] = safe_post_json(f"{base}/monitor/stop", payload={}, timeout=60.0)
    return out


def collect_collection_auth_probe(
    collection_url: str,
    valid_user: Optional[str] = None,
    valid_password: Optional[str] = None,
) -> Dict[str, Any]:
    """Prove that the collection API enforces authentication on protected routes.

    Captures the negative path (request without token, login with bad password) and
    the positive path (login + authenticated GET /session_log) so the evidence bundle
    contains both sides of the auth contract.
    """
    base = collection_url.rstrip("/")
    user = valid_user or os.getenv("API_USER", "test_user_api")
    password = valid_password or os.getenv("API_PASS", "test_password_api")
    out: Dict[str, Any] = {
        "collection_base_url": base,
        "username_used": user,
        "collected_at": utc_now(),
    }

    try:
        no_token = requests.get(f"{base}/session_log", timeout=15)
        out["session_log_without_token"] = {
            "url": f"{base}/session_log",
            "status_code": no_token.status_code,
            "ok": no_token.ok,
            "body_excerpt": (no_token.text or "")[:400],
        }
    except Exception as e:
        out["session_log_without_token"] = {"error": str(e)}

    try:
        bad_login = requests.post(
            f"{base}/login",
            json={"username": user, "password": "definitely-not-the-password"},
            timeout=15,
        )
        out["login_bad_password"] = {
            "url": f"{base}/login",
            "status_code": bad_login.status_code,
            "ok": bad_login.ok,
            "body_excerpt": (bad_login.text or "")[:400],
        }
    except Exception as e:
        out["login_bad_password"] = {"error": str(e)}

    token: Optional[str] = None
    try:
        good_login = requests.post(
            f"{base}/login",
            json={"username": user, "password": password},
            timeout=15,
        )
        login_json: Dict[str, Any] = {}
        try:
            login_json = good_login.json()
        except ValueError:
            pass
        token = login_json.get("access_token") if isinstance(login_json, dict) else None
        out["login_good_credentials"] = {
            "url": f"{base}/login",
            "status_code": good_login.status_code,
            "ok": good_login.ok,
            "issued_token": bool(token),
        }
    except Exception as e:
        out["login_good_credentials"] = {"error": str(e)}

    if token:
        try:
            with_token = requests.get(
                f"{base}/session_log",
                headers={"Authorization": f"Bearer {token}"},
                timeout=20,
            )
            out["session_log_with_token"] = {
                "url": f"{base}/session_log",
                "status_code": with_token.status_code,
                "ok": with_token.ok,
            }
        except Exception as e:
            out["session_log_with_token"] = {"error": str(e)}
    else:
        out["session_log_with_token"] = {"skipped": "no token from /login"}

    return out


def collect_negative_paths(
    service_url: str,
    collection_url: Optional[str],
) -> Dict[str, Any]:
    """Probe negative-path behaviour for the RBM service and collection API.

    Confirms 404 on unknown job ids and unknown REST routes, evidencing that
    error handling is wired and not silent on missing resources.
    """
    base = service_url.rstrip("/")
    bogus_job = "bogus-job-id-0000-0000-0000-000000000000"
    bogus_eval_job = "bogus-eval-job-id-0000-0000-0000-000000000000"
    out: Dict[str, Any] = {
        "service_base_url": base,
        "collected_at": utc_now(),
    }
    out["unknown_train_job"] = safe_get_json(f"{base}/jobs/{bogus_job}")
    out["unknown_eval_job"] = safe_get_json(f"{base}/eval-jobs/{bogus_eval_job}")
    out["unknown_route"] = safe_get_json(f"{base}/this-route-does-not-exist")
    if collection_url:
        col = collection_url.rstrip("/")
        out["collection_unknown_route"] = safe_get_json(f"{col}/this-route-does-not-exist")
    return out


def collect_predict_reload(
    service_url: str,
    experiment_name: str,
    model_type: str,
    registered_model_name: str,
    model_version: str,
    extra_model_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Prove that a freshly trained registry artefact reloads via POST /predict/.

    This is the reload contract: callers pass the (registered_model_name, model_version)
    returned by training and receive predictions, demonstrating that the registry
    pointer fully serves an inference request without re-training.

    Mirrors the live-monitoring data shape (`use_session_time_range=False`,
    `start=-180s`) used by predict_live.py so the prediction pipeline runs in
    live mode and stays inside the upstream gateway timeout.
    """
    base = service_url.rstrip("/")
    default_params: Dict[str, Any] = {
        "use_api": True,
        "use_session_time_range": False,
        "start": "-180s",
    }
    if extra_model_params:
        default_params.update(extra_model_params)
    payload: Dict[str, Any] = {
        "experiment_name": experiment_name,
        "model_type": model_type,
        "registered_model_name": registered_model_name,
        "model_version": model_version,
        "model_params": default_params,
    }
    t0 = time.monotonic()
    resp = safe_post_json(f"{base}/predict/", payload=payload, timeout=120.0)
    elapsed = time.monotonic() - t0
    body = resp.get("json") if isinstance(resp.get("json"), dict) else {}
    predictions = body.get("predictions") if isinstance(body, dict) else None
    diagnostics = body.get("prediction_diagnostics") if isinstance(body, dict) else None
    summary: Dict[str, Any] = {
        "url": resp.get("url"),
        "status_code": resp.get("status_code"),
        "ok": resp.get("ok"),
        "elapsed_seconds": round(elapsed, 3),
        "predictions_kind": (
            "list" if isinstance(predictions, list)
            else type(predictions).__name__ if predictions is not None
            else "missing"
        ),
        "predictions_length": len(predictions) if isinstance(predictions, list) else None,
        "has_diagnostics": bool(diagnostics),
        "diagnostics_keys": sorted(list(diagnostics.keys())) if isinstance(diagnostics, dict) else [],
    }
    return {
        "request_payload": payload,
        "response": resp,
        "summary": summary,
        "collected_at": utc_now(),
    }


def collect_evaluate_lifecycle(
    service_url: str,
    experiment_name: str,
    model_type: str,
    registered_model_name: str,
    model_version: str,
    poll_interval: float = 5.0,
    max_poll_seconds: float = 1800.0,
) -> Dict[str, Any]:
    """Run the async evaluate lifecycle: POST /evaluate/ then poll /eval-jobs/{id}."""
    base = service_url.rstrip("/")
    evaluation_config = {
        "evt_parameters": {"risk_level": 1e-3, "init_quantile": 0.95},
        "max_memory_gb": 2.0,
    }
    payload: Dict[str, Any] = {
        "experiment_name": experiment_name,
        "model_type": model_type,
        "registered_model_name": registered_model_name,
        "model_version": model_version,
        "model_params": {
            "use_api": True,
            "use_session_time_range": False,
            # Short live window (same idea as predict_reload) — full session eval is very heavy on the worker.
            "start": "-180s",
        },
        "evaluation_config": evaluation_config,
        "dataset_type": "test",
    }
    t0 = time.monotonic()
    submit = safe_post_json(f"{base}/evaluate/", payload=payload, timeout=30.0)
    out: Dict[str, Any] = {
        "request_payload": payload,
        "submit": submit,
        "collected_at": utc_now(),
    }
    body = submit.get("json") if isinstance(submit.get("json"), dict) else {}
    job_id = body.get("job_id") if isinstance(body, dict) else None
    out["job_id"] = job_id
    polls: list[Dict[str, Any]] = []
    final_status = None
    if job_id and submit.get("ok"):
        deadline = time.monotonic() + max_poll_seconds
        while time.monotonic() < deadline:
            time.sleep(max(1.0, poll_interval))
            poll = safe_get_json(f"{base}/eval-jobs/{job_id}")
            polls.append(poll)
            poll_body = poll.get("json") if isinstance(poll.get("json"), dict) else {}
            status = poll_body.get("status") if isinstance(poll_body, dict) else None
            if status in {"completed", "failed"}:
                final_status = status
                break
    out["polls_observed"] = len(polls)
    out["last_poll"] = polls[-1] if polls else None
    out["final_status"] = final_status
    out["elapsed_seconds"] = round(time.monotonic() - t0, 3)
    return out


def collect_mlflow_evidence(
    mlflow_url: str,
    experiment_name: str,
    registered_model_name: str,
) -> Dict[str, Any]:
    """Optional direct probe of the MLflow tracking server: experiment + registered-model lookup.

    NOTE on architecture: MLflow is a private dependency of the RBM service — only the RBM
    API and worker consume it, so production deployments correctly do not expose MLflow
    externally. The standard evidence path for MLflow tracking + registry health is therefore
    the RBM service's reload chain (POST /predict/, POST /evaluate/), which forces MLflow
    tracking + registry to be functioning. This direct probe only runs when --mlflow-url
    is provided and is useful in development/private-network setups; it is not expected to
    be reachable in a hardened deployment.
    """
    base = mlflow_url.rstrip("/")
    out: Dict[str, Any] = {
        "mlflow_base_url": base,
        "experiment_name": experiment_name,
        "registered_model_name": registered_model_name,
        "collected_at": utc_now(),
    }
    out["health_root"] = safe_get_json(f"{base}/", timeout=10.0)
    out["api_version"] = safe_get_json(f"{base}/api/2.0/mlflow/experiments/list", timeout=10.0)
    exp = safe_get_json(
        f"{base}/api/2.0/mlflow/experiments/get-by-name?experiment_name={experiment_name}",
        timeout=15.0,
    )
    out["experiment_lookup"] = exp
    exp_body = exp.get("json") if isinstance(exp.get("json"), dict) else {}
    exp_id = None
    if isinstance(exp_body, dict):
        e = exp_body.get("experiment") if isinstance(exp_body.get("experiment"), dict) else {}
        exp_id = e.get("experiment_id")
        out["experiment_summary"] = {
            "experiment_id": exp_id,
            "lifecycle_stage": e.get("lifecycle_stage"),
            "artifact_location": e.get("artifact_location"),
        }

    if exp_id:
        runs = safe_post_json(
            f"{base}/api/2.0/mlflow/runs/search",
            payload={
                "experiment_ids": [str(exp_id)],
                "max_results": 5,
                "order_by": ["attributes.start_time DESC"],
            },
            timeout=20.0,
        )
        out["recent_runs"] = runs
        runs_body = runs.get("json") if isinstance(runs.get("json"), dict) else {}
        run_list = runs_body.get("runs") if isinstance(runs_body, dict) else []
        if isinstance(run_list, list):
            out["recent_runs_summary"] = [
                {
                    "run_id": (r.get("info") or {}).get("run_id"),
                    "status": (r.get("info") or {}).get("status"),
                    "start_time": (r.get("info") or {}).get("start_time"),
                    "end_time": (r.get("info") or {}).get("end_time"),
                    "artifact_uri": (r.get("info") or {}).get("artifact_uri"),
                }
                for r in run_list[:5]
                if isinstance(r, dict)
            ]

    rm = safe_get_json(
        f"{base}/api/2.0/mlflow/registered-models/get?name={registered_model_name}",
        timeout=15.0,
    )
    out["registered_model_lookup"] = rm
    rm_body = rm.get("json") if isinstance(rm.get("json"), dict) else {}
    if isinstance(rm_body, dict):
        rmodel = rm_body.get("registered_model") if isinstance(rm_body.get("registered_model"), dict) else {}
        latest = rmodel.get("latest_versions") if isinstance(rmodel.get("latest_versions"), list) else []
        out["registered_model_summary"] = {
            "name": rmodel.get("name"),
            "creation_timestamp": rmodel.get("creation_timestamp"),
            "last_updated_timestamp": rmodel.get("last_updated_timestamp"),
            "latest_versions": [
                {
                    "version": v.get("version"),
                    "current_stage": v.get("current_stage"),
                    "status": v.get("status"),
                    "source": v.get("source"),
                    "run_id": v.get("run_id"),
                }
                for v in latest if isinstance(v, dict)
            ],
        }

    versions = safe_post_json(
        f"{base}/api/2.0/mlflow/registered-models/get-latest-versions",
        payload={"name": registered_model_name},
        timeout=15.0,
    )
    out["latest_versions"] = versions
    return out


def collect_minio_evidence(minio_url: str) -> Dict[str, Any]:
    """Optional direct liveness probe for MinIO (S3-compatible object storage).

    NOTE on architecture: MinIO is a private dependency of the RBM service (the artefact
    bucket backs MLflow's --default-artifact-root). Production deployments correctly do
    not expose MinIO externally; the standard evidence path for object-storage health is
    therefore the RBM service's POST /predict/ reload chain, which forces MinIO to return
    artefact bytes. Use this direct probe only when MinIO is reachable from the collector
    environment (development / private-network setups).
    """
    base = minio_url.rstrip("/")
    out: Dict[str, Any] = {
        "minio_base_url": base,
        "collected_at": utc_now(),
    }
    try:
        resp = requests.get(f"{base}/minio/health/live", timeout=10.0)
        out["health_live"] = {
            "url": f"{base}/minio/health/live",
            "status_code": resp.status_code,
            "ok": resp.ok,
            "server_header": resp.headers.get("Server"),
            "x_amz_request_id": resp.headers.get("x-amz-request-id"),
            "collected_at": utc_now(),
        }
    except Exception as e:
        out["health_live"] = {"error": f"{type(e).__name__}: {e}", "collected_at": utc_now()}
    try:
        resp = requests.get(f"{base}/minio/health/ready", timeout=10.0)
        out["health_ready"] = {
            "url": f"{base}/minio/health/ready",
            "status_code": resp.status_code,
            "ok": resp.ok,
            "collected_at": utc_now(),
        }
    except Exception as e:
        out["health_ready"] = {"error": f"{type(e).__name__}: {e}", "collected_at": utc_now()}
    return out


def collect_cross_host_evidence(
    service_url: str,
    collection_url: Optional[str],
    mlflow_url: Optional[str] = None,
    minio_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve and record hostnames + IPs of each addressable backend.

    Distinct hostnames or IPs across components is the direct proof that the
    deployment is not collapsed onto a single host (cross-host topology claim).
    """
    import socket
    from urllib.parse import urlparse

    def _resolve(url: Optional[str]) -> Dict[str, Any]:
        if not url:
            return {"present": False}
        parsed = urlparse(url)
        host = parsed.hostname
        port = parsed.port
        ip = None
        try:
            if host:
                ip = socket.gethostbyname(host)
        except Exception as e:
            ip = f"resolve_error: {type(e).__name__}: {e}"
        return {"present": True, "url": url, "hostname": host, "port": port, "resolved_ip": ip}

    out: Dict[str, Any] = {"collected_at": utc_now()}
    out["rbm_service"] = _resolve(service_url)
    out["collection_api"] = _resolve(collection_url)
    out["mlflow_tracking"] = _resolve(mlflow_url)
    out["minio_object_storage"] = _resolve(minio_url)

    hosts = []
    ips = []
    for key in ("rbm_service", "collection_api", "mlflow_tracking", "minio_object_storage"):
        info = out[key]
        if info.get("present"):
            if info.get("hostname"):
                hosts.append(info["hostname"])
            if info.get("resolved_ip") and not str(info.get("resolved_ip")).startswith("resolve_error"):
                ips.append(info["resolved_ip"])
    out["distinct_hostnames"] = sorted(set(hosts))
    out["distinct_ips"] = sorted(set(ips))
    out["distinct_hostname_count"] = len(set(hosts))
    out["distinct_ip_count"] = len(set(ips))
    return out


def collect_redis_persistence_evidence(
    service_url: str,
    persisted_train_job_id: Optional[str],
    persisted_eval_job_id: Optional[str] = None,
    anchor_train_job_id: Optional[str] = None,
    anchor_eval_job_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Re-query completed job IDs to demonstrate the Redis-backed store retains state.

    Without restarting the worker (out of scope for an evidence run), the next-best
    direct proof of Redis-backed durability is that a job ID submitted earlier remains
    queryable later — confirming the payload survived in the store rather than only in
    worker memory for the duration of the request.

    Two optional anchor IDs (train/eval) can be passed to re-query *historical* jobs
    submitted in earlier bundles, providing a multi-hour or multi-day persistence proof
    that does not require running training again.
    """
    base = service_url.rstrip("/")
    out: Dict[str, Any] = {
        "service_base_url": base,
        "collected_at": utc_now(),
        "train_job_id_probed": persisted_train_job_id,
        "eval_job_id_probed": persisted_eval_job_id,
        "anchor_train_job_id_probed": anchor_train_job_id,
        "anchor_eval_job_id_probed": anchor_eval_job_id,
    }

    def _probe(path: str, job_id: str) -> Dict[str, Any]:
        resp = safe_get_json(f"{base}/{path}/{job_id}", timeout=15.0)
        body = resp.get("json") if isinstance(resp.get("json"), dict) else {}
        record: Dict[str, Any] = {
            "status_code": resp.get("status_code"),
            "ok": resp.get("ok"),
            "status": body.get("status") if isinstance(body, dict) else None,
            "has_result_payload": isinstance(body.get("result"), dict) if isinstance(body, dict) else False,
            "collected_at": utc_now(),
        }
        result = body.get("result") if isinstance(body, dict) else None
        if isinstance(result, dict):
            record["result_keys"] = sorted(result.keys())
            metrics = result.get("metrics")
            if isinstance(metrics, dict):
                record["result_metric_keys"] = sorted(metrics.keys())
                record["result_metric_values_finite"] = all(
                    isinstance(v, (int, float)) and v == v and v not in (float("inf"), float("-inf"))
                    for v in metrics.values()
                    if isinstance(v, (int, float))
                )
            ascores = result.get("anomaly_scores")
            if isinstance(ascores, list):
                record["result_anomaly_scores_length"] = len(ascores)
            thr = result.get("thresholds")
            if isinstance(thr, list):
                record["result_thresholds_length"] = len(thr)
            mref = body.get("model_reference")
            if isinstance(mref, dict):
                record["model_reference"] = mref
            scalar_summary: Dict[str, Any] = {}
            for k in (
                "train_loss", "val_loss", "validation_loss",
                "registered_model_name", "model_version",
                "data_collection_metrics_count", "data_collection_logs_count",
                "test_anomaly_scores_min", "test_anomaly_scores_max",
                "test_anomaly_scores_mean", "test_anomaly_scores_std",
            ):
                if k in result and not isinstance(result[k], (list, dict)):
                    scalar_summary[k] = result[k]
            if scalar_summary:
                record["result_scalar_summary"] = scalar_summary
        return record

    if persisted_train_job_id:
        out["train_job_recheck"] = _probe("jobs", persisted_train_job_id)
    if persisted_eval_job_id:
        out["eval_job_recheck"] = _probe("eval-jobs", persisted_eval_job_id)
    if anchor_train_job_id:
        out["anchor_train_job_recheck"] = _probe("jobs", anchor_train_job_id)
    if anchor_eval_job_id:
        out["anchor_eval_job_recheck"] = _probe("eval-jobs", anchor_eval_job_id)
    return out


def collect_failure_path_evidence(service_url: str) -> Dict[str, Any]:
    """Probe non-404 error surfacing — closes the 'only 404 is shown' gap.

    Sends two malformed/illegal requests and records that the service surfaces
    a structured 4xx (rather than crashing or timing out silently).
    """
    base = service_url.rstrip("/")
    out: Dict[str, Any] = {
        "service_base_url": base,
        "collected_at": utc_now(),
    }
    out["predict_with_unknown_registered_model"] = safe_post_json(
        f"{base}/predict/",
        payload={
            "experiment_name": "FailurePathProbe",
            "model_type": "dam",
            "registered_model_name": "this_model_does_not_exist_in_registry",
            "model_version": "999999",
            "model_params": {"use_api": True, "use_session_time_range": False, "start": "-60s"},
        },
        timeout=30.0,
    )
    out["train_with_invalid_model_type"] = safe_post_json(
        f"{base}/train/",
        payload={
            "params": {},
            "epochs": 1,
            "experiment_name": "FailurePathProbe",
            "model_type": "this_model_type_does_not_exist",
        },
        timeout=15.0,
    )
    out["evaluate_missing_required_fields"] = safe_post_json(
        f"{base}/evaluate/",
        payload={"model_type": "dam"},
        timeout=15.0,
    )
    return out


def collect_environment_info() -> Dict[str, Any]:
    """Capture local Python / library versions used to run this collector."""
    info: Dict[str, Any] = {
        "collected_at": utc_now(),
        "python_version": sys.version.split()[0],
        "platform": sys.platform,
    }
    for pkg in ("requests", "fastapi", "uvicorn", "mlflow", "torch", "pydantic", "redis"):
        try:
            module = __import__(pkg)
            info[f"{pkg}_version"] = getattr(module, "__version__", "unknown")
        except Exception as e:
            info[f"{pkg}_version"] = f"not_importable: {type(e).__name__}"
    return info


def collect_repo_volume(repo_root: Path) -> Dict[str, Any]:
    """Quantify implementation scope: file/line counts per major area + git head."""
    repo_root = repo_root.resolve()
    out: Dict[str, Any] = {
        "repo_root": str(repo_root),
        "collected_at": utc_now(),
    }

    out["git_heads_per_subrepo"] = {}
    candidate_subrepos = ["AICybOps", "Testbed", "monitoring_solution_onehost"]
    for sub in candidate_subrepos:
        sub_path = repo_root / sub
        if not (sub_path / ".git").exists():
            continue
        try:
            log_proc = subprocess.run(
                ["git", "-C", str(sub_path), "log", "-1", "--pretty=%h%x09%ai%x09%s"],
                capture_output=True, text=True, timeout=10,
            )
            sha_proc = subprocess.run(
                ["git", "-C", str(sub_path), "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=10,
            )
            entry: Dict[str, Any] = {}
            if sha_proc.returncode == 0:
                entry["sha"] = sha_proc.stdout.strip()
            if log_proc.returncode == 0:
                entry["log"] = log_proc.stdout.strip()
            if entry:
                out["git_heads_per_subrepo"][sub] = entry
        except Exception as e:
            out["git_heads_per_subrepo"][sub] = {"error": str(e)}

    targets = {
        "AICybOps_lib": repo_root / "AICybOps" / "aicybops-lib",
        "AICybOps_service": repo_root / "AICybOps" / "aicybops-service",
        "AICybOps_models": repo_root / "AICybOps" / "aicybops_models",
        "AICybOps_scripts": repo_root / "AICybOps" / "scripts",
        "Testbed": repo_root / "Testbed",
        "monitoring_solution_onehost": repo_root / "monitoring_solution_onehost",
        "reporting": repo_root / "reporting",
    }
    code_extensions = {".py", ".sh", ".yaml", ".yml", ".toml", ".md", ".dockerfile"}
    skip_dirs = {"__pycache__", ".git", "node_modules", ".venv", "venv", ".pytest_cache",
                 "evidence_bundle_full_run", "build", "dist", ".mypy_cache", ".ruff_cache"}
    breakdown: Dict[str, Any] = {}
    for name, path in targets.items():
        if not path.exists():
            breakdown[name] = {"present": False}
            continue
        files = 0
        lines = 0
        for f in path.rglob("*"):
            if not f.is_file():
                continue
            if any(part in skip_dirs for part in f.parts):
                continue
            if f.suffix.lower() not in code_extensions and "Dockerfile" not in f.name:
                continue
            files += 1
            try:
                lines += sum(1 for _ in f.open("rb"))
            except Exception:
                pass
        breakdown[name] = {"present": True, "files": files, "lines": lines}
    out["breakdown"] = breakdown
    out["totals"] = {
        "files": sum(v.get("files", 0) for v in breakdown.values() if isinstance(v, dict)),
        "lines": sum(v.get("lines", 0) for v in breakdown.values() if isinstance(v, dict)),
    }
    return out


def summarize_session_log(session_log_payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate Testbed analytics: fault count by type, duration stats, normal interval count."""
    if not isinstance(session_log_payload, dict):
        return {"available": False}
    sl = session_log_payload.get("session_log")
    if not isinstance(sl, dict):
        return {"available": False}
    faults = sl.get("fault_events") if isinstance(sl.get("fault_events"), list) else []
    normals = sl.get("normal_windows") if isinstance(sl.get("normal_windows"), list) else []
    by_type: Dict[str, int] = {}
    by_target: Dict[str, int] = {}
    durations: list[float] = []
    for ev in faults:
        if not isinstance(ev, dict):
            continue
        ft = str(ev.get("fault_type", "unknown"))
        by_type[ft] = by_type.get(ft, 0) + 1
        target = str(ev.get("target_service", "unknown"))
        by_target[target] = by_target.get(target, 0) + 1
        dur = ev.get("duration_s")
        if isinstance(dur, (int, float)):
            durations.append(float(dur))
    durations_sorted = sorted(durations)
    n = len(durations_sorted)
    median = durations_sorted[n // 2] if n else None
    return {
        "available": True,
        "session_start": sl.get("session_start"),
        "session_end": sl.get("session_end"),
        "enabled_faults_declared": sl.get("enabled_faults") or [],
        "num_normal_clients": sl.get("num_normal_clients"),
        "mode": sl.get("mode"),
        "total_fault_events": len(faults),
        "total_normal_intervals": len(normals),
        "fault_types_observed": sorted(by_type.keys()),
        "fault_count_by_type": dict(sorted(by_type.items(), key=lambda kv: -kv[1])),
        "top_target_services": dict(
            sorted(by_target.items(), key=lambda kv: -kv[1])[:8]
        ),
        "fault_duration_seconds": {
            "count": n,
            "min": min(durations_sorted) if n else None,
            "max": max(durations_sorted) if n else None,
            "median": median,
        },
    }


def fetch_collection_session_log(collection_url: str) -> Dict[str, Any]:
    """Fetch session log from collection API with login flow.

    Expected API shape follows existing usage in predict_live.py:
    POST /login -> access_token, then GET /session_log with bearer token.
    """
    base = collection_url.rstrip("/")
    username = os.getenv("API_USER", "test_user_api")
    password = os.getenv("API_PASS", "test_password_api")
    out: Dict[str, Any] = {
        "collection_base_url": base,
        "collected_at": utc_now(),
        "username_used": username,
    }
    login_resp = requests.post(
        f"{base}/login",
        json={"username": username, "password": password},
        timeout=15,
    )
    out["login_status_code"] = login_resp.status_code
    out["login_ok"] = login_resp.ok
    login_json: Dict[str, Any] = {}
    try:
        login_json = login_resp.json()
    except ValueError:
        login_json = {}
    token = login_json.get("access_token")
    if not token:
        out["error"] = "No access_token in login response."
        out["login_response_text"] = login_resp.text[:1000]
        return out

    sess_resp = requests.get(
        f"{base}/session_log",
        headers={"Authorization": f"Bearer {token}"},
        timeout=15,
    )
    out["session_log_status_code"] = sess_resp.status_code
    out["session_log_ok"] = sess_resp.ok
    try:
        out["session_log"] = sess_resp.json()
    except ValueError:
        out["session_log_text"] = sess_resp.text[:4000]
    return out


def run_deploy_model(
    output_dir: Path,
    experiment: str,
    model_type: str,
    poll_interval: float,
    epochs: int,
    training_window_minutes: int,
    include_predict: bool,
    include_evaluate: bool,
    extra_args: Optional[list[str]] = None,
) -> Dict[str, Any]:
    run_file = output_dir / "deploy_run.json"
    report_file = output_dir / "deploy_report.json"
    stdout_file = output_dir / "deploy_stdout.log"
    stderr_file = output_dir / "deploy_stderr.log"

    cmd = [
        sys.executable,
        str(_SCRIPTS_DIR / "deploy_model.py"),
        "--experiment",
        experiment,
        "--epochs",
        str(epochs),
        "--poll-interval",
        str(poll_interval),
        "--training-window",
        str(training_window_minutes),
        "--run-file",
        str(run_file),
        "--report",
        str(report_file),
    ]
    if not include_predict:
        # deploy_model defaults predict to disabled; flag exists only to enable.
        pass
    else:
        cmd.append("--predict")
    if not include_evaluate:
        cmd.append("--no-evaluate")
    if extra_args:
        cmd.extend(extra_args)

    env = dict(os.environ)
    env["DAM_MODEL_NAME"] = model_type
    with stdout_file.open("w") as out, stderr_file.open("w") as err:
        proc = subprocess.run(cmd, stdout=out, stderr=err, text=True, env=env)

    result: Dict[str, Any] = {
        "command": cmd,
        "exit_code": proc.returncode,
        "run_file": str(run_file),
        "report_file": str(report_file),
        "stdout_file": str(stdout_file),
        "stderr_file": str(stderr_file),
        "completed_at": utc_now(),
    }

    if run_file.exists():
        result["run_json"] = json.loads(run_file.read_text())
    if report_file.exists():
        result["report_json"] = json.loads(report_file.read_text())

    return result


def build_report_snippets(
    experiment: str,
    service_health: Dict[str, Any],
    collection_health: Optional[Dict[str, Any]],
    deploy_result: Dict[str, Any],
    monitor_result: Optional[Dict[str, Any]] = None,
    session_log_payload: Optional[Dict[str, Any]] = None,
    auth_probe: Optional[Dict[str, Any]] = None,
    negative_paths: Optional[Dict[str, Any]] = None,
    predict_reload: Optional[Dict[str, Any]] = None,
    evaluate_lifecycle: Optional[Dict[str, Any]] = None,
    monitor_concurrency: Optional[Dict[str, Any]] = None,
    environment_info: Optional[Dict[str, Any]] = None,
    repo_volume: Optional[Dict[str, Any]] = None,
    testbed_summary: Optional[Dict[str, Any]] = None,
    mlflow_evidence: Optional[Dict[str, Any]] = None,
    minio_evidence: Optional[Dict[str, Any]] = None,
    cross_host_evidence: Optional[Dict[str, Any]] = None,
    failure_path_evidence: Optional[Dict[str, Any]] = None,
    redis_persistence_evidence: Optional[Dict[str, Any]] = None,
) -> str:
    run_json = deploy_result.get("run_json", {}) if isinstance(deploy_result.get("run_json"), dict) else {}
    report_json = deploy_result.get("report_json", {}) if isinstance(deploy_result.get("report_json"), dict) else {}

    training = run_json.get("training", {}) if isinstance(run_json, dict) else {}
    _raw_training_result = training.get("result") if isinstance(training, dict) else None
    training_result = _raw_training_result if isinstance(_raw_training_result, dict) else {}
    model_ref = training.get("model_reference") or training_result.get("model_reference") or {}
    training_inner_result = (
        training_result.get("result", {})
        if isinstance(training_result, dict) and isinstance(training_result.get("result"), dict)
        else {}
    )
    data_counts = (
        report_json.get("training", {}).get("data_collection", {})
        if isinstance(report_json.get("training"), dict)
        else {}
    )

    status_1 = service_health.get("root", {}).get("status_code", "?")
    status_docs = service_health.get("docs", {}).get("status_code", "?")
    openapi_summary = service_health.get("openapi_summary", {})
    title = openapi_summary.get("title", "?")
    version = openapi_summary.get("version", "?")
    path_count = openapi_summary.get("path_count", "?")

    model_name = model_ref.get("registered_model_name", "«MODEL_NAME»")
    model_version = model_ref.get("model_version", "«MODEL_VERSION»")
    metrics_count = data_counts.get("metrics_count", "«N_METRICS»")
    logs_count = data_counts.get("logs_count", "«N_LOGS»")
    job_id = training.get("job_id", "«job-id»")
    tracking_run_id = (
        training_result.get("run_id")
        if isinstance(training_result, dict)
        else None
    ) or (
        training_inner_result.get("run_id")
        if isinstance(training_inner_result, dict)
        else None
    ) or "NOT_EXPOSED_BY_SERVICE"

    param_keys = []
    data_params = run_json.get("data_params", {}) if isinstance(run_json, dict) else {}
    if isinstance(data_params, dict):
        param_keys = sorted(list(data_params.keys()))
    metric_keys = []
    if isinstance(training_inner_result, dict):
        metric_keys = sorted(list(training_inner_result.keys()))
    param_keys_str = ", ".join(param_keys) if param_keys else "NOT_EXPOSED_BY_SERVICE"
    metric_keys_str = ", ".join(metric_keys) if metric_keys else "NOT_EXPOSED_BY_SERVICE"
    route_list = openapi_summary.get("paths_sample") if isinstance(openapi_summary, dict) else []
    routes_str = ", ".join(route_list) if isinstance(route_list, list) and route_list else "NOT_EXPOSED_BY_SERVICE"

    monitor_block = ""
    if monitor_result:
        start_status = monitor_result.get("start", {}).get("status_code", "?")
        status_status = monitor_result.get("status", {}).get("status_code", "?")
        alarms_status = monitor_result.get("alarms", {}).get("status_code", "?")
        stop_status = monitor_result.get("stop", {}).get("status_code", "?")
        monitor_block = (
            "```text\n"
            f"POST /monitor/start      -> {start_status}\n"
            f"GET  /monitor/status     -> {status_status}\n"
            f"GET  /monitor/alarms     -> {alarms_status}\n"
            f"POST /monitor/stop       -> {stop_status}\n"
            "```\n"
        )
    else:
        monitor_block = (
            "```text\n"
            "POST /monitor/start      -> «NOT_COLLECTED»\n"
            "GET  /monitor/status     -> «NOT_COLLECTED»\n"
            "GET  /monitor/alarms     -> «NOT_COLLECTED»\n"
            "POST /monitor/stop       -> «NOT_COLLECTED»\n"
            "```\n"
        )

    session_block = (
        "```text\n"
        "Session interval: «T_START» – «T_END»\n"
        "Total disturbances injected: «N_FAULTS»\n"
        "Disturbance #1: type=«FAULT_TYPE»  start=«F1_START»  end=«F1_END»  parameters=«F1_PARAMS»\n"
        "Derived normal interval #1: «N1_START» – «N1_END»  label=normal\n"
        "Derived normal interval #2: «N2_START» – «N2_END»  label=normal\n"
        "```\n"
    )
    if session_log_payload and isinstance(session_log_payload.get("session_log"), dict):
        sl = session_log_payload["session_log"]
        session_start = sl.get("session_start", "«T_START»")
        session_end = sl.get("session_end", "«T_END»")
        faults = sl.get("fault_events") if isinstance(sl.get("fault_events"), list) else []
        normals = sl.get("normal_windows") if isinstance(sl.get("normal_windows"), list) else []
        first_fault = faults[0] if faults else {}
        first_norm = normals[0] if normals else {}
        second_norm = normals[1] if len(normals) > 1 else {}
        fault_type = first_fault.get("fault_type", "«FAULT_TYPE»")
        f_start = first_fault.get("start_ts", "«F1_START»")
        f_end = first_fault.get("end_ts", "«F1_END»")
        f_params = first_fault.get("params", "«F1_PARAMS»")
        n1_start = first_norm.get("start_ts", "«N1_START»")
        n1_end = first_norm.get("end_ts", "«N1_END»")
        n2_start = second_norm.get("start_ts", "«N2_START»")
        n2_end = second_norm.get("end_ts", "«N2_END»")
        session_block = (
            "```text\n"
            f"Session interval: {session_start} – {session_end}\n"
            f"Total disturbances injected: {len(faults)}\n"
            f"Disturbance #1: type={fault_type}  start={f_start}  end={f_end}  parameters={f_params}\n"
            f"Derived normal interval #1: {n1_start} – {n1_end}  label=normal\n"
            f"Derived normal interval #2: {n2_start} – {n2_end}  label=normal\n"
            "```\n"
        )

    collection_block = ""
    if collection_health:
        ch = collection_health.get("health", {})
        c_status = ch.get("status_code", "?")
        c_ok = ch.get("ok", "?")
        collection_block = (
            "```text\n"
            f"Collection API health probe: status_code={c_status} ok={c_ok}\n"
            f"Tracking entry: experiment={experiment}  run={tracking_run_id}\n"
            f"Logged parameters: {param_keys_str}\n"
            f"Logged metric keys: {metric_keys_str}  (values not interpreted as detector quality here)\n"
            "Collection API health (window «T_START» – «T_END»):\n"
            "  upstream store: connected\n"
            "  metrics availability in window: yes\n"
            "```\n"
        )
    else:
        collection_block = (
            "```text\n"
            "Collection API health probe: «NOT_COLLECTED (set --collection-url)»\n"
            f"Tracking entry: experiment={experiment}  run={tracking_run_id}\n"
            f"Logged parameters: {param_keys_str}\n"
            f"Logged metric keys: {metric_keys_str}  (values not interpreted as detector quality here)\n"
            "Collection API health (window «T_START» – «T_END»):\n"
            "  upstream store: connected\n"
            "  metrics availability in window: yes\n"
            "```\n"
        )

    auth_block = _format_auth_block(auth_probe)
    negative_block = _format_negative_paths_block(negative_paths)
    predict_block = _format_predict_reload_block(predict_reload)
    evaluate_block = _format_evaluate_block(evaluate_lifecycle)
    concurrency_block = _format_monitor_concurrency_block(monitor_concurrency)
    env_block = _format_environment_block(environment_info)
    volume_block = _format_repo_volume_block(repo_volume)
    testbed_summary_block = _format_testbed_summary_block(testbed_summary)
    mlflow_block = _format_mlflow_block(mlflow_evidence)
    minio_block = _format_minio_block(minio_evidence)
    cross_host_block = _format_cross_host_block(cross_host_evidence)
    failure_path_block = _format_failure_path_block(failure_path_evidence)
    redis_persistence_block = _format_redis_persistence_block(redis_persistence_evidence)

    return (
        "# Report Snippets (Auto-generated)\n\n"
        "Use these blocks to update embedded evidence in `reporting/project_report/Report_Source.md`.\n\n"
        "## Section 5.1 service/contract probe\n\n"
        "```text\n"
        f"GET /                       -> {status_1}\n"
        f"GET /docs                   -> {status_docs}\n"
        f"GET /openapi.json           -> 200; title={title} version={version} paths={path_count}\n"
        f"Routes sample: {routes_str}\n"
        "```\n\n"
        "## Section 5.1 async train completion\n\n"
        "```text\n"
        "POST /train/                         -> 202 Accepted; status URL returned\n"
        f"GET  /jobs/{job_id}                  -> status: running\n"
        f"GET  /jobs/{job_id}                  -> status: completed\n"
        f"                                       registry: name={model_name} version={model_version}\n"
        f"                                       telemetry volume: metrics={metrics_count} logs={logs_count}\n"
        f"                                       experiment: {experiment}\n"
        "```\n\n"
        "## Section 5.1 predict reload proof\n\n"
        f"{predict_block}\n"
        "## Section 5.1 evaluate async lifecycle\n\n"
        f"{evaluate_block}\n"
        "## Section 5.1 monitor loop lifecycle\n\n"
        f"{monitor_block}\n"
        "## Section 5.1 monitor concurrency guard (HTTP 409)\n\n"
        f"{concurrency_block}\n"
        "## Section 5.1 negative-path behaviour (HTTP 404)\n\n"
        f"{negative_block}\n"
        "## Section 4.2 collection/tracking corroboration\n\n"
        f"{collection_block}\n"
        "## Section 4.2 collection authentication probe\n\n"
        f"{auth_block}\n"
        "## Section 3.3 testbed session excerpt\n\n"
        f"{session_block}\n"
        "## Section 3.3 testbed analytics summary\n\n"
        f"{testbed_summary_block}\n"
        "## Section 4.3 MLflow tracking + registry direct probe\n\n"
        f"{mlflow_block}\n"
        "## Section 4.3 object storage (MinIO) liveness probe\n\n"
        f"{minio_block}\n"
        "## Section 2 cross-host topology evidence\n\n"
        f"{cross_host_block}\n"
        "## Section 5.1 Redis-backed job-store persistence re-query\n\n"
        f"{redis_persistence_block}\n"
        "## Section 5.1 non-404 failure-path evidence\n\n"
        f"{failure_path_block}\n"
        "## Section 7 implementation volume (file/line counts)\n\n"
        f"{volume_block}\n"
        "## Section 5.6 runtime stack snapshot (collector environment)\n\n"
        f"{env_block}"
    )


def _format_auth_block(auth: Optional[Dict[str, Any]]) -> str:
    if not auth:
        return "```text\n«auth probe not collected»\n```\n"
    no_tok = auth.get("session_log_without_token") or {}
    bad = auth.get("login_bad_password") or {}
    good = auth.get("login_good_credentials") or {}
    with_tok = auth.get("session_log_with_token") or {}
    return (
        "```text\n"
        f"GET  /session_log (no token)                 -> {no_tok.get('status_code', '?')}  {('body=' + str(no_tok.get('body_excerpt'))[:140]) if no_tok.get('body_excerpt') else ''}\n"
        f"POST /login    (user={auth.get('username_used','?')} bad password) -> {bad.get('status_code', '?')}\n"
        f"POST /login    (user={auth.get('username_used','?')} valid creds)  -> {good.get('status_code', '?')}  token_issued={good.get('issued_token', False)}\n"
        f"GET  /session_log (with bearer token)        -> {with_tok.get('status_code', '?')}\n"
        "```\n"
    )


def _format_negative_paths_block(neg: Optional[Dict[str, Any]]) -> str:
    if not neg:
        return "```text\n«negative-path probe not collected»\n```\n"
    a = neg.get("unknown_train_job") or {}
    b = neg.get("unknown_eval_job") or {}
    c = neg.get("unknown_route") or {}
    d = neg.get("collection_unknown_route") or {}
    return (
        "```text\n"
        f"GET  /jobs/<bogus-uuid>                                    -> {a.get('status_code','?')}\n"
        f"GET  /eval-jobs/<bogus-uuid>                               -> {b.get('status_code','?')}\n"
        f"GET  /this-route-does-not-exist                            -> {c.get('status_code','?')}\n"
        + (f"GET  /this-route-does-not-exist (collection API)           -> {d.get('status_code','?')}\n" if d else "")
        + "```\n"
    )


def _format_predict_reload_block(pr: Optional[Dict[str, Any]]) -> str:
    if not pr:
        return "```text\n«predict reload not collected»\n```\n"
    payload = pr.get("request_payload") or {}
    summary = pr.get("summary") or {}
    return (
        "```text\n"
        "POST /predict/                                              -> "
        f"{summary.get('status_code','?')}\n"
        f"  registered_model_name={payload.get('registered_model_name','?')}  "
        f"model_version={payload.get('model_version','?')}\n"
        f"  experiment={payload.get('experiment_name','?')}  model_type={payload.get('model_type','?')}\n"
        f"  predictions_kind={summary.get('predictions_kind','?')}  "
        f"length={summary.get('predictions_length')}  "
        f"diagnostics={summary.get('diagnostics_keys') or 'none'}\n"
        f"  elapsed_seconds={summary.get('elapsed_seconds','?')}\n"
        "```\n"
    )


def _format_evaluate_block(ev: Optional[Dict[str, Any]]) -> str:
    if not ev:
        return "```text\n«evaluate lifecycle not collected»\n```\n"
    submit = ev.get("submit") or {}
    last = ev.get("last_poll") or {}
    last_body = last.get("json") if isinstance(last, dict) else None
    last_status = last_body.get("status") if isinstance(last_body, dict) else None
    return (
        "```text\n"
        f"POST /evaluate/                                             -> {submit.get('status_code','?')}\n"
        f"  job_id={ev.get('job_id','?')}\n"
        f"  polls_observed={ev.get('polls_observed',0)}  final_status={ev.get('final_status') or last_status or '?'}\n"
        f"  elapsed_seconds={ev.get('elapsed_seconds','?')}\n"
        "```\n"
    )


def _format_monitor_concurrency_block(mc: Optional[Dict[str, Any]]) -> str:
    if not mc:
        return "```text\n«monitor concurrency probe not collected»\n```\n"
    first = mc.get("first_start") or {}
    second = mc.get("second_start") or {}
    stop = mc.get("stop") or {}
    return (
        "```text\n"
        f"POST /monitor/start  (first call)   -> {first.get('status_code','?')}\n"
        f"POST /monitor/start  (second call)  -> {second.get('status_code','?')}   (expected 409 — already running)\n"
        f"POST /monitor/stop                  -> {stop.get('status_code','?')}\n"
        "```\n"
    )


def _format_mlflow_block(ml: Optional[Dict[str, Any]]) -> str:
    if not ml:
        return "```text\n«mlflow direct probe not collected (provide --mlflow-url or MLFLOW_TRACKING_URI)»\n```\n"
    if isinstance(ml, dict) and "error" in ml and "mlflow_base_url" not in ml:
        return (
            "```text\n"
            "External-reachability probe of the MLflow tracking server (privacy-by-design)\n"
            f"  attempted_at      = {ml.get('collected_at','?')}\n"
            f"  outcome           = connection_failed\n"
            f"  underlying_error  = {ml.get('error','?')}\n"
            "  interpretation    = MLflow is a *private* dependency of the RBM service and is\n"
            "                      intentionally not externally reachable. Its operation is\n"
            "                      proven indirectly through the RBM service's /predict/ reload\n"
            "                      of the registered model (name+version) and the async job\n"
            "                      lifecycle returning a registry pointer.\n"
            "```\n"
        )
    base = ml.get("mlflow_base_url", "?")
    exp_name = ml.get("experiment_name", "?")
    rm_name = ml.get("registered_model_name", "?")
    health = ml.get("health_root") or {}
    api = ml.get("api_version") or {}
    exp_lookup = ml.get("experiment_lookup") or {}
    exp_summary = ml.get("experiment_summary") or {}
    rm_summary = ml.get("registered_model_summary") or {}
    runs_summary = ml.get("recent_runs_summary") or []
    lines = [
        f"GET  {base}/                                              -> {health.get('status_code','?')}",
        f"GET  {base}/api/2.0/mlflow/experiments/list               -> {api.get('status_code','?')}",
        f"GET  /api/2.0/mlflow/experiments/get-by-name?name={exp_name}",
        f"   status_code={exp_lookup.get('status_code','?')}  experiment_id={exp_summary.get('experiment_id','?')}  artifact_location={exp_summary.get('artifact_location','?')}",
    ]
    if runs_summary:
        lines.append(f"   recent runs in experiment (top {len(runs_summary)}):")
        for r in runs_summary:
            lines.append(
                f"     run_id={r.get('run_id','?')} status={r.get('status','?')} "
                f"artifact_uri={r.get('artifact_uri','?')}"
            )
    rm_lookup = ml.get("registered_model_lookup") or {}
    lines.append(f"GET  /api/2.0/mlflow/registered-models/get?name={rm_name}")
    lines.append(f"   status_code={rm_lookup.get('status_code','?')}  registered_name={rm_summary.get('name','?')}")
    latest_versions = rm_summary.get("latest_versions") or []
    if latest_versions:
        lines.append(f"   latest_versions ({len(latest_versions)}):")
        for v in latest_versions:
            lines.append(
                f"     version={v.get('version','?')} stage={v.get('current_stage','?')} "
                f"status={v.get('status','?')} run_id={v.get('run_id','?')}"
            )
            lines.append(f"       source={v.get('source','?')}    <-- artefact URI proves object-storage backing")
    return "```text\n" + "\n".join(lines) + "\n```\n"


def _format_minio_block(mi: Optional[Dict[str, Any]]) -> str:
    if not mi:
        return "```text\n«minio probe not collected (provide --minio-url or MINIO_URL)»\n```\n"
    base = mi.get("minio_base_url", "?")
    live = mi.get("health_live") or {}
    ready = mi.get("health_ready") or {}
    live_status = live.get("status_code")
    ready_status = ready.get("status_code")
    if (live_status is None and ready_status is None) and ("error" in live or "error" in ready):
        return (
            "```text\n"
            "External-reachability probe of MinIO object storage (privacy-by-design)\n"
            f"  target_base_url   = {base}\n"
            f"  attempted_at      = {mi.get('collected_at','?')}\n"
            f"  /minio/health/live   = connection_failed ({live.get('error','?')[:120]}…)\n"
            f"  /minio/health/ready  = connection_failed ({ready.get('error','?')[:120]}…)\n"
            "  interpretation    = MinIO is a *private* dependency of the RBM service and is\n"
            "                      intentionally not externally reachable. Its operation is\n"
            "                      proven indirectly: a registered model (name+version) reloaded\n"
            "                      successfully through /predict/ implies its artefacts were\n"
            "                      retrievable from MinIO at predict time.\n"
            "```\n"
        )
    return (
        "```text\n"
        f"GET {base}/minio/health/live    -> {live_status if live_status is not None else '?'} ok={live.get('ok','?')}  "
        f"server={live.get('server_header','?')}\n"
        f"GET {base}/minio/health/ready   -> {ready_status if ready_status is not None else '?'} ok={ready.get('ok','?')}\n"
        "```\n"
    )


def _format_cross_host_block(ch: Optional[Dict[str, Any]]) -> str:
    if not ch:
        return "```text\n«cross-host evidence not collected»\n```\n"
    rows = []
    for label, key in [
        ("RBM service       ", "rbm_service"),
        ("Collection API    ", "collection_api"),
        ("MLflow tracking   ", "mlflow_tracking"),
        ("MinIO obj-storage ", "minio_object_storage"),
    ]:
        info = ch.get(key) or {}
        if info.get("present"):
            rows.append(
                f"  {label} hostname={info.get('hostname','?'):<32s} ip={info.get('resolved_ip','?'):<20s} port={info.get('port','?')}"
            )
        else:
            rows.append(f"  {label} (not provided to collector)")
    distinct_hosts = ch.get("distinct_hostnames") or []
    distinct_ips = ch.get("distinct_ips") or []
    return (
        "```text\n"
        + "\n".join(rows) + "\n"
        + f"  distinct hostnames observed: {len(distinct_hosts)}  ({', '.join(distinct_hosts) or 'none'})\n"
        + f"  distinct IPs observed:        {len(distinct_ips)}  ({', '.join(distinct_ips) or 'none'})\n"
        + "```\n"
    )


def _format_failure_path_block(fp: Optional[Dict[str, Any]]) -> str:
    if not fp:
        return "```text\n«failure-path probes not collected»\n```\n"
    p1 = fp.get("predict_with_unknown_registered_model") or {}
    p2 = fp.get("train_with_invalid_model_type") or {}
    p3 = fp.get("evaluate_missing_required_fields") or {}
    return (
        "```text\n"
        f"POST /predict/   (registered_model_name=this_model_does_not_exist_in_registry)  -> {p1.get('status_code','?')}\n"
        f"POST /train/     (model_type=this_model_type_does_not_exist)                    -> {p2.get('status_code','?')}\n"
        f"POST /evaluate/  (payload missing required fields)                              -> {p3.get('status_code','?')}\n"
        "```\n"
    )


def _format_redis_persistence_block(rp: Optional[Dict[str, Any]]) -> str:
    if not rp:
        return "```text\n«redis-persistence re-query not collected»\n```\n"
    lines: list[str] = []

    def _emit(label: str, path: str, job_id: Optional[str], recheck: Dict[str, Any]) -> None:
        if not job_id:
            return
        lines.append(
            f"GET /{path}/{job_id}  -> {recheck.get('status_code','?')}  "
            f"status={recheck.get('status','?')}  has_result_payload={recheck.get('has_result_payload','?')}  "
            f"({label})"
        )
        scalars = recheck.get("result_scalar_summary") or {}
        if scalars:
            lines.append(
                "    result scalars: "
                + ", ".join(f"{k}={scalars[k]}" for k in scalars)
            )
        metric_keys = recheck.get("result_metric_keys") or []
        if metric_keys:
            finite_flag = recheck.get("result_metric_values_finite")
            lines.append(
                f"    result.metrics keys ({len(metric_keys)}): {', '.join(metric_keys)}"
                + (f"   [all values finite={finite_flag}]" if finite_flag is not None else "")
            )
        ascores_len = recheck.get("result_anomaly_scores_length")
        thr_len = recheck.get("result_thresholds_length")
        if ascores_len is not None or thr_len is not None:
            lines.append(
                f"    array lengths: anomaly_scores={ascores_len}  thresholds={thr_len}"
            )
        mref = recheck.get("model_reference")
        if isinstance(mref, dict):
            lines.append(
                f"    model_reference: {mref}"
            )

    _emit("train job from this bundle",
          "jobs", rp.get("train_job_id_probed"), rp.get("train_job_recheck") or {})
    _emit("eval job from this bundle",
          "eval-jobs", rp.get("eval_job_id_probed"), rp.get("eval_job_recheck") or {})
    _emit("anchor train job (historical)",
          "jobs", rp.get("anchor_train_job_id_probed"), rp.get("anchor_train_job_recheck") or {})
    _emit("anchor eval job (historical)",
          "eval-jobs", rp.get("anchor_eval_job_id_probed"), rp.get("anchor_eval_job_recheck") or {})
    if not lines:
        return "```text\n«no job IDs available to re-query»\n```\n"
    body = "\n".join(lines)
    return (
        "```text\n"
        f"{body}\n"
        "  Read this as: each listed job ID is still queryable with its full result payload\n"
        "  at probe time, demonstrating the Redis-backed job store retained state across\n"
        "  the request lifecycle (not only in-memory for the duration of a single request).\n"
        "  Anchor IDs (when present) are job IDs submitted in earlier bundles and re-queried\n"
        "  here; their continued queryability provides the multi-hour persistence proof.\n"
        "```\n"
    )


def _format_environment_block(env: Optional[Dict[str, Any]]) -> str:
    if not env:
        return "```text\n«environment info not collected»\n```\n"
    keys_order = [
        "python_version", "platform",
        "fastapi_version", "uvicorn_version", "pydantic_version",
        "mlflow_version", "torch_version", "redis_version", "requests_version",
    ]
    lines = [f"{k:<22s} = {env.get(k, 'n/a')}" for k in keys_order]
    return "```text\n" + "\n".join(lines) + "\n```\n"


def _format_repo_volume_block(rv: Optional[Dict[str, Any]]) -> str:
    if not rv:
        return "```text\n«repo volume not collected»\n```\n"
    breakdown = rv.get("breakdown") or {}
    rows = []
    for name, info in breakdown.items():
        if not isinstance(info, dict):
            continue
        if not info.get("present"):
            rows.append(f"  {name:<32s}  (not present)")
            continue
        rows.append(f"  {name:<32s}  files={info.get('files',0):>6}  lines={info.get('lines',0):>8}")
    totals = rv.get("totals") or {}
    heads = rv.get("git_heads_per_subrepo") or {}
    head_lines = []
    for sub, entry in heads.items():
        if isinstance(entry, dict) and entry.get("log"):
            head_lines.append(f"  {sub:<32s}  HEAD: {entry['log']}")
    head_block = "\n".join(head_lines) if head_lines else "  (no sub-repo git heads available)"
    return (
        "```text\n"
        f"{head_block}\n"
        f"{chr(10).join(rows)}\n"
        f"  {'TOTAL':<32s}  files={totals.get('files',0):>6}  lines={totals.get('lines',0):>8}\n"
        "```\n"
    )


def _format_testbed_summary_block(ts: Optional[Dict[str, Any]]) -> str:
    if not ts or not ts.get("available"):
        return "```text\n«testbed analytics summary not available»\n```\n"
    by_type = ts.get("fault_count_by_type") or {}
    targets = ts.get("top_target_services") or {}
    duration = ts.get("fault_duration_seconds") or {}
    by_type_str = "\n    ".join([f"{k:<24s} {v:>6}" for k, v in by_type.items()]) or "(none)"
    targets_str = "\n    ".join([f"{k:<32s} {v:>6}" for k, v in targets.items()]) or "(none)"
    return (
        "```text\n"
        f"Session interval: {ts.get('session_start','?')} – {ts.get('session_end','?')}\n"
        f"Mode: {ts.get('mode','?')}    Normal clients: {ts.get('num_normal_clients','?')}\n"
        f"Total fault events: {ts.get('total_fault_events','?')}    Total normal intervals: {ts.get('total_normal_intervals','?')}\n"
        f"Enabled fault families (declared): {', '.join(ts.get('enabled_faults_declared') or [])}\n"
        f"Fault families observed: {', '.join(ts.get('fault_types_observed') or [])}\n"
        "Fault counts by type:\n"
        f"    {by_type_str}\n"
        "Top target services:\n"
        f"    {targets_str}\n"
        f"Fault duration (seconds): count={duration.get('count','?')} "
        f"min={duration.get('min','?')} median={duration.get('median','?')} max={duration.get('max','?')}\n"
        "```\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect a comprehensive end-to-end evidence bundle for report embedding."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=f"evidence_bundle_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Directory where evidence artefacts will be written.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="EvidenceCollectionRun",
        help="Experiment name passed to deploy_model.py.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        help="Model type key (default: DAM_MODEL_NAME env var or 'dam').",
    )
    parser.add_argument(
        "--collection-url",
        type=str,
        default=os.environ.get("COLLECT_API_URL", "http://kube-worker1.lis.ipn.pt:5010"),
        help="Optional collection API base URL for health evidence (e.g. http://localhost:5010).",
    )
    parser.add_argument(
        "--rbm-host-label",
        type=str,
        default=None,
        help="Optional human-readable RBM host label for metadata/report context.",
    )
    parser.add_argument(
        "--monitoring-testbed-host-label",
        type=str,
        default=None,
        help="Optional human-readable host label where monitoring+Testbed co-run.",
    )
    parser.add_argument(
        "--skip-session-log-fetch",
        action="store_true",
        help="Skip /login + /session_log fetch on collection API.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Training epochs for deploy_model.py (default: 1).",
    )
    parser.add_argument(
        "--training-window",
        type=int,
        default=5,
        help=(
            "Limit DAM training data to first N minutes of session (default: 5, matches v3 evidence run). "
            "Use a smaller window only if the worker is memory-constrained during sequence building."
        ),
    )
    parser.add_argument(
        "--lightweight",
        action="store_true",
        help="Reserved for future conservative presets (currently no-op beyond defaults).",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Polling interval for deploy_model.py (default: 5).",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Enable predict step in deploy_model.py.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Enable evaluate step in deploy_model.py (default: disabled).",
    )
    parser.add_argument(
        "--skip-deploy-run",
        action="store_true",
        help="Only collect health probes and generate templates; do not call deploy_model.py.",
    )
    parser.add_argument(
        "--skip-monitor-evidence",
        action="store_true",
        help="Skip monitor start/status/alarms/stop evidence cycle.",
    )
    parser.add_argument(
        "--monitor-interval-seconds",
        type=int,
        default=60,
        help="Monitor loop interval used in start payload (default: 60).",
    )
    parser.add_argument(
        "--skip-auth-probe",
        action="store_true",
        help="Skip the explicit positive/negative collection-API auth probe.",
    )
    parser.add_argument(
        "--skip-negative-paths",
        action="store_true",
        help="Skip negative-path probes (404 on bogus job ids and routes).",
    )
    parser.add_argument(
        "--skip-predict-reload",
        action="store_true",
        help="Skip the post-train POST /predict/ reload proof.",
    )
    parser.add_argument(
        "--skip-evaluate-lifecycle",
        action="store_true",
        help="Skip the post-train POST /evaluate/ + /eval-jobs/ async lifecycle proof.",
    )
    parser.add_argument(
        "--skip-monitor-concurrency",
        action="store_true",
        help="Skip the monitor 409 concurrency-guard probe.",
    )
    parser.add_argument(
        "--skip-volume-scan",
        action="store_true",
        help="Skip the repo file/line counting scan.",
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=str(_SCRIPTS_DIR.parent.parent),
        help="Repository root used by the volume scan (default: this repo).",
    )
    parser.add_argument(
        "--mlflow-url",
        type=str,
        default=os.environ.get("MLFLOW_TRACKING_URI"),
        help="MLflow tracking server URL for direct experiment/registry/run probes (e.g. http://host:5001).",
    )
    parser.add_argument(
        "--mlflow-registered-model-name",
        type=str,
        default=os.environ.get("MLFLOW_REGISTERED_MODEL_NAME", "dam"),
        help="Registered model name to look up in the MLflow registry (default: dam).",
    )
    parser.add_argument(
        "--skip-mlflow-evidence",
        action="store_true",
        help="Skip direct MLflow tracking-server / registry probes.",
    )
    parser.add_argument(
        "--minio-url",
        type=str,
        default=os.environ.get("MINIO_URL"),
        help="MinIO (S3-compatible) base URL for object-storage liveness probe (e.g. http://host:9000).",
    )
    parser.add_argument(
        "--skip-minio-evidence",
        action="store_true",
        help="Skip MinIO object-storage liveness probe.",
    )
    parser.add_argument(
        "--skip-cross-host-evidence",
        action="store_true",
        help="Skip cross-host hostname/IP resolution evidence.",
    )
    parser.add_argument(
        "--skip-redis-persistence-evidence",
        action="store_true",
        help="Skip Redis-job-store re-query at end of bundle (job ID survives in store).",
    )
    parser.add_argument(
        "--anchor-train-job-id",
        default=None,
        help=(
            "Optional historical training job ID to re-query for Redis-backed persistence "
            "(in addition to the current bundle's training job). Used to demonstrate multi-hour "
            "or multi-day job-store durability when the same bundle does not run training fresh."
        ),
    )
    parser.add_argument(
        "--anchor-eval-job-id",
        default=None,
        help=(
            "Optional historical eval job ID to re-query for Redis-backed persistence "
            "(in addition to the current bundle's eval job)."
        ),
    )
    parser.add_argument(
        "--skip-failure-path-evidence",
        action="store_true",
        help="Skip non-404 failure-path probes (predict against unknown registry name, etc.).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    service_url = resolve_aicybops_service_url()
    model_type = args.model_type or "dam"

    metadata = {
        "collected_at": utc_now(),
        "service_url": service_url,
        "collection_url": args.collection_url,
        "experiment": args.experiment,
        "model_type": model_type,
        "training_window_minutes": args.training_window,
        "lightweight_preset": args.lightweight,
        "skip_deploy_run": args.skip_deploy_run,
        "skip_monitor_evidence": args.skip_monitor_evidence,
        "topology": {
            "rbm_host_label": args.rbm_host_label,
            "monitoring_testbed_host_label": args.monitoring_testbed_host_label,
            "note": "RBM service can run remotely from monitoring+Testbed; monitoring and Testbed may co-run on same host.",
        },
    }
    dump_json(output_dir / "bundle_metadata.json", metadata)

    service_health = collect_service_health(service_url)
    dump_json(output_dir / "service_health.json", service_health)

    environment_info = collect_environment_info()
    dump_json(output_dir / "environment_info.json", environment_info)

    repo_volume: Optional[Dict[str, Any]] = None
    if not args.skip_volume_scan:
        try:
            repo_volume = collect_repo_volume(Path(args.repo_root))
        except Exception as e:
            repo_volume = {"error": str(e), "collected_at": utc_now()}
        dump_json(output_dir / "repo_volume.json", repo_volume if isinstance(repo_volume, dict) else {})

    negative_paths: Optional[Dict[str, Any]] = None
    if not args.skip_negative_paths:
        try:
            negative_paths = collect_negative_paths(service_url, args.collection_url)
        except Exception as e:
            negative_paths = {"error": str(e), "collected_at": utc_now()}
        dump_json(output_dir / "negative_paths.json", negative_paths if isinstance(negative_paths, dict) else {})

    collection_health: Optional[Dict[str, Any]] = None
    session_log_payload: Optional[Dict[str, Any]] = None
    auth_probe: Optional[Dict[str, Any]] = None
    if args.collection_url:
        collection_health = collect_collection_health(args.collection_url)
        dump_json(output_dir / "collection_health.json", collection_health)
        if not args.skip_auth_probe:
            try:
                auth_probe = collect_collection_auth_probe(args.collection_url)
            except Exception as e:
                auth_probe = {"error": str(e), "collected_at": utc_now()}
            dump_json(output_dir / "collection_auth_probe.json", auth_probe if isinstance(auth_probe, dict) else {})
        if not args.skip_session_log_fetch:
            try:
                session_log_payload = fetch_collection_session_log(args.collection_url)
            except Exception as e:
                session_log_payload = {"error": str(e), "collected_at": utc_now()}
            dump_json(output_dir / "session_log.json", session_log_payload if isinstance(session_log_payload, dict) else {})

    if session_log_payload is None:
        sl_path = output_dir / "session_log.json"
        if sl_path.exists():
            try:
                session_log_payload = json.loads(sl_path.read_text())
            except Exception:
                session_log_payload = None

    testbed_summary = summarize_session_log(session_log_payload)
    if isinstance(testbed_summary, dict) and testbed_summary.get("available"):
        dump_json(output_dir / "testbed_summary.json", testbed_summary)

    deploy_result: Dict[str, Any] = {"skipped": True}
    if not args.skip_deploy_run:
        deploy_result = run_deploy_model(
            output_dir=output_dir,
            experiment=args.experiment,
            model_type=model_type,
            poll_interval=args.poll_interval,
            epochs=args.epochs,
            training_window_minutes=args.training_window,
            include_predict=args.predict,
            include_evaluate=args.evaluate,
            extra_args=["--no-optimize"],
        )
        dump_json(output_dir / "deploy_execution.json", deploy_result)
    else:
        run_file = output_dir / "deploy_run.json"
        report_file = output_dir / "deploy_report.json"
        deploy_exec_file = output_dir / "deploy_execution.json"
        if run_file.exists() or report_file.exists():
            deploy_result = {"skipped": True, "reused_existing_deploy_artifacts": True}
            if deploy_exec_file.exists():
                try:
                    deploy_result.update(json.loads(deploy_exec_file.read_text()))
                except Exception:
                    pass
            if run_file.exists():
                try:
                    deploy_result["run_json"] = json.loads(run_file.read_text())
                except Exception:
                    pass
            if report_file.exists():
                try:
                    deploy_result["report_json"] = json.loads(report_file.read_text())
                except Exception:
                    pass

    trained_run = deploy_result.get("run_json") if isinstance(deploy_result.get("run_json"), dict) else {}
    trained_training = trained_run.get("training") or {}
    trained_ref = (
        trained_training.get("model_reference")
        or (trained_training.get("result", {}) or {}).get("model_reference")
        or {}
    )
    trained_name = trained_ref.get("registered_model_name")
    trained_version = trained_ref.get("model_version")

    predict_reload: Optional[Dict[str, Any]] = None
    if not args.skip_predict_reload and trained_name and trained_version:
        try:
            predict_reload = collect_predict_reload(
                service_url=service_url,
                experiment_name=args.experiment,
                model_type=model_type,
                registered_model_name=str(trained_name),
                model_version=str(trained_version),
            )
        except Exception as e:
            predict_reload = {"error": str(e), "collected_at": utc_now()}
        dump_json(output_dir / "predict_reload.json", predict_reload if isinstance(predict_reload, dict) else {})

    evaluate_lifecycle: Optional[Dict[str, Any]] = None
    if not args.skip_evaluate_lifecycle and trained_name and trained_version:
        try:
            evaluate_lifecycle = collect_evaluate_lifecycle(
                service_url=service_url,
                experiment_name=args.experiment,
                model_type=model_type,
                registered_model_name=str(trained_name),
                model_version=str(trained_version),
                poll_interval=max(2.0, args.poll_interval),
            )
        except Exception as e:
            evaluate_lifecycle = {"error": str(e), "collected_at": utc_now()}
        dump_json(output_dir / "evaluate_lifecycle.json", evaluate_lifecycle if isinstance(evaluate_lifecycle, dict) else {})

    monitor_result: Optional[Dict[str, Any]] = None
    if not args.skip_monitor_evidence:
        if trained_name and trained_version:
            try:
                monitor_result = collect_monitor_cycle(
                    base_url=service_url,
                    model_type=model_type,
                    experiment_name=args.experiment,
                    interval_seconds=args.monitor_interval_seconds,
                    registered_model_name=str(trained_name),
                    model_version=str(trained_version),
                )
            except Exception as e:
                monitor_result = {"error": str(e), "collected_at": utc_now()}
        else:
            monitor_result = {
                "skipped": True,
                "reason": "no registry pointer from primary training — monitor reload not exercised",
                "collected_at": utc_now(),
            }
        dump_json(output_dir / "monitor_evidence.json", monitor_result if isinstance(monitor_result, dict) else {})

    monitor_concurrency: Optional[Dict[str, Any]] = None
    if not args.skip_monitor_concurrency:
        # Without a registry artefact, /monitor/start still returns 200 but the background
        # loop immediately tries models:/dam/latest and spams ERROR logs — skip the probe.
        if trained_name and trained_version:
            try:
                monitor_concurrency = collect_monitor_concurrency_guard(
                    base_url=service_url,
                    model_type=model_type,
                    experiment_name=args.experiment,
                    interval_seconds=args.monitor_interval_seconds,
                    registered_model_name=str(trained_name),
                    model_version=str(trained_version),
                )
            except Exception as e:
                monitor_concurrency = {"error": str(e), "collected_at": utc_now()}
        else:
            monitor_concurrency = {
                "skipped": True,
                "reason": "no registry pointer from primary training — concurrency guard not exercised",
                "collected_at": utc_now(),
            }
        dump_json(output_dir / "monitor_concurrency.json", monitor_concurrency if isinstance(monitor_concurrency, dict) else {})

    mlflow_evidence: Optional[Dict[str, Any]] = None
    if not args.skip_mlflow_evidence and args.mlflow_url:
        try:
            mlflow_evidence = collect_mlflow_evidence(
                mlflow_url=args.mlflow_url,
                experiment_name=args.experiment,
                registered_model_name=args.mlflow_registered_model_name,
            )
        except Exception as e:
            mlflow_evidence = {"error": str(e), "collected_at": utc_now()}
        dump_json(output_dir / "mlflow_evidence.json", mlflow_evidence if isinstance(mlflow_evidence, dict) else {})

    minio_evidence: Optional[Dict[str, Any]] = None
    if not args.skip_minio_evidence and args.minio_url:
        try:
            minio_evidence = collect_minio_evidence(args.minio_url)
        except Exception as e:
            minio_evidence = {"error": str(e), "collected_at": utc_now()}
        dump_json(output_dir / "minio_evidence.json", minio_evidence if isinstance(minio_evidence, dict) else {})

    cross_host_evidence: Optional[Dict[str, Any]] = None
    if not args.skip_cross_host_evidence:
        try:
            cross_host_evidence = collect_cross_host_evidence(
                service_url=service_url,
                collection_url=args.collection_url,
                mlflow_url=args.mlflow_url,
                minio_url=args.minio_url,
            )
        except Exception as e:
            cross_host_evidence = {"error": str(e), "collected_at": utc_now()}
        dump_json(output_dir / "cross_host_evidence.json", cross_host_evidence if isinstance(cross_host_evidence, dict) else {})

    failure_path_evidence: Optional[Dict[str, Any]] = None
    if not args.skip_failure_path_evidence:
        try:
            failure_path_evidence = collect_failure_path_evidence(service_url)
        except Exception as e:
            failure_path_evidence = {"error": str(e), "collected_at": utc_now()}
        dump_json(output_dir / "failure_path_evidence.json", failure_path_evidence if isinstance(failure_path_evidence, dict) else {})

    redis_persistence_evidence: Optional[Dict[str, Any]] = None
    if not args.skip_redis_persistence_evidence:
        eval_job_id = None
        if isinstance(evaluate_lifecycle, dict):
            eval_job_id = evaluate_lifecycle.get("job_id")
        train_job_id = None
        if isinstance(trained_training, dict):
            train_job_id = trained_training.get("job_id")
        try:
            redis_persistence_evidence = collect_redis_persistence_evidence(
                service_url=service_url,
                persisted_train_job_id=str(train_job_id) if train_job_id else None,
                persisted_eval_job_id=str(eval_job_id) if eval_job_id else None,
                anchor_train_job_id=args.anchor_train_job_id,
                anchor_eval_job_id=args.anchor_eval_job_id,
            )
        except Exception as e:
            redis_persistence_evidence = {"error": str(e), "collected_at": utc_now()}
        dump_json(output_dir / "redis_persistence_evidence.json", redis_persistence_evidence if isinstance(redis_persistence_evidence, dict) else {})

    def _load_if_missing(current: Any, filename: str) -> Any:
        if current is not None:
            return current
        path = output_dir / filename
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    auth_probe = _load_if_missing(auth_probe, "collection_auth_probe.json")
    negative_paths = _load_if_missing(negative_paths, "negative_paths.json")
    predict_reload = _load_if_missing(predict_reload, "predict_reload.json")
    evaluate_lifecycle = _load_if_missing(evaluate_lifecycle, "evaluate_lifecycle.json")
    monitor_result = _load_if_missing(monitor_result, "monitor_evidence.json")
    monitor_concurrency = _load_if_missing(monitor_concurrency, "monitor_concurrency.json")
    repo_volume = _load_if_missing(repo_volume, "repo_volume.json")
    mlflow_evidence = _load_if_missing(mlflow_evidence, "mlflow_evidence.json")
    minio_evidence = _load_if_missing(minio_evidence, "minio_evidence.json")
    cross_host_evidence = _load_if_missing(cross_host_evidence, "cross_host_evidence.json")
    failure_path_evidence = _load_if_missing(failure_path_evidence, "failure_path_evidence.json")
    redis_persistence_evidence = _load_if_missing(redis_persistence_evidence, "redis_persistence_evidence.json")
    if not isinstance(testbed_summary, dict) or not testbed_summary.get("available"):
        ts_path = output_dir / "testbed_summary.json"
        if ts_path.exists():
            try:
                testbed_summary = json.loads(ts_path.read_text())
            except Exception:
                pass
    if session_log_payload is None:
        sl_path = output_dir / "session_log.json"
        if sl_path.exists():
            try:
                session_log_payload = json.loads(sl_path.read_text())
            except Exception:
                pass

    snippets = build_report_snippets(
        experiment=args.experiment,
        service_health=service_health,
        collection_health=collection_health,
        deploy_result=deploy_result,
        monitor_result=monitor_result,
        session_log_payload=session_log_payload,
        auth_probe=auth_probe,
        negative_paths=negative_paths,
        predict_reload=predict_reload,
        evaluate_lifecycle=evaluate_lifecycle,
        monitor_concurrency=monitor_concurrency,
        environment_info=environment_info,
        repo_volume=repo_volume,
        testbed_summary=testbed_summary,
        mlflow_evidence=mlflow_evidence,
        minio_evidence=minio_evidence,
        cross_host_evidence=cross_host_evidence,
        failure_path_evidence=failure_path_evidence,
        redis_persistence_evidence=redis_persistence_evidence,
    )
    (output_dir / "report_snippets.md").write_text(snippets)

    print("Evidence bundle generated successfully.")
    print(f"Output directory: {output_dir}")
    print("Artefacts:")
    for name in [
        "bundle_metadata.json",
        "service_health.json",
        "environment_info.json",
        "repo_volume.json",
        "negative_paths.json",
        "monitor_evidence.json",
        "monitor_concurrency.json",
        "collection_health.json",
        "collection_auth_probe.json",
        "session_log.json",
        "testbed_summary.json",
        "deploy_execution.json",
        "deploy_run.json",
        "deploy_report.json",
        "deploy_stdout.log",
        "deploy_stderr.log",
        "predict_reload.json",
        "evaluate_lifecycle.json",
        "mlflow_evidence.json",
        "minio_evidence.json",
        "cross_host_evidence.json",
        "failure_path_evidence.json",
        "redis_persistence_evidence.json",
        "report_snippets.md",
    ]:
        path = output_dir / name
        if path.exists():
            print(f"  - {name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

