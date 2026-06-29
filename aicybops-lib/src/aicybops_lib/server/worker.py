import faulthandler
import gc
import logging
import os
import resource
import signal
import sys
import time
import traceback
from typing import Any, Dict, Optional, Tuple

from .app import EvaluateRequest, TrainRequest, _run_evaluate_sync, _run_train_sync
from .job_store_redis import RedisJobStore

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

_active_job: Optional[Tuple[str, str]] = None  # (job_type, job_id)
_active_store: Optional[RedisJobStore] = None


def _enable_native_crash_handler() -> None:
    """Dump a Python traceback on SIGSEGV/SIGABRT/SIGBUS/SIGFPE.

    Without this, native crashes (PyTorch/TF/numpy C++) leave no trace
    because they bypass Python's exception machinery.
    """
    faulthandler.enable(file=sys.stderr, all_threads=True)
    for sig in ("SIGUSR1",):
        try:
            faulthandler.register(getattr(signal, sig), file=sys.stderr, all_threads=True)
        except (AttributeError, ValueError):
            pass


def _rss_mb() -> float:
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux ru_maxrss is in KiB.
        return usage / 1024.0
    except Exception:
        return -1.0


def _log_stage(tag: str, job_type: str, job_id: str) -> None:
    print(
        f"[worker] {tag} job={job_type}/{job_id} rss_mb={_rss_mb():.1f}",
        flush=True,
    )


def _configure_cpu_parallelism() -> None:
    """
    Apply optional PyTorch thread caps from environment.

    This prevents a single worker process from unexpectedly using all host cores.
    """
    if torch is None:
        return

    num_threads = os.environ.get("TORCH_NUM_THREADS")
    if num_threads:
        try:
            torch.set_num_threads(int(num_threads))
        except ValueError:
            logger.warning("Invalid TORCH_NUM_THREADS=%s", num_threads)

    num_interop_threads = os.environ.get("TORCH_NUM_INTEROP_THREADS")
    if num_interop_threads:
        try:
            torch.set_num_interop_threads(int(num_interop_threads))
        except ValueError:
            logger.warning("Invalid TORCH_NUM_INTEROP_THREADS=%s", num_interop_threads)

    logger.info(
        "Torch CPU threads configured: intraop=%s interop=%s",
        torch.get_num_threads(),
        torch.get_num_interop_threads(),
    )


def _mark_active_job_failed(reason: str) -> None:
    if _active_job is None or _active_store is None:
        return
    job_type, job_id = _active_job
    _active_store.update_job(
        job_type,
        job_id,
        {"status": "failed", "error": reason},
    )
    logger.warning("Marked active job %s/%s as failed: %s", job_type, job_id, reason)


def _handle_shutdown_signal(signum: int, frame: Any) -> None:
    sig_name = signal.Signals(signum).name
    print(f"[worker] received {sig_name}; current stack:", flush=True)
    traceback.print_stack(frame, file=sys.stderr)
    sys.stderr.flush()
    _mark_active_job_failed(f"Worker received {sig_name}")
    sys.exit(128 + signum)


def _run_train_job(store: RedisJobStore, job_id: str, request_payload: Dict[str, Any]) -> None:
    global _active_job, _active_store
    _active_job = ("train", job_id)
    _active_store = store
    store.update_job("train", job_id, {"status": "running"})
    _log_stage("start", "train", job_id)
    try:
        request = TrainRequest.model_construct(**request_payload)
        out = _run_train_sync(request)
        updates: Dict[str, Any] = {
            "status": "completed",
            "result": out["result"],
            "model_reference": out.get("model_reference"),
            "data_collection_metrics_count": out.get("data_collection_metrics_count"),
            "data_collection_logs_count": out.get("data_collection_logs_count"),
        }
        store.update_job("train", job_id, updates)
        _log_stage("done", "train", job_id)
    except Exception as exc:
        logger.exception("Train job %s failed: %s", job_id, exc)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        store.update_job("train", job_id, {"status": "failed", "error": str(exc)})
        _log_stage("error", "train", job_id)
    finally:
        _release_worker_memory()
        _active_job = None
        _active_store = None


def _release_worker_memory() -> None:
    """Drop peak memory between jobs (TF/PyTorch caches, collected API data)."""
    gc.collect()
    try:
        import tensorflow as tf

        tf.keras.backend.clear_session()
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _run_eval_job(store: RedisJobStore, job_id: str, request_payload: Dict[str, Any]) -> None:
    global _active_job, _active_store
    _active_job = ("eval", job_id)
    _active_store = store
    store.update_job("eval", job_id, {"status": "running"})
    _log_stage("start", "eval", job_id)
    try:
        request = EvaluateRequest.model_construct(**request_payload)
        out = _run_evaluate_sync(request)
        store.update_job("eval", job_id, {"status": "completed", "result": out["result"]})
        _log_stage("done", "eval", job_id)
    except Exception as exc:
        logger.exception("Eval job %s failed: %s", job_id, exc)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        store.update_job("eval", job_id, {"status": "failed", "error": str(exc)})
        _log_stage("error", "eval", job_id)
    finally:
        _release_worker_memory()
        _active_job = None
        _active_store = None


def run_worker_loop() -> None:
    _enable_native_crash_handler()
    redis_url = os.environ.get("REDIS_URL", "redis://redis:6379/0")
    store = RedisJobStore(redis_url=redis_url)
    store.ping()
    _configure_cpu_parallelism()

    recovered = store.recover_interrupted_jobs()
    if recovered:
        logger.warning("Recovered %d interrupted job(s) left in 'running' state", recovered)
        print(
            f"[worker] recovered {recovered} interrupted job(s) — last run was killed externally (SIGKILL/OOM/host restart). Check `docker inspect` and dmesg.",
            flush=True,
        )

    signal.signal(signal.SIGTERM, _handle_shutdown_signal)
    signal.signal(signal.SIGINT, _handle_shutdown_signal)

    logger.info("Worker connected to Redis at %s", redis_url)
    print(f"[worker] ready; pid={os.getpid()} rss_mb={_rss_mb():.1f}", flush=True)

    while True:
        try:
            job = store.dequeue_job(timeout_seconds=5)
            if job is None:
                continue
            job_type, job_id, request_payload = job
            logger.info("Worker picked job type=%s id=%s", job_type, job_id)
            if job_type == "train":
                _run_train_job(store, job_id, request_payload)
            elif job_type == "eval":
                _run_eval_job(store, job_id, request_payload)
            else:
                logger.error("Unknown job type: %s", job_type)
        except Exception as exc:
            logger.exception("Worker loop error: %s", exc)
            time.sleep(1)
