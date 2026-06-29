"""Background monitor loop and REST endpoints for DAM prediction."""
import logging
import threading
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..base_model.registry import ModelRegistry
from .serialization import np_encoder

logger = logging.getLogger(__name__)

_monitor_lock = threading.Lock()
_monitor_state: Dict[str, Any] = {
    "running": False,
    "thread": None,
    "stop_event": None,
    "latest_alarms": None,
    "last_alarm_at": None,
    "error": None,
}


class MonitorStartRequest(BaseModel):
    """Request to start the continuous monitoring loop (DAM prediction on live API data)."""
    experiment_name: str = Field(..., min_length=1)
    model_type: str = Field(..., description="e.g. dam")
    model_params: dict = Field(default_factory=dict)
    registered_model_name: Optional[str] = None
    model_version: Optional[str] = None
    interval_seconds: int = Field(default=60, ge=5, le=3600)


def _monitor_loop(
    experiment_name: str,
    model_type: str,
    model_params: dict,
    registered_model_name: str,
    model_version: str,
    interval_seconds: int,
    registry: ModelRegistry,
    stop_event: threading.Event,
) -> None:
    try:
        model_class = registry.get(model_type)
        if model_class is None:
            with _monitor_lock:
                _monitor_state["error"] = f"Invalid model type: {model_type}"
            return
        params = dict(model_params)
        if "registered_model_name" not in params:
            params["registered_model_name"] = registered_model_name or model_type
        model = model_class(experiment_name=experiment_name, **params)
        prediction_data = {"use_api": True}
        while not stop_event.is_set():
            try:
                alarms = model.get_prediction(
                    registered_model_name=registered_model_name or model_type,
                    model_version=model_version or "latest",
                    data=prediction_data,
                )
                with _monitor_lock:
                    _monitor_state["latest_alarms"] = np_encoder(alarms) if alarms is not None else None
                    _monitor_state["last_alarm_at"] = time.time()
                    _monitor_state["error"] = None
            except Exception as e:
                logger.exception("Monitor loop predict failed: %s", e)
                with _monitor_lock:
                    _monitor_state["error"] = str(e)
            stop_event.wait(interval_seconds)
    finally:
        with _monitor_lock:
            _monitor_state["running"] = False
            _monitor_state["thread"] = None
            _monitor_state["stop_event"] = None


router = APIRouter(prefix="/monitor", tags=["Monitoring"])


@router.post(
    "/start",
    summary="Start monitoring loop",
    description="Start a background loop that periodically runs DAM prediction on live API data.",
    responses={200: {"description": "Monitor started"}, 409: {"description": "Monitor already running"}},
)
async def monitor_start(body: MonitorStartRequest, request: Request):
    registry = request.app.state.model_registry
    with _monitor_lock:
        if _monitor_state["running"]:
            raise HTTPException(status_code=409, detail="Monitor already running")
        stop_event = threading.Event()
        thread = threading.Thread(
            target=_monitor_loop,
            kwargs={
                "experiment_name": body.experiment_name,
                "model_type": body.model_type,
                "model_params": body.model_params or {},
                "registered_model_name": body.registered_model_name or body.model_type,
                "model_version": body.model_version or "latest",
                "interval_seconds": body.interval_seconds,
                "registry": registry,
                "stop_event": stop_event,
            },
            daemon=True,
        )
        _monitor_state["running"] = True
        _monitor_state["thread"] = thread
        _monitor_state["stop_event"] = stop_event
        _monitor_state["latest_alarms"] = None
        _monitor_state["last_alarm_at"] = None
        _monitor_state["error"] = None
    thread.start()
    return {"status": "started", "interval_seconds": body.interval_seconds}


@router.get(
    "/status",
    summary="Monitor status",
    description="Return whether the monitor loop is running and last alarm time.",
)
async def monitor_status():
    with _monitor_lock:
        out = {
            "running": _monitor_state["running"],
            "last_alarm_at": _monitor_state.get("last_alarm_at"),
        }
        if _monitor_state.get("error"):
            out["error"] = _monitor_state["error"]
    return out


@router.post(
    "/stop",
    summary="Stop monitoring loop",
    description="Stop the background monitoring loop.",
)
async def monitor_stop():
    with _monitor_lock:
        ev = _monitor_state.get("stop_event")
        thread = _monitor_state.get("thread")
    if ev is not None:
        ev.set()
    if thread is not None and thread.is_alive():
        thread.join(timeout=15)
    with _monitor_lock:
        _monitor_state["running"] = False
        _monitor_state["thread"] = None
        _monitor_state["stop_event"] = None
    return {"status": "stopped"}


@router.get(
    "/alarms",
    summary="Latest alarms",
    description="Return the latest alarm result from the monitoring loop.",
)
async def monitor_alarms():
    with _monitor_lock:
        return {"alarms": _monitor_state.get("latest_alarms"), "last_alarm_at": _monitor_state.get("last_alarm_at")}
