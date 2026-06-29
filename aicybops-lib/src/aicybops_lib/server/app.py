from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Union, Optional, List, Dict, Any
from ..base_model.registry import ModelRegistry
from .serialization import np_encoder
from .job_store_redis import RedisJobStore
from .monitoring import router as monitor_router
import mlflow
import os
import uuid
import asyncio
import logging
import time

logger = logging.getLogger(__name__)

_job_store: Dict[str, Dict[str, Any]] = {}
_eval_job_store: Dict[str, Dict[str, Any]] = {}
_job_backend: str = "memory"

app = FastAPI(
    title="AICybOps API",
    description="REST API for training and deploying ML models for cybersecurity operations.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

def init_app(model_registry: ModelRegistry, lifespan=None):
    setup_mlflow()
    app.state.model_registry = model_registry
    _init_job_backend()
    
    if lifespan:
        app.lifespan = lifespan

    app.post("/train/")(train_model)
    app.get("/jobs/{job_id}")(get_job_status)
    app.post("/predict/")(predict)
    app.post("/evaluate/")(evaluate_model)
    app.get("/eval-jobs/{job_id}")(get_eval_job_status)
    app.include_router(monitor_router)

    return app


def _init_job_backend() -> None:
    global _job_backend
    _job_backend = os.environ.get("JOB_BACKEND", "redis").strip().lower()
    if _job_backend not in {"redis", "memory"}:
        raise RuntimeError(f"Invalid JOB_BACKEND={_job_backend}. Use 'redis' or 'memory'.")
    if _job_backend == "redis":
        redis_url = os.environ.get("REDIS_URL", "redis://redis:6379/0")
        store = RedisJobStore(redis_url=redis_url)
        store.ping()
        app.state.redis_job_store = store
        logger.info("Job backend: redis (%s)", redis_url)
    else:
        logger.warning("Job backend: memory (single-process, non-durable)")

def setup_mlflow():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError(
            "MLFLOW_TRACKING_URI must be set. Use the full stack with MinIO: "
            "MLflow server (e.g. http://mlflow:5001) with MLFLOW_S3_ENDPOINT_URL, "
            "AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY for MinIO. No local/sqlite backend."
        )
    logger.info("MLflow tracking URI: %s", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)


class TrainRequest(BaseModel):
    """Request model for training machine learning models."""
    params: dict = Field(..., description="Training parameters (learning rate, batch size, etc.) - example: {'lr': 0.001, 'batch_size': 32}")
    epochs: int = Field(..., gt=0, description="Number of training epochs")
    experiment_name: str = Field(..., min_length=1, description="MLflow experiment name")
    model_type: str = Field(..., description="Model type (e.g. dam or other registered models)")
    model_params: dict = Field(default_factory=dict, description="Model-specific parameters")
    run_optimization: bool = Field(default=False, description="If True call model.optimize(); if False call model.train_test_validate()")
    param_space: Optional[List[dict]] = Field(default=None, description="List of param dicts for optimize() (used when run_optimization=True)")
    max_evals: Optional[int] = Field(default=None, description="Max evaluations for optimize() (used when run_optimization=True)")
    objective: str = Field(default="val_loss", description="Optimization objective: 'val_loss', 'f1_score', or 'test_anomaly_scores_mean'")

    @validator('model_type')
    def validate_model_type(cls, v, values, **kwargs):
        if v not in app.state.model_registry.list():
            raise ValueError(f"Invalid model type: {v}")
        return v
    
    class Config:
        protected_namespaces = ()
        json_schema_extra = {
            "example": {
                "params": {"lr": 0.001, "batch_size": 32},
                "epochs": 10,
                "experiment_name": "MyExperiment",
                "model_type": "dam",
                "model_params": {},
                "run_optimization": False
            }
        }

class PredictionRequest(BaseModel):
    """Request model for making predictions with trained models. Use Model Registry (registered_model_name, model_version) by default."""
    experiment_name: str = Field(..., min_length=1, description="MLflow experiment name")
    model_type: str = Field(default="dam", description="Model type")
    model_params: dict = Field(default_factory=dict, description="Model-specific parameters (include registered_model_name to set default for load)")
    registered_model_name: Optional[str] = Field(default=None, description="Model Registry name; when omitted uses model_type or default")
    model_version: Optional[str] = Field(default=None, description="Model Registry version; when omitted uses 'latest'")
    run_name: Optional[str] = Field(default=None, description="(Deprecated) Specific MLflow run name; prefer registered_model_name")
    nested_run_name: Optional[str] = Field(default=None, description="(Deprecated) Nested run name")
    metric_name: Optional[str] = Field(default=None, description="(Deprecated) Metric for run-based selection")
    mode: Optional[str] = Field(default=None, pattern="^(max|min)$", description="(Deprecated) Optimization mode")

    class Config:
        protected_namespaces = ()
        json_schema_extra = {
            "example": {
                "experiment_name": "MyExperiment",
                "model_type": "dam",
                "registered_model_name": "dam",
                "model_version": "latest"
            }
        }

class TrainingResponse(BaseModel):
    """Response model for training endpoint (sync wait=true)."""
    model_config = {"protected_namespaces": ()}

    result: dict = Field(..., description="Training results and metrics")
    model_reference: Optional[dict] = Field(default=None, description="Registered model name and version for predict")
    data_collection_metrics_count: Optional[str] = Field(default=None, description="When data was collected from API: number of metrics rows")
    data_collection_logs_count: Optional[str] = Field(default=None, description="When data was collected from API: number of log entries")

class JobAcceptedResponse(BaseModel):
    """Response when train is accepted (202)."""
    job_id: str = Field(..., description="Job id for polling")
    status_url: str = Field(..., description="URL for GET job status")


class EvaluateRequest(BaseModel):
    """Request for model evaluation (generic; evaluation_config shape depends on model)."""
    experiment_name: str = Field(..., min_length=1, description="MLflow experiment name")
    model_type: str = Field(..., description="Registered model class key (e.g. dam)")
    model_params: dict = Field(default_factory=dict, description="Model constructor kwargs; merged into evaluation data loading")
    registered_model_name: Optional[str] = Field(default=None, description="Model Registry name; defaults to model_type")
    model_version: Optional[str] = Field(default=None, description="Model Registry version; defaults to latest")
    evaluation_config: dict = Field(..., description="Model-specific evaluation config (e.g. DAM: evt_parameters, max_memory_gb)")
    dataset_type: str = Field(default="test", description="Dataset split passed to evaluate() (e.g. test)")
    output_dir: Optional[str] = Field(default=None, description="Optional directory for evaluation artifacts")

    @validator("model_type")
    def validate_model_type_eval(cls, v):
        try:
            if app.state.model_registry and v not in app.state.model_registry.list():
                raise ValueError(f"Invalid model type: {v}")
        except (AttributeError, RuntimeError):
            pass
        return v

    class Config:
        protected_namespaces = ()


class EvaluationResponse(BaseModel):
    """Response: result dict from model.evaluate() (metrics, thresholds, etc.)."""
    result: dict = Field(..., description="Evaluation result dict")

class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    predictions: Union[List[float], List[int], bool] = Field(..., description="Model predictions")
    prediction_diagnostics: Optional[dict] = Field(
        default=None,
        description="Optional diagnostics from the model (e.g. DAM); omitted when not returned",
    )

class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(..., description="Service status")

class ErrorResponse(BaseModel):
    """Standard error response model."""
    detail: str = Field(..., description="Error message")


def _run_train_sync(request: TrainRequest) -> dict:
    """Run training synchronously; returns dict with 'result' and optionally 'model_reference'."""
    model_class = app.state.model_registry.get(request.model_type)
    if model_class is None:
        raise ValueError(f"Invalid model type: {request.model_type}")
    model_params = dict(request.model_params)
    if "registered_model_name" not in model_params:
        model_params["registered_model_name"] = request.model_type
    model = model_class(experiment_name=request.experiment_name, **model_params)
    train_params = dict(request.params)
    if os.environ.get("DATA_SOURCE") == "api" and "use_api" not in train_params:
        train_params["use_api"] = True
    dam_model_name = os.environ.get("DAM_MODEL_NAME", "dam")
    train_kwargs = {"params": train_params, "epochs": request.epochs}
    if request.model_type == dam_model_name:
        train_kwargs["data"] = {
            "config_path": os.environ.get("DAM_CONFIG_PATH"),
            "use_api": train_params.get("use_api", True),
            "use_session_time_range": train_params.get("use_session_time_range", True),
            "batch_size": train_params.get("batch_size", 32),
            "train_ratio": 0.6,
            "val_ratio": 0.2,
            "random_state": 42,
        }
        if train_params.get("start") is not None:
            train_kwargs["data"]["start"] = train_params["start"]
        if train_params.get("training_window_minutes") is not None:
            train_kwargs["data"]["training_window_minutes"] = train_params["training_window_minutes"]
    if request.run_optimization:
        param_space = request.param_space if request.param_space is not None else [train_params]
        max_evals = request.max_evals if request.max_evals is not None else len(param_space)
        result = model.optimize(param_space=param_space, max_evals=max_evals, epochs=request.epochs, objective=request.objective, **{k: v for k, v in train_kwargs.items() if k != "epochs" and k != "params"})
        if isinstance(result, tuple):
            best_params, best_loss = result
            result = {"best_params": best_params, "best_loss": best_loss}
    else:
        result = model.train_test_validate(**train_kwargs)
    out = {"result": result}
    if isinstance(result, dict) and "registered_model_name" in result and "model_version" in result:
        out["model_reference"] = {"registered_model_name": result["registered_model_name"], "model_version": result["model_version"]}
    if isinstance(result, dict):
        if result.get("data_collection_metrics_count") is not None:
            out["data_collection_metrics_count"] = result["data_collection_metrics_count"]
        if result.get("data_collection_logs_count") is not None:
            out["data_collection_logs_count"] = result["data_collection_logs_count"]
    return out


@app.post(
    "/train/",
    summary="Train a Machine Learning Model",
    description="Train a machine learning model. By default returns 202 Accepted with job_id; poll GET /jobs/{job_id}. Use ?wait=true to block and get 200 with result.",
    responses={
        200: {"description": "Training completed (when wait=true)", "model": TrainingResponse},
        202: {"description": "Training accepted", "model": JobAcceptedResponse},
        400: {"description": "Invalid request parameters", "model": ErrorResponse},
        422: {"description": "Validation error", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse}
    },
    tags=["Training"]
)
async def train_model(request: TrainRequest, wait: bool = Query(False, alias="wait", description="If true, block until job completes and return 200 with result")):
    """Train a machine learning model with the specified parameters."""
    logger.info("Received training request: %s", request)
    try:
        if wait:
            job_id = _submit_train_job(request)
            job = await _wait_for_job("train", job_id)
            if job.get("status") == "failed":
                raise HTTPException(status_code=500, detail=job.get("error", "training failed"))
            response = {
                "result": np_encoder(job.get("result")),
                "model_reference": job.get("model_reference"),
            }
            if job.get("data_collection_metrics_count") is not None:
                response["data_collection_metrics_count"] = job["data_collection_metrics_count"]
            if job.get("data_collection_logs_count") is not None:
                response["data_collection_logs_count"] = job["data_collection_logs_count"]
            return response
        job_id = _submit_train_job(request)
        return JSONResponse(
            status_code=202,
            content={"job_id": job_id, "status_url": f"/jobs/{job_id}"},
            headers={"Location": f"/jobs/{job_id}", "Retry-After": "10"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Server error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

def _run_evaluate_sync(request: EvaluateRequest) -> dict:
    """Run evaluation synchronously; returns dict with 'result'."""
    model_class = app.state.model_registry.get(request.model_type)
    if model_class is None:
        raise ValueError(f"Invalid model type: {request.model_type}")
    model_params = dict(request.model_params)
    if "registered_model_name" not in model_params:
        model_params["registered_model_name"] = request.registered_model_name or request.model_type
    model = model_class(experiment_name=request.experiment_name, **model_params)
    name = request.registered_model_name or request.model_type
    ver = request.model_version or "latest"
    if hasattr(model, "_get_model"):
        loaded, _ = model._get_model(name, ver)
        if loaded is not None:
            model.model = loaded
    evaluation_data = dict(request.model_params) if request.model_params else {}
    if not evaluation_data.get("use_api"):
        evaluation_data.setdefault("use_api", True)
        # Keep fallback bounded when callers do not provide explicit evaluation model_params.
        evaluation_data.setdefault("use_session_time_range", False)
        evaluation_data.setdefault("start", "-600s")
    result = model.evaluate(
        data=evaluation_data,
        config=request.evaluation_config,
        output_dir=request.output_dir,
        dataset_type=request.dataset_type,
    )
    return {"result": result}


def _create_train_job_record() -> Dict[str, Any]:
    now = time.time()
    return {
        "status": "pending",
        "created_at": now,
        "updated_at": now,
        "result": None,
        "model_reference": None,
        "data_collection_metrics_count": None,
        "data_collection_logs_count": None,
        "error": None,
    }


def _create_eval_job_record() -> Dict[str, Any]:
    now = time.time()
    return {
        "status": "pending",
        "created_at": now,
        "updated_at": now,
        "result": None,
        "error": None,
    }


def _submit_train_job(request: TrainRequest) -> str:
    job_id = str(uuid.uuid4())
    if _job_backend == "redis":
        store: RedisJobStore = app.state.redis_job_store
        store.create_job("train", job_id, _create_train_job_record())
        store.enqueue_job("train", job_id, request.dict())
        return job_id

    _job_store[job_id] = _create_train_job_record()

    def run_job():
        _job_store[job_id]["status"] = "running"
        _job_store[job_id]["updated_at"] = time.time()
        try:
            out = _run_train_sync(request)
            _job_store[job_id]["result"] = out["result"]
            _job_store[job_id]["model_reference"] = out.get("model_reference")
            _job_store[job_id]["data_collection_metrics_count"] = out.get("data_collection_metrics_count")
            _job_store[job_id]["data_collection_logs_count"] = out.get("data_collection_logs_count")
            _job_store[job_id]["status"] = "completed"
        except Exception as e:
            _job_store[job_id]["status"] = "failed"
            _job_store[job_id]["error"] = str(e)
        _job_store[job_id]["updated_at"] = time.time()

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, run_job)
    return job_id


def _submit_eval_job(request: EvaluateRequest) -> str:
    job_id = str(uuid.uuid4())
    if _job_backend == "redis":
        store: RedisJobStore = app.state.redis_job_store
        store.create_job("eval", job_id, _create_eval_job_record())
        store.enqueue_job("eval", job_id, request.dict())
        return job_id

    _eval_job_store[job_id] = _create_eval_job_record()

    def run_eval_job():
        _eval_job_store[job_id]["status"] = "running"
        _eval_job_store[job_id]["updated_at"] = time.time()
        try:
            out = _run_evaluate_sync(request)
            _eval_job_store[job_id]["result"] = out["result"]
            _eval_job_store[job_id]["status"] = "completed"
        except Exception as e:
            _eval_job_store[job_id]["status"] = "failed"
            _eval_job_store[job_id]["error"] = str(e)
        _eval_job_store[job_id]["updated_at"] = time.time()

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, run_eval_job)
    return job_id


def _read_job(job_type: str, job_id: str) -> Optional[Dict[str, Any]]:
    if _job_backend == "redis":
        store: RedisJobStore = app.state.redis_job_store
        return store.get_job(job_type, job_id)
    return _job_store.get(job_id) if job_type == "train" else _eval_job_store.get(job_id)


async def _wait_for_job(job_type: str, job_id: str, timeout_seconds: int = 3600, poll_interval: float = 1.0) -> Dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        job = _read_job(job_type, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        status = job.get("status")
        if status in {"completed", "failed"}:
            return job
        await asyncio.sleep(poll_interval)
    raise HTTPException(status_code=504, detail=f"{job_type} job timed out after {timeout_seconds}s")


@app.post(
    "/evaluate/",
    summary="Evaluate a trained model",
    description="Run model evaluation. By default returns 202 Accepted with job_id; poll GET /eval-jobs/{job_id}. Use ?wait=true to block and get 200 with result. Returns 501 if not supported.",
    responses={
        200: {"description": "Evaluation result (when wait=true)", "model": EvaluationResponse},
        202: {"description": "Evaluation accepted", "model": JobAcceptedResponse},
        501: {"description": "Evaluation not supported for this model"},
    },
    tags=["Evaluation"]
)
async def evaluate_model(request: EvaluateRequest, wait: bool = Query(False, alias="wait", description="If true, block until evaluation completes and return 200 with result")):
    try:
        if wait:
            try:
                job_id = _submit_eval_job(request)
                job = await _wait_for_job("eval", job_id)
            except NotImplementedError as e:
                raise HTTPException(status_code=501, detail=f"Evaluation not supported for this model: {e}")
            if job.get("status") == "failed":
                detail = job.get("error", "evaluation failed")
                if "not supported" in detail.lower():
                    raise HTTPException(status_code=501, detail=detail)
                raise HTTPException(status_code=500, detail=detail)
            return {"result": np_encoder(job.get("result")) if job.get("result") is not None else {}}

        job_id = _submit_eval_job(request)
        return JSONResponse(
            status_code=202,
            content={"job_id": job_id, "status_url": f"/eval-jobs/{job_id}"},
            headers={"Location": f"/eval-jobs/{job_id}", "Retry-After": "10"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Evaluation failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/jobs/{job_id}",
    summary="Get job status",
    description="Poll training job status. Returns status (pending|running|completed|failed). When completed, includes result, model_reference, and optionally data_collection_metrics_count / data_collection_logs_count when training used API-sourced data. When failed, includes error.",
    responses={200: {"description": "Job status"}, 404: {"description": "Job not found"}},
    tags=["Training"]
)
async def get_job_status(job_id: str):
    job = _read_job("train", job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    out = {"status": job["status"], "job_id": job_id}
    if job["status"] == "completed":
        out["result"] = np_encoder(job["result"]) if job.get("result") is not None else None
        out["model_reference"] = job.get("model_reference")
        if job.get("data_collection_metrics_count") is not None:
            out["data_collection_metrics_count"] = job["data_collection_metrics_count"]
        if job.get("data_collection_logs_count") is not None:
            out["data_collection_logs_count"] = job["data_collection_logs_count"]
    if job["status"] == "failed":
        out["error"] = job.get("error")
    return out


@app.get(
    "/eval-jobs/{job_id}",
    summary="Get evaluation job status",
    description="Poll evaluation job status. Returns status (pending|running|completed|failed); when completed includes result; when failed includes error.",
    responses={200: {"description": "Job status"}, 404: {"description": "Job not found"}},
    tags=["Evaluation"]
)
async def get_eval_job_status(job_id: str):
    job = _read_job("eval", job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Evaluation job not found")
    out = {"status": job["status"], "job_id": job_id}
    if job["status"] == "completed":
        out["result"] = np_encoder(job["result"]) if job.get("result") is not None else None
    if job["status"] == "failed":
        out["error"] = job.get("error")
    return out


@app.post(
    "/predict/",
    response_model=PredictionResponse,
    summary="Make Predictions with Trained Model",
    description=(
        "Load a model from the MLflow Model Registry (registered_model_name / model_version; defaults from model_type). "
        "Run-based selection via run_name / nested_run_name / metric_name is deprecated. "
        "Pass model_params as prediction data (e.g. use_api); when DATA_SOURCE=api, use_api and use_session_time_range may be set automatically."
    ),
    responses={
        200: {"description": "Predictions generated successfully"},
        400: {"description": "Invalid request parameters", "model": ErrorResponse},
        404: {"description": "Experiment or model not found", "model": ErrorResponse},
        422: {"description": "Validation error", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse}
    },
    tags=["Prediction"]
)
async def predict(request: PredictionRequest):
    """Make predictions using a trained model from MLflow Model Registry."""
    logger.info("Predicting with request: %s", request)
    try:
        model_class = app.state.model_registry.get(request.model_type)
        if model_class is None:
            raise HTTPException(status_code=400, detail=f"Invalid model type: {request.model_type}")
        model_params = dict(request.model_params)
        if "registered_model_name" not in model_params:
            model_params["registered_model_name"] = request.registered_model_name or request.model_type
        model = model_class(experiment_name=request.experiment_name, **model_params)
        prediction_data = dict(request.model_params) if request.model_params else {}
        if os.environ.get("DATA_SOURCE") == "api" and "use_api" not in prediction_data:
            prediction_data["use_api"] = True
        if os.environ.get("DATA_SOURCE") == "api" and "use_session_time_range" not in prediction_data:
            prediction_data["use_session_time_range"] = False
        out = model.get_prediction(
            run_name=request.run_name,
            nested_run_name=request.nested_run_name,
            metric_name=request.metric_name,
            mode=request.mode,
            registered_model_name=request.registered_model_name or request.model_type,
            model_version=request.model_version or "latest",
            data=prediction_data,
        )
        if isinstance(out, dict) and "predictions" in out:
            response = {"predictions": np_encoder(out["predictions"])}
            if out.get("prediction_diagnostics"):
                response["prediction_diagnostics"] = np_encoder(out["prediction_diagnostics"])
            return response
        return {"predictions": np_encoder(out)}
    except Exception as e:
        logger.error("Prediction failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check the health status of the AICybOps API service.",
    responses={200: {"description": "Service is healthy and running"}},
    tags=["Health"]
)
async def root():
    return {"status": "ok"}
