# DAM model scripts

These scripts **test and run the DAM model locally** (direct execution of the model and pipelines). They do **not** call the AICybOps service; they exercise the DAM code in `aicybops_models/dam_model/` on this machine.

**Definition of success / test failure:** Every script must complete without errors. Any non-zero exit code, unhandled exception, or failed assertion is a **test failure**. Any HTTP call (e.g. to MLflow) that returns 4xx or 5xx is also a failure. Scripts must actually work as intended (models used as requested, no error responses).

- **run_experiments.py** – Experiment sweeps (configs × seeds) with MLflow; run from `dam_model` with `--configs` and optional `--quick-test`.
- **evaluate_dam.py** – Train and/or evaluate a DAM model; uses config and data paths from the repo.
- **full_pipeline.py** – Full pipeline: optimization (optional), training, evaluation; requires `--config-path`.
- **test_dam_with_pipeline.py** – Local dev/test: train or evaluate with `--quick-test` for faster runs.
- **test_processing_pipeline.py** – Tests processing pipeline components (config, metrics, logs, labels, etc.).

Run from the `aicybops_models/dam_model` directory (or set up paths accordingly). For service-based usage (train/predict via the API), use the repo root `scripts/` (e.g. `scripts/example_client.py`, `scripts/remote_deployment_example.py`) or the AICybOps client against a running service.

### Required environment (MLflow + MinIO)

Scripts that log models to MLflow (e.g. **run_experiments.py**, **evaluate_dam.py**, **full_pipeline.py**, **test_dam_with_pipeline.py**) need the MLflow tracking server and MinIO artifact store. Set these before running (adjust hosts if not using localhost):

```bash
export MLFLOW_TRACKING_URI="http://localhost:5001"
export MLFLOW_S3_ENDPOINT_URL="http://localhost:9000"
export AWS_ACCESS_KEY_ID="minio"
export AWS_SECRET_ACCESS_KEY="minio123"
```

Example (from `aicybops_models/dam_model`):

```bash
python scripts/run_experiments.py --configs configs/dam_config.json --seeds 0 --quick-test --epochs 1
```
