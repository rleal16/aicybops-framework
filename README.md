# AICybOps Service Deployment

Companion technical report (operational validation referenced in the paper): [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md).

A machine learning service with the DAM (Deep Anomaly Model) for metrics and log anomaly detection.

## Models

### DAM (Deep Anomaly Model)
- **Purpose**: Anomaly detection on metrics and log sequences
- **Location**: `aicybops_models/dam_model/`
- **Class**: `DAMAnomalyDetector`
- **Data**: Config-driven (local files or API via `use_api`)

## Prerequisites

- **Docker** and **Docker Compose**
- **Conda** (for running the example client)
- **8GB+ RAM** recommended

## Quick Start

```bash
# 1. Go to AICybOps and activate the environment
cd AICybOps
conda activate aicybops

# 2. Start all services
docker compose up -d

# 3. Run the example client
python scripts/example_client.py
```

## Detailed Steps

### 1. Start Services

```bash
docker compose up -d
```

This starts:
- **AICybOps Service** (port 8000): Main API
- **MLflow** (port 5001): Experiment tracking
- **PostgreSQL** (port 5432): Database
- **MinIO** (ports 9000-9001): Storage

### 2. Verify Services

```bash
# Check all services are running
docker compose ps

# Test API
curl http://localhost:8000/
```

### 3. Run Example Client

```bash
# From the AICybOps directory
cd AICybOps

# Create conda environment (if it doesn't exist)
conda create -n aicybops python=3.12
conda activate aicybops

# Install required packages
pip install requests

# Run client
python scripts/example_client.py
```

## API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Key Endpoints

**Train DAM Model:**
```bash
POST /train/
{
  "experiment_name": "DAM_Experiment",
  "model_type": "dam",
  "params": {"lr": 0.0001, "batch_size": 32},
  "epochs": 10
}
```

**Predict with DAM:**
```bash
POST /predict/
{
  "experiment_name": "DAM_Experiment",
  "model_type": "dam",
  "registered_model_name": "dam",
  "model_version": "latest"
}
```

## Troubleshooting

### Services Not Starting
```bash
docker compose logs aicybops-service
docker compose down && docker compose up -d
```

### Port Conflicts
```bash
# Check what's using ports
lsof -i :8000
lsof -i :5432
```

### Client Issues
```bash
# From AICybOps directory: activate environment and ensure deps
cd AICybOps
conda activate aicybops
pip install requests
```

### Data Source Issues
```bash
# Verify data source is set to local
docker compose exec aicybops-service env | grep DATA_SOURCE
```

## Environment Variables

The service uses several environment variables for configuration:

### Data Source Configuration
- **`DATA_SOURCE`** / **`MAIN_DATA_SOURCE`**: Data source for DAM
  - `local`: Uses local data files (paths from `DAM_CONFIG_PATH` config)
  - `api`: Fetches data from external API
- **`LOCAL_DATA_SOURCE`**: Path to local data directory (`/app/data/`)
- **`API_URL`**: External API URL when using API data source
- **`DAM_CONFIG_PATH`**: Path to DAM config JSON inside the container

### MLflow Configuration
- **`MLFLOW_TRACKING_URI`**: MLflow server URL (`http://mlflow:5001`)
- **`MLFLOW_S3_ENDPOINT_URL`**: MinIO endpoint for artifacts (`http://minio:9000`)
- **`AWS_ACCESS_KEY_ID`**: MinIO access key (`minio`)
- **`AWS_SECRET_ACCESS_KEY`**: MinIO secret key (`minio123`)

### Service Configuration
- **`DATA_DIR`**: Data directory path (`/app/data`)
- **`PYTHONPATH`**: Python module search path (set in Dockerfile)
- **`PYTHONUNBUFFERED`**: Python output buffering (`1`)

### Dockerfile Configuration
The `PYTHONPATH` is configured in the Dockerfile:
```dockerfile
ENV PYTHONPATH="/app:${PYTHONPATH}"
```
This ensures Python can import modules from the `/app` directory inside the container.

## Code Structure

### Model Files
- **DAM model**: `aicybops_models/dam_model/`
  - Core: `aicybops_models/dam_model/core/dam_anomaly_detector.py`
  - Local scripts (train/evaluate/experiments): `aicybops_models/dam_model/scripts/` — run DAM **locally**; they do not call the AICybOps service.

### Key Components
- **API Server**: `aicybops-lib/src/aicybops_lib/server/app.py`
- **Client Library**: `aicybops-lib/src/aicybops_lib/client/client.py`
- **Example Client**: `scripts/example_client.py`
- **Docker Configuration**: `docker-compose.yml`

## Services

Replace `localhost` with your server's IP address or hostname:

- **AICybOps API**: http://localhost:8000
- **MLflow UI**: http://localhost:5001
- **MinIO Console**: http://localhost:9001
