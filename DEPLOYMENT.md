# AICybOps Remote Deployment Guide

This guide explains how to deploy the AICybOps service on a remote server and configure clients to connect to it.

## **Prototype deployment (two hosts)**

- **Host A (monitoring host):** Runs monitoring_solution_onehost (collect_metrics_api on port 5010) and Testbed. InfluxDB and collectors (e.g. cAdvisor, Telegraf) run here so the API has metrics and logs. Ensure port **5010** is open for inbound from Host B.
- **Host B (AICybOps host):** Runs only the AICybOps stack (PostgreSQL, MLflow, MinIO, aicybops-service). It fetches all data via the RestAPI from Host A.

**Order:** On Host A, start InfluxDB and collectors, then collect_metrics_api, then run Testbed with `--api-url http://localhost:5010` so the session log is stored. On Host B, set `API_URL=http://<Host_A_IP>:5010` (collect_metrics_api URL, not InfluxDB) and start the AICybOps stack. Use `deployment.env.example` as reference; copy to `.env` and set `API_URL` to the monitoring host.

## **Remote Server Deployment**

### **Smoke test (after deploy)**

1. Start the stack (from repo root):  
   `docker compose up -d`  
   To pick up code changes, rebuild the service first:  
   `docker compose build aicybops-service && docker compose up -d`
2. Wait for services to be healthy (e.g. `docker compose ps`; postgres healthy, mlflow and aicybops-service running).
3. Run a client script to verify the service:  
   `python scripts/example_client.py` or `python scripts/remote_deployment_example.py`  
   (Optionally set `AICYBOPS_SERVICE_URL=http://localhost:8000` or your service URL.)  
   These use the client API with wait=True and model_reference for predict.

The service **requires the full stack** (PostgreSQL, MLflow, MinIO). Artifacts and model storage use **MinIO only**; no local or sqlite backend is supported.

### **1. Deploy the Service**

On the remote server, clone the repository and deploy:

```bash
# Clone the repository
git clone <repository-url>
cd AICybOps

# Copy and configure environment variables
cp deployment.env.example .env
# Edit .env with your remote server configuration

# Deploy with Docker Compose
docker compose up -d
```

### **2. Configure Environment Variables**

Create a `.env` file on the remote server:

```bash
# Service Configuration
AICYBOPS_SERVICE_HOST=0.0.0.0  # Listen on all interfaces
AICYBOPS_SERVICE_PORT=8000
AICYBOPS_SERVICE_PROTOCOL=http

# Data Source Configuration
DATA_SOURCE=api  # default: fetch from collect_metrics_api (RestAPI)
API_URL=http://<MONITORING_HOST_IP>:5010  # collect_metrics_api on monitoring host (port 5010)

# Model Configuration
DAM_MODEL_NAME=dam

# DAM model: path to JSON config (required for train/predict when using DAM)
DAM_CONFIG_PATH=/app/configs/dam_config.json

# MLflow Configuration
MLFLOW_TRACKING_URI=http://mlflow:5001
MLFLOW_S3_ENDPOINT_URL=http://minio:9000

# MinIO Configuration
AWS_ACCESS_KEY_ID=minio
AWS_SECRET_ACCESS_KEY=minio123

# DAM model (for train/predict via API and scripts/example_client.py)
DAM_CONFIG_PATH=/app/aicybops_models/dam_model/configs/dam_config.json
```

### **DAM model setup**

For the DAM model to work via the service (e.g. `scripts/example_client.py` or `/train` with `model_type=dam`):

- **Docker Compose**: The stack sets `DAM_CONFIG_PATH` and mounts the DAM config and data directories from the repo into the container. **Run `docker compose` from the repository root** (the directory containing `docker-compose.yml`) so the volume paths `./aicybops_models/...` resolve to your repo. The existing data you generated under `aicybops_models/dam_model/data_generation/` is then mapped into the container. Required layout on the host:
  - `aicybops_models/dam_model/data_generation/data/generated/` must contain `metrics.csv` and (if using labels) `anomaly_labels.csv`
  - `aicybops_models/dam_model/data_generation/logs/` must contain `logs.txt`
  That layout matches the output of the data_generation scripts (e.g. `config_main_test.json` → `data/generated/metrics.csv`, `logs/logs.txt`). If those files are missing, run the DAM data generation once so they exist, then start the stack.
- **Local run** (e.g. uvicorn on the host): Set `DAM_CONFIG_PATH` to the **absolute** path to the DAM config file, e.g.  
  `export DAM_CONFIG_PATH=/path/to/repo/aicybops_models/dam_model/configs/dam_config.json`  
  The config’s paths (metrics, logs, labels) are resolved relative to the config file; the same repo layout used by the DAM scripts must be available to the process.

## **Client Configuration**

### **1. Local Client Configuration**

To connect from a local machine to the remote service:

```bash
# Set environment variables
export AICYBOPS_SERVICE_HOST=remote-server.com
export AICYBOPS_SERVICE_PORT=8000
export AICYBOPS_SERVICE_PROTOCOL=http
export DATA_SOURCE=api
export API_URL=http://<MONITORING_HOST_IP>:5010

# Run the client
python scripts/example_client.py
```

### **2. Programmatic Configuration**

```python
import os
from aicybops_lib.client import AICybOpsClient

# Configure for remote server
os.environ['AICYBOPS_SERVICE_HOST'] = 'remote-server.com'
os.environ['AICYBOPS_SERVICE_PORT'] = '8000'
os.environ['AICYBOPS_SERVICE_PROTOCOL'] = 'http'
os.environ['DATA_SOURCE'] = 'api'
os.environ['API_URL'] = 'http://<MONITORING_HOST_IP>:5010'

# Create client
client = AICybOpsClient(base_url="http://remote-server.com:8000")

# Use the client
result = client.train_model(
    experiment_name="RemoteTest",
    model_type="pytorch",
    params={"lr": 0.01},
    epochs=1
)
```

## **Network Configuration**

### **1. Firewall Settings**

Ensure the following ports are open on your remote server:

```bash
# AICybOps Service
8000/tcp

# MLflow (if accessing directly)
5001/tcp

# MinIO (if accessing directly)
9000/tcp
9001/tcp

# PostgreSQL (if accessing directly)
5432/tcp
```

### **2. Docker Compose Network**

The service is configured to listen on `0.0.0.0:8000`, making it accessible from external clients.

## **Security Considerations**

### **1. HTTPS Configuration**

For production deployments, consider:

```bash
# Use HTTPS
export AICYBOPS_SERVICE_PROTOCOL=https

# Configure SSL certificates
# Add reverse proxy (nginx/traefik) for SSL termination
```

### **2. Authentication**

The current implementation doesn't include authentication. For production:

- Add API key authentication
- Implement OAuth2/JWT tokens
- Use reverse proxy with authentication

## **Monitoring and Logging**

### **1. Service Health**

Check service health:

```bash
curl http://remote-server.com:8000/
```

### **2. Logs**

View service logs:

```bash
docker compose logs aicybops-service
```

## **Testing Remote Deployment**

Use the provided test script:

```bash
# Configure for your remote server
export AICYBOPS_SERVICE_HOST=remote-server.com
export AICYBOPS_SERVICE_PORT=8000

# Run the test
python scripts/remote_deployment_example.py
```

## **Troubleshooting**

### **Debug Commands**

```bash
# Check service status
docker compose ps

# View logs
docker compose logs aicybops-service

# Test connectivity
curl -v http://remote-server.com:8000/

# Check network
telnet remote-server.com 8000
```

## **Environment Variables Reference**

- `AICYBOPS_SERVICE_HOST`: Service hostname/IP (default: `localhost`, required: No)
- `AICYBOPS_SERVICE_PORT`: Service port (default: `8000`, required: No)
- `AICYBOPS_SERVICE_PROTOCOL`: Protocol (http/https) (default: `http`, required: No)
- `DATA_SOURCE`: Data source type (default: `local`, required: No)
- `API_URL`: collect_metrics_api URL on the monitoring host (e.g. `http://<MONITORING_HOST_IP>:5010`, port 5010; required when DATA_SOURCE=api)
- `DAM_MODEL_NAME`: DAM model type name (default: `dam`, required: No)
- `DAM_CONFIG_PATH`: Path to DAM JSON config file; must exist in container for DAM train/predict (e.g. `/app/configs/dam_config.json` or mount). Optional if using default.
- `MLFLOW_REGISTERED_MODEL_NAME`: Default Model Registry name when not provided per request (default: `aicybops_model`, required: No)
