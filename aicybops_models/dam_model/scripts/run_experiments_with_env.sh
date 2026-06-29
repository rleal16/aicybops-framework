#!/bin/bash
# Set MLflow/S3 env vars and run run_experiments.py (so artifact logging works).
# Usage: ./scripts/run_experiments_with_env.sh [run_experiments.py args...]
# Example: ./scripts/run_experiments_with_env.sh --configs configs/dam_config.json --seeds 0 1

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAM_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$DAM_ROOT"

# Load .env if present (same dir as dam_model root)
if [ -f "$DAM_ROOT/.env" ]; then
  set -a
  # shellcheck source=/dev/null
  source "$DAM_ROOT/.env"
  set +a
fi

# Set defaults so artifact logging to MinIO works when running from host
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:5001}"
export MLFLOW_S3_ENDPOINT_URL="${MLFLOW_S3_ENDPOINT_URL:-http://localhost:9000}"
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-minio}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-minio123}"

exec python scripts/run_experiments.py "$@"
