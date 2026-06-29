#!/usr/bin/env bash
# Deploy script for AICybOps service (Host B).
#
# Usage: ./deploy.sh start [<HOST_A_IP>] [--build]
#                    stop
#                    clean
#                    status
#
#   start                  - Deploy AICybOps; default Host A is kube-worker1.lis.ipn.pt:5010.
#   start <HOST_A_IP>      - Override collect_metrics_api host (port 5010 fixed in checks).
#   start --build          - Build aicybops-service image before starting.
#   start --no-cache       - Build aicybops-service image without Docker cache before starting.
#   stop                   - Stop all AICybOps services.
#   clean                  - Remove containers, volumes, and orphans (keeps images).
#   clean-all              - Full reset: stop worker, remove containers/volumes/images/networks, prune build cache.
#   status                 - Show running containers and health checks.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONNECT_TIMEOUT=60
SERVICE_HEALTH_TIMEOUT=120
DEFAULT_HOST_A_MONITORING="kube-worker1.lis.ipn.pt"

check_prerequisites() {
  for cmd in docker curl; do
    command -v "$cmd" >/dev/null || { echo "ERROR: $cmd not found"; exit 1; }
  done
  docker compose version >/dev/null 2>&1 || { echo "ERROR: docker compose v2 required"; exit 1; }
}

check_monitoring_connection() {
  local host_a_ip="$1"
  echo ""
  echo "Checking connectivity to Host A monitoring API (http://${host_a_ip}:5010)..."
  echo "  Retrying for up to ${CONNECT_TIMEOUT}s..."

  local elapsed=0
  while [ $elapsed -lt $CONNECT_TIMEOUT ]; do
    if curl -sf --max-time 5 "http://${host_a_ip}:5010/test_connection" >/dev/null 2>&1; then
      echo "  Host A monitoring API is reachable."
      return 0
    fi
    sleep 3
    elapsed=$((elapsed + 3))
  done

  echo ""
  echo "ERROR: Cannot reach Host A monitoring API at http://${host_a_ip}:5010"
  echo ""
  echo "  Possible causes:"
  echo "    - Host A monitoring stack is not deployed yet"
  echo "    - Port 5010 is not open / firewall is blocking"
  echo "    - Wrong IP address for Host A"
  echo ""
  echo "  Deploy Host A first:  cd monitoring_solution_onehost && ./deploy.sh start"
  exit 1
}

configure_env() {
  local host_a_ip="$1"

  if [ ! -f .env ]; then
    if [ -f deployment.env.example ]; then
      cp deployment.env.example .env
      echo "  Created .env from deployment.env.example"
    fi
  fi

  local docker_api_url
  if [[ "$host_a_ip" == "localhost" || "$host_a_ip" == "127.0.0.1" ]]; then
    docker_api_url="http://host.docker.internal:5010"
  else
    docker_api_url="http://${host_a_ip}:5010"
  fi

  if [ -f .env ]; then
    if grep -q "^API_URL=" .env; then
      sed -i.bak "s|^API_URL=.*|API_URL=${docker_api_url}|" .env && rm -f .env.bak
    else
      echo "API_URL=${docker_api_url}" >> .env
    fi
    echo "  Configured API_URL=${docker_api_url}"
  fi
}

wait_for_service() {
  echo ""
  echo "Waiting for AICybOps service to become ready..."
  local elapsed=0
  while [ $elapsed -lt $SERVICE_HEALTH_TIMEOUT ]; do
    if curl -sf http://localhost:8000/ >/dev/null 2>&1; then
      echo "  AICybOps service ready on port 8000"
      return 0
    fi
    sleep 3
    elapsed=$((elapsed + 3))
  done
  echo "  WARNING: AICybOps service not responding on port 8000 — check logs with:"
  echo "    docker compose logs aicybops-service"
}

cmd_start() {
  local host_a_ip=""
  local used_default_host=false
  local do_build=false
  local no_cache=false

  shift
  for arg in "$@"; do
    case "$arg" in
      --build)     do_build=true ;;
      --no-cache)  do_build=true; no_cache=true ;;
      -*) echo "Unknown option: $arg"; exit 1 ;;
      *)  host_a_ip="$arg" ;;
    esac
  done

  if [ -z "$host_a_ip" ]; then
    host_a_ip="$DEFAULT_HOST_A_MONITORING"
    used_default_host=true
  fi

  echo "=== Deploying Host B: AICybOps Service ==="
  echo "  Host A (monitoring API): http://${host_a_ip}:5010"
  if [ "$used_default_host" = true ]; then
    echo "  (default Host A; override with: $0 start <HOST_A_IP>)"
  fi

  check_prerequisites

  check_monitoring_connection "$host_a_ip"

  configure_env "$host_a_ip"

  echo ""
  if [ "$do_build" = true ]; then
    if [ "$no_cache" = true ]; then
      echo "Building AICybOps service (no cache)..."
      docker compose build --no-cache aicybops-service
    else
      echo "Building AICybOps service..."
      docker compose build aicybops-service
    fi
  fi

  echo "Starting AICybOps services..."
  docker compose up -d

  wait_for_service

  echo ""
  echo "=== Host B deployment complete ==="
}

cmd_stop() {
  echo "=== Stopping Host B: AICybOps Service ==="
  docker compose down 2>/dev/null || true
  echo "AICybOps services stopped."
}

cmd_clean() {
  echo "=== Cleaning Host B: AICybOps Service ==="
  docker compose down -v --remove-orphans 2>/dev/null || true
  echo "AICybOps containers, volumes, and orphans removed."
}

cmd_clean_all() {
  echo "=== Full clean: AICybOps (containers, volumes, images, job queue) ==="
  echo "  This removes:"
  echo "    - All compose containers and named volumes (Redis jobs, MLflow DB, MinIO, Drain3)"
  echo "    - Images built for this project (aicybops-service, aicybops-worker, mlflow)"
  echo "    - Legacy project 'aicybops-dev' if present"
  echo "    - Leftover aicybops containers/networks/volumes"
  echo "    - Docker build cache for this host"
  echo ""
  echo "  Stop any client jobs first (evidence collector / deploy scripts) with Ctrl+C."
  echo ""

  # Stop worker first so restart policy does not immediately respawn during teardown.
  docker compose stop aicybops-worker 2>/dev/null || true
  docker compose down -v --remove-orphans --rmi all 2>/dev/null || true

  docker compose -p aicybops-dev stop aicybops-worker 2>/dev/null || true
  docker compose -p aicybops-dev down -v --remove-orphans --rmi all 2>/dev/null || true

  # Force-remove any stopped/exited containers still named aicybops*.
  while IFS= read -r cid; do
    [ -n "$cid" ] || continue
    echo "  Removing container: $cid"
    docker rm -f "$cid" 2>/dev/null || true
  done < <(docker ps -aq --filter "name=aicybops" 2>/dev/null || true)

  # Remove any leftover named volumes (e.g. after project rename).
  while IFS= read -r vol; do
    [ -n "$vol" ] || continue
    echo "  Removing volume: $vol"
    docker volume rm "$vol" 2>/dev/null || true
  done < <(docker volume ls -q 2>/dev/null | grep -E '^aicybops' || true)

  # Remove unused project networks.
  while IFS= read -r net; do
    [ -n "$net" ] || continue
    echo "  Removing network: $net"
    docker network rm "$net" 2>/dev/null || true
  done < <(docker network ls -q --filter "name=aicybops" 2>/dev/null || true)

  docker builder prune -af 2>/dev/null || true

  echo ""
  echo "=== Full clean complete ==="
  echo "  Redeploy with: $0 start --build"
  echo ""
  echo "  Verify nothing left:"
  echo "    docker ps -a | grep aicybops || echo 'no aicybops containers'"
  echo "    docker volume ls | grep aicybops || echo 'no aicybops volumes'"
  echo "    docker images | grep aicybops || echo 'no aicybops images'"
}

cmd_status() {
  echo "Expected containers: postgres, minio, mlflow, redis, aicybops-service, aicybops-worker"
  docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null \
    | grep -E "postgres|minio|mlflow|aicybops" || true
  echo ""
  if curl -sf http://localhost:8000/ >/dev/null 2>&1; then
    echo "AICybOps API (8000): OK"
  else
    echo "AICybOps API (8000): not reachable"
  fi
  if curl -sf http://localhost:5001/ >/dev/null 2>&1; then
    echo "MLflow UI (5001): OK"
  else
    echo "MLflow UI (5001): not reachable"
  fi
}

case "${1:-}" in
  start)     cmd_start "$@" ;;
  stop)      cmd_stop ;;
  clean)     cmd_clean ;;
  clean-all) cmd_clean_all ;;
  status)    cmd_status ;;
  *)
    echo "Usage: $0 start [<HOST_A_IP>] [--build] [--no-cache] | stop | clean | clean-all | status"
    exit 1
    ;;
esac
