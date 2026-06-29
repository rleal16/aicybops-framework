#!/usr/bin/env bash
# Smoke-test a single DAM train job and capture *every* diagnostic we need
# if the worker dies.
#
# Usage:
#   scripts/smoke_train.sh [--epochs N] [--window-min N] [--batch-size N]
#                          [--timeout SECONDS] [--api-host HOST:PORT]
#
# Outputs (in /tmp/aicybops_smoke/<timestamp>/):
#   worker.log         - full timestamped worker logs during the run
#   train_resp.json    - response body from POST /train/
#   job_status.json    - final Redis job record
#   inspect.txt        - docker inspect summary (status/exit/oom/restarts)
#   dmesg.txt          - last 200 dmesg lines filtered for OOM/kill
#   summary.txt        - one-line verdict (success | python_error | native_crash | oom_kill | timeout)

set -uo pipefail

EPOCHS=1
WINDOW_MIN=5
BATCH_SIZE=32
TIMEOUT_SECONDS=600
API="http://localhost:8000"
WORKER="aicybops-aicybops-worker-1"
REDIS="aicybops-redis-1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --epochs) EPOCHS="$2"; shift 2 ;;
    --window-min) WINDOW_MIN="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --timeout) TIMEOUT_SECONDS="$2"; shift 2 ;;
    --api-host) API="http://$2"; shift 2 ;;
    -h|--help)
      sed -n '2,15p' "$0"; exit 0 ;;
    *)
      echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

TS="$(date +%Y%m%d-%H%M%S)"
OUT="/tmp/aicybops_smoke/$TS"
mkdir -p "$OUT"

# Choose a sudo prefix only if we need it.
if docker ps >/dev/null 2>&1; then
  SUDO=""
else
  SUDO="sudo"
fi

log() { printf '[smoke %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

cleanup() {
  if [[ -n "${LOGPID:-}" ]] && kill -0 "$LOGPID" 2>/dev/null; then
    kill "$LOGPID" 2>/dev/null || true
    wait "$LOGPID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

log "output dir: $OUT"

# --- 0. Pre-flight: worker has the new code? --------------------------------
log "verifying running container has faulthandler + swapaxes"
$SUDO docker exec "$WORKER" python -c '
import inspect, aicybops_lib.server.worker as w
src = inspect.getsource(w)
print("worker_file=", w.__file__)
print("has_faulthandler=", "faulthandler" in src)
' 2>&1 | tee "$OUT/preflight.txt" || {
  log "ERROR: worker container not running or python import failed"
  exit 3
}

$SUDO docker exec "$WORKER" grep -n "swapaxes" \
  /app/aicybops_models/dam_model/processing/data_analysis.py >> "$OUT/preflight.txt" 2>&1 || true

# --- 1. Start the log stream -------------------------------------------------
log "streaming worker logs to $OUT/worker.log"
$SUDO docker logs --tail 0 -f --timestamps "$WORKER" >"$OUT/worker.log" 2>&1 &
LOGPID=$!
sleep 1

# --- 2. Submit the job -------------------------------------------------------
log "submitting train job (epochs=$EPOCHS window_min=$WINDOW_MIN batch_size=$BATCH_SIZE)"
HTTP_CODE=$(curl -s -o "$OUT/train_resp.json" -w '%{http_code}' \
  -X POST "$API/train/" \
  -H 'Content-Type: application/json' \
  -d "{\"params\":{\"batch_size\":$BATCH_SIZE,\"training_window_minutes\":$WINDOW_MIN},\"epochs\":$EPOCHS,\"experiment_name\":\"smoke\",\"model_type\":\"dam\",\"run_optimization\":false}")

if [[ "$HTTP_CODE" != "200" && "$HTTP_CODE" != "202" ]]; then
  log "ERROR: POST /train/ returned HTTP $HTTP_CODE"
  cat "$OUT/train_resp.json"
  exit 4
fi

JOB=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["job_id"])' "$OUT/train_resp.json")
log "job_id=$JOB"

# --- 3. Poll until terminal state or timeout --------------------------------
START=$(date +%s)
VERDICT="unknown"
LAST_STATUS=""
while :; do
  RAW=$($SUDO docker exec "$REDIS" redis-cli --no-raw GET "aicybops:job:train:$JOB" 2>/dev/null || true)
  RAW=${RAW%\"}; RAW=${RAW#\"}
  STATUS=$(python3 - <<PY 2>/dev/null || echo "unknown"
import json
try:
    raw = """$RAW"""
    raw = raw.encode().decode('unicode_escape')
    obj = json.loads(raw)
    print(obj.get("status", "unknown"))
except Exception:
    print("unknown")
PY
)

  if [[ "$STATUS" != "$LAST_STATUS" ]]; then
    log "job status: $STATUS"
    LAST_STATUS="$STATUS"
  fi

  case "$STATUS" in
    completed) VERDICT="success"; break ;;
    failed)    VERDICT="python_error"; break ;;
  esac

  # Container died?
  CSTATE=$($SUDO docker inspect "$WORKER" --format '{{.State.Status}}|{{.State.ExitCode}}|{{.State.OOMKilled}}|{{.RestartCount}}' 2>/dev/null || echo "?|?|?|?")
  if [[ "$CSTATE" == *"exited"* ]]; then
    log "worker container in exited state: $CSTATE"
    break
  fi

  NOW=$(date +%s)
  if (( NOW - START >= TIMEOUT_SECONDS )); then
    log "timeout after ${TIMEOUT_SECONDS}s with status=$STATUS"
    VERDICT="timeout"
    break
  fi
  sleep 3
done

# Stop the log tail.
cleanup
LOGPID=""

# --- 4. Collect post-mortem artifacts ---------------------------------------
log "collecting post-mortem artifacts"

$SUDO docker exec "$REDIS" redis-cli --no-raw GET "aicybops:job:train:$JOB" \
  > "$OUT/job_status.json" 2>/dev/null || true

$SUDO docker inspect "$WORKER" --format \
  'status={{.State.Status}}
exit={{.State.ExitCode}}
oom={{.State.OOMKilled}}
error={{.State.Error}}
restarts={{.RestartCount}}
started={{.State.StartedAt}}
finished={{.State.FinishedAt}}
mem_limit={{.HostConfig.Memory}}
memswap_limit={{.HostConfig.MemorySwap}}' \
  > "$OUT/inspect.txt" 2>/dev/null || true

$SUDO dmesg -T 2>/dev/null | tail -n 400 \
  | grep -iE 'oom|killed process|out of memory|aicybops' \
  > "$OUT/dmesg.txt" || echo "no OOM lines in dmesg" > "$OUT/dmesg.txt"

# --- 5. Decide verdict from artifacts ---------------------------------------
EXIT_CODE=$(grep '^exit=' "$OUT/inspect.txt" | cut -d= -f2 | tr -d '[:space:]' || echo "")
OOM=$(grep '^oom=' "$OUT/inspect.txt" | cut -d= -f2 | tr -d '[:space:]' || echo "false")

if [[ "$VERDICT" == "unknown" || "$VERDICT" == "python_error" ]]; then
  if grep -q 'Fatal Python error' "$OUT/worker.log"; then
    VERDICT="native_crash"
  elif [[ "$OOM" == "true" ]] || [[ "$EXIT_CODE" == "137" ]] || grep -qi 'out of memory' "$OUT/dmesg.txt"; then
    VERDICT="oom_kill"
  elif grep -q 'Traceback (most recent call last):' "$OUT/worker.log"; then
    VERDICT="python_error"
  fi
fi

# --- 6. Print summary --------------------------------------------------------
{
  echo "verdict=$VERDICT"
  echo "job_id=$JOB"
  echo "output_dir=$OUT"
  echo
  echo "--- inspect ---"
  cat "$OUT/inspect.txt" 2>/dev/null || true
  echo
  echo "--- last 30 worker log lines ---"
  tail -n 30 "$OUT/worker.log" 2>/dev/null || true
  echo
  echo "--- sequence shapes seen ---"
  grep -E "Sequences for|Aligned frames|Tensors ready|View run" "$OUT/worker.log" || true
  echo
  echo "--- error/traceback excerpts ---"
  grep -nE 'Fatal Python error|Traceback|Error|Killed' "$OUT/worker.log" | head -n 40 || true
  echo
  echo "--- dmesg OOM excerpts ---"
  cat "$OUT/dmesg.txt"
} | tee "$OUT/summary.txt"

log "done. summary -> $OUT/summary.txt"

case "$VERDICT" in
  success) exit 0 ;;
  python_error|native_crash|oom_kill) exit 10 ;;
  timeout) exit 20 ;;
  *) exit 30 ;;
esac
