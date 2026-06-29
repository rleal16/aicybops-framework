#!/usr/bin/env bash
set -u

# Run integration scripts sequentially and print per-step status.

CONDA_ENV="${CONDA_ENV:-aicybops}"
REPO_ROOT="${REPO_ROOT:-$(pwd)}"

if [[ ! -d "${REPO_ROOT}/scripts" ]]; then
  echo "[ERROR] REPO_ROOT does not look like AICybOps: ${REPO_ROOT}"
  exit 2
fi

if ! source "$(conda info --base)/etc/profile.d/conda.sh"; then
  echo "[ERROR] Failed to load conda shell integration"
  exit 2
fi
if ! conda activate "${CONDA_ENV}"; then
  echo "[ERROR] Failed to activate conda env: ${CONDA_ENV}"
  exit 2
fi

export PYTHONPATH="${REPO_ROOT}/aicybops-lib/src:${REPO_ROOT}"
export PYTHONUNBUFFERED=1

cd "${REPO_ROOT}" || exit 2

declare -a NAMES=(
  "deploy_model_modelos_uc.py"
  "predict_live_modelos_uc.py"
  "deploy_model_uc2.py"
  "predict_live_uc2.py"
  "deploy_model.py"
  "predict_live.py"
)

declare -a CMDS=(
  "python scripts/deploy_model_modelos_uc.py"
  "python scripts/predict_live_modelos_uc.py"
  "python scripts/deploy_model_uc2.py"
  "python scripts/predict_live_uc2.py"
  "python scripts/deploy_model.py --training-window 20 --epochs 2 --max-evals 2"
  "python scripts/predict_live.py"
)

declare -a EXITS

echo "Running integration scripts from: ${REPO_ROOT}"
echo "Conda env: ${CONDA_ENV}"
echo

for i in "${!NAMES[@]}"; do
  idx=$((i + 1))
  name="${NAMES[$i]}"
  cmd="${CMDS[$i]}"
  echo "=== ${idx}/${#NAMES[@]} ${name} ==="
  bash -lc "${cmd}"
  rc=$?
  EXITS[$i]=$rc
  echo "exit=${rc}"
  echo
done

echo "=== SUMMARY ==="
for i in "${!NAMES[@]}"; do
  printf "%-30s %s\n" "${NAMES[$i]}" "${EXITS[$i]}"
done
echo

ok=0
if [[ "${EXITS[0]}" -eq 0 \
   && "${EXITS[2]}" -eq 0 \
   && "${EXITS[3]}" -eq 0 \
   && "${EXITS[4]}" -eq 0 \
   && "${EXITS[5]}" -eq 0 \
   && ( "${EXITS[1]}" -eq 0 || "${EXITS[1]}" -eq 1 ) ]]; then
  ok=1
fi

if [[ "${ok}" -eq 1 ]]; then
  echo "Overall result: PASS"
  exit 0
fi

echo "Overall result: FAIL"
exit 1
