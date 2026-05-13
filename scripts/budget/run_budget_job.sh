#!/usr/bin/env bash
set -euo pipefail

ROOT="${TGS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CONDA_BIN="${CONDA_BIN:-conda}"
CONDA_ENV="${TGS_CONDA_ENV:-dino}"
RUN_NAME="${RUN_NAME:-mainline_v1}"
DATASET="${DATASET:?DATASET is required}"
SPLIT="${SPLIT:?SPLIT is required}"
BUDGET_MODE="${BUDGET_MODE:?BUDGET_MODE is required}"
RUN_ROOT="${ROOT}/budget_results/${RUN_NAME}"
ANALYSIS_ROOT="${ROOT}/budget_results/analysis/${RUN_NAME}"
LOG_ROOT="${ROOT}/logs/budget/${RUN_NAME}"
ANALYSIS_LOCK="${LOG_ROOT}/refresh.lock"
CKPT_BASE="${TGS_CKPT_BASE:-${ROOT}/results_real/epochs6_lr1e-4_bs4_gradacc8_lora_r8alpha16dropout0.05}"
CKPT_DIR="${TGS_CKPT_DIR:-${CKPT_BASE}/checkpoint-551}"
DEVICE="${TGS_DEVICE:-${DEVICE:-cuda:0}}"
OUTPUT_DIR="${RUN_ROOT}/${DATASET}/${SPLIT}/${BUDGET_MODE}"
METRICS_PATH="${OUTPUT_DIR}/metrics.json"
RUNNING_FLAG="${METRICS_PATH}.running"
JOB_NAME="${JOB_NAME:-${DATASET}_${SPLIT}_${BUDGET_MODE}}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/${JOB_NAME}.log}"

mkdir -p "${RUN_ROOT}" "${ANALYSIS_ROOT}" "${LOG_ROOT}" "${OUTPUT_DIR}"

timestamp() {
  date '+%F %T'
}

log() {
  echo "[$(timestamp)] $*"
}

cleanup() {
  rm -f "${RUNNING_FLAG}"
}

refresh_analysis() {
  exec 9>"${ANALYSIS_LOCK}"
  flock 9
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python "${ROOT}/scripts/budget/refresh_budget_analysis.py" \
    --ckpt-dir "${CKPT_DIR}" \
    --budget-root "${RUN_ROOT}" \
    --output-root "${ANALYSIS_ROOT}/current"
  flock -u 9
  exec 9>&-
}

if [[ -f "${METRICS_PATH}" ]]; then
  log "SKIP: metrics already exist at ${METRICS_PATH}"
  exit 0
fi

if [[ -f "${RUNNING_FLAG}" ]]; then
  log "SKIP: running flag already exists at ${RUNNING_FLAG}"
  exit 0
fi

touch "${RUNNING_FLAG}"
trap cleanup EXIT

CMD=(
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python "${ROOT}/scripts/budget/infer_budget.py"
  --dataset "${DATASET}"
  --split "${SPLIT}"
  --budget-mode "${BUDGET_MODE}"
  --output-root "${RUN_ROOT}"
  --device "${DEVICE}"
  --overwrite
)

if [[ -n "${META_CSV:-}" ]]; then
  CMD+=(--meta-csv "${META_CSV}")
fi

if [[ -n "${LONG_JSONL:-}" ]]; then
  CMD+=(--long-jsonl "${LONG_JSONL}")
fi

if [[ -n "${PREDICTIONS_JSONL:-}" ]]; then
  CMD+=(--predictions-jsonl "${PREDICTIONS_JSONL}")
fi

if [[ "${SKIP_INFERENCE:-0}" == "1" ]]; then
  CMD+=(--skip-inference)
fi

if [[ "${SKIP_GROUND:-0}" == "1" ]]; then
  CMD+=(--skip-ground)
fi

log "START ${JOB_NAME}"
printf '%q ' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
log "DONE ${JOB_NAME}"

log "START refresh_analysis"
refresh_analysis
log "DONE refresh_analysis"
