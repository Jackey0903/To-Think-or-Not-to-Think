#!/usr/bin/env bash
set -euo pipefail

ROOT="${TGS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RUN_NAME="${RUN_NAME:-mainline_v1}"
DATASET="${DATASET:?DATASET is required}"
SPLIT="${SPLIT:?SPLIT is required}"
BUDGET_MODE="${BUDGET_MODE:?BUDGET_MODE is required}"
JOB_NAME="${JOB_NAME:-${DATASET}_${SPLIT}_${BUDGET_MODE}}"
LOG_ROOT="${ROOT}/logs/budget/${RUN_NAME}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/${JOB_NAME}.log}"
PID_FILE="${PID_FILE:-${LOG_ROOT}/${JOB_NAME}.pid}"

mkdir -p "${LOG_ROOT}"

nohup env \
  RUN_NAME="${RUN_NAME}" \
  DATASET="${DATASET}" \
  SPLIT="${SPLIT}" \
  BUDGET_MODE="${BUDGET_MODE}" \
  JOB_NAME="${JOB_NAME}" \
  DEVICE="${DEVICE:-cuda:0}" \
  META_CSV="${META_CSV:-}" \
  LONG_JSONL="${LONG_JSONL:-}" \
  PREDICTIONS_JSONL="${PREDICTIONS_JSONL:-}" \
  SKIP_INFERENCE="${SKIP_INFERENCE:-0}" \
  SKIP_GROUND="${SKIP_GROUND:-0}" \
  LOG_FILE="${LOG_FILE}" \
  setsid bash "${ROOT}/scripts/budget/run_budget_job.sh" > "${LOG_FILE}" 2>&1 < /dev/null &
echo $! | tee "${PID_FILE}"
