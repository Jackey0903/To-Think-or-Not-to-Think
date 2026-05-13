#!/usr/bin/env bash
set -euo pipefail

ROOT="${TGS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CONDA_BIN="${CONDA_BIN:-conda}"
CONDA_ENV="${TGS_CONDA_ENV:-dino}"
RUN_NAME="${RUN_NAME:-mainline_v1}"
RUN_ROOT="${ROOT}/budget_results/${RUN_NAME}"
ANALYSIS_ROOT="${ROOT}/budget_results/analysis/${RUN_NAME}"
LOG_ROOT="${ROOT}/logs/budget/${RUN_NAME}"
CKPT_BASE="${TGS_CKPT_BASE:-${ROOT}/results_real/epochs6_lr1e-4_bs4_gradacc8_lora_r8alpha16dropout0.05}"
CKPT_DIR="${TGS_CKPT_DIR:-${CKPT_BASE}/checkpoint-551}"
REF_META="${ROOT}/R2AVSBench/RefAVSBench_metadata.csv"
R2_META="${ROOT}/R2AVSBench/R2AVSBench_metadata.csv"
DEVICE="${TGS_DEVICE:-${DEVICE:-cuda:0}}"

mkdir -p "${RUN_ROOT}" "${ANALYSIS_ROOT}" "${LOG_ROOT}"
STATUS_TSV="${LOG_ROOT}/stage_status.tsv"
: > "${STATUS_TSV}"

timestamp() {
  date '+%F %T'
}

log() {
  echo "[$(timestamp)] $*"
}

run_step() {
  local name="$1"
  shift
  log "START: ${name}"
  echo -e "$(timestamp)\tSTART\t${name}" >> "${STATUS_TSV}"
  "$@"
  log "DONE : ${name}"
  echo -e "$(timestamp)\tDONE\t${name}" >> "${STATUS_TSV}"
}

refresh_analysis() {
  run_step "Refresh analysis" \
    "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python "${ROOT}/scripts/budget/refresh_budget_analysis.py" \
      --ckpt-dir "${CKPT_DIR}" \
      --budget-root "${RUN_ROOT}" \
      --output-root "${ANALYSIS_ROOT}/current"
}

cd "${ROOT}"
export PYTHONUNBUFFERED=1

log "Pipeline root: ${ROOT}"
log "Run root: ${RUN_ROOT}"
log "Analysis root: ${ANALYSIS_ROOT}"
log "Device: ${DEVICE}"

run_step "EnvCheck nvidia-smi" nvidia-smi
run_step "EnvCheck ${CONDA_ENV} python" "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"

run_step "Pilot RefAVS metadata test_s 100" \
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python "${ROOT}/scripts/budget/make_pilot_metadata.py" \
    --input-csv "${REF_META}" \
    --split test_s \
    --limit 100 \
    --sort-by-uid \
    --output-csv "${RUN_ROOT}/pilots/refavs_test_s_100.csv"

run_step "Collect existing baselines" \
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python "${ROOT}/scripts/budget/collect_existing_results.py" \
    --ckpt-dir "${CKPT_DIR}" \
    --output-root "${ANALYSIS_ROOT}/legacy_snapshot"

refresh_analysis

run_step "RefAVS test_u zero" \
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python "${ROOT}/scripts/budget/infer_budget.py" \
    --dataset refavs \
    --split test_u \
    --budget-mode zero \
    --output-root "${RUN_ROOT}" \
    --device "${DEVICE}" \
    --overwrite
refresh_analysis

run_step "RefAVS test_u long" \
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python "${ROOT}/scripts/budget/infer_budget.py" \
    --dataset refavs \
    --split test_u \
    --budget-mode long \
    --output-root "${RUN_ROOT}" \
    --device "${DEVICE}" \
    --overwrite
refresh_analysis

run_step "R2AVS test_s corrected long" \
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python "${ROOT}/scripts/budget/infer_budget.py" \
    --dataset r2avs \
    --split test_s \
    --budget-mode long \
    --meta-csv "${R2_META}" \
    --output-root "${RUN_ROOT}" \
    --device "${DEVICE}" \
    --overwrite
refresh_analysis

run_step "RefAVS pilot100 short" \
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python "${ROOT}/scripts/budget/infer_budget.py" \
    --dataset refavs \
    --split test_s \
    --budget-mode short \
    --meta-csv "${RUN_ROOT}/pilots/refavs_test_s_100.csv" \
    --output-root "${RUN_ROOT}" \
    --device "${DEVICE}" \
    --overwrite
refresh_analysis

run_step "RefAVS pilot100 rewrite" \
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python "${ROOT}/scripts/budget/infer_budget.py" \
    --dataset refavs \
    --split test_s \
    --budget-mode rewrite \
    --meta-csv "${RUN_ROOT}/pilots/refavs_test_s_100.csv" \
    --output-root "${RUN_ROOT}" \
    --device "${DEVICE}" \
    --overwrite
refresh_analysis

run_step "RefAVS pilot100 trunc_long" \
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python "${ROOT}/scripts/budget/infer_budget.py" \
    --dataset refavs \
    --split test_s \
    --budget-mode trunc_long \
    --meta-csv "${RUN_ROOT}/pilots/refavs_test_s_100.csv" \
    --long-jsonl "${CKPT_DIR}/inference_cot_ref_avs_test_s_bs/inference_results.jsonl" \
    --output-root "${RUN_ROOT}" \
    --device "${DEVICE}" \
    --overwrite
refresh_analysis

log "PIPELINE COMPLETED"
