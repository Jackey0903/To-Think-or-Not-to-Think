#!/usr/bin/env bash
set -euo pipefail

ROOT="${TGS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CONDA_BIN="${CONDA_BIN:-conda}"
CONDA_ENV="${TGS_CONDA_ENV:-dino}"
RUN_NAME="${RUN_NAME:-mainline_v2}"
RUN_ROOT="${ROOT}/budget_results/${RUN_NAME}"
ANALYSIS_ROOT="${ROOT}/budget_results/analysis/${RUN_NAME}"
LOG_ROOT="${ROOT}/logs/budget/${RUN_NAME}"
STATUS_TSV="${LOG_ROOT}/stage_status.tsv"
ANALYSIS_LOCK="${LOG_ROOT}/refresh.lock"
CKPT_BASE="${TGS_CKPT_BASE:-${ROOT}/results_real/epochs6_lr1e-4_bs4_gradacc8_lora_r8alpha16dropout0.05}"
CKPT_DIR="${TGS_CKPT_DIR:-${CKPT_BASE}/checkpoint-551}"
REF_META="${ROOT}/R2AVSBench/RefAVSBench_metadata.csv"
R2_META="${ROOT}/R2AVSBench/R2AVSBench_metadata.csv"
DEVICE="${TGS_DEVICE:-${DEVICE:-cuda:0}}"

mkdir -p "${RUN_ROOT}" "${ANALYSIS_ROOT}" "${LOG_ROOT}" "${RUN_ROOT}/pilots"
: > "${STATUS_TSV}"

timestamp() {
  date '+%F %T'
}

log() {
  echo "[$(timestamp)] $*"
}

record() {
  echo -e "$(timestamp)\t$1\t$2" >> "${STATUS_TSV}"
}

run_cmd_logged() {
  local name="$1"
  local logfile="$2"
  shift 2
  log "START: ${name}"
  record "START" "${name}"
  (
    set -x
    "$@"
  ) 2>&1 | tee "${logfile}"
  local rc=${PIPESTATUS[0]}
  if [[ ${rc} -ne 0 ]]; then
    log "FAIL : ${name} (rc=${rc})"
    record "FAIL" "${name}"
    return ${rc}
  fi
  log "DONE : ${name}"
  record "DONE" "${name}"
}

refresh_analysis() {
  exec 9>"${ANALYSIS_LOCK}"
  flock 9
  run_cmd_logged \
    "Refresh analysis" \
    "${LOG_ROOT}/refresh_analysis_$(date +%Y%m%d_%H%M%S).log" \
    "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python "${ROOT}/scripts/budget/refresh_budget_analysis.py" \
      --ckpt-dir "${CKPT_DIR}" \
      --budget-root "${RUN_ROOT}" \
      --output-root "${ANALYSIS_ROOT}/current"
  local rc=$?
  flock -u 9
  exec 9>&-
  return ${rc}
}

maybe_run_budget() {
  local name="$1"
  local logfile="$2"
  local metrics_path="$3"
  shift 3
  local running_flag="${metrics_path}.running"
  if [[ -f "${metrics_path}" ]]; then
    log "SKIP : ${name} because metrics already exist at ${metrics_path}"
    record "SKIP" "${name}"
    return 0
  fi
  if [[ -f "${running_flag}" ]]; then
    log "SKIP : ${name} because running flag exists at ${running_flag}"
    record "SKIP" "${name}"
    return 0
  fi
  mkdir -p "$(dirname "${running_flag}")"
  touch "${running_flag}"
  trap 'rm -f "${running_flag}"' RETURN
  run_cmd_logged "${name}" "${logfile}" "$@"
  rm -f "${running_flag}"
  trap - RETURN
}

run_refavs_testu_long() {
  maybe_run_budget \
    "RefAVS test_u long" \
    "${LOG_ROOT}/refavs_testu_long.log" \
    "${RUN_ROOT}/refavs/test_u/long/metrics.json" \
    "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python "${ROOT}/scripts/budget/infer_budget.py" \
      --dataset refavs \
      --split test_u \
      --budget-mode long \
      --output-root "${RUN_ROOT}" \
      --device "${DEVICE}" \
      --overwrite
  refresh_analysis
}

run_pilot_sequence() {
  local pilot_meta="${RUN_ROOT}/pilots/refavs_test_s_100.csv"
  run_cmd_logged \
    "Pilot RefAVS metadata test_s 100" \
    "${LOG_ROOT}/pilot_metadata.log" \
    "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python "${ROOT}/scripts/budget/make_pilot_metadata.py" \
      --input-csv "${REF_META}" \
      --split test_s \
      --limit 100 \
      --sort-by-uid \
      --output-csv "${pilot_meta}"

  maybe_run_budget \
    "RefAVS pilot100 short" \
    "${LOG_ROOT}/pilot_short.log" \
    "${RUN_ROOT}/refavs/test_s/short/metrics.json" \
    "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python "${ROOT}/scripts/budget/infer_budget.py" \
      --dataset refavs \
      --split test_s \
      --budget-mode short \
      --meta-csv "${pilot_meta}" \
      --output-root "${RUN_ROOT}" \
      --device "${DEVICE}" \
      --overwrite
  refresh_analysis

  maybe_run_budget \
    "RefAVS pilot100 rewrite" \
    "${LOG_ROOT}/pilot_rewrite.log" \
    "${RUN_ROOT}/refavs/test_s/rewrite/metrics.json" \
    "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python "${ROOT}/scripts/budget/infer_budget.py" \
      --dataset refavs \
      --split test_s \
      --budget-mode rewrite \
      --meta-csv "${pilot_meta}" \
      --output-root "${RUN_ROOT}" \
      --device "${DEVICE}" \
      --overwrite
  refresh_analysis

  maybe_run_budget \
    "RefAVS pilot100 trunc_long" \
    "${LOG_ROOT}/pilot_trunc_long.log" \
    "${RUN_ROOT}/refavs/test_s/trunc_long/metrics.json" \
    "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python "${ROOT}/scripts/budget/infer_budget.py" \
      --dataset refavs \
      --split test_s \
      --budget-mode trunc_long \
      --meta-csv "${pilot_meta}" \
      --long-jsonl "${CKPT_DIR}/inference_cot_ref_avs_test_s_bs/inference_results.jsonl" \
      --output-root "${RUN_ROOT}" \
      --device "${DEVICE}" \
      --overwrite
  refresh_analysis
}

run_r2_corrected_long() {
  maybe_run_budget \
    "R2AVS test_s corrected long" \
    "${LOG_ROOT}/r2avs_tests_long.log" \
    "${RUN_ROOT}/r2avs/test_s/long/metrics.json" \
    "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python "${ROOT}/scripts/budget/infer_budget.py" \
      --dataset r2avs \
      --split test_s \
      --budget-mode long \
      --meta-csv "${R2_META}" \
      --output-root "${RUN_ROOT}" \
      --device "${DEVICE}" \
      --overwrite
  refresh_analysis
}

cd "${ROOT}"
export PYTHONUNBUFFERED=1

log "Resume root: ${ROOT}"
log "Run root: ${RUN_ROOT}"
log "Analysis root: ${ANALYSIS_ROOT}"
log "Device: ${DEVICE}"

run_cmd_logged "EnvCheck nvidia-smi" "${LOG_ROOT}/env_nvidia_smi.log" nvidia-smi
run_cmd_logged "EnvCheck ${CONDA_ENV} python" "${LOG_ROOT}/env_python.log" "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"
refresh_analysis

log "Launching parallel tracks: refavs_testu_long + pilot_sequence"
record "INFO" "parallel launch refavs_testu_long + pilot_sequence"
( run_refavs_testu_long ) &
REFAVS_LONG_PID=$!
( run_pilot_sequence ) &
PILOT_PID=$!

wait "${PILOT_PID}"
log "Pilot sequence finished; launching R2 corrected long"
record "INFO" "pilot finished; launch r2 corrected long"
( run_r2_corrected_long ) &
R2_PID=$!

wait "${REFAVS_LONG_PID}"
wait "${R2_PID}"

refresh_analysis
log "RESUME PIPELINE COMPLETED"
record "DONE" "resume pipeline completed"
