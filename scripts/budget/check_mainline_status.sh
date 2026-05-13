#!/usr/bin/env bash
set -euo pipefail

ROOT="${TGS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RUN_NAME="${RUN_NAME:-mainline_v1}"
LOG_DIR="${ROOT}/logs/budget/${RUN_NAME}"
PID_FILE="${LOG_DIR}/nohup.pid"
LOG_FILE="${LOG_DIR}/nohup.log"
STATUS_FILE="${LOG_DIR}/stage_status.tsv"
RUN_ROOT="${ROOT}/budget_results/${RUN_NAME}"
ANALYSIS_ROOT="${ROOT}/budget_results/analysis/${RUN_NAME}/current"

echo "RUN_NAME=${RUN_NAME}"
if [[ -f "${PID_FILE}" ]]; then
  PID="$(cat "${PID_FILE}")"
  echo "PID=${PID}"
  if ps -p "${PID}" > /dev/null 2>&1; then
    echo "PROCESS=running"
  else
    echo "PROCESS=stopped"
  fi
else
  echo "PID=missing"
fi

echo "---- recent log ----"
if [[ -f "${LOG_FILE}" ]]; then
  tail -n 30 "${LOG_FILE}"
else
  echo "no log file yet"
fi

echo "---- stage status ----"
if [[ -f "${STATUS_FILE}" ]]; then
  tail -n 20 "${STATUS_FILE}"
else
  echo "no stage status yet"
fi

echo "---- generated metrics ----"
if [[ -d "${RUN_ROOT}" ]]; then
  find "${RUN_ROOT}" -path "*/metrics.json" | sort
else
  echo "run root missing"
fi

echo "---- analysis outputs ----"
if [[ -d "${ANALYSIS_ROOT}" ]]; then
  find "${ANALYSIS_ROOT}" -maxdepth 2 -type f | sort | sed -n '1,80p'
else
  echo "analysis root missing"
fi
