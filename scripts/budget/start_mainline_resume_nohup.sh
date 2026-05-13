#!/usr/bin/env bash
set -euo pipefail

ROOT="${TGS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RUN_NAME="${RUN_NAME:-mainline_v2}"
LOG_DIR="${ROOT}/logs/budget/${RUN_NAME}"
LOG_FILE="${LOG_DIR}/nohup.log"
PID_FILE="${LOG_DIR}/nohup.pid"

mkdir -p "${LOG_DIR}"
cd "${ROOT}"
nohup setsid bash "${ROOT}/scripts/budget/run_mainline_resume_parallel.sh" > "${LOG_FILE}" 2>&1 < /dev/null &
echo $! > "${PID_FILE}"
echo "RUN_NAME=${RUN_NAME}"
echo "PID=$(cat "${PID_FILE}")"
echo "LOG_FILE=${LOG_FILE}"
