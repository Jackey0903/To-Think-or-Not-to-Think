#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"

SESSION_NAME="${1:-tgs_repro}"
RUN_LOG="${LOG_DIR}/${SESSION_NAME}.log"

cd "${ROOT_DIR}"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION_NAME}"
  echo "attach: tmux attach -t ${SESSION_NAME}"
  exit 0
fi

tmux new-session -d -s "${SESSION_NAME}" "bash scripts/repro/repro_full_pipeline.sh |& tee -a '${RUN_LOG}'"

echo "started tmux session: ${SESSION_NAME}"
echo "log file: ${RUN_LOG}"
echo "list sessions: tmux ls"
echo "attach: tmux attach -t ${SESSION_NAME}"
echo "tail log: tail -f '${RUN_LOG}'"
