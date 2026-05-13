#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION_NAME="${1:-tgs_repro}"
RUN_LOG="${ROOT_DIR}/logs/${SESSION_NAME}.log"

echo "=== tmux sessions ==="
tmux ls || true
echo

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "=== session ${SESSION_NAME} panes ==="
  tmux list-panes -t "${SESSION_NAME}" -F '#{session_name}:#{window_index}.#{pane_index} pid=#{pane_pid} cmd=#{pane_current_command}'
  echo
  echo "=== last screen output ==="
  tmux capture-pane -pt "${SESSION_NAME}" -S -30 || true
else
  echo "session not found: ${SESSION_NAME}"
fi

echo
echo "=== log tail (${RUN_LOG}) ==="
if [[ -f "${RUN_LOG}" ]]; then
  tail -n 40 "${RUN_LOG}"
else
  echo "log file not found"
fi
