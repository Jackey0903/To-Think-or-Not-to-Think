#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_BIN="${CONDA_BIN:-conda}"
CONDA_ENV="${TGS_CONDA_ENV:-dino}"
GS_DIR="${ROOT_DIR}/ground_segment_scripts"
CKPT_BASE="${TGS_CKPT_BASE:-${ROOT_DIR}/results_real/epochs6_lr1e-4_bs4_gradacc8_lora_r8alpha16dropout0.05}"
CKPT_DIR="${TGS_CKPT_DIR:-${CKPT_BASE}/checkpoint-551}"

STEP2_SCRIPT="${GS_DIR}/ground_segment_with_direct_reference_of_RefAVSBench.py"
STEP3_SCRIPT="${GS_DIR}/ground_segment_with_object_text_after_thinking_for_R2AVSBench.py"
STEP4_SCRIPT="${GS_DIR}/ground_segment_with_direct_reference_of_R2AVSBench.py"

timestamp() {
  date "+%F %T"
}

log() {
  echo "[$(timestamp)] $*"
}

run_step() {
  local name="$1"
  shift
  log "START: ${name}"
  "$@"
  log "DONE : ${name}"
}

cd "${ROOT_DIR}"
log "Resume pipeline root: ${ROOT_DIR}"

for f in "${STEP2_SCRIPT}" "${STEP3_SCRIPT}" "${STEP4_SCRIPT}"; do
  if [[ ! -f "${f}" ]]; then
    log "Missing script: ${f}"
    exit 1
  fi
done

run_step "EnvCheck conda ${CONDA_ENV}" "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python -u -c "import torch; print('torch', torch.__version__)"

run_step "Ground RefAVS direct ref" \
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" env PYTHONUNBUFFERED=1 PYTHONPATH=. python -u "${STEP2_SCRIPT}"

run_step "Ground R2AVS thinker text" \
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" env PYTHONUNBUFFERED=1 PYTHONPATH=. python -u "${STEP3_SCRIPT}"

run_step "Ground R2AVS direct ref" \
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" env PYTHONUNBUFFERED=1 PYTHONPATH=. python -u "${STEP4_SCRIPT}"

log "Done step2-step4."

python - <<'PY'
import json, os
base=os.environ.get('TGS_CKPT_DIR', os.path.join(os.getcwd(), 'results_real', 'epochs6_lr1e-4_bs4_gradacc8_lora_r8alpha16dropout0.05', 'checkpoint-551'))
targets=[
  ('RefAVS_thinker','inference_cot_ref_avs_test_s_bs/s_object_eval_results_v2_boxthre0.1_textthre0.25.json'),
  ('RefAVS_direct','inference_cot_ref_avs_test_s_bs/direct_ref_eval_results_v2_boxthre0.1_textthre0.25_reproduce.json'),
  ('R2AVS_thinker','R2AVSBenchinference_cot_ref_avs_test_s_bs/s_object_eval_results_v2_boxthre0.1_textthre0.25_reproduce.json'),
  ('R2AVS_direct','inference_cot_ref_avs_test_s_bs/R2AVSBench_butOriref_eval_results_v2_boxthre0.1_textthre0.25v2.json'),
]
for name,rel in targets:
    p=os.path.join(base,rel)
    if not os.path.exists(p):
        print(name,'missing',rel)
        continue
    d=json.load(open(p))
    miou=d.get('overall_average_iou')
    fs=d.get('overall_average_fscore')
    print(name,'mIoU',miou,'Fscore',fs,'file',rel)
PY
