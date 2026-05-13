#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"

CONDA_BIN="${CONDA_BIN:-conda}"
CONDA_ENV="${TGS_CONDA_ENV:-dino}"

CKPT_BASE="${TGS_CKPT_BASE:-${ROOT_DIR}/results_real/epochs6_lr1e-4_bs4_gradacc8_lora_r8alpha16dropout0.05}"
CKPT_DIR="${TGS_CKPT_DIR:-${CKPT_BASE}/checkpoint-551}"
INFER_DIR="${CKPT_DIR}/inference_cot_ref_avs_test_s_bs"
INFER_JSONL="${INFER_DIR}/inference_results.jsonl"
R2_ALIAS_DIR="${CKPT_DIR}/R2AVSBenchinference_cot_ref_avs_test_s_bs"
GS_DIR="${ROOT_DIR}/ground_segment_scripts"
STEP1_SCRIPT="${GS_DIR}/ground_segment_with_object_text_after_thinking_for_RefAVSBench.py"
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
log "Pipeline root: ${ROOT_DIR}"

for f in "${STEP1_SCRIPT}" "${STEP2_SCRIPT}" "${STEP3_SCRIPT}" "${STEP4_SCRIPT}"; do
  if [[ ! -f "${f}" ]]; then
    log "Missing script: ${f}"
    exit 1
  fi
done

run_step "EnvCheck tmux" tmux -V
run_step "EnvCheck conda ${CONDA_ENV}" "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python -u -c "import torch; print('torch', torch.__version__)"

if [[ ! -f "${INFER_JSONL}" ]]; then
  log "Inference result missing, running Ref-Thinker inference (test_s)..."
  run_step "RefThinker inference test_s" "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python -u scripts/finetune/inference_hyper_lora.py \
    --llm_name llama \
    --model_name_or_path "${TGS_MODEL_PATH:-./pretrained_weights/Llama-2-7b-chat-hf}" \
    --freeze_backbone True \
    --lora_enable True \
    --use_hyper_lora False \
    --use_process True \
    --bits 32 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --bf16 True \
    --tf32 False \
    --fp16 False \
    --ckpt_dir "${CKPT_DIR}" \
    --avqa_task False \
    --ave_task False \
    --avvp_task False \
    --arig_task False \
    --avcap_task False \
    --ms3_task False \
    --s4_task False \
    --avss_task False \
    --ref_avs_task True \
    --avs_ckpt_dir "${CKPT_BASE}" \
    --test_name test_s \
    --device "${TGS_DEVICE:-cuda:0}" \
    --multi_frames False \
    --visual_branch True \
    --video_frame_nums 10 \
    --vit_ckpt_path "${TGS_VIT_PATH:-./pretrained_weights/clip-vit-large-patch14}" \
    --select_feature patch \
    --image_size 224 \
    --patch_size 14 \
    --visual_query_token_nums 32 \
    --audio_branch True \
    --BEATs_ckpt_path "${TGS_BEATS_PATH:-./pretrained_weights/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt}" \
    --audio_query_token_nums 32 \
    --seg_branch False \
    --prompt_embed_dim 256 \
    --mask_decoder_transformer_depth 2 \
    --low_res_mask_size 112 \
    --image_scale_nums 2 \
    --token_nums_per_scale 3 \
    --avs_query_num 300 \
    --num_classes 1 \
    --query_generator_num_layers 2 \
    --output_dir test
else
  log "Found existing inference jsonl: ${INFER_JSONL}"
fi

if [[ ! -e "${R2_ALIAS_DIR}" ]]; then
  log "Creating compatibility symlink for R2 script path: ${R2_ALIAS_DIR}"
  ln -s "${INFER_DIR}" "${R2_ALIAS_DIR}"
fi

run_step "Ground RefAVS thinker text" \
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" env PYTHONUNBUFFERED=1 PYTHONPATH=. python -u "${STEP1_SCRIPT}"

run_step "Ground RefAVS direct ref" \
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" env PYTHONUNBUFFERED=1 PYTHONPATH=. python -u "${STEP2_SCRIPT}"

run_step "Ground R2AVS thinker text" \
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" env PYTHONUNBUFFERED=1 PYTHONPATH=. python -u "${STEP3_SCRIPT}"

run_step "Ground R2AVS direct ref" \
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" env PYTHONUNBUFFERED=1 PYTHONPATH=. python -u "${STEP4_SCRIPT}"

log "All pipeline steps completed."
