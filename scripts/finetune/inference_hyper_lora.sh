#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

llama2_ckpt_path="${TGS_MODEL_PATH:-./pretrained_weights/Llama-2-7b-chat-hf}"

export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM='true'

pretrained_ckpt_base_dir="${TGS_CKPT_BASE:-./results_real/epochs6_lr1e-4_bs4_gradacc8_lora_r8alpha16dropout0.05}"
ckpt_dir="${TGS_CKPT_DIR:-${pretrained_ckpt_base_dir}/checkpoint-551}"
test_name="${TGS_TEST_NAME:-test_u}"
device="${TGS_DEVICE:-cuda:0}"
vit_ckpt_path="${TGS_VIT_PATH:-./pretrained_weights/clip-vit-large-patch14}"
beats_ckpt_path="${TGS_BEATS_PATH:-./pretrained_weights/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt}"
budget_mode="${TGS_BUDGET_MODE:-long}"
max_new_tokens="${TGS_MAX_NEW_TOKENS:-256}"
save_name="${TGS_SAVE_NAME:-}"

cmd=(python scripts/finetune/inference_hyper_lora.py \
    --llm_name llama \
    --model_name_or_path "${llama2_ckpt_path}" \
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
    --ckpt_dir "${ckpt_dir}" \
    --avqa_task False \
    --ave_task False \
    --avvp_task False \
    --arig_task False \
    --avcap_task False \
    --ms3_task False \
    --s4_task False \
    --avss_task False \
    --ref_avs_task True \
    --avs_ckpt_dir "${pretrained_ckpt_base_dir}" \
    --test_name "${test_name}" \
    --device "${device}" \
    --multi_frames False \
    --visual_branch True \
    --video_frame_nums 10 \
    --vit_ckpt_path "${vit_ckpt_path}" \
    --select_feature patch \
    --image_size 224 \
    --patch_size 14 \
    --visual_query_token_nums 32 \
    --audio_branch True \
    --BEATs_ckpt_path "${beats_ckpt_path}"  \
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
    --output_dir 'test' \
    --max_new_tokens "${max_new_tokens}" \
    --budget_mode "${budget_mode}")

if [[ -n "${save_name}" ]]; then
    cmd+=(--save_name "${save_name}")
fi

printf '%q ' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
