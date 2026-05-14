#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

HF_BIN="${HF_BIN:-huggingface-cli}"

if ! command -v "${HF_BIN}" >/dev/null 2>&1; then
  echo "huggingface-cli not found. Install it with: pip install -U huggingface_hub"
  exit 1
fi

mkdir -p pretrained_weights results_real

download_hf() {
  local repo="$1"
  local local_dir="$2"
  shift 2
  echo "==> ${repo} -> ${local_dir}"
  "${HF_BIN}" download "${repo}" "$@" \
    --local-dir "${local_dir}" \
    --local-dir-use-symlinks False
}

download_hf "openai/clip-vit-large-patch14" "pretrained_weights/clip-vit-large-patch14"
download_hf "ahsgdxhs/Crab" "pretrained_weights" audio_pretrain.bin visual_pretrain.bin
download_hf "Jinxing1/TGS-Agent" "results_real"

cat <<'EOF'

Downloaded public Hugging Face resources.

Manual remaining steps:
1. LLaMA-2-7B-Chat-HF is gated. After access approval:
   huggingface-cli login
   huggingface-cli download meta-llama/Llama-2-7b-chat-hf \
     --local-dir pretrained_weights/Llama-2-7b-chat-hf \
     --local-dir-use-symlinks False

2. Download BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt from:
   https://github.com/microsoft/unilm/tree/master/beats
   Put it at:
   pretrained_weights/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt

3. Install Grounded-SAM-2 and run its checkpoint download scripts. See docs/WEIGHTS.md.
EOF

