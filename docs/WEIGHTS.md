# Model Weights and Download Links

This repository does not track model weights. Download them into the paths below before running inference or reproduction scripts.

## Required Weights

| Component | Target path | Download link | Notes |
| --- | --- | --- | --- |
| LLaMA-2-7B-Chat-HF | `pretrained_weights/Llama-2-7b-chat-hf/` | https://huggingface.co/meta-llama/Llama-2-7b-chat-hf | Gated by Meta/Hugging Face. Request access and run `huggingface-cli login` first. |
| CLIP ViT-L/14 | `pretrained_weights/clip-vit-large-patch14/` | https://huggingface.co/openai/clip-vit-large-patch14 | Visual encoder backbone. |
| BEATs Iter3+ AS2M cpt2 | `pretrained_weights/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt` | https://github.com/microsoft/unilm/tree/master/beats | Use the official BEATs table: `Fine-tuned BEATs_iter3+ (AS2M) (cpt2)`. |
| Crab audio projector | `pretrained_weights/audio_pretrain.bin` | https://huggingface.co/ahsgdxhs/Crab/tree/main | Pretrained audio projector used by the Ref-Thinker initialization. |
| Crab visual projector | `pretrained_weights/visual_pretrain.bin` | https://huggingface.co/ahsgdxhs/Crab/tree/main | Pretrained visual projector used by the Ref-Thinker initialization. |
| Ref-Thinker checkpoint | `results_real/epochs6_lr1e-4_bs4_gradacc8_lora_r8alpha16dropout0.05/` | https://huggingface.co/Jinxing1/TGS-Agent/tree/main | Contains `checkpoint-551/finetune_weights.bin` and `non_lora_trainables.bin`. |
| GroundingDINO Swin-T | `gdino_checkpoints/groundingdino_swint_ogc.pth` | https://github.com/IDEA-Research/Grounded-SAM-2 | Download via the upstream `gdino_checkpoints/download_ckpts.sh`. |
| SAM2.1 Hiera-Large | `checkpoints/sam2.1_hiera_large.pt` | https://github.com/IDEA-Research/Grounded-SAM-2 | Download via the upstream `checkpoints/download_ckpts.sh`. |

## One-Command Helper

After installing `huggingface_hub`, this helper downloads Hugging Face resources that do not require manual browser interaction:

```bash
bash scripts/download_weights.sh
```

For gated LLaMA weights:

```bash
huggingface-cli login
huggingface-cli download meta-llama/Llama-2-7b-chat-hf \
  --local-dir pretrained_weights/Llama-2-7b-chat-hf \
  --local-dir-use-symlinks False
```

For Grounded-SAM-2:

```bash
cd ground_segment_scripts
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
cd ..
ln -s ground_segment_scripts/Grounded-SAM-2/grounding_dino grounding_dino
ln -s ground_segment_scripts/Grounded-SAM-2/sam2 sam2
ln -s ground_segment_scripts/Grounded-SAM-2/checkpoints checkpoints
ln -s ground_segment_scripts/Grounded-SAM-2/gdino_checkpoints gdino_checkpoints
bash ground_segment_scripts/Grounded-SAM-2/gdino_checkpoints/download_ckpts.sh
bash ground_segment_scripts/Grounded-SAM-2/checkpoints/download_ckpts.sh
```

## Expected Final Layout

```text
pretrained_weights/
  Llama-2-7b-chat-hf/
  clip-vit-large-patch14/
  BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
  audio_pretrain.bin
  visual_pretrain.bin

results_real/
  epochs6_lr1e-4_bs4_gradacc8_lora_r8alpha16dropout0.05/
    checkpoint-551/
      finetune_weights.bin
    non_lora_trainables.bin

grounding_dino -> ground_segment_scripts/Grounded-SAM-2/grounding_dino
sam2 -> ground_segment_scripts/Grounded-SAM-2/sam2
gdino_checkpoints/
  groundingdino_swint_ogc.pth
checkpoints/
  sam2.1_hiera_large.pt
```

