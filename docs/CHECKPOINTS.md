# Checkpoints

Expected local layout:

```text
pretrained_weights/
  Llama-2-7b-chat-hf/
  clip-vit-large-patch14/
  BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
  audio_pretrain.bin
  visual_pretrain.bin

results_real/
  <released-ref-thinker-checkpoint>/
    checkpoint-551/
      finetune_weights.bin
    non_lora_trainables.bin
```

Ground-Segment weights are expected through the Grounded-SAM-2 layout:

```text
gdino_checkpoints/groundingdino_swint_ogc.pth
checkpoints/sam2.1_hiera_large.pt
```

The main scripts allow path overrides:

```bash
python scripts/budget/infer_budget.py \
  --dataset refavs \
  --split test_u \
  --budget-mode long \
  --model-name-or-path /path/to/Llama-2-7b-chat-hf \
  --vit-ckpt-path /path/to/clip-vit-large-patch14 \
  --beats-ckpt-path /path/to/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt \
  --ckpt-dir /path/to/checkpoint-551 \
  --avs-ckpt-dir /path/to/released-ref-thinker-checkpoint
```

For download links and helper commands, see [WEIGHTS.md](WEIGHTS.md).
