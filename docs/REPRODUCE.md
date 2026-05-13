# Reproduction

## Smoke Check

```bash
python scripts/check_smoke.py
```

The smoke check validates repository structure and required local resources. It does not load the large models.

## Ref-Thinker Inference

```bash
conda activate think
TGS_DEVICE=cuda:0 bash scripts/finetune/inference_hyper_lora.sh
```

Useful overrides:

```bash
TGS_MODEL_PATH=/path/to/Llama-2-7b-chat-hf
TGS_CKPT_BASE=/path/to/released-ref-thinker-checkpoint
TGS_TEST_NAME=test_u
TGS_DEVICE=cuda:0
```

## Single Budget Evaluation

```bash
conda activate dino
python scripts/budget/infer_budget.py \
  --dataset refavs \
  --split test_u \
  --budget-mode long \
  --output-root budget_results/main \
  --device cuda:0 \
  --overwrite
```

Supported `--budget-mode` values:

```text
zero, short, long, rewrite, trunc_long
```

## Compact Pipeline

```bash
TGS_DEVICE=cuda:0 \
TGS_CONDA_ENV=dino \
bash scripts/repro/repro_full_pipeline.sh
```

The pipeline uses local paths by default and can be configured with environment variables:

```bash
TGS_CKPT_BASE=results_real/<released-ref-thinker-checkpoint>
TGS_CKPT_DIR=results_real/<released-ref-thinker-checkpoint>/checkpoint-551
TGS_MODEL_PATH=pretrained_weights/Llama-2-7b-chat-hf
TGS_VIT_PATH=pretrained_weights/clip-vit-large-patch14
TGS_BEATS_PATH=pretrained_weights/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
```

