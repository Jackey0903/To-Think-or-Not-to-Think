<div align="center">

# To Think or Not to Think

### Pre-Decisional Reasoning Budgets for Referring Audio-Visual Segmentation

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white)](#installation)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white)](#installation)
[![Task](https://img.shields.io/badge/Task-Referring%20Audio--Visual%20Segmentation-2E8B57.svg)](#overview)
[![Code](https://img.shields.io/badge/Code-Released-1f6feb.svg)](https://github.com/Jackey0903/To-Think-or-Not-to-Think)
[![License](https://img.shields.io/badge/License-Pending-lightgrey.svg)](LICENSE)

**A lightweight research codebase for adaptive reasoning in referring audio-visual segmentation.**

</div>

## Overview

Recent multimodal segmentation systems often apply long chain-of-thought reasoning before grounding every query. This repository studies a more selective question: **can the model decide how much to think before it starts generating reasoning tokens?**

We provide code for evaluating reasoning budgets in referring audio-visual segmentation:

- **Zero reasoning**: ground directly from the original reference.
- **Short reasoning**: generate a compact object-aware description.
- **Long reasoning**: use full Ref-Thinker reasoning before grounding.
- **Budget analysis**: compare per-sample trade-offs between segmentation quality and generated token cost.

The repository is intentionally lightweight. Datasets, pretrained backbones, Ref-Thinker checkpoints, logs, and generated masks are downloaded or generated locally rather than committed to Git.

## Method at a Glance

```text
video + audio + reference
        |
        v
   Ref-Thinker
        |
        +---- zero  reasoning: original reference
        +---- short reasoning: compact object description
        +---- long  reasoning: full object-aware reasoning
        |
        v
 GroundingDINO + SAM2
        |
        v
 Referring audio-visual segmentation mask
```

The main experimental scripts compare these budgets on Ref-AVSBench and R2-AVSBench style metadata splits.

## News

- Code skeleton, metadata, documentation, and reproduction scripts are released.
- Dataset, checkpoint, and third-party model weights are intentionally kept outside the Git repository. See [Data](docs/DATA.md) and [Checkpoints](docs/CHECKPOINTS.md).

## Repository Structure

| Path | Description |
| --- | --- |
| `configs/` | Model and SAM2 configuration files. |
| `dataset/` | Ref-AVS data loading and multimodal preprocessing. |
| `models/` | Ref-Thinker and multimodal model components. |
| `scripts/finetune/` | Ref-Thinker training and inference entrypoints. |
| `scripts/budget/` | Reasoning-budget inference, grounding, evaluation, and aggregation. |
| `scripts/repro/` | Compact end-to-end reproduction helpers. |
| `ground_segment_scripts/` | Legacy Ground-Segment compatibility scripts. |
| `R2AVSBench/` | Lightweight metadata CSV files. |
| `docs/` | Installation, data, checkpoint, and reproduction guides. |

## Installation

Clone the repository:

```bash
git clone https://github.com/Jackey0903/To-Think-or-Not-to-Think.git
cd To-Think-or-Not-to-Think
```

Create the Ref-Thinker environment:

```bash
conda env create -f think_environment.yml
conda activate think
```

Create the Ground-Segment environment:

```bash
conda env create -f dinosam2_environment.yml
conda activate dino
```

Install Grounded-SAM-2 following [docs/INSTALL.md](docs/INSTALL.md).

## Data Preparation

Expected local layout:

```text
REFAVS/
  media/
  gt_mask/

R2AVSBench/
  RefAVSBench_metadata.csv
  R2AVSBench_metadata.csv
  RefAVSBenchRef_R2AVSBenchVideo.csv
  RefThinker_instruction_tuning_set.json
```

The metadata CSV files are included. The full videos, masks, and instruction-tuning JSON should be placed locally following [docs/DATA.md](docs/DATA.md).

## Checkpoint Preparation

Expected local layout:

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
```

GroundingDINO and SAM2 weights are expected through the Grounded-SAM-2-compatible symlinks described in [docs/CHECKPOINTS.md](docs/CHECKPOINTS.md).

## Quick Sanity Check

After preparing local resources, run:

```bash
python scripts/check_smoke.py
```

Before downloading data and weights, the smoke check will still pass but report missing local resources.

## Reproduction

### Ref-Thinker Inference

```bash
conda activate think
TGS_DEVICE=cuda:0 bash scripts/finetune/inference_hyper_lora.sh
```

Useful environment overrides:

```bash
TGS_MODEL_PATH=/path/to/Llama-2-7b-chat-hf
TGS_CKPT_BASE=/path/to/ref-thinker-checkpoint
TGS_CKPT_DIR=/path/to/ref-thinker-checkpoint/checkpoint-551
TGS_TEST_NAME=test_u
TGS_DEVICE=cuda:0
```

### Single Budget Evaluation

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

Supported budget modes:

```text
zero, short, long, rewrite, trunc_long
```

### Compact Pipeline

```bash
TGS_DEVICE=cuda:0 \
TGS_CONDA_ENV=dino \
bash scripts/repro/repro_full_pipeline.sh
```

More reproduction details are available in [docs/REPRODUCE.md](docs/REPRODUCE.md).

## Main Configurable Paths

| Variable | Default |
| --- | --- |
| `TGS_MODEL_PATH` | `pretrained_weights/Llama-2-7b-chat-hf` |
| `TGS_VIT_PATH` | `pretrained_weights/clip-vit-large-patch14` |
| `TGS_BEATS_PATH` | `pretrained_weights/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt` |
| `TGS_CKPT_BASE` | `results_real/epochs6_lr1e-4_bs4_gradacc8_lora_r8alpha16dropout0.05` |
| `TGS_CKPT_DIR` | `$TGS_CKPT_BASE/checkpoint-551` |
| `TGS_DEVICE` | `cuda:0` for Ref-Thinker scripts; `cuda` for Ground-Segment scripts |

## Notes on Lightweight Release

This repository does **not** track:

- Ref-AVSBench video frames and ground-truth masks.
- LLaMA, CLIP, BEATs, GroundingDINO, SAM2, or Ref-Thinker weights.
- Generated predictions, masks, logs, and intermediate experiment outputs.
- Full third-party source trees.

This keeps the repository suitable for GitHub while preserving all code paths needed to reproduce the main experiments after local resource preparation.

## Citation

If this repository helps your research, please cite the paper. The final BibTeX entry will be updated when public metadata is available.

```bibtex
@misc{to_think_or_not_to_think,
  title  = {To Think or Not to Think: Pre-Decisional Reasoning Budgets for Referring Audio-Visual Segmentation},
  author = {Anonymous},
  year   = {2026}
}
```

## Acknowledgements

This project builds on Crab, Grounded-SAM-2, GroundingDINO, SAM2, BEATs, CLIP, Transformers, and PEFT-style adaptation. Please follow the licenses and terms of the upstream projects, datasets, and pretrained models.

