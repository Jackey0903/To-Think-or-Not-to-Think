<div align="center">

# To Think or Not to Think

### Pre-Decisional Reasoning Budgets for Referring Audio-Visual Segmentation

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white)](#installation)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white)](#installation)
[![Task](https://img.shields.io/badge/Task-Ref--AVS-2E8B57.svg)](#overview)
[![Code](https://img.shields.io/badge/Code-Released-1f6feb.svg)](https://github.com/Jackey0903/To-Think-or-Not-to-Think)
[![Weights](https://img.shields.io/badge/Weights-Links%20Provided-f59e0b.svg)](#model-zoo)
[![License](https://img.shields.io/badge/License-Pending-lightgrey.svg)](LICENSE)

**A lightweight, reproducible codebase for adaptive reasoning in referring audio-visual segmentation.**

</div>

## Overview

Long chain-of-thought reasoning is not always beneficial for referring audio-visual segmentation. Some queries are already visually and acoustically clear; forcing a long reasoning chain can introduce drift. Other queries genuinely require multi-step reasoning. This repository provides the code to evaluate and reproduce this **reasoning-budget** perspective.

We compare three core inference modes:

- **Zero reasoning**: directly ground the original referring expression.
- **Short reasoning**: generate a compact object-aware description.
- **Long reasoning**: generate the full Ref-Thinker reasoning output.

The codebase is organized around a **Think-Ground-Segment** pipeline:

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
 segmentation mask + budget metrics
```

Large datasets, model weights, generated masks, logs, and third-party source trees are intentionally excluded from Git. All required download links and expected local paths are documented below.

## News

- **Code released**: core model code, metadata CSV files, budget scripts, and reproduction helpers are available.
- **Reproducibility docs added**: installation, data preparation, checkpoint links, and smoke checks are documented.

## TODO

- Add final paper link and citation after public metadata is available.
- Add released result tables once the checkpoint hosting layout is finalized.

## Repository Structure

| Path | Description |
| --- | --- |
| `configs/` | Model and SAM2 configuration files. |
| `dataset/` | Ref-AVS data loading and multimodal preprocessing. |
| `models/` | Ref-Thinker and multimodal model components. |
| `scripts/finetune/` | Ref-Thinker training and inference entrypoints. |
| `scripts/budget/` | Reasoning-budget inference, grounding, evaluation, and aggregation. |
| `scripts/repro/` | Compact end-to-end reproduction helpers. |
| `scripts/download_weights.sh` | Helper for downloading public Hugging Face resources. |
| `ground_segment_scripts/` | Legacy Ground-Segment compatibility scripts. |
| `R2AVSBench/` | Lightweight metadata CSV files. |
| `docs/` | Installation, data, checkpoint, and reproduction guides. |

## Installation

Clone this repository:

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

Install Grounded-SAM-2 under `ground_segment_scripts/`:

```bash
cd ground_segment_scripts
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
cd ..
ln -s ground_segment_scripts/Grounded-SAM-2/grounding_dino grounding_dino
ln -s ground_segment_scripts/Grounded-SAM-2/sam2 sam2
ln -s ground_segment_scripts/Grounded-SAM-2/checkpoints checkpoints
ln -s ground_segment_scripts/Grounded-SAM-2/gdino_checkpoints gdino_checkpoints
```

See [docs/INSTALL.md](docs/INSTALL.md) for additional notes.

## Model Zoo

Download the following resources before reproduction:

| Component | Target path | Link |
| --- | --- | --- |
| LLaMA-2-7B-Chat-HF | `pretrained_weights/Llama-2-7b-chat-hf/` | [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) |
| CLIP ViT-L/14 | `pretrained_weights/clip-vit-large-patch14/` | [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) |
| BEATs cpt2 | `pretrained_weights/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt` | [microsoft/unilm BEATs](https://github.com/microsoft/unilm/tree/master/beats) |
| Audio projector | `pretrained_weights/audio_pretrain.bin` | [ahsgdxhs/Crab](https://huggingface.co/ahsgdxhs/Crab/tree/main) |
| Visual projector | `pretrained_weights/visual_pretrain.bin` | [ahsgdxhs/Crab](https://huggingface.co/ahsgdxhs/Crab/tree/main) |
| Ref-Thinker checkpoint | `results_real/epochs6_lr1e-4_bs4_gradacc8_lora_r8alpha16dropout0.05/` | [Jinxing1/TGS-Agent](https://huggingface.co/Jinxing1/TGS-Agent/tree/main) |
| GroundingDINO Swin-T | `gdino_checkpoints/groundingdino_swint_ogc.pth` | [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) |
| SAM2.1 Hiera-Large | `checkpoints/sam2.1_hiera_large.pt` | [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) |

For public Hugging Face files, run:

```bash
pip install -U huggingface_hub
bash scripts/download_weights.sh
```

LLaMA-2 is gated. Request access on Hugging Face and run:

```bash
huggingface-cli login
huggingface-cli download meta-llama/Llama-2-7b-chat-hf \
  --local-dir pretrained_weights/Llama-2-7b-chat-hf \
  --local-dir-use-symlinks False
```

For detailed path checks and manual download commands, see [docs/WEIGHTS.md](docs/WEIGHTS.md).

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

The metadata CSV files are included. Download the full Ref-AVSBench videos and masks from the official [Ref-AVS repository](https://github.com/GeWu-Lab/Ref-AVS). Download the instruction-tuning JSON from [Jinxing1/TGSAgent-FT-data](https://huggingface.co/datasets/Jinxing1/TGSAgent-FT-data/tree/main), then place it at `R2AVSBench/RefThinker_instruction_tuning_set.json`.

See [docs/DATA.md](docs/DATA.md) for details.

## Verify Your Setup

Run the smoke check before launching expensive jobs:

```bash
python scripts/check_smoke.py
```

Use strict mode after downloading data and weights:

```bash
python scripts/check_smoke.py --strict
```

The non-strict smoke check validates repository structure and reports missing local resources without failing.

## Reproduction

### Ref-Thinker Inference

```bash
conda activate think
TGS_DEVICE=cuda:0 bash scripts/finetune/inference_hyper_lora.sh
```

Common overrides:

```bash
TGS_MODEL_PATH=/path/to/Llama-2-7b-chat-hf
TGS_VIT_PATH=/path/to/clip-vit-large-patch14
TGS_BEATS_PATH=/path/to/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
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

More commands are available in [docs/REPRODUCE.md](docs/REPRODUCE.md).

## Reproducibility Notes

- All scripts use repository-relative paths by default.
- Machine-specific paths can be overridden through `TGS_*` environment variables or CLI arguments.
- Heavy resources are ignored by `.gitignore` and should be placed locally.
- `scripts/check_smoke.py --strict` is the recommended pre-flight check on a new machine.
- The legacy scripts in `ground_segment_scripts/` are kept for compatibility; the preferred public entrypoint is `scripts/budget/infer_budget.py`.

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

This project builds on Crab, Ref-AVS, Grounded-SAM-2, GroundingDINO, SAM2, BEATs, CLIP, Transformers, and PEFT-style adaptation. Please follow the licenses and terms of the upstream projects, datasets, and pretrained models.

