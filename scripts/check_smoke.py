#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


REQUIRED_FILES = [
    "configs/unified_config.py",
    "dataset/unified_dataset.py",
    "models/unified_llama.py",
    "scripts/finetune/inference_hyper_lora.py",
    "scripts/budget/infer_budget.py",
    "scripts/budget/ground_budget.py",
    "R2AVSBench/RefAVSBench_metadata.csv",
    "R2AVSBench/R2AVSBench_metadata.csv",
]


OPTIONAL_LOCAL_RESOURCES = [
    "REFAVS/media",
    "REFAVS/gt_mask",
    "R2AVSBench/RefThinker_instruction_tuning_set.json",
    "pretrained_weights/Llama-2-7b-chat-hf",
    "pretrained_weights/clip-vit-large-patch14",
    "pretrained_weights/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
    "results_real/epochs6_lr1e-4_bs4_gradacc8_lora_r8alpha16dropout0.05/checkpoint-551/finetune_weights.bin",
    "results_real/epochs6_lr1e-4_bs4_gradacc8_lora_r8alpha16dropout0.05/non_lora_trainables.bin",
    "grounding_dino",
    "sam2",
    "gdino_checkpoints/groundingdino_swint_ogc.pth",
    "checkpoints/sam2.1_hiera_large.pt",
]


def exists(path: str) -> bool:
    return (ROOT / path).exists()


def main() -> int:
    parser = argparse.ArgumentParser(description="Check repository layout and local resources.")
    parser.add_argument("--strict", action="store_true", help="Fail if optional local resources are missing.")
    args = parser.parse_args()

    missing_required = [path for path in REQUIRED_FILES if not exists(path)]
    missing_optional = [path for path in OPTIONAL_LOCAL_RESOURCES if not exists(path)]

    print(f"Repository root: {ROOT}")
    print("\nRequired files:")
    for path in REQUIRED_FILES:
        print(f"  {'OK     ' if exists(path) else 'MISSING'} {path}")

    print("\nLocal resources:")
    for path in OPTIONAL_LOCAL_RESOURCES:
        print(f"  {'OK     ' if exists(path) else 'MISSING'} {path}")

    if missing_required:
        print("\nMissing required files. The repository copy is incomplete.")
        return 1
    if args.strict and missing_optional:
        print("\nMissing local resources. See docs/DATA.md and docs/CHECKPOINTS.md.")
        return 1

    print("\nSmoke check passed.")
    if missing_optional:
        print("Some local resources are missing; this is expected before downloading data and weights.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
