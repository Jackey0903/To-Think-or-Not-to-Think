import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from scripts.budget.common import DEFAULT_BUDGET_ROOT, ROOT, default_meta_csv, ensure_dir


DEFAULT_PRETRAIN_BASE = Path(os.environ.get(
    "TGS_CKPT_BASE",
    ROOT / "results_real" / "epochs6_lr1e-4_bs4_gradacc8_lora_r8alpha16dropout0.05",
))
DEFAULT_CKPT_DIR = DEFAULT_PRETRAIN_BASE / "checkpoint-551"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["refavs", "r2avs"], required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--budget-mode", choices=["zero", "long", "short", "rewrite", "trunc_long"], required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_BUDGET_ROOT))
    parser.add_argument("--ckpt-dir", default=str(DEFAULT_CKPT_DIR))
    parser.add_argument("--avs-ckpt-dir", default=str(DEFAULT_PRETRAIN_BASE))
    parser.add_argument("--device", default=os.environ.get("TGS_DEVICE", "cuda:0"))
    parser.add_argument("--model-name-or-path", default=os.environ.get("TGS_MODEL_PATH", "./pretrained_weights/Llama-2-7b-chat-hf"))
    parser.add_argument("--vit-ckpt-path", default=os.environ.get("TGS_VIT_PATH", "./pretrained_weights/clip-vit-large-patch14"))
    parser.add_argument("--beats-ckpt-path", default=os.environ.get("TGS_BEATS_PATH", "./pretrained_weights/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"))
    parser.add_argument("--meta-csv", default=None)
    parser.add_argument("--long-jsonl", default=None)
    parser.add_argument("--predictions-jsonl", default=None)
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--skip-ground", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_command(cmd, cwd=ROOT):
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def budget_defaults(budget_mode):
    if budget_mode == "long":
        return {"max_new_tokens": 256, "budget_mode": "long"}
    if budget_mode == "short":
        return {"max_new_tokens": 60, "budget_mode": "short"}
    if budget_mode == "rewrite":
        return {"max_new_tokens": 60, "budget_mode": "rewrite"}
    return {"max_new_tokens": 60, "budget_mode": budget_mode}


def copy_predictions(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main():
    args = parse_args()
    meta_csv = Path(args.meta_csv) if args.meta_csv else default_meta_csv(args.dataset)
    budget_dir = ensure_dir(Path(args.output_root) / args.dataset / args.split / args.budget_mode)
    predictions_jsonl = Path(args.predictions_jsonl) if args.predictions_jsonl else budget_dir / "predictions.jsonl"

    if args.budget_mode in {"long", "short", "rewrite"} and not args.skip_inference:
        defaults = budget_defaults(args.budget_mode)
        save_name = f"budget_{args.dataset}_{args.split}_{args.budget_mode}"
        run_command(
            [
                sys.executable,
                "scripts/finetune/inference_hyper_lora.py",
                "--llm_name", "llama",
                "--model_name_or_path", args.model_name_or_path,
                "--freeze_backbone", "True",
                "--lora_enable", "True",
                "--use_hyper_lora", "False",
                "--use_process", "True",
                "--bits", "32",
                "--lora_r", "8",
                "--lora_alpha", "16",
                "--lora_dropout", "0.05",
                "--bf16", "True",
                "--tf32", "False",
                "--fp16", "False",
                "--ckpt_dir", args.ckpt_dir,
                "--avqa_task", "False",
                "--ave_task", "False",
                "--avvp_task", "False",
                "--arig_task", "False",
                "--avcap_task", "False",
                "--ms3_task", "False",
                "--s4_task", "False",
                "--avss_task", "False",
                "--ref_avs_task", "True",
                "--avs_ckpt_dir", args.avs_ckpt_dir,
                "--test_name", args.split,
                "--device", args.device,
                "--multi_frames", "False",
                "--visual_branch", "True",
                "--video_frame_nums", "10",
                "--vit_ckpt_path", args.vit_ckpt_path,
                "--select_feature", "patch",
                "--image_size", "224",
                "--patch_size", "14",
                "--visual_query_token_nums", "32",
                "--audio_branch", "True",
                "--BEATs_ckpt_path", args.beats_ckpt_path,
                "--audio_query_token_nums", "32",
                "--seg_branch", "False",
                "--prompt_embed_dim", "256",
                "--mask_decoder_transformer_depth", "2",
                "--low_res_mask_size", "112",
                "--image_scale_nums", "2",
                "--token_nums_per_scale", "3",
                "--avs_query_num", "300",
                "--num_classes", "1",
                "--query_generator_num_layers", "2",
                "--output_dir", "test",
                "--max_new_tokens", str(defaults["max_new_tokens"]),
                "--save_name", save_name,
                "--budget_mode", defaults["budget_mode"],
                "--ref_avs_meta_csv", str(meta_csv),
                "--ref_avs_budget_mode", defaults["budget_mode"],
            ]
        )
        src = Path(args.ckpt_dir) / save_name / "inference_results.jsonl"
        copy_predictions(src, predictions_jsonl)

    elif args.budget_mode == "trunc_long" and not args.skip_inference:
        source_jsonl = Path(args.long_jsonl) if args.long_jsonl else budget_dir.parent / "long" / "predictions.jsonl"
        run_command(
            [
                sys.executable,
                "scripts/budget/truncate_long.py",
                "--input-jsonl", str(source_jsonl),
                "--output-jsonl", str(predictions_jsonl),
            ]
        )

    if args.budget_mode == "zero":
        predictions_jsonl = None

    if not args.skip_ground:
        cmd = [
            sys.executable,
            "scripts/budget/ground_budget.py",
            "--dataset", args.dataset,
            "--split", args.split,
            "--budget-mode", args.budget_mode,
            "--meta-csv", str(meta_csv),
            "--output-root", args.output_root,
        ]
        if predictions_jsonl is not None:
            cmd.extend(["--predictions-jsonl", str(predictions_jsonl)])
        if args.overwrite:
            cmd.append("--overwrite")
        run_command(cmd)

    print(budget_dir)


if __name__ == "__main__":
    main()
