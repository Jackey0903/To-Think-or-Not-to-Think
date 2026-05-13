import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from scripts.budget.common import (
    DEFAULT_BUDGET_ROOT,
    ROOT,
    build_metadata_index,
    default_meta_csv,
    ensure_dir,
    normalize_metric_result,
    pivot_budget_records,
    read_json,
    summarize_by_keys,
    write_json,
    write_jsonl,
)


DEFAULT_CKPT_DIR = Path(os.environ.get(
    "TGS_CKPT_DIR",
    ROOT / "results_real" / "epochs6_lr1e-4_bs4_gradacc8_lora_r8alpha16dropout0.05" / "checkpoint-551",
))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", default=str(DEFAULT_CKPT_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_BUDGET_ROOT))
    return parser.parse_args()


def existing_sources(ckpt_dir):
    ckpt_dir = Path(ckpt_dir)
    return [
        {
            "dataset": "refavs",
            "split": "test_s",
            "budget_mode": "long",
            "path": ckpt_dir / "inference_cot_ref_avs_test_s_bs" / "s_object_eval_results_v2_boxthre0.1_textthre0.25.json",
            "valid_for_paper": True,
        },
        {
            "dataset": "refavs",
            "split": "test_s",
            "budget_mode": "zero",
            "path": ckpt_dir / "inference_cot_ref_avs_test_s_bs" / "direct_ref_eval_results_v2_boxthre0.1_textthre0.25_reproduce.json",
            "valid_for_paper": True,
        },
        {
            "dataset": "r2avs",
            "split": "test_s",
            "budget_mode": "long",
            "path": ckpt_dir / "R2AVSBenchinference_cot_ref_avs_test_s_bs" / "s_object_eval_results_v2_boxthre0.1_textthre0.25_reproduce.json",
            "valid_for_paper": False,
        },
        {
            "dataset": "r2avs",
            "split": "test_s",
            "budget_mode": "zero",
            "path": ckpt_dir / "inference_cot_ref_avs_test_s_bs" / "R2AVSBench_butOriref_eval_results_v2_boxthre0.1_textthre0.25v2.json",
            "valid_for_paper": True,
        },
    ]


def main():
    args = parse_args()
    output_root = ensure_dir(args.output_root)
    long_records = []

    for source in existing_sources(args.ckpt_dir):
        if not source["path"].exists():
            continue
        metrics = read_json(source["path"])
        meta_index = build_metadata_index(default_meta_csv(source["dataset"]), split=source["split"])
        for uid, payload in metrics.items():
            if uid.startswith("overall_average_"):
                continue
            meta_row = meta_index.get(uid, {})
            ref_text = meta_row.get("exp", payload.get("ref", ""))
            row = normalize_metric_result(
                uid=uid,
                payload=payload,
                dataset=source["dataset"],
                split=source["split"],
                budget_mode=source["budget_mode"],
                ref_text=ref_text,
            )
            row["source_path"] = str(source["path"])
            row["valid_for_paper"] = source["valid_for_paper"]
            long_records.append(row)

    long_records = sorted(long_records, key=lambda row: (row["dataset"], row["split"], row["uid"], row["budget_mode"]))
    wide_records = pivot_budget_records(long_records)
    split_summary = summarize_by_keys(wide_records, ["dataset", "split"], ["jf_zero", "jf_long"])

    write_jsonl(output_root / "existing_budget_records.jsonl", long_records)
    write_json(output_root / "existing_budget_records.json", long_records)
    write_json(output_root / "existing_budget_wide.json", wide_records)
    write_json(output_root / "existing_budget_split_summary.json", split_summary)
    print(output_root / "existing_budget_wide.json")


if __name__ == "__main__":
    main()
