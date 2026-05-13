import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from scripts.budget.common import pivot_budget_records, summarize_by_keys, write_json, write_jsonl
from scripts.budget.collect_existing_results import existing_sources


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", default=os.environ.get(
        "TGS_CKPT_DIR",
        str(ROOT / "results_real" / "epochs6_lr1e-4_bs4_gradacc8_lora_r8alpha16dropout0.05" / "checkpoint-551"),
    ))
    parser.add_argument("--budget-root", default=str(ROOT / "budget_results"))
    parser.add_argument("--output-root", default=str(ROOT / "budget_results" / "analysis" / "current"))
    return parser.parse_args()


def load_json(path):
    import json
    with open(path, "r") as f:
        return json.load(f)


def collect_existing(ckpt_dir):
    from scripts.budget.collect_existing_results import build_metadata_index, default_meta_csv, normalize_metric_result, read_json

    records = []
    for source in existing_sources(ckpt_dir):
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
            records.append(row)
    return records


def collect_budget_outputs(budget_root):
    records = []
    for metrics_path in sorted(Path(budget_root).glob("*/*/*/metrics.json")):
        payload = load_json(metrics_path)
        for sample in payload.get("samples", []):
            row = dict(sample)
            row["source_path"] = str(metrics_path)
            row["valid_for_paper"] = True
            records.append(row)
    return records


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    existing_records = collect_existing(args.ckpt_dir)
    new_records = collect_budget_outputs(args.budget_root)

    merged = {}
    for row in existing_records:
        key = (row["dataset"], row["split"], row["budget_mode"], row["uid"])
        merged[key] = row
    for row in new_records:
        key = (row["dataset"], row["split"], row["budget_mode"], row["uid"])
        merged[key] = row

    merged_records = sorted(merged.values(), key=lambda row: (row["dataset"], row["split"], row["uid"], row["budget_mode"]))
    wide = pivot_budget_records(merged_records)
    metric_fields = sorted({key for row in wide for key in row if key.startswith("jf_")})
    split_summary = summarize_by_keys(wide, ["dataset", "split"], metric_fields)

    write_jsonl(output_root / "all_budget_records.jsonl", merged_records)
    write_json(output_root / "all_budget_records.json", merged_records)
    write_json(output_root / "all_budget_wide.json", wide)
    write_json(output_root / "all_budget_split_summary.json", split_summary)

    import subprocess
    subprocess.run(
        [
            sys.executable,
            "scripts/budget/label_budget.py",
            "--input", str(output_root / "all_budget_records.jsonl"),
            "--output-root", str(output_root / "labels"),
        ],
        cwd=str(ROOT),
        check=True,
    )

    print(output_root / "all_budget_wide.json")


if __name__ == "__main__":
    main()
