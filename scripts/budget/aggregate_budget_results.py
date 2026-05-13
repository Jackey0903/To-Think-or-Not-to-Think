import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from scripts.budget.common import pivot_budget_records, summarize_by_keys, write_json, write_jsonl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget-root", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    budget_root = Path(args.budget_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    records = []
    for metrics_path in sorted(budget_root.glob("*/*/*/metrics.json")):
        dataset = metrics_path.parents[2].name
        split = metrics_path.parents[1].name
        budget_mode = metrics_path.parents[0].name
        payload = __import__("json").loads(metrics_path.read_text())
        for sample in payload.get("samples", []):
            row = dict(sample)
            row["dataset"] = dataset
            row["split"] = split
            row["budget_mode"] = budget_mode
            row["source_path"] = str(metrics_path)
            row["valid_for_paper"] = True
            records.append(row)

    wide = pivot_budget_records(records)
    split_summary = summarize_by_keys(wide, ["dataset", "split"], [key for key in sorted({k for row in wide for k in row if k.startswith("jf_")})])

    write_jsonl(output_root / "budget_records.jsonl", records)
    write_json(output_root / "budget_records.json", records)
    write_json(output_root / "budget_wide.json", wide)
    write_json(output_root / "budget_split_summary.json", split_summary)
    print(output_root / "budget_wide.json")


if __name__ == "__main__":
    main()
