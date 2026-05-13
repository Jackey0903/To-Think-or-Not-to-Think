import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from scripts.budget.common import (
    CHEAP_TO_EXPENSIVE,
    choose_budget_label,
    pivot_budget_records,
    read_json,
    read_jsonl,
    safe_float,
    summarize_by_keys,
    write_json,
    write_jsonl,
)


PRIMARY_BUDGETS = ["zero", "short", "long"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Long-format json/jsonl with one row per uid-budget result.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--thresholds", default="0.00,0.01,0.02,0.03")
    return parser.parse_args()


def load_records(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    return read_json(path)


def annotate_record(row, threshold):
    score_map = {}
    for budget in PRIMARY_BUDGETS:
        score_map[budget] = safe_float(row.get(f"jf_{budget}"), default=-1e9)

    best_mode = max(score_map, key=score_map.get)
    assigned_mode = choose_budget_label(score_map, threshold=threshold)
    oracle_score = score_map[best_mode]
    assigned_score = score_map[assigned_mode]
    result = dict(row)
    result["threshold"] = threshold
    result["oracle_budget"] = best_mode
    result["budget_label"] = f"need-{assigned_mode}"
    result["assigned_budget"] = assigned_mode
    result["oracle_jf"] = oracle_score
    result["assigned_jf"] = assigned_score
    result["reasoning_gain"] = safe_float(row.get("jf_long")) - safe_float(row.get("jf_zero"))
    result["overthink_gap"] = safe_float(row.get("jf_long")) - safe_float(row.get("jf_short"))
    result["valid_flag"] = int(all(row.get(f"jf_{budget}") is not None for budget in PRIMARY_BUDGETS))
    for budget in PRIMARY_BUDGETS:
        result[f"is_{budget}"] = int(assigned_mode == budget)
    return result


def summarize_labels(records, key_fields):
    grouped = {}
    for row in records:
        key = tuple(row[field] for field in key_fields)
        bucket = grouped.setdefault(
            key,
            {field: row[field] for field in key_fields} | {
                "count": 0,
                "need_zero": 0,
                "need_short": 0,
                "need_long": 0,
                "oracle_jf_sum": 0.0,
                "assigned_jf_sum": 0.0,
                "reasoning_gain_sum": 0.0,
                "overthink_gap_sum": 0.0,
            },
        )
        bucket["count"] += 1
        bucket["need_zero"] += row["is_zero"]
        bucket["need_short"] += row["is_short"]
        bucket["need_long"] += row["is_long"]
        bucket["oracle_jf_sum"] += row["oracle_jf"]
        bucket["assigned_jf_sum"] += row["assigned_jf"]
        bucket["reasoning_gain_sum"] += row["reasoning_gain"]
        bucket["overthink_gap_sum"] += row["overthink_gap"]

    summary = []
    for bucket in grouped.values():
        count = bucket["count"] or 1
        item = {field: bucket[field] for field in key_fields}
        item["count"] = bucket["count"]
        item["need_zero_rate"] = bucket["need_zero"] / count
        item["need_short_rate"] = bucket["need_short"] / count
        item["need_long_rate"] = bucket["need_long"] / count
        item["oracle_jf"] = bucket["oracle_jf_sum"] / count
        item["assigned_jf"] = bucket["assigned_jf_sum"] / count
        item["reasoning_gain"] = bucket["reasoning_gain_sum"] / count
        item["overthink_gap"] = bucket["overthink_gap_sum"] / count
        summary.append(item)
    return summary


def main():
    args = parse_args()
    thresholds = [float(item) for item in args.thresholds.split(",") if item]
    records = load_records(args.input)
    if records and isinstance(records[0], dict) and "budget_mode" in records[0]:
        wide_records = pivot_budget_records(records)
    else:
        wide_records = records

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for threshold in thresholds:
        labeled = [annotate_record(row, threshold=threshold) for row in wide_records]
        label_dir = output_root / f"threshold_{threshold:.2f}"
        label_dir.mkdir(parents=True, exist_ok=True)
        split_summary = summarize_labels(labeled, ["dataset", "split", "threshold"])
        expr_summary = summarize_labels(labeled, ["dataset", "split", "expression_type", "threshold"])
        table1 = summarize_by_keys(labeled, ["dataset", "split", "threshold"], ["jf_zero", "jf_short", "jf_long"])

        write_jsonl(label_dir / "sample_labels.jsonl", labeled)
        write_json(label_dir / "sample_labels.json", labeled)
        write_json(label_dir / "split_summary.json", split_summary)
        write_json(label_dir / "expression_type_summary.json", expr_summary)
        write_json(label_dir / "table1_data.json", table1)

    print(output_root)


if __name__ == "__main__":
    main()
