import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from scripts.budget.common import extract_tagged_text, read_jsonl, write_json, write_jsonl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = read_jsonl(args.input_jsonl)
    output_rows = []
    summary = {
        "input_jsonl": args.input_jsonl,
        "output_jsonl": args.output_jsonl,
        "count": 0,
        "missing_s_object": 0,
    }

    for row in rows:
        s_object = extract_tagged_text(row.get("predict", ""), "s_object")
        f_object = extract_tagged_text(row.get("predict", ""), "f_object")
        concise = s_object or f_object
        if not s_object:
            summary["missing_s_object"] += 1
        output_rows.append(
            {
                "uid": row["uid"],
                "ref": row.get("ref", ""),
                "predict": f"<answer>\n<s_object> {concise} </s_object>\n</answer>",
                "source_predict": row.get("predict", ""),
                "source_budget_mode": "long",
            }
        )
        summary["count"] += 1

    output_path = Path(args.output_jsonl)
    write_jsonl(output_path, output_rows)
    write_json(output_path.with_suffix(".summary.json"), summary)
    print(output_path)


if __name__ == "__main__":
    main()
