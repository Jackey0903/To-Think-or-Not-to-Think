import argparse
import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--sort-by-uid", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(input_csv, "r") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if row["split"] == args.split]

    if args.sort_by_uid:
        rows = sorted(rows, key=lambda row: row["uid"])

    rows = rows[: args.limit]

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["vid", "uid", "split", "fid", "exp"])
        writer.writeheader()
        writer.writerows(rows)

    print(output_csv)


if __name__ == "__main__":
    main()
