import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["refavs", "r2avs"], required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--budgets", default="zero,short,long,rewrite,trunc_long")
    return parser.parse_known_args()


def main():
    args, passthrough = parse_args()
    budgets = [item.strip() for item in args.budgets.split(",") if item.strip()]
    for budget in budgets:
        cmd = [
            sys.executable,
            "scripts/budget/infer_budget.py",
            "--dataset", args.dataset,
            "--split", args.split,
            "--budget-mode", budget,
            *passthrough,
        ]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
