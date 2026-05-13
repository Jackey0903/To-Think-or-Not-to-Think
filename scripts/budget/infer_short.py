import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from scripts.budget.infer_budget import main


if __name__ == "__main__":
    sys.argv.extend(["--budget-mode", "short"]) if "--budget-mode" not in sys.argv else None
    main()
