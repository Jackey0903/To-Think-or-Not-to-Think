# Ref-AVS Budget Calibration Scripts

Current experiment tracking document:

- `refavs_budget_experiment_log.md`

## What is implemented

- `infer_budget.py`: unified driver for `zero`, `long`, `short`, `rewrite`, `trunc_long`
- `ground_budget.py`: unified grounding + segmentation evaluation
- `truncate_long.py`: post-process long predictions into `trunc_long`
- `collect_existing_results.py`: normalize legacy reproduced results into one sample-level table
- `label_budget.py`: CBL labels, threshold sweep, split/expression summaries
- `infer_short.py`, `infer_rewrite.py`, `infer_zero.py`, `infer_all_budgets.py`: thin wrappers

## Example Commands

Use existing reproduced `test_s` results:

```bash
cd To-Think-or-Not-to-Think
python scripts/budget/collect_existing_results.py
python scripts/budget/label_budget.py \
  --input budget_results/existing_budget_records.jsonl \
  --output-root budget_results/labels
```

Run a new `short` condition:

```bash
python scripts/budget/infer_budget.py \
  --dataset refavs \
  --split test_u \
  --budget-mode short \
  --device cuda:0
```

Run `trunc_long` from an existing long prediction file:

```bash
python scripts/budget/infer_budget.py \
  --dataset refavs \
  --split test_s \
  --budget-mode trunc_long \
  --long-jsonl budget_results/refavs/test_s/long/predictions.jsonl
```

Ground an already generated prediction file:

```bash
python scripts/budget/infer_budget.py \
  --dataset r2avs \
  --split test_s \
  --budget-mode long \
  --predictions-jsonl /path/to/predictions.jsonl \
  --skip-inference
```
