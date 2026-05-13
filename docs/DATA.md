# Data

Expected local layout:

```text
REFAVS/
  media/
  gt_mask/
R2AVSBench/
  RefAVSBench_metadata.csv
  R2AVSBench_metadata.csv
  RefAVSBenchRef_R2AVSBenchVideo.csv
  RefThinker_instruction_tuning_set.json
```

`R2AVSBench/*.csv` metadata files are kept in this repository because they are small and needed by the scripts. The full Ref-AVSBench videos, masks, and Ref-Thinker instruction-tuning JSON should be downloaded separately according to the dataset release instructions.

The default paths can be overridden by script arguments such as:

```bash
python scripts/budget/infer_budget.py \
  --dataset refavs \
  --split test_u \
  --budget-mode long \
  --meta-csv /path/to/metadata.csv
```

For Ground-Segment evaluation, `scripts/budget/ground_budget.py` also accepts:

```bash
--media-dir /path/to/REFAVS/media
--gt-mask-dir /path/to/REFAVS/gt_mask
```

