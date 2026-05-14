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

`R2AVSBench/*.csv` metadata files are kept in this repository because they are small and needed by the scripts.

Download the full Ref-AVSBench videos and masks from:

- https://github.com/GeWu-Lab/Ref-AVS

Download the Ref-Thinker instruction-tuning JSON from:

- https://huggingface.co/datasets/Jinxing1/TGSAgent-FT-data/tree/main

Place the JSON at:

```text
R2AVSBench/RefThinker_instruction_tuning_set.json
```

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
