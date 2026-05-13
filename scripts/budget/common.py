import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUDGET_ROOT = ROOT / "budget_results"
DEFAULT_REFAVS_META = ROOT / "R2AVSBench" / "RefAVSBench_metadata.csv"
DEFAULT_R2AVS_META = ROOT / "R2AVSBench" / "R2AVSBench_metadata.csv"
DEFAULT_REFAVS_MEDIA = ROOT / "REFAVS" / "media"
DEFAULT_REFAVS_MEDIA_CROSS = ROOT / "REFAVS" / "media_cross"
DEFAULT_GT_MASK = ROOT / "REFAVS" / "gt_mask"


CHEAP_TO_EXPENSIVE = ["zero", "rewrite", "trunc_long", "short", "long"]

SPATIAL_KEYWORDS = (
    "left", "right", "behind", "front", "in front of", "next to", "beside",
    "between", "near", "closest", "furthest", "under", "below", "above",
    "on top of", "around",
)
TEMPORAL_KEYWORDS = (
    "before", "after", "while", "when", "during", "first", "last", "then",
    "start", "begin", "end", "ending", "at all times", "continues", "keep",
    "keeps", "still",
)
AUDIO_KEYWORDS = (
    "sound", "sounding", "hear", "heard", "audio", "voice", "sing", "singing",
    "music", "noisy", "ring", "playing", "makes sound", "speaking",
)
ABSTRACT_KEYWORDS = (
    "thing", "object", "one", "something", "someone", "the one", "the thing",
    "that is", "which is", "keeps", "appears", "seems",
)


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def read_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_meta_csv(dataset_name):
    if dataset_name == "r2avs":
        return DEFAULT_R2AVS_META
    return DEFAULT_REFAVS_META


def load_metadata_rows(meta_csv_path, split=None):
    meta_csv_path = Path(meta_csv_path)
    rows = []
    with open(meta_csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if split is not None and row["split"] != split:
                continue
            rows.append(row)
    return rows


def build_metadata_index(meta_csv_path, split=None):
    rows = load_metadata_rows(meta_csv_path, split=split)
    return {row["uid"]: row for row in rows}


def extract_tagged_text(text, tag):
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text or "", re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def normalize_metric_result(uid, payload, dataset, split, budget_mode, ref_text=None):
    record = {
        "uid": uid,
        "dataset": dataset,
        "split": split,
        "budget_mode": budget_mode,
        "vid": payload.get("vid", uid.rsplit("_", 2)[0]),
        "ref": ref_text if ref_text is not None else payload.get("ref", ""),
        "f_object": payload.get("f_object", ""),
        "s_object": payload.get("s_object", ""),
        "j": safe_float(payload.get("s_mean_iou", payload.get("f_mean_iou", 0.0))),
        "f": safe_float(payload.get("s_fscore", payload.get("f_fscore", 0.0))),
    }
    record["jf"] = (record["j"] + record["f"]) / 2.0
    record["expression_type"] = classify_expression_type(record["ref"])
    return record


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def classify_expression_type(ref_text):
    text = (ref_text or "").strip().lower()
    if not text:
        return "simple object"
    if contains_any(text, AUDIO_KEYWORDS):
        return "audio-dominant"
    if contains_any(text, TEMPORAL_KEYWORDS):
        return "temporal"
    if contains_any(text, SPATIAL_KEYWORDS):
        return "spatial relation"
    if contains_any(text, ABSTRACT_KEYWORDS):
        return "abstract/indirect"
    return "simple object"


def contains_any(text, patterns):
    return any(pattern in text for pattern in patterns)


def pivot_budget_records(records):
    grouped = {}
    for row in records:
        key = (row["dataset"], row["split"], row["uid"])
        bucket = grouped.setdefault(
            key,
            {
                "dataset": row["dataset"],
                "split": row["split"],
                "uid": row["uid"],
                "vid": row.get("vid", row["uid"].rsplit("_", 2)[0]),
                "ref": row.get("ref", ""),
                "expression_type": row.get("expression_type", classify_expression_type(row.get("ref", ""))),
            },
        )
        budget = row["budget_mode"]
        bucket[f"j_{budget}"] = safe_float(row.get("j"))
        bucket[f"f_{budget}"] = safe_float(row.get("f"))
        bucket[f"jf_{budget}"] = safe_float(row.get("jf"))
    return list(grouped.values())


def summarize_by_keys(records, key_fields, metric_fields):
    grouped = {}
    for row in records:
        key = tuple(row[field] for field in key_fields)
        bucket = grouped.setdefault(
            key,
            {field: row[field] for field in key_fields} | {"count": 0},
        )
        bucket["count"] += 1
        for metric in metric_fields:
            bucket.setdefault(f"sum_{metric}", 0.0)
            bucket[f"sum_{metric}"] += safe_float(row.get(metric))

    summary = []
    for bucket in grouped.values():
        item = {field: bucket[field] for field in key_fields}
        item["count"] = bucket["count"]
        for metric in metric_fields:
            denom = bucket["count"] or 1
            item[metric] = bucket[f"sum_{metric}"] / denom
        summary.append(item)
    return summary


def choose_budget_label(score_map, threshold):
    best_score = max(score_map.values())
    candidates = {
        mode: score for mode, score in score_map.items()
        if score >= best_score - threshold
    }
    for mode in CHEAP_TO_EXPENSIVE:
        if mode in candidates:
            return mode
    return max(score_map, key=score_map.get)
