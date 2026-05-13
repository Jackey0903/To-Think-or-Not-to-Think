import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from scripts.budget.common import (
    DEFAULT_BUDGET_ROOT,
    DEFAULT_GT_MASK,
    DEFAULT_REFAVS_MEDIA,
    build_metadata_index,
    classify_expression_type,
    default_meta_csv,
    ensure_dir,
    extract_tagged_text,
    safe_float,
    write_json,
    write_jsonl,
)


GROUNDING_DINO_CONFIG = ROOT / "grounding_dino" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = ROOT / "gdino_checkpoints" / "groundingdino_swint_ogc.pth"
SAM2_CHECKPOINT = ROOT / "checkpoints" / "sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = ROOT / "configs" / "sam2.1" / "sam2.1_hiera_l.yaml"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["refavs", "r2avs"], required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--budget-mode", choices=["zero", "long", "short", "rewrite", "trunc_long"], required=True)
    parser.add_argument("--predictions-jsonl", default=None)
    parser.add_argument("--meta-csv", default=None)
    parser.add_argument("--media-dir", default=str(DEFAULT_REFAVS_MEDIA))
    parser.add_argument("--gt-mask-dir", default=str(DEFAULT_GT_MASK))
    parser.add_argument("--output-root", default=str(DEFAULT_BUDGET_ROOT))
    parser.add_argument("--box-threshold", type=float, default=0.1)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--device", default=os.environ.get("TGS_DEVICE", "cuda"))
    parser.add_argument("--grounding-dino-config", default=str(GROUNDING_DINO_CONFIG))
    parser.add_argument("--grounding-dino-checkpoint", default=str(GROUNDING_DINO_CHECKPOINT))
    parser.add_argument("--sam2-config", default=str(SAM2_MODEL_CONFIG))
    parser.add_argument("--sam2-checkpoint", default=str(SAM2_CHECKPOINT))
    parser.add_argument("--save-vis", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_prediction_map(predictions_jsonl):
    pred_map = {}
    if not predictions_jsonl:
        return pred_map
    with open(predictions_jsonl, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pred_map[row["uid"]] = row
    return pred_map


def resolve_reference(row, prediction_row, budget_mode):
    if budget_mode == "zero":
        return {
            "ref_text": row["exp"],
            "f_object": "",
            "s_object": "",
            "predict": "",
            "caption_suffix": "",
        }
    predict_text = prediction_row.get("predict", "") if prediction_row else ""
    f_object = extract_tagged_text(predict_text, "f_object")
    s_object = extract_tagged_text(predict_text, "s_object")
    ref_text = s_object or f_object or row["exp"]
    return {
        "ref_text": ref_text,
        "f_object": f_object,
        "s_object": s_object,
        "predict": predict_text,
        "caption_suffix": ".",
    }


def main():
    args = parse_args()
    import cv2
    import numpy as np
    import supervision as sv
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from tqdm import tqdm
    from torchvision.ops import box_convert

    from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from utils.avss_utils import Eval_Fmeasure, mask_iou, metric_s_for_null_batch

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    meta_csv = Path(args.meta_csv) if args.meta_csv else default_meta_csv(args.dataset)
    prediction_map = load_prediction_map(args.predictions_jsonl)
    metadata = build_metadata_index(meta_csv, split=args.split)
    if args.budget_mode != "zero" and not prediction_map:
        raise ValueError(f"No predictions loaded for budget mode {args.budget_mode}.")

    output_dir = ensure_dir(Path(args.output_root) / args.dataset / args.split / args.budget_mode)
    metrics_json = output_dir / "metrics.json"
    sample_metrics_jsonl = output_dir / "sample_metrics.jsonl"
    vis_root = output_dir / "visualizations"

    existing = {}
    if metrics_json.exists() and not args.overwrite:
        loaded = json.loads(metrics_json.read_text())
        for row in loaded.get("samples", []):
            existing[row["uid"]] = row

    gdino_model = load_model(
        model_config_path=args.grounding_dino_config,
        model_checkpoint_path=args.grounding_dino_checkpoint,
        device=args.device,
    )
    sam2_model = build_sam2(args.sam2_config, args.sam2_checkpoint, device=args.device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    samples = list(existing.values())
    media_dir = Path(args.media_dir)
    gt_mask_dir = Path(args.gt_mask_dir)
    missing_predictions = []

    for uid, row in tqdm(metadata.items(), desc=f"ground {args.dataset} {args.split} {args.budget_mode}"):
        if uid in existing:
            continue
        prediction_row = prediction_map.get(uid)
        if args.budget_mode != "zero" and prediction_row is None:
            missing_predictions.append(uid)
            continue
        resolved = resolve_reference(row=row, prediction_row=prediction_row, budget_mode=args.budget_mode)
        ref_text = resolved["ref_text"]

        vid = uid.rsplit("_", 2)[0]
        frame_dir = media_dir / vid / "frames"
        fid = int(row["fid"])
        pred_masks = []
        pred_logits_for_fmeasure = []
        gt_masks = []

        if args.save_vis:
            ensure_dir(vis_root / uid)

        for img_id in range(10):
            img_path = frame_dir / f"{img_id}.jpg"
            gt_mask_path = gt_mask_dir / vid / f"fid_{fid}" / f"{img_id:05d}.png"
            if not img_path.exists() or not gt_mask_path.exists():
                continue

            image_source, image = load_image(str(img_path))
            image_source = cv2.resize(image_source, (224, 224))
            image_source = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)
            h, w, _ = image_source.shape

            boxes, confidences, labels = predict(
                model=gdino_model,
                image=image,
                caption=ref_text.lower().strip() + resolved["caption_suffix"],
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                device=args.device,
            )

            boxes = boxes * torch.Tensor([w, h, w, h])
            boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

            current_pred_mask = torch.zeros((1, 224, 224), dtype=torch.uint8)
            current_pred_logit = torch.zeros((1, 224, 224), dtype=torch.float32)
            gt = np.array(Image.open(gt_mask_path).convert("L").resize((224, 224))).astype(bool)

            if boxes.shape[0] == 0:
                pred_masks.append(current_pred_mask)
                pred_logits_for_fmeasure.append(current_pred_logit)
                gt_masks.append(torch.from_numpy(gt).unsqueeze(0).to(torch.uint8))
                continue

            boxes = boxes[:1]
            sam2_predictor.set_image(image_source)
            masks, scores, logits = sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=boxes,
                multimask_output=False,
            )

            if masks is None or len(masks) == 0:
                pred_masks.append(current_pred_mask)
                pred_logits_for_fmeasure.append(current_pred_logit)
                gt_masks.append(torch.from_numpy(gt).unsqueeze(0).to(torch.uint8))
                continue

            if isinstance(masks, torch.Tensor):
                masks = masks.cpu().numpy()
            if masks.ndim == 4:
                masks = np.squeeze(masks, axis=1)
            mask = masks[0].astype(np.uint8)
            pred_masks.append(torch.from_numpy(mask).unsqueeze(0))

            processed_logits_tensor = current_pred_logit
            if logits is not None:
                if isinstance(logits, torch.Tensor):
                    temp_logits = logits.cpu().squeeze(1)
                else:
                    temp_logits = torch.from_numpy(logits).float().squeeze(1)
                if temp_logits.ndim == 2:
                    temp_logits = temp_logits.unsqueeze(0).unsqueeze(0)
                elif temp_logits.ndim == 3 and temp_logits.shape[0] == 1:
                    temp_logits = temp_logits.unsqueeze(0)
                processed_logits_tensor = F.interpolate(
                    temp_logits,
                    size=(224, 224),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).to(torch.float32)
                processed_logits_tensor = torch.sigmoid(processed_logits_tensor)

            pred_logits_for_fmeasure.append(processed_logits_tensor.to(torch.float32))
            gt_masks.append(torch.from_numpy(gt).unsqueeze(0).to(torch.uint8))

            if args.save_vis:
                detections = sv.Detections(
                    xyxy=boxes,
                    mask=np.array([mask], dtype=bool),
                    class_id=np.array([0]),
                )
                annotated = sv.BoxAnnotator().annotate(image_source.copy(), detections)
                annotated = sv.MaskAnnotator().annotate(annotated, detections)
                annotated = sv.LabelAnnotator().annotate(annotated, detections, labels=[ref_text])
                cv2.imwrite(str(vis_root / uid / f"{img_id}_vis.jpg"), annotated)

        if pred_masks:
            pred_masks_tensor = torch.cat(pred_masks, dim=0)
            gt_masks_tensor = torch.cat(gt_masks, dim=0)
            pred_logits_tensor = torch.cat(pred_logits_for_fmeasure, dim=0)
            if args.split == "test_n":
                mean_iou = 0.0 if ref_text == "null" else metric_s_for_null_batch(pred_masks_tensor)
                fscore = -999
            else:
                mean_iou = mask_iou(pred_masks_tensor, gt_masks_tensor)
                fscore = Eval_Fmeasure(pred=pred_logits_tensor.cpu(), gt=gt_masks_tensor.float().cpu())
        else:
            mean_iou = 0.0
            fscore = 0.0

        j = safe_float(mean_iou.item() if isinstance(mean_iou, torch.Tensor) else mean_iou)
        f = safe_float(fscore.item() if isinstance(fscore, torch.Tensor) else fscore)
        sample = {
            "uid": uid,
            "dataset": args.dataset,
            "split": args.split,
            "budget_mode": args.budget_mode,
            "vid": vid,
            "ref": row["exp"],
            "resolved_text": ref_text,
            "f_object": resolved["f_object"],
            "s_object": resolved["s_object"],
            "predict": resolved["predict"],
            "expression_type": classify_expression_type(row["exp"]),
            "j": j,
            "f": f,
            "jf": (j + f) / 2.0,
        }
        samples.append(sample)
        write_jsonl(sample_metrics_jsonl, samples)

    average_j = sum(row["j"] for row in samples) / len(samples) if samples else 0.0
    average_f = sum(row["f"] for row in samples) / len(samples) if samples else 0.0
    payload = {
        "dataset": args.dataset,
        "split": args.split,
        "budget_mode": args.budget_mode,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "count": len(samples),
        "overall_average_iou": round(average_j, 4),
        "overall_average_fscore": round(average_f, 4),
        "overall_average_jf": round((average_j + average_f) / 2.0, 4),
        "missing_prediction_count": len(missing_predictions),
        "missing_prediction_uids": missing_predictions[:50],
        "samples": sorted(samples, key=lambda row: row["uid"]),
    }
    write_json(metrics_json, payload)
    print(metrics_json)


if __name__ == "__main__":
    main()
