import sys
import json
import argparse
from collections import Counter
from pathlib import Path
import torch

_VISION_ROOT = Path(__file__).resolve().parent.parent
if str(_VISION_ROOT) not in sys.path:
    sys.path.insert(0, str(_VISION_ROOT))

from perception.vehicle_taxonomy import CANONICAL_CLASS_PCU, CANONICAL_ID_TO_NAME, pcu_for_class_id


def detect_device():
    """Detects best available hardware accelerator (CUDA, Apple Silicon MPS, or CPU)."""
    if torch.cuda.is_available():
        return "0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def compute_pcu_evaluation_metrics(gt_by_image, pred_by_image, is_multiclass: bool = True):
    """
    Computes traffic-control PCU-weighted evaluation metrics comparing ground-truth
    annotations against predicted detections across images.

    Args:
        gt_by_image: dict[image_id, list[int]] of canonical ground-truth class IDs.
        pred_by_image: dict[image_id, list[int]] of canonical predicted class IDs.
    """
    all_images = sorted(set(gt_by_image.keys()) | set(pred_by_image.keys()))
    total_gt_pcu = 0.0
    total_pred_pcu = 0.0
    abs_err_sum = 0.0

    gt_class_counts = Counter()
    pred_class_counts = Counter()
    gt_class_pcu = Counter()
    pred_class_pcu = Counter()

    for img_id in all_images:
        gt_classes = gt_by_image.get(img_id, [])
        pred_classes = pred_by_image.get(img_id, [])

        img_gt_pcu = 0.0
        for cid in gt_classes:
            pcu = pcu_for_class_id(cid, is_multiclass=is_multiclass)
            if pcu is None:
                continue
            cname = CANONICAL_ID_TO_NAME.get(int(cid), "vehicle") if is_multiclass else "vehicle"
            gt_class_counts[cname] += 1
            gt_class_pcu[cname] += pcu
            img_gt_pcu += pcu

        img_pred_pcu = 0.0
        for cid in pred_classes:
            pcu = pcu_for_class_id(cid, is_multiclass=is_multiclass)
            if pcu is None:
                continue
            cname = CANONICAL_ID_TO_NAME.get(int(cid), "vehicle") if is_multiclass else "vehicle"
            pred_class_counts[cname] += 1
            pred_class_pcu[cname] += pcu
            img_pred_pcu += pcu

        total_gt_pcu += img_gt_pcu
        total_pred_pcu += img_pred_pcu
        abs_err_sum += abs(img_pred_pcu - img_gt_pcu)

    net_pcu_error = total_pred_pcu - total_gt_pcu
    mae_pcu = abs_err_sum / len(all_images) if all_images else 0.0
    relative_pcu_error = abs(net_pcu_error) / total_gt_pcu if total_gt_pcu > 0.0 else 0.0
    mean_image_rel_error = (abs_err_sum / total_gt_pcu) if total_gt_pcu > 0.0 else 0.0

    all_class_names = list(CANONICAL_CLASS_PCU.keys()) if is_multiclass else ["vehicle"]
    per_class_pcu_error = {}
    for cname in all_class_names:
        g_pcu = round(gt_class_pcu.get(cname, 0.0), 4)
        p_pcu = round(pred_class_pcu.get(cname, 0.0), 4)
        diff = round(p_pcu - g_pcu, 4)
        per_class_pcu_error[cname] = {
            "gt_count": gt_class_counts.get(cname, 0),
            "pred_count": pred_class_counts.get(cname, 0),
            "gt_pcu": g_pcu,
            "pred_pcu": p_pcu,
            "signed_pcu_error": diff,
            "abs_pcu_error": round(abs(diff), 4),
        }

    return {
        "num_images": len(all_images),
        "total_gt_pcu": round(total_gt_pcu, 4),
        "total_pred_pcu": round(total_pred_pcu, 4),
        "absolute_pcu_error_sum": round(abs_err_sum, 4),
        "mean_absolute_pcu_error_per_image": round(mae_pcu, 4),
        "net_pcu_error": round(net_pcu_error, 4),
        "relative_net_pcu_error": round(relative_pcu_error, 4),
        "relative_total_abs_pcu_error": round(mean_image_rel_error, 4),
        "per_class_pcu_error": per_class_pcu_error,
    }


def parse_args():
    base_dir = _VISION_ROOT
    default_weights = base_dir / "runs" / "detect" / "model_p2_960_safe" / "weights" / "best.pt"
    if not default_weights.exists():
        default_weights = "yolov8n.pt"

    default_data = base_dir / "configs" / "data_multiclass.yaml"
    if not default_data.exists():
        default_data = base_dir / "configs" / "data_combined.yaml"

    parser = argparse.ArgumentParser(description="Evaluate ATCS Vehicle Detection Model (mAP + PCU metrics).")
    parser.add_argument("--weights", type=str, default=str(default_weights))
    parser.add_argument("--data", type=str, default=str(default_data))
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test", "train"])
    parser.add_argument("--rect", action="store_true", default=True)
    parser.add_argument("--no-rect", dest="rect", action="store_false")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-json", action="store_true", default=True)
    parser.add_argument("--plots", action="store_true", default=True)
    parser.add_argument("--report-out", type=str, default=None, help="Optional JSON path for evaluation report.")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: Ultralytics is not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    weights_path = Path(args.weights).resolve()
    if not weights_path.exists():
        print(f"Error: Model weights not found at: {weights_path}")
        sys.exit(1)

    data_yaml = Path(args.data).resolve()
    if not data_yaml.exists():
        print(f"Error: Dataset YAML not found at: {data_yaml}")
        sys.exit(1)

    target_device = args.device if args.device is not None else detect_device()
    eval_device = "cpu" if target_device == "mps" and args.imgsz >= 960 else target_device

    model = YOLO(str(weights_path))
    try:
        metrics = model.val(
            data=str(data_yaml),
            imgsz=args.imgsz,
            split=args.split,
            rect=args.rect,
            device=eval_device,
            save_json=args.save_json,
            plots=args.plots,
        )
    except Exception as e:
        print(f"[Warning] Validation on {eval_device} failed ({e}). Retrying on CPU...")
        metrics = model.val(
            data=str(data_yaml),
            imgsz=args.imgsz,
            split=args.split,
            rect=args.rect,
            device="cpu",
            save_json=args.save_json,
            plots=args.plots,
        )

    per_class = {}
    names = getattr(metrics, "names", CANONICAL_ID_TO_NAME)
    if hasattr(metrics.box, "ap_class_index") and metrics.box.ap_class_index is not None:
        for idx, cid in enumerate(metrics.box.ap_class_index):
            cname = names.get(int(cid), str(cid)) if isinstance(names, dict) else str(cid)
            p_val, r_val, ap50_val, ap_val = metrics.box.class_result(idx)
            per_class[cname] = {
                "precision": round(float(p_val), 4),
                "recall": round(float(r_val), 4),
                "map50": round(float(ap50_val), 4),
                "map50_95": round(float(ap_val), 4),
            }

    cm_matrix = None
    if hasattr(metrics, "confusion_matrix") and metrics.confusion_matrix is not None:
        cm_matrix = metrics.confusion_matrix.matrix.tolist()

    report = {
        "weights": str(weights_path),
        "data": str(data_yaml),
        "split": args.split,
        "overall": {
            "precision": round(float(metrics.box.mp), 4),
            "recall": round(float(metrics.box.mr), 4),
            "map50": round(float(metrics.box.map50), 4),
            "map50_95": round(float(metrics.box.map), 4),
        },
        "per_class": per_class,
        "confusion_matrix": cm_matrix,
    }

    print("VALIDATION METRICS SUMMARY")
    print(f" Precision (P):    {metrics.box.mp:.4f}")
    print(f" Recall (R):       {metrics.box.mr:.4f}")
    print(f" mAP@50:           {metrics.box.map50:.4f}")
    print(f" mAP@50-95:        {metrics.box.map:.4f}")
    if per_class:
        print(" Per-class metrics:", json.dumps(per_class, indent=2))

    if args.report_out:
        Path(args.report_out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
