import sys
import argparse
from pathlib import Path
import torch


# Detects best available hardware accelerator (CUDA, Apple Silicon MPS, or CPU)
def detect_device():
    if torch.cuda.is_available():
        return "0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args():
    default_weights = Path(__file__).resolve().parent / "runs" / "detect" / "model_p2_960_safe" / "weights" / "best.pt"
    if not default_weights.exists():
        default_weights = Path(__file__).resolve().parent / "runs" / "detect" / "atcs_high_recall_model" / "weights" / "best.pt"
    if not default_weights.exists():
        default_weights = "yolov8n.pt"

    default_data = Path(__file__).resolve().parent / "configs" / "data_combined.yaml"
    if not default_data.exists():
        default_data = Path(__file__).resolve().parent / "configs" / "data.yaml"

    parser = argparse.ArgumentParser(description="Evaluate ATCS Vehicle Detection Model on Validation Dataset.")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(default_weights),
        help="Path to trained model weights checkpoint (.pt file).",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(default_data),
        help="Path to dataset configuration YAML file.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="Input image resolution (default: 960).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test", "train"],
        help="Dataset split to evaluate on (default: 'val').",
    )
    parser.add_argument(
        "--rect",
        action="store_true",
        default=True,
        help="Enable rectangular aspect-ratio evaluation (default: True).",
    )
    parser.add_argument(
        "--no-rect",
        dest="rect",
        action="store_false",
        help="Disable rectangular evaluation and use square letterboxing.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run evaluation on ('0', 'mps', 'cpu'). Auto-detects if omitted.",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        default=True,
        help="Save predictions in COCO format JSON.",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        default=True,
        help="Generate and save precision-recall curves and confusion matrix plots.",
    )
    return parser.parse_args()


# Executes validation evaluation with automatic MPS channel overflow protection
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

    # Automatically switches to CPU if MPS device is requested with high anchor counts (>65536)
    if target_device == "mps" and args.imgsz >= 960:
        print("[Notice] Image resolution >= 960px exceeds Apple MPS 65,536 channel limit. Running evaluation on CPU.")
        eval_device = "cpu"
    else:
        eval_device = target_device

    print("=" * 70)
    print(" ATCS Vehicle Detector: Model Evaluation & Validation")
    print("=" * 70)
    print(f" Model Weights:     {weights_path}")
    print(f" Dataset Config:    {data_yaml}")
    print(f" Evaluation Split:  {args.split}")
    print(f" Image Resolution:  {args.imgsz}px (Rectangular: {args.rect})")
    print(f" Target Device:     {eval_device.upper()}")
    print("=" * 70)

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
        print(f"\n[Warning] Validation on {eval_device} failed ({e}). Retrying on CPU...")
        metrics = model.val(
            data=str(data_yaml),
            imgsz=args.imgsz,
            split=args.split,
            rect=args.rect,
            device="cpu",
            save_json=args.save_json,
            plots=args.plots,
        )

    print("\n" + "=" * 70)
    print(" VALIDATION METRICS SUMMARY")
    print("=" * 70)
    print(f" Precision (P):    {metrics.box.mp:.4f}")
    print(f" Recall (R):       {metrics.box.mr:.4f}")
    print(f" mAP@50:           {metrics.box.map50:.4f}")
    print(f" mAP@50-95:        {metrics.box.map:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
