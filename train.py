import sys
import gc
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
    default_data = Path(__file__).resolve().parent / "configs" / "data_combined.yaml"
    if not default_data.exists():
        default_data = Path(__file__).resolve().parent / "configs" / "data.yaml"

    parser = argparse.ArgumentParser(description="Train High-Recall ATCS Vehicle Detector (Memory Safe).")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n-p2.yaml",
        help="Model architecture or weights file (e.g., yolov8n-p2.yaml, yolov8n.pt, yolo11n.pt).",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(default_data),
        help="Path to dataset configuration YAML file.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="Number of training epochs (default: 60).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size (default: 8 for safe 960px memory footprint on 16GB Mac).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="Input image resolution (default: 960 for high small-vehicle recall).",
    )
    parser.add_argument(
        "--rect",
        action="store_true",
        default=True,
        help="Enable 16:9 rectangular training (no square distortion).",
    )
    parser.add_argument(
        "--no-rect",
        dest="rect",
        action="store_false",
        help="Disable rectangular training and use standard square letterboxing.",
    )
    parser.add_argument(
        "--cls",
        type=float,
        default=1.5,
        help="Classification loss gain (default: 1.5 to penalize missed detections and boost Recall).",
    )
    parser.add_argument(
        "--box",
        type=float,
        default=7.5,
        help="Bounding box loss gain (default: 7.5).",
    )
    parser.add_argument(
        "--dfl",
        type=float,
        default=1.5,
        help="Distribution Focal Loss gain (default: 1.5).",
    )
    parser.add_argument(
        "--cos-lr",
        action="store_true",
        default=True,
        help="Use cosine learning rate scheduler (default: True).",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help="Optimizer (default: 'AdamW' for smooth gradient convergence).",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.6,
        help="Scale augmentation gain for multi-scale vehicle learning (default: 0.6).",
    )
    parser.add_argument(
        "--mosaic",
        type=float,
        default=1.0,
        help="Mosaic augmentation probability (default: 1.0).",
    )
    parser.add_argument(
        "--close-mosaic",
        type=int,
        default=10,
        help="Disable mosaic for the final N epochs for sharp fine-tuning (default: 10).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (e.g., '0', 'mps', 'cpu'). If not specified, auto-detects.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of DataLoader worker threads (default: 2 for low RAM overhead).",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=str(Path(__file__).resolve().parent / "runs" / "detect"),
        help="Project directory to save training runs.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="atcs_high_recall_model",
        help="Name of this specific training run.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the most recent checkpoint.",
    )
    return parser.parse_args()


# Executes training and runs post-training validation with safe device fallback
def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: Ultralytics is not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    device = args.device if args.device is not None else detect_device()
    data_yaml = Path(args.data).resolve()

    if not data_yaml.exists():
        print(f"Error: Dataset YAML not found at: {data_yaml}")
        sys.exit(1)

    print("=" * 70)
    print(" ATCS High-Recall Vehicle Detector Training (Memory-Safe Mode)")
    print("=" * 70)
    print(f" Model Architecture:  {args.model}")
    print(f" Dataset Config:      {data_yaml}")
    print(f" Target Device:       {device.upper()}")
    print(f" Image Resolution:    {args.imgsz}px (Rectangular: {args.rect})")
    print(f" Safe Batch Size:     {args.batch} (Gradient Accumulation Active)")
    print(f" Epochs:              {args.epochs} (Close-mosaic: {args.close_mosaic})")
    print(f" Optimizer:           {args.optimizer} (Cosine LR: {args.cos_lr})")
    print(f" Loss Weights:        cls={args.cls} (Recall-Boosted), box={args.box}, dfl={args.dfl}")
    print(f" DataLoader Workers:  {args.workers} (RAM-Safe Mode, cache=False)")
    print(f" Output Project:      {args.project}/{args.name}")
    print("=" * 70)

    # Loads custom architecture with pretrained backbone weights if yaml specified
    if args.model.endswith(".yaml"):
        model = YOLO(args.model).load("yolov8n.pt")
    else:
        model = YOLO(args.model)

    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        rect=args.rect,
        cls=args.cls,
        box=args.box,
        dfl=args.dfl,
        cos_lr=args.cos_lr,
        optimizer=args.optimizer,
        scale=args.scale,
        mosaic=args.mosaic,
        close_mosaic=args.close_mosaic,
        device=device,
        workers=args.workers,
        cache=False,
        project=args.project,
        name=args.name,
        resume=args.resume,
        exist_ok=True,
        pretrained=True,
        val=False,
        verbose=True,
        plots=True,
    )

    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    gc.collect()

    print("\n" + "=" * 70)
    print(" Evaluating Best Model on Validation Split...")
    print("=" * 70)

    # Automatically uses CPU for validation if MPS device is active with high anchor counts (>65536)
    val_device = "cpu" if device == "mps" and args.imgsz >= 960 else device

    try:
        val_results = model.val(data=str(data_yaml), imgsz=args.imgsz, rect=args.rect, device=val_device)
    except Exception as e:
        print(f"[Notice] Validation on {val_device} encountered: {e}. Falling back to CPU...")
        val_results = model.val(data=str(data_yaml), imgsz=args.imgsz, rect=args.rect, device="cpu")

    best_weights_path = Path(args.project) / args.name / "weights" / "best.pt"
    print("\n" + "=" * 70)
    print(" TRAINING COMPLETE!")
    print("=" * 70)
    print(f" Best Weights Saved To: {best_weights_path}")
    print(f" Validation Recall:     {val_results.box.mr:.4f}  (Target: >= 0.80)")
    print(f" Validation Precision:  {val_results.box.mp:.4f}")
    print(f" Validation mAP@50:     {val_results.box.map50:.4f}")
    print(f" Validation mAP@50-95:  {val_results.box.map:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
