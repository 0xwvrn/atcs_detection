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
    default_data = Path(__file__).resolve().parent / "configs" / "data.yaml"

    parser = argparse.ArgumentParser(description="Train ATCS Vehicle Detection Model on UA-DETRAC dataset.")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Initial model weights or architecture definition (default: yolov8n.pt).",
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
        default=50,
        help="Number of training epochs (default: 50).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Training batch size (default: 16).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image resolution (default: 640).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run training on ('0', 'mps', 'cpu'). Auto-detects if omitted.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker threads for DataLoader (default: 4).",
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
        default="atcs_vehicle_detector",
        help="Experiment name for saving run logs and weights.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the most recent checkpoint.",
    )
    return parser.parse_args()


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
    print(" ATCS Vehicle Detector Model Training")
    print("=" * 70)
    print(f" Model:      {args.model}")
    print(f" Dataset:    {data_yaml}")
    print(f" Epochs:     {args.epochs}")
    print(f" Batch Size: {args.batch}")
    print(f" Image Size: {args.imgsz}")
    print(f" Device:     {device.upper()}")
    print(f" Output:     {args.project}/{args.name}")
    print("=" * 70)

    model = YOLO(args.model)

    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        resume=args.resume,
        exist_ok=True,
        verbose=True,
        plots=True,
    )

    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    gc.collect()

    print("\n" + "=" * 70)
    print(" Evaluating Best Model on Validation Split...")
    print("=" * 70)
    val_results = model.val(data=str(data_yaml), imgsz=args.imgsz, device=device)

    best_weights_path = Path(args.project) / args.name / "weights" / "best.pt"
    print("\n" + "=" * 70)
    print(" Training Complete!")
    print(f" Best Weights Saved To: {best_weights_path}")
    print(f" Validation mAP@50:     {val_results.box.map50:.4f}")
    print(f" Validation mAP@50-95:  {val_results.box.map:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
