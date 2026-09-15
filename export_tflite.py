import sys
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Export trained YOLO PyTorch weights to edge-optimized TFLite format.")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(Path(__file__).resolve().parent / "runs" / "detect" / "atcs_high_recall_model" / "weights" / "best.pt"),
        help="Path to trained model checkpoint (.pt).",
    )
    parser.add_argument(
        "--imgsz",
        nargs="+",
        type=int,
        default=[320, 192],
        help="Input image resolution [width, height] or single square int (default: 320 192).",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Enable full INT8 post-training quantization.",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Enable FP16 half-precision export.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(Path(__file__).resolve().parent / "configs" / "data_combined.yaml"),
        help="Dataset YAML file for INT8 calibration representative dataset.",
    )
    return parser.parse_args()


# Converts PyTorch model weights to TFLite format with optional INT8 post-training quantization
def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: Ultralytics is not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    weights_path = Path(args.weights).resolve()
    if not weights_path.exists():
        print(f"Error: Weights file not found at: {weights_path}")
        sys.exit(1)

    img_size = args.imgsz if len(args.imgsz) > 1 else args.imgsz[0]

    print("=" * 70)
    print(" ATCS Model Exporter: PyTorch -> TFLite (Edge Optimization)")
    print("=" * 70)
    print(f" Source Weights:      {weights_path}")
    print(f" Target Resolution:   {img_size}")
    print(f" INT8 Quantization:   {args.int8}")
    print(f" FP16 Half Precision: {args.half}")
    if args.int8:
        print(f" Calibration Data:    {args.data}")
    print("=" * 70)

    model = YOLO(str(weights_path))

    export_path = model.export(
        format="tflite",
        imgsz=img_size,
        int8=args.int8,
        half=args.half,
        data=args.data if args.int8 else None,
    )

    print("\n" + "=" * 70)
    print(" Export completed successfully!")
    print(f" Exported Artifact: {export_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
