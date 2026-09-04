import sys
import time
import argparse
from pathlib import Path
import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="ATCS Vehicle Detector Inference & Live Visualization.")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(Path(__file__).resolve().parent / "runs" / "detect" / "atcs_high_recall_model" / "weights" / "best.pt"),
        help="Path to trained model weights (.pt file).",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source image, directory, video file, or camera index (e.g., 0).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold (default: 0.45).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="Inference image size (default: 960).",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save annotated results to disk.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "runs" / "inference"),
        help="Directory to save inference results.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display visualization window.",
    )
    return parser.parse_args()


# Draws status banner overlay showing detected vehicle counts and processing FPS
def draw_hud(frame, count, fps=None):
    h, w = frame.shape[:2]
    header_h = 55
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, header_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(
        frame,
        f"VEHICLES DETECTED: {count}",
        (15, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 128),
        2,
        cv2.LINE_AA,
    )

    if fps is not None:
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (w - 150, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )


# Performs inference and displays or saves detections for a single static image
def process_image(model, img_path, conf, iou, imgsz, show, save, out_dir):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Error: Could not read image from: {img_path}")
        return

    results = model.predict(source=img, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
    boxes = results[0].boxes
    vehicle_count = len(boxes)

    print(f"Inference on {img_path.name}: Found {vehicle_count} vehicles.")

    annotated_frame = results[0].plot(line_width=2)
    draw_hud(annotated_frame, vehicle_count)

    if save:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"pred_{Path(img_path).name}"
        cv2.imwrite(str(out_file), annotated_frame)
        print(f"Saved annotated image to: {out_file}")

    if show:
        cv2.imshow("ATCS Vehicle Detection", annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# Runs real-time inference on continuous video stream or video file
def process_stream_or_video(model, source, conf, iou, imgsz, show, save, out_dir):
    src_input = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src_input)

    if not cap.isOpened():
        print(f"Error: Could not open video source: {source}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0

    print(f"Processing stream/video: {source} ({width}x{height} @ {fps_in:.1f} FPS)...")

    video_writer = None
    if save:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "annotated_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(out_file), fourcc, fps_in, (width, height))
        print(f"Saving output video to: {out_file}")

    prev_time = time.time()
    frame_idx = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            curr_time = time.time()
            fps_calc = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
            prev_time = curr_time

            results = model.predict(source=frame, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
            boxes = results[0].boxes
            vehicle_count = len(boxes)

            annotated_frame = results[0].plot(line_width=2)
            draw_hud(annotated_frame, vehicle_count, fps_calc)

            if video_writer is not None:
                video_writer.write(annotated_frame)

            if show:
                cv2.imshow("ATCS Vehicle Detection & Counting", annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    break

            frame_idx += 1
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        if show:
            cv2.destroyAllWindows()


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: Ultralytics is not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    weights_path = Path(args.weights).resolve()
    if not weights_path.exists():
        print(f"Warning: Specified weights '{weights_path}' not found, falling back to 'yolov8n.pt'.")
        weights_path = "yolov8n.pt"

    print("=" * 70)
    print(" ATCS Vehicle Detector: Inference & Visualization")
    print("=" * 70)
    print(f" Model Weights: {weights_path}")
    print(f" Input Source:  {args.source}")
    print(f" Confidence:    {args.conf}")
    print(f" IoU Threshold: {args.iou}")
    print(f" Image Size:    {args.imgsz}")
    print("=" * 70)

    model = YOLO(str(weights_path))
    src_path = Path(args.source)

    if src_path.is_file() and src_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
        process_image(
            model, src_path, args.conf, args.iou, args.imgsz, not args.no_show, args.save, args.output_dir
        )
    else:
        process_stream_or_video(
            model, args.source, args.conf, args.iou, args.imgsz, not args.no_show, args.save, args.output_dir
        )


if __name__ == "__main__":
    main()
