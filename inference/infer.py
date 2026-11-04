"""
ATCS Vehicle Detector: Benchmarking, Tracking & Inference Utility.

Runs detection or ByteTrack/BoT-SORT tracking on images, image folders, video files, or webcams.
Provides HUD overlays, latency benchmarks, and annotated output saving.
"""

import argparse
from collections import deque
from datetime import datetime
from pathlib import Path
import sys
import time
import cv2
import numpy as np


def parse_args():
    base_dir = Path(__file__).resolve().parents[1]
    default_weights = base_dir / "runs" / "detect" / "atcs_high_recall_model" / "weights" / "best.pt"

    parser = argparse.ArgumentParser(description="ATCS Vehicle Detector: Benchmarking, Tracking & Inference Utility.")
    parser.add_argument("--weights", type=str, default=str(default_weights), help="Path to trained model weights (.pt file).")
    parser.add_argument("--allow-fallback", action="store_true", help="Permit fallback to 'yolov8n.pt' if custom weights are missing.")
    parser.add_argument("--source", type=str, required=True, help="Image file, image directory, video file, or camera index (e.g. '0').")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold (default: 0.25).")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (default: 0.45).")
    parser.add_argument("--imgsz", type=int, default=960, help="Inference image resolution (default: 960).")
    parser.add_argument("--device", type=str, default="", help="Inference device: 'mps', 'cuda:0', 'cpu', or '' (auto).")
    parser.add_argument("--track", action="store_true", help="Enable multi-object tracking across video frames.")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml", choices=["bytetrack.yaml", "botsort.yaml"], help="Tracker configuration.")
    parser.add_argument("--vid-stride", type=int, default=1, help="Video frame stride (default: 1).")
    parser.add_argument("--save", action="store_true", help="Save annotated results to disk.")
    parser.add_argument("--output-dir", type=str, default=str(base_dir / "runs" / "inference"), help="Directory to save inference results.")
    parser.add_argument("--no-show", action="store_true", help="Do not display visualization window.")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark mode: disables display and file writes to measure latency.")
    return parser.parse_args()


def validate_args(args):
    if not (0.0 <= args.conf <= 1.0):
        raise ValueError(f"Confidence threshold --conf must be between 0.0 and 1.0, got {args.conf}")
    if not (0.0 <= args.iou <= 1.0):
        raise ValueError(f"IoU threshold --iou must be between 0.0 and 1.0, got {args.iou}")
    if args.imgsz <= 0:
        raise ValueError(f"Image size --imgsz must be positive, got {args.imgsz}")
    if args.vid_stride < 1:
        raise ValueError(f"Video stride --vid-stride must be >= 1, got {args.vid_stride}")


def resolve_weights(weights_arg: str, allow_fallback: bool) -> str:
    weights_path = Path(weights_arg).resolve()
    if weights_path.is_file():
        return str(weights_path)

    alt_p2 = Path(__file__).resolve().parents[1] / "runs" / "detect" / "model_p2_960_safe" / "weights" / "best.pt"
    if alt_p2.is_file():
        print(f"[INFO] Using alternative project weights: {alt_p2}")
        return str(alt_p2)

    if allow_fallback:
        print("[WARNING] Custom trained weights not found. Falling back to pretrained 'yolov8n.pt' (--allow-fallback was specified).")
        return "yolov8n.pt"

    raise FileNotFoundError(
        f"Custom model weights not found at: {weights_path}\n"
        f"To avoid invalidating your experiment with arbitrary pretrained classes, execution is halted.\n"
        f"Please provide valid weights via '--weights <path>' or explicitly pass '--allow-fallback' to use yolov8n.pt."
    )


def draw_hud(frame, count, fps=None, infer_ms=None, is_tracking=False):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    mode_label = "TRACKED VEHICLES" if is_tracking else "VEHICLES DETECTED"
    cv2.putText(frame, f"{mode_label}: {count}", (15, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 128), 2, cv2.LINE_AA)

    stats = []
    if infer_ms is not None:
        stats.append(f"INFER: {infer_ms:.1f}ms")
    if fps is not None:
        stats.append(f"FPS: {fps:.1f}")

    if stats:
        text = " | ".join(stats)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0]
        cv2.putText(frame, text, (w - text_size[0] - 15, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2, cv2.LINE_AA)


def process_image(model, img_path: Path, args, out_dir: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[ERROR] Could not decode image: {img_path}")
        return

    t0 = time.perf_counter()
    results = model.predict(
        source=img,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device if args.device else None,
        verbose=False,
    )
    t_infer = (time.perf_counter() - t0) * 1000.0
    vehicle_count = len(results[0].boxes)
    print(f"[{img_path.name}] Found {vehicle_count} vehicles | Latency: {t_infer:.1f} ms")

    if args.benchmark:
        return

    annotated_frame = results[0].plot(line_width=2)
    draw_hud(annotated_frame, vehicle_count, infer_ms=t_infer, is_tracking=False)

    if args.save:
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = out_dir / f"pred_{img_path.stem}_{timestamp}{img_path.suffix}"
        if cv2.imwrite(str(out_file), annotated_frame):
            print(f"[SAVED] {out_file}")
        else:
            print(f"[ERROR] Failed to write image to: {out_file}")

    if not args.no_show:
        cv2.imshow("ATCS Vehicle Detection", annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_stream_or_video(model, source_str: str, args, out_dir: Path):
    is_live = source_str.isdigit()
    src_input = int(source_str) if is_live else source_str

    cap = cv2.VideoCapture(src_input)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source_str}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    if raw_fps <= 0.0 or np.isnan(raw_fps) or raw_fps > 120.0:
        print(f"[WARNING] Invalid input FPS reported ({raw_fps}). Defaulting to 25.0 FPS.")
        fps_in = 25.0
    else:
        fps_in = raw_fps

    print(f"[STREAM] Source: {source_str} ({width}x{height} @ {fps_in:.1f} FPS, Stride: {args.vid_stride})")

    video_writer = None
    if args.save and not args.benchmark:
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        src_stem = Path(source_str).stem if not is_live else f"cam_{source_str}"
        out_file = out_dir / f"annotated_{src_stem}_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        effective_fps = max(1.0, fps_in / args.vid_stride)
        video_writer = cv2.VideoWriter(str(out_file), fourcc, effective_fps, (width, height))
        if not video_writer.isOpened():
            print(f"[ERROR] Failed to initialize VideoWriter for: {out_file}")
            video_writer = None
        else:
            print(f"[SAVED] Recording annotated video to: {out_file}")

    frame_latencies = deque(maxlen=30)
    infer_latencies = deque(maxlen=30)
    all_infer_times = []
    frame_count, processed_count = 0, 0
    prev_loop_time = time.perf_counter()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if args.vid_stride > 1 and (frame_count % args.vid_stride != 0):
                continue

            processed_count += 1
            t_infer_start = time.perf_counter()

            if args.track:
                results = model.track(
                    source=frame,
                    conf=args.conf,
                    iou=args.iou,
                    imgsz=args.imgsz,
                    tracker=args.tracker,
                    persist=True,
                    device=args.device if args.device else None,
                    verbose=False,
                )
            else:
                results = model.predict(
                    source=frame,
                    conf=args.conf,
                    iou=args.iou,
                    imgsz=args.imgsz,
                    device=args.device if args.device else None,
                    verbose=False,
                )

            infer_ms = (time.perf_counter() - t_infer_start) * 1000.0
            infer_latencies.append(infer_ms)
            all_infer_times.append(infer_ms)

            curr_loop_time = time.perf_counter()
            dt_loop = curr_loop_time - prev_loop_time
            prev_loop_time = curr_loop_time
            if dt_loop > 0:
                frame_latencies.append(1.0 / dt_loop)

            rolling_fps = float(np.mean(frame_latencies)) if frame_latencies else 0.0
            mean_infer_ms = float(np.mean(infer_latencies)) if infer_latencies else infer_ms
            vehicle_count = len(results[0].boxes)

            if args.benchmark:
                if processed_count % 30 == 0:
                    print(f"Frame {processed_count:04d} | Infer: {mean_infer_ms:.1f} ms | Rolling FPS: {rolling_fps:.1f} | Detections: {vehicle_count}")
                continue

            annotated_frame = results[0].plot(line_width=2)
            draw_hud(annotated_frame, vehicle_count, fps=rolling_fps, infer_ms=mean_infer_ms, is_tracking=args.track)

            if video_writer is not None:
                video_writer.write(annotated_frame)

            if not args.no_show:
                cv2.imshow("ATCS Vehicle Detection & Tracking", annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    print("[INFO] Terminated by user.")
                    break
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        if not args.no_show and not args.benchmark:
            cv2.destroyAllWindows()

    if all_infer_times:
        p50 = np.percentile(all_infer_times, 50)
        p95 = np.percentile(all_infer_times, 95)
        avg_infer = np.mean(all_infer_times)
        avg_fps = 1000.0 / avg_infer if avg_infer > 0 else 0.0
        print("\n" + "=" * 60)
        print(" INFERENCE BENCHMARK SUMMARY")
        print("=" * 60)
        print(f" Total Frames Processed: {processed_count}")
        print(f" Mean Inference Latency: {avg_infer:.2f} ms")
        print(f" Median Latency (p50):   {p50:.2f} ms")
        print(f" 95th Percentile (p95):  {p95:.2f} ms")
        print(f" Theoretical Model FPS:  {avg_fps:.1f} FPS")
        print("=" * 60)


def main():
    args = parse_args()
    validate_args(args)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Ultralytics is not installed in the active environment.")
        print("Please activate your virtual environment: source venv/bin/activate")
        sys.exit(1)

    weights_path = resolve_weights(args.weights, args.allow_fallback)

    print("=" * 70)
    print(" ATCS Vehicle Detector: Inference & Benchmarking Pipeline")
    print("=" * 70)
    print(f" Weights:        {weights_path}")
    print(f" Source:         {args.source}")
    print(f" Confidence:     {args.conf}")
    print(f" IoU:            {args.iou}")
    print(f" Image Size:     {args.imgsz}")
    print(f" Device:         {args.device if args.device else 'auto'}")
    print(f" Tracking:       {'Enabled (ByteTrack)' if args.track else 'Disabled (Frame Detections)'}")
    print(f" Benchmark Mode: {'Active' if args.benchmark else 'Inactive'}")
    print("=" * 70)

    model = YOLO(weights_path)
    out_dir = Path(args.output_dir)
    src_path = Path(args.source)

    if src_path.is_dir():
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_files = sorted([p for p in src_path.iterdir() if p.suffix.lower() in image_extensions])
        if not image_files:
            print(f"[ERROR] No valid image files found in directory: {src_path}")
            return
        print(f"[INFO] Found {len(image_files)} images in directory. Processing sequentially...")
        for img_p in image_files:
            process_image(model, img_p, args, out_dir)

    elif src_path.is_file() and src_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        process_image(model, src_path, args, out_dir)

    else:
        process_stream_or_video(model, args.source, args, out_dir)


if __name__ == "__main__":
    main()
