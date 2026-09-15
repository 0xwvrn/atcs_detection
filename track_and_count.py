import sys
import time
import argparse
from pathlib import Path
from collections import defaultdict, deque
import cv2
import numpy as np


def parse_args():
    default_weights = Path(__file__).resolve().parent / "runs" / "detect" / "atcs_high_recall_model" / "weights" / "best.pt"
    if not default_weights.exists():
        default_weights = Path(__file__).resolve().parent / "runs" / "detect" / "atcs_vehicle_detector" / "weights" / "best.pt"
    if not default_weights.exists():
        default_weights = "yolov8n.pt"

    parser = argparse.ArgumentParser(description="ATCS Multi-Object Tracking & Vehicle Queue / Throughput Counting.")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(default_weights),
        help="Path to trained model weights (.pt file).",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source video file, RTSP stream URL, or webcam device index (e.g., 0).",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack.yaml",
        choices=["bytetrack.yaml", "botsort.yaml"],
        help="Tracking algorithm configuration ('bytetrack.yaml' or 'botsort.yaml').",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold (default: 0.25).",
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
        help="Inference input image size (default: 960).",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save annotated tracking video to output directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "runs" / "tracking"),
        help="Directory to save tracking output videos.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display interactive video visualization window.",
    )
    return parser.parse_args()


# Manages multi-object trajectory association, virtual line crossing, and queue occupancy counting
class VehicleTrafficTracker:
    def __init__(self, line_y_ratio=0.65, history_len=30):
        self.track_history = defaultdict(lambda: deque(maxlen=history_len))
        self.counted_ids = set()
        self.active_queue_ids = set()
        self.line_y_ratio = line_y_ratio
        self.total_throughput_count = 0

    # Updates vehicle trajectory history and checks if a persistent Track ID crossed the stopline
    def process_tracks(self, frame, boxes, track_ids):
        h, w = frame.shape[:2]
        line_y = int(h * self.line_y_ratio)

        cv2.line(frame, (0, line_y), (w, line_y), (0, 165, 255), 2)
        cv2.putText(
            frame,
            "STOPLINE",
            (20, line_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )

        current_frame_ids = set()

        if track_ids is not None:
            for box, track_id in zip(boxes, track_ids):
                tid = int(track_id)
                current_frame_ids.add(tid)

                x1, y1, x2, y2 = map(int, box)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                self.track_history[tid].append((center_x, center_y))

                # Evaluates directional line intersection using preceding trajectory history
                if len(self.track_history[tid]) >= 2:
                    prev_y = self.track_history[tid][-2][1]
                    if prev_y < line_y <= center_y and tid not in self.counted_ids:
                        self.counted_ids.add(tid)
                        self.total_throughput_count += 1

                box_color = (0, 255, 0) if tid in self.counted_ids else (255, 200, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(
                    frame,
                    f"ID: {tid}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    box_color,
                    2,
                    cv2.LINE_AA,
                )

                points = np.array(self.track_history[tid], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=(0, 255, 255), thickness=2)

        active_queue_count = len(current_frame_ids)
        return active_queue_count, self.total_throughput_count


# Draws heads-up display overlay with active lane queue and cumulative throughput
def draw_tracking_hud(frame, queue_count, throughput_count, fps):
    h, w = frame.shape[:2]
    header_h = 65

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, header_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    cv2.putText(
        frame,
        f"ACTIVE QUEUE: {queue_count}",
        (15, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 255, 128),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"TOTAL PASSED: {throughput_count}",
        (w // 2 - 80, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 200, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (w - 140, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (200, 200, 200),
        2,
        cv2.LINE_AA,
    )


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
    print(" ATCS Multi-Object Tracking & Throughput Engine")
    print("=" * 70)
    print(f" Model Weights:     {weights_path}")
    print(f" Tracker Config:    {args.tracker}")
    print(f" Video Source:      {args.source}")
    print(f" Confidence / IoU:  {args.conf} / {args.iou}")
    print(f" Image Resolution:  {args.imgsz}px")
    print("=" * 70)

    model = YOLO(str(weights_path))
    tracker_manager = VehicleTrafficTracker()

    src_input = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src_input)

    if not cap.isOpened():
        print(f"Error: Could not open video source: {args.source}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0

    video_writer = None
    if args.save:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"tracking_{Path(str(args.source)).stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(out_file), fourcc, fps_in, (width, height))
        print(f"Saving tracking video stream to: {out_file}")

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

            results = model.track(
                source=frame,
                persist=True,
                tracker=args.tracker,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                verbose=False,
            )

            boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []
            track_ids = results[0].boxes.id.int().cpu().numpy() if results[0].boxes is not None and results[0].boxes.id is not None else None

            queue_cnt, throughput_cnt = tracker_manager.process_tracks(frame, boxes, track_ids)
            draw_tracking_hud(frame, queue_cnt, throughput_cnt, fps_calc)

            if video_writer is not None:
                video_writer.write(frame)

            if not args.no_show:
                cv2.imshow("ATCS Multi-Object Tracking & Vehicle Counting", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    break

            frame_idx += 1
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        if not args.no_show:
            cv2.destroyAllWindows()

    print("\n" + "=" * 70)
    print(" Tracking session completed.")
    print(f" Total Cumulative Throughput: {tracker_manager.total_throughput_count} vehicles")
    print("=" * 70)


if __name__ == "__main__":
    main()
