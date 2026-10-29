import sys
import time
import json
import argparse
import threading
import queue
from pathlib import Path
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

from ultralytics import YOLO


def parse_args():
    default_weights = (
        Path(__file__).resolve().parent / "runs" / "detect" / "model_p2_960_safe" / "weights" / "best.pt"
    )
    if not default_weights.exists():
        default_weights = (
            Path(__file__).resolve().parent / "runs" / "detect" / "atcs_high_recall_model" / "weights" / "best.pt"
        )
    if not default_weights.exists():
        default_weights = "yolov8n.pt"

    parser = argparse.ArgumentParser(
        description="ATCS Lane-Aware Queue & Speed Estimator with Calibrated Telemetry Bridge (v3.2)."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(default_weights),
        help="Path to trained YOLO weights (.pt file).",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Input source: video file path, RTSP/HTTP stream URL, or webcam index (default: 0).",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack.yaml",
        choices=["bytetrack.yaml", "botsort.yaml"],
        help="Tracking algorithm configuration.",
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
        nargs="+",
        type=int,
        default=[960],
        help="Inference image size (default: 960).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Inference device: 'mps', 'cuda:0', 'cpu', or '' (auto).",
    )
    parser.add_argument(
        "--serial-port",
        type=str,
        default="/dev/cu.usbmodem5B900309611",
        help="Serial port of connected ESP32-S3 (default: /dev/cu.usbmodem5B900309611).",
    )
    parser.add_argument(
        "--baudrate",
        type=int,
        default=115200,
        help="Serial baudrate (default: 115200).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable interactive OpenCV visualization window.",
    )
    parser.add_argument(
        "--approach-leg",
        type=str,
        default="N",
        choices=["N", "S", "E", "W"],
        help="Intersection approach monitored by this video stream (default: N).",
    )
    parser.add_argument(
        "--calib-file",
        type=str,
        default="",
        help="Path to JSON file with measured 4-point ground homography correspondences.",
    )
    parser.add_argument(
        "--num-lanes",
        type=int,
        default=2,
        help="Number of approach lanes (default: 2).",
    )
    parser.add_argument(
        "--max-vehicle-gap-m",
        type=float,
        default=6.0,
        help="Maximum distance gap (meters) between adjacent vehicles in the same lane queue (default: 6.0m).",
    )
    parser.add_argument(
        "--stopline-threshold-m",
        type=float,
        default=14.0,
        help="Maximum distance from stopline (meters) for a vehicle to be considered at head of stopline queue.",
    )
    return parser.parse_args()


def validate_args(args):
    if args.num_lanes <= 0:
        raise ValueError(f"--num-lanes must be > 0, got {args.num_lanes}")
    if args.max_vehicle_gap_m <= 0.0:
        raise ValueError(f"--max-vehicle-gap-m must be positive, got {args.max_vehicle_gap_m}")
    if args.stopline_threshold_m < 0.0:
        raise ValueError(f"--stopline-threshold-m must be non-negative, got {args.stopline_threshold_m}")
    if not (0.0 <= args.conf <= 1.0):
        raise ValueError(f"--conf must be within [0.0, 1.0], got {args.conf}")
    if not (0.0 <= args.iou <= 1.0):
        raise ValueError(f"--iou must be within [0.0, 1.0], got {args.iou}")
    for dim in args.imgsz:
        if dim <= 0:
            raise ValueError(f"--imgsz dimensions must be positive integers, got {args.imgsz}")


# Planar Homography: Maps camera pixels (u, v) on the road plane into metric ground meters (X, Y)
class RoadHomographyCalibrator:
    def __init__(self, frame_w, frame_h, calib_file="", num_lanes=2, lane_width_m=3.5, approach_length_m=50.0):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.num_lanes = num_lanes
        self.total_width_m = num_lanes * lane_width_m
        self.approach_length_m = approach_length_m
        self.is_calibrated = False

        if calib_file and Path(calib_file).exists():
            success = self.load_and_validate_calibration(calib_file)
            if not success:
                print(f"[CALIB WARNING] Calibration file {calib_file} failed validation. Using unconfigured fallback.")
                self.setup_default_trapezoid()
        else:
            self.setup_default_trapezoid()

        self.H = cv2.getPerspectiveTransform(self.src_trap, self.dst_metric)
        det = np.linalg.det(self.H)
        if abs(det) < 1e-7:
            print(f"[CALIB WARNING] Degenerate homography matrix (det={det:.2e}). Falling back to uncalibrated.")
            self.setup_default_trapezoid()
            self.H = cv2.getPerspectiveTransform(self.src_trap, self.dst_metric)

    def load_and_validate_calibration(self, calib_file: str) -> bool:
        try:
            with open(calib_file, "r") as f:
                data = json.load(f)

            src_pts = np.float32(data.get("image_points", []))
            dst_pts = np.float32(data.get("ground_points_metric", []))

            # Validation 1: Exactly 4 points required for homography
            if len(src_pts) != 4 or len(dst_pts) != 4:
                print("[CALIB ERROR] Calibration must contain exactly 4 image_points and 4 ground_points_metric.")
                return False

            # Validation 2: Coordinates must be finite numbers
            if not (np.all(np.isfinite(src_pts)) and np.all(np.isfinite(dst_pts))):
                print("[CALIB ERROR] Calibration coordinates contain NaN or Inf values.")
                return False

            # Validation 3: Image points must lie within frame boundaries
            for pt in src_pts:
                if not (0 <= pt[0] <= self.frame_w and 0 <= pt[1] <= self.frame_h):
                    print(f"[CALIB ERROR] Image point {pt} is outside frame dimensions ({self.frame_w}x{self.frame_h}).")
                    return False

            # Validation 4: Quadrilateral area and convexity check
            area_img = cv2.contourArea(src_pts)
            if area_img < 1000.0:
                print(f"[CALIB ERROR] Degenerate image quadrilateral (area={area_img:.1f}).")
                return False

            if not cv2.isContourConvex(src_pts.astype(np.int32)):
                print("[CALIB ERROR] Image quadrilateral points are not convex or out of order.")
                return False

            # Validation 5: Ground points ordering and bounds
            x_coords = [p[0] for p in dst_pts]
            y_coords = [p[1] for p in dst_pts]
            if min(x_coords) < -0.1 or max(x_coords) > self.total_width_m * 1.5:
                print(f"[CALIB ERROR] Ground lateral coordinates out of expected range: {x_coords}")
                return False
            if min(y_coords) < -0.1 or max(y_coords) > self.approach_length_m * 1.5:
                print(f"[CALIB ERROR] Ground longitudinal coordinates out of expected range: {y_coords}")
                return False

            self.src_trap = src_pts
            self.dst_metric = dst_pts
            self.is_calibrated = True
            print(f"[CALIB] Successfully loaded and geometrically validated calibration from {calib_file}.")
            return True

        except Exception as e:
            print(f"[CALIB ERROR] Exception during calibration parsing: {e}")
            return False

    def setup_default_trapezoid(self):
        # Unconfigured geometric placeholder: marked uncalibrated
        self.src_trap = np.float32([
            [self.frame_w * 0.35, self.frame_h * 0.25],
            [self.frame_w * 0.65, self.frame_h * 0.25],
            [self.frame_w * 0.90, self.frame_h * 0.80],
            [self.frame_w * 0.10, self.frame_h * 0.80],
        ])
        self.dst_metric = np.float32([
            [0.0, self.approach_length_m],
            [self.total_width_m, self.approach_length_m],
            [self.total_width_m, 0.0],
            [0.0, 0.0],
        ])
        self.is_calibrated = False

    def contact_point_to_ground(self, u: float, v: float) -> Tuple[float, float, int]:
        pt = np.array([[[float(u), float(v)]]], dtype=np.float32)
        ground_pt = cv2.perspectiveTransform(pt, self.H)
        x_m = float(ground_pt[0, 0, 0])
        dist_m = float(ground_pt[0, 0, 1])

        lane_width = self.total_width_m / self.num_lanes

        # Explicit road boundary validation: Reject points outside approach corridor
        if not (0.0 <= x_m < self.total_width_m):
            return x_m, dist_m, -1
        if not (0.0 <= dist_m <= self.approach_length_m):
            return x_m, dist_m, -1

        lane_id = int(x_m / lane_width)
        return x_m, dist_m, lane_id


# Vehicle Track State with Dual-Threshold Hysteresis and Delta-Time Guard
class VehicleTrackState:
    def __init__(self, track_id: int, class_name: str, pcu: float):
        self.track_id = track_id
        self.class_name = class_name
        self.pcu = pcu

        self.history = deque(maxlen=20)  # Stores (time_s, x_m, dist_m)
        self.smoothed_speed_mps = None   # None until 2 observations exist
        self.stopped_since = None
        self.is_queued = False
        self.last_seen_time = 0.0
        self.lane_id = 0

    def update_position(
        self,
        t_src: float,
        x_m: float,
        dist_m: float,
        lane_id: int,
        enter_speed_mps=0.4,
        exit_speed_mps=0.8,
        min_dwell_time=1.0,
    ) -> Tuple[bool, Optional[float]]:
        self.last_seen_time = t_src
        self.lane_id = lane_id

        if len(self.history) > 0:
            prev_t, prev_x, prev_dist = self.history[-1]
            dt = t_src - prev_t

            # Critical gap guard: reset speed and reset queued status if tracking interrupted
            if dt > 0.8:
                self.smoothed_speed_mps = None
                self.stopped_since = None
                self.is_queued = False
                self.history.clear()
            elif dt > 0.005:
                dx = x_m - prev_x
                ddist = dist_m - prev_dist
                instant_speed = np.sqrt(dx**2 + ddist**2) / dt

                # Exponential moving average filter (alpha = 0.35)
                if self.smoothed_speed_mps is None:
                    self.smoothed_speed_mps = instant_speed
                else:
                    self.smoothed_speed_mps = 0.35 * instant_speed + 0.65 * self.smoothed_speed_mps
        self.history.append((t_src, x_m, dist_m))

        # Dual-Threshold Hysteresis for Stationary Classification
        if self.smoothed_speed_mps is not None and np.isfinite(self.smoothed_speed_mps):
            if not self.is_queued:
                if self.smoothed_speed_mps <= enter_speed_mps:
                    if self.stopped_since is None:
                        self.stopped_since = t_src
                    elif (t_src - self.stopped_since) >= min_dwell_time:
                        self.is_queued = True
                else:
                    self.stopped_since = None
            else:
                # Must exceed higher exit threshold to break out of queued state
                if self.smoothed_speed_mps >= exit_speed_mps:
                    self.is_queued = False
                    self.stopped_since = None

        return self.is_queued, self.smoothed_speed_mps


@dataclass
class LaneQueue:
    lane_id: int
    count: int
    pcu: float
    tail_m: float
    track_ids: Tuple[int, ...]
    connected_to_stopline: bool


# Lane-Aware Spatial Queue Estimator
class LaneAwareQueueEstimator:
    def __init__(self, max_vehicle_gap_m=6.0, stopline_threshold_m=14.0, history_size=5):
        self.max_vehicle_gap_m = max_vehicle_gap_m
        self.stopline_threshold_m = stopline_threshold_m
        self.history = deque(maxlen=history_size)

    def estimate_lane(self, lane_id: int, lane_vehicles: List[Tuple[float, float, int, tuple, str]]) -> LaneQueue:
        if not lane_vehicles:
            return LaneQueue(
                lane_id=lane_id, count=0, pcu=0.0, tail_m=0.0, track_ids=(), connected_to_stopline=False
            )

        # Sort stopped vehicles by ascending distance from stopline
        sorted_v = sorted(lane_vehicles, key=lambda v: v[0])

        lead_dist = sorted_v[0][0]
        # Distinguish stopline-connected queue from upstream stopped cluster
        is_connected = lead_dist <= self.stopline_threshold_m

        cluster = [sorted_v[0]]
        tail_m = lead_dist

        for i in range(1, len(sorted_v)):
            dist_m = sorted_v[i][0]
            gap = dist_m - tail_m
            if gap <= self.max_vehicle_gap_m:
                cluster.append(sorted_v[i])
                tail_m = dist_m
            else:
                break

        return LaneQueue(
            lane_id=lane_id,
            count=len(cluster),
            pcu=sum(v[1] for v in cluster),
            tail_m=tail_m,
            track_ids=tuple(v[2] for v in cluster),
            connected_to_stopline=is_connected,
        )

    def estimate(self, vehicles_by_lane: Dict[int, List], lane_ids: List[int]) -> Dict[int, LaneQueue]:
        results = {}
        for lid in lane_ids:
            vehicles = vehicles_by_lane.get(lid, [])
            results[lid] = self.estimate_lane(lid, vehicles)
        return results

    def smooth(
        self,
        q_stopline: int,
        pcu_stopline: float,
        tail_stopline_m: float,
        q_upstream: int,
        pcu_upstream: float,
    ):
        # Store co-observed frame snapshot
        self.history.append((q_stopline, pcu_stopline, tail_stopline_m, q_upstream, pcu_upstream))
        rows = list(self.history)
        if not rows:
            return {
                "Q_stopline": q_stopline,
                "PCU_stopline": pcu_stopline,
                "D_tail_m": tail_stopline_m,
                "Q_upstream": q_upstream,
                "PCU_upstream": pcu_upstream,
            }

        # Select the representative frame snapshot whose stopline count matches median count
        counts = [r[0] for r in rows]
        median_count = int(np.median(counts))
        best_row = min(rows, key=lambda r: abs(r[0] - median_count))

        return {
            "Q_stopline": int(best_row[0]),
            "PCU_stopline": round(float(best_row[1]), 2),
            "D_tail_m": round(float(best_row[2]), 2),
            "Q_upstream": int(best_row[3]),
            "PCU_upstream": round(float(best_row[4]), 2),
        }


# Vehicle Taxonomy Allowlist Filter (Rejects non-vehicle COCO classes such as persons, chairs, etc.)
VEHICLE_TAXONOMY = {
    "car", "motorcycle", "bus", "truck", "van", "bicycle", "auto_rickshaw", "vehicle"
}


def resolve_pcu(class_name: Optional[str], is_multiclass: bool) -> Tuple[Optional[float], Optional[str]]:
    if not class_name:
        return None, None

    name = class_name.lower().strip()

    if not is_multiclass:
        return 1.0, "VEHICLE"

    is_vehicle_class = (name in VEHICLE_TAXONOMY) or any(
        sub in name for sub in ["car", "bus", "truck", "motor", "bike", "auto", "van", "vehicle"]
    )
    if not is_vehicle_class:
        return None, None

    if "bus" in name or "truck" in name:
        return 2.5, "BUS/TRUCK"
    if "motorcycle" in name or "bike" in name or "two_wheeler" in name:
        return 0.5, "TWO_WHEELER"
    if "auto" in name or "rickshaw" in name:
        return 1.2, "AUTO_RICKSHAW"
    return 1.0, "CAR"


# Non-Blocking Threaded Telemetry Bridge with Write Timeout & Stale Packet Discard
class NonBlockingTelemetryBridge:
    def __init__(self, port: str, baudrate: int, approach_leg="N"):
        self.port = port
        self.baudrate = baudrate
        self.approach_leg = approach_leg

        self.serial_conn = None
        self.seq = 0
        self.controller_feedback = "STANDALONE"

        # Bounded queue with worker thread ensures inference loop NEVER stalls on serial I/O
        self.tx_queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def _worker_loop(self):
        last_connect_attempt = 0.0

        while not self.stop_event.is_set():
            # Handle background auto-reconnection every 5s
            now = time.time()
            if self.serial_conn is None and (now - last_connect_attempt >= 5.0):
                last_connect_attempt = now
                if SERIAL_AVAILABLE:
                    try:
                        self.serial_conn = serial.Serial(
                            self.port,
                            self.baudrate,
                            timeout=0.0,
                            write_timeout=0.05,  # 50 ms max write timeout guarantees non-blocking
                        )
                        print(f"[BRIDGE] Connected to ESP32-S3 on {self.port} @ {self.baudrate} baud.")
                    except Exception:
                        self.serial_conn = None

            # Pull telemetry payload from queue
            try:
                payload, seq_num = self.tx_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if self.serial_conn and self.serial_conn.is_open:
                try:
                    self.serial_conn.write(payload.encode("utf-8"))
                    if self.serial_conn.in_waiting > 0:
                        fb = self.serial_conn.readline().decode("utf-8", errors="ignore").strip()
                        if fb:
                            self.controller_feedback = fb
                except Exception as e:
                    print(f"[BRIDGE] Serial transmission error: {e}. Reconnection queued.")
                    self.safe_close()
            else:
                print(f"[TELEMETRY TX (Seq: {seq_num})] {payload.strip()}")

    def send_telemetry(
        self,
        smoothed_q: dict,
        is_calibrated: bool,
        lane_queues: Dict[int, LaneQueue],
        frame_wall_ms: int,
        source_timestamp_ms: int,
        frame_valid: bool = True,
        estimate_valid: bool = True,
    ):
        self.seq += 1

        now_wall_ms = int(time.time_ns() // 1_000_000)
        measurement_age_ms = max(0, now_wall_ms - frame_wall_ms)

        q_stopline = smoothed_q["Q_stopline"]
        pcu_stopline = smoothed_q["PCU_stopline"]
        q_upstream = smoothed_q["Q_upstream"]
        pcu_upstream = smoothed_q["PCU_upstream"]
        d_tail_m = smoothed_q["D_tail_m"]

        telemetry = {
            "seq": self.seq,
            "approach": self.approach_leg,
            # Control gate: valid requires both successful estimation and verified metric calibration
            "valid": bool(frame_valid and estimate_valid and is_calibrated),
            "frame_valid": frame_valid,
            "estimate_valid": estimate_valid,
            "calibrated": is_calibrated,
            # Core demand for signal scheduling (strictly stopline-connected)
            "Q": q_stopline,
            "PCU": pcu_stopline,
            "D_tail_m": d_tail_m,
            # Explicit academic telemetry breakdown
            "Q_stopline": q_stopline,
            "Q_upstream": q_upstream,
            "Q_stopped_total": q_stopline + q_upstream,
            "PCU_stopline": pcu_stopline,
            "PCU_upstream": pcu_upstream,
            "PCU_total": round(pcu_stopline + pcu_upstream, 2),
            "lanes_stopline": {
                f"lane_{lid}": lq.count for lid, lq in lane_queues.items() if lq.connected_to_stopline
            },
            "lanes_upstream": {
                f"lane_{lid}": lq.count for lid, lq in lane_queues.items() if not lq.connected_to_stopline
            },
            "timestamp_ms": frame_wall_ms,
            "source_timestamp_ms": source_timestamp_ms,
            "age_ms": measurement_age_ms,
        }

        json_str = json.dumps(telemetry) + "\n"

        # Discard stale packet if queue is full to preserve lowest telemetry latency
        if self.tx_queue.full():
            try:
                self.tx_queue.get_nowait()
            except queue.Empty:
                pass

        try:
            self.tx_queue.put_nowait((json_str, self.seq))
        except queue.Full:
            pass

    def safe_close(self):
        if self.serial_conn:
            try:
                if self.serial_conn.is_open:
                    self.serial_conn.close()
            except Exception:
                pass
            finally:
                self.serial_conn = None

    def close(self):
        self.stop_event.set()
        self.safe_close()
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=0.5)


# Image Sequence Reader: Seamlessly reads folders of frames (e.g. DETRAC sequences) like a video stream
class ImageSequenceCapture:
    def __init__(self, dir_path: Path, fps=25.0):
        self.dir_path = dir_path
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        self.files = sorted([p for p in self.dir_path.iterdir() if p.suffix.lower() in exts])
        self.idx = 0
        self.fps = fps
        self.is_open = len(self.files) > 0
        self.total_frames = len(self.files)

    def isOpened(self) -> bool:
        return self.is_open and (self.idx < self.total_frames)

    def read(self):
        if not self.is_open or self.idx >= self.total_frames:
            return False, None
        img = cv2.imread(str(self.files[self.idx]))
        self.idx += 1
        return (img is not None), img

    def get(self, propId):
        if propId == cv2.CAP_PROP_FPS:
            return self.fps
        if propId == cv2.CAP_PROP_POS_MSEC:
            return (self.idx / self.fps) * 1000.0
        if propId == cv2.CAP_PROP_FRAME_COUNT:
            return float(self.total_frames)
        return 0.0

    def release(self):
        self.is_open = False


def main():
    args = parse_args()
    validate_args(args)

    print("==================================================")
    print("  ATCS Lane-Aware Queue & Speed Estimator (v3.2)")
    print(f"  Weights:    {args.weights}")
    print(f"  Source:     {args.source}")
    print(f"  Approach:   {args.approach_leg} | Lanes: {args.num_lanes}")
    print(f"  Max Gap:    {args.max_vehicle_gap_m}m | Stopline Thresh: {args.stopline_threshold_m}m")
    print("==================================================")

    model = YOLO(args.weights)
    is_multiclass = len(model.names) > 1 and not (len(model.names) == 1 and model.names[0] == "vehicle")

    bridge = NonBlockingTelemetryBridge(args.serial_port, args.baudrate, args.approach_leg)

    # Differentiate live stream vs pre-recorded video file vs image sequence directory
    is_live_stream = False
    src_path = Path(args.source)

    if args.source.isdigit():
        source_input = int(args.source)
        is_live_stream = True
        cap = cv2.VideoCapture(source_input)
    elif args.source.lower().startswith(("rtsp://", "http://", "https://")):
        source_input = args.source
        is_live_stream = True
        cap = cv2.VideoCapture(source_input)
    elif src_path.is_dir():
        is_live_stream = False
        cap = ImageSequenceCapture(src_path, fps=25.0)
    else:
        source_input = args.source
        is_live_stream = False
        cap = cv2.VideoCapture(source_input)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {args.source}")
        sys.exit(1)

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(source_fps) or source_fps <= 0 or source_fps > 120:
        source_fps = 25.0

    ret, sample_frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read initial frame.")
        sys.exit(1)

    frame_h, frame_w = sample_frame.shape[:2]

    # Initialize Calibrator and Lane-Aware Estimator
    calibrator = RoadHomographyCalibrator(
        frame_w,
        frame_h,
        calib_file=args.calib_file,
        num_lanes=args.num_lanes,
        lane_width_m=3.5,
        approach_length_m=50.0,
    )
    estimator = LaneAwareQueueEstimator(
        max_vehicle_gap_m=args.max_vehicle_gap_m,
        stopline_threshold_m=args.stopline_threshold_m,
        history_size=5,
    )

    active_tracks: Dict[int, VehicleTrackState] = {}
    frame_idx = 0
    last_tx_time = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Video stream ended or loop completed.")
                break

            # Timestamp taken immediately after frame is decoded into memory
            frame_wall_ms = int(time.time_ns() // 1_000_000)
            frame_idx += 1

            # SOURCE TIME BASE (for vehicle velocity tracking)
            if is_live_stream:
                t_src = frame_wall_ms / 1000.0
                t_src_ms = frame_wall_ms
            else:
                msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                if msec > 0:
                    t_src = msec / 1000.0
                else:
                    t_src = frame_idx / source_fps
                t_src_ms = int(t_src * 1000)

            results = model.track(
                frame,
                persist=True,
                tracker=args.tracker,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device if args.device else None,
                verbose=False,
            )

            current_frame_ids = set()
            vehicles_by_lane = defaultdict(list)

            if results[0].boxes and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()
                classes = results[0].boxes.cls.int().cpu().numpy()

                for box, track_id, cls_id in zip(boxes, track_ids, classes):
                    tid = int(track_id)
                    current_frame_ids.add(tid)

                    class_name = model.names[cls_id] if cls_id in model.names else "vehicle"
                    pcu, category_label = resolve_pcu(class_name, is_multiclass)

                    # Filter out non-vehicle classes
                    if pcu is None or category_label is None:
                        continue

                    x1, y1, x2, y2 = map(int, box)
                    cx = (x1 + x2) / 2.0
                    base_y = float(y2)

                    # Project road-contact point through homography
                    x_m, dist_m, lane_id = calibrator.contact_point_to_ground(cx, base_y)

                    # Reject detections lying outside the calibrated road boundaries
                    if lane_id < 0:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 70, 70), 1)
                        continue

                    if tid not in active_tracks:
                        active_tracks[tid] = VehicleTrackState(tid, category_label, pcu)

                    track = active_tracks[tid]
                    is_queued, speed_mps = track.update_position(
                        t_src, x_m, dist_m, lane_id, enter_speed_mps=0.4, exit_speed_mps=0.8, min_dwell_time=1.0
                    )

                    # Stopped vehicles in approach corridor are queued candidates
                    if is_queued:
                        vehicles_by_lane[lane_id].append((dist_m, pcu, tid, (x1, y1, x2, y2), category_label))
                    else:
                        speed_kmh = (speed_mps * 3.6) if (speed_mps is not None and np.isfinite(speed_mps)) else 0.0
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 1)
                        cv2.putText(
                            frame,
                            f"L{lane_id} ID:{tid} {speed_kmh:.0f}km/h",
                            (x1, max(15, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (180, 180, 180),
                            1,
                            cv2.LINE_AA,
                        )

            # Purge stale tracks not seen in 2.5s
            active_tracks = {tid: t for tid, t in active_tracks.items() if t_src - t.last_seen_time < 2.5}

            # Lane-Aware Queue Estimation
            lane_ids = list(range(args.num_lanes))
            lane_queues = estimator.estimate(vehicles_by_lane, lane_ids)

            # Separate stopline queues from upstream clusters
            stopline_lanes = {lid: lq for lid, lq in lane_queues.items() if lq.connected_to_stopline}
            upstream_lanes = {lid: lq for lid, lq in lane_queues.items() if not lq.connected_to_stopline}

            raw_q_stopline = sum(lq.count for lq in stopline_lanes.values())
            raw_pcu_stopline = sum(lq.pcu for lq in stopline_lanes.values())
            stopline_tails = [lq.tail_m for lq in stopline_lanes.values() if lq.count > 0]
            approach_tail_stopline = max(stopline_tails) if stopline_tails else 0.0

            raw_q_upstream = sum(lq.count for lq in upstream_lanes.values())
            raw_pcu_upstream = sum(lq.pcu for lq in upstream_lanes.values())

            smoothed_q = estimator.smooth(
                raw_q_stopline,
                raw_pcu_stopline,
                approach_tail_stopline,
                raw_q_upstream,
                raw_pcu_upstream,
            )

            # Measurement validity criteria
            frame_valid = bool(ret and frame is not None and frame.size > 0)
            estimate_valid = frame_valid

            # Highlight queued vehicles
            for lid, lq in lane_queues.items():
                for item in vehicles_by_lane.get(lid, []):
                    dist_m, pcu, tid, box, cat_name = item
                    if tid in lq.track_ids:
                        x1, y1, x2, y2 = map(int, box)
                        # Color coding: Green for stopline-connected queue, Cyan for upstream stopped cluster
                        box_color = (0, 255, 128) if lq.connected_to_stopline else (255, 200, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                        prefix = "Q-L" if lq.connected_to_stopline else "UP-L"
                        cv2.putText(
                            frame,
                            f"{prefix}{lid} ID:{tid} [{cat_name}] {dist_m:.1f}m",
                            (x1, max(18, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.48,
                            box_color,
                            2,
                            cv2.LINE_AA,
                        )

            # Rate-limited telemetry transmission (1 Hz)
            now_s = time.time()
            if now_s - last_tx_time >= 1.0:
                last_tx_time = now_s
                bridge.send_telemetry(
                    smoothed_q,
                    calibrator.is_calibrated,
                    lane_queues,
                    frame_wall_ms,
                    t_src_ms,
                    frame_valid=frame_valid,
                    estimate_valid=estimate_valid,
                )

            # Heads-Up Display
            if not args.no_show:
                trap_pts = calibrator.src_trap.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [trap_pts], isClosed=True, color=(255, 128, 0), thickness=2)

                pt3 = tuple(map(int, calibrator.src_trap[3]))
                pt2 = tuple(map(int, calibrator.src_trap[2]))
                cv2.line(frame, pt3, pt2, (0, 0, 255), 3)
                cv2.putText(frame, "STOPLINE (0m)", (pt3[0] + 10, pt3[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

                hud = frame.copy()
                cv2.rectangle(hud, (0, 0), (frame_w, 68), (20, 20, 20), -1)
                cv2.addWeighted(hud, 0.85, frame, 0.15, 0, frame)

                lane_breakdown_str = " | ".join(
                    f"L{lid}:{lq.count}{'*' if lq.connected_to_stopline else '^'}" for lid, lq in lane_queues.items()
                )
                cv2.putText(
                    frame,
                    f"APPROACH [{args.approach_leg}] STOPLINE: {smoothed_q['Q_stopline']} ({lane_breakdown_str}) | UPSTREAM: {smoothed_q['Q_upstream']} | DEMAND: {smoothed_q['PCU_stopline']} PCU",
                    (15, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.54,
                    (0, 255, 128) if smoothed_q["Q_stopline"] > 0 else (200, 200, 200),
                    2,
                    cv2.LINE_AA,
                )

                calib_str = "CALIBRATION LOADED" if calibrator.is_calibrated else "UNCONFIGURED APPROX"
                cv2.putText(
                    frame,
                    f"TAIL: {smoothed_q['D_tail_m']}m | TIME: {t_src:.1f}s | CALIB: {calib_str}",
                    (15, 52),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )

                conn_status = "HW SERIAL ONLINE" if (bridge.serial_conn and bridge.serial_conn.is_open) else "STANDALONE"
                cv2.putText(
                    frame,
                    f"LINK: {conn_status}",
                    (frame_w - 240, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0) if (bridge.serial_conn and bridge.serial_conn.is_open) else (0, 165, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow("ATCS Lane-Aware Queue & Speed Estimator (v3.2)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        cap.release()
        bridge.close()
        if not args.no_show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
