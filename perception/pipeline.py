"""
ATCS Traffic Perception Pipeline.

Main entry point and orchestrator coordinating YOLO tracking, traffic analysis,
serial telemetry transmission, and optional OpenCV visualization.
"""

import argparse
from collections import defaultdict
from pathlib import Path
import sys
import time
from typing import Dict, Tuple

import cv2
import numpy as np

try:
    from .telemetry import NonBlockingTelemetryBridge, SERIAL_AVAILABLE
    from .traffic_analysis import (
        CalibrationStatus,
        CameraHealthState,
        CameraStreamHealthMonitor,
        LaneAwareQueueEstimator,
        RoadHomographyCalibrator,
        UpstreamArrivalTracker,
        VehicleTrackState,
        resolve_pcu,
    )
    from .visualization import (
        draw_approach_corridor,
        draw_heads_up_display,
        draw_moving_vehicle,
        draw_queued_vehicle,
    )
except ImportError:
    from telemetry import NonBlockingTelemetryBridge, SERIAL_AVAILABLE
    from traffic_analysis import (
        CalibrationStatus,
        CameraHealthState,
        CameraStreamHealthMonitor,
        LaneAwareQueueEstimator,
        RoadHomographyCalibrator,
        UpstreamArrivalTracker,
        VehicleTrackState,
        resolve_pcu,
    )
    from visualization import (
        draw_approach_corridor,
        draw_heads_up_display,
        draw_moving_vehicle,
        draw_queued_vehicle,
    )


def parse_args():
    base_dir = Path(__file__).resolve().parents[1]
    default_weights = next(
        (p for p in (
            base_dir / "runs" / "detect" / "model_p2_960_safe" / "weights" / "best.pt",
            base_dir / "runs" / "detect" / "atcs_high_recall_model" / "weights" / "best.pt",
        ) if p.exists()),
        "yolov8n.pt",
    )

    parser = argparse.ArgumentParser(
        description="ATCS Traffic Perception: Lane-Aware Queue & Speed Estimator with Telemetry Bridge."
    )
    parser.add_argument("--weights", type=str, default=str(default_weights), help="Path to trained YOLO checkpoint (.pt).")
    parser.add_argument("--source", type=str, default="0", help="Video file path, stream URL, or camera index.")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml", choices=["bytetrack.yaml", "botsort.yaml"], help="MOT tracker config.")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold (default: 0.25).")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (default: 0.45).")
    parser.add_argument("--imgsz", nargs="+", type=int, default=[960], help="Inference resolution (default: 960).")
    parser.add_argument("--device", type=str, default="", help="Compute device: 'mps', 'cuda:0', 'cpu', or '' (auto).")
    parser.add_argument("--serial-port", type=str, default="/dev/cu.usbmodem5B900309611", help="Serial port of ESP32-S3 (empty string disables serial).")
    parser.add_argument("--standalone", action="store_true", help="Run perception without opening a serial connection.")
    parser.add_argument("--baudrate", type=int, default=115200, help="Serial baud rate (default: 115200).")
    parser.add_argument("--approach-leg", type=str, default="N", choices=["N", "S", "E", "W"], help="Monitored firmware approach (default: N).")
    parser.add_argument("--calib-file", type=str, default="", help="Path to 4-point homography calibration JSON.")
    parser.add_argument("--num-lanes", type=int, default=2, help="Number of approach lanes (default: 2).")
    parser.add_argument("--max-vehicle-gap-m", type=float, default=6.0, help="Max bumper clearance gap in queue (default: 6.0m).")
    parser.add_argument("--stopline-threshold-m", type=float, default=14.0, help="Max distance to stopline for queue head (default: 14.0m).")
    parser.add_argument("--allow-approx-calib", action="store_true", help="Permit default geometric trapezoid for valid telemetry.")
    parser.add_argument("--telemetry-rate", type=float, default=2.0, help="Telemetry transmission rate in Hz (default: 2.0).")
    parser.add_argument("--no-show", action="store_true", help="Disable interactive OpenCV visualization window.")
    return parser.parse_args()


def validate_args(args):
    if args.num_lanes <= 0:
        raise ValueError(f"--num-lanes must be > 0, got {args.num_lanes}")
    if args.max_vehicle_gap_m <= 0.0 or args.telemetry_rate <= 0.0:
        raise ValueError("max-vehicle-gap-m and telemetry-rate must be positive")
    if args.stopline_threshold_m < 0.0:
        raise ValueError("stopline-threshold-m must be non-negative")
    if not (0.0 <= args.conf <= 1.0 and 0.0 <= args.iou <= 1.0):
        raise ValueError("--conf and --iou must be within [0.0, 1.0]")
    if any(dim <= 0 for dim in args.imgsz):
        raise ValueError(f"--imgsz dimensions must be positive integers, got {args.imgsz}")


def create_video_capture(source: str) -> Tuple[cv2.VideoCapture, bool]:
    """Factory creating appropriate video capture; returns (cap, is_live_stream)."""
    if source.isdigit():
        return cv2.VideoCapture(int(source)), True
    if source.lower().startswith(("rtsp://", "http://", "https://")):
        return cv2.VideoCapture(source), True
    return cv2.VideoCapture(source), False


def get_model_class_name(names, cls_id: int) -> str:
    """Safely retrieves class name string regardless of whether names is dict or list/tuple."""
    if isinstance(names, dict):
        return str(names.get(cls_id, "vehicle"))
    if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
        return str(names[cls_id])
    return "vehicle"


def resolve_serial_mode(args, argv=None) -> Tuple[bool, str]:
    """
    Validates serial dependency handling:
      - Fails fast with an actionable error if serial was explicitly requested without pyserial.
      - Preserves intentional standalone mode when --standalone or --serial-port "" is used,
        or when serial was not explicitly requested and pyserial is unavailable.
    Returns (use_standalone, link_label).
    """
    cli_args = sys.argv[1:] if argv is None else list(argv)
    serial_flag_passed = any(a == "--serial-port" or a.startswith("--serial-port=") for a in cli_args)
    standalone_flag = bool(getattr(args, "standalone", False))
    port = str(getattr(args, "serial_port", "") or "").strip()

    if standalone_flag or not port:
        return True, "STANDALONE"

    if serial_flag_passed and not SERIAL_AVAILABLE:
        raise RuntimeError(
            f"Serial transmission requested (--serial-port {port}), but 'pyserial' is not installed. "
            "Install it with 'pip install pyserial>=3.5' (or pass '--standalone' to run without serial)."
        )

    if not SERIAL_AVAILABLE:
        return True, "STANDALONE (pyserial missing)"

    return False, f"Port={port}"


def run_perception(args):
    try:
        use_standalone, link_label = resolve_serial_mode(args)
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print(
        f"# ATCS Traffic Perception Pipeline (v4.0) | Approach={args.approach_leg} "
        f"| Lanes={args.num_lanes} | Tx={args.telemetry_rate}Hz | {link_label}"
    )

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Ultralytics is not installed. Please activate your virtual environment: source venv/bin/activate")
        sys.exit(1)

    model = YOLO(args.weights)
    names_list = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)
    is_multiclass = len(names_list) > 1 and not (len(names_list) == 1 and names_list[0] == "vehicle")

    bridge = None
    cap = None

    try:
        bridge = NonBlockingTelemetryBridge(
            args.serial_port,
            args.baudrate,
            args.approach_leg,
            allow_approx_calib=args.allow_approx_calib,
            standalone=use_standalone,
        )

        cap, is_live_stream = create_video_capture(args.source)
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
        calibrator = RoadHomographyCalibrator(frame_w, frame_h, calib_file=args.calib_file, num_lanes=args.num_lanes)
        estimator = LaneAwareQueueEstimator(
            max_vehicle_gap_m=args.max_vehicle_gap_m,
            stopline_threshold_m=args.stopline_threshold_m,
            history_size=5,
        )
        arrival_tracker = UpstreamArrivalTracker(
            window_duration_s=10.0,
            corridor_width_m=calibrator.total_width_m,
            stopline_threshold_m=args.stopline_threshold_m,
        )
        health_monitor = CameraStreamHealthMonitor(stale_timeout_s=2.0)

        active_tracks: Dict[int, VehicleTrackState] = {}
        frame_idx, last_tx_mono = 0, 0.0
        last_media_t_src = 0.0
        tx_interval = 1.0 / max(0.1, args.telemetry_rate)
        invalid_observation = {
            "Q_stopline": 0,
            "PCU_stopline": 0.0,
            "D_tail_m": 0.0,
            "Q_upstream": 0,
            "PCU_upstream": 0.0,
            "inflow_pcu": 0.0,
            "arrival_rate_pcu_s": 0.0,
            "metric_valid": False,
        }

        curr_frame = sample_frame
        ret = True
        while True:
            mono_now = time.monotonic()

            if not ret or curr_frame is None or curr_frame.size == 0:
                if not is_live_stream:
                    print("[INFO] Video feed ended or completed.")
                    break

                health_monitor.record_read_failure(mono_now)
                if mono_now - last_tx_mono >= tx_interval:
                    last_tx_mono = mono_now
                    health_monitor.record_telemetry_published(mono_now)
                    bridge.send_telemetry(
                        invalid_observation,
                        calibrator.metric_valid,
                        frame_valid=False,
                        fresh=False,
                    )

                if health_monitor.should_attempt_reconnect(mono_now):
                    health_monitor.mark_reconnect_attempt(mono_now)
                    if cap is not None:
                        cap.release()
                    cap, _ = create_video_capture(args.source)
                elif health_monitor.state == CameraHealthState.DISCONNECTED and health_monitor.reconnect_attempts >= health_monitor.max_reconnect_attempts:
                    print("[ERROR] Maximum camera reconnection attempts exhausted.")
                    break

                time.sleep(0.05)
                ret, curr_frame = cap.read() if (cap is not None and cap.isOpened()) else (False, None)
                continue

            frame = curr_frame
            frame_idx += 1
            fh, fw = frame.shape[:2]
            calibrator.check_frame_dimensions(fw, fh)
            health_monitor.record_frame_acquired(frame, mono_now)

            if health_monitor.consume_tracker_reset_needed():
                active_tracks.clear()
                arrival_tracker.reset()
                estimator.history.clear()

            if is_live_stream:
                t_src = mono_now
            else:
                msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                cand_t = (msec / 1000.0) if msec > 0 else (frame_idx / source_fps)
                t_src = cand_t if cand_t >= last_media_t_src else (last_media_t_src + (1.0 / source_fps))
                last_media_t_src = t_src

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

            vehicles_by_lane = defaultdict(list)
            frame_observations = []

            if results[0].boxes and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()
                classes = results[0].boxes.cls.int().cpu().numpy()

                for box, track_id, cls_id in zip(boxes, track_ids, classes):
                    tid = int(track_id)
                    class_name = get_model_class_name(model.names, int(cls_id))
                    pcu, cat = resolve_pcu(class_name, is_multiclass)
                    if pcu is None or cat is None:
                        continue

                    x1, y1, x2, y2 = map(int, box)
                    x_m, dist_m, lane_id = calibrator.contact_point_to_ground((x1 + x2) / 2.0, float(y2))
                    if lane_id < 0:
                        if not args.no_show:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 70, 70), 1)
                        continue

                    if tid not in active_tracks:
                        active_tracks[tid] = VehicleTrackState(tid, cat, pcu)
                    else:
                        active_tracks[tid].update_class(cat, pcu)

                    track = active_tracks[tid]
                    is_queued, speed_mps = track.update_position(t_src, x_m, dist_m)
                    frame_observations.append((tid, x_m, dist_m, track.pcu, track.class_name, is_queued))

                    if is_queued:
                        vehicles_by_lane[lane_id].append((dist_m, track.pcu, tid, (x1, y1, x2, y2), track.class_name))
                    else:
                        if not args.no_show:
                            speed_kmh = (speed_mps * 3.6) if (speed_mps is not None and np.isfinite(speed_mps)) else 0.0
                            draw_moving_vehicle(frame, (x1, y1, x2, y2), lane_id, tid, speed_kmh)

            # Purge stale tracks not observed within 2.5s
            active_tracks = {tid: t for tid, t in active_tracks.items() if t_src - t.last_seen_time < 2.5}

            health_monitor.record_frame_processed(True, mono_now)
            obs_fresh = health_monitor.is_observation_fresh(mono_now)
            arrival_rate_pcu_s = arrival_tracker.update(t_src, frame_observations) if (obs_fresh and calibrator.metric_valid) else 0.0

            # Lane-aware queue aggregation and smoothing
            lane_queues = estimator.estimate(vehicles_by_lane, list(range(args.num_lanes)))
            raw_q_stop = sum(lq.count for lq in lane_queues.values())
            raw_pcu_stop = sum(lq.pcu for lq in lane_queues.values())
            tails = [lq.tail_m for lq in lane_queues.values() if lq.count > 0]
            raw_tail = max(tails) if tails else 0.0

            raw_q_up = sum(lq.upstream_count for lq in lane_queues.values())
            raw_pcu_up = sum(lq.upstream_pcu for lq in lane_queues.values())

            smoothed_q = estimator.smooth(
                raw_q_stop,
                raw_pcu_stop,
                raw_tail,
                raw_q_up,
                raw_pcu_up,
                arrival_rate_pcu_s,
                metric_valid=calibrator.metric_valid,
                calibration_status=calibrator.status.value,
            )

            # Rate-limited telemetry transmission using monotonic clock
            now_mono = time.monotonic()
            if now_mono - last_tx_mono >= tx_interval:
                last_tx_mono = now_mono
                health_monitor.record_telemetry_published(now_mono)
                payload_q = smoothed_q if obs_fresh else invalid_observation
                bridge.send_telemetry(
                    payload_q,
                    calibrator.metric_valid,
                    frame_valid=bool(obs_fresh and frame.size > 0),
                    fresh=obs_fresh,
                )

            # Interactive visualization
            if not args.no_show:
                draw_approach_corridor(frame, calibrator)
                for lid, lq in lane_queues.items():
                    for dist_m, pcu, tid, box, cat in vehicles_by_lane.get(lid, []):
                        draw_queued_vehicle(frame, box, lid, tid, cat, dist_m, tid in lq.track_ids, calibrator.is_calibrated)

                draw_heads_up_display(
                    frame,
                    frame_w,
                    args.approach_leg,
                    smoothed_q,
                    lane_queues,
                    t_src,
                    calibrator.is_calibrated,
                    args.allow_approx_calib,
                    bridge.is_connected,
                    bridge.controller_feedback,
                )

                cv2.imshow("ATCS Traffic Perception", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            ret, curr_frame = cap.read()

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        if cap is not None:
            cap.release()
        if bridge is not None:
            bridge.close()
        if not args.no_show:
            cv2.destroyAllWindows()


def main():
    args = parse_args()
    validate_args(args)
    run_perception(args)


if __name__ == "__main__":
    main()
