from collections import defaultdict
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock
import numpy as np

if "cv2" not in sys.modules:
    sys.modules["cv2"] = MagicMock()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "controller"))

from perception.telemetry import NonBlockingTelemetryBridge, serialize_observation_packet
from perception.traffic_analysis import (
    CalibrationStatus,
    CameraHealthState,
    CameraStreamHealthMonitor,
    LaneAwareQueueEstimator,
    RoadHomographyCalibrator,
    UpstreamArrivalTracker,
    VehicleTrackState,
    resolve_pcu,
)
from adaptive_controller import AdaptiveTrafficController, SignalState
from telemetry_schema import parse_and_validate_telemetry


def _valid_calib():
    return {
        "image_width": 1280,
        "image_height": 720,
        "image_points": [[450.0, 180.0], [830.0, 180.0], [1150.0, 600.0], [130.0, 600.0]],
        "ground_points_metric": [[0.0, 50.0], [7.0, 50.0], [7.0, 0.0], [0.0, 0.0]],
    }


def _ground_to_pixel(cal: RoadHomographyCalibrator, x_m: float, dist_m: float):
    H_inv = np.linalg.inv(cal.H)
    pt = np.array([x_m, dist_m, 1.0], dtype=np.float64) @ H_inv.T
    return float(pt[0] / pt[2]), float(pt[1] / pt[2])


class TestPhase1EEndToEnd(unittest.TestCase):
    def test_end_to_end_pipeline_normal_and_failure_scenarios(self):
        clock = [0.0]
        cal = RoadHomographyCalibrator(1280, 720, num_lanes=2)
        self.assertTrue(cal.validate_calibration_dict(_valid_calib()))
        est = LaneAwareQueueEstimator(max_vehicle_gap_m=6.0, stopline_threshold_m=14.0, history_size=3)
        arr_tracker = UpstreamArrivalTracker(window_duration_s=10.0, corridor_width_m=cal.total_width_m)
        health = CameraStreamHealthMonitor(
            stale_timeout_s=2.0,
            frozen_frame_threshold=4,
            min_freeze_duration_s=0.3,
            disconnect_failure_threshold=3,
            recovery_frames_required=2,
            clock_fn=lambda: clock[0],
        )
        ctrl = AdaptiveTrafficController(startup_all_red_s=2.0)
        tracks = {}
        rng = np.random.default_rng(7)

        def step_frame(detections_ground):
            clock[0] += 0.1
            frame = rng.integers(20, 220, size=(64, 64, 3), dtype=np.uint8)
            health.record_frame_acquired(frame)
            if health.consume_tracker_reset_needed():
                tracks.clear()
                arr_tracker.reset()
                est.history.clear()

            vehicles_by_lane = defaultdict(list)
            obs_list = []
            for tid, cls_name, x_m, d_m in detections_ground:
                u, v = _ground_to_pixel(cal, x_m, d_m)
                px_m, pdist_m, lane_id = cal.contact_point_to_ground(u, v)
                pcu, cat = resolve_pcu(cls_name, True)
                if lane_id < 0 or pcu is None or cat is None:
                    continue
                if tid not in tracks:
                    tracks[tid] = VehicleTrackState(tid, cat, pcu)
                else:
                    tracks[tid].update_class(cat, pcu)
                is_q, _ = tracks[tid].update_position(clock[0], px_m, pdist_m)
                obs_list.append((tid, px_m, pdist_m, tracks[tid].pcu, tracks[tid].class_name, is_q))
                if is_q:
                    vehicles_by_lane[lane_id].append((pdist_m, tracks[tid].pcu, tid, (0, 0, 10, 10), tracks[tid].class_name))

            health.record_frame_processed(True)
            fresh = health.is_observation_fresh()
            rate = arr_tracker.update(clock[0], obs_list) if (fresh and cal.metric_valid) else 0.0
            lqs = est.estimate(vehicles_by_lane, [0, 1])
            q_stop = sum(lq.count for lq in lqs.values())
            pcu_stop = sum(lq.pcu for lq in lqs.values())
            tails = [lq.tail_m for lq in lqs.values() if lq.count > 0]
            tail = max(tails) if tails else 0.0
            q_up = sum(lq.upstream_count for lq in lqs.values())
            pcu_up = sum(lq.upstream_pcu for lq in lqs.values())
            smoothed = est.smooth(
                q_stop, pcu_stop, tail, q_up, pcu_up, rate,
                metric_valid=cal.metric_valid,
                calibration_status=cal.status.value,
            )
            raw_pkt = serialize_observation_packet(
                "N", smoothed, is_calibrated=cal.metric_valid, frame_valid=fresh, fresh=fresh, seq=int(clock[0] * 10)
            )
            ok, parsed, err = parse_and_validate_telemetry(raw_pkt)
            self.assertTrue(ok, msg=err)
            ctrl.sim_time_s = clock[0]
            ctrl.update_approach(
                parsed["approach"], parsed["Q"], parsed["D_tail_m"], parsed["PCU"],
                is_valid=parsed["valid"], timestamp_s=clock[0], inflow_pcu=parsed["inflow_pcu"],
            )
            return parsed

        # 1 & 2. Normal moving traffic across many frames -> counted once, queue=0
        for step in range(15):
            parsed = step_frame([(101, "car", 1.75, 42.0 - step * 1.2)])
        self.assertEqual(arr_tracker.total_unique_arrivals, 1)
        self.assertAlmostEqual(parsed["arrival_rate_pcu_s"], 0.1, places=3)
        self.assertEqual(parsed["Q"], 0)
        self.assertTrue(parsed["valid"])

        # 3. Vehicle reaches 6.0m and stops (EMA decay + 1.0s dwell + median smoothing) -> Queue and arrival measurements remain distinct
        for _ in range(35):
            parsed = step_frame([(101, "car", 1.75, 6.0)])
        self.assertEqual(arr_tracker.total_unique_arrivals, 1)  # Still 1 unique arrival
        self.assertEqual(parsed["Q"], 1)
        self.assertAlmostEqual(parsed["PCU"], 1.0)
        self.assertAlmostEqual(parsed["D_tail_m"], 6.0, places=1)
        self.assertEqual(parsed["inflow_pcu"], 0.0)

        # 4. Camera returns identical frames -> Stream marked frozen -> telemetry valid=False, fresh=False
        frozen_img = np.full((64, 64, 3), 80, dtype=np.uint8)
        for _ in range(5):
            clock[0] += 0.1
            health.record_frame_acquired(frozen_img.copy())
            health.record_frame_processed(True)
        self.assertTrue(health.is_frozen)
        self.assertEqual(health.state, CameraHealthState.STALE)
        raw_frozen = serialize_observation_packet(
            "N", {"Q_stopline": 5, "PCU_stopline": 5.0, "D_tail_m": 20.0},
            is_calibrated=True, frame_valid=True, fresh=health.is_observation_fresh(), seq=99,
        )
        ok, parsed_frozen, _ = parse_and_validate_telemetry(raw_frozen)
        self.assertTrue(ok)
        self.assertFalse(parsed_frozen["valid"])
        self.assertFalse(parsed_frozen["fresh"])
        self.assertEqual(parsed_frozen["PCU"], 0.0)

        # 5 & 6. Camera disconnects and reconnects -> requires 2 fresh frames before recovery
        for _ in range(3):
            clock[0] += 0.1
            health.record_read_failure()
        self.assertEqual(health.state, CameraHealthState.DISCONNECTED)
        clock[0] += 0.1
        health.record_frame_acquired(rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8))
        health.record_frame_processed(True)
        self.assertEqual(health.state, CameraHealthState.RECOVERING)
        self.assertFalse(health.is_observation_fresh())
        clock[0] += 0.1
        health.record_frame_acquired(rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8))
        health.record_frame_processed(True)
        self.assertEqual(health.state, CameraHealthState.HEALTHY)
        self.assertTrue(health.is_observation_fresh())

        # 7. Invalid calibration -> metric measurements marked invalid
        cal.invalidate("Bad calibration")
        parsed_uncal = step_frame([(102, "car", 1.75, 6.0)])
        self.assertFalse(parsed_uncal["calibrated"])
        self.assertFalse(parsed_uncal["valid"])
        self.assertEqual(parsed_uncal["D_tail_m"], 0.0)

        # 8, 9, 10, 11. Partial and full sensor invalidity fallback behaviour on controller
        ctrl2 = AdaptiveTrafficController()
        ctrl2.sim_time_s = 10.0
        ctrl2.update_approach("N", 6, 20.0, 6.0, is_valid=True, timestamp_s=10.0)
        ctrl2.update_approach("S", 0, 0.0, 0.0, is_valid=False, timestamp_s=10.0)
        ns_pcu, ns_measured = ctrl2.get_phase_demand("N", "S")
        # N (6.0 measured) + S (2.0 fallback) = 8.0 PCU, partially measured (ns_measured=False)
        self.assertAlmostEqual(ns_pcu, 8.0)
        self.assertFalse(ns_measured)


if __name__ == "__main__":
    unittest.main()
