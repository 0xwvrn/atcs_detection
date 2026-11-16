import itertools
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock
import numpy as np

try:
    import cv2
    REAL_CV2_AVAILABLE = hasattr(cv2, "VideoCapture") and not isinstance(cv2, MagicMock)
except ImportError:
    cv2 = MagicMock()
    sys.modules["cv2"] = cv2
    REAL_CV2_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "controller"))

from perception.telemetry import serialize_observation_packet
from perception.traffic_analysis import (
    CalibrationStatus,
    CameraHealthState,
    CameraStreamHealthMonitor,
    LaneAwareQueueEstimator,
    RoadHomographyCalibrator,
    UpstreamArrivalTracker,
    VehicleTrackState,
)
from adaptive_controller import AdaptiveTrafficController, SignalState
from telemetry_schema import (
    MAX_PAYLOAD_BYTES,
    build_telemetry_packet,
    parse_and_validate_telemetry,
)


class TestPhase1FIndependentVerification(unittest.TestCase):
    def test_1_all_8_combinations_of_valid_calibrated_fresh(self):
        """Section 1: Exhaustively tests all 8 (valid, calibrated, fresh) combinations."""
        smoothed_q = {
            "Q_stopline": 7,
            "PCU_stopline": 8.5,
            "D_tail_m": 31.2,
            "Q_upstream": 2,
            "PCU_upstream": 3.0,
            "inflow_pcu": 3.0,
            "arrival_rate_pcu_s": 0.45,
        }

        for valid_flag, cal_flag, fresh_flag in itertools.product([True, False], repeat=3):
            with self.subTest(valid=valid_flag, calibrated=cal_flag, fresh=fresh_flag):
                raw = build_telemetry_packet(
                    approach="N",
                    q=7,
                    pcu=8.5,
                    d_tail_m=31.2,
                    valid=valid_flag,
                    inflow_pcu=3.0,
                    pcu_upstream=3.0,
                    arrival_rate_pcu_s=0.45,
                    calibrated=cal_flag,
                    fresh=fresh_flag,
                    seq=10,
                )
                ok, parsed, err = parse_and_validate_telemetry(raw)
                self.assertTrue(ok, msg=err)

                expected_effective_valid = bool(valid_flag and fresh_flag)
                self.assertEqual(parsed["valid"], expected_effective_valid)
                self.assertEqual(parsed["calibrated"], cal_flag)
                self.assertEqual(parsed["fresh"], fresh_flag)

                if not expected_effective_valid:
                    self.assertEqual(parsed["Q"], 0)
                    self.assertEqual(parsed["PCU"], 0.0)
                    self.assertEqual(parsed["D_tail_m"], 0.0)
                    self.assertEqual(parsed["pcu_upstream"], 0.0)
                    self.assertEqual(parsed["inflow_pcu"], 0.0)
                    self.assertEqual(parsed["arrival_rate_pcu_s"], 0.0)
                elif not cal_flag:
                    # Approximate image-space queue counts preserved, metric-dependent fields strictly 0.0
                    self.assertEqual(parsed["Q"], 7)
                    self.assertAlmostEqual(parsed["PCU"], 8.5)
                    self.assertEqual(parsed["D_tail_m"], 0.0)
                    self.assertEqual(parsed["pcu_upstream"], 0.0)
                    self.assertEqual(parsed["inflow_pcu"], 0.0)
                    self.assertEqual(parsed["arrival_rate_pcu_s"], 0.0)
                else:
                    # Fully valid, fresh, and calibrated
                    self.assertEqual(parsed["Q"], 7)
                    self.assertAlmostEqual(parsed["PCU"], 8.5)
                    self.assertAlmostEqual(parsed["D_tail_m"], 31.2)
                    self.assertAlmostEqual(parsed["pcu_upstream"], 3.0)
                    self.assertAlmostEqual(parsed["inflow_pcu"], 3.0)
                    self.assertAlmostEqual(parsed["arrival_rate_pcu_s"], 0.45)

                # Verify controller demand ingestion never uses invalid observations or uncalibrated metric inflow
                ctrl = AdaptiveTrafficController(enable_moderato=True)
                ctrl.sim_time_s = 5.0
                ctrl.update_approach("N", 4, 15.0, 4.0, is_valid=True, timestamp_s=5.0, inflow_pcu=1.5)
                ctrl.update_approach(
                    parsed["approach"],
                    parsed["Q"],
                    parsed["D_tail_m"],
                    parsed["PCU"],
                    is_valid=parsed["valid"],
                    timestamp_s=6.0,
                    inflow_pcu=parsed["inflow_pcu"],
                )
                if not expected_effective_valid:
                    self.assertFalse(ctrl.approaches["N"].is_valid)
                    pcu_val, measured = ctrl.get_approach_demand("N")
                    self.assertFalse(measured)
                    self.assertEqual(pcu_val, ctrl.nominal_fallback_pcu)
                    self.assertEqual(ctrl.get_approach_inflow("N"), 0.0)
                elif not cal_flag:
                    self.assertTrue(ctrl.approaches["N"].is_valid)
                    self.assertEqual(ctrl.get_approach_inflow("N"), 0.0)
                    self.assertEqual(ctrl.approaches["N"].d_tail_m, 0.0)

    def test_2_arrival_tracking_simultaneous_multi_vehicle_reacquisition_and_window_boundary(self):
        """Section 2: Simultaneous arrivals, upstream-following vehicle non-suppression, prolonged occlusion, and exact window boundary."""
        tracker = UpstreamArrivalTracker(
            window_duration_s=10.0,
            zone_min_m=18.0,
            zone_max_m=48.0,
            reacquire_window_s=2.0,
            reacquire_radius_m=4.0,
            id_reuse_jump_m=10.0,
        )

        # A. 4 vehicles enter in the exact same frames at t=0.0 -> t=0.5
        tracker.update(0.0, [
            (1, 1.5, 45.0, 1.0, "CAR", False),
            (2, 5.0, 44.0, 2.5, "BUS/TRUCK", False),
            (3, 1.5, 38.0, 1.2, "AUTO_RICKSHAW", False),
            (4, 5.0, 35.0, 0.5, "TWO_WHEELER", False),
        ])
        rate = tracker.update(0.5, [
            (1, 1.5, 43.0, 1.0, "CAR", False),
            (2, 5.0, 42.0, 2.5, "BUS/TRUCK", False),
            (3, 1.5, 36.0, 1.2, "AUTO_RICKSHAW", False),
            (4, 5.0, 33.0, 0.5, "TWO_WHEELER", False),
        ])
        self.assertEqual(tracker.total_unique_arrivals, 4)
        self.assertAlmostEqual(rate, (1.0 + 2.5 + 1.2 + 0.5) / 10.0, places=3)

        # B. Vehicle 1 (at 43.0m) disappears at t=0.5, and a brand-new Vehicle 5 enters UPSTREAM at 45.5m at t=0.7.
        # Vehicle 5 must NOT be falsely merged into Vehicle 1 (since 45.5m > 43.0m + 0.5m).
        tracker.update(0.7, [(5, 1.5, 45.5, 1.0, "CAR", False)])
        tracker.update(1.0, [(5, 1.5, 44.0, 1.0, "CAR", False)])
        self.assertEqual(tracker.total_unique_arrivals, 5)

        # C. Exact rolling-window boundary: events at t=0.5 expire at t=10.5 (since 0.5 <= 10.5 - 10.0),
        # leaving only Vehicle 5 (counted at t=1.0, 1.0 PCU -> rate = 0.1 PCU/s).
        rate_at_10_5 = tracker.update(10.5, [])
        self.assertAlmostEqual(rate_at_10_5, 1.0 / 10.0, places=3)

        # D. Prolonged occlusion (> reacquire_window_s = 2.0s): documented limitation verified
        tracker.update(11.0, [(6, 1.5, 42.0, 1.0, "CAR", False)])
        tracker.update(11.5, [(6, 1.5, 40.0, 1.0, "CAR", False)])
        self.assertEqual(tracker.total_unique_arrivals, 6)
        # Disappears for 2.5s (> 2.0s reacquire_window_s) and reappears with new ID 77
        tracker.update(14.0, [(77, 1.5, 32.0, 1.0, "CAR", False)])
        tracker.update(14.5, [(77, 1.5, 30.0, 1.0, "CAR", False)])
        self.assertEqual(tracker.total_unique_arrivals, 7)

    def test_3_camera_cold_start_and_real_recorded_detrac_stream_recovery(self):
        """Section 3: Cold start is immediately HEALTHY, and real DETRAC image stream recovery resets state."""
        clock = [0.0]
        mon = CameraStreamHealthMonitor(
            stale_timeout_s=2.0,
            frozen_frame_threshold=4,
            min_freeze_duration_s=0.3,
            disconnect_failure_threshold=3,
            recovery_frames_required=2,
            reconnect_interval_s=1.0,
            clock_fn=lambda: clock[0],
        )

        # Load real recorded UA-DETRAC frames if present on disk
        detrac_dir = Path("/Users/dhritiman/Documents/dev/atcs/data/DETRAC/DETRAC-Images/MVI_20011")
        frames = []
        if REAL_CV2_AVAILABLE and detrac_dir.exists():
            for idx in range(1, 10):
                img_path = detrac_dir / f"img{idx:05d}.jpg"
                if img_path.exists():
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        frames.append(img)

        if len(frames) < 6:
            rng = np.random.default_rng(99)
            frames = [rng.integers(0, 255, size=(120, 160, 3), dtype=np.uint8) for _ in range(8)]

        # 1. Cold start: Frame 1 immediately transitions to HEALTHY (fresh=True on Frame 1)
        clock[0] = 1.0
        st1 = mon.record_frame_acquired(frames[0])
        mon.record_frame_processed(True)
        self.assertEqual(st1, CameraHealthState.HEALTHY)
        self.assertTrue(mon.is_observation_fresh())

        # 2. Simulate 3 consecutive read failures -> DISCONNECTED
        for _ in range(3):
            clock[0] += 0.1
            mon.record_read_failure()
        self.assertEqual(mon.state, CameraHealthState.DISCONNECTED)
        self.assertFalse(mon.is_observation_fresh())
        self.assertTrue(mon.should_attempt_reconnect())

        # 3. Reconnect and feed real frame 1 -> RECOVERING (tracker reset triggered, not yet fresh)
        mon.mark_reconnect_attempt()
        self.assertTrue(mon.consume_tracker_reset_needed())
        clock[0] += 0.1
        mon.record_frame_acquired(frames[1])
        mon.record_frame_processed(True)
        self.assertEqual(mon.state, CameraHealthState.RECOVERING)
        self.assertFalse(mon.is_observation_fresh())

        # 4. Feed real frame 2 -> HEALTHY (fresh=True)
        clock[0] += 0.1
        mon.record_frame_acquired(frames[2])
        mon.record_frame_processed(True)
        self.assertEqual(mon.state, CameraHealthState.HEALTHY)
        self.assertTrue(mon.is_observation_fresh())

    def test_4_pre_vs_post_phase1_controller_scheduling_equivalence(self):
        """Section 5: Confirms new Phase 1 telemetry fields do NOT alter existing queue-based controller scheduling."""
        for enable_mod in (False, True):
            ctrl_legacy = AdaptiveTrafficController(enable_moderato=enable_mod, timing_profile="URBAN_ARTERIAL")
            ctrl_phase1 = AdaptiveTrafficController(enable_moderato=enable_mod, timing_profile="URBAN_ARTERIAL")

            # Simulate 300.0s (3000 ticks at dt=0.1s) with identical traffic demand waves
            for step in range(3000):
                t = round(step * 0.1, 2)
                if step % 10 == 0:  # 1 Hz telemetry updates
                    ns_q = 12 if (step < 1000) else (0 if step < 1500 else 22)
                    ew_q = 4 if (step < 1000) else (15 if step < 2200 else 25)
                    inflow_n = 3.0 if step >= 1500 else 0.0

                    for leg, q in (("N", ns_q), ("S", ns_q), ("E", ew_q), ("W", ew_q)):
                        inf_val = inflow_n if leg in ("N", "S") else 0.0
                        # Legacy 5-field packet
                        legacy_raw = json.dumps({
                            "approach": leg,
                            "Q": q,
                            "PCU": float(q),
                            "D_tail_m": float(q * 3.5),
                            "inflow_pcu": inf_val,
                            "valid": True,
                        })
                        # Phase 1 Canonical v1 packet (with v, pcu_upstream, arrival_rate_pcu_s, calibrated, fresh, seq)
                        phase1_raw = build_telemetry_packet(
                            approach=leg,
                            q=q,
                            pcu=float(q),
                            d_tail_m=float(q * 3.5),
                            valid=True,
                            inflow_pcu=inf_val,
                            pcu_upstream=inf_val,
                            arrival_rate_pcu_s=0.48,  # Non-zero arrival rate must NOT alter scheduling
                            calibrated=True,
                            fresh=True,
                            seq=step,
                        )
                        ok1, p1, _ = parse_and_validate_telemetry(legacy_raw)
                        ok2, p2, _ = parse_and_validate_telemetry(phase1_raw)
                        self.assertTrue(ok1 and ok2)

                        ctrl_legacy.update_approach(
                            p1["approach"], p1["Q"], p1["D_tail_m"], p1["PCU"],
                            is_valid=p1["valid"], timestamp_s=t, inflow_pcu=p1["inflow_pcu"],
                        )
                        ctrl_phase1.update_approach(
                            p2["approach"], p2["Q"], p2["D_tail_m"], p2["PCU"],
                            is_valid=p2["valid"], timestamp_s=t, inflow_pcu=p2["inflow_pcu"],
                        )

                ctrl_legacy.tick(dt=0.1)
                ctrl_phase1.tick(dt=0.1)

                self.assertEqual(ctrl_legacy.state, ctrl_phase1.state)
                self.assertAlmostEqual(ctrl_legacy.allocated_green_s, ctrl_phase1.allocated_green_s, places=5)
                self.assertAlmostEqual(ctrl_legacy.ns_wait_time_s, ctrl_phase1.ns_wait_time_s, places=5)
                self.assertAlmostEqual(ctrl_legacy.ew_wait_time_s, ctrl_phase1.ew_wait_time_s, places=5)

            self.assertEqual(ctrl_legacy.transition_log, ctrl_phase1.transition_log)

    def test_5_serial_dependency_handling_explicit_vs_standalone(self):
        """Stage 1: Explicit --serial-port fails clearly without pyserial; standalone mode is preserved."""
        from types import SimpleNamespace
        from unittest.mock import patch
        import perception.pipeline as pipeline_mod

        with patch.object(pipeline_mod, "SERIAL_AVAILABLE", False):
            args_explicit = SimpleNamespace(serial_port="/dev/cu.usbmodem5B900309611", standalone=False)
            with self.assertRaises(RuntimeError) as ctx:
                pipeline_mod.resolve_serial_mode(args_explicit, argv=["--serial-port", "/dev/cu.usbmodem5B900309611"])
            self.assertIn("pyserial", str(ctx.exception).lower())

            args_standalone = SimpleNamespace(serial_port="/dev/cu.usbmodem5B900309611", standalone=True)
            is_sa, label = pipeline_mod.resolve_serial_mode(args_standalone, argv=["--standalone"])
            self.assertTrue(is_sa)
            self.assertEqual(label, "STANDALONE")

            args_default = SimpleNamespace(serial_port="/dev/cu.usbmodem5B900309611", standalone=False)
            is_sa2, label2 = pipeline_mod.resolve_serial_mode(args_default, argv=["--source", "0"])
            self.assertTrue(is_sa2)
            self.assertIn("STANDALONE", label2)


if __name__ == "__main__":
    unittest.main()
