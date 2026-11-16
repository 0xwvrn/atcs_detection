from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock
import numpy as np

if "cv2" not in sys.modules:
    sys.modules["cv2"] = MagicMock()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perception.traffic_analysis import (
    CameraHealthState,
    CameraStreamHealthMonitor,
)


class TestPhase1CCameraHealth(unittest.TestCase):
    def test_1_repeated_identical_frames_trigger_frozen_and_stale(self):
        clock = [100.0]
        mon = CameraStreamHealthMonitor(
            stale_timeout_s=2.0,
            frozen_frame_threshold=5,
            min_freeze_duration_s=0.4,
            recovery_frames_required=2,
            clock_fn=lambda: clock[0],
        )
        base_frame = np.full((64, 64, 3), 120, dtype=np.uint8)
        mon.record_frame_acquired(base_frame)
        mon.record_frame_processed(True)
        # Second distinct frame completes initial recovery from DISCONNECTED -> HEALTHY
        frame2 = base_frame.copy()
        frame2[0, 0, 0] = 200
        clock[0] += 0.1
        mon.record_frame_acquired(frame2)
        mon.record_frame_processed(True)
        self.assertEqual(mon.state, CameraHealthState.HEALTHY)
        self.assertTrue(mon.is_observation_fresh())

        # Now feed identical frame2 5 times over 0.5s
        for _ in range(5):
            clock[0] += 0.1
            mon.record_frame_acquired(frame2.copy())
            mon.record_frame_processed(True)

        self.assertTrue(mon.is_frozen)
        self.assertEqual(mon.state, CameraHealthState.STALE)
        self.assertFalse(mon.is_observation_fresh())
        self.assertTrue(mon.consume_tracker_reset_needed())

    def test_2_slightly_changed_frames_and_static_road_with_sensor_noise_stay_healthy(self):
        clock = [10.0]
        mon = CameraStreamHealthMonitor(
            frozen_frame_threshold=5,
            min_freeze_duration_s=0.2,
            freeze_mad_threshold=0.02,
            recovery_frames_required=1,
            clock_fn=lambda: clock[0],
        )
        rng = np.random.default_rng(42)
        base = np.full((64, 64, 3), 90, dtype=np.uint8)
        for _ in range(20):
            clock[0] += 0.1
            # Simulate empty road with natural +/-1 LSB camera thermal noise
            noisy = np.clip(base.astype(np.int16) + rng.integers(-1, 2, size=base.shape), 0, 255).astype(np.uint8)
            mon.record_frame_acquired(noisy)
            mon.record_frame_processed(True)

        self.assertFalse(mon.is_frozen)
        self.assertEqual(mon.state, CameraHealthState.HEALTHY)
        self.assertTrue(mon.is_observation_fresh())

    def test_3_completely_static_frames_with_advancing_hw_sequence_and_slight_diff(self):
        clock = [50.0]
        mon = CameraStreamHealthMonitor(
            frozen_frame_threshold=4,
            min_freeze_duration_s=0.2,
            recovery_frames_required=1,
            clock_fn=lambda: clock[0],
        )
        # Completely static frame where hardware sequence counter advances -> stays HEALTHY
        static_frame = np.full((64, 64, 3), 60, dtype=np.uint8)
        for seq in range(1, 12):
            clock[0] += 0.1
            mon.record_frame_acquired(static_frame.copy(), hw_frame_seq=seq)
            mon.record_frame_processed(True)
        self.assertFalse(mon.is_frozen)
        self.assertEqual(mon.state, CameraHealthState.HEALTHY)

        # If hardware sequence stops advancing (stalled at 11 for 4 frames), detect frozen stream
        for _ in range(4):
            clock[0] += 0.1
            mon.record_frame_acquired(static_frame.copy(), hw_frame_seq=11)
            mon.record_frame_processed(True)
        self.assertTrue(mon.is_frozen)
        self.assertEqual(mon.state, CameraHealthState.STALE)

    def test_4_intermittent_frame_updates_and_sudden_recovery_after_freezing(self):
        clock = [0.0]
        mon = CameraStreamHealthMonitor(
            frozen_frame_threshold=4,
            min_freeze_duration_s=0.3,
            recovery_frames_required=2,
            clock_fn=lambda: clock[0],
        )
        f1 = np.zeros((64, 64, 3), dtype=np.uint8)
        f2 = np.full((64, 64, 3), 50, dtype=np.uint8)
        mon.record_frame_acquired(f1)
        clock[0] += 0.1
        mon.record_frame_acquired(f2)
        mon.record_frame_processed(True)
        self.assertEqual(mon.state, CameraHealthState.HEALTHY)

        # Freeze on f2 for 4 frames
        for _ in range(4):
            clock[0] += 0.1
            mon.record_frame_acquired(f2.copy())
            mon.record_frame_processed(True)
        self.assertTrue(mon.is_frozen)
        self.assertEqual(mon.state, CameraHealthState.STALE)

        # Sudden recovery: 1st distinct frame -> RECOVERING (not yet fresh), 2nd distinct frame -> HEALTHY
        f3 = np.full((64, 64, 3), 100, dtype=np.uint8)
        clock[0] += 0.1
        mon.record_frame_acquired(f3)
        mon.record_frame_processed(True)
        self.assertFalse(mon.is_frozen)
        self.assertEqual(mon.state, CameraHealthState.RECOVERING)
        self.assertFalse(mon.is_observation_fresh())

        f4 = np.full((64, 64, 3), 150, dtype=np.uint8)
        clock[0] += 0.1
        mon.record_frame_acquired(f4)
        mon.record_frame_processed(True)
        self.assertEqual(mon.state, CameraHealthState.HEALTHY)
        self.assertTrue(mon.is_observation_fresh())

    def test_5_disconnection_bounded_reconnection_and_stale_timeout(self):
        clock = [200.0]
        mon = CameraStreamHealthMonitor(
            stale_timeout_s=2.0,
            degraded_failure_threshold=2,
            disconnect_failure_threshold=4,
            recovery_frames_required=2,
            reconnect_interval_s=2.0,
            max_reconnect_attempts=3,
            clock_fn=lambda: clock[0],
        )
        f1 = np.full((64, 64, 3), 10, dtype=np.uint8)
        f2 = np.full((64, 64, 3), 40, dtype=np.uint8)
        mon.record_frame_acquired(f1)
        clock[0] += 0.1
        mon.record_frame_acquired(f2)
        mon.record_frame_processed(True)
        self.assertEqual(mon.state, CameraHealthState.HEALTHY)

        # 2 read failures -> DEGRADED
        clock[0] += 0.1
        mon.record_read_failure()
        clock[0] += 0.1
        mon.record_read_failure()
        self.assertEqual(mon.state, CameraHealthState.DEGRADED)

        # 4 read failures -> DISCONNECTED
        clock[0] += 0.1
        mon.record_read_failure()
        clock[0] += 0.1
        mon.record_read_failure()
        self.assertEqual(mon.state, CameraHealthState.DISCONNECTED)
        self.assertFalse(mon.is_observation_fresh())
        self.assertTrue(mon.should_attempt_reconnect())

        # Bounded reconnect backoff check
        mon.mark_reconnect_attempt()
        self.assertEqual(mon.reconnect_attempts, 1)
        self.assertFalse(mon.should_attempt_reconnect())  # Interval 2.0s not elapsed yet
        clock[0] += 2.1
        self.assertTrue(mon.should_attempt_reconnect())


if __name__ == "__main__":
    unittest.main()
