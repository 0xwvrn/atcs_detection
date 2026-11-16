from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock

if "cv2" not in sys.modules:
    sys.modules["cv2"] = MagicMock()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perception import tracking, traffic_analysis


class TestVehicleTracking(unittest.TestCase):
    def test_backward_compatible_reexport(self):
        self.assertIs(traffic_analysis.VehicleTrackState, tracking.VehicleTrackState)

    def test_class_majority_voting_and_pcu_consistency(self):
        t = tracking.VehicleTrackState(7, "CAR", 1.0)
        self.assertEqual((t.class_name, t.pcu), ("CAR", 1.0))

        # Single noisy VAN detection does not flip majority (1 CAR vs 1 VAN -> CAR inserted first)
        t.update_class("VAN", 1.4)
        self.assertEqual((t.class_name, t.pcu), ("CAR", 1.0))

        # Consistent BUS/TRUCK detections override initial CAR vote
        t.update_class("BUS/TRUCK", 2.5)
        t.update_class("BUS/TRUCK", 2.5)
        self.assertEqual((t.class_name, t.pcu), ("BUS/TRUCK", 2.5))

    def test_hysteresis_gap_guard_and_queue_exit(self):
        t = tracking.VehicleTrackState(12, "CAR", 1.0)
        # Moving toward stopline at 4.0 m/s -> not queued
        for i in range(6):
            is_q, spd = t.update_position(i * 0.2, 1.75, 30.0 - i * 0.8)
            self.assertFalse(is_q)
        self.assertGreater(spd, 0.8)

        # Stops at 20.0m (within 8cm jitter deadband) -> enters queue after EMA <= 0.4 and 1.0s dwell
        ts = 1.2
        while not t.is_queued and ts < 6.0:
            is_q, _ = t.update_position(ts, 1.75, 20.02)
            ts += 0.2
        self.assertTrue(t.is_queued)

        # Brief 2.0s detector dropout (> 1.5s gap guard) preserves queued status
        is_q_after_drop, spd_after_drop = t.update_position(ts + 2.0, 1.75, 20.0)
        self.assertTrue(is_q_after_drop)
        self.assertIsNone(spd_after_drop)

        # Resumes moving at >= 0.8 m/s -> exits queue
        is_q_exit, spd_exit = t.update_position(ts + 2.5, 1.75, 18.5)
        self.assertFalse(is_q_exit)
        self.assertGreaterEqual(spd_exit, 0.8)

    def test_non_monotonic_and_duplicate_timestamps_ignored(self):
        t = tracking.VehicleTrackState(21, "CAR", 1.0)
        t.update_position(1.0, 1.75, 30.0)
        is_q, spd = t.update_position(1.2, 1.75, 29.2)  # 4.0 m/s
        self.assertFalse(is_q)
        self.assertAlmostEqual(spd, 4.0, places=5)
        saved_obs = t.prev_obs
        saved_seen = t.last_seen_time

        # Duplicate timestamp (dt == 0.0) must not overwrite prev_obs or last_seen_time
        is_q_dup, spd_dup = t.update_position(1.2, 5.0, 10.0)
        self.assertFalse(is_q_dup)
        self.assertAlmostEqual(spd_dup, 4.0, places=5)
        self.assertEqual(t.prev_obs, saved_obs)
        self.assertEqual(t.last_seen_time, saved_seen)

        # Backwards timestamp (dt < 0.0) must not overwrite prev_obs or last_seen_time
        is_q_back, spd_back = t.update_position(0.8, 0.0, 40.0)
        self.assertFalse(is_q_back)
        self.assertAlmostEqual(spd_back, 4.0, places=5)
        self.assertEqual(t.prev_obs, saved_obs)
        self.assertEqual(t.last_seen_time, saved_seen)

        # Next valid monotonic timestamp computes speed against preserved 1.2s observation
        is_q_next, spd_next = t.update_position(1.4, 1.75, 28.4)  # 0.8m / 0.2s = 4.0 m/s
        self.assertFalse(is_q_next)
        self.assertAlmostEqual(spd_next, 0.25 * 4.0 + 0.75 * 4.0, places=5)
        self.assertEqual(t.last_seen_time, 1.4)
        self.assertEqual(t.prev_obs, (1.4, 1.75, 28.4))

    def test_speed_ema_exact_calculation(self):
        t = tracking.VehicleTrackState(22, "CAR", 1.0)
        t.update_position(0.0, 1.0, 30.0)
        _, spd1 = t.update_position(0.5, 1.0, 28.0)  # 2.0m / 0.5s = 4.0 m/s
        self.assertAlmostEqual(spd1, 4.0, places=5)
        _, spd2 = t.update_position(1.0, 1.0, 27.0)  # 1.0m / 0.5s = 2.0 m/s -> EMA: 0.25*2 + 0.75*4 = 3.5
        self.assertAlmostEqual(spd2, 3.5, places=5)


if __name__ == "__main__":
    unittest.main()
