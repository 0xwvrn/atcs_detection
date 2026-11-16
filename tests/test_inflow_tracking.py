import math
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock

if "cv2" not in sys.modules:
    sys.modules["cv2"] = MagicMock()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perception.traffic_analysis import (
    LaneAwareQueueEstimator,
    UpstreamArrivalTracker,
    VehicleTrackState,
)


class TestPhase1AInflowModel(unittest.TestCase):
    def test_1_single_vehicle_across_many_frames_counted_once(self):
        tracker = UpstreamArrivalTracker(window_duration_s=10.0, zone_min_m=18.0, zone_max_m=48.0)
        # Vehicle 101 travels from 45m down to 20m across 25 frames (0.1s step)
        for i in range(25):
            t = i * 0.1
            dist_m = 45.0 - i * 1.0
            rate = tracker.update(t, [(101, 1.75, dist_m, 1.0, "CAR", False)])

        self.assertEqual(tracker.total_unique_arrivals, 1)
        self.assertAlmostEqual(rate, 1.0 / 10.0, places=3)

    def test_2_multiple_distinct_vehicles_entering(self):
        tracker = UpstreamArrivalTracker(window_duration_s=10.0)
        for i in range(10):
            t = i * 0.2
            obs = [
                (1, 1.75, 44.0 - i * 1.5, 1.0, "CAR", False),
                (2, 5.25, 42.0 - i * 1.5, 1.0, "CAR", False),
                (3, 1.75, 38.0 - i * 1.5, 1.0, "CAR", False),
            ]
            rate = tracker.update(t, obs)

        self.assertEqual(tracker.total_unique_arrivals, 3)
        self.assertAlmostEqual(rate, 3.0 / 10.0, places=3)

    def test_3_different_vehicle_classes_and_pcu_weights(self):
        tracker = UpstreamArrivalTracker(window_duration_s=10.0)
        # Bus (2.5) + Auto (1.2) + Two-Wheeler (0.5) = 4.2 PCU over 10s -> 0.42 PCU/s
        for i in range(6):
            t = i * 0.2
            obs = [
                (10, 1.75, 40.0 - i * 1.0, 2.5, "BUS/TRUCK", False),
                (11, 5.25, 35.0 - i * 1.0, 1.2, "AUTO_RICKSHAW", False),
                (12, 1.50, 30.0 - i * 1.0, 0.5, "TWO_WHEELER", False),
            ]
            rate = tracker.update(t, obs)

        self.assertEqual(tracker.total_unique_arrivals, 3)
        self.assertAlmostEqual(rate, 4.2 / 10.0, places=3)

    def test_4_vehicle_leaving_and_reentering_region_or_oscillating(self):
        tracker = UpstreamArrivalTracker(window_duration_s=10.0, zone_min_m=20.0, zone_max_m=45.0)
        # Enters at 44m -> 42m (counted), then jitter pushes it to 45.5m (outside) and back to 43.0m
        positions = [44.5, 43.8, 43.0, 45.5, 44.2, 43.5, 19.5, 20.4, 19.2]
        for i, d in enumerate(positions):
            rate = tracker.update(i * 0.2, [(77, 2.0, d, 1.0, "CAR", False)])

        self.assertEqual(tracker.total_unique_arrivals, 1)
        self.assertAlmostEqual(rate, 0.1, places=3)

    def test_5_tracker_id_reuse_for_new_upstream_vehicle(self):
        tracker = UpstreamArrivalTracker(window_duration_s=20.0, id_reuse_jump_m=10.0)
        # Vehicle with ID 5 moves from 40m down to 8m (near stopline)
        for i, d in enumerate([40.0, 38.0, 30.0, 20.0, 12.0, 8.0]):
            tracker.update(i * 0.5, [(5, 1.75, d, 1.0, "CAR", False)])
        self.assertEqual(tracker.total_unique_arrivals, 1)

        # Tracker reuses ID 5 for a brand new vehicle appearing at 44m and moving to 42m
        tracker.update(3.5, [(5, 1.75, 44.0, 2.5, "BUS/TRUCK", False)])
        rate = tracker.update(4.0, [(5, 1.75, 42.5, 2.5, "BUS/TRUCK", False)])
        self.assertEqual(tracker.total_unique_arrivals, 2)
        self.assertAlmostEqual(rate, (1.0 + 2.5) / 20.0, places=3)

    def test_6_track_loss_and_reacquisition_not_double_counted(self):
        tracker = UpstreamArrivalTracker(window_duration_s=10.0, reacquire_window_s=2.0, reacquire_radius_m=4.0)
        # Track 20 enters and is counted at 38m moving at ~5 m/s
        tracker.update(0.0, [(20, 2.0, 40.0, 1.0, "CAR", False)])
        tracker.update(0.2, [(20, 2.0, 39.0, 1.0, "CAR", False)])
        tracker.update(0.4, [(20, 2.0, 38.0, 1.0, "CAR", False)])
        self.assertEqual(tracker.total_unique_arrivals, 1)

        # Track 20 drops for 1 frame (t=0.6) and reappears as Track 99 at t=0.8 near 36.2m
        tracker.update(0.6, [])
        tracker.update(0.8, [(99, 2.1, 36.2, 1.0, "CAR", False)])
        rate = tracker.update(1.0, [(99, 2.1, 35.0, 1.0, "CAR", False)])

        self.assertEqual(tracker.total_unique_arrivals, 1)
        self.assertAlmostEqual(rate, 0.1, places=3)

    def test_7_empty_measurement_windows_and_decay(self):
        tracker = UpstreamArrivalTracker(window_duration_s=5.0)
        self.assertEqual(tracker.get_arrival_rate(), 0.0)
        tracker.update(0.0, [(1, 2.0, 40.0, 1.0, "CAR", False)])
        tracker.update(0.5, [(1, 2.0, 38.5, 1.0, "CAR", False)])
        self.assertAlmostEqual(tracker.get_arrival_rate(1.0), 0.2, places=3)

        # After window expires (t = 6.0 > 0.5 + 5.0), rate decays to 0.0
        rate_after = tracker.update(6.0, [])
        self.assertEqual(rate_after, 0.0)

    def test_8_irregular_frame_intervals_and_frame_drops(self):
        tracker = UpstreamArrivalTracker(window_duration_s=8.0)
        # Irregular frame timestamps: 0.0s, 0.03s, 0.45s, 1.90s
        timestamps = [0.0, 0.03, 0.45, 1.90]
        dists = [42.0, 41.8, 39.5, 31.0]
        for t, d in zip(timestamps, dists):
            rate = tracker.update(t, [(33, 2.0, d, 1.0, "CAR", False)])

        # Rate depends solely on window_duration_s (8.0s), never frame count
        self.assertEqual(tracker.total_unique_arrivals, 1)
        self.assertAlmostEqual(rate, 1.0 / 8.0, places=3)

    def test_9_invalid_and_non_monotonic_timestamps_and_backwards_motion(self):
        tracker = UpstreamArrivalTracker(window_duration_s=10.0)
        # Vehicle moving backwards (away from stopline: 30m -> 32m -> 35m) must NOT be counted
        tracker.update(1.0, [(50, 2.0, 30.0, 1.0, "CAR", False)])
        tracker.update(1.5, [(50, 2.0, 32.0, 1.0, "CAR", False)])
        tracker.update(2.0, [(50, 2.0, 35.0, 1.0, "CAR", False)])
        self.assertEqual(tracker.total_unique_arrivals, 0)

        # Non-monotonic (t=1.2 < 2.0) and non-finite timestamps ignored safely
        tracker.update(1.2, [(51, 2.0, 40.0, 1.0, "CAR", False)])
        tracker.update(float("nan"), [(51, 2.0, 38.0, 1.0, "CAR", False)])
        tracker.update(float("inf"), [(51, 2.0, 36.0, 1.0, "CAR", False)])
        self.assertEqual(tracker.total_unique_arrivals, 0)

        with self.assertRaises(ValueError):
            UpstreamArrivalTracker(window_duration_s=0.0)
        with self.assertRaises(ValueError):
            UpstreamArrivalTracker(window_duration_s=-5.0)

    def test_10_stopline_queue_exclusion_and_smooth_quantity_separation(self):
        tracker = UpstreamArrivalTracker(window_duration_s=10.0, stopline_threshold_m=14.0)
        # Vehicle already in queue at 10m creeping to 9m must not count as upstream arrival
        tracker.update(0.0, [(88, 2.0, 10.0, 1.0, "CAR", True)])
        tracker.update(0.5, [(88, 2.0, 9.0, 1.0, "CAR", False)])
        self.assertEqual(tracker.total_unique_arrivals, 0)

        estimator = LaneAwareQueueEstimator()
        res = estimator.smooth(
            q_stopline=4,
            pcu_stopline=4.5,
            tail_stopline_m=18.2,
            q_upstream=2,
            pcu_upstream=3.5,
            arrival_rate_pcu_s=0.42,
        )
        self.assertEqual(res["Q_stopline"], 4)
        self.assertEqual(res["PCU_stopline"], 4.5)
        self.assertEqual(res["PCU_upstream"], 3.5)
        self.assertEqual(res["inflow_pcu"], 3.5)  # Equals PCU_upstream, NOT mixed with arrival_rate_pcu_s
        self.assertEqual(res["arrival_rate_pcu_s"], 0.42)
        self.assertIsNone(res["predicted_demand_pcu"])

    def test_11_competing_reacquisition_candidates_selects_smallest_error(self):
        tracker = UpstreamArrivalTracker(window_duration_s=10.0, reacquire_window_s=2.0, reacquire_radius_m=4.0)
        # Track 10 at 40m -> 39m -> 38m (speed ~5 m/s) and Track 11 at 35m -> 34m -> 33m (speed ~5 m/s)
        for i in range(3):
            t = i * 0.2
            tracker.update(t, [
                (10, 2.0, 40.0 - i * 1.0, 1.0, "CAR", False),
                (11, 2.2, 35.0 - i * 1.0, 1.0, "CAR", False),
            ])
        self.assertEqual(tracker.total_unique_arrivals, 2)

        # Both drop at t=0.6; at t=0.8, Track 99 appears at 36.0m (matches Track 10's predicted 36.0m exactly vs 11's 31.0m)
        tracker.update(0.6, [])
        tracker.update(0.8, [(99, 2.0, 36.0, 1.0, "CAR", False)])
        tracker.update(1.0, [(99, 2.0, 35.0, 1.0, "CAR", False)])
        self.assertEqual(tracker.total_unique_arrivals, 2)

        # Track 11 should still be available for reacquisition at t=1.0, while Track 10 was already consumed
        tracker.update(1.2, [(100, 2.2, 29.0, 1.0, "CAR", False)])
        tracker.update(1.4, [(100, 2.2, 28.0, 1.0, "CAR", False)])
        self.assertEqual(tracker.total_unique_arrivals, 2)

    def test_12_reacquisition_time_boundary(self):
        # Exact 2.0s boundary is accepted
        tracker_ok = UpstreamArrivalTracker(window_duration_s=10.0, reacquire_window_s=2.0, reacquire_radius_m=4.0)
        tracker_ok.update(0.0, [(1, 2.0, 40.0, 1.0, "CAR", False)])
        tracker_ok.update(0.2, [(1, 2.0, 39.5, 1.0, "CAR", False)])
        self.assertEqual(tracker_ok.total_unique_arrivals, 1)
        tracker_ok.update(2.2, [(2, 2.0, 37.5, 1.0, "CAR", False)])
        tracker_ok.update(2.4, [(2, 2.0, 36.5, 1.0, "CAR", False)])
        self.assertEqual(tracker_ok.total_unique_arrivals, 1)

        # Beyond 2.0s (2.01s) is rejected and counted as a new arrival
        tracker_late = UpstreamArrivalTracker(window_duration_s=10.0, reacquire_window_s=2.0, reacquire_radius_m=4.0)
        tracker_late.update(0.0, [(1, 2.0, 40.0, 1.0, "CAR", False)])
        tracker_late.update(0.2, [(1, 2.0, 39.5, 1.0, "CAR", False)])
        self.assertEqual(tracker_late.total_unique_arrivals, 1)
        tracker_late.update(2.21, [(2, 2.0, 37.5, 1.0, "CAR", False)])
        tracker_late.update(2.41, [(2, 2.0, 36.5, 1.0, "CAR", False)])
        self.assertEqual(tracker_late.total_unique_arrivals, 2)

    def test_13_reacquisition_lateral_boundary(self):
        # Exact 2.5m lateral offset is accepted
        tracker_ok = UpstreamArrivalTracker(window_duration_s=10.0, reacquire_window_s=2.0, reacquire_radius_m=4.0)
        tracker_ok.update(0.0, [(1, 2.0, 40.0, 1.0, "CAR", False)])
        tracker_ok.update(0.2, [(1, 2.0, 39.0, 1.0, "CAR", False)])
        tracker_ok.update(0.4, [(2, 4.5, 38.0, 1.0, "CAR", False)])
        tracker_ok.update(0.6, [(2, 4.5, 37.0, 1.0, "CAR", False)])
        self.assertEqual(tracker_ok.total_unique_arrivals, 1)

        # Beyond 2.5m (2.51m) is rejected
        tracker_far = UpstreamArrivalTracker(window_duration_s=10.0, reacquire_window_s=2.0, reacquire_radius_m=4.0)
        tracker_far.update(0.0, [(1, 2.0, 40.0, 1.0, "CAR", False)])
        tracker_far.update(0.2, [(1, 2.0, 39.0, 1.0, "CAR", False)])
        tracker_far.update(0.4, [(2, 4.51, 38.0, 1.0, "CAR", False)])
        tracker_far.update(0.6, [(2, 4.51, 37.0, 1.0, "CAR", False)])
        self.assertEqual(tracker_far.total_unique_arrivals, 2)

    def test_14_reacquisition_class_and_pcu_mismatch_rejected(self):
        tracker = UpstreamArrivalTracker(window_duration_s=10.0, reacquire_window_s=2.0, reacquire_radius_m=4.0)
        tracker.update(0.0, [(1, 2.0, 40.0, 2.5, "BUS/TRUCK", False)])
        tracker.update(0.2, [(1, 2.0, 39.0, 2.5, "BUS/TRUCK", False)])
        self.assertEqual(tracker.total_unique_arrivals, 1)

        # Incompatible class/PCU (CAR 1.0 vs BUS/TRUCK 2.5) at exact predicted position must NOT reacquire
        tracker.update(0.4, [(2, 2.0, 38.0, 1.0, "CAR", False)])
        tracker.update(0.6, [(2, 2.0, 37.0, 1.0, "CAR", False)])
        self.assertEqual(tracker.total_unique_arrivals, 2)

    def test_15_reset_clears_all_tracker_state(self):
        tracker = UpstreamArrivalTracker(window_duration_s=10.0)
        tracker.update(0.0, [(1, 2.0, 40.0, 1.0, "CAR", False)])
        tracker.update(0.5, [(1, 2.0, 38.0, 1.0, "CAR", False)])
        self.assertEqual(tracker.total_unique_arrivals, 1)
        self.assertGreater(tracker.get_arrival_rate(), 0.0)

        tracker.reset()
        self.assertEqual(tracker.active_tracks, {})
        self.assertEqual(len(tracker.arrival_events), 0)
        self.assertEqual(tracker.total_unique_arrivals, 0)
        self.assertIsNone(tracker.last_timestamp_s)
        self.assertEqual(tracker.get_arrival_rate(), 0.0)

    def test_16_track_ttl_expiry_and_same_id_reuse(self):
        tracker = UpstreamArrivalTracker(window_duration_s=20.0, track_ttl_s=5.0, id_reuse_jump_m=10.0)
        tracker.update(0.0, [(7, 2.0, 40.0, 1.0, "CAR", False)])
        tracker.update(0.5, [(7, 2.0, 38.0, 1.0, "CAR", False)])
        self.assertEqual(tracker.total_unique_arrivals, 1)

        # Exceed 5.0s TTL (0.5 -> 5.6, dt=5.1s) and reuse same ID 7 without upstream jump >= 10m
        tracker.update(5.6, [(7, 2.0, 37.0, 1.0, "CAR", False)])
        tracker.update(6.0, [(7, 2.0, 35.5, 1.0, "CAR", False)])
        self.assertEqual(tracker.total_unique_arrivals, 2)


if __name__ == "__main__":
    unittest.main()
