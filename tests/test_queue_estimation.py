from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock

if "cv2" not in sys.modules:
    sys.modules["cv2"] = MagicMock()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perception import queue_estimation, traffic_analysis


class TestQueueEstimation(unittest.TestCase):
    def test_backward_compatible_reexports(self):
        self.assertIs(traffic_analysis.LaneQueue, queue_estimation.LaneQueue)
        self.assertIs(traffic_analysis.LaneAwareQueueEstimator, queue_estimation.LaneAwareQueueEstimator)

    def test_bumper_clearance_clustering_and_upstream_partition(self):
        est = queue_estimation.LaneAwareQueueEstimator(max_vehicle_gap_m=6.0, stopline_threshold_m=14.0)
        # Lane 0: Car at 5m + Car at 12m (gap = 7 - 4.5 = 2.5m <= 6m -> stopline cluster),
        # plus Bus at 30m (gap = 18 - (4.5+10)/2 = 10.75m > 6m -> upstream cluster).
        vehicles = {
            0: [
                (12.0, 1.0, 2, (0, 0, 10, 10), "CAR"),
                (5.0, 1.0, 1, (0, 0, 10, 10), "CAR"),
                (30.0, 2.5, 3, (0, 0, 10, 10), "BUS/TRUCK"),
            ],
            1: [],
        }
        lqs = est.estimate(vehicles, [0, 1])
        self.assertTrue(lqs[0].connected_to_stopline)
        self.assertEqual(lqs[0].count, 2)
        self.assertAlmostEqual(lqs[0].pcu, 2.0)
        self.assertAlmostEqual(lqs[0].tail_m, 12.0)
        self.assertEqual(lqs[0].track_ids, (1, 2))
        self.assertEqual(lqs[0].upstream_count, 1)
        self.assertAlmostEqual(lqs[0].upstream_pcu, 2.5)
        self.assertFalse(lqs[1].connected_to_stopline)
        self.assertEqual(lqs[1].count, 0)

    def test_median_smoothing_and_uncalibrated_fallback(self):
        est = queue_estimation.LaneAwareQueueEstimator(history_size=3)
        est.smooth(2, 2.0, 10.0, 1, 1.0, arrival_rate_pcu_s=0.2)
        est.smooth(8, 8.0, 35.0, 1, 1.0, arrival_rate_pcu_s=0.2)  # Single-frame spike
        res = est.smooth(3, 3.0, 14.0, 1, 1.5, arrival_rate_pcu_s=0.3)
        # Median of [2, 8, 3] is 3
        self.assertEqual(res["Q_stopline"], 3)
        self.assertAlmostEqual(res["D_tail_m"], 14.0)
        self.assertAlmostEqual(res["PCU_upstream"], 1.5)
        self.assertAlmostEqual(res["inflow_pcu"], 1.5)
        self.assertAlmostEqual(res["arrival_rate_pcu_s"], 0.3)
        self.assertIsNone(res["predicted_demand_pcu"])

        uncal = est.smooth(3, 3.0, 14.0, 1, 1.5, arrival_rate_pcu_s=0.3, metric_valid=False, calibration_status="UNCALIBRATED")
        self.assertEqual(uncal["Q_stopline"], 3)
        self.assertEqual(uncal["D_tail_m"], 0.0)
        self.assertAlmostEqual(uncal["D_tail_approx"], 14.0)
        self.assertEqual(uncal["inflow_pcu"], 0.0)
        self.assertEqual(uncal["arrival_rate_pcu_s"], 0.0)
        self.assertFalse(uncal["metric_valid"])


if __name__ == "__main__":
    unittest.main()
