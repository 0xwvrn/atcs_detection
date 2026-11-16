import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

if "cv2" not in sys.modules:
    sys.modules["cv2"] = MagicMock()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perception.traffic_analysis import (
    CalibrationStatus,
    LaneAwareQueueEstimator,
    RoadHomographyCalibrator,
)


def make_valid_calib_dict(w: int = 1280, h: int = 720):
    return {
        "image_width": w,
        "image_height": h,
        "image_points": [
            [450.0, 180.0],
            [830.0, 180.0],
            [1150.0, 600.0],
            [130.0, 600.0],
        ],
        "ground_points_metric": [
            [0.0, 50.0],
            [7.0, 50.0],
            [7.0, 0.0],
            [0.0, 0.0],
        ],
        "reference_points": [
            {"pixel": [640.0, 600.0], "ground_m": [3.5, 0.0]},
            {"pixel": [640.0, 180.0], "ground_m": [3.5, 50.0]},
        ],
        "reference_distances": [
            {"p1": [130.0, 600.0], "p2": [1150.0, 600.0], "distance_m": 7.0},
        ],
    }


class TestPhase1BCalibration(unittest.TestCase):
    def test_1_valid_perspective_transformation_and_known_reference_projections(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            json.dump(make_valid_calib_dict(), tf)
            calib_path = tf.name

        cal = RoadHomographyCalibrator(1280, 720, calib_file=calib_path, num_lanes=2)
        self.assertEqual(cal.status, CalibrationStatus.VALID)
        self.assertTrue(cal.is_calibrated)
        self.assertTrue(cal.metric_valid)
        self.assertIsNotNone(cal.quality_metrics)
        self.assertLess(cal.quality_metrics.corner_reprojection_rmse_m, 1e-4)
        self.assertLess(cal.quality_metrics.stopline_projection_error_m, 1e-4)
        self.assertLess(cal.quality_metrics.lane_width_consistency_error_m, 1e-4)
        self.assertLess(cal.quality_metrics.reference_point_rmse_m, 1e-4)
        self.assertLess(cal.quality_metrics.reference_distance_rmse_m, 1e-4)

        # Stopline center pixel (640, 600) -> (3.5m, 0.0m, lane 1)
        x_m, dist_m, lane_id = cal.contact_point_to_ground(640.0, 600.0)
        self.assertAlmostEqual(x_m, 3.5, places=3)
        self.assertAlmostEqual(dist_m, 0.0, places=3)
        self.assertEqual(lane_id, 1)

    def test_2_degenerate_and_collinear_points_rejected(self):
        cal = RoadHomographyCalibrator(1280, 720)
        self.assertEqual(cal.status, CalibrationStatus.UNCALIBRATED)

        # Collinear image points (all on y = 300)
        d = make_valid_calib_dict()
        d["image_points"] = [[100.0, 300.0], [300.0, 300.0], [600.0, 300.0], [900.0, 300.0]]
        self.assertFalse(cal.validate_calibration_dict(d))
        self.assertEqual(cal.status, CalibrationStatus.INVALID)
        self.assertFalse(cal.metric_valid)

        # 3 collinear points + 1 offset point
        d2 = make_valid_calib_dict()
        d2["image_points"] = [[200.0, 200.0], [400.0, 200.0], [600.0, 200.0], [400.0, 500.0]]
        self.assertFalse(cal.validate_calibration_dict(d2))
        self.assertEqual(cal.status, CalibrationStatus.INVALID)

        # Bowtie / self-intersecting non-convex quad
        d3 = make_valid_calib_dict()
        d3["image_points"] = [[450.0, 180.0], [1150.0, 600.0], [830.0, 180.0], [130.0, 600.0]]
        self.assertFalse(cal.validate_calibration_dict(d3))
        self.assertEqual(cal.status, CalibrationStatus.INVALID)

    def test_3_out_of_bounds_and_non_finite_inputs_rejected(self):
        cal = RoadHomographyCalibrator(1280, 720)
        d_oob = make_valid_calib_dict()
        d_oob["image_points"][2] = [1350.0, 600.0]  # > 1280
        self.assertFalse(cal.validate_calibration_dict(d_oob))
        self.assertEqual(cal.status, CalibrationStatus.INVALID)

        d_nan = make_valid_calib_dict()
        d_nan["ground_points_metric"][0] = [float("nan"), 50.0]
        self.assertFalse(cal.validate_calibration_dict(d_nan))
        self.assertEqual(cal.status, CalibrationStatus.INVALID)

        d_inf = make_valid_calib_dict()
        d_inf["image_points"][1] = [float("inf"), 180.0]
        self.assertFalse(cal.validate_calibration_dict(d_inf))
        self.assertEqual(cal.status, CalibrationStatus.INVALID)

    def test_4_image_resolution_mismatch_and_runtime_recalibration_state(self):
        cal = RoadHomographyCalibrator(1280, 720)
        d_res = make_valid_calib_dict(w=1920, h=1080)
        self.assertFalse(cal.validate_calibration_dict(d_res))
        self.assertEqual(cal.status, CalibrationStatus.REQUIRES_RECALIBRATION)
        self.assertFalse(cal.metric_valid)

        # Load valid 1280x720, then simulate camera stream switching to 640x480
        self.assertTrue(cal.validate_calibration_dict(make_valid_calib_dict(1280, 720)))
        self.assertEqual(cal.status, CalibrationStatus.VALID)
        self.assertFalse(cal.check_frame_dimensions(640, 480))
        self.assertEqual(cal.status, CalibrationStatus.REQUIRES_RECALIBRATION)
        self.assertFalse(cal.metric_valid)

    def test_5_implausible_ground_geometry_and_invalidation_recovery(self):
        cal = RoadHomographyCalibrator(1280, 720)
        # Inverted camera geometry (upstream below stopline in image)
        d_inv = make_valid_calib_dict()
        d_inv["image_points"] = [
            [130.0, 600.0],
            [1150.0, 600.0],
            [830.0, 180.0],
            [450.0, 180.0],
        ]
        self.assertFalse(cal.validate_calibration_dict(d_inv))
        self.assertEqual(cal.status, CalibrationStatus.INVALID)

        # Recovery with valid dict
        self.assertTrue(cal.validate_calibration_dict(make_valid_calib_dict()))
        self.assertEqual(cal.status, CalibrationStatus.VALID)

        # Manual invalidation falls back safely and marks status INVALID
        cal.invalidate("Camera bumped")
        self.assertEqual(cal.status, CalibrationStatus.INVALID)
        self.assertFalse(cal.metric_valid)

    def test_6_uncalibrated_estimator_suppresses_metric_distances_while_preserving_counts(self):
        est = LaneAwareQueueEstimator()
        out_uncal = est.smooth(
            q_stopline=5,
            pcu_stopline=6.0,
            tail_stopline_m=24.5,
            q_upstream=1,
            pcu_upstream=2.5,
            arrival_rate_pcu_s=0.35,
            metric_valid=False,
            calibration_status=CalibrationStatus.UNCALIBRATED.value,
        )
        # Counts and approximate queue PCU preserved, but metric D_tail_m and metric rates are not labeled valid meters
        self.assertEqual(out_uncal["Q_stopline"], 5)
        self.assertEqual(out_uncal["PCU_stopline"], 6.0)
        self.assertEqual(out_uncal["D_tail_m"], 0.0)
        self.assertEqual(out_uncal["D_tail_approx"], 24.5)
        self.assertEqual(out_uncal["arrival_rate_pcu_s"], 0.0)
        self.assertFalse(out_uncal["metric_valid"])
        self.assertEqual(out_uncal["calibration_status"], "UNCALIBRATED")


if __name__ == "__main__":
    unittest.main()
