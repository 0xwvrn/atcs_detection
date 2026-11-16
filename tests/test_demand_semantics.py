from collections import defaultdict
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock

if "cv2" not in sys.modules:
    sys.modules["cv2"] = MagicMock()

_ROOT = Path(__file__).resolve().parents[2]
_VISION = _ROOT / "vision"
_CONTROLLER = _ROOT / "controller"
for p in (str(_VISION), str(_CONTROLLER)):
    if p not in sys.path:
        sys.path.insert(0, p)

from perception.arrival_tracking import UpstreamArrivalTracker
from perception.calibration import RoadHomographyCalibrator
from perception.queue_estimation import LaneAwareQueueEstimator
from perception.telemetry import serialize_observation_packet
from perception.tracking import VehicleTrackState
from telemetry_schema import parse_and_validate_telemetry
from adaptive_controller import AdaptiveTrafficController


def _make_calibrated_calibrator() -> RoadHomographyCalibrator:
    cal = RoadHomographyCalibrator(960, 540, calib_file="", num_lanes=2)
    ok = cal.validate_calibration_dict({
        "image_width": 960,
        "image_height": 540,
        "image_points": [[336, 162], [624, 162], [864, 496], [96, 496]],
        "ground_points_metric": [[0.0, 50.0], [7.0, 50.0], [7.0, 0.0], [0.0, 0.0]],
    })
    assert ok and cal.metric_valid
    return cal


def _run_synthetic_step(
    cal: RoadHomographyCalibrator,
    tracks: dict,
    est: LaneAwareQueueEstimator,
    arr: UpstreamArrivalTracker,
    t_src: float,
    obs_fresh: bool,
    allow_approx_calib: bool,
    detections,
):
    vbl = defaultdict(list)
    frame_obs = []
    for tid, xm, dm, lane_id, pcu, cat in detections:
        if lane_id < 0:
            continue
        if tid not in tracks:
            tracks[tid] = VehicleTrackState(tid, cat, pcu)
        is_q, _ = tracks[tid].update_position(t_src, xm, dm)
        frame_obs.append((tid, xm, dm, pcu, cat, is_q))
        if is_q:
            vbl[lane_id].append((dm, pcu, tid, (0, 0, 10, 10), cat))

    rate = arr.update(t_src, frame_obs) if (obs_fresh and cal.metric_valid) else 0.0
    lqs = est.estimate(vbl, [0, 1])
    tails = [lq.tail_m for lq in lqs.values() if lq.count > 0]
    smoothed = est.smooth(
        sum(lq.count for lq in lqs.values()),
        sum(lq.pcu for lq in lqs.values()),
        max(tails) if tails else 0.0,
        sum(lq.upstream_count for lq in lqs.values()),
        sum(lq.upstream_pcu for lq in lqs.values()),
        rate,
        metric_valid=cal.metric_valid,
        calibration_status=cal.status.value,
    )
    raw = serialize_observation_packet(
        "N",
        smoothed,
        is_calibrated=cal.metric_valid,
        frame_valid=obs_fresh,
        fresh=obs_fresh,
        allow_approx_calib=allow_approx_calib,
        seq=int(t_src * 10) + 1,
    )
    ok, parsed, err = parse_and_validate_telemetry(raw)
    assert ok, err
    return parsed


class TestDemandSemanticsCases(unittest.TestCase):
    """Stage 5G deterministic verification of Cases A-E using synthetic ground-plane observations."""

    def test_case_a_moving_vehicles_not_counted_as_queue(self):
        cal = _make_calibrated_calibrator()
        tracks, est, arr = {}, LaneAwareQueueEstimator(), UpstreamArrivalTracker(10.0)
        for i in range(15):
            t = i * 0.1
            pkt = _run_synthetic_step(
                cal, tracks, est, arr, t, True, False,
                [(1, 1.75, 44.0 - i * 1.5, 0, 1.0, "CAR")],  # 15 m/s moving
            )
        self.assertTrue(pkt["valid"])
        self.assertEqual(pkt["Q"], 0)
        self.assertEqual(pkt["PCU"], 0.0)
        self.assertEqual(pkt["inflow_pcu"], 0.0)
        self.assertGreater(pkt["arrival_rate_pcu_s"], 0.0)

    def test_case_b_and_c_stationary_queue_and_upstream_cluster(self):
        cal = _make_calibrated_calibrator()
        tracks, est, arr = {}, LaneAwareQueueEstimator(max_vehicle_gap_m=6.0, stopline_threshold_m=14.0), UpstreamArrivalTracker(10.0)
        # Vehicle 1 stopped at stopline (6.0m); Vehicle 2 stopped upstream beyond 6.0m clearance gap (26.0m)
        for i in range(18):
            t = i * 0.1
            pkt = _run_synthetic_step(
                cal, tracks, est, arr, t, True, False,
                [
                    (1, 1.75, 6.0, 0, 1.0, "CAR"),
                    (2, 1.75, 26.0, 0, 2.5, "BUS/TRUCK"),
                ],
            )
        self.assertTrue(pkt["valid"])
        self.assertTrue(pkt["calibrated"])
        self.assertEqual(pkt["Q"], 1)
        self.assertAlmostEqual(pkt["PCU"], 1.0)
        self.assertAlmostEqual(pkt["D_tail_m"], 6.0)
        self.assertAlmostEqual(pkt["pcu_upstream"], 2.5)
        self.assertAlmostEqual(pkt["inflow_pcu"], 2.5)

    def test_case_d_uncalibrated_approx_preserves_queue_pcu_and_suppresses_metrics(self):
        cal = RoadHomographyCalibrator(960, 540, calib_file="", num_lanes=2)  # UNCALIBRATED
        tracks, est, arr = {}, LaneAwareQueueEstimator(), UpstreamArrivalTracker(10.0)
        for i in range(18):
            t = i * 0.1
            pkt = _run_synthetic_step(
                cal, tracks, est, arr, t, True, True,  # allow_approx_calib=True
                [(1, 1.75, 6.0, 0, 1.0, "CAR")],
            )
        self.assertTrue(pkt["valid"])
        self.assertFalse(pkt["calibrated"])
        self.assertEqual(pkt["Q"], 1)
        self.assertAlmostEqual(pkt["PCU"], 1.0)
        self.assertEqual(pkt["D_tail_m"], 0.0)
        self.assertEqual(pkt["pcu_upstream"], 0.0)
        self.assertEqual(pkt["inflow_pcu"], 0.0)
        self.assertEqual(pkt["arrival_rate_pcu_s"], 0.0)

        ctrl = AdaptiveTrafficController()
        ctrl.update_approach("N", pkt["Q"], pkt["D_tail_m"], pkt["PCU"], is_valid=pkt["valid"], inflow_pcu=pkt["inflow_pcu"])
        pcu_n, measured_n = ctrl.get_approach_demand("N")
        self.assertTrue(measured_n)
        self.assertAlmostEqual(pcu_n, 1.0)

    def test_case_e_invalid_or_stale_cannot_masquerade_as_zero_traffic(self):
        cal = _make_calibrated_calibrator()
        tracks, est, arr = {}, LaneAwareQueueEstimator(), UpstreamArrivalTracker(10.0)
        pkt = _run_synthetic_step(
            cal, tracks, est, arr, 0.0, False, False,  # obs_fresh=False
            [(1, 1.75, 6.0, 0, 1.0, "CAR")],
        )
        self.assertFalse(pkt["fresh"])
        self.assertFalse(pkt["valid"])
        self.assertEqual(pkt["Q"], 0)
        self.assertEqual(pkt["PCU"], 0.0)

        ctrl = AdaptiveTrafficController()
        ctrl.update_approach("N", pkt["Q"], pkt["D_tail_m"], pkt["PCU"], is_valid=pkt["valid"], inflow_pcu=pkt["inflow_pcu"])
        pcu_n, measured_n = ctrl.get_approach_demand("N")
        self.assertFalse(measured_n)
        self.assertAlmostEqual(pcu_n, ctrl.nominal_fallback_pcu)  # 2.0 fallback, NOT 0.0 measured


if __name__ == "__main__":
    unittest.main()
