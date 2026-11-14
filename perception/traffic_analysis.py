"""
ATCS Traffic Analysis Module.

Contains road homography calibration, vehicle tracking and speed estimation,
stopline and upstream queue estimation, and IRC-standard PCU classification.
"""

from collections import defaultdict, deque
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Allowlist of recognized vehicular classes to reject non-traffic detections
VEHICLE_TAXONOMY = {
    "car", "motorcycle", "bus", "truck", "van", "bicycle", "auto_rickshaw", "vehicle"
}

# Metric vehicle lengths (meters) used to compute bumper-to-bumper clearance gaps
CATEGORY_LENGTH_M = {
    "BUS/TRUCK": 10.0,
    "CAR": 4.5,
    "VAN": 5.0,
    "AUTO_RICKSHAW": 2.8,
    "TWO_WHEELER": 1.8,
    "VEHICLE": 4.5,
}

# Standard Indian IRC Passenger Car Unit (PCU) equivalents
CATEGORY_PCU = {
    "BUS/TRUCK": 2.5,
    "CAR": 1.0,
    "VAN": 1.4,
    "AUTO_RICKSHAW": 1.2,
    "TWO_WHEELER": 0.5,
    "VEHICLE": 1.0,
}


def resolve_pcu(class_name: Optional[str], is_multiclass: bool) -> Tuple[Optional[float], Optional[str]]:
    """
    Maps detected YOLO object class to Indian IRC standard Passenger Car Unit (PCU).
    Returns (pcu_value, display_label) or (None, None) if class is not a road vehicle.
    """
    if not class_name:
        return None, None

    name = class_name.lower().strip()
    if not is_multiclass:
        return 1.0, "VEHICLE"

    is_vehicle = (name in VEHICLE_TAXONOMY) or any(
        sub in name for sub in ("car", "bus", "truck", "motor", "bike", "auto", "van", "vehicle")
    )
    if not is_vehicle:
        return None, None

    if "bus" in name or "truck" in name:
        return CATEGORY_PCU["BUS/TRUCK"], "BUS/TRUCK"
    if "van" in name:
        return CATEGORY_PCU["VAN"], "VAN"
    if "motorcycle" in name or "bike" in name or "two_wheeler" in name:
        return CATEGORY_PCU["TWO_WHEELER"], "TWO_WHEELER"
    if "auto" in name or "rickshaw" in name:
        return CATEGORY_PCU["AUTO_RICKSHAW"], "AUTO_RICKSHAW"
    return CATEGORY_PCU["CAR"], "CAR"


class RoadHomographyCalibrator:
    """
    Planar Homography: Maps camera pixels (u, v) on the road plane into
    metric ground coordinates (X lateral meters, Y longitudinal meters to stopline).
    """

    def __init__(
        self,
        frame_w: int,
        frame_h: int,
        calib_file: str = "",
        num_lanes: int = 2,
        lane_width_m: float = 3.5,
        approach_length_m: float = 50.0,
    ):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.num_lanes = num_lanes
        self.total_width_m = num_lanes * lane_width_m
        self.approach_length_m = approach_length_m
        self.is_calibrated = False

        if calib_file and Path(calib_file).exists():
            if not self.load_and_validate_calibration(calib_file):
                print(f"[CALIB WARNING] Calibration {calib_file} invalid. Using approximate trapezoid.")
                self.setup_default_trapezoid()
        else:
            self.setup_default_trapezoid()

        self.H = cv2.getPerspectiveTransform(self.src_trap, self.dst_metric)
        det = np.linalg.det(self.H)
        if abs(det) < 1e-7:
            print(f"[CALIB WARNING] Degenerate homography matrix (det={det:.2e}). Falling back.")
            self.setup_default_trapezoid()
            self.H = cv2.getPerspectiveTransform(self.src_trap, self.dst_metric)

    def load_and_validate_calibration(self, calib_file: str) -> bool:
        try:
            with open(calib_file, "r") as f:
                data = json.load(f)

            src_pts = np.float32(data.get("image_points", []))
            dst_pts = np.float32(data.get("ground_points_metric", []))

            if len(src_pts) != 4 or len(dst_pts) != 4 or not (np.all(np.isfinite(src_pts)) and np.all(np.isfinite(dst_pts))):
                return False

            if any(not (0 <= pt[0] <= self.frame_w and 0 <= pt[1] <= self.frame_h) for pt in src_pts):
                return False

            if cv2.contourArea(src_pts) < 1000.0 or not cv2.isContourConvex(src_pts.astype(np.int32)):
                return False

            if min(dst_pts[:, 0]) < -0.1 or max(dst_pts[:, 0]) > self.total_width_m * 1.5:
                return False
            if min(dst_pts[:, 1]) < -0.1 or max(dst_pts[:, 1]) > self.approach_length_m * 1.5:
                return False

            self.src_trap, self.dst_metric = src_pts, dst_pts
            self.is_calibrated = True
            print(f"[CALIB] Successfully loaded calibrated geometry from {calib_file}.")
            return True

        except Exception as e:
            print(f"[CALIB ERROR] Calibration parsing error: {e}")
            return False

    def setup_default_trapezoid(self):
        w, h = self.frame_w, self.frame_h
        self.src_trap = np.float32([
            [w * 0.35, h * 0.25],
            [w * 0.65, h * 0.25],
            [w * 0.90, h * 0.80],
            [w * 0.10, h * 0.80],
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
        x_m, dist_m = float(ground_pt[0, 0, 0]), float(ground_pt[0, 0, 1])

        if not (0.0 <= x_m < self.total_width_m and 0.0 <= dist_m <= self.approach_length_m):
            return x_m, dist_m, -1

        lane_id = int(x_m / (self.total_width_m / self.num_lanes))
        return x_m, dist_m, lane_id


class VehicleTrackState:
    """
    Tracks vehicle identity, speed, and stationary queue state.
    Uses temporal majority voting to guarantee class name and PCU consistency.
    """

    def __init__(self, track_id: int, class_name: str, pcu: float):
        self.track_id = track_id
        self.class_name = class_name
        self.pcu = pcu
        self.class_votes: Dict[str, int] = defaultdict(int)
        self.class_votes[class_name] += 1

        self.prev_obs: Optional[Tuple[float, float, float]] = None  # (t_src, x_m, dist_m)
        self.smoothed_speed_mps: Optional[float] = None
        self.stopped_since: Optional[float] = None
        self.is_queued = False
        self.last_seen_time = 0.0

    def update_class(self, class_name: str, pcu: float):
        self.class_votes[class_name] += 1
        best_cls = max(self.class_votes.items(), key=lambda item: item[1])[0]
        self.class_name = best_cls
        self.pcu = CATEGORY_PCU.get(best_cls, pcu)

    def update_position(
        self,
        t_src: float,
        x_m: float,
        dist_m: float,
        enter_speed_mps: float = 0.4,
        exit_speed_mps: float = 0.8,
        min_dwell_time: float = 1.0,
    ) -> Tuple[bool, Optional[float]]:
        self.last_seen_time = t_src

        if self.prev_obs is not None:
            prev_t, prev_x, prev_dist = self.prev_obs
            dt = t_src - prev_t

            # 1.5s gap guard preserves queued status across brief detector drops
            if dt > 1.5:
                self.smoothed_speed_mps = None
                if not self.is_queued:
                    self.stopped_since = None
            elif dt > 0.005:
                dist_moved = np.hypot(x_m - prev_x, dist_m - prev_dist)

                # Sub-centimeter displacement deadband (< 8cm) suppresses homography jitter
                instant_speed = 0.0 if dist_moved < 0.08 else dist_moved / dt

                # Exponential moving average filter (alpha = 0.25)
                if self.smoothed_speed_mps is None:
                    self.smoothed_speed_mps = instant_speed
                else:
                    self.smoothed_speed_mps = 0.25 * instant_speed + 0.75 * self.smoothed_speed_mps

        self.prev_obs = (t_src, x_m, dist_m)

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
            elif self.smoothed_speed_mps >= exit_speed_mps:
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
    upstream_count: int = 0
    upstream_pcu: float = 0.0


class LaneAwareQueueEstimator:
    """
    Computes lane-wise stopped queues, clusters vehicles by physical clearance gap,
    partitions stopline-connected from upstream queues, and filters temporal jitter.
    """

    def __init__(self, max_vehicle_gap_m: float = 6.0, stopline_threshold_m: float = 14.0, history_size: int = 5):
        self.max_vehicle_gap_m = max_vehicle_gap_m
        self.stopline_threshold_m = stopline_threshold_m
        self.history = deque(maxlen=history_size)

    def estimate_lane(self, lane_id: int, lane_vehicles: List[Tuple[float, float, int, tuple, str]]) -> LaneQueue:
        if not lane_vehicles:
            return LaneQueue(
                lane_id=lane_id, count=0, pcu=0.0, tail_m=0.0, track_ids=(),
                connected_to_stopline=False, upstream_count=0, upstream_pcu=0.0
            )

        sorted_v = sorted(lane_vehicles, key=lambda v: v[0])
        is_connected = sorted_v[0][0] <= self.stopline_threshold_m

        # Partition vehicles into contiguous clusters using bumper-to-bumper clearance
        clusters: List[List[Tuple[float, float, int, tuple, str]]] = [[sorted_v[0]]]

        for v in sorted_v[1:]:
            prev_v = clusters[-1][-1]
            half_lens = (CATEGORY_LENGTH_M.get(prev_v[4], 4.5) + CATEGORY_LENGTH_M.get(v[4], 4.5)) / 2.0
            clearance = max(0.0, (v[0] - prev_v[0]) - half_lens)
            if clearance <= self.max_vehicle_gap_m:
                clusters[-1].append(v)
            else:
                clusters.append([v])

        stopline_cluster = clusters[0] if is_connected else []
        upstream_clusters = clusters[1:] if is_connected else clusters

        return LaneQueue(
            lane_id=lane_id,
            count=len(stopline_cluster),
            pcu=sum(v[1] for v in stopline_cluster),
            tail_m=stopline_cluster[-1][0] if stopline_cluster else 0.0,
            track_ids=tuple(v[2] for v in stopline_cluster),
            connected_to_stopline=is_connected,
            upstream_count=sum(len(c) for c in upstream_clusters),
            upstream_pcu=sum(sum(v[1] for v in c) for c in upstream_clusters),
        )

    def estimate(self, vehicles_by_lane: Dict[int, List], lane_ids: List[int]) -> Dict[int, LaneQueue]:
        return {lid: self.estimate_lane(lid, vehicles_by_lane.get(lid, [])) for lid in lane_ids}

    def smooth(
        self,
        q_stopline: int,
        pcu_stopline: float,
        tail_stopline_m: float,
        q_upstream: int,
        pcu_upstream: float,
        moving_inflow_pcu: float = 0.0,
    ) -> Dict[str, Any]:
        inflow_pcu = max(0.0, float(pcu_upstream) + float(moving_inflow_pcu))
        self.history.append((q_stopline, pcu_stopline, tail_stopline_m, q_upstream, pcu_upstream, inflow_pcu))
        rows = list(self.history)
        median_count = int(np.median([r[0] for r in rows]))
        best_row = rows[-1]
        min_diff = float("inf")

        # Reverse iteration picks latest frame on tie to avoid smoothing delay
        for r in reversed(rows):
            diff = abs(r[0] - median_count)
            if diff < min_diff:
                min_diff, best_row = diff, r
                if diff == 0:
                    break

        return {
            "Q_stopline": int(best_row[0]),
            "PCU_stopline": round(float(best_row[1]), 2),
            "D_tail_m": round(float(best_row[2]), 2),
            "Q_upstream": int(best_row[3]),
            "PCU_upstream": round(float(best_row[4]), 2),
            "inflow_pcu": round(float(best_row[5]), 2),
        }
