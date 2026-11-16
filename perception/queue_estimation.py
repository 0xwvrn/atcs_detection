"""
ATCS Lane-Aware Queue Estimation Module.

Computes lane-wise stopped queues, clusters vehicles by bumper-to-bumper clearance
gap, partitions stopline-connected queues from upstream stopped clusters, and applies
median-count temporal smoothing with explicit separation of metric vs. approximate fields.
"""

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    from .calibration import CalibrationStatus
    from .vehicle_taxonomy import CATEGORY_LENGTH_M
except ImportError:
    from calibration import CalibrationStatus
    from vehicle_taxonomy import CATEGORY_LENGTH_M


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
        arrival_rate_pcu_s: float = 0.0,
        metric_valid: bool = True,
        calibration_status: str = CalibrationStatus.VALID.value,
    ) -> Dict[str, Any]:
        """
        Smooths stopline and upstream stationary queue observations using a median count filter.
        Quantities are strictly separated:
          - PCU_stopline (PCU): vehicles in the stopline-connected queue
          - PCU_upstream (PCU): vehicles in upstream stopped clusters not yet connected to stopline
          - inflow_pcu (PCU): physical stationary upstream demand (PCU_upstream), never per-frame moving sum
          - arrival_rate_pcu_s (PCU/s): rolling-window unique vehicle arrival rate
          - predicted_demand_pcu: None (not enabled without validated horizon model)
        When metric_valid is False (uncalibrated/invalid calibration), D_tail_m and metric arrival
        rates are zeroed so approximate pixel-derived geometry is never labeled as validated meters.
        """
        clean_upstream_pcu = max(0.0, float(pcu_upstream)) if np.isfinite(pcu_upstream) else 0.0
        clean_rate = max(0.0, float(arrival_rate_pcu_s)) if (metric_valid and np.isfinite(arrival_rate_pcu_s)) else 0.0
        clean_tail = max(0.0, float(tail_stopline_m)) if np.isfinite(tail_stopline_m) else 0.0
        self.history.append((q_stopline, pcu_stopline, clean_tail, q_upstream, clean_upstream_pcu))
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

        smoothed_upstream_pcu = round(float(best_row[4]), 2) if metric_valid else 0.0
        approx_tail = round(float(best_row[2]), 2)
        return {
            "Q_stopline": int(best_row[0]),
            "PCU_stopline": round(float(best_row[1]), 2),
            "D_tail_m": approx_tail if metric_valid else 0.0,
            "D_tail_approx": approx_tail,
            "Q_upstream": int(best_row[3]),
            "PCU_upstream": smoothed_upstream_pcu,
            "inflow_pcu": smoothed_upstream_pcu,
            "arrival_rate_pcu_s": round(clean_rate, 3),
            "predicted_demand_pcu": None,
            "metric_valid": bool(metric_valid),
            "calibration_status": str(calibration_status),
        }
