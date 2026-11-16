"""
ATCS Vehicle Track State Module.

Maintains per-track vehicle class majority voting, ground-plane speed estimation
via exponential moving average, and dual-threshold hysteresis for stationary queue state.
"""

from collections import defaultdict
from typing import Dict, Optional, Tuple

import numpy as np

try:
    from .vehicle_taxonomy import CATEGORY_PCU
except ImportError:
    from vehicle_taxonomy import CATEGORY_PCU


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
        self.longitudinal_speed_mps: Optional[float] = None  # Positive toward stopline (decreasing dist_m)
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
        if not (np.isfinite(t_src) and np.isfinite(x_m) and np.isfinite(dist_m)):
            return self.is_queued, self.smoothed_speed_mps

        if self.prev_obs is not None and t_src <= self.prev_obs[0]:
            # Ignore duplicate or backwards timestamps (dt <= 0) to protect state and speed EMA
            return self.is_queued, self.smoothed_speed_mps

        self.last_seen_time = t_src

        if self.prev_obs is not None:
            prev_t, prev_x, prev_dist = self.prev_obs
            dt = t_src - prev_t

            # 1.5s gap guard preserves queued status across brief detector drops
            if dt > 1.5:
                self.smoothed_speed_mps = None
                self.longitudinal_speed_mps = None
                if not self.is_queued:
                    self.stopped_since = None
            elif dt > 0.005:
                dist_moved = float(np.hypot(x_m - prev_x, dist_m - prev_dist))
                long_delta = float(prev_dist - dist_m)

                # Sub-centimeter displacement deadband (< 8cm) suppresses homography jitter
                instant_speed = 0.0 if dist_moved < 0.08 else dist_moved / dt
                inst_long_speed = 0.0 if abs(long_delta) < 0.08 else long_delta / dt

                # Exponential moving average filter (alpha = 0.25)
                if self.smoothed_speed_mps is None:
                    self.smoothed_speed_mps = instant_speed
                    self.longitudinal_speed_mps = inst_long_speed
                else:
                    self.smoothed_speed_mps = 0.25 * instant_speed + 0.75 * self.smoothed_speed_mps
                    self.longitudinal_speed_mps = 0.25 * inst_long_speed + 0.75 * (self.longitudinal_speed_mps or 0.0)

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
