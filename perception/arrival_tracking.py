"""
ATCS Upstream Arrival Tracking Module.

Counts unique approaching vehicles entering an upstream ground-plane measurement
region and computes a rolling-window arrival rate (in PCU/s) with tracker ID reuse
detection and brief-occlusion reacquisition.
"""

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class _TrackArrivalRecord:
    track_id: int
    class_name: str
    pcu: float
    first_t: float
    last_t: float
    last_x_m: float
    last_dist_m: float
    anchor_dist_m: float
    min_dist_m: float
    last_long_speed_mps: float = 0.0
    origin_in_queue: bool = False
    ever_queued: bool = False
    counted: bool = False


class UpstreamArrivalTracker:
    """
    Counts unique approaching vehicles entering an upstream ground-plane measurement zone
    and computes a rolling-window arrival rate (in PCU/s).

    Measurement geometry (ground-plane meters):
      - Lateral bounds: x_m in [0.0, corridor_width_m]
      - Longitudinal zone: dist_m in [zone_min_m, zone_max_m] upstream of stopline (0.0m)
      - Stopline queue exclusion: vehicles originating at dist_m <= stopline_threshold_m
        or already classified as queued are excluded from new arrival events.
    """

    def __init__(
        self,
        window_duration_s: float = 10.0,
        zone_min_m: float = 18.0,
        zone_max_m: float = 48.0,
        corridor_width_m: float = 7.0,
        stopline_threshold_m: float = 14.0,
        min_forward_advance_m: float = 0.4,
        track_ttl_s: float = 5.0,
        reacquire_window_s: float = 2.0,
        reacquire_radius_m: float = 4.0,
        id_reuse_jump_m: float = 10.0,
    ):
        if not np.isfinite(window_duration_s) or window_duration_s <= 0.0:
            raise ValueError(f"window_duration_s must be finite and > 0, got {window_duration_s}")
        if not (0.0 <= zone_min_m < zone_max_m):
            raise ValueError(f"Invalid upstream zone bounds [{zone_min_m}, {zone_max_m}]")

        self.window_duration_s = float(window_duration_s)
        self.zone_min_m = float(zone_min_m)
        self.zone_max_m = float(zone_max_m)
        self.corridor_width_m = float(corridor_width_m)
        self.stopline_threshold_m = float(stopline_threshold_m)
        self.min_forward_advance_m = float(min_forward_advance_m)
        self.track_ttl_s = float(track_ttl_s)
        self.reacquire_window_s = float(reacquire_window_s)
        self.reacquire_radius_m = float(reacquire_radius_m)
        self.id_reuse_jump_m = float(id_reuse_jump_m)

        self.active_tracks: Dict[int, _TrackArrivalRecord] = {}
        self.arrival_events: deque = deque()  # (timestamp_s, pcu, track_id)
        self.last_timestamp_s: Optional[float] = None
        self.total_unique_arrivals: int = 0

    def reset(self):
        self.active_tracks.clear()
        self.arrival_events.clear()
        self.last_timestamp_s = None
        self.total_unique_arrivals = 0

    def _try_reacquire(
        self,
        t_src: float,
        x_m: float,
        dist_m: float,
        pcu: float,
        class_name: str,
        current_frame_ids: set,
    ) -> Optional[_TrackArrivalRecord]:
        best_tid = None
        best_err = self.reacquire_radius_m

        for cand_tid, rec in list(self.active_tracks.items()):
            if cand_tid in current_frame_ids:
                continue
            dt = t_src - rec.last_t
            if dt <= 0.0 or dt > self.reacquire_window_s:
                continue
            if abs(rec.last_x_m - x_m) > 2.5:
                continue
            # Cannot reacquire a vehicle upstream of where it was last observed
            if dist_m > rec.last_dist_m + 0.5:
                continue
            # Require class or PCU weight compatibility
            if rec.class_name != class_name and abs(rec.pcu - pcu) > 0.3:
                continue
            pred_dist = rec.last_dist_m - max(0.0, rec.last_long_speed_mps) * dt
            err = abs(pred_dist - dist_m)
            if err <= best_err:
                best_err = err
                best_tid = cand_tid

        if best_tid is not None:
            return self.active_tracks.pop(best_tid)
        return None

    def update(
        self,
        t_src: float,
        observations: List[Tuple[int, float, float, float, str, bool]],
    ) -> float:
        """
        Processes one frame of ground-plane vehicle observations:
        observations: list of (track_id, x_m, dist_m, pcu, class_name, is_queued).
        Returns rolling-window arrival_rate_pcu_s (in PCU/s).
        """
        if t_src is None or isinstance(t_src, bool) or not isinstance(t_src, (int, float)) or not np.isfinite(t_src):
            return self.get_arrival_rate()
        t_val = float(t_src)
        if self.last_timestamp_s is not None and t_val < self.last_timestamp_s:
            return self.get_arrival_rate()

        self.last_timestamp_s = t_val

        # Expire stale tracks
        for tid, rec in list(self.active_tracks.items()):
            age = t_val - rec.last_t
            if age > self.track_ttl_s:
                del self.active_tracks[tid]

        current_ids = {int(obs[0]) for obs in observations if obs and len(obs) >= 6}

        for obs in observations:
            tid, x_m, dist_m, pcu, class_name, is_queued = (
                int(obs[0]), float(obs[1]), float(obs[2]), float(obs[3]), str(obs[4]), bool(obs[5])
            )
            if not (np.isfinite(x_m) and np.isfinite(dist_m) and np.isfinite(pcu)) or pcu <= 0.0:
                continue
            if not (0.0 <= x_m <= self.corridor_width_m):
                continue

            rec = self.active_tracks.get(tid)
            was_id_reused = False
            if rec is not None:
                dt = t_val - rec.last_t
                # Detect tracker ID reuse: expired gap or implausible upstream jump
                if dt > self.track_ttl_s or (dist_m - rec.min_dist_m) >= self.id_reuse_jump_m:
                    rec = None
                    was_id_reused = True
                    del self.active_tracks[tid]

            if rec is None:
                reacquired = None if was_id_reused else self._try_reacquire(t_val, x_m, dist_m, pcu, class_name, current_ids)
                if reacquired is not None:
                    reacquired.track_id = tid
                    rec = reacquired
                else:
                    in_stopline_zone = bool(is_queued or dist_m <= self.stopline_threshold_m)
                    rec = _TrackArrivalRecord(
                        track_id=tid,
                        class_name=class_name,
                        pcu=pcu,
                        first_t=t_val,
                        last_t=t_val,
                        last_x_m=x_m,
                        last_dist_m=dist_m,
                        anchor_dist_m=dist_m,
                        min_dist_m=dist_m,
                        origin_in_queue=in_stopline_zone,
                        ever_queued=in_stopline_zone,
                        counted=False,
                    )
                self.active_tracks[tid] = rec

            dt = t_val - rec.last_t
            long_delta = rec.last_dist_m - dist_m  # Positive when approaching stopline (dist_m decreasing)
            if dt > 0.005:
                inst_long_v = 0.0 if abs(long_delta) < 0.08 else (long_delta / dt)
                rec.last_long_speed_mps = 0.4 * inst_long_v + 0.6 * rec.last_long_speed_mps

            rec.pcu = pcu
            rec.class_name = class_name
            rec.last_t = t_val
            rec.last_x_m = x_m
            rec.last_dist_m = dist_m
            rec.min_dist_m = min(rec.min_dist_m, dist_m)
            if is_queued:
                rec.ever_queued = True
            if dist_m > rec.anchor_dist_m and not rec.counted:
                rec.anchor_dist_m = dist_m

            in_upstream_zone = (self.zone_min_m <= dist_m <= self.zone_max_m)
            forward_progress = (rec.anchor_dist_m - dist_m) >= self.min_forward_advance_m
            moving_forward = (long_delta >= -0.05) and (rec.last_long_speed_mps >= 0.1)

            if (
                in_upstream_zone
                and not rec.counted
                and not rec.origin_in_queue
                and not rec.ever_queued
                and not is_queued
                and forward_progress
                and moving_forward
            ):
                rec.counted = True
                self.total_unique_arrivals += 1
                self.arrival_events.append((t_val, pcu, tid))

        self._purge_old_events(t_val)
        return self.get_arrival_rate(t_val)

    def _purge_old_events(self, now_s: float):
        cutoff = now_s - self.window_duration_s
        while self.arrival_events and self.arrival_events[0][0] <= cutoff:
            self.arrival_events.popleft()

    def get_arrival_rate(self, query_time_s: Optional[float] = None) -> float:
        if self.last_timestamp_s is None:
            return 0.0
        ref_t = self.last_timestamp_s if query_time_s is None else float(query_time_s)
        if not np.isfinite(ref_t) or ref_t < self.last_timestamp_s:
            ref_t = self.last_timestamp_s
        self._purge_old_events(ref_t)
        if not self.arrival_events:
            return 0.0
        total_pcu = sum(ev[1] for ev in self.arrival_events)
        return round(float(total_pcu / self.window_duration_s), 3)
