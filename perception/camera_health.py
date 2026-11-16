"""
ATCS Camera Stream Health Monitoring Module.

Provides explicit camera stream health states, freshness evaluation,
frozen-frame detection, and bounded reconnection tracking.
"""

from enum import Enum
import hashlib
import time
from typing import Callable, Optional

import numpy as np


class CameraHealthState(str, Enum):
    """Explicit camera stream health state machine."""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    STALE = "STALE"
    DISCONNECTED = "DISCONNECTED"
    RECOVERING = "RECOVERING"


class CameraStreamHealthMonitor:
    """
    Independent camera freshness, frozen-frame, and disconnection state tracker.
    Uses a monotonic clock for all timeout calculations.

    Frozen-frame detection limitations:
      - Physical cameras viewing an empty road exhibit natural sensor thermal readout noise
        (typical pixel mean absolute difference > 0.10 gray levels), which keeps the stream HEALTHY.
      - A frozen frame buffer or stalled RTSP repeater repeats the exact same decoded frame
        (identical byte digest / MAD <= freeze_mad_threshold) or an unchanging hardware sequence ID.
      - Purely synthetic noise-free static test patterns without an advancing hardware sequence ID
        cannot be distinguished from a frozen buffer by pixel inspection alone.
    """

    def __init__(
        self,
        stale_timeout_s: float = 2.0,
        degraded_failure_threshold: int = 2,
        disconnect_failure_threshold: int = 5,
        frozen_frame_threshold: int = 12,
        min_freeze_duration_s: float = 1.0,
        freeze_mad_threshold: float = 0.02,
        recovery_frames_required: int = 2,
        reconnect_interval_s: float = 2.0,
        max_reconnect_attempts: int = 10,
        clock_fn: Callable[[], float] = time.monotonic,
    ):
        self.stale_timeout_s = float(stale_timeout_s)
        self.degraded_failure_threshold = max(1, int(degraded_failure_threshold))
        self.disconnect_failure_threshold = max(self.degraded_failure_threshold + 1, int(disconnect_failure_threshold))
        self.frozen_frame_threshold = max(2, int(frozen_frame_threshold))
        self.min_freeze_duration_s = max(0.0, float(min_freeze_duration_s))
        self.freeze_mad_threshold = max(0.0, float(freeze_mad_threshold))
        self.recovery_frames_required = max(1, int(recovery_frames_required))
        self.reconnect_interval_s = max(0.1, float(reconnect_interval_s))
        self.max_reconnect_attempts = max(1, int(max_reconnect_attempts))
        self.clock_fn = clock_fn

        self.last_acq_mono_s: Optional[float] = None
        self.last_proc_mono_s: Optional[float] = None
        self.last_tx_mono_s: Optional[float] = None
        self.consecutive_read_failures: int = 0
        self.consecutive_proc_failures: int = 0
        self.reconnect_attempts: int = 0
        self.last_reconnect_attempt_mono_s: Optional[float] = None

        self._prev_sample: Optional[np.ndarray] = None
        self._prev_digest: Optional[bytes] = None
        self._prev_seq_id: Optional[int] = None
        self._last_distinct_mono_s: Optional[float] = None
        self._freeze_start_mono_s: Optional[float] = None
        self.identical_frame_streak: int = 0
        self.is_frozen: bool = False

        self._recovery_streak: int = 0
        self._needs_tracker_reset: bool = False
        self._has_ever_failed_or_staled: bool = False
        self.state: CameraHealthState = CameraHealthState.DISCONNECTED

    def _now(self, mono_now: Optional[float] = None) -> float:
        return float(self.clock_fn() if mono_now is None else mono_now)

    @staticmethod
    def _extract_subsample(frame: np.ndarray) -> np.ndarray:
        arr = np.asarray(frame)
        if arr.ndim == 3:
            arr = arr[::8, ::8, 0]
        elif arr.ndim == 2:
            arr = arr[::8, ::8]
        return np.ascontiguousarray(arr, dtype=np.uint8)

    def record_frame_acquired(
        self,
        frame: Optional[np.ndarray],
        mono_now: Optional[float] = None,
        hw_frame_seq: Optional[int] = None,
    ) -> CameraHealthState:
        now = self._now(mono_now)
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return self.record_read_failure(now)

        was_unhealthy = self._has_ever_failed_or_staled and (
            self.state in (CameraHealthState.DISCONNECTED, CameraHealthState.STALE) or self.is_frozen
        )
        self.last_acq_mono_s = now
        self.consecutive_read_failures = 0
        self.reconnect_attempts = 0

        sample = self._extract_subsample(frame)
        digest = hashlib.blake2b(sample.tobytes(), digest_size=8).digest()

        is_duplicate = False
        if hw_frame_seq is not None and self._prev_seq_id is not None:
            is_duplicate = int(hw_frame_seq) <= self._prev_seq_id
        elif self._prev_sample is not None and self._prev_sample.shape == sample.shape:
            if digest == self._prev_digest:
                is_duplicate = True
            else:
                mad = float(np.mean(np.abs(sample.astype(np.float32) - self._prev_sample.astype(np.float32))))
                if mad <= self.freeze_mad_threshold:
                    is_duplicate = True

        self._prev_sample = sample
        self._prev_digest = digest
        if hw_frame_seq is not None:
            self._prev_seq_id = int(hw_frame_seq)

        if is_duplicate:
            if self.identical_frame_streak == 0:
                self._freeze_start_mono_s = self._last_distinct_mono_s if self._last_distinct_mono_s is not None else now
            self.identical_frame_streak += 1
            freeze_dur = now - (self._freeze_start_mono_s if self._freeze_start_mono_s is not None else now)
            if self.identical_frame_streak >= self.frozen_frame_threshold and freeze_dur >= self.min_freeze_duration_s:
                self.is_frozen = True
                self._has_ever_failed_or_staled = True
                self._recovery_streak = 0
                self._needs_tracker_reset = True
                self.state = CameraHealthState.STALE
            elif self.state == CameraHealthState.HEALTHY and self.identical_frame_streak >= max(2, self.frozen_frame_threshold // 2):
                self.state = CameraHealthState.DEGRADED
        else:
            self._last_distinct_mono_s = now
            self.identical_frame_streak = 0
            self._freeze_start_mono_s = None
            self.is_frozen = False
            if was_unhealthy or self.state == CameraHealthState.RECOVERING:
                self._recovery_streak += 1
                if self._recovery_streak >= self.recovery_frames_required:
                    self.state = CameraHealthState.HEALTHY
                else:
                    self.state = CameraHealthState.RECOVERING
            else:
                self.state = CameraHealthState.HEALTHY

        return self.state

    def record_frame_processed(self, success: bool = True, mono_now: Optional[float] = None) -> CameraHealthState:
        now = self._now(mono_now)
        if not success:
            self.consecutive_proc_failures += 1
            if self.consecutive_proc_failures >= self.disconnect_failure_threshold:
                self._has_ever_failed_or_staled = True
                self.state = CameraHealthState.STALE
            elif self.consecutive_proc_failures >= self.degraded_failure_threshold and self.state == CameraHealthState.HEALTHY:
                self.state = CameraHealthState.DEGRADED
            return self.state

        self.consecutive_proc_failures = 0
        if not self.is_frozen:
            self.last_proc_mono_s = now
        return self.evaluate_state(now)

    def record_read_failure(self, mono_now: Optional[float] = None) -> CameraHealthState:
        now = self._now(mono_now)
        self.consecutive_read_failures += 1
        self._recovery_streak = 0
        if self.consecutive_read_failures >= self.disconnect_failure_threshold:
            self._has_ever_failed_or_staled = True
            self.state = CameraHealthState.DISCONNECTED
            self._needs_tracker_reset = True
        elif self.consecutive_read_failures >= self.degraded_failure_threshold:
            if self.state == CameraHealthState.HEALTHY:
                self.state = CameraHealthState.DEGRADED
        if self.last_proc_mono_s is not None and (now - self.last_proc_mono_s) > self.stale_timeout_s:
            self._has_ever_failed_or_staled = True
            if self.state != CameraHealthState.DISCONNECTED:
                self.state = CameraHealthState.STALE
            self._needs_tracker_reset = True
        return self.state

    def record_telemetry_published(self, mono_now: Optional[float] = None):
        self.last_tx_mono_s = self._now(mono_now)

    def evaluate_state(self, mono_now: Optional[float] = None) -> CameraHealthState:
        now = self._now(mono_now)
        if self.state == CameraHealthState.DISCONNECTED:
            return self.state
        if self.is_frozen:
            self._has_ever_failed_or_staled = True
            self.state = CameraHealthState.STALE
            return self.state
        if self.last_proc_mono_s is None or (now - self.last_proc_mono_s) > self.stale_timeout_s:
            self._has_ever_failed_or_staled = True
            self.state = CameraHealthState.STALE
            self._needs_tracker_reset = True
            self._recovery_streak = 0
        return self.state

    def is_observation_fresh(self, mono_now: Optional[float] = None) -> bool:
        st = self.evaluate_state(mono_now)
        return st in (CameraHealthState.HEALTHY, CameraHealthState.DEGRADED) and not self.is_frozen

    def should_attempt_reconnect(self, mono_now: Optional[float] = None) -> bool:
        now = self._now(mono_now)
        if self.state != CameraHealthState.DISCONNECTED:
            return False
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            return False
        if self.last_reconnect_attempt_mono_s is None or (now - self.last_reconnect_attempt_mono_s) >= self.reconnect_interval_s:
            return True
        return False

    def mark_reconnect_attempt(self, mono_now: Optional[float] = None):
        self.last_reconnect_attempt_mono_s = self._now(mono_now)
        self.reconnect_attempts += 1
        self._prev_sample = None
        self._prev_digest = None
        self._prev_seq_id = None
        self._needs_tracker_reset = True

    def consume_tracker_reset_needed(self) -> bool:
        if self._needs_tracker_reset:
            self._needs_tracker_reset = False
            return True
        return False
