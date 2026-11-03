"""
ATCS Perception Visualization Module.

Provides OpenCV overlay rendering for vehicle bounding boxes, approach corridors,
stopline queues, metric distance tags, and the real-time HUD banner.
"""

from typing import Any, Dict, Tuple
import cv2
import numpy as np


def _put_text(frame: np.ndarray, text: str, org: Tuple[int, int], scale: float, color: Tuple[int, int, int], thickness: int = 1):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_approach_corridor(frame: np.ndarray, calibrator: Any):
    """Draws boundary quadrilateral of approach corridor and highlighted stopline."""
    trap_pts = calibrator.src_trap.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [trap_pts], isClosed=True, color=(255, 128, 0), thickness=2)
    pt3 = tuple(map(int, calibrator.src_trap[3]))
    pt2 = tuple(map(int, calibrator.src_trap[2]))
    cv2.line(frame, pt3, pt2, (0, 0, 255), 3)
    _put_text(frame, "STOPLINE (0m)", (pt3[0] + 10, pt3[1] - 8), 0.55, (0, 0, 255), 2)


def draw_moving_vehicle(frame: np.ndarray, box: Tuple[int, int, int, int], lane_id: int, tid: int, speed_kmh: float):
    """Draws thin grey box with speed annotation for moving vehicle."""
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 1)
    _put_text(frame, f"L{lane_id} ID:{tid} {speed_kmh:.0f}km/h", (x1, max(15, y1 - 4)), 0.45, (180, 180, 180), 1)


def draw_queued_vehicle(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
    lane_id: int,
    tid: int,
    category_label: str,
    dist_m: float,
    is_stopline: bool,
):
    """Draws colored box for stationary vehicle (Green: Stopline queue, Cyan: Upstream queue)."""
    x1, y1, x2, y2 = box
    color = (0, 255, 128) if is_stopline else (255, 200, 0)
    prefix = "Q-L" if is_stopline else "UP-L"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    _put_text(frame, f"{prefix}{lane_id} ID:{tid} [{category_label}] {dist_m:.1f}m", (x1, max(18, y1 - 6)), 0.48, color, 2)


def draw_heads_up_display(
    frame: np.ndarray,
    frame_w: int,
    approach_leg: str,
    smoothed_q: dict,
    lane_queues: Dict[int, Any],
    t_src: float,
    is_calibrated: bool,
    allow_approx_calib: bool,
    is_connected: bool,
    feedback: str,
):
    """Renders semi-transparent HUD banner with approach metrics, tail, link status, and ESP32 echo."""
    hud = frame.copy()
    cv2.rectangle(hud, (0, 0), (frame_w, 68), (20, 20, 20), -1)
    cv2.addWeighted(hud, 0.85, frame, 0.15, 0, frame)

    lanes_str = " | ".join(f"L{lid}:{lq.count}{'*' if lq.connected_to_stopline else '^'}" for lid, lq in lane_queues.items())
    q_color = (0, 255, 128) if smoothed_q["Q_stopline"] > 0 else (200, 200, 200)
    _put_text(frame, f"APPROACH [{approach_leg}] STOPLINE: {smoothed_q['Q_stopline']} ({lanes_str}) | UPSTREAM: {smoothed_q['Q_upstream']} | DEMAND: {smoothed_q['PCU_stopline']} PCU", (15, 26), 0.54, q_color, 2)

    calib_str = "CALIBRATION LOADED" if is_calibrated else ("APPROX CALIB ALLOWED" if allow_approx_calib else "UNCONFIGURED APPROX")
    echo_str = f" | ECHO: {feedback}" if feedback else ""
    _put_text(frame, f"TAIL: {smoothed_q['D_tail_m']}m | TIME: {t_src:.1f}s | CALIB: {calib_str}{echo_str}", (15, 52), 0.50, (200, 200, 200), 1)

    link_str = "HW SERIAL ONLINE" if is_connected else "STANDALONE"
    link_color = (0, 255, 0) if is_connected else (0, 165, 255)
    _put_text(frame, f"LINK: {link_str}", (frame_w - 240, 26), 0.55, link_color, 2)
