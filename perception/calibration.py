"""
ATCS Planar Homography Calibration Module.

Maps camera image pixels (u, v) on the road plane into metric ground coordinates
(X lateral meters, Y longitudinal meters to stopline) with explicit geometric
validation, calibration state tracking, and reference quality metrics.
"""

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


class CalibrationStatus(str, Enum):
    """Explicit homography calibration state."""
    UNCALIBRATED = "UNCALIBRATED"
    VALID = "VALID"
    INVALID = "INVALID"
    REQUIRES_RECALIBRATION = "REQUIRES_RECALIBRATION"


@dataclass(frozen=True)
class CalibrationQualityMetrics:
    """
    Calibration quality metrics kept strictly separate from mathematical/geometric validity.
    A homography can be mathematically VALID (4 non-collinear points) without real-world
    metric accuracy unless verified against independent reference points/distances.
    """
    corner_reprojection_rmse_m: float
    stopline_projection_error_m: float
    lane_width_consistency_error_m: float
    ground_projection_valid: bool
    reference_point_rmse_m: Optional[float] = None
    reference_distance_rmse_m: Optional[float] = None


def _compute_homography_4pt(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """Computes exact 3x3 planar homography H from 4 point correspondences via DLT / linear system."""
    A = np.zeros((8, 8), dtype=np.float64)
    b = np.zeros((8,), dtype=np.float64)
    for i in range(4):
        u, v = float(src_pts[i, 0]), float(src_pts[i, 1])
        x, y = float(dst_pts[i, 0]), float(dst_pts[i, 1])
        A[2 * i] = [u, v, 1.0, 0.0, 0.0, 0.0, -u * x, -v * x]
        b[2 * i] = x
        A[2 * i + 1] = [0.0, 0.0, 0.0, u, v, 1.0, -u * y, -v * y]
        b[2 * i + 1] = y
    h, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    if rank < 8 or not np.all(np.isfinite(h)):
        return np.zeros((3, 3), dtype=np.float64)
    return np.array([
        [h[0], h[1], h[2]],
        [h[3], h[4], h[5]],
        [h[6], h[7], 1.0],
    ], dtype=np.float64)


def _project_points_homogeneous(H: np.ndarray, pts_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Projects Nx2 points (u, v) using 3x3 homography H, returning (projected_Nx2, w_denom_N)."""
    pts = np.asarray(pts_2d, dtype=np.float64).reshape(-1, 2)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    hom = np.hstack([pts, ones]) @ H.T
    w = hom[:, 2]
    safe_w = np.where(np.abs(w) > 1e-12, w, np.nan)
    proj = hom[:, :2] / safe_w[:, None]
    return proj, w


def _quad_cross_products(pts: np.ndarray) -> np.ndarray:
    """Returns the 4 directed edge cross products around a 4-vertex polygon."""
    crosses = np.zeros(4, dtype=np.float64)
    for i in range(4):
        p0 = pts[i]
        p1 = pts[(i + 1) % 4]
        p2 = pts[(i + 2) % 4]
        e1 = p1 - p0
        e2 = p2 - p1
        crosses[i] = float(e1[0] * e2[1] - e1[1] * e2[0])
    return crosses


def _min_triplet_triangle_area(pts: np.ndarray) -> float:
    """Returns the minimum area among all 4 vertex triplets of a 4-point set (detects collinearity)."""
    areas = []
    for indices in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
        a, b, c = pts[indices[0]], pts[indices[1]], pts[indices[2]]
        area = 0.5 * abs(float((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])))
        areas.append(area)
    return min(areas)


class RoadHomographyCalibrator:
    """
    Planar Homography: Maps camera pixels (u, v) on the road plane into
    metric ground coordinates (X lateral meters, Y longitudinal meters to stopline).

    Coordinate Conventions:
      - Image coordinates (u, v): pixels, origin (0, 0) top-left, u in [0, frame_w], v in [0, frame_h].
      - Ground coordinates (X, Y): meters, X in [0, total_width_m] left-to-right across approach lanes,
        Y in [0, approach_length_m] longitudinal distance upstream from the stopline (Y = 0.0m at stopline).
      - Expected 4-point ordering: [upstream-left, upstream-right, stopline-right, stopline-left].
    """

    # Documented geometric validation thresholds
    MIN_IMAGE_QUAD_AREA_PX2 = 1000.0      # Rejects collapsed pixel regions smaller than ~32x32 px
    MIN_IMAGE_TRIPLET_AREA_PX2 = 250.0    # Rejects collinear/near-collinear 3-point pixel configurations
    MIN_GROUND_QUAD_AREA_M2 = 10.0        # Rejects implausibly tiny road corridors (< 10 m^2)
    MIN_GROUND_TRIPLET_AREA_M2 = 2.0      # Rejects collinear ground calibration points
    MIN_HOMOGRAPHY_DET = 1e-7             # Rejects singular perspective matrices
    MAX_HOMOGRAPHY_COND = 1e8             # Rejects numerically ill-conditioned transformations
    MIN_HOMOGENEOUS_W = 1e-4              # Rejects horizon-crossing / inverted projective rays

    def __init__(
        self,
        frame_w: int,
        frame_h: int,
        calib_file: str = "",
        num_lanes: int = 2,
        lane_width_m: float = 3.5,
        approach_length_m: float = 50.0,
    ):
        if frame_w <= 0 or frame_h <= 0 or num_lanes <= 0 or lane_width_m <= 0.0 or approach_length_m <= 0.0:
            raise ValueError("Frame dimensions, lane count, lane width, and approach length must be positive")

        self.frame_w = int(frame_w)
        self.frame_h = int(frame_h)
        self.num_lanes = int(num_lanes)
        self.lane_width_m = float(lane_width_m)
        self.total_width_m = float(num_lanes * lane_width_m)
        self.approach_length_m = float(approach_length_m)
        self.calib_file = str(calib_file) if calib_file else ""

        self.status: CalibrationStatus = CalibrationStatus.UNCALIBRATED
        self.is_calibrated: bool = False
        self.validation_error: str = ""
        self.quality_metrics: Optional[CalibrationQualityMetrics] = None
        self.calibrated_size: Optional[Tuple[int, int]] = None

        if self.calib_file:
            if Path(self.calib_file).exists():
                if not self.load_and_validate_calibration(self.calib_file):
                    print(f"[CALIB WARNING] Calibration {self.calib_file} rejected ({self.status.value}: {self.validation_error}). Using approximate trapezoid.")
                    self._apply_fallback_trapezoid(preserve_status=True)
            else:
                self.status = CalibrationStatus.INVALID
                self.validation_error = f"Calibration file not found: {self.calib_file}"
                self._apply_fallback_trapezoid(preserve_status=True)
        else:
            self.setup_default_trapezoid()

    @property
    def metric_valid(self) -> bool:
        """True only when homography is geometrically valid and matches current frame resolution."""
        return self.status == CalibrationStatus.VALID and self.is_calibrated

    def check_frame_dimensions(self, actual_w: int, actual_h: int) -> bool:
        """
        Verifies runtime frame dimensions match calibrated dimensions.
        If resolution changes mid-stream, invalidates metric calibration to REQUIRES_RECALIBRATION.
        """
        if actual_w != self.frame_w or actual_h != self.frame_h:
            self.status = CalibrationStatus.REQUIRES_RECALIBRATION
            self.is_calibrated = False
            self.validation_error = (
                f"Runtime frame resolution ({actual_w}x{actual_h}) does not match "
                f"calibrator dimensions ({self.frame_w}x{self.frame_h})"
            )
            return False
        return self.metric_valid

    def load_and_validate_calibration(self, calib_file: str) -> bool:
        try:
            with open(calib_file, "r") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                self._set_invalid("Calibration JSON root must be an object")
                return False
            return self.validate_calibration_dict(data)
        except Exception as e:
            self._set_invalid(f"Calibration parsing error: {e}")
            return False

    def _set_invalid(self, reason: str, status: CalibrationStatus = CalibrationStatus.INVALID):
        self.status = status
        self.is_calibrated = False
        self.validation_error = reason
        self.quality_metrics = None

    def validate_calibration_dict(self, data: Dict[str, Any]) -> bool:
        # 1. Optional or explicit calibration resolution check
        cal_w = data.get("image_width")
        cal_h = data.get("image_height")
        if cal_w is not None or cal_h is not None:
            if cal_w != self.frame_w or cal_h != self.frame_h:
                self._set_invalid(
                    f"Calibration image dimensions ({cal_w}x{cal_h}) do not match camera ({self.frame_w}x{self.frame_h})",
                    status=CalibrationStatus.REQUIRES_RECALIBRATION,
                )
                return False

        raw_src = data.get("image_points")
        raw_dst = data.get("ground_points_metric")
        if not isinstance(raw_src, list) or not isinstance(raw_dst, list) or len(raw_src) != 4 or len(raw_dst) != 4:
            self._set_invalid("Exactly 4 image_points and 4 ground_points_metric are required")
            return False

        try:
            src_pts = np.asarray(raw_src, dtype=np.float64)
            dst_pts = np.asarray(raw_dst, dtype=np.float64)
        except Exception:
            self._set_invalid("Non-numeric calibration coordinates")
            return False

        if src_pts.shape != (4, 2) or dst_pts.shape != (4, 2):
            self._set_invalid("Calibration point arrays must have shape (4, 2)")
            return False

        # 2. Reject NaN or Infinity
        if not (np.all(np.isfinite(src_pts)) and np.all(np.isfinite(dst_pts))):
            self._set_invalid("Calibration points contain NaN or Infinity")
            return False

        # 3. Reject out-of-bounds image coordinates
        if np.any(src_pts[:, 0] < 0.0) or np.any(src_pts[:, 0] > self.frame_w) or np.any(src_pts[:, 1] < 0.0) or np.any(src_pts[:, 1] > self.frame_h):
            self._set_invalid("One or more image_points lie outside frame bounds")
            return False

        # 4. Reject collinear, degenerate, or non-convex quadrilaterals in image and ground space
        src_crosses = _quad_cross_products(src_pts)
        dst_crosses = _quad_cross_products(dst_pts)
        if not (np.all(src_crosses > 0.0) or np.all(src_crosses < 0.0)):
            self._set_invalid("image_points do not form a strictly convex quadrilateral")
            return False
        if not (np.all(dst_crosses > 0.0) or np.all(dst_crosses < 0.0)):
            self._set_invalid("ground_points_metric do not form a strictly convex quadrilateral")
            return False

        src_min_tri = _min_triplet_triangle_area(src_pts)
        dst_min_tri = _min_triplet_triangle_area(dst_pts)
        src_quad_area = 0.5 * abs(float(np.sum(src_pts[:, 0] * np.roll(src_pts[:, 1], -1) - np.roll(src_pts[:, 0], -1) * src_pts[:, 1])))
        dst_quad_area = 0.5 * abs(float(np.sum(dst_pts[:, 0] * np.roll(dst_pts[:, 1], -1) - np.roll(dst_pts[:, 0], -1) * dst_pts[:, 1])))

        if src_min_tri < self.MIN_IMAGE_TRIPLET_AREA_PX2 or src_quad_area < self.MIN_IMAGE_QUAD_AREA_PX2:
            self._set_invalid(f"Degenerate or collinear image_points (min_tri={src_min_tri:.1f}px^2, area={src_quad_area:.1f}px^2)")
            return False
        if dst_min_tri < self.MIN_GROUND_TRIPLET_AREA_M2 or dst_quad_area < self.MIN_GROUND_QUAD_AREA_M2:
            self._set_invalid(f"Degenerate or collinear ground_points_metric (min_tri={dst_min_tri:.2f}m^2, area={dst_quad_area:.2f}m^2)")
            return False

        # 5. Plausible road geometry bounds and orientation in ground space
        if np.min(dst_pts[:, 0]) < -0.1 or np.max(dst_pts[:, 0]) > self.total_width_m * 1.5:
            self._set_invalid("ground_points_metric lateral X coordinates outside plausible road width")
            return False
        if np.min(dst_pts[:, 1]) < -0.1 or np.max(dst_pts[:, 1]) > self.approach_length_m * 1.5:
            self._set_invalid("ground_points_metric longitudinal Y coordinates outside plausible approach length")
            return False

        # Upstream points (0, 1) must be further from stopline than stopline points (2, 3),
        # and in image space upstream (0, 1) must appear above stopline (2, 3) (smaller v).
        if not (dst_pts[0, 1] > dst_pts[3, 1] + 5.0 and dst_pts[1, 1] > dst_pts[2, 1] + 5.0):
            self._set_invalid("Implausible longitudinal ground ordering: upstream points must exceed stopline by >= 5m")
            return False
        if not (dst_pts[1, 0] > dst_pts[0, 0] + 1.5 and dst_pts[2, 0] > dst_pts[3, 0] + 1.5):
            self._set_invalid("Implausible lateral ground ordering: right boundary must exceed left boundary by >= 1.5m")
            return False
        if not (max(src_pts[0, 1], src_pts[1, 1]) < min(src_pts[2, 1], src_pts[3, 1])):
            self._set_invalid("Implausible camera geometry: upstream road boundary must be above stopline in image")
            return False

        # 6. Compute Homography and verify non-singularity and numerical conditioning
        H = _compute_homography_4pt(src_pts, dst_pts)
        if not np.all(np.isfinite(H)):
            self._set_invalid("Homography matrix contains non-finite values")
            return False
        det = float(np.linalg.det(H))
        if abs(det) < self.MIN_HOMOGRAPHY_DET:
            self._set_invalid(f"Singular homography matrix (det={det:.2e})")
            return False
        cond = float(np.linalg.cond(H))
        if not np.isfinite(cond) or cond > self.MAX_HOMOGRAPHY_COND:
            self._set_invalid(f"Ill-conditioned homography matrix (cond={cond:.2e})")
            return False

        # 7. Project corridor test points (corners, midpoints, centroid) and check finite & positive depth
        centroid = np.mean(src_pts, axis=0, keepdims=True)
        stopline_mid = 0.5 * (src_pts[2:3] + src_pts[3:4])
        upstream_mid = 0.5 * (src_pts[0:1] + src_pts[1:2])
        test_pts = np.vstack([src_pts, centroid, stopline_mid, upstream_mid])
        proj_pts, w_vals = _project_points_homogeneous(H, test_pts)
        if not np.all(np.isfinite(proj_pts)) or np.any(w_vals <= self.MIN_HOMOGENEOUS_W):
            self._set_invalid("Homography produces non-finite or horizon-inverted projected coordinates")
            return False

        # 8. Compute calibration quality metrics (separate from validity)
        corner_err = np.linalg.norm(proj_pts[:4] - dst_pts, axis=1)
        corner_rmse = float(np.sqrt(np.mean(corner_err ** 2)))
        stopline_err = float(max(abs(proj_pts[2, 1]), abs(proj_pts[3, 1]), abs(proj_pts[5, 1])))
        up_width = float(np.linalg.norm(proj_pts[1] - proj_pts[0]))
        stop_width = float(np.linalg.norm(proj_pts[2] - proj_pts[3]))
        lane_consistency_err = float(max(abs(up_width - self.total_width_m), abs(stop_width - self.total_width_m)))

        ref_pt_rmse: Optional[float] = None
        if isinstance(data.get("reference_points"), list) and len(data["reference_points"]) > 0:
            ref_px = np.asarray([item["pixel"] for item in data["reference_points"]], dtype=np.float64)
            ref_gt = np.asarray([item["ground_m"] for item in data["reference_points"]], dtype=np.float64)
            ref_proj, ref_w = _project_points_homogeneous(H, ref_px)
            if np.all(np.isfinite(ref_proj)) and np.all(ref_w > self.MIN_HOMOGENEOUS_W):
                ref_pt_rmse = float(np.sqrt(np.mean(np.sum((ref_proj - ref_gt) ** 2, axis=1))))

        ref_dist_rmse: Optional[float] = None
        if isinstance(data.get("reference_distances"), list) and len(data["reference_distances"]) > 0:
            d_errs = []
            for item in data["reference_distances"]:
                pair_px = np.asarray([item["p1"], item["p2"]], dtype=np.float64)
                pair_proj, pair_w = _project_points_homogeneous(H, pair_px)
                if np.all(np.isfinite(pair_proj)) and np.all(pair_w > self.MIN_HOMOGENEOUS_W):
                    meas_d = float(np.linalg.norm(pair_proj[1] - pair_proj[0]))
                    d_errs.append((meas_d - float(item["distance_m"])) ** 2)
            if d_errs:
                ref_dist_rmse = float(np.sqrt(np.mean(d_errs)))

        self.src_trap = src_pts.astype(np.float32)
        self.dst_metric = dst_pts.astype(np.float32)
        self.H = H
        self.calibrated_size = (self.frame_w, self.frame_h)
        self.status = CalibrationStatus.VALID
        self.is_calibrated = True
        self.validation_error = ""
        self.quality_metrics = CalibrationQualityMetrics(
            corner_reprojection_rmse_m=round(corner_rmse, 4),
            stopline_projection_error_m=round(stopline_err, 4),
            lane_width_consistency_error_m=round(lane_consistency_err, 4),
            ground_projection_valid=True,
            reference_point_rmse_m=round(ref_pt_rmse, 4) if ref_pt_rmse is not None else None,
            reference_distance_rmse_m=round(ref_dist_rmse, 4) if ref_dist_rmse is not None else None,
        )
        print(f"[CALIB] Loaded valid geometry from {self.calib_file or 'dict'} (status={self.status.value}).")
        return True

    def _apply_fallback_trapezoid(self, preserve_status: bool = False):
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
        self.H = _compute_homography_4pt(self.src_trap, self.dst_metric)
        self.is_calibrated = False
        if not preserve_status:
            self.status = CalibrationStatus.UNCALIBRATED
            self.validation_error = ""
            self.quality_metrics = None

    def setup_default_trapezoid(self):
        self._apply_fallback_trapezoid(preserve_status=False)

    def invalidate(self, reason: str = "Manual invalidation"):
        self._set_invalid(reason, status=CalibrationStatus.INVALID)
        self._apply_fallback_trapezoid(preserve_status=True)

    def contact_point_to_ground(self, u: float, v: float) -> Tuple[float, float, int]:
        if not (np.isfinite(u) and np.isfinite(v)):
            return float("nan"), float("nan"), -1
        proj, w = _project_points_homogeneous(self.H, np.array([[float(u), float(v)]], dtype=np.float64))
        if w[0] <= self.MIN_HOMOGENEOUS_W or not np.all(np.isfinite(proj)):
            return float("nan"), float("nan"), -1

        x_m, dist_m = float(proj[0, 0]), float(proj[0, 1])
        if not (0.0 <= x_m < self.total_width_m and 0.0 <= dist_m <= self.approach_length_m):
            return x_m, dist_m, -1

        lane_id = int(x_m / (self.total_width_m / self.num_lanes))
        return x_m, dist_m, lane_id
