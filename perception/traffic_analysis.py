"""
ATCS Traffic Analysis Module.

Backward-compatibility facade re-exporting road homography calibration,
vehicle tracking and speed estimation, stopline and upstream queue estimation,
camera stream health monitoring, and IRC-standard PCU classification.
"""

try:
    from .arrival_tracking import UpstreamArrivalTracker, _TrackArrivalRecord
    from .calibration import (
        CalibrationQualityMetrics,
        CalibrationStatus,
        RoadHomographyCalibrator,
        _compute_homography_4pt,
        _min_triplet_triangle_area,
        _project_points_homogeneous,
        _quad_cross_products,
    )
    from .camera_health import CameraHealthState, CameraStreamHealthMonitor
    from .queue_estimation import LaneAwareQueueEstimator, LaneQueue
    from .tracking import VehicleTrackState
    from .vehicle_taxonomy import CATEGORY_LENGTH_M, CATEGORY_PCU, VEHICLE_TAXONOMY, resolve_pcu
except ImportError:
    from arrival_tracking import UpstreamArrivalTracker, _TrackArrivalRecord
    from calibration import (
        CalibrationQualityMetrics,
        CalibrationStatus,
        RoadHomographyCalibrator,
        _compute_homography_4pt,
        _min_triplet_triangle_area,
        _project_points_homogeneous,
        _quad_cross_products,
    )
    from camera_health import CameraHealthState, CameraStreamHealthMonitor
    from queue_estimation import LaneAwareQueueEstimator, LaneQueue
    from tracking import VehicleTrackState
    from vehicle_taxonomy import CATEGORY_LENGTH_M, CATEGORY_PCU, VEHICLE_TAXONOMY, resolve_pcu
