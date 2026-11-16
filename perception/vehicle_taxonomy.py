"""
ATCS Vehicle Taxonomy and IRC PCU Classification Module.

Defines recognized road vehicle classes, metric bumper-to-bumper lengths,
Indian IRC Passenger Car Unit (PCU) weights, and class-to-PCU mapping.
"""

from typing import Optional, Tuple

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
    if "motorcycle" in name or "bicycle" in name or "bike" in name or "two_wheeler" in name:
        return CATEGORY_PCU["TWO_WHEELER"], "TWO_WHEELER"
    if "auto" in name or "rickshaw" in name:
        return CATEGORY_PCU["AUTO_RICKSHAW"], "AUTO_RICKSHAW"
    return CATEGORY_PCU["CAR"], "CAR"
