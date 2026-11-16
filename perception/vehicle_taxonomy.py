"""
ATCS Vehicle Taxonomy and IRC PCU Classification Module.

Defines the single authoritative canonical vehicle taxonomy, stable class IDs,
metric bumper-to-bumper lengths, Indian IRC Passenger Car Unit (PCU) weights,
source dataset label mappings, and class-to-PCU resolution.
"""

from typing import Dict, Optional, Tuple

# Canonical vehicular classes with IRC PCU weights: (id, name, pcu, length_m, display_category)
CANONICAL_CLASSES = (
    (0, "car", 1.0, 4.5, "CAR"),
    (1, "bus", 2.5, 10.0, "BUS/TRUCK"),
    (2, "motorcycle", 0.5, 1.8, "TWO_WHEELER"),
    (3, "auto_rickshaw", 1.2, 2.8, "AUTO_RICKSHAW"),
    (4, "van", 1.4, 5.0, "VAN"),
    (5, "truck", 2.5, 10.0, "BUS/TRUCK"),
    (6, "bicycle", 0.5, 1.8, "TWO_WHEELER"),
)

# Non-vehicle detector classes (identified by YOLO, excluded from vehicle PCU/queue counts)
NON_VEHICLE_CLASSES = (
    (7, "pedestrian"),
    (8, "people"),
)

CANONICAL_ID_TO_NAME: Dict[int, str] = {
    **{cid: name for cid, name, _, _, _ in CANONICAL_CLASSES},
    **{cid: name for cid, name in NON_VEHICLE_CLASSES},
}
CANONICAL_NAME_TO_ID: Dict[str, int] = {name: cid for cid, name in CANONICAL_ID_TO_NAME.items()}
CANONICAL_CLASS_PCU: Dict[str, float] = {name: pcu for _, name, pcu, _, _ in CANONICAL_CLASSES}
CANONICAL_CLASS_LENGTH_M: Dict[str, float] = {name: length for _, name, _, length, _ in CANONICAL_CLASSES}
CANONICAL_NAME_TO_CATEGORY: Dict[str, str] = {name: cat for _, name, _, _, cat in CANONICAL_CLASSES}
NON_VEHICLE_CLASS_IDS = {cid for cid, _ in NON_VEHICLE_CLASSES}

# Source dataset mappings -> canonical class name (or None when excluded)
DETRAC_SOURCE_MAP: Dict[str, Optional[str]] = {
    "car": "car",
    "bus": "bus",
    "van": "van",
    "others": None,  # Ambiguous (mixed trucks/pickups/utility); excluded from canonical classes
}
DETRAC_AMBIGUOUS_LABELS = {"others"}

VISDRONE_CATEGORIES: Dict[int, str] = {
    0: "ignored-regions",
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
    11: "others",
}

VISDRONE_SOURCE_MAP: Dict[int, Optional[str]] = {
    0: None,  # ignored-regions
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "auto_rickshaw",  # tricycle -> auto_rickshaw
    8: "auto_rickshaw",  # awning-tricycle -> auto_rickshaw
    9: "bus",
    10: "motorcycle",
    11: None,  # others (ambiguous)
}
VISDRONE_NON_VEHICLE_IDS: set = set()
VISDRONE_AMBIGUOUS_IDS = {11}

# Allowlist of recognized vehicular classes (excludes non-vehicle classes like pedestrian/people)
VEHICLE_TAXONOMY = {name for _, name, _, _, _ in CANONICAL_CLASSES} | {"vehicle"}

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


def map_source_label(dataset: str, label) -> Tuple[Optional[int], Optional[str], str]:
    """
    Maps a source dataset label to (canonical_id, canonical_name, status).

    Status is one of: 'mapped', 'ambiguous', 'non_vehicle', 'ignored', 'unsupported'.
    """
    ds = (dataset or "").strip().lower()
    if ds == "detrac":
        key = str(label).strip().lower()
        if key in DETRAC_SOURCE_MAP:
            canonical = DETRAC_SOURCE_MAP[key]
            if canonical is None:
                return None, None, "ambiguous"
            return CANONICAL_NAME_TO_ID[canonical], canonical, "mapped"
        return None, None, "unsupported"

    if ds == "visdrone":
        try:
            cat_id = int(label)
        except (TypeError, ValueError):
            return None, None, "unsupported"
        if cat_id == 0:
            return None, None, "ignored"
        if cat_id in VISDRONE_NON_VEHICLE_IDS:
            return None, None, "non_vehicle"
        if cat_id in VISDRONE_AMBIGUOUS_IDS:
            return None, None, "ambiguous"
        if cat_id in VISDRONE_SOURCE_MAP:
            canonical = VISDRONE_SOURCE_MAP[cat_id]
            return CANONICAL_NAME_TO_ID[canonical], canonical, "mapped"
        return None, None, "unsupported"

    return None, None, "unsupported"


def pcu_for_class_id(class_id: int, is_multiclass: bool = True) -> Optional[float]:
    """Returns canonical PCU weight for a vehicular class ID, None for non-vehicles, or 1.0 in single-class mode."""
    if not is_multiclass:
        return 1.0
    name = CANONICAL_ID_TO_NAME.get(int(class_id))
    if name is None:
        return None
    return CANONICAL_CLASS_PCU.get(name)


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

    if name in CANONICAL_CLASS_PCU:
        cat = CANONICAL_NAME_TO_CATEGORY[name]
        return CANONICAL_CLASS_PCU[name], cat

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
