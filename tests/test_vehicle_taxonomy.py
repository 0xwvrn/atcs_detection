from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock

if "cv2" not in sys.modules:
    sys.modules["cv2"] = MagicMock()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perception import traffic_analysis, vehicle_taxonomy


class TestVehicleTaxonomy(unittest.TestCase):
    def test_backward_compatible_reexports(self):
        self.assertIs(traffic_analysis.VEHICLE_TAXONOMY, vehicle_taxonomy.VEHICLE_TAXONOMY)
        self.assertIs(traffic_analysis.CATEGORY_LENGTH_M, vehicle_taxonomy.CATEGORY_LENGTH_M)
        self.assertIs(traffic_analysis.CATEGORY_PCU, vehicle_taxonomy.CATEGORY_PCU)
        self.assertIs(traffic_analysis.resolve_pcu, vehicle_taxonomy.resolve_pcu)

    def test_single_class_and_empty_inputs(self):
        self.assertEqual(vehicle_taxonomy.resolve_pcu(None, True), (None, None))
        self.assertEqual(vehicle_taxonomy.resolve_pcu("", True), (None, None))
        self.assertEqual(vehicle_taxonomy.resolve_pcu("vehicle", False), (1.0, "VEHICLE"))
        self.assertEqual(vehicle_taxonomy.resolve_pcu("person", False), (1.0, "VEHICLE"))

    def test_multiclass_pcu_weights_and_non_vehicle_rejection(self):
        cases = [
            ("car", (1.0, "CAR")),
            ("bus", (2.5, "BUS/TRUCK")),
            ("truck", (2.5, "BUS/TRUCK")),
            ("van", (1.4, "VAN")),
            ("motorcycle", (0.5, "TWO_WHEELER")),
            ("motorbike", (0.5, "TWO_WHEELER")),
            ("bicycle", (0.5, "TWO_WHEELER")),
            ("  Bicycle  ", (0.5, "TWO_WHEELER")),
            ("auto_rickshaw", (1.2, "AUTO_RICKSHAW")),
            ("person", (None, None)),
            ("traffic light", (None, None)),
        ]
        for cls_name, expected in cases:
            with self.subTest(cls_name=cls_name):
                self.assertEqual(vehicle_taxonomy.resolve_pcu(cls_name, True), expected)
                self.assertEqual(traffic_analysis.resolve_pcu(cls_name, True), expected)


if __name__ == "__main__":
    unittest.main()
