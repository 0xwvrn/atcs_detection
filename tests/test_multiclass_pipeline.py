"""
Automated tests for Stage 6A multiclass taxonomy, dataset conversion,
preprocessing, PCU-weighted evaluation, and perception integration.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from PIL import Image

_VISION_ROOT = Path(__file__).resolve().parent.parent
if str(_VISION_ROOT) not in sys.path:
    sys.path.insert(0, str(_VISION_ROOT))

from converters.convert_detrac import process_split as convert_detrac_split
from converters.convert_visdrone import extract_and_convert_split as convert_visdrone_split
from perception.arrival_tracking import UpstreamArrivalTracker
from perception.queue_estimation import LaneAwareQueueEstimator
from perception.telemetry import serialize_observation_packet
from perception.tracking import VehicleTrackState
from perception.vehicle_taxonomy import (
    CANONICAL_CLASSES,
    CANONICAL_CLASS_PCU,
    CANONICAL_ID_TO_NAME,
    CANONICAL_NAME_TO_ID,
    map_source_label,
    pcu_for_class_id,
    resolve_pcu,
)
from training.prepare_multiclass_dataset import audit_yolo_dataset, prepare_dataset
from training.val import compute_pcu_evaluation_metrics


class TestCanonicalTaxonomy(unittest.TestCase):
    def test_stable_class_ids_and_pcu(self):
        expected_vehicle_ids = {
            0: ("car", 1.0),
            1: ("bus", 2.5),
            2: ("motorcycle", 0.5),
            3: ("auto_rickshaw", 1.2),
            4: ("van", 1.4),
            5: ("truck", 2.5),
            6: ("bicycle", 0.5),
        }
        self.assertEqual(len(CANONICAL_CLASSES), 7)
        self.assertEqual(len(CANONICAL_ID_TO_NAME), 9)
        for cid, (name, pcu) in expected_vehicle_ids.items():
            self.assertEqual(CANONICAL_ID_TO_NAME[cid], name)
            self.assertEqual(CANONICAL_NAME_TO_ID[name], cid)
            self.assertEqual(CANONICAL_CLASS_PCU[name], pcu)
            self.assertEqual(pcu_for_class_id(cid, is_multiclass=True), pcu)
        self.assertEqual(CANONICAL_ID_TO_NAME[7], "pedestrian")
        self.assertEqual(CANONICAL_ID_TO_NAME[8], "people")
        self.assertIsNone(pcu_for_class_id(7, is_multiclass=True))
        self.assertIsNone(pcu_for_class_id(8, is_multiclass=True))

    def test_unknown_and_ambiguous_handling(self):
        self.assertEqual(resolve_pcu("person", is_multiclass=True), (None, None))
        self.assertEqual(resolve_pcu("pedestrian", is_multiclass=True), (None, None))
        self.assertEqual(resolve_pcu("people", is_multiclass=True), (None, None))
        self.assertEqual(resolve_pcu("tricycle", is_multiclass=True), (None, None))
        self.assertEqual(resolve_pcu("auto_rickshaw", is_multiclass=True), (1.2, "AUTO_RICKSHAW"))
        self.assertEqual(resolve_pcu("motorcycle", is_multiclass=True), (0.5, "TWO_WHEELER"))
        # Single-class backward compatibility
        self.assertEqual(resolve_pcu("vehicle", is_multiclass=False), (1.0, "VEHICLE"))
        self.assertEqual(pcu_for_class_id(0, is_multiclass=False), 1.0)

    def test_source_dataset_mappings(self):
        self.assertEqual(map_source_label("detrac", "car"), (0, "car", "mapped"))
        self.assertEqual(map_source_label("detrac", "bus"), (1, "bus", "mapped"))
        self.assertEqual(map_source_label("detrac", "van"), (4, "van", "mapped"))
        self.assertEqual(map_source_label("detrac", "others"), (None, None, "ambiguous"))
        self.assertEqual(map_source_label("detrac", "ufo"), (None, None, "unsupported"))

        self.assertEqual(map_source_label("visdrone", 1), (7, "pedestrian", "mapped"))
        self.assertEqual(map_source_label("visdrone", 2), (8, "people", "mapped"))
        self.assertEqual(map_source_label("visdrone", 4), (0, "car", "mapped"))
        self.assertEqual(map_source_label("visdrone", 9), (1, "bus", "mapped"))
        self.assertEqual(map_source_label("visdrone", 10), (2, "motorcycle", "mapped"))
        self.assertEqual(map_source_label("visdrone", 5), (4, "van", "mapped"))
        self.assertEqual(map_source_label("visdrone", 6), (5, "truck", "mapped"))
        self.assertEqual(map_source_label("visdrone", 3), (6, "bicycle", "mapped"))
        self.assertEqual(map_source_label("visdrone", 7), (3, "auto_rickshaw", "mapped"))
        self.assertEqual(map_source_label("visdrone", 8), (3, "auto_rickshaw", "mapped"))
        self.assertEqual(map_source_label("visdrone", 11), (None, None, "ambiguous"))
        self.assertEqual(map_source_label("visdrone", 0), (None, None, "ignored"))


class TestDatasetConversionAndPreprocessing(unittest.TestCase):
    def _make_dummy_image(self, path: Path, size=(200, 100), color=(20, 40, 60)):
        path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", size, color=color)
        # Draw a unique pixel patch so JPEG compression preserves distinctness
        img.putpixel((10, 10), (color[0] ^ 255, color[1] ^ 127, color[2] ^ 63))
        img.save(path)

    def test_detrac_conversion_canonical_and_empty_and_ambiguous(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            xml_dir = root / "xml"
            img_dir = root / "imgs" / "MVI_00001"
            out_dir = root / "out"
            xml_dir.mkdir(parents=True)
            self._make_dummy_image(img_dir / "img00001.jpg", size=(960, 540))
            self._make_dummy_image(img_dir / "img00002.jpg", size=(960, 540))

            xml_content = """<sequence name="MVI_00001">
  <ignored_region><box left="0" top="0" width="50" height="50"/></ignored_region>
  <frame num="1">
    <target_list>
      <target id="1"><box left="100" top="100" width="96" height="54"/><attribute vehicle_type="car" truncation_ratio="0.0"/></target>
      <target id="2"><box left="200" top="100" width="120" height="80"/><attribute vehicle_type="bus" truncation_ratio="0.0"/></target>
      <target id="3"><box left="350" top="100" width="100" height="60"/><attribute vehicle_type="van" truncation_ratio="0.0"/></target>
      <target id="4"><box left="500" top="100" width="80" height="50"/><attribute vehicle_type="others" truncation_ratio="0.0"/></target>
      <target id="5"><box left="600" top="100" width="0.5" height="50"/><attribute vehicle_type="car" truncation_ratio="0.0"/></target>
    </target_list>
  </frame>
  <frame num="2"></frame>
</sequence>"""
            (xml_dir / "MVI_00001.xml").write_text(xml_content)

            stats = convert_detrac_split("detrac_train", xml_dir, root / "imgs", out_dir, 1, 960.0, 540.0, False, "multiclass")
            self.assertEqual(stats["images_processed"], 2)
            self.assertEqual(stats["empty_label_images"], 1)
            self.assertEqual(stats["annotations_processed"], 3)
            self.assertEqual(stats["annotations_rejected"], 1)
            self.assertEqual(stats["excluded_ambiguous"], {"detrac:others": 1})

            lbl1 = (out_dir / "labels" / "detrac_train" / "MVI_00001_img00001.txt").read_text().splitlines()
            self.assertEqual(len(lbl1), 3)
            cls_ids = [int(line.split()[0]) for line in lbl1]
            self.assertEqual(cls_ids, [0, 1, 4])
            # Verify first box coords: left=100, top=100, w=96, h=54 on 960x540 -> cx=148/960, cy=127/540, nw=0.1, nh=0.1
            parts = [float(x) for x in lbl1[0].split()]
            self.assertAlmostEqual(parts[1], 148.0 / 960.0, places=5)
            self.assertAlmostEqual(parts[2], 127.0 / 540.0, places=5)
            self.assertAlmostEqual(parts[3], 0.1, places=5)
            self.assertAlmostEqual(parts[4], 0.1, places=5)

    def test_visdrone_conversion_and_preprocessing_no_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            detrac_dir = root / "DETRAC"
            visdrone_dir = root / "VisDrone"
            out_dir = root / "dataset_multiclass"
            yaml_path = root / "data_multiclass.yaml"

            # Minimal DETRAC: 2 train seqs, 1 test seq
            for split_offset, (split_xml, seqs) in enumerate([("DETRAC-Train-Annotations-XML", ["MVI_10001", "MVI_10002"]), ("DETRAC-Test-Annotations-XML", ["MVI_20001"])]):
                (detrac_dir / split_xml).mkdir(parents=True, exist_ok=True)
                for idx, seq in enumerate(seqs):
                    c = ((split_offset * 100 + idx * 40 + 15) % 255, 20 + split_offset * 30, 30)
                    self._make_dummy_image(detrac_dir / "DETRAC-Images" / seq / "img00001.jpg", size=(200, 100), color=c)
                    (detrac_dir / split_xml / f"{seq}.xml").write_text(
                        f'<sequence name="{seq}"><frame num="1"><target_list>'
                        '<target id="1"><box left="20" top="20" width="40" height="20"/><attribute vehicle_type="car" truncation_ratio="0.0"/></target>'
                        '</target_list></frame></sequence>'
                    )

            # Minimal VisDrone: 2 train seqs, 1 val seq
            for split_offset, (vd_split, seqs) in enumerate([("VisDrone2019-DET-train", ["00001_00001_d", "00002_00001_d"]), ("VisDrone2019-DET-val", ["00003_00001_d"])]):
                (visdrone_dir / vd_split / "images").mkdir(parents=True, exist_ok=True)
                (visdrone_dir / vd_split / "annotations").mkdir(parents=True, exist_ok=True)
                for idx, stem in enumerate(seqs):
                    c = ((100 + split_offset * 70 + idx * 35) % 255, 50 + split_offset * 40, 10)
                    self._make_dummy_image(visdrone_dir / vd_split / "images" / f"{stem}.jpg", size=(200, 100), color=c)
                    (visdrone_dir / vd_split / "annotations" / f"{stem}.txt").write_text(
                        "10,10,40,20,1,4,0,0\n"   # car -> 0
                        "60,10,30,20,1,10,0,0\n"  # motor -> 2
                        "90,10,20,20,1,7,0,0\n"   # tricycle -> ambiguous
                        "120,10,10,20,1,1,0,0\n"  # pedestrian -> non-vehicle
                        "140,10,20,20,0,4,0,0\n"  # score=0 -> ignored
                    )

            manifest = prepare_dataset(
                detrac_dir=detrac_dir,
                visdrone_dir=visdrone_dir,
                output_dir=out_dir,
                yaml_path=yaml_path,
                stride=1,
                val_ratio=0.5,
                seed=42,
                symlink=False,
            )
            self.assertEqual(manifest["audit"]["cross_split_duplicate_count"], 0)
            self.assertEqual(manifest["audit"]["splits"]["train"]["images"], 2)
            self.assertEqual(manifest["audit"]["splits"]["val"]["images"], 2)
            self.assertEqual(manifest["audit"]["splits"]["test"]["images"], 2)
            self.assertTrue(yaml_path.exists())
            self.assertIn("nc: 9", yaml_path.read_text())
            self.assertGreater(manifest["audit"]["total_class_counts"]["auto_rickshaw"], 0)
            self.assertGreater(manifest["audit"]["total_class_counts"]["pedestrian"], 0)
            # Ensure no DETRAC train sequence appears in val
            tr_seqs = set(manifest["split_sequences"]["detrac_train_seqs"])
            va_seqs = set(manifest["split_sequences"]["detrac_val_seqs"])
            self.assertEqual(len(tr_seqs & va_seqs), 0)

            # Corrupt a label intentionally to test audit detection
            bad_lbl = out_dir / "labels" / "train" / sorted((out_dir / "labels" / "train").glob("*.txt"))[0].name
            bad_lbl.write_text("99 0.5 0.5 0.1 0.1\n0 0.5 0.5 0.0 0.1\n0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n")
            bad_audit = audit_yolo_dataset(out_dir, check_hashes=False)
            self.assertEqual(bad_audit["splits"]["train"]["invalid_class_ids"], 1)
            self.assertEqual(bad_audit["splits"]["train"]["zero_area_boxes"], 1)
            self.assertEqual(bad_audit["splits"]["train"]["duplicate_annotations"], 1)


class TestPCUEvaluationAndPerceptionIntegration(unittest.TestCase):
    def test_pcu_weighted_evaluation_metrics(self):
        # img1: GT=[car(1.0), bus(2.5)] -> 3.5; Pred=[car(1.0), car(1.0)] -> 2.0 (abs err 1.5)
        # img2: GT=[motorcycle(0.5), auto_rickshaw(1.2)] -> 1.7; Pred=[motorcycle(0.5), auto_rickshaw(1.2)] -> 1.7 (abs err 0.0)
        gt = {"img1": [0, 1], "img2": [2, 3]}
        pred = {"img1": [0, 0], "img2": [2, 3]}
        res = compute_pcu_evaluation_metrics(gt, pred, is_multiclass=True)
        self.assertAlmostEqual(res["total_gt_pcu"], 5.2, places=4)
        self.assertAlmostEqual(res["total_pred_pcu"], 3.7, places=4)
        self.assertAlmostEqual(res["absolute_pcu_error_sum"], 1.5, places=4)
        self.assertAlmostEqual(res["mean_absolute_pcu_error_per_image"], 0.75, places=4)
        self.assertAlmostEqual(res["per_class_pcu_error"]["bus"]["signed_pcu_error"], -2.5, places=4)
        self.assertAlmostEqual(res["per_class_pcu_error"]["car"]["signed_pcu_error"], 1.0, places=4)

    def test_multiclass_perception_to_telemetry_and_single_class_compat(self):
        estimator = LaneAwareQueueEstimator(max_vehicle_gap_m=6.0, stopline_threshold_m=14.0)
        tracker = UpstreamArrivalTracker()
        tracks = {}

        # Stationary bus (2.5 PCU) + stationary auto_rickshaw (1.2 PCU) + stationary motorcycle (0.5 PCU)
        detections = [
            (1, "bus", (0.0, 4.0)),
            (2, "auto_rickshaw", (0.0, 10.0)),
            (3, "motorcycle", (2.0, 5.0)),
        ]
        vehicles_by_lane = {0: [], 1: []}
        for tid, cname, pos in detections:
            pcu, cat = resolve_pcu(cname, is_multiclass=True)
            ts = VehicleTrackState(tid, cat, pcu)
            for t in (0.0, 0.5, 1.0, 1.5):
                ts.update_position(pos[0], pos[1], t)
            tracks[tid] = ts
            lane_id = 0 if pos[0] < 1.0 else 1
            vehicles_by_lane[lane_id].append((pos[1], ts.pcu, tid, (0, 0, 10, 10), ts.class_name))

        tracker.update(tracks, 1.5)
        arrival_rate = tracker.get_arrival_rate(1.5)
        lane_qs = estimator.estimate(vehicles_by_lane, [0, 1])
        q_stopline = sum(lq.count for lq in lane_qs.values())
        pcu_stopline = sum(lq.pcu for lq in lane_qs.values())
        tail_stopline = max((lq.tail_m for lq in lane_qs.values()), default=0.0)
        q_upstream = sum(lq.upstream_count for lq in lane_qs.values())
        pcu_upstream = sum(lq.upstream_pcu for lq in lane_qs.values())
        smoothed = estimator.smooth(q_stopline, pcu_stopline, tail_stopline, q_upstream, pcu_upstream, arrival_rate, metric_valid=True)
        self.assertEqual(smoothed["Q_stopline"], 3)
        self.assertAlmostEqual(smoothed["PCU_stopline"], 4.2, places=2)  # 2.5 + 1.2 + 0.5

        pkt_str = serialize_observation_packet("N", smoothed, is_calibrated=True, frame_valid=True, fresh=True, seq=1)
        pkt = json.loads(pkt_str)
        self.assertEqual(pkt["Q"], 3)
        self.assertAlmostEqual(pkt["PCU"], 4.2, places=2)

        # Single-class backward compatibility check
        sc_pcu, sc_cat = resolve_pcu("vehicle", is_multiclass=False)
        self.assertEqual((sc_pcu, sc_cat), (1.0, "VEHICLE"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
