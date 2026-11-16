import glob
import importlib
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import MagicMock, patch

if isinstance(sys.modules.get("cv2"), MagicMock):
    del sys.modules["cv2"]

try:
    import cv2
except ImportError:
    cv2 = MagicMock()
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_VISION = _ROOT / "vision"
_CONTROLLER = _ROOT / "controller"
for p in (str(_VISION), str(_CONTROLLER)):
    if p not in sys.path:
        sys.path.insert(0, p)

from perception import pipeline, telemetry
if not isinstance(cv2, MagicMock):
    importlib.reload(pipeline)
from telemetry_schema import parse_and_validate_telemetry


class _FakeTensor:
    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def cpu(self):
        return self

    def int(self):
        return _FakeTensor(self._arr.astype(np.int32))

    def numpy(self):
        return self._arr


class _FakeBoxes:
    def __init__(self, detections):
        if detections:
            xyxy = np.array([d[0] for d in detections], dtype=np.float32)
            tids = np.array([d[1] for d in detections], dtype=np.int32)
            clss = np.array([d[2] for d in detections], dtype=np.int32)
        else:
            xyxy = np.zeros((0, 4), dtype=np.float32)
            tids = np.zeros((0,), dtype=np.int32)
            clss = np.zeros((0,), dtype=np.int32)
        self.xyxy = _FakeTensor(xyxy)
        self.id = _FakeTensor(tids) if detections else None
        self.cls = _FakeTensor(clss)


class _FakeResult:
    def __init__(self, detections):
        self.boxes = _FakeBoxes(detections)


class _FakeSerialPort:
    def __init__(self):
        self.is_open = True
        self.in_waiting = 0
        self.lines = []
        self._lock = threading.Lock()

    def write(self, data: bytes):
        text = data.decode("utf-8")
        with self._lock:
            for line in text.splitlines():
                if line.strip():
                    self.lines.append(line.strip())

    def read(self, n: int) -> bytes:
        return b""

    def close(self):
        self.is_open = False


def _load_or_build_frames(count: int, freeze: bool = False):
    detrac_dir = _ROOT / "data" / "DETRAC" / "DETRAC-Images" / "MVI_20011"
    paths = sorted(glob.glob(str(detrac_dir / "img*.jpg")))
    if paths:
        base = cv2.imread(paths[0])
        if freeze:
            return [base.copy() for _ in range(count)]
        frames = []
        for idx in range(count):
            img = cv2.imread(paths[idx % len(paths)])
            frames.append(img if img is not None else base.copy())
        return frames

    rng = np.random.default_rng(42)
    base = rng.integers(30, 200, size=(540, 960, 3), dtype=np.uint8)
    if freeze:
        return [base.copy() for _ in range(count)]
    return [rng.integers(30, 200, size=(540, 960, 3), dtype=np.uint8) for _ in range(count)]


def _write_temp_mp4(path: Path, frames, fps: float = 10.0):
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    assert writer.isOpened(), "Failed to open cv2.VideoWriter"
    for f in frames:
        writer.write(f)
    writer.release()


def _make_calib_json(path: Path, w: int = 960, h: int = 540):
    # Matches RoadHomographyCalibrator default trapezoid proportions for 960x540
    # [upstream-left, upstream-right, stopline-right, stopline-left]
    data = {
        "image_width": w,
        "image_height": h,
        "image_points": [
            [int(w * 0.35), int(h * 0.30)],
            [int(w * 0.65), int(h * 0.30)],
            [int(w * 0.90), int(h * 0.92)],
            [int(w * 0.10), int(h * 0.92)],
        ],
        "ground_points_metric": [
            [0.0, 50.0],
            [7.0, 50.0],
            [7.0, 0.0],
            [0.0, 0.0],
        ],
    }
    path.write_text(json.dumps(data))


@unittest.skipIf(isinstance(cv2, MagicMock), "Real OpenCV required for recorded-video test")
class TestRecordedPipelineEndToEnd(unittest.TestCase):
    def _run_pipeline_with_mock_yolo(self, video_path: Path, calib_file: str, names_dict, frame_detections_fn):
        fake_serial = _FakeSerialPort()
        frame_counter = {"n": 0}

        class _FakeYOLO:
            def __init__(self, weights):
                self.names = names_dict

            def track(self, frame, **kwargs):
                idx = frame_counter["n"]
                frame_counter["n"] += 1
                time.sleep(0.08)  # Ensure real monotonic dwell & telemetry interval progression
                return [_FakeResult(frame_detections_fn(idx))]

        fake_ultralytics = types.SimpleNamespace(YOLO=_FakeYOLO)
        args = types.SimpleNamespace(
            weights="mock.pt",
            source=str(video_path),
            tracker="bytetrack.yaml",
            conf=0.25,
            iou=0.45,
            imgsz=[960],
            device="",
            serial_port="/dev/mock0",
            standalone=False,
            baudrate=115200,
            approach_leg="N",
            calib_file=calib_file,
            num_lanes=2,
            max_vehicle_gap_m=6.0,
            stopline_threshold_m=14.0,
            allow_approx_calib=False,
            telemetry_rate=50.0,
            no_show=True,
        )

        with patch.dict(sys.modules, {"ultralytics": fake_ultralytics}), \
             patch.object(sys, "argv", ["pipeline.py", "--serial-port", "/dev/mock0"]), \
             patch.object(telemetry, "SERIAL_AVAILABLE", True), \
             patch.object(telemetry, "serial", create=True) as mock_serial_mod:
            mock_serial_mod.Serial.return_value = fake_serial
            pipeline.run_perception(args)

        return [json.loads(line) for line in fake_serial.lines]

    def test_1_recorded_video_calibrated_single_class_detection_to_telemetry(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            video_path = tmp_dir / "clip.mp4"
            calib_path = tmp_dir / "calib.json"
            _write_temp_mp4(video_path, _load_or_build_frames(18, freeze=False), fps=10.0)
            _make_calib_json(calib_path)

            # Vehicle 1 stationary near stopline (x=280..360, y=410..470); Vehicle 2 moving upstream
            def dets(idx):
                up_y = 180 + idx * 5
                return [
                    ([280, 410, 360, 470], 1, 0),
                    ([540, up_y, 600, up_y + 40], 2, 0),
                ]

            packets = self._run_pipeline_with_mock_yolo(video_path, str(calib_path), {0: "vehicle"}, dets)
            self.assertGreater(len(packets), 0)
            for pkt in packets:
                ok, _, err = parse_and_validate_telemetry(json.dumps(pkt))
                self.assertTrue(ok, msg=err)
                self.assertEqual(pkt["approach"], "N")
                self.assertTrue(pkt["calibrated"])
                self.assertTrue(pkt["fresh"])
                self.assertTrue(pkt["valid"])

            # After > 1.0s dwell, stationary vehicle 1 enters stopline queue
            last = packets[-1]
            self.assertGreaterEqual(last["Q"], 1)
            self.assertGreaterEqual(last["PCU"], 1.0)
            self.assertGreater(last["D_tail_m"], 0.0)

    def test_2_uncalibrated_pipeline_suppresses_metrics_and_marks_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_path = Path(tmp) / "clip.mp4"
            _write_temp_mp4(video_path, _load_or_build_frames(6, freeze=False), fps=10.0)

            packets = self._run_pipeline_with_mock_yolo(
                video_path,
                "",
                {0: "vehicle"},
                lambda idx: [([280, 410, 360, 470], 1, 0)],
            )
            self.assertGreater(len(packets), 0)
            for pkt in packets:
                ok, _, err = parse_and_validate_telemetry(json.dumps(pkt))
                self.assertTrue(ok, msg=err)
                self.assertFalse(pkt["calibrated"])
                self.assertFalse(pkt["valid"])
                self.assertEqual(pkt["Q"], 0)
                self.assertEqual(pkt["PCU"], 0.0)
                self.assertEqual(pkt["D_tail_m"], 0.0)
                self.assertEqual(pkt["inflow_pcu"], 0.0)
                self.assertEqual(pkt["arrival_rate_pcu_s"], 0.0)

    def test_3_frozen_recorded_stream_triggers_stale_and_zeros_telemetry(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            calib_path = tmp_dir / "calib.json"
            frozen_frame = _load_or_build_frames(1, freeze=True)[0]
            for i in range(22):
                cv2.imwrite(str(tmp_dir / f"frozen_{i:02d}.png"), frozen_frame)
            _make_calib_json(calib_path)

            packets = self._run_pipeline_with_mock_yolo(
                tmp_dir / "frozen_%02d.png",
                str(calib_path),
                {0: "vehicle"},
                lambda idx: [([280, 410, 360, 470], 1, 0)],
            )
            self.assertGreater(len(packets), 0)
            # After frozen_frame_threshold (12) + min_freeze_duration_s (1.0s), stream becomes STALE -> fresh=False, valid=False
            stale_pkts = [p for p in packets if (not p["fresh"] and not p["valid"])]
            self.assertGreater(len(stale_pkts), 0)
            last = stale_pkts[-1]
            self.assertFalse(last["fresh"])
            self.assertFalse(last["valid"])
            self.assertEqual(last["Q"], 0)
            self.assertEqual(last["PCU"], 0.0)
            self.assertEqual(last["D_tail_m"], 0.0)

    def test_4_multiclass_model_pcu_weights_propagate_to_telemetry(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            video_path = tmp_dir / "multi.mp4"
            calib_path = tmp_dir / "calib.json"
            _write_temp_mp4(video_path, _load_or_build_frames(16, freeze=False), fps=10.0)
            _make_calib_json(calib_path)

            names = {0: "car", 1: "bicycle", 2: "bus", 3: "auto_rickshaw"}

            # 4 stationary vehicles near stopline in lane 0 & lane 1 -> 1.0 + 0.5 + 2.5 + 1.2 = 5.2 PCU
            def dets(idx):
                return [
                    ([270, 430, 330, 475], 1, 0),  # car (1.0)
                    ([340, 430, 380, 475], 2, 1),  # bicycle (0.5)
                    ([520, 430, 600, 475], 3, 2),  # bus (2.5)
                    ([610, 430, 660, 475], 4, 3),  # auto_rickshaw (1.2)
                ]

            packets = self._run_pipeline_with_mock_yolo(video_path, str(calib_path), names, dets)
            self.assertGreater(len(packets), 0)
            last = packets[-1]
            self.assertTrue(last["valid"])
            self.assertEqual(last["Q"], 4)
            self.assertAlmostEqual(last["PCU"], 5.2, places=2)


if __name__ == "__main__":
    unittest.main()
