import json
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

if "cv2" not in sys.modules:
    sys.modules["cv2"] = MagicMock()

_ROOT = Path(__file__).resolve().parents[2]
_VISION = _ROOT / "vision"
_CONTROLLER = _ROOT / "controller"
for p in (str(_VISION), str(_CONTROLLER)):
    if p not in sys.path:
        sys.path.insert(0, p)

from perception import telemetry
from telemetry_schema import parse_and_validate_telemetry


class _MockSerialPort:
    def __init__(self, write_event: threading.Event):
        self.is_open = True
        self.in_waiting = 0
        self.written = []
        self._write_event = write_event

    def write(self, data: bytes):
        self.written.append(data.decode("utf-8"))
        self._write_event.set()

    def read(self, n: int) -> bytes:
        return b""

    def close(self):
        self.is_open = False


class TestTelemetryBridge(unittest.TestCase):
    def test_1_fresh_valid_observation_serializes_canonical_v1(self):
        bridge = telemetry.NonBlockingTelemetryBridge("", 115200, "N", standalone=True)
        try:
            obs = {
                "Q_stopline": 4,
                "PCU_stopline": 5.5,
                "D_tail_m": 18.2,
                "PCU_upstream": 2.0,
                "inflow_pcu": 2.0,
                "arrival_rate_pcu_s": 0.25,
            }
            raw = bridge.send_telemetry(obs, is_calibrated=True, frame_valid=True, fresh=True)
            data = json.loads(raw)
            self.assertNotIn("stream_state", data)
            self.assertEqual(data["v"], 1)
            self.assertEqual(data["approach"], "N")
            self.assertEqual(data["Q"], 4)
            self.assertAlmostEqual(data["PCU"], 5.5)
            self.assertAlmostEqual(data["D_tail_m"], 18.2)
            self.assertAlmostEqual(data["pcu_upstream"], 2.0)
            self.assertAlmostEqual(data["inflow_pcu"], 2.0)
            self.assertAlmostEqual(data["arrival_rate_pcu_s"], 0.25)
            self.assertTrue(data["calibrated"])
            self.assertTrue(data["fresh"])
            self.assertTrue(data["valid"])
            self.assertEqual(data["seq"], 1)

            ok, _, err = parse_and_validate_telemetry(raw)
            self.assertTrue(ok, msg=err)
        finally:
            bridge.close()

    def test_2_stale_or_disconnected_observation_zeros_demand(self):
        bridge = telemetry.NonBlockingTelemetryBridge("", 115200, "E", standalone=True)
        try:
            obs = {
                "Q_stopline": 7,
                "PCU_stopline": 9.0,
                "D_tail_m": 32.0,
                "PCU_upstream": 3.0,
                "inflow_pcu": 3.0,
                "arrival_rate_pcu_s": 0.4,
            }
            raw = bridge.send_telemetry(obs, is_calibrated=True, frame_valid=False, fresh=False)
            data = json.loads(raw)
            self.assertNotIn("stream_state", data)
            self.assertFalse(data["fresh"])
            self.assertFalse(data["valid"])
            self.assertEqual(data["Q"], 0)
            self.assertEqual(data["PCU"], 0.0)
            self.assertEqual(data["D_tail_m"], 0.0)
            self.assertEqual(data["pcu_upstream"], 0.0)
            self.assertEqual(data["inflow_pcu"], 0.0)
            self.assertEqual(data["arrival_rate_pcu_s"], 0.0)

            ok, parsed, err = parse_and_validate_telemetry(raw)
            self.assertTrue(ok, msg=err)
            self.assertFalse(parsed["valid"])
            self.assertFalse(parsed["fresh"])
        finally:
            bridge.close()

    def test_3_sequence_numbers_and_removed_stream_state_kwarg(self):
        bridge = telemetry.NonBlockingTelemetryBridge("", 115200, "S", standalone=True)
        try:
            obs = {"Q_stopline": 1, "PCU_stopline": 1.0, "D_tail_m": 5.0}
            seqs = [json.loads(bridge.send_telemetry(obs, is_calibrated=True))["seq"] for _ in range(3)]
            self.assertEqual(seqs, [1, 2, 3])

            with self.assertRaises(TypeError):
                bridge.send_telemetry(obs, is_calibrated=True, stream_state="HEALTHY")  # type: ignore[call-arg]
        finally:
            bridge.close()

    def test_4_non_blocking_latest_wins_queue_and_mock_serial_write(self):
        write_event = threading.Event()
        mock_port = _MockSerialPort(write_event)

        with patch.object(telemetry, "SERIAL_AVAILABLE", True), patch.object(
            telemetry, "serial", create=True
        ) as mock_serial_mod:
            mock_serial_mod.Serial.return_value = mock_port
            bridge = telemetry.NonBlockingTelemetryBridge("/dev/mock0", 115200, "W", standalone=False)
            try:
                obs1 = {"Q_stopline": 1, "PCU_stopline": 1.0, "D_tail_m": 5.0}
                obs2 = {"Q_stopline": 2, "PCU_stopline": 2.0, "D_tail_m": 9.0}
                bridge.send_telemetry(obs1, is_calibrated=True)
                bridge.send_telemetry(obs2, is_calibrated=True)

                # Wait briefly for worker thread to write at least one packet to mock serial
                self.assertTrue(write_event.wait(timeout=2.0), "Worker thread did not write to mock serial")
                time.sleep(0.05)
                self.assertGreaterEqual(len(mock_port.written), 1)
                last_pkt = json.loads(mock_port.written[-1])
                self.assertEqual(last_pkt["approach"], "W")
                self.assertIn(last_pkt["seq"], (1, 2))
                self.assertNotIn("stream_state", last_pkt)
            finally:
                bridge.close()


if __name__ == "__main__":
    unittest.main()
