"""
ATCS Edge Telemetry Communication Module.

Provides non-blocking threaded UART telemetry bridge to ESP32-S3 signal controller
with automatic background reconnection and state feedback parsing.
"""

import json
import math
import queue
import threading
import time
from typing import Any, Dict, Optional

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    serial = None  # type: ignore
    SERIAL_AVAILABLE = False

PROTOCOL_VERSION = 1
FIRMWARE_APPROACHES = {"N", "S", "E", "W"}
MAX_PAYLOAD_BYTES = 511
MAX_QUEUE_COUNT = 500
MAX_METRIC_FLOAT = 2000.0
MAX_ARRIVAL_RATE_PCU_S = 50.0


def _sanitize_float(val: Any, max_val: float = MAX_METRIC_FLOAT, decimals: int = 2) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
        return 0.0
    return round(min(max(0.0, float(val)), max_val), decimals)


def _sanitize_int(val: Any, max_val: int = MAX_QUEUE_COUNT) -> int:
    if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
        return 0
    return min(max(0, int(val)), max_val)


def serialize_observation_packet(
    approach_leg: str,
    smoothed_q: Dict[str, Any],
    is_calibrated: bool,
    frame_valid: bool = True,
    fresh: bool = True,
    allow_approx_calib: bool = False,
    seq: int = 0,
) -> str:
    """
    Serializes a perception observation into a Canonical Schema v1 newline-delimited JSON packet.
    Enforces:
      - Only firmware-supported approaches {"N", "S", "E", "W"} are emitted.
      - Non-finite numeric values are sanitized to 0.0 and observation marked invalid.
      - Metric-only fields (D_tail_m, pcu_upstream, inflow_pcu, arrival_rate_pcu_s) are zeroed
        when is_calibrated is False so approximate pixel geometry never enters metric demand.
      - When frame_valid or fresh is False, valid is False and demand fields are zeroed.
    """
    app = str(approach_leg).strip().upper()
    if app not in FIRMWARE_APPROACHES:
        raise ValueError(f"Unsupported firmware approach '{approach_leg}'. Must be one of {sorted(FIRMWARE_APPROACHES)}")

    raw_q = smoothed_q.get("Q_stopline", 0)
    raw_pcu = smoothed_q.get("PCU_stopline", 0.0)
    raw_tail = smoothed_q.get("D_tail_m", 0.0)
    raw_up_pcu = smoothed_q.get("PCU_upstream", 0.0)
    raw_inflow = smoothed_q.get("inflow_pcu", raw_up_pcu)
    raw_rate = smoothed_q.get("arrival_rate_pcu_s", 0.0)

    numbers_finite = all(
        not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v) and v >= 0
        for v in (raw_q, raw_pcu, raw_tail, raw_up_pcu, raw_inflow, raw_rate)
    )

    is_fresh = bool(frame_valid and fresh and numbers_finite)
    is_cal = bool(is_calibrated and numbers_finite)
    is_packet_valid = bool(is_fresh and (is_cal or allow_approx_calib))

    telemetry: Dict[str, Any] = {
        "v": PROTOCOL_VERSION,
        "approach": app,
        "Q": _sanitize_int(raw_q) if is_packet_valid else 0,
        "PCU": _sanitize_float(raw_pcu) if is_packet_valid else 0.0,
        "D_tail_m": _sanitize_float(raw_tail) if (is_packet_valid and is_cal) else 0.0,
        "pcu_upstream": _sanitize_float(raw_up_pcu) if (is_packet_valid and is_cal) else 0.0,
        "inflow_pcu": _sanitize_float(raw_inflow) if (is_packet_valid and is_cal) else 0.0,
        "arrival_rate_pcu_s": _sanitize_float(raw_rate, MAX_ARRIVAL_RATE_PCU_S, 3) if (is_packet_valid and is_cal) else 0.0,
        "calibrated": is_cal,
        "fresh": is_fresh,
        "valid": is_packet_valid,
        "seq": max(0, int(seq)),
    }
    raw_json = json.dumps(telemetry, separators=(",", ":"), allow_nan=False)
    if len(raw_json.encode("utf-8")) > MAX_PAYLOAD_BYTES:
        raise ValueError(f"Telemetry payload exceeds {MAX_PAYLOAD_BYTES} bytes limit")
    return raw_json + "\n"


class NonBlockingTelemetryBridge:
    """Thread-safe non-blocking telemetry bridge transmitting JSON queue demand over UART."""

    def __init__(
        self,
        port: str,
        baudrate: int,
        approach_leg: str = "N",
        allow_approx_calib: bool = False,
        standalone: bool = False,
    ):
        app = str(approach_leg).strip().upper()
        if app not in FIRMWARE_APPROACHES:
            raise ValueError(f"Unsupported firmware approach '{approach_leg}'. Must be one of {sorted(FIRMWARE_APPROACHES)}")
        self.port = port
        self.baudrate = baudrate
        self.approach_leg = app
        self.allow_approx_calib = allow_approx_calib
        self.standalone = bool(standalone or not port)

        self.lock = threading.Lock()
        self.serial_conn: Optional[Any] = None
        self.seq = 0
        self.controller_feedback = "STANDALONE"
        self._last_print_connected: Optional[bool] = None

        self.tx_queue: queue.Queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    @property
    def is_connected(self) -> bool:
        with self.lock:
            return self.serial_conn is not None and getattr(self.serial_conn, "is_open", False)

    def _worker_loop(self):
        last_connect_attempt = 0.0
        rx_buffer = ""

        try:
            while not self.stop_event.is_set():
                now = time.time()
                if self.standalone:
                    if self._last_print_connected is not False:
                        print("[BRIDGE] Standalone mode active (serial transmission disabled).")
                        self._last_print_connected = False
                elif not self.is_connected and (now - last_connect_attempt >= 5.0):
                    last_connect_attempt = now
                    if SERIAL_AVAILABLE:
                        try:
                            conn = serial.Serial(
                                self.port, self.baudrate, timeout=0.0, write_timeout=0.05, dsrdtr=False, rtscts=False
                            )
                            with self.lock:
                                if not self.stop_event.is_set():
                                    self.serial_conn = conn
                                else:
                                    try:
                                        conn.close()
                                    except Exception:
                                        pass
                                    self.serial_conn = None

                            if self._last_print_connected is not True and not self.stop_event.is_set():
                                print(f"[BRIDGE] Connected to ESP32-S3 on {self.port} @ {self.baudrate} baud.")
                                self._last_print_connected = True
                        except Exception:
                            with self.lock:
                                self.serial_conn = None
                            if self._last_print_connected is not False:
                                print(f"[BRIDGE] Hardware link unavailable on {self.port}. Running standalone.")
                                self._last_print_connected = False
                    else:
                        if self._last_print_connected is not False:
                            print("[BRIDGE] PySerial not available on system. Running standalone.")
                            self._last_print_connected = False

                try:
                    payload, seq_num = self.tx_queue.get(timeout=0.1)
                except queue.Empty:
                    payload = None

                with self.lock:
                    conn = self.serial_conn

                if conn and getattr(conn, "is_open", False):
                    try:
                        if payload is not None:
                            conn.write(payload.encode("utf-8"))
                        if conn.in_waiting > 0:
                            rx_buffer += conn.read(conn.in_waiting).decode("utf-8", errors="ignore")
                            if "\n" in rx_buffer:
                                lines = rx_buffer.split("\n")
                                rx_buffer = lines[-1]
                                for line in lines[:-1]:
                                    if line.strip():
                                        self.controller_feedback = line.strip()
                    except Exception as e:
                        print(f"[BRIDGE] Serial error: {e}. Reconnection queued.")
                        self.safe_close()
                elif payload is not None and self._last_print_connected is False:
                    print(f"[STANDALONE TELEMETRY (Seq: {seq_num})] {payload.strip()}")
        finally:
            self.safe_close()

    def send_telemetry(
        self,
        smoothed_q: dict,
        is_calibrated: bool,
        frame_valid: bool = True,
        fresh: bool = True,
    ) -> str:
        self.seq += 1
        json_str = serialize_observation_packet(
            approach_leg=self.approach_leg,
            smoothed_q=smoothed_q,
            is_calibrated=is_calibrated,
            frame_valid=frame_valid,
            fresh=fresh,
            allow_approx_calib=self.allow_approx_calib,
            seq=self.seq,
        )

        if self.tx_queue.full():
            try:
                self.tx_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.tx_queue.put_nowait((json_str, self.seq))
        except queue.Full:
            pass
        return json_str

    def safe_close(self):
        with self.lock:
            conn = self.serial_conn
            self.serial_conn = None
        if conn:
            try:
                if getattr(conn, "is_open", False):
                    conn.close()
            except Exception:
                pass

    def close(self):
        self.stop_event.set()
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        self.safe_close()
