"""
ATCS Edge Telemetry Communication Module.

Provides non-blocking threaded UART telemetry bridge to ESP32-S3 signal controller
with automatic background reconnection and state feedback parsing.
"""

import json
import queue
import threading
import time
from typing import Any, Optional

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    serial = None  # type: ignore
    SERIAL_AVAILABLE = False


class NonBlockingTelemetryBridge:
    """Thread-safe non-blocking telemetry bridge transmitting JSON queue demand over UART."""

    def __init__(self, port: str, baudrate: int, approach_leg: str = "N", allow_approx_calib: bool = False):
        self.port = port
        self.baudrate = baudrate
        self.approach_leg = approach_leg
        self.allow_approx_calib = allow_approx_calib

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
                if not self.is_connected and (now - last_connect_attempt >= 5.0):
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
                            print(f"[BRIDGE] PySerial not available on system. Running standalone.")
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
                    print(f"[TELEMETRY TX (Seq: {seq_num})] {payload.strip()}")
        finally:
            self.safe_close()

    def send_telemetry(self, smoothed_q: dict, is_calibrated: bool, frame_valid: bool = True):
        self.seq += 1
        is_packet_valid = bool(frame_valid and (is_calibrated or self.allow_approx_calib))
        telemetry = {
            "approach": self.approach_leg,
            "Q": int(smoothed_q["Q_stopline"]),
            "PCU": round(float(smoothed_q["PCU_stopline"]), 2),
            "D_tail_m": round(float(smoothed_q["D_tail_m"]), 2),
            "valid": is_packet_valid,
        }
        json_str = json.dumps(telemetry) + "\n"

        if self.tx_queue.full():
            try:
                self.tx_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.tx_queue.put_nowait((json_str, self.seq))
        except queue.Full:
            pass

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
