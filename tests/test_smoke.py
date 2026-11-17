"""Camera-free smoke test for tracking, counting, config, and drawing."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ubuncam.config import ConfigError, apply_thread_limits, load_config  # noqa: E402
from ubuncam.dashboard import self_check_dashboard, self_check_hourly  # noqa: E402
from ubuncam.pipeline import self_check, self_check_bus_captures, self_check_inference_gate, self_check_log  # noqa: E402


class SmokeTest(unittest.TestCase):
    def test_self_check(self):
        result = self_check()
        self.assertEqual(result["counts"]["bus"], 1)
        self.assertEqual(result["counts"]["car"], 3)
        self.assertTrue(result["drawn"])

    def test_log_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            lines = self_check_log(Path(tmp))
        self.assertEqual(lines, 2)

    def test_example_config(self):
        cfg = load_config(ROOT / "config.example.yaml", use_env=False)
        self.assertEqual(cfg.camera_device, "/dev/video0")
        self.assertEqual(cfg.classes, ("car", "truck", "bus"))
        self.assertAlmostEqual(cfg.confidence, 0.40)
        self.assertEqual(cfg.imgsz, 320)
        self.assertAlmostEqual(cfg.infer_fps, 5.0)
        self.assertEqual(cfg.threads, 4)
        self.assertTrue(str(cfg.log_path).endswith("logs/events.jsonl"))
        self.assertTrue(str(cfg.weights).endswith("models/yolo11n.onnx"))
        self.assertTrue(cfg.dashboard_enabled)
        self.assertEqual(cfg.dashboard_host, "127.0.0.1")
        self.assertEqual(cfg.dashboard_port, 8765)
        self.assertTrue(str(cfg.captures_dir).endswith("logs/captures"))

    def test_infer_defaults_and_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text("camera: {device: /dev/video1}\n", encoding="utf-8")
            cfg = load_config(path, use_env=False)
            self.assertEqual(cfg.imgsz, 320)
            self.assertAlmostEqual(cfg.infer_fps, 5.0)
            self.assertEqual(cfg.threads, 4)

            path.write_text("model: {infer_fps: 2.5, threads: 1, imgsz: 640}\n", encoding="utf-8")
            cfg = load_config(path, use_env=False)
            self.assertEqual(cfg.imgsz, 640)
            self.assertAlmostEqual(cfg.infer_fps, 2.5)
            self.assertEqual(cfg.threads, 1)

            for text in (
                "model: {infer_fps: 0}\n",
                "model: {infer_fps: -1}\n",
                "model: {infer_fps: true}\n",
                "model: {threads: 0}\n",
                "model: {threads: 1.5}\n",
            ):
                path.write_text(text, encoding="utf-8")
                with self.assertRaises(ConfigError):
                    load_config(path, use_env=False)

    def test_thread_limits_env(self):
        keys = (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "NUMEXPR_MAX_THREADS",
            "OPENCV_FOR_THREADS_NUM",
            "ORT_INTRA_OP_NUM_THREADS",
            "ORT_INTER_OP_NUM_THREADS",
            "OMP_WAIT_POLICY",
        )
        import cv2

        previous = {key: os.environ.get(key) for key in keys}
        previous_threads = cv2.getNumThreads()
        try:
            apply_thread_limits(4)
            self.assertEqual(os.environ["OMP_NUM_THREADS"], "4")
            self.assertEqual(os.environ["ORT_INTRA_OP_NUM_THREADS"], "4")
            self.assertEqual(os.environ["ORT_INTER_OP_NUM_THREADS"], "1")
            self.assertEqual(os.environ["OPENCV_FOR_THREADS_NUM"], "4")
            self.assertEqual(os.environ["OMP_WAIT_POLICY"], "PASSIVE")
            self.assertEqual(cv2.getNumThreads(), 4)
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            cv2.setNumThreads(previous_threads)

    def test_tracker_updates_only_on_inference(self):
        result = self_check_inference_gate()
        self.assertEqual(result["counts"], {"car": 2, "truck": 0, "bus": 0})
        self.assertEqual(result["overlays"], 4)
        self.assertEqual(result["inferences"], 5)
        self.assertTrue(result["held_track"])

    def test_infer_worker_one_inflight_and_schedule(self):
        import threading
        import time

        import numpy as np

        from ubuncam.__main__ import _InferWorker, next_infer_at

        self.assertEqual(next_infer_at(10.0, 0.0, True, 0.2), 10.2)
        self.assertEqual(next_infer_at(10.5, 10.2, False, 0.2), 10.2)
        self.assertEqual(next_infer_at(10.5, 10.2, True, 0.2), 10.7)

        class Gate:
            def __init__(self):
                self.calls = 0
                self.entered = threading.Event()
                self.block = threading.Event()
                self.seen: list[int] = []

            def detect(self, frame):
                self.calls += 1
                self.seen.append(int(frame[0, 0, 0]))
                self.entered.set()
                if not self.block.wait(2):
                    raise TimeoutError("inference was not released")
                return []

        gate = Gate()
        worker = _InferWorker(gate)
        first = np.zeros((2, 2, 3), dtype=np.uint8)
        second = np.zeros((2, 2, 3), dtype=np.uint8)
        second[0, 0, 0] = 7
        try:
            self.assertTrue(worker.submit(first))
            self.assertTrue(gate.entered.wait(2))
            first[0, 0, 0] = 9
            self.assertFalse(worker.submit(second))
            self.assertIsNone(worker.poll())
            gate.block.set()
            result = None
            deadline = time.time() + 2
            while result is None and time.time() < deadline:
                result = worker.poll()
                time.sleep(0.005)
            self.assertIsNotNone(result)
            detections, frame, elapsed = result
            self.assertEqual(detections, [])
            self.assertEqual(gate.seen, [0])
            self.assertEqual(int(frame[0, 0, 0]), 0)
            self.assertGreaterEqual(elapsed, 0.0)
            self.assertEqual(gate.calls, 1)
            gate.entered.clear()
            self.assertTrue(worker.submit(second))
            self.assertTrue(gate.entered.wait(2))
            self.assertEqual(gate.calls, 2)
            self.assertEqual(gate.seen[-1], 7)
        finally:
            gate.block.set()
            worker.close()

    def test_rejects_custom_class(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text("classes: [car, double-decker]\n", encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(path, use_env=False)

    def test_env_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(
                "camera: {device: /dev/video0}\nmodel: {confidence: 0.4}\nlog: {path: logs/events.jsonl}\n",
                encoding="utf-8",
            )
            previous = {key: os.environ.get(key) for key in ("UBUNCAM_CAMERA", "UBUNCAM_CONF", "UBUNCAM_PREVIEW")}
            os.environ["UBUNCAM_CAMERA"] = "/dev/video2"
            os.environ["UBUNCAM_CONF"] = "0.55"
            os.environ["UBUNCAM_PREVIEW"] = "0"
            try:
                cfg = load_config(path)
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
            self.assertEqual(cfg.camera_device, "/dev/video2")
            self.assertAlmostEqual(cfg.confidence, 0.55)
            self.assertFalse(cfg.preview)

    def test_bus_capture_once_per_track(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self_check_bus_captures(Path(tmp))
        self.assertEqual(result["saved"], 2)
        self.assertEqual(result["files"], 2)
        self.assertEqual(result["counts"], {"car": 1, "truck": 0, "bus": 2})
        self.assertTrue(result["logged_capture"])

    def test_hourly_buckets(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self_check_hourly(Path(tmp))
        self.assertEqual(result["today"], 6)
        self.assertEqual(result["at_9"], 2)

    def test_dashboard_counts(self):
        payload = self_check_dashboard()
        self.assertEqual(payload["counts"], {"car": 4, "truck": 1, "bus": 2})
        self.assertEqual(payload["active"], {"car": 1, "truck": 0, "bus": 1})
        self.assertEqual(payload["captures"], [])
        self.assertEqual(len(payload["hourly"]["hours"]), 24)
        self.assertEqual(len(payload["hourly"]["series"]["total"]), 24)
        self.assertEqual(sum(payload["hourly"]["series"]["bus"]), 1)
        self.assertEqual(sum(payload["hourly"]["series"]["total"]), 1)
        self.assertEqual(sum(payload["hourly"]["series"]["car"]), 0)

    def test_dashboard_serves_capture_and_rejects_escape(self):
        import json
        import urllib.error
        import urllib.request

        from ubuncam.dashboard import Dashboard

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            name = "bus-7-20261113T081200123456Z.jpg"
            payload_bytes = b"\xff\xd8bus\xff\xd9"
            (root / name).write_bytes(payload_bytes)
            (root / "notes.txt").write_text("secret", encoding="utf-8")
            dashboard = Dashboard(("car", "truck", "bus"), root, port=0)
            try:
                dashboard.start()
                self.assertEqual(dashboard.host, "127.0.0.1")
                dashboard.update({"car": 0, "truck": 0, "bus": 1}, {"car": 0, "truck": 0, "bus": 0})
                with urllib.request.urlopen(dashboard.url + "api/status", timeout=2) as response:
                    body = json.load(response)
                self.assertEqual(body["counts"]["bus"], 1)
                self.assertEqual(body["active"]["bus"], 0)
                self.assertEqual(body["captures"], [{"track_id": 7, "file": name, "url": "/captures/" + name}])
                with urllib.request.urlopen(dashboard.url, timeout=2) as response:
                    main_page = response.read().decode("utf-8")
                self.assertIn('href="/captures"', main_page)
                self.assertNotIn(name, main_page)
                with urllib.request.urlopen(dashboard.url + "captures", timeout=2) as response:
                    captures_page = response.read().decode("utf-8")
                self.assertIn(name, captures_page)
                self.assertIn('href="/"', captures_page)
                with urllib.request.urlopen(dashboard.url + "captures/" + name, timeout=2) as response:
                    self.assertEqual(response.read(), payload_bytes)
                import socket

                dashboard.update(
                    {"car": 1, "truck": 0, "bus": 1},
                    {"car": 1, "truck": 0, "bus": 1},
                    jpeg=b"\xff\xd8\xff\xd9",
                )
                sock = socket.create_connection(("127.0.0.1", dashboard.port), timeout=2)
                try:
                    sock.sendall(b"GET /stream HTTP/1.0\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
                    streamed = b""
                    sock.settimeout(2)
                    while b"\xff\xd8" not in streamed and len(streamed) < 20000:
                        chunk = sock.recv(4096)
                        if not chunk:
                            break
                        streamed += chunk
                finally:
                    sock.close()
                self.assertIn(b"multipart/x-mixed-replace", streamed)
                self.assertIn(b"\xff\xd8\xff\xd9", streamed)

                for blocked in (
                    "captures/../notes.txt",
                    "captures/%2e%2e%2fnotes.txt",
                    "captures/notes.txt",
                    "captures/bus-7-20261113T081200123456Z.jpg/../../notes.txt",
                ):
                    with self.assertRaises(urllib.error.HTTPError) as caught:
                        urllib.request.urlopen(dashboard.url + blocked, timeout=2)
                    self.assertEqual(caught.exception.code, 404)
            finally:
                dashboard.stop()

    def test_dashboard_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(
                "dashboard: {enabled: false, port: 9001}\ncaptures: {dir: logs/buses}\n",
                encoding="utf-8",
            )
            cfg = load_config(path, use_env=False)
            self.assertFalse(cfg.dashboard_enabled)
            self.assertEqual(cfg.dashboard_host, "127.0.0.1")
            self.assertEqual(cfg.dashboard_port, 9001)
            self.assertTrue(str(cfg.captures_dir).endswith("logs/buses"))

            path.write_text("dashboard: {enabled: 'false'}\n", encoding="utf-8")
            self.assertFalse(load_config(path, use_env=False).dashboard_enabled)

            path.write_text("dashboard: {host: 0.0.0.0, port: 8765}\n", encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(path, use_env=False)


if __name__ == "__main__":
    unittest.main()
