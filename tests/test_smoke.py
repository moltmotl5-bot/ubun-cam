"""Camera-free smoke test for tracking, counting, config, and drawing."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ubuncam.config import ConfigError, load_config  # noqa: E402
from ubuncam.pipeline import self_check, self_check_log  # noqa: E402


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
        self.assertTrue(str(cfg.log_path).endswith("logs/events.jsonl"))
        self.assertTrue(str(cfg.weights).endswith("models/yolo11n.onnx"))

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


if __name__ == "__main__":
    unittest.main()
