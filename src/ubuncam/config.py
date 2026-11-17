"""Load the YAML config and apply a few environment overrides."""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

ALLOWED_CLASSES = ("car", "truck", "bus")
DASHBOARD_HOST = "127.0.0.1"
DEFAULT_IMGSZ = 320
DEFAULT_INFER_FPS = 5.0
DEFAULT_THREADS = 4


class ConfigError(ValueError):
    """The config file or an override is not usable."""


@dataclass(frozen=True)
class Config:
    camera_device: str
    camera_width: int
    camera_height: int
    camera_fourcc: str
    weights: Path
    source_weights: Path
    confidence: float
    iou: float
    imgsz: int
    infer_fps: float
    threads: int
    classes: tuple[str, ...]
    tracker_iou: float
    tracker_max_age: int
    log_path: Path
    summary_interval_s: float
    preview: bool
    window: str
    captures_dir: Path
    dashboard_enabled: bool
    dashboard_host: str
    dashboard_port: int


def load_config(path: str | Path, use_env: bool = True) -> Config:
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(
            f"Config not found: {config_path}. "
            "Copy config.example.yaml to config.yaml or run scripts/setup.sh."
        )
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path} must be a mapping")

    base = config_path.resolve().parent
    camera = _section(raw, "camera")
    model = _section(raw, "model")
    tracker = _section(raw, "tracker")
    log = _section(raw, "log")
    preview = _section(raw, "preview")
    dashboard = _section(raw, "dashboard")
    captures = _section(raw, "captures")

    device = str(camera.get("device", "/dev/video0"))
    classes = _classes(raw.get("classes", list(ALLOWED_CLASSES)))
    confidence = _unit_float(model.get("confidence", 0.40), "model.confidence")
    iou = _unit_float(model.get("iou", 0.50), "model.iou")
    tracker_iou = _unit_float(tracker.get("iou_threshold", 0.30), "tracker.iou_threshold")
    max_age = _positive_int(tracker.get("max_age", 15), "tracker.max_age")
    imgsz = _positive_int(model.get("imgsz", DEFAULT_IMGSZ), "model.imgsz")
    infer_fps = _positive_number(model.get("infer_fps", DEFAULT_INFER_FPS), "model.infer_fps")
    threads = _positive_int(model.get("threads", DEFAULT_THREADS), "model.threads")
    width = _non_negative_int(camera.get("width", 1280), "camera.width")
    height = _non_negative_int(camera.get("height", 720), "camera.height")
    try:
        summary = float(log.get("summary_interval", 60))
    except (TypeError, ValueError) as exc:
        raise ConfigError("log.summary_interval must be a number > 0") from exc
    if summary <= 0:
        raise ConfigError("log.summary_interval must be > 0")

    fourcc = str(camera.get("fourcc", "MJPG"))
    if fourcc and len(fourcc) != 4:
        raise ConfigError("camera.fourcc must be 4 characters or empty")

    preview_enabled = bool(preview.get("enabled", True))
    log_path = _resolve(base, str(log.get("path", "logs/events.jsonl")))
    weights = _resolve(base, str(model.get("weights", "models/yolo11n.onnx")))
    source = _resolve(base, str(model.get("source_weights", "models/yolo11n.pt")))
    window = str(preview.get("window", "ubun-cam"))
    if "host" in dashboard or "bind" in dashboard:
        raise ConfigError(
            "dashboard listens on 127.0.0.1 only. Remove dashboard.host and dashboard.bind."
        )
    dashboard_enabled = _flag(dashboard.get("enabled", True), "dashboard.enabled")
    dashboard_port = _port(dashboard.get("port", 8765))
    captures_dir = _resolve(base, str(captures.get("dir", "logs/captures")))

    if use_env:
        if os.environ.get("UBUNCAM_CAMERA"):
            device = os.environ["UBUNCAM_CAMERA"]
        if os.environ.get("UBUNCAM_CONF"):
            confidence = _unit_float(os.environ["UBUNCAM_CONF"], "UBUNCAM_CONF")
        if os.environ.get("UBUNCAM_LOG"):
            log_path = _resolve(base, os.environ["UBUNCAM_LOG"])
        if os.environ.get("UBUNCAM_PREVIEW"):
            preview_enabled = os.environ["UBUNCAM_PREVIEW"].strip().lower() not in {
                "0",
                "false",
                "no",
                "off",
            }

    return Config(
        camera_device=device,
        camera_width=width,
        camera_height=height,
        camera_fourcc=fourcc,
        weights=weights,
        source_weights=source,
        confidence=confidence,
        iou=iou,
        imgsz=imgsz,
        infer_fps=infer_fps,
        threads=threads,
        classes=classes,
        tracker_iou=tracker_iou,
        tracker_max_age=max_age,
        log_path=log_path,
        summary_interval_s=summary,
        preview=preview_enabled,
        window=window,
        captures_dir=captures_dir,
        dashboard_enabled=dashboard_enabled,
        dashboard_host=DASHBOARD_HOST,
        dashboard_port=dashboard_port,
    )


def apply_thread_limits(threads: int) -> None:
    """Cap OpenCV, OpenMP, and ONNX Runtime before those pools start.

    Call this before importing cv2, torch, or ultralytics. ONNX Runtime reads
    ``ORT_INTRA_OP_NUM_THREADS`` when the session is created, and OpenMP reads
    ``OMP_NUM_THREADS`` when its runtime starts. OpenCV also honors
    ``cv2.setNumThreads`` if it is already imported.
    """
    count = _positive_int(threads, "model.threads")
    value = str(count)
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NUMEXPR_MAX_THREADS",
        "OPENCV_FOR_THREADS_NUM",
        "ORT_INTRA_OP_NUM_THREADS",
    ):
        os.environ[key] = value
    # One graph at a time. Inter-op workers would sit on top of the intra-op pool.
    os.environ["ORT_INTER_OP_NUM_THREADS"] = "1"
    # Idle OpenMP threads otherwise spin between inferences and peg the CPU.
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    cv2 = sys.modules.get("cv2")
    if cv2 is not None:
        cv2.setNumThreads(count)
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            torch.set_num_threads(count)
        except RuntimeError:
            pass


def _flag(value: object, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise ConfigError(f"{label} must be true or false")


def _port(value: object) -> int:
    port = _positive_int(value, "dashboard.port")
    if port > 65535:
        raise ConfigError("dashboard.port must be from 1 to 65535")
    return port


def _section(raw: dict, name: str) -> dict:
    value = raw.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a mapping")
    return value


def _classes(value: object) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, (list, tuple)) or not value:
        raise ConfigError("classes must be a non-empty list of car, truck, bus")
    names = tuple(str(item) for item in value)
    unknown = [name for name in names if name not in ALLOWED_CLASSES]
    if unknown:
        raise ConfigError(
            f"unsupported classes {unknown}. This MVP detects only car, truck, and bus."
        )
    if len(set(names)) != len(names):
        raise ConfigError(f"duplicate classes in {list(names)}")
    return names


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return base / path


def _unit_float(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label} must be a number from 0 to 1") from exc
    if not 0.0 <= number <= 1.0:
        raise ConfigError(f"{label} must be a number from 0 to 1")
    return number


def _positive_number(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"{label} must be a number > 0")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label} must be a number > 0") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ConfigError(f"{label} must be a number > 0")
    return number


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value or int(value) < 1:
        raise ConfigError(f"{label} must be an integer >= 1")
    return int(value)


def _non_negative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value or int(value) < 0:
        raise ConfigError(f"{label} must be an integer >= 0")
    return int(value)
