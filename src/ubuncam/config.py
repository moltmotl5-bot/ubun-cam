"""Load the YAML config and apply a few environment overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

ALLOWED_CLASSES = ("car", "truck", "bus")


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
    classes: tuple[str, ...]
    tracker_iou: float
    tracker_max_age: int
    log_path: Path
    summary_interval_s: float
    preview: bool
    window: str


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

    device = str(camera.get("device", "/dev/video0"))
    classes = _classes(raw.get("classes", list(ALLOWED_CLASSES)))
    confidence = _unit_float(model.get("confidence", 0.40), "model.confidence")
    iou = _unit_float(model.get("iou", 0.50), "model.iou")
    tracker_iou = _unit_float(tracker.get("iou_threshold", 0.30), "tracker.iou_threshold")
    max_age = _positive_int(tracker.get("max_age", 15), "tracker.max_age")
    imgsz = _positive_int(model.get("imgsz", 640), "model.imgsz")
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
        classes=classes,
        tracker_iou=tracker_iou,
        tracker_max_age=max_age,
        log_path=log_path,
        summary_interval_s=summary,
        preview=preview_enabled,
        window=window,
    )


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


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value or int(value) < 1:
        raise ConfigError(f"{label} must be an integer >= 1")
    return int(value)


def _non_negative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value or int(value) < 0:
        raise ConfigError(f"{label} must be an integer >= 0")
    return int(value)
