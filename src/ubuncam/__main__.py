"""ubun-cam entrypoint: capture -> detect -> track -> count -> preview."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import tempfile
import time
from pathlib import Path

import cv2

from ubuncam import __version__
from ubuncam.config import Config, ConfigError, load_config
from ubuncam.dashboard import Dashboard, self_check_dashboard
from ubuncam.pipeline import (
    EventLog,
    IoUTracker,
    VehicleCounter,
    record_frame,
    self_check,
    self_check_bus_captures,
    self_check_log,
)

_STOP = False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ubuncam",
        description="Detect cars, trucks, and buses on a local USB webcam.",
    )
    parser.add_argument(
        "--config",
        default=os.environ.get("UBUNCAM_CONFIG", "config.yaml"),
        help="YAML config (default: config.yaml or UBUNCAM_CONFIG)",
    )
    parser.add_argument("--self-check", action="store_true", help="Run the camera-free smoke check and exit")
    parser.add_argument("--list-cameras", action="store_true", help="Print /dev/video* devices and exit")
    parser.add_argument("--export-model", action="store_true", help="Download YOLO11n and export ONNX, then exit")
    parser.add_argument("--image", help="Run one still image through the pipeline instead of the camera")
    args = parser.parse_args(argv)

    if args.self_check:
        return _run_self_check()
    if args.list_cameras:
        return _list_cameras()

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 1

    try:
        if args.export_model:
            return _export(cfg)
        if args.image:
            return _run_image(cfg, Path(args.image))
        return _run_camera(cfg)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1


def cli() -> None:
    raise SystemExit(main())


def _run_self_check() -> int:
    result = self_check()
    with tempfile.TemporaryDirectory(prefix="ubuncam-") as tmp:
        root = Path(tmp)
        lines = self_check_log(root)
        captures = self_check_bus_captures(root / "bus")
    status = self_check_dashboard()
    print(
        "self-check ok  "
        f"counts={result['counts']}  "
        f"log_lines={lines}  "
        f"bus_captures={captures['saved']}  "
        f"dashboard_bus={status['counts']['bus']}  "
        "camera=not used  model=not used"
    )
    return 0


def _list_cameras() -> int:
    devices = sorted(Path("/dev").glob("video*"))
    if not devices:
        print("No /dev/video* devices found.", file=sys.stderr)
        return 1
    for device in devices:
        print(device)
    return 0


def _export(cfg: Config) -> int:
    from ubuncam.detect import ensure_onnx

    path = ensure_onnx(cfg.source_weights, cfg.weights, cfg.imgsz)
    print(f"model ready: {path}")
    return 0


def _detector(cfg: Config):
    from ubuncam.detect import Detector, ensure_onnx

    weights = ensure_onnx(cfg.source_weights, cfg.weights, cfg.imgsz)
    return Detector(weights, cfg.classes, cfg.confidence, cfg.iou, cfg.imgsz)


def _run_image(cfg: Config, image_path: Path) -> int:
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"Cannot read image {image_path}", file=sys.stderr)
        return 1
    detector = _detector(cfg)
    tracker = IoUTracker(cfg.tracker_iou, cfg.tracker_max_age)
    counter = VehicleCounter(cfg.classes)
    log = EventLog(cfg.log_path)
    _tracked, events, _active, vis = record_frame(
        frame,
        detector.detect(frame),
        tracker,
        counter,
        cfg.captures_dir,
        draw=True,
    )
    log.write_events(events)
    log.write_summary(counter.counts)
    out = cfg.log_path.parent / "preview.jpg"
    cv2.imwrite(str(out), vis)
    for event in events:
        _print_detection(event)
    print(f"counts {counter.counts}")
    print(f"wrote {out}")
    print(f"log {cfg.log_path}")
    print(f"captures {cfg.captures_dir}")
    return 0


def _run_camera(cfg: Config) -> int:
    global _STOP
    detector = _detector(cfg)
    tracker = IoUTracker(cfg.tracker_iou, cfg.tracker_max_age)
    counter = VehicleCounter(cfg.classes)
    log = EventLog(cfg.log_path)
    cfg.captures_dir.mkdir(parents=True, exist_ok=True)
    cap = _open_camera(cfg)
    preview = _preview_enabled(cfg.preview)
    window_ready = False
    dashboard = _start_dashboard(cfg)

    def _handle(_signum, _frame):
        global _STOP
        _STOP = True

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    print(f"ubun-cam {__version__}")
    print(f"camera {cfg.camera_device}")
    print(f"model {cfg.weights}")
    print(f"classes {', '.join(cfg.classes)}  conf {cfg.confidence}")
    print(f"log {cfg.log_path}")
    print(f"captures {cfg.captures_dir}")
    print("preview " + ("on (press q to quit)" if preview else "off"))
    print(f"dashboard {dashboard.url}" if dashboard is not None else "dashboard off")

    failures = 0
    last_summary = time.monotonic()
    try:
        while not _STOP:
            started = time.perf_counter()
            ok, frame = cap.read()
            if not ok or frame is None:
                failures += 1
                if failures >= 30:
                    print(f"Camera {cfg.camera_device} stopped producing frames.", file=sys.stderr)
                    return 1
                time.sleep(0.05)
                continue
            failures = 0

            detections = detector.detect(frame)
            elapsed = time.perf_counter() - started
            fps = (1.0 / elapsed) if elapsed > 0 else 0.0
            _tracked, events, active, vis = record_frame(
                frame,
                detections,
                tracker,
                counter,
                cfg.captures_dir,
                draw=preview or dashboard is not None,
                fps=fps,
            )
            log.write_events(events)
            for event in events:
                _print_detection(event)

            now = time.monotonic()
            if now - last_summary >= cfg.summary_interval_s:
                log.write_summary(counter.counts)
                print(f"counts {counter.counts} active {active}", flush=True)
                last_summary = now

            if dashboard is not None:
                dashboard.update(counter.counts, active, _encode_jpeg(vis) if vis is not None else None)

            if preview and vis is not None:
                if not window_ready:
                    try:
                        cv2.namedWindow(cfg.window, cv2.WINDOW_NORMAL)
                    except cv2.error:
                        preview = False
                        print("Preview unavailable. Continuing.", file=sys.stderr)
                        continue
                    window_ready = True
                action = _show(vis, cfg.window)
                if action == "quit":
                    break
                if action == "disabled":
                    preview = False
                    window_ready = False
                    print("Preview unavailable. Continuing.", file=sys.stderr)
    finally:
        log.write_summary(counter.counts)
        print(f"counts {counter.counts}", flush=True)
        cap.release()
        if window_ready:
            cv2.destroyAllWindows()
        if dashboard is not None:
            dashboard.stop()
    return 0


def _start_dashboard(cfg: Config) -> Dashboard | None:
    if not cfg.dashboard_enabled:
        return None
    dashboard = Dashboard(cfg.classes, cfg.captures_dir, port=cfg.dashboard_port, log_path=cfg.log_path)
    try:
        dashboard.start()
    except (OSError, RuntimeError) as exc:
        print(f"Dashboard not started on 127.0.0.1:{cfg.dashboard_port} ({exc}). Continuing.", file=sys.stderr)
        dashboard.stop()
        return None
    return dashboard


def _encode_jpeg(frame) -> bytes | None:
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return None
    return encoded.tobytes()


def _print_detection(event: dict) -> None:
    message = f"detection {event['class']} id={event['track_id']} conf={event['conf']}"
    if event.get("capture"):
        message += f" capture={event['capture']}"
    print(message, flush=True)


def _preview_enabled(requested: bool) -> bool:
    if not requested:
        return False
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        print(
            "No DISPLAY or WAYLAND_DISPLAY. Preview window off. "
            "Logging, bus captures, and the dashboard continue.",
            file=sys.stderr,
        )
        return False
    return True


def _show(frame, window: str) -> str:
    try:
        cv2.imshow(window, frame)
        key = cv2.waitKey(1) & 0xFF
    except cv2.error:
        return "disabled"
    if key in (ord("q"), 27):
        return "quit"
    return "ok"


def _open_camera(cfg: Config):
    device = cfg.camera_device
    source: int | str = int(device) if device.isdigit() else device
    api = cv2.CAP_V4L2 if sys.platform.startswith("linux") else 0
    cap = cv2.VideoCapture(source, api)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(source)
    if cfg.camera_fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg.camera_fourcc))
    if cfg.camera_width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(cfg.camera_width))
    if cfg.camera_height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(cfg.camera_height))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera {device}. "
            "Check `ls -l /dev/video*` and `python -m ubuncam --list-cameras`. "
            "If you see Permission denied, join the video group and log in again: "
            'sudo usermod -aG video "$USER"'
        )
    return cap


if __name__ == "__main__":
    raise SystemExit(main())
