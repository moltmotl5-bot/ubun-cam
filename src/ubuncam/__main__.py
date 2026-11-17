"""ubun-cam entrypoint: capture -> detect -> track -> count -> preview."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import tempfile
import threading
import time
from pathlib import Path

from ubuncam import __version__
from ubuncam.config import DEFAULT_THREADS, Config, ConfigError, apply_thread_limits, load_config

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

    if args.list_cameras:
        return _list_cameras()
    if args.self_check:
        apply_thread_limits(DEFAULT_THREADS)
        return _run_self_check()

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 1

    # Before cv2, torch, and the ONNX session start their thread pools.
    apply_thread_limits(cfg.threads)

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


def next_infer_at(now: float, scheduled: float, accepted: bool, interval: float) -> float:
    """Next time YOLO may start.

    A refused submit leaves the deadline where it is, so a slow pass is
    followed immediately by the newest frame. An accepted submit waits
    ``interval`` from this start and does not queue the ones in between.
    """
    if accepted:
        return now + interval
    return scheduled


class _InferWorker:
    """One inference at a time. The frame is copied before the worker sees it."""

    def __init__(self, detector):
        self._detector = detector
        self._cond = threading.Condition()
        self._pending = None
        self._busy = False
        self._stop = False
        self._result = None
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._loop, name="ubuncam-infer", daemon=True)
        self._thread.start()

    def submit(self, frame) -> bool:
        """Accept a copy of ``frame`` when nothing is in flight and no result is unread."""
        with self._cond:
            if (
                self._stop
                or self._busy
                or self._pending is not None
                or self._result is not None
                or self._error is not None
            ):
                return False
            self._pending = frame.copy()
            self._cond.notify()
            return True

    def poll(self):
        """Return ``(detections, frame, seconds)`` once, or None if it was already taken."""
        with self._cond:
            if self._error is not None:
                raise self._error
            result = self._result
            self._result = None
            return result

    def close(self) -> None:
        with self._cond:
            self._stop = True
            self._pending = None
            self._cond.notify_all()
        self._thread.join(timeout=5)

    def _loop(self) -> None:
        while True:
            with self._cond:
                while self._pending is None and not self._stop:
                    self._cond.wait()
                if self._pending is None:
                    return
                frame = self._pending
                self._pending = None
                self._busy = True
            try:
                started = time.perf_counter()
                detections = self._detector.detect(frame)
                elapsed = time.perf_counter() - started
            except Exception as exc:
                with self._cond:
                    self._busy = False
                    self._error = exc
                    self._stop = True
                return
            with self._cond:
                self._busy = False
                self._result = (detections, frame, elapsed)


def _run_self_check() -> int:
    import cv2

    cv2.setNumThreads(DEFAULT_THREADS)
    from ubuncam.dashboard import self_check_dashboard, self_check_hourly
    from ubuncam.pipeline import self_check, self_check_bus_captures, self_check_log

    result = self_check()
    with tempfile.TemporaryDirectory(prefix="ubuncam-") as tmp:
        root = Path(tmp)
        lines = self_check_log(root)
        captures = self_check_bus_captures(root / "bus")
        hourly = self_check_hourly(root / "hourly")
    status = self_check_dashboard()
    print(
        "self-check ok  "
        f"counts={result['counts']}  "
        f"log_lines={lines}  "
        f"bus_captures={captures['saved']}  "
        f"hourly_today={hourly['today']}  "
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
    return Detector(weights, cfg.classes, cfg.confidence, cfg.iou, cfg.imgsz, cfg.threads)


def _run_image(cfg: Config, image_path: Path) -> int:
    import cv2

    from ubuncam.pipeline import EventLog, IoUTracker, VehicleCounter, record_frame

    cv2.setNumThreads(cfg.threads)
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"Cannot read image {image_path}", file=sys.stderr)
        return 1
    detector = _detector(cfg)
    cv2.setNumThreads(cfg.threads)
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
    import cv2

    from ubuncam.pipeline import EventLog, IoUTracker, TrackSession, VehicleCounter

    cv2.setNumThreads(cfg.threads)
    detector = _detector(cfg)
    cv2.setNumThreads(cfg.threads)
    session = TrackSession(
        IoUTracker(cfg.tracker_iou, cfg.tracker_max_age),
        VehicleCounter(cfg.classes),
    )
    log = EventLog(cfg.log_path)
    cfg.captures_dir.mkdir(parents=True, exist_ok=True)
    cap = _open_camera(cfg, cv2)
    preview = _preview_enabled(cfg.preview)
    window_ready = False
    dashboard = _start_dashboard(cfg)
    worker = _InferWorker(detector)
    interval = 1.0 / cfg.infer_fps
    next_at = 0.0
    display_fps = 0.0
    infer_fps = 0.0
    last_infer_done: float | None = None
    previous = time.perf_counter()

    def _handle(_signum, _frame):
        global _STOP
        _STOP = True

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    print(f"ubun-cam {__version__}")
    print(f"camera {cfg.camera_device}")
    print(f"model {cfg.weights}")
    print(
        f"classes {', '.join(cfg.classes)}  conf {cfg.confidence}  "
        f"imgsz {cfg.imgsz}  infer {cfg.infer_fps:g} fps  threads {cfg.threads}"
    )
    print(f"log {cfg.log_path}")
    print(f"captures {cfg.captures_dir}")
    print("preview " + ("on (press q to quit)" if preview else "off"))
    print(f"dashboard {dashboard.url}" if dashboard is not None else "dashboard off")

    failures = 0
    last_summary = time.monotonic()
    try:
        while not _STOP:
            ok, frame = cap.read()
            if not ok or frame is None:
                failures += 1
                if failures >= 30:
                    print(f"Camera {cfg.camera_device} stopped producing frames.", file=sys.stderr)
                    return 1
                time.sleep(0.05)
                continue
            failures = 0

            result = worker.poll()
            if result is not None:
                detections, inferred, infer_dt = result
                done = time.perf_counter()
                if last_infer_done is not None:
                    gap = done - last_infer_done
                    if gap > 0:
                        sample = 1.0 / gap
                        infer_fps = sample if infer_fps <= 0 else (0.8 * infer_fps + 0.2 * sample)
                elif infer_dt > 0:
                    infer_fps = min(cfg.infer_fps, 1.0 / infer_dt)
                last_infer_done = done
                events, _vis = session.step(
                    inferred,
                    detections,
                    cfg.captures_dir,
                    draw=False,
                    fps=display_fps or None,
                )
                log.write_events(events)
                for event in events:
                    _print_detection(event)

            now = time.monotonic()
            if now >= next_at:
                accepted = worker.submit(frame)
                next_at = next_infer_at(now, next_at, accepted, interval)

            tick = time.perf_counter()
            elapsed = tick - previous
            previous = tick
            if elapsed > 0:
                sample = 1.0 / elapsed
                display_fps = sample if display_fps <= 0 else (0.9 * display_fps + 0.1 * sample)

            need_view = preview or dashboard is not None
            vis = None
            if need_view:
                vis = session.overlay(
                    frame,
                    fps=display_fps,
                    infer_fps=infer_fps if infer_fps > 0 else None,
                )

            now = time.monotonic()
            if now - last_summary >= cfg.summary_interval_s:
                log.write_summary(session.counter.counts)
                print(f"counts {session.counter.counts} active {session.active}", flush=True)
                last_summary = now

            if dashboard is not None:
                dashboard.update(
                    session.counter.counts,
                    session.active,
                    _encode_jpeg(vis, cv2) if vis is not None else None,
                )

            if preview and vis is not None:
                if not window_ready:
                    try:
                        cv2.namedWindow(cfg.window, cv2.WINDOW_NORMAL)
                    except cv2.error:
                        preview = False
                        print("Preview unavailable. Continuing.", file=sys.stderr)
                        continue
                    window_ready = True
                action = _show(vis, cfg.window, cv2)
                if action == "quit":
                    break
                if action == "disabled":
                    preview = False
                    window_ready = False
                    print("Preview unavailable. Continuing.", file=sys.stderr)
    finally:
        worker.close()
        log.write_summary(session.counter.counts)
        print(f"counts {session.counter.counts}", flush=True)
        cap.release()
        if window_ready:
            cv2.destroyAllWindows()
        if dashboard is not None:
            dashboard.stop()
    return 0


def _start_dashboard(cfg: Config):
    if not cfg.dashboard_enabled:
        return None
    from ubuncam.dashboard import Dashboard

    dashboard = Dashboard(cfg.classes, cfg.captures_dir, port=cfg.dashboard_port, log_path=cfg.log_path)
    try:
        dashboard.start()
    except (OSError, RuntimeError) as exc:
        print(f"Dashboard not started on 127.0.0.1:{cfg.dashboard_port} ({exc}). Continuing.", file=sys.stderr)
        dashboard.stop()
        return None
    return dashboard


def _encode_jpeg(frame, cv2) -> bytes | None:
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


def _show(frame, window: str, cv2) -> str:
    try:
        cv2.imshow(window, frame)
        key = cv2.waitKey(1) & 0xFF
    except cv2.error:
        return "disabled"
    if key in (ord("q"), 27):
        return "quit"
    return "ok"


def _open_camera(cfg: Config, cv2):
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
