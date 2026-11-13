"""Tracking, counting, logging, and drawing. No camera and no model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

COLORS = {
    "car": (0, 200, 0),
    "truck": (0, 140, 255),
    "bus": (255, 80, 0),
}


@dataclass
class Detection:
    xyxy: tuple[float, float, float, float]
    class_name: str
    confidence: float
    track_id: int | None = None


@dataclass
class _Track:
    class_name: str
    xyxy: tuple[float, float, float, float]
    missed: int = 0


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def keep_classes(detections: list[Detection], allowed: tuple[str, ...] | list[str]) -> list[Detection]:
    names = set(allowed)
    return [det for det in detections if det.class_name in names]


class IoUTracker:
    """Greedy same-class IoU matcher. Enough to count a vehicle once while it overlaps."""

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 15):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self._next_id = 1
        self._tracks: dict[int, _Track] = {}

    def update(self, detections: list[Detection]) -> list[Detection]:
        pairs: list[tuple[float, int, int]] = []
        for track_id, track in self._tracks.items():
            for index, det in enumerate(detections):
                if det.class_name != track.class_name:
                    continue
                score = iou(track.xyxy, det.xyxy)
                if score >= self.iou_threshold:
                    pairs.append((score, track_id, index))
        pairs.sort(key=lambda item: item[0], reverse=True)

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        for _score, track_id, index in pairs:
            if track_id in matched_tracks or index in matched_dets:
                continue
            matched_tracks.add(track_id)
            matched_dets.add(index)
            det = detections[index]
            det.track_id = track_id
            track = self._tracks[track_id]
            track.xyxy = det.xyxy
            track.missed = 0

        for index, det in enumerate(detections):
            if index in matched_dets:
                continue
            track_id = self._next_id
            self._next_id += 1
            det.track_id = track_id
            self._tracks[track_id] = _Track(det.class_name, det.xyxy, missed=0)
            matched_tracks.add(track_id)

        for track_id in list(self._tracks):
            if track_id in matched_tracks:
                continue
            track = self._tracks[track_id]
            track.missed += 1
            if track.missed > self.max_age:
                del self._tracks[track_id]
        return detections


class VehicleCounter:
    """Count each track id once, the first frame it is seen."""

    def __init__(self, classes: tuple[str, ...] | list[str]):
        self.counts = {name: 0 for name in classes}
        self._seen: set[int] = set()

    def observe(self, detections: list[Detection]) -> list[dict]:
        events = []
        for det in detections:
            if det.track_id is None or det.track_id in self._seen:
                continue
            if det.class_name not in self.counts:
                continue
            self._seen.add(det.track_id)
            self.counts[det.class_name] += 1
            events.append(
                {
                    "event": "detection",
                    "track_id": det.track_id,
                    "class": det.class_name,
                    "conf": round(float(det.confidence), 4),
                    "bbox": [round(float(v), 1) for v in det.xyxy],
                }
            )
        return events


class EventLog:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, payload: dict) -> None:
        record = {"ts": datetime.now(timezone.utc).isoformat(), **payload}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    def write_events(self, events: list[dict]) -> None:
        for event in events:
            self.write(event)

    def write_summary(self, counts: dict[str, int]) -> None:
        self.write({"event": "summary", "counts": dict(counts)})


def active_counts(detections: list[Detection], classes: tuple[str, ...] | list[str]) -> dict[str, int]:
    counts = {name: 0 for name in classes}
    for det in detections:
        if det.class_name in counts:
            counts[det.class_name] += 1
    return counts


def annotate(
    frame: np.ndarray,
    detections: list[Detection],
    counts: dict[str, int],
    fps: float | None = None,
) -> np.ndarray:
    vis = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in det.xyxy)
        color = COLORS.get(det.class_name, (0, 200, 0))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        if det.track_id is not None:
            label += f" #{det.track_id}"
        text_y = y1 - 4
        if text_y < 32:
            text_y = min(y2 + 16, vis.shape[0] - 2)
        _label(vis, label, (x1, text_y), color)

    hud = "  ".join(f"{name}:{count}" for name, count in counts.items())
    if fps is not None:
        hud = f"{fps:4.1f} fps   {hud}"
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(vis, hud, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def bus_capture_filename(track_id: int, when: datetime | None = None) -> str:
    """``bus-<id>-YYYYmmddTHHMMSSffffffZ.jpg``. The dashboard only serves this shape."""
    moment = when or datetime.now(timezone.utc)
    return f"bus-{int(track_id)}-{moment.strftime('%Y%m%dT%H%M%S%fZ')}.jpg"


def save_bus_captures(frame: np.ndarray, events: list[dict], directory: Path) -> list[Path]:
    """Write one annotated JPEG per new bus detection event.

    Sets ``capture`` on those events to the absolute path. Other classes are
    left unchanged. Pass only the events from one ``VehicleCounter.observe``
    call so each track is saved once.
    """
    bus_events = [
        event
        for event in events
        if event.get("class") == "bus" and event.get("event", "detection") == "detection"
    ]
    saved: list[Path] = []
    if not bus_events:
        return saved
    directory.mkdir(parents=True, exist_ok=True)
    for event in bus_events:
        path = directory / bus_capture_filename(int(event["track_id"]))
        if not cv2.imwrite(str(path), frame):
            continue
        event["capture"] = str(path.resolve())
        saved.append(path.resolve())
    return saved


def record_frame(
    frame: np.ndarray,
    detections: list[Detection],
    tracker: IoUTracker,
    counter: VehicleCounter,
    captures_dir: Path,
    *,
    draw: bool = False,
    fps: float | None = None,
) -> tuple[list[Detection], list[dict], dict[str, int], np.ndarray | None]:
    """Track, count, and save one annotated JPEG per new bus track.

    ``draw`` forces an annotated frame for the preview or the dashboard.
    A new bus still produces that frame when ``draw`` is false, so headless
    mode keeps the captures.
    """
    tracked = tracker.update(detections)
    events = counter.observe(tracked)
    active = active_counts(tracked, tuple(counter.counts))
    needs_image = draw or any(event.get("class") == "bus" for event in events)
    vis = annotate(frame, tracked, counter.counts, fps=fps) if needs_image else None
    if vis is not None:
        save_bus_captures(vis, events, captures_dir)
    return tracked, events, active, vis


def _label(image: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    y = max(height + 2, y)
    cv2.rectangle(image, (x, y - height - 2), (x + width + 2, y + baseline), (0, 0, 0), -1)
    cv2.putText(image, text, (x + 1, y), font, scale, color, thickness, cv2.LINE_AA)


def self_check() -> dict:
    """Exercise track, count, and drawing on a synthetic frame. No camera or weights."""
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    allowed = ("car", "truck", "bus")
    mixed = [
        Detection((10, 10, 80, 60), "car", 0.88),
        Detection((0, 0, 5, 5), "person", 0.99),
    ]
    kept = keep_classes(mixed, allowed)
    assert [det.class_name for det in kept] == ["car"]

    tracker = IoUTracker(iou_threshold=0.3, max_age=1)
    counter = VehicleCounter(allowed)
    first = tracker.update(
        [
            Detection((10, 10, 110, 80), "car", 0.91),
            Detection((200, 20, 310, 90), "car", 0.80),
        ]
    )
    events = counter.observe(first)
    assert counter.counts == {"car": 2, "truck": 0, "bus": 0}
    assert len(events) == 2
    car_ids = {det.track_id for det in first}
    assert len(car_ids) == 2

    moved = tracker.update(
        [
            Detection((14, 12, 114, 82), "car", 0.90),
            Detection((198, 22, 308, 92), "car", 0.79),
            Detection((20, 120, 160, 200), "bus", 0.77),
        ]
    )
    events = counter.observe(moved)
    assert counter.counts == {"car": 2, "truck": 0, "bus": 1}
    assert [event["class"] for event in events] == ["bus"]
    moved_ids = {det.class_name: det.track_id for det in moved}
    assert moved_ids["bus"] not in car_ids
    assert {det.track_id for det in moved if det.class_name == "car"} == car_ids

    # max_age is 1 missed frame, so two empty updates drop the tracks.
    tracker.update([])
    tracker.update([])
    again = tracker.update([Detection((10, 10, 110, 80), "car", 0.92)])
    events = counter.observe(again)
    assert counter.counts["car"] == 3
    assert again[0].track_id not in car_ids
    assert events[0]["event"] == "detection"

    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    vis = annotate(frame, again, counter.counts, fps=8.0)
    assert vis.shape == frame.shape
    assert int(vis[0:28].sum()) > 0
    # Left edge of the box sits below the HUD strip.
    assert int(vis[40, 10].sum()) > 0

    return {
        "counts": dict(counter.counts),
        "drawn": True,
        "classes": list(allowed),
    }


def self_check_log(directory: Path) -> int:
    """Write a real JSONL file the way the live loop does, and return the line count."""
    counter = VehicleCounter(("car", "truck", "bus"))
    tracker = IoUTracker()
    tracked = tracker.update([Detection((1, 1, 20, 20), "truck", 0.66)])
    log = EventLog(directory / "events.jsonl")
    log.write_events(counter.observe(tracked))
    log.write_summary(counter.counts)
    lines = (directory / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    detection = json.loads(lines[0])
    summary = json.loads(lines[1])
    assert detection["event"] == "detection"
    assert detection["class"] == "truck"
    assert summary["event"] == "summary"
    assert summary["counts"]["truck"] == 1
    assert "ts" in detection
    return len(lines)


def self_check_bus_captures(directory: Path) -> dict:
    """Save one annotated JPEG per new bus track, and record the path in JSONL.

    No camera and no weights. Headless (``draw=False``) still saves. Drawing
    every frame does not save the same track again.
    """
    from ubuncam.dashboard import CAPTURE_NAME

    tracker = IoUTracker(iou_threshold=0.3, max_age=1)
    counter = VehicleCounter(("car", "truck", "bus"))
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    captures = directory / "captures"
    log = EventLog(directory / "events.jsonl")

    _tracked, events, _active, vis = record_frame(
        frame,
        [Detection((10, 10, 80, 70), "bus", 0.91)],
        tracker,
        counter,
        captures,
        draw=False,
    )
    assert vis is not None
    assert int(vis[:28].sum()) > 0
    assert counter.counts == {"car": 0, "truck": 0, "bus": 1}
    assert len(events) == 1 and events[0]["class"] == "bus"
    saved = Path(events[0]["capture"])
    assert saved.is_file()
    assert CAPTURE_NAME.match(saved.name)
    decoded = cv2.imdecode(np.frombuffer(saved.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None and decoded.shape == frame.shape
    assert int(decoded.sum()) > 0
    log.write_events(events)

    _tracked, events, _active, vis = record_frame(
        frame,
        [Detection((12, 12, 82, 72), "bus", 0.90)],
        tracker,
        counter,
        captures,
        draw=True,
    )
    assert events == []
    assert vis is not None
    assert counter.counts["bus"] == 1
    assert len(list(captures.glob("bus-*.jpg"))) == 1
    log.write_events(events)

    _tracked, events, _active, _vis = record_frame(
        frame,
        [
            Detection((14, 14, 84, 74), "bus", 0.89),
            Detection((180, 10, 260, 50), "car", 0.70),
        ],
        tracker,
        counter,
        captures,
        draw=False,
    )
    assert [event["class"] for event in events] == ["car"]
    assert "capture" not in events[0]
    assert len(list(captures.glob("bus-*.jpg"))) == 1
    log.write_events(events)

    _tracked, events, _active, _vis = record_frame(
        frame,
        [
            Detection((14, 14, 84, 74), "bus", 0.89),
            Detection((180, 80, 300, 160), "bus", 0.82),
        ],
        tracker,
        counter,
        captures,
        draw=False,
    )
    assert [event["class"] for event in events] == ["bus"]
    second = Path(events[0]["capture"])
    assert second.is_file() and second != saved
    assert len(list(captures.glob("bus-*.jpg"))) == 2
    log.write_events(events)

    lines = (directory / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    first = json.loads(lines[0])
    car = json.loads(lines[1])
    again = json.loads(lines[2])
    assert first["event"] == "detection" and first["class"] == "bus"
    assert first["capture"] == str(saved)
    assert car["class"] == "car" and "capture" not in car
    assert again["class"] == "bus" and again["capture"] == str(second)
    assert counter.counts == {"car": 1, "truck": 0, "bus": 2}
    return {"saved": 2, "files": 2, "counts": dict(counter.counts), "logged_capture": True}
