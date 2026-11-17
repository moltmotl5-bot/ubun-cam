"""Local dashboard on 127.0.0.1. Stdlib HTTP only."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

CAPTURE_NAME = re.compile(r"^bus-(\d+)-(\d{8}T\d{12}Z)\.jpg$")
_HOST = "127.0.0.1"
_CAPTURE_LIMIT = 24
_CLASSES = ("car", "truck", "bus")

_CSS = """
  body { margin: 0; font: 16px/1.4 system-ui, sans-serif; background: #141414; color: #f2f2f2; }
  main { max-width: 1100px; margin: 0 auto; padding: 1.25rem; }
  h1 { font-size: 1.25rem; margin: 0 0 0.25rem; }
  h2 { font-size: 1rem; margin: 1.25rem 0 0.5rem; }
  a { color: #9ec1ff; }
  p.note { margin: 0 0 1rem; color: #bdbdbd; }
  .stats, .now { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.75rem; }
  .stats { margin-bottom: 0.5rem; }
  .stats div, .now div { background: #222; padding: 0.75rem 1rem; }
  .stats strong { display: block; font-size: 2.75rem; font-variant-numeric: tabular-nums; }
  .now strong { font-variant-numeric: tabular-nums; }
  img.live { display: block; width: 100%; max-height: 42vh; object-fit: contain; object-position: center top; background: #000; min-height: 120px; }
  svg#chart { width: 100%; height: auto; background: #1b1b1b; }
  .legend span { display: inline-block; width: 1.1rem; height: 0.35rem; margin: 0 0.35rem 0 0.8rem; vertical-align: middle; }
  .swatch-total { background: #f2f2f2; }
  .swatch-car { background: #00c800; }
  .swatch-truck { background: #ff8c00; }
  .swatch-bus { background: #3d7dff; }
  .captures { display: flex; flex-wrap: wrap; gap: 0.75rem; }
  .captures figure { margin: 0; width: 240px; }
  .captures img { width: 100%; background: #000; }
  figcaption { font-size: 0.8rem; color: #ccc; word-break: break-all; }
  @media (max-width: 640px) {
    main { padding: 0.75rem; }
    .stats strong { font-size: 2rem; }
    .stats div, .now div { padding: 0.55rem 0.6rem; }
    .captures figure { width: 100%; }
  }
"""

_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ubun-cam</title>
<style>__CSS__</style>
</head>
<body>
<main>
  <h1>ubun-cam</h1>
  <p class="note">Session totals count each tracked vehicle once since this process started. The chart is today, in this machine's local time, read from the JSONL log. A double-decker is a bus. This page is only on this machine.</p>
  <p><a href="/captures">Bus captures</a></p>
  <div class="stats">
    <div>car <strong id="total-car">__TOTAL_CAR__</strong></div>
    <div>truck <strong id="total-truck">__TOTAL_TRUCK__</strong></div>
    <div>bus <strong id="total-bus">__TOTAL_BUS__</strong></div>
  </div>
  <h2>In frame</h2>
  <div class="now">
    <div>car <strong id="now-car">__NOW_CAR__</strong></div>
    <div>truck <strong id="now-truck">__NOW_TRUCK__</strong></div>
    <div>bus <strong id="now-bus">__NOW_BUS__</strong></div>
  </div>
  <h2>Today by hour</h2>
  <p class="note" id="chart-summary">__CHART_SUMMARY__</p>
  <p class="legend"><span class="swatch-total"></span>total<span class="swatch-car"></span>car<span class="swatch-truck"></span>truck<span class="swatch-bus"></span>bus</p>
  <svg id="chart" viewBox="0 0 720 240" role="img" aria-label="Vehicle counts by hour"></svg>
  <h2>Live view</h2>
  <img class="live" src="/stream" alt="Annotated camera view">
  <p class="note" id="paths">Log: __LOG_PATH__</p>
</main>
<script>
const totals = ["car", "truck", "bus"];
const chartColors = {total: "#f2f2f2", car: "#00c800", truck: "#ff8c00", bus: "#3d7dff"};

function svgEl(name, attrs) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const key in attrs) el.setAttribute(key, attrs[key]);
  return el;
}

function drawChart(hourly) {
  const svg = document.getElementById("chart");
  const summary = document.getElementById("chart-summary");
  if (!svg || !hourly || !hourly.series) return;
  const names = ["total", "car", "truck", "bus"];
  let max = 1;
  let dayTotal = 0;
  for (const name of names) {
    const values = hourly.series[name] || [];
    for (let i = 0; i < values.length; i++) {
      const value = Number(values[i]) || 0;
      if (value > max) max = value;
      if (name === "total") dayTotal += value;
    }
  }
  if (summary) {
    summary.textContent = "Today " + (hourly.date || "") + " (local time): " + dayTotal;
  }
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  const width = 720, height = 240, left = 36, right = 12, top = 16, bottom = 28;
  const innerW = width - left - right;
  const innerH = height - top - bottom;
  function x(i) { return left + (i / 23) * innerW; }
  function y(value) { return top + innerH - (value / max) * innerH; }
  svg.appendChild(svgEl("line", {x1: left, y1: top + innerH, x2: width - right, y2: top + innerH, stroke: "#444"}));
  svg.appendChild(svgEl("line", {x1: left, y1: top, x2: left, y2: top + innerH, stroke: "#444"}));
  for (const label of [0, max]) {
    const text = svgEl("text", {x: 4, y: y(label) + 4, fill: "#aaa", "font-size": "11"});
    text.textContent = String(label);
    svg.appendChild(text);
  }
  for (const hour of [0, 6, 12, 18]) {
    const text = svgEl("text", {x: x(hour) - 8, y: height - 8, fill: "#aaa", "font-size": "11"});
    text.textContent = String(hour).padStart(2, "0");
    svg.appendChild(text);
  }
  for (const name of names) {
    const values = hourly.series[name] || [];
    const points = [];
    for (let i = 0; i < 24; i++) points.push(x(i).toFixed(1) + "," + y(Number(values[i]) || 0).toFixed(1));
    const attrs = {fill: "none", stroke: chartColors[name], "stroke-width": name === "total" ? "2.5" : "1.75", points: points.join(" ")};
    if (name === "total") attrs["stroke-dasharray"] = "5 4";
    svg.appendChild(svgEl("polyline", attrs));
  }
}

async function tick() {
  try {
    const response = await fetch("/api/status", {cache: "no-store"});
    const data = await response.json();
    for (const name of totals) {
      const total = document.getElementById("total-" + name);
      const now = document.getElementById("now-" + name);
      if (total && data.counts) total.textContent = data.counts[name] ?? 0;
      if (now && data.active) now.textContent = data.active[name] ?? 0;
    }
    drawChart(data.hourly);
  } catch (err) {
    /* keep the last numbers on screen */
  }
}
tick();
setInterval(tick, 1000);
</script>
<noscript><meta http-equiv="refresh" content="2"></noscript>
</body>
</html>
"""

_CAPTURES_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ubun-cam captures</title>
<style>__CSS__</style>
</head>
<body>
<main>
  <h1>Bus captures</h1>
  <p class="note">One annotated image for each newly counted bus track. <a href="/">Back to the dashboard</a></p>
  <div class="captures" id="captures">__CAPTURES__</div>
  <p class="note">Captures: __CAPTURES_DIR__</p>
</main>
<script>
async function tick() {
  try {
    const response = await fetch("/api/status", {cache: "no-store"});
    const data = await response.json();
    const box = document.getElementById("captures");
    const caps = Array.isArray(data.captures) ? data.captures : [];
    const key = caps.map(function (item) { return item.url; }).join("|");
    if (box.dataset.key === key) return;
    box.dataset.key = key;
    box.replaceChildren();
    if (!caps.length) {
      const empty = document.createElement("p");
      empty.textContent = "No bus captures yet.";
      box.append(empty);
      return;
    }
    for (const cap of caps) {
      if (!cap.url || String(cap.url).indexOf("/captures/") !== 0 || String(cap.url).indexOf("..") !== -1) continue;
      const figure = document.createElement("figure");
      const link = document.createElement("a");
      link.href = cap.url;
      const img = document.createElement("img");
      img.src = cap.url;
      img.alt = cap.file || "bus";
      link.append(img);
      const caption = document.createElement("figcaption");
      caption.textContent = cap.file || "";
      figure.append(link, caption);
      box.append(figure);
    }
  } catch (err) {
    /* keep the last gallery on screen */
  }
}
tick();
setInterval(tick, 2000);
</script>
<noscript><meta http-equiv="refresh" content="2"></noscript>
</body>
</html>
"""


def list_captures(directory: Path, limit: int = _CAPTURE_LIMIT) -> list[dict]:
    if not directory.is_dir():
        return []
    found: list[tuple[float, str, int]] = []
    for path in directory.iterdir():
        if not path.is_file():
            continue
        match = CAPTURE_NAME.match(path.name)
        if match is None:
            continue
        found.append((path.stat().st_mtime, path.name, int(match.group(1))))
    found.sort(key=lambda item: item[0], reverse=True)
    return [
        {"track_id": track_id, "file": name, "url": "/captures/" + name}
        for _mtime, name, track_id in found[:limit]
    ]


def hourly_counts(path: Path | None, now: datetime | None = None) -> dict:
    """Count today's detection events by local hour.

    Hours are 00–23 on the machine's local calendar day of ``now`` (or the
    current local day). ``total`` is car + truck + bus. Summary lines, other
    classes, and other days are ignored. Log timestamps are UTC.
    """
    moment = now.astimezone() if now is not None else datetime.now().astimezone()
    day = moment.date()
    buckets = {name: [0] * 24 for name in _CLASSES}
    if path is not None and Path(path).is_file():
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                record = _detection(line)
                if record is None:
                    continue
                name = record.get("class")
                if name not in buckets:
                    continue
                local = _local_stamp(record.get("ts"))
                if local is None or local.date() != day:
                    continue
                buckets[name][local.hour] += 1
    series = {
        "total": [sum(buckets[name][hour] for name in _CLASSES) for hour in range(24)],
        "car": buckets["car"],
        "truck": buckets["truck"],
        "bus": buckets["bus"],
    }
    return {
        "date": day.isoformat(),
        "hours": [f"{hour:02d}" for hour in range(24)],
        "series": series,
    }


def _detection(line: str) -> dict | None:
    text = line.strip()
    if not text:
        return None
    try:
        record = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(record, dict) or record.get("event") != "detection":
        return None
    return record


def _local_stamp(value: object) -> datetime | None:
    try:
        stamp = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone()


class Dashboard:
    """In-process counts plus a localhost page. Bind address is always 127.0.0.1."""

    def __init__(
        self,
        classes: tuple[str, ...] | list[str],
        capture_dir: Path,
        port: int = 8765,
        log_path: Path | None = None,
    ):
        self.host = _HOST
        self.port = port
        self.capture_dir = Path(capture_dir)
        self.log_path = Path(log_path) if log_path is not None else None
        self._classes = tuple(classes)
        self.counts = {name: 0 for name in self._classes}
        self.active = {name: 0 for name in self._classes}
        self._jpeg: bytes | None = None
        self._seq = 0
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._serving = threading.Event()
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._closed = False
        self._hourly_lock = threading.Lock()
        self._hourly_key: tuple | None = None
        self._hourly_cached: dict | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def start(self) -> None:
        if self._httpd is not None:
            return
        httpd = _Server((self.host, self.port), self)
        bound_host, bound_port = httpd.server_address[:2]
        if bound_host != self.host:
            httpd.server_close()
            raise RuntimeError(f"dashboard refused to bind {self.host}")
        self._httpd = httpd
        self.port = int(bound_port)
        self._thread = threading.Thread(target=self._serve, name="ubuncam-dashboard", daemon=True)
        self._thread.start()
        if not self._serving.wait(timeout=2):
            raise RuntimeError("dashboard did not start")

    def stop(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        thread = self._thread
        httpd = self._httpd
        if thread is not None:
            self._serving.wait(timeout=2)
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
            self._httpd = None
        if thread is not None:
            thread.join(timeout=3)
            self._thread = None

    def update(self, counts: dict, active: dict, jpeg: bytes | None = None) -> None:
        with self._cond:
            self.counts = self._numbers(counts)
            self.active = self._numbers(active)
            if jpeg is not None:
                self._jpeg = jpeg
                self._seq += 1
            self._cond.notify_all()

    def snapshot(self) -> dict:
        with self._cond:
            counts = dict(self.counts)
            active = dict(self.active)
        return {
            "counts": counts,
            "active": active,
            "captures_dir": str(self.capture_dir.resolve()),
            "log_path": None if self.log_path is None else str(self.log_path.resolve()),
            "captures": list_captures(self.capture_dir),
            "hourly": self.hourly_snapshot(),
        }

    def hourly_snapshot(self) -> dict:
        path = self.log_path
        today = datetime.now().astimezone().date().isoformat()
        if path is None or not path.is_file():
            return hourly_counts(None)
        stat = path.stat()
        key = (stat.st_mtime_ns, stat.st_size, today)
        with self._hourly_lock:
            if key == self._hourly_key and self._hourly_cached is not None:
                return self._hourly_cached
            parsed = hourly_counts(path)
            self._hourly_key = key
            self._hourly_cached = parsed
            return parsed

    def latest_jpeg(self, after_seq: int, timeout: float) -> tuple[int, bytes | None]:
        with self._cond:
            if self._seq <= after_seq and not self._stop.is_set():
                self._cond.wait(timeout)
            return self._seq, self._jpeg

    def _numbers(self, values: dict) -> dict[str, int]:
        merged = {name: 0 for name in self._classes}
        for key, value in values.items():
            merged[str(key)] = int(value)
        return merged

    def _serve(self) -> None:
        self._serving.set()
        if self._httpd is not None:
            self._httpd.serve_forever(poll_interval=0.2)


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address, dashboard: Dashboard):
        self.dashboard = dashboard
        super().__init__(server_address, _Handler)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def do_GET(self) -> None:  # noqa: N802
        try:
            self._route()
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            self.close_connection = True

    def log_message(self, fmt: str, *args) -> None:
        return

    def _dashboard(self) -> Dashboard:
        return self.server.dashboard  # type: ignore[attr-defined]

    def _route(self) -> None:
        path = urlsplit(self.path).path
        dashboard = self._dashboard()
        if path == "/":
            self._send(200, "text/html; charset=utf-8", _page(dashboard.snapshot()))
        elif path in ("/captures", "/captures/"):
            self._send(200, "text/html; charset=utf-8", _captures_page(dashboard.snapshot()))
        elif path == "/api/status":
            body = json.dumps(dashboard.snapshot()).encode("utf-8")
            self._send(200, "application/json", body)
        elif path == "/stream":
            self._stream()
        elif path.startswith("/captures/"):
            self._capture(path)
        else:
            self.send_error(404)

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _capture(self, path: str) -> None:
        rel = unquote(path.removeprefix("/captures/"))
        if rel != Path(rel).name or CAPTURE_NAME.match(rel) is None:
            self.send_error(404)
            return
        root = self._dashboard().capture_dir.resolve()
        target = (root / rel).resolve()
        if target.parent != root or not target.is_file():
            self.send_error(404)
            return
        self._send(200, "image/jpeg", target.read_bytes())

    def _stream(self) -> None:
        dash = self._dashboard()
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.connection.settimeout(2.0)
        seq = 0
        while not dash.stopped:
            seq, jpeg = dash.latest_jpeg(seq, 0.5)
            if jpeg is None:
                continue
            if dash.stopped:
                break
            header = (
                b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                + str(len(jpeg)).encode("ascii")
                + b"\r\n\r\n"
            )
            self.wfile.write(header + jpeg + b"\r\n")
            self.wfile.flush()


def _num(mapping: dict, name: str) -> str:
    try:
        return str(int(mapping.get(name, 0)))
    except (TypeError, ValueError):
        return "0"


def _gallery(snapshot: dict) -> str:
    import html

    figures = []
    for item in snapshot.get("captures") or []:
        url = str(item.get("url") or "")
        if not url.startswith("/captures/") or ".." in url:
            continue
        safe_url = html.escape(url, quote=True)
        label = html.escape(str(item.get("file") or ""))
        figures.append(
            f'<figure><a href="{safe_url}"><img src="{safe_url}" alt="{label}"></a>'
            f"<figcaption>{label}</figcaption></figure>"
        )
    return "".join(figures) or "<p>No bus captures yet.</p>"


def _chart_summary(hourly: dict) -> str:
    series = (hourly or {}).get("series") or {}
    total = series.get("total") or []
    try:
        day_total = sum(int(value) for value in total)
    except (TypeError, ValueError):
        day_total = 0
    date = (hourly or {}).get("date") or ""
    return f"Today {date} (local time): {day_total}"


def _page(snapshot: dict) -> bytes:
    import html

    counts = snapshot.get("counts") or {}
    active = snapshot.get("active") or {}
    log_path = snapshot.get("log_path")
    log_text = html.escape(str(log_path)) if log_path else "—"
    page = (
        _PAGE.replace("__CSS__", _CSS)
        .replace("__TOTAL_CAR__", _num(counts, "car"))
        .replace("__TOTAL_TRUCK__", _num(counts, "truck"))
        .replace("__TOTAL_BUS__", _num(counts, "bus"))
        .replace("__NOW_CAR__", _num(active, "car"))
        .replace("__NOW_TRUCK__", _num(active, "truck"))
        .replace("__NOW_BUS__", _num(active, "bus"))
        .replace("__CHART_SUMMARY__", html.escape(_chart_summary(snapshot.get("hourly") or {})))
        .replace("__LOG_PATH__", log_text)
    )
    return page.encode("utf-8")


def _captures_page(snapshot: dict) -> bytes:
    import html

    page = (
        _CAPTURES_PAGE.replace("__CSS__", _CSS)
        .replace("__CAPTURES__", _gallery(snapshot))
        .replace("__CAPTURES_DIR__", html.escape(str(snapshot.get("captures_dir") or "")))
    )
    return page.encode("utf-8")


def _fetch_json(url: str) -> dict:
    last: Exception | None = None
    for _ in range(20):
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            time.sleep(0.05)
    raise AssertionError(f"dashboard did not respond at {url}: {last}")


def self_check_hourly(directory: Path) -> dict:
    """Bucket one local day of detection events. No camera and no weights."""
    directory.mkdir(parents=True, exist_ok=True)
    now = datetime.now().astimezone().replace(minute=30, second=0, microsecond=0)
    morning = now.replace(hour=9)
    afternoon = now.replace(hour=15)
    midnight = now.replace(hour=0, minute=5)
    late = now.replace(hour=23, minute=50)
    yesterday = morning - timedelta(days=1)
    path = directory / "events.jsonl"

    def detection(moment: datetime, name: str, track_id: int) -> str:
        return json.dumps(
            {
                "ts": moment.astimezone(timezone.utc).isoformat(),
                "event": "detection",
                "track_id": track_id,
                "class": name,
                "conf": 0.8,
                "bbox": [0, 0, 1, 1],
            },
            separators=(",", ":"),
        )

    lines = [
        detection(morning, "car", 1),
        detection(morning, "bus", 2),
        detection(afternoon, "truck", 3),
        detection(afternoon, "car", 4),
        detection(yesterday, "car", 5),
        detection(midnight, "car", 6),
        detection(late, "bus", 7),
        detection(morning, "person", 8),
        json.dumps({"ts": morning.astimezone(timezone.utc).isoformat(), "event": "summary", "counts": {"car": 1}}),
        "not-json",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = hourly_counts(path, now=now)
    series = result["series"]
    assert result["date"] == now.date().isoformat()
    assert result["hours"][0] == "00" and result["hours"][23] == "23"
    assert len(series["total"]) == 24
    for hour in range(24):
        assert series["total"][hour] == series["car"][hour] + series["truck"][hour] + series["bus"][hour]
    assert series["car"][9] == 1
    assert series["bus"][9] == 1
    assert series["truck"][9] == 0
    assert series["total"][9] == 2
    assert series["car"][15] == 1
    assert series["truck"][15] == 1
    assert series["total"][15] == 2
    assert series["car"][0] == 1
    assert series["bus"][23] == 1
    assert series["total"][0] == 1
    assert series["total"][23] == 1
    assert sum(series["car"]) == 3
    assert sum(series["truck"]) == 1
    assert sum(series["bus"]) == 2
    assert sum(series["total"]) == 6
    return {"today": sum(series["total"]), "at_9": series["total"][9]}


def self_check_dashboard() -> dict:
    """Serve counts, today's hourly buckets, and the captures page. No camera or weights."""
    tmp = Path(tempfile.mkdtemp(prefix="ubuncam-dash-"))
    log_path = tmp / "events.jsonl"
    dashboard = Dashboard(("car", "truck", "bus"), tmp, port=0, log_path=log_path)
    try:
        dashboard.start()
        assert dashboard.host == "127.0.0.1"
        dashboard.update(
            {"car": 4, "truck": 1, "bus": 2},
            {"car": 1, "truck": 0, "bus": 1},
        )
        now = datetime.now().astimezone()
        log_path.write_text(
            json.dumps(
                {
                    "ts": now.astimezone(timezone.utc).isoformat(),
                    "event": "detection",
                    "track_id": 3,
                    "class": "bus",
                    "conf": 0.5,
                    "bbox": [0, 0, 1, 1],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        payload = _fetch_json(dashboard.url + "api/status")
        assert payload["counts"] == {"car": 4, "truck": 1, "bus": 2}
        assert payload["active"] == {"car": 1, "truck": 0, "bus": 1}
        assert payload["captures"] == []
        assert payload["captures_dir"] == str(tmp.resolve())
        hourly = payload["hourly"]
        assert len(hourly["series"]["total"]) == 24
        assert hourly["series"]["bus"][now.hour] == 1
        assert hourly["series"]["total"][now.hour] == 1
        assert hourly["series"]["car"][now.hour] == 0
        assert hourly["date"] == now.date().isoformat()
        with urllib.request.urlopen(dashboard.url, timeout=2) as response:
            page = response.read().decode("utf-8")
        assert 'id="total-bus">2<' in page
        assert 'id="now-bus">1<' in page
        assert 'href="/captures"' in page
        assert 'id="chart"' in page
        assert "Recent bus captures" not in page
        with urllib.request.urlopen(dashboard.url + "captures", timeout=2) as response:
            captures_page = response.read().decode("utf-8")
        assert "Bus captures" in captures_page
        assert 'href="/"' in captures_page
        assert "No bus captures yet." in captures_page
        with urllib.request.urlopen(dashboard.url + "captures/", timeout=2) as response:
            assert "Bus captures" in response.read().decode("utf-8")
        return payload
    finally:
        dashboard.stop()
        shutil.rmtree(tmp, ignore_errors=True)
