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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

CAPTURE_NAME = re.compile(r"^bus-(\d+)-(\d{8}T\d{12}Z)\.jpg$")
_HOST = "127.0.0.1"
_CAPTURE_LIMIT = 24

_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ubun-cam</title>
<style>
  body { margin: 0; font: 16px/1.4 system-ui, sans-serif; background: #141414; color: #f2f2f2; }
  main { max-width: 1100px; margin: 0 auto; padding: 1.25rem; }
  h1 { font-size: 1.25rem; margin: 0 0 0.25rem; }
  h2 { font-size: 1rem; margin: 1.25rem 0 0.5rem; }
  p.note { margin: 0 0 1rem; color: #bdbdbd; }
  .stats, .now { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.75rem; }
  .stats { margin-bottom: 0.5rem; }
  .stats div, .now div { background: #222; padding: 0.75rem 1rem; }
  .stats strong { display: block; font-size: 2.75rem; font-variant-numeric: tabular-nums; }
  .now strong { font-variant-numeric: tabular-nums; }
  img.live { display: block; width: 100%; max-height: 42vh; object-fit: contain; object-position: center top; background: #000; min-height: 120px; }
  @media (max-width: 640px) {
    main { padding: 0.75rem; }
    .stats strong { font-size: 2rem; }
    .stats div, .now div { padding: 0.55rem 0.6rem; }
    .captures figure { width: 100%; }
  }
  .captures { display: flex; flex-wrap: wrap; gap: 0.75rem; }
  .captures figure { margin: 0; width: 240px; }
  .captures img { width: 100%; background: #000; }
  figcaption { font-size: 0.8rem; color: #ccc; word-break: break-all; }
</style>
</head>
<body>
<main>
  <h1>ubun-cam</h1>
  <p class="note">Session totals count each tracked vehicle once since this process started. A double-decker is a bus. This page is only on this machine.</p>
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
  <h2>Live view</h2>
  <img class="live" src="/stream" alt="Annotated camera view">
  <h2>Recent bus captures</h2>
  <div class="captures" id="captures">__CAPTURES__</div>
  <p class="note" id="paths">Captures: __CAPTURES_DIR__<br>Log: __LOG_PATH__</p>
</main>
<script>
const totals = ["car", "truck", "bus"];
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
      if (!cap.url || String(cap.url).indexOf("/captures/") !== 0) continue;
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
        }

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
        if path == "/":
            self._send(200, "text/html; charset=utf-8", _page(self._dashboard().snapshot()))
        elif path == "/api/status":
            body = json.dumps(self._dashboard().snapshot()).encode("utf-8")
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


def _page(snapshot: dict) -> bytes:
    import html

    counts = snapshot.get("counts") or {}
    active = snapshot.get("active") or {}

    def num(mapping: dict, name: str) -> str:
        try:
            return str(int(mapping.get(name, 0)))
        except (TypeError, ValueError):
            return "0"

    figures = []
    for item in snapshot.get("captures") or []:
        url = str(item.get("url") or "")
        if not url.startswith("/captures/"):
            continue
        safe_url = html.escape(url, quote=True)
        label = html.escape(str(item.get("file") or ""))
        figures.append(
            f'<figure><a href="{safe_url}"><img src="{safe_url}" alt="{label}"></a>'
            f"<figcaption>{label}</figcaption></figure>"
        )
    gallery = "".join(figures) or "<p>No bus captures yet.</p>"
    captures_dir = html.escape(str(snapshot.get("captures_dir") or ""))
    log_path = snapshot.get("log_path")
    log_text = html.escape(str(log_path)) if log_path else "—"
    page = (
        _PAGE.replace("__TOTAL_CAR__", num(counts, "car"))
        .replace("__TOTAL_TRUCK__", num(counts, "truck"))
        .replace("__TOTAL_BUS__", num(counts, "bus"))
        .replace("__NOW_CAR__", num(active, "car"))
        .replace("__NOW_TRUCK__", num(active, "truck"))
        .replace("__NOW_BUS__", num(active, "bus"))
        .replace("__CAPTURES__", gallery)
        .replace("__CAPTURES_DIR__", captures_dir)
        .replace("__LOG_PATH__", log_text)
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


def self_check_dashboard() -> dict:
    """Serve the counts payload on 127.0.0.1. No camera and no weights."""
    tmp = Path(tempfile.mkdtemp(prefix="ubuncam-dash-"))
    dashboard = Dashboard(("car", "truck", "bus"), tmp, port=0, log_path=tmp / "events.jsonl")
    try:
        dashboard.start()
        assert dashboard.host == "127.0.0.1"
        dashboard.update(
            {"car": 4, "truck": 1, "bus": 2},
            {"car": 1, "truck": 0, "bus": 1},
        )
        payload = _fetch_json(dashboard.url + "api/status")
        assert payload["counts"] == {"car": 4, "truck": 1, "bus": 2}
        assert payload["active"] == {"car": 1, "truck": 0, "bus": 1}
        assert payload["captures"] == []
        assert payload["captures_dir"] == str(tmp.resolve())
        with urllib.request.urlopen(dashboard.url, timeout=2) as response:
            page = response.read().decode("utf-8")
        assert 'id="total-bus">2<' in page
        assert 'id="now-bus">1<' in page
        return payload
    finally:
        dashboard.stop()
        shutil.rmtree(tmp, ignore_errors=True)
