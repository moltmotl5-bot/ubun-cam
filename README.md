# ubun-cam

Local Ubuntu service that uses a USB webcam to detect road vehicles and show a live annotated preview.

Phase 1 detects the COCO classes **car**, **truck**, and **bus**, counts each tracked vehicle once, appends a JSONL log, and draws boxes with OpenCV. A double-decker bus is counted as **bus**. Inference is YOLO11n on CPU, exported once to ONNX and run through Ultralytics.

Target machine this was written for: x86 Intel i7-8700, 32 GB RAM, Ubuntu 26.04.1 LTS, USB webcam that already works in VLC. Capture is OpenCV / V4L2 on `/dev/video*`. VLC is not part of the pipeline.

## Architecture

```
/dev/video*  --OpenCV V4L2-->  frame
                                |
                                v
                    YOLO11n ONNX (ultralytics, CPU)
                                |
                                v
                    keep car / truck / bus
                                |
                                v
                    greedy IoU tracker (same class)
                                |
                    +-----------+-----------+
                    |                       |
                    v                       v
            JSONL log + stdout      cv2.imshow boxes
```

A new track id is one count. The same vehicle stays one count while its box still overlaps the previous box. If it leaves the frame long enough for the track to expire, the next sighting is a new count.

Each new track appends one `detection` line. Every `log.summary_interval` seconds, and again on exit, a `summary` line records the totals.

```json
{"ts":"2026-11-03T09:50:00+00:00","event":"detection","track_id":3,"class":"car","conf":0.83,"bbox":[10.0,20.0,110.0,80.0]}
{"ts":"2026-11-03T09:51:00+00:00","event":"summary","counts":{"car":4,"truck":1,"bus":0}}
```

## Run it

From a desktop session on the Ubuntu machine (preview needs a display):

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip v4l-utils libgl1 libglib2.0-0t64 libgtk-3-0t64

git clone https://github.com/moltmotl5-bot/ubun-cam.git
cd ubun-cam
bash scripts/setup.sh
source .venv/bin/activate
python -m ubuncam
```

`scripts/setup.sh` creates `.venv`, installs `requirements.txt`, copies `config.example.yaml` to `config.yaml` if you do not already have one, runs a camera-free self-check, and downloads YOLO11n into `models/` as ONNX. Requirements pin a **CPU** build of PyTorch, so this Intel machine does not download the CUDA wheels. That PyTorch download is still the slow part of setup. Later starts reuse `models/yolo11n.onnx` and do not download weights again.

Press **q** in the preview window to quit. Ctrl+C also stops the loop and writes a final summary.

If `libglib2.0-0t64` or `libgtk-3-0t64` is not in your apt index, install `libglib2.0-0` and `libgtk-3-0` instead. More detail, including the `video` group, is in [docs/INSTALL.md](docs/INSTALL.md).

### Camera

```bash
python -m ubuncam --list-cameras
v4l2-ctl --list-devices
```

Set `camera.device` in `config.yaml` to the node that shows a picture (often `/dev/video0` or `/dev/video1`). If opening the device says permission denied:

```bash
sudo usermod -aG video "$USER"
```

Log out and back in after that. Confirm with `groups` (the word `video` should be listed).

### Useful commands

```bash
python -m ubuncam --self-check          # no camera, no weights
python -m ubuncam --export-model        # download + ONNX export only
python -m ubuncam --image photo.jpg     # one frame, writes logs/preview.jpg
python -m unittest tests.test_smoke -v  # same checks as --self-check, plus config
```

Run these from the repo root so `config.yaml` and `models/` resolve.

## Config

`config.example.yaml` is the template. `config.yaml` is local and gitignored.

| Key | Meaning |
| --- | --- |
| `camera.device` | `0` or `/dev/video0` |
| `camera.width` / `height` | Requested capture size |
| `camera.fourcc` | `MJPG` is a good USB default. `""` leaves the camera alone |
| `model.confidence` | Detection threshold |
| `model.weights` | ONNX path. Exported if missing |
| `classes` | Any non-empty subset of `car`, `truck`, `bus` |
| `log.path` | JSONL event log |
| `preview.enabled` | Live window. Turn off for a headless service |

Environment overrides, applied on top of the file:

| Variable | Effect |
| --- | --- |
| `UBUNCAM_CONFIG` | Config path |
| `UBUNCAM_CAMERA` | Device index or path |
| `UBUNCAM_CONF` | Confidence 0–1 |
| `UBUNCAM_LOG` | Log path |
| `UBUNCAM_PREVIEW` | `0` / `1` |

Paths in the file are relative to the config file's directory.

## Service

An optional user unit lives in `deploy/ubuncam.service`. It assumes the repo is `~/ubun-cam`. For a long-running logger, set `preview.enabled: false` so it does not need a display. Enable steps are in the unit file and in [docs/INSTALL.md](docs/INSTALL.md).

## Hardware notes

- CPU only. YOLO11n at 640 is the small model, which is the point of the `n` variant on an i7-8700.
- ONNX Runtime (via Ultralytics) is the runtime. OpenVINO is a possible later speed-up on this Intel CPU and is not installed here.
- 32 GB RAM is plenty. The heavy part is the first CPU PyTorch download, not the steady-state process.
- The webcam must already appear as a V4L2 device. This app does not open VLC.

Ultralytics and the YOLO11n weights are AGPL-3.0.

## MVP and later

In this phase:

- COCO `car`, `truck`, `bus` only. There is no custom double-decker class. A double-decker is a bus.
- Counts and the JSONL log stay on disk. Nothing is published off the machine.
- Preview is a local OpenCV window.

Not in this phase:

- A separate double-decker class (that needs a custom dataset and weights).
- MQTT or any other network publish.
- OpenVINO, cloud APIs, or a container.

## Layout

```
config.example.yaml
src/ubuncam/          # python -m ubuncam
scripts/setup.sh
deploy/ubuncam.service
tests/test_smoke.py
docs/INSTALL.md
```
