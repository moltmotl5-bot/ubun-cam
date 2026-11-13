# Install ubun-cam on Ubuntu

The short path is also in the README. This page is the package list, camera permissions, and the service unit.

Written for Ubuntu 26.04.1 LTS on x86_64 (the i7-8700 machine). Python 3.11 or newer is required. The distro `python3` is the one to use.

## 1. Packages

```bash
sudo apt update
sudo apt install -y \
  python3 \
  python3-venv \
  python3-pip \
  v4l-utils \
  libgl1 \
  libglib2.0-0t64 \
  libgtk-3-0t64
```

`libgl1` and GTK are there so the pip OpenCV build can open a preview window. `v4l-utils` gives you `v4l2-ctl`.

If apt does not have the `t64` names, install the unversioned ones instead:

```bash
sudo apt install -y libglib2.0-0 libgtk-3-0
```

Do not install `python3-opencv` from apt into the same environment. The venv uses the pinned `opencv-python` wheel, which includes `imshow`.

## 2. Setup script

```bash
git clone https://github.com/moltmotl5-bot/ubun-cam.git
cd ubun-cam
bash scripts/setup.sh
```

The script:

1. Creates `.venv` in the repo.
2. Installs the pinned packages in `requirements.txt` (Ultralytics, OpenCV, PyYAML, ONNX, ONNX Runtime, a CPU build of PyTorch). The CPU index is declared at the top of that file, so do not replace it with a plain `pip install ultralytics` — that pulls the much larger CUDA build, which this machine does not use.
3. Installs the `ubuncam` package in that venv.
4. Copies `config.example.yaml` to `config.yaml` when `config.yaml` is absent.
5. Runs `python -m ubuncam --self-check` (no camera, no weights).
6. Downloads `yolo11n.pt` and exports `models/yolo11n.onnx`.

Step 6 needs network access. The PyTorch wheels pulled in by Ultralytics are the large download. Re-running the script is safe: an existing `config.yaml` is left alone, and an existing ONNX file is not exported again.

Start the preview:

```bash
cd ubun-cam
source .venv/bin/activate
python -m ubuncam
```

Quit with **q** in the window, or Ctrl+C.

## 3. Camera permissions

The process must be allowed to open `/dev/video*`. That is the `video` group, not VLC.

```bash
ls -l /dev/video*
groups
v4l2-ctl --list-devices
python -m ubuncam --list-cameras
```

If `ls` shows `crw-rw----` owned by `root:video` and `groups` does not list `video`:

```bash
sudo usermod -aG video "$USER"
```

Group membership is read at login. Log out and back in, then check `groups` again. Unplug and replug the camera if the node never appears.

Pick the device that actually streams pictures. A lot of USB webcams create a capture node and a metadata node; VLC may have opened `/dev/video0` while OpenCV needs `/dev/video1`, or the other way around. A Microsoft LifeCam HD-3000 often captures on `/dev/video1`, not `/dev/video2`. Set it in `config.yaml`:

```yaml
camera:
  device: /dev/video0
```

Or for one run:

```bash
UBUNCAM_CAMERA=/dev/video1 python -m ubuncam
```

`camera.fourcc: MJPG` asks the camera for Motion-JPEG, which is usually the right mode at 1280x720. If the image looks wrong, set `fourcc: ""` and try a smaller `width` / `height`.

## 4. Check the model without a camera

```bash
python -m ubuncam --image /path/to/street.jpg
```

That writes `logs/preview.jpg` and a detection line in `logs/events.jsonl` when the model finds a car, truck, or bus. The synthetic self-check does not load the model:

```bash
python -m ubuncam --self-check
python -m unittest tests.test_smoke -v
```

`--self-check` and the unit tests also cover two camera-free cases: one annotated JPEG per new bus track, and the dashboard counts payload on `127.0.0.1`.

## 5. Dashboard and bus captures

With the service running, open this URL in a browser on the same machine:

```text
http://127.0.0.1:8765/
```

The page uses the same session totals as the tracker. It also shows how many cars, trucks, and buses are in the current frame, a live view of the annotated frames, and the newest bus images. The process binds to `127.0.0.1` only. Config keys are `dashboard.enabled`, `dashboard.port`, and `captures.dir` (see `config.example.yaml`).

Each new bus track saves one annotated JPEG (boxes and labels) under `logs/captures/`. The matching `detection` line in `logs/events.jsonl` has a `capture` path. Later frames of the same track do not add another file. `logs/` is gitignored.

To check a session:

1. Watch the road and the preview, or the dashboard live view.
2. Compare the session totals with what passed. **In frame** falls when the vehicle leaves.
3. Open a file in `logs/captures/` (also listed on the dashboard) and confirm that a counted bus is a bus.
4. Read `logs/events.jsonl` for the `detection` and `summary` lines.

Headless mode still does the count, the log, the bus JPEGs, and the dashboard:

```yaml
preview:
  enabled: false
dashboard:
  enabled: true
  port: 8765
```

From another computer, forward the local port and open the same URL there:

```bash
ssh -L 8765:127.0.0.1:8765 user@the-ubuntu-machine
```

## 6. systemd user service

Optional. Enable it after `scripts/setup.sh` has succeeded and `config.yaml` points at the right camera.

For a service, turn the window off so it does not need a desktop session. Leave the dashboard on if you want the counts page on this machine:

```yaml
preview:
  enabled: false
dashboard:
  enabled: true
```

Install the unit (edit the paths inside it first if the clone is not `~/ubun-cam`):

```bash
mkdir -p ~/.config/systemd/user
cp deploy/ubuncam.service ~/.config/systemd/user/ubuncam.service
systemctl --user daemon-reload
systemctl --user enable --now ubuncam.service
journalctl --user -u ubuncam -f
```

Keep it running after logout:

```bash
sudo loginctl enable-linger "$USER"
```

To show a window from the user service, set `preview.enabled: true` and uncomment the `DISPLAY` and `XAUTHORITY` lines in the unit. The display name on a Wayland session with XWayland is often still `:0`.

## 7. Troubleshooting

| What you see | What to change |
| --- | --- |
| `Cannot open camera` / `Permission denied` | `video` group, then log in again. Confirm with `ls -l /dev/video*`. |
| Opens, but the frame is black | Try the other `/dev/videoN` from `v4l2-ctl --list-devices`. |
| `No DISPLAY or WAYLAND_DISPLAY` | You are on a text console. Export `DISPLAY` or set `preview.enabled: false`. Logging, bus captures, and the dashboard still run. |
| Dashboard page does not load | Open it on this machine at `http://127.0.0.1:<dashboard.port>/`. The process does not listen on other interfaces. If the port is taken, the log says so and detection continues. |
| Preview backend error | Install the GTK / libgl packages in section 1, then retry. The loop drops the window and keeps logging. |
| Very low fps | Keep `imgsz: 640` and the `n` model. MJPEG capture avoids a slow YUYV conversion. OpenVINO is not wired up. |
| Setup fails downloading weights | Check network and run `python -m ubuncam --export-model` again. The `.pt` file lands in `models/`. |
| `unsupported classes` | Only `car`, `truck`, and `bus`. Double-decker is not its own class yet. |
