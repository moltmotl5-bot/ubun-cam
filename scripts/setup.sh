#!/usr/bin/env bash
# Create the venv, install pinned deps, self-check, and export YOLO11n to ONNX.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required. Install the packages in docs/INSTALL.md first." >&2
  exit 1
fi

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e . --no-deps

if [[ ! -f config.yaml ]]; then
  cp config.example.yaml config.yaml
  echo "Wrote config.yaml from config.example.yaml"
fi

echo
echo "Self-check (no camera, no model):"
.venv/bin/python -m ubuncam --self-check

echo
CURRENT_USER="${USER:-$(id -un)}"
if id -nG "$CURRENT_USER" | tr ' ' '\n' | grep -qx video; then
  echo "User $CURRENT_USER is in the video group."
else
  echo "Camera permission: this user is not in the video group."
  echo "Run the next command, then log out and back in (or reboot):"
  echo "  sudo usermod -aG video \"$CURRENT_USER\""
fi
echo "See which node is the real camera:"
echo "  ls -l /dev/video*"
echo "  v4l2-ctl --list-devices"

echo
echo "Downloading YOLO11n and exporting ONNX into models/ (one time, needs network)..."
.venv/bin/python -m ubuncam --export-model

echo
echo "Start from the repo root:"
echo "  source .venv/bin/activate"
echo "  python -m ubuncam"
echo "Dashboard (this machine only): http://127.0.0.1:8765/"
echo "Bus images: logs/captures/   Events: logs/events.jsonl"
echo "Press q in the preview window to quit. Ctrl+C also stops."
