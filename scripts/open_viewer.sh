#!/usr/bin/env bash
# Open the EgoInfinity viser 3D viewer for a processed clip, in Chrome with
# hardware WebGL forced on (new GPUs like the RTX 5080 get blocklisted by
# default -> "WebGL unavailable").
#
# Usage: ./scripts/open_viewer.sh <clip_dir> [port]
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
source "${CONFIG:-$HERE/config.sh}"

CLIP="${1:?usage: open_viewer.sh <clip_dir> [port]}"
PORT="${2:-8080}"

# start viser (load-cache view mode) if not already serving
if ! curl -s -o /dev/null "http://localhost:$PORT"; then
    ( cd "$EI_REPO" && nohup "$EI_PYTHON" scripts/exo_pipeline.py --load-cache \
        --cache-dir "$CLIP" --frames_dir "$CLIP/extract_frames/frames" \
        --port "$PORT" >/tmp/ei_viser.log 2>&1 & )
    for _ in $(seq 1 30); do curl -s -o /dev/null "http://localhost:$PORT" && break; sleep 1; done
fi
echo "[viewer] viser serving: http://localhost:$PORT"

# Chrome with hardware WebGL (native GL backend, GPU blocklist overridden).
# If your X display can't do hardware GLX, swap --use-angle=gl for
# --enable-unsafe-swiftshader (software WebGL, always works).
nohup google-chrome --user-data-dir=/tmp/ei_chrome_gpu \
    --ignore-gpu-blocklist --use-angle=gl \
    --new-window "http://localhost:$PORT" >/tmp/ei_chrome.log 2>&1 &
echo "[viewer] opened Chrome at http://localhost:$PORT"
