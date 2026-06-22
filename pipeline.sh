#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# EgoInfinity + MANO pipeline
#   input clip  ->  EgoInfinity (HaMeR hands + open-vocab detector)
#               ->  rendered videos: MANO overlay, 3D reconstruction, 2D tracking
#
# Usage:
#   cp config.example.sh config.sh   # then edit paths
#   ./pipeline.sh <clip_dir> [out_dir] [extra `egoinfinity process` args...]
#
# <clip_dir> is an artifact dir containing:
#     manifest.json                 {"video_uri": "...", "objects": ["red mug", ...]}
#     (frames are auto-extracted by EgoInfinity's extract_frames stage)
# Outputs (default <clip_dir>/outputs/):
#     mano_overlay.mp4   reconstruction_3d.mp4   tracking_2d.mp4
# ─────────────────────────────────────────────────────────────────────────
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${CONFIG:-$HERE/config.sh}"

CLIP="${1:?usage: ./pipeline.sh <clip_dir> [out_dir] [process args...]}"
OUT="${2:-$CLIP/outputs}"
shift || true; shift 2>/dev/null || true
mkdir -p "$OUT"

PY="$EI_PYTHON"
export HAMER_REPO
SOCK="/tmp/egoinfinity_sam3_${USER}.sock"
export SAM3_WORKER_SOCKET="$SOCK"

cd "$EI_REPO"

# ── 1. start the object-detector worker on EgoInfinity's SAM3 socket ──────
#   DETECTOR=sam3  -> real SAM 3.1 (sam3 env, local checkpoint)
#   DETECTOR=gdino -> GroundingDINO + SAM2 (ungated fallback)
rm -f "$SOCK" "$SOCK.ready"
if [ "${DETECTOR:-sam3}" = "sam3" ]; then
    echo "[pipeline] starting SAM 3.1 detector worker ..."
    BPE="$SAM3_REPO/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    HF_HUB_OFFLINE=1 SAM3_CKPT="$SAM3_CKPT" \
      "$SAM3_PYTHON" scripts/sam3_worker.py --socket "$SOCK" \
      --version sam3.1 --bpe_path "$BPE" >/tmp/sam3_worker.log 2>&1 &
else
    echo "[pipeline] starting GroundingDINO+SAM2 detector worker ..."
    "$PY" scripts/gdino_sam2_worker.py --socket "$SOCK" >/tmp/gdino_worker.log 2>&1 &
fi
WPID=$!
trap 'kill $WPID 2>/dev/null' EXIT
for _ in $(seq 1 150); do
  [ -f "$SOCK.ready" ] && break
  kill -0 $WPID 2>/dev/null || { echo "[pipeline] detector worker died (see /tmp/gdino_worker.log)"; exit 1; }
  sleep 2
done
[ -f "$SOCK.ready" ] || { echo "[pipeline] detector worker not ready"; exit 1; }
echo "[pipeline] detector ready."

# ── 2. run EgoInfinity (HaMeR hands; FP++ rotation bake disabled) ─────────
echo "[pipeline] running EgoInfinity (hands=$HANDS) ..."
"$PY" -m egoinfinity process "$CLIP" \
    --hand-recon-backend "$HANDS" \
    --set bake_fp.enabled=false "$@" || { echo "[pipeline] EgoInfinity failed"; exit 1; }

PKL="$CLIP/pipeline_result.pkl.gz"
FRAMES="$CLIP/extract_frames/frames"
[ -f "$PKL" ] || { echo "[pipeline] no pkl produced at $PKL"; exit 1; }

# ── 3. render the output videos ──────────────────────────────────────────
enc() { ffmpeg -y -hide_banner -loglevel error -framerate 15 \
        -i "$1/%06d.png" -c:v libx264 -pix_fmt yuv420p -crf 20 "$2"; }
TMP="$(mktemp -d)"

echo "[pipeline] rendering MANO overlay ..."
"$PY" "$HERE/render/render_mano_overlay.py" "$PKL" "$TMP/mano" && enc "$TMP/mano" "$OUT/mano_overlay.mp4"

echo "[pipeline] rendering 3D reconstruction ..."
"$PY" "$HERE/render/render_3d.py" "$PKL" "$TMP/r3d" && enc "$TMP/r3d" "$OUT/reconstruction_3d.mp4"

echo "[pipeline] rendering 2D tracking ..."
"$PY" "$HERE/render/render_tracking_2d.py" "$PKL" "$FRAMES" "$TMP/t2d" && enc "$TMP/t2d" "$OUT/tracking_2d.mp4"

rm -rf "$TMP"
echo "[pipeline] DONE. Outputs:"
ls -lh "$OUT"/*.mp4
