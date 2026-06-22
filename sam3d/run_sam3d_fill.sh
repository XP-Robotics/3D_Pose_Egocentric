#!/usr/bin/env bash
# Host B (>=32 GB GPU): reconstruct per-object 3D meshes for a clip whose
# detection/tracking was already done on Host A. Spawns ONLY the SAM 3D Objects
# worker (reuses the SAM3.1 masks already in pipeline_result.pkl.gz).
#
# Usage: ./sam3d/run_sam3d_fill.sh <clip_artifact_dir>
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
source "${CONFIG:-$HERE/config.sh}"

CLIP="${1:?usage: run_sam3d_fill.sh <clip_artifact_dir>}"
: "${SAM3D_REPO:?set SAM3D_REPO in config.sh (sam-3d-objects checkout)}"
: "${SAM3D_PYTHON:?set SAM3D_PYTHON in config.sh (sam3d-objects env python)}"
export SAM3D_REPO SAM3D_PYTHON EGOINFINITY_RUN_SAM3D=1

# sanity: gated checkpoints present?
[ -f "$SAM3D_REPO/checkpoints/hf/pipeline.yaml" ] || {
    echo "ERROR: $SAM3D_REPO/checkpoints/hf/pipeline.yaml not found."
    echo "       Download the gated weights first (see sam3d/SETUP.md)."; exit 1; }

cd "$EI_REPO"
echo "[sam3d] reconstructing object meshes (SAM 3D Objects) for $CLIP ..."
echo "[sam3d]   SAM3D_REPO=$SAM3D_REPO"
echo "[sam3d]   SAM3D_PYTHON=$SAM3D_PYTHON"

# refresh_sam3d reuses the SAM3 masks in the pkl; only the SAM3D worker loads.
"$EI_PYTHON" -m egoinfinity run refresh_sam3d "$CLIP" --with-deps

echo "[sam3d] done. meshes:"
ls -lh "$CLIP/sam3_meshes/"*.ply 2>/dev/null || echo "  (no PLYs written — check the log)"
echo "[sam3d] now ship $CLIP back to Host A and run: egoinfinity process $CLIP"
