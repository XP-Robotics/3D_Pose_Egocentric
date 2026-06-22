#!/usr/bin/env bash
# One-time SAM 3D Objects setup on a >=32 GB GPU host (Host B).
# Builds the sam3d-objects env and downloads the GATED checkpoints.
#
# Usage: bash setup_hostB.sh <path-to-sam-3d-objects-checkout>
# Prereq: mamba (or conda), and HF access to facebook/sam-3d-objects granted.
set -euo pipefail
SAM3D_REPO="${1:?usage: setup_hostB.sh <sam-3d-objects checkout>}"
cd "$SAM3D_REPO"

echo "== 1/4: create sam3d-objects env (CUDA 12.1 / PyTorch3D / Kaolin) =="
# Build on a GPU node — PyTorch3D/Kaolin need a GPU to compile.
mamba env create -f environments/default.yml || conda env create -f environments/default.yml
# shellcheck disable=SC1091
eval "$(conda shell.bash hook)"; conda activate sam3d-objects

echo "== 2/4: install sam3d-objects + deps =="
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
pip install -e '.[dev]'
pip install -e '.[p3d]'      # 2-step p3d install (pytorch3d/pytorch ordering)
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
pip install -e '.[inference]'
./patching/hydra || true     # hydra patch (facebookresearch/hydra#2863)

echo "== 3/4: authenticate HuggingFace (need access to facebook/sam-3d-objects) =="
pip install -q 'huggingface-hub[cli]<1.0'
hf auth whoami >/dev/null 2>&1 || { echo "Run: hf auth login   (token with sam-3d-objects access)"; exit 1; }

echo "== 4/4: download gated checkpoints -> checkpoints/hf/ =="
if [ ! -f checkpoints/hf/pipeline.yaml ]; then
    hf download --repo-type model --local-dir checkpoints/hf-download \
        --max-workers 1 facebook/sam-3d-objects
    mv checkpoints/hf-download/checkpoints checkpoints/hf
    rm -rf checkpoints/hf-download
fi
test -f checkpoints/hf/pipeline.yaml && echo "OK: checkpoints/hf/pipeline.yaml present"

echo ""
echo "Done. Set in your config.sh:"
echo "  export SAM3D_REPO=$SAM3D_REPO"
echo "  export SAM3D_PYTHON=\$(conda run -n sam3d-objects which python)"
