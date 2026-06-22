#!/usr/bin/env bash
# Copy this to `config.sh` and edit the paths for your machine.
#   cp config.example.sh config.sh && $EDITOR config.sh
# `pipeline.sh` sources config.sh.

# ── Paths (EDIT THESE) ───────────────────────────────────────────────────
# Your EgoInfinity checkout (the dir containing egoinfinity/, scripts/, third_party/).
export EI_REPO="${EI_REPO:-$HOME/EgoInfinity}"
# Python interpreter of the env where EgoInfinity is installed
# (`pip install -e .` + MoGe/GeoCalib/MEMFOF/open3d; cu128 torch if Blackwell GPU).
export EI_PYTHON="${EI_PYTHON:-$HOME/egoinfinity-venv/bin/python}"
# HaMeR checkout (weights live under $HAMER_REPO/_DATA/hamer_ckpts/...).
export HAMER_REPO="${HAMER_REPO:-$HOME/hamer}"

# ── Pipeline options ─────────────────────────────────────────────────────
export HANDS="${HANDS:-hamer}"                 # hand reconstructor: hamer | wilor
export EGOINFINITY_CKPT_DIR="${EGOINFINITY_CKPT_DIR:-$EI_REPO/pretrained_models}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# 16 GB GPU: skip the SAM 3D Objects mesh worker (~10 GB, also gated).
export EGOINFINITY_RUN_SAM3D="${EGOINFINITY_RUN_SAM3D:-0}"
export EGOINFINITY_LOW_VRAM="${EGOINFINITY_LOW_VRAM:-1}"

# First run must be ONLINE to fetch MoGe-2 / DINOv2 / MEMFOF (ungated HF).
# Flip to 1 after the HF cache is warm to avoid HEAD-request latency.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"

# Headless GL for the renderers + WiLoR/HaMeR pyrender.
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

# ── Object detector ──────────────────────────────────────────────────────
# sam3  = real SAM 3.1 (best quality; needs gated weights, see below)
# gdino = GroundingDINO + SAM2 (ungated fallback; no Meta access needed)
export DETECTOR="${DETECTOR:-sam3}"

# -- SAM 3.1 (DETECTOR=sam3) --
# Separate env (numpy<2): clone https://github.com/facebookresearch/sam3 to
# $SAM3_REPO, make a py3.12 venv, `pip install -e $SAM3_REPO` (+ webdataset
# pycocotools psutil pandas). Weights are gated — download sam3.1_multiplex.pt
# from https://huggingface.co/facebook/sam3.1 and point SAM3_CKPT at it (the
# worker loads it locally, no HF access at runtime).
export SAM3_REPO="${SAM3_REPO:-$HOME/sam3}"
export SAM3_PYTHON="${SAM3_PYTHON:-$HOME/sam3-venv/bin/python}"
export SAM3_CKPT="${SAM3_CKPT:-$HOME/sam3_weights/sam3.1/sam3.1_multiplex.pt}"

# -- GroundingDINO fallback (DETECTOR=gdino) -- ungated HF weights
export GDINO_MODEL="${GDINO_MODEL:-IDEA-Research/grounding-dino-base}"

# ── SAM 3D Objects (optional, multi-host) — per-object 3D meshes ──────────
# Needs a >=32 GB GPU (Host B). See sam3d/SETUP.md. Only used by
# sam3d/run_sam3d_fill.sh on the big-GPU host; not started by pipeline.sh.
export SAM3D_REPO="${SAM3D_REPO:-$HOME/sam-3d-objects}"
export SAM3D_PYTHON="${SAM3D_PYTHON:-$HOME/miniconda3/envs/sam3d-objects/bin/python}"
