# Ego_Infinity_MANO

End-to-end pipeline that takes a **static exo-view RGB clip** and produces **3D
hand (MANO) + object reconstruction videos** — built on top of
[EgoInfinity](https://github.com/Rice-RobotPI-Lab/EgoInfinity), with two
additions that make it run without gated models:

- **HaMeR** hand-reconstruction backend (drop-in alongside EgoInfinity's WiLoR)
- a pluggable object detector — **SAM 3.1** (default, best quality, loaded from a
  local checkpoint) or a **GroundingDINO + SAM2** fallback that needs no gated
  access — selected with one `DETECTOR=sam3|gdino` switch

One command (`pipeline.sh`) goes from a clip to the three videos below.

---

## Outputs

> The `<video>` players below play the actual MP4s on GitHub once this repo is
> pushed to `XP-Robotics/3D_Pose_Egocentric` (branch `main`). The animated GIF
> under each one renders inline everywhere (GitHub + local Markdown preview), so
> something always shows.

### 1. MANO hand-mesh overlay
HaMeR meshes projected onto the input video.

<video src="https://github.com/XP-Robotics/3D_Pose_Egocentric/raw/main/outputs/mano_overlay_hamer.mp4" controls muted loop width="640"></video>

![MANO overlay](outputs/gif/mano_overlay.gif)

### 2. 3D reconstruction
Metric depth point cloud + MANO hand meshes, orbiting camera.

<video src="https://github.com/XP-Robotics/3D_Pose_Egocentric/raw/main/outputs/reconstruction_3d.mp4" controls muted loop width="640"></video>

![3D reconstruction](outputs/gif/reconstruction_3d.gif)

### 3. 2D tracking
21-joint hand skeletons + open-vocab object masks.

<video src="https://github.com/XP-Robotics/3D_Pose_Egocentric/raw/main/outputs/tracking_2d.mp4" controls muted loop width="640"></video>

![2D tracking](outputs/gif/tracking_2d.gif)

---

## What `pipeline.sh` does

```
clip (manifest.json + video)
   │  extract frames
   ▼
EgoInfinity  ──  MoGe-2 depth · GeoCalib gravity · YOLO hand detect
   │             HaMeR hand reconstruction (MANO)
   │             GroundingDINO→SAM2 object detection + tracking
   │             MEMFOF flow · infiller · biomech · post-tracking
   ▼
pipeline_result.pkl.gz
   │  render (this repo)
   ▼
mano_overlay.mp4 · reconstruction_3d.mp4 · tracking_2d.mp4
```

---

## Quick start

**Prereqs** (one-time): a working EgoInfinity install + a HaMeR checkout.
See [`egoinfinity_patch/REGISTER.md`](egoinfinity_patch/REGISTER.md) for wiring
the two patch files in, and the **MANO note** below.

```bash
# 1. point the config at your machine
cp config.example.sh config.sh
$EDITOR config.sh            # set EI_REPO, EI_PYTHON, HAMER_REPO

# 2. run on a clip directory (contains manifest.json)
./pipeline.sh /path/to/clip_dir
#   -> /path/to/clip_dir/outputs/{mano_overlay,reconstruction_3d,tracking_2d}.mp4
```

`manifest.json` (see [`manifest.example.json`](manifest.example.json)):

```json
{ "video_uri": "/abs/path/video.mov",
  "objects": ["black tool box", "screwdriver", "pliers"],
  "fps": 15 }
```

`objects` are the text prompts the open-vocab detector grounds on — use clean,
concrete nouns.

Optional 3D viewer (viser, in browser): `./scripts/open_viewer.sh /path/to/clip_dir`

---

## Contents

```
Ego_Infinity_MANO/
├── pipeline.sh                 ★ end-to-end entry point
├── config.example.sh          paths + options (copy to config.sh, edit)
├── manifest.example.json
├── egoinfinity_patch/          drop-ins for an EgoInfinity checkout
│   ├── hamer_reconstructor.py    HaMeR backend (-> egoinfinity/pipeline/)
│   ├── gdino_sam2_worker.py       GroundingDINO+SAM2 detector (-> scripts/)
│   └── REGISTER.md                how to wire them in
├── render/
│   ├── render_mano_overlay.py     MANO meshes on the input video
│   ├── render_3d.py               point cloud + meshes, orbiting camera
│   └── render_tracking_2d.py      skeletons + object masks
├── scripts/
│   └── open_viewer.sh             launch the viser 3D viewer in Chrome
├── outputs/                    sample result videos (+ gif/ previews)
├── NOTICE.md                   attribution & licenses
└── .gitignore
```

---

## Why HaMeR and the open detector

- **HaMeR backend** mirrors EgoInfinity's `HandReconstructor` output contract
  exactly (same crop / `cam_crop_to_full` / focal conventions), so it's a true
  drop-in. Select with `--hand-recon-backend hamer`; default stays WiLoR.
  *(On small/far hands the two are close — the accuracy ceiling is input
  resolution, not the model.)*
- **Object detector is pluggable** (`DETECTOR` in `config.sh`); both speak
  EgoInfinity's exact SAM3 Unix-socket protocol, so the pipeline is unchanged:
  - `sam3` (default) — real **SAM 3.1**, loaded from a **local checkpoint** (no
    runtime HF access). Best quality — e.g. *"black tool box"* @ **0.95**.
  - `gdino` — **GroundingDINO + SAM2**, **ungated** weights, for when you don't
    have SAM 3.1 access. Lower confidence (*~0.5*) but works out of the box.

## MANO note

MANO is non-commercial and **not redistributed** here. Register at
https://mano.is.tue.mpg.de, get `MANO_RIGHT.pkl`, and place a **chumpy-free**
copy at `$HAMER_REPO/_DATA/data/mano/MANO_RIGHT.pkl` (modern MANO `.pkl`s embed
chumpy arrays that break under NumPy ≥ 2 — load once in a NumPy 1.23 env, convert
each value to plain `np.array`, re-pickle).

## Notes / hardware

- Designed around a single 16 GB+ NVIDIA GPU. On 16 GB the SAM-3D-Objects mesh
  (Phase D-sam3d) is skipped (`EGOINFINITY_RUN_SAM3D=0`) — you still get depth,
  hands, and object mask/OBB tracking. To add the **per-object 3D meshes** (the
  paper's object reconstructions), run the mesh step on a **≥32 GB GPU** —
  see [`sam3d/SETUP.md`](sam3d/SETUP.md) (multi-host: detect here, mesh there,
  ship the `.ply`s back).
- **Blackwell GPUs (RTX 50xx)** need CUDA 12.8 PyTorch (`--index-url .../cu128`).
- First run must be **online** (fetches MoGe-2 / DINOv2 / MEMFOF, all ungated).

See [`NOTICE.md`](NOTICE.md) for licenses. Original code here is MIT.
