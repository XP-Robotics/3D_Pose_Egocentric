# SAM 3D Objects — per-object 3D meshes (multi-host)

Adds the **3D object meshes** (Gaussian-splat reconstructions) the EgoInfinity
paper shows — the one thing missing from the 16 GB run. SAM 3D Objects needs
**≥ 32 GB VRAM**, so it can't run on a 16 GB card. The fix is EgoInfinity's
**multi-host** design: do detection/tracking on your machine (already done),
then run *only* the mesh step on a 32 GB+ GPU, and ship the meshes back.

```
Host A (16 GB, your box)                 Host B (≥32 GB: A100 / L40S / H100)
  detection + SAM3.1 + tracking   ──ship──►  refresh_sam3d (SAM 3D Objects)
  (pipeline_result.pkl.gz)        ◄─ship──   (writes sam3_meshes/*.ply into pkl)
  resume post-tracking
```

> **GPU note:** SAM 3D Objects' env is **CUDA 12.1** (PyTorch cu121 + PyTorch3D
> + Kaolin). Use an **A100 / L40S / H100** (cu121-friendly). A 32 GB **RTX 5090**
> is Blackwell and needs cu128 — possible but you'd have to rebuild SAM3D's torch
> stack, so prefer A100/L40S/H100.

---

## Host B — one-time setup

```bash
# 0. clone both repos on Host B (next to each other)
git clone https://github.com/Rice-RobotPI-Lab/EgoInfinity.git
git clone https://github.com/facebookresearch/sam-3d-objects.git

# 1. EgoInfinity env (same as your main env) — see EgoInfinity README
#    (needs the egoinfinity package so `egoinfinity run refresh_sam3d` works)

# 2. SAM 3D Objects env + weights:
bash sam3d/setup_hostB.sh /path/to/sam-3d-objects
```

`setup_hostB.sh` builds the `sam3d-objects` conda/mamba env and downloads the
**gated** checkpoints.

### Getting the gated weights
1. Request access at **https://huggingface.co/facebook/sam-3d-objects** (agree to
   the SAM license; Meta approves manually; rejected in sanctioned regions).
2. `hf auth login` with a Read token from that account.
3. `setup_hostB.sh` runs the official download:
   ```bash
   hf download --repo-type model --local-dir checkpoints/hf-download \
       --max-workers 1 facebook/sam-3d-objects
   mv checkpoints/hf-download/checkpoints checkpoints/hf   # -> checkpoints/hf/pipeline.yaml
   ```
   (Already-downloaded? Just drop the files into `$SAM3D_REPO/checkpoints/hf/`.)

---

## Run the mesh fill (Host B)

```bash
# point config at this machine's EgoInfinity + sam3d env
cp config.example.sh config.sh && $EDITOR config.sh   # set SAM3D_REPO, SAM3D_PYTHON

# receive the clip from Host A (rsync/scp the artifact dir), then:
bash sam3d/run_sam3d_fill.sh /path/to/artifacts/<CLIP>
```

That runs `egoinfinity run refresh_sam3d <CLIP>`, which spawns **only** the SAM3D
worker (the SAM3.1 masks are already in the pkl), reconstructs each object's
mesh, and writes `<CLIP>/sam3_meshes/*.ply` + the mesh info into the pkl.

---

## Ship back + finish (Host A)

```bash
# Host A: pull the clip back (now with sam3_meshes/), then resume post-tracking
python -m egoinfinity process /path/to/artifacts/<CLIP>
# regenerate the videos (objects now render as 3D meshes)
./pipeline.sh /path/to/artifacts/<CLIP>     # or just the render step
```

The viser viewer's **"SAM3D Mesh (6DoF)"** layer will now be populated, and the
3D reconstruction video shows real object meshes instead of point clouds.

## What to ship between hosts
```
artifacts/<CLIP>/
├── pipeline_result.pkl.gz   # masks/depth/joints  (A→B, then B→A with meshes)
├── manifest.json            # objects[]            (A→B)
├── extract_frames/frames/   # init-frame images    (A→B)
└── sam3_meshes/*.ply        # produced on B        (B→A)
```
~100 MB for a few-second clip. Use rsync/scp/s3/NFS — any file copy.
