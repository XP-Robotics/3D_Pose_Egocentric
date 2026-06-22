# Wiring the patch into EgoInfinity

These two files extend an existing [EgoInfinity](https://github.com/Rice-RobotPI-Lab/EgoInfinity)
checkout. Nothing else in EgoInfinity changes.

## 1. HaMeR hand-reconstruction backend

Copy the adapter into the pipeline package:

```bash
cp egoinfinity_patch/hamer_reconstructor.py  $EI_REPO/egoinfinity/pipeline/
```

Register it in `egoinfinity/pipeline/backends/__init__.py` — add the loader and
one `register(...)` line next to the existing WiLoR one:

```python
def _hamer() -> type:
    from egoinfinity.pipeline.hamer_reconstructor import HaMeRReconstructor
    return HaMeRReconstructor

# ... alongside the other register() calls:
register("hand_recon", "hamer", _hamer)
```

Then select it per run with `--hand-recon-backend hamer`
(or `EGOINFINITY_HAND_RECON_BACKEND=hamer`). Default stays `wilor`.

**HaMeR setup:** clone https://github.com/geopavlakos/hamer to `$HAMER_REPO`,
download its weights (`hamer_demo_data.tar.gz` → `_DATA/`), `pip install webdataset`,
and drop a chumpy-free `MANO_RIGHT.pkl` into `$HAMER_REPO/_DATA/data/mano/`
(see README "MANO note"). The core model needs **no** detectron2/ViTPose.

## 2. Object detector — `DETECTOR=sam3` (default) or `gdino` (fallback)

Both bind the same Unix socket EgoInfinity's SAM3 client expects
(`/tmp/egoinfinity_sam3_$USER.sock`) and speak the identical protocol, so the
pipeline talks to whichever is running with **no code change**. `pipeline.sh`
starts the one selected by `$DETECTOR`.

### 2a. SAM 3.1 (best quality — `DETECTOR=sam3`)

SAM 3.1 runs in its **own env** (it needs `numpy<2`). Weights are gated, but the
worker can load them **locally** with no runtime HF access:

```bash
# env (next to EgoInfinity)
git clone https://github.com/facebookresearch/sam3  $SAM3_REPO
python3.12 -m venv ~/sam3-venv
$SAM3_PYTHON -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
$SAM3_PYTHON -m pip install -e $SAM3_REPO webdataset pycocotools psutil pandas

# weights: download sam3.1_multiplex.pt from https://huggingface.co/facebook/sam3.1
#   (gated — request access), put it at $SAM3_CKPT
```

One-line patch to EgoInfinity's worker so it uses the local checkpoint instead
of the gated HF download — in `$EI_REPO/scripts/sam3_worker.py`, change:

```python
# ckpt_path = download_ckpt_from_hf(version=args.version)
ckpt_path = os.environ.get("SAM3_CKPT") or download_ckpt_from_hf(version=args.version)
```

Then `DETECTOR=sam3` in `config.sh`. (Higher confidence + cleaner masks than the
GroundingDINO fallback — e.g. "black tool box" 0.95 vs 0.52.)

### 2b. GroundingDINO + SAM2 (ungated fallback — `DETECTOR=gdino`)

No Meta access needed.

```bash
cp egoinfinity_patch/gdino_sam2_worker.py  $EI_REPO/scripts/
$EI_PYTHON -m pip install transformers     # GroundingDINO weights are ungated
```

Uses **GroundingDINO** for boxes + **SAM2** for masks. Set `DETECTOR=gdino`.
