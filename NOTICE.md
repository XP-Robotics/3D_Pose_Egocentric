# Attribution & licenses

This repo contains **only original integration/render code** plus rendered
result media. It does **not** redistribute any third-party model weights or
non-redistributable assets. To run it you fetch those yourself, under their own
licenses:

| Component | Where it comes from | License / terms |
|---|---|---|
| **EgoInfinity** (the pipeline this extends) | https://github.com/Rice-RobotPI-Lab/EgoInfinity | MIT (own code) |
| **HaMeR** (hand reconstruction, this repo's adapter targets it) | https://github.com/geopavlakos/hamer | see upstream |
| **WiLoR** (default hand backend in EgoInfinity) | upstream, fetched at install | CC-BY-NC-ND 4.0 — research only, **not redistributed here** |
| **MANO** (hand model) | https://mano.is.tue.mpg.de (register) | non-commercial research, **not redistributed here** |
| **GroundingDINO** (open-vocab detector) | IDEA-Research / HuggingFace | Apache-2.0 (ungated) |
| **SAM2** (mask tracking) | Meta | Apache-2.0 |
| **MoGe-2 / GeoCalib / MEMFOF** | upstream HF | each upstream's license |

**Not included on purpose:** `MANO_*.pkl`, WiLoR/HaMeR weights, the EgoInfinity
source tree, conda/venvs, per-clip `.pkl.gz` artifacts. See `.gitignore`.

The sample result videos in `outputs/` are derived from a capture session owned
by the repo author; they depict an identifiable person — remove them if that
isn't intended for public hosting.

Original code in this repo (pipeline.sh, render/*, the HaMeR adapter, the
GroundingDINO+SAM2 worker): **MIT**.
