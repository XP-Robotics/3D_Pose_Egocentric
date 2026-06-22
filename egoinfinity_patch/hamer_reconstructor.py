"""HaMeR-based 3D hand reconstruction backend (drop-in for WiLoR).

HaMeR (Pavlakos et al., CVPR 2024) shares WiLoR's reconstruction conventions
(ViTDetDataset crop, cam_crop_to_full, FOCAL_LENGTH/IMAGE_SIZE weak-perspective,
left-hand x-flip), so this mirrors ``hand_reconstructor.HandReconstructor`` and
emits the SAME ``HandResult`` so the rest of Phase B is unchanged.

Select via: EGOINFINITY_HAND_RECON_BACKEND=hamer  (or `process --hand-recon-backend hamer`).
Repo + weights resolved from $HAMER_REPO (default <ego_root>/hamer).
"""
import os
import sys
import numpy as np
import torch
from typing import List, Optional

# ── locate the HaMeR checkout + weights ──────────────────────────────────
HAMER_DIR = os.environ.get(
    "HAMER_REPO", os.path.join(os.path.expanduser("~"), "hamer"))
if HAMER_DIR not in sys.path:
    sys.path.insert(0, HAMER_DIR)

# HaMeR configs resolve MANO paths relative to CACHE_DIR_HAMER ("./_DATA");
# pin it absolute so it works regardless of cwd.
import hamer.configs as _hcfg          # noqa: E402
_hcfg.CACHE_DIR_HAMER = os.path.join(HAMER_DIR, "_DATA")

from hamer.models import load_hamer                       # noqa: E402
from hamer.datasets.vitdet_dataset import ViTDetDataset   # noqa: E402
from hamer.utils import recursive_to                      # noqa: E402
from hamer.utils.renderer import cam_crop_to_full         # noqa: E402

from .hand_reconstructor import HandResult                # noqa: E402  (reuse dataclass)
from .hand_detector import HandDetection                  # noqa: E402

HAMER_CKPT = os.path.join(
    HAMER_DIR, "_DATA", "hamer_ckpts", "checkpoints", "hamer.ckpt")
RESCALE_FACTOR = float(os.environ.get("HAMER_RESCALE_FACTOR", "2.0"))


class HaMeRReconstructor:
    """HaMeR single-frame hand reconstruction (WiLoR-compatible output)."""

    def __init__(self, checkpoint: str = HAMER_CKPT, device: str = "cuda",
                 **_ignore):
        self.device = torch.device(device)
        self.model, self.model_cfg = load_hamer(checkpoint)
        self.model = self.model.to(self.device).eval()
        # HaMeR runs fp32 (its demo does); keep fp32 for numerical stability.
        self.cfg_focal = self.model_cfg.EXTRA.FOCAL_LENGTH      # 5000
        self.img_size_cfg = self.model_cfg.MODEL.IMAGE_SIZE     # 224

    @torch.no_grad()
    def reconstruct(self, img_bgr: np.ndarray,
                    detections: List[HandDetection],
                    focal_length: Optional[float] = None,
                    ) -> List[HandResult]:
        if len(detections) == 0:
            return []

        H, W = img_bgr.shape[:2]
        boxes = np.array([d.bbox for d in detections])
        right = np.array([d.is_right for d in detections])

        dataset = ViTDetDataset(
            self.model_cfg, img_bgr, boxes, right,
            rescale_factor=RESCALE_FACTOR)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=len(detections), shuffle=False, num_workers=0)

        results: List[HandResult] = []
        for batch in loader:
            batch = recursive_to(batch, self.device)
            out = self.model(batch)

            multiplier = (2 * batch["right"] - 1)
            pred_cam = out["pred_cam"].float()
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]

            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()

            scaled_focal_render = (self.cfg_focal / self.img_size_cfg
                                   * img_size.max(dim=1).values)
            cam_t_render = cam_crop_to_full(
                pred_cam, box_center, box_size, img_size,
                scaled_focal_render).detach().cpu().numpy()

            if focal_length is not None:
                focal_tensor = torch.tensor(
                    focal_length, device=self.device, dtype=torch.float32)
                cam_t_metric = cam_crop_to_full(
                    pred_cam, box_center, box_size, img_size,
                    focal_tensor).detach().cpu().numpy()
            else:
                cam_t_metric = cam_t_render

            for n in range(len(detections)):
                det = detections[n]
                is_right_n = batch["right"][n].item()

                joints_3d = out["pred_keypoints_3d"][n].detach().float().cpu().numpy()
                vertices = out["pred_vertices"][n].detach().float().cpu().numpy()

                joints_3d[:, 0] = (2 * is_right_n - 1) * joints_3d[:, 0]
                vertices[:, 0] = (2 * is_right_n - 1) * vertices[:, 0]
                joints_3d_rel = joints_3d.copy()

                cam_t_m = cam_t_metric[n]
                joints_3d_cam = joints_3d + cam_t_m
                vertices_cam = vertices + cam_t_m

                cam_t_r = cam_t_render[n]
                scaled_f = float(scaled_focal_render[n].cpu())
                cx, cy = W / 2.0, H / 2.0
                pts = joints_3d_rel + cam_t_r
                joints_2d = np.zeros((21, 2))
                joints_2d[:, 0] = pts[:, 0] / pts[:, 2] * scaled_f + cx
                joints_2d[:, 1] = pts[:, 1] / pts[:, 2] * scaled_f + cy

                effective_focal = (float(focal_length)
                                   if focal_length is not None else scaled_f)

                mano_params = out["pred_mano_params"]
                global_orient = mano_params["global_orient"][n].detach().float().cpu().numpy()
                hand_pose = mano_params["hand_pose"][n].detach().float().cpu().numpy()
                betas = mano_params["betas"][n].detach().float().cpu().numpy()

                results.append(HandResult(
                    global_orient=global_orient,
                    hand_pose=hand_pose,
                    betas=betas,
                    joints_3d=joints_3d_cam,
                    joints_3d_rel=joints_3d_rel,
                    vertices=vertices_cam,
                    cam_t=cam_t_m,
                    joints_2d=joints_2d,
                    scaled_focal=effective_focal,
                    is_right=bool(is_right_n > 0.5),
                    bbox=det.bbox.copy(),
                    confidence=det.confidence,
                    track_id=det.track_id,
                ))
        return results
