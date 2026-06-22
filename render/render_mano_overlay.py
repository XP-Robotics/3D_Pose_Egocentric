#!/usr/bin/env python
"""Overlay the reconstructed MANO hand meshes on the original input frames.

Renders the 778-vertex MANO meshes through the SAME pinhole intrinsics the
pipeline estimated (focal, cx, cy) so they project pixel-aligned onto each
RGB frame, then alpha-composites. Headless via EGL.
"""
import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import sys
import gzip
import pickle
import numpy as np
import cv2
import trimesh
import pyrender

PKL = sys.argv[1]
OUTDIR = sys.argv[2]
ONLY = int(sys.argv[3]) if len(sys.argv) > 3 else -1
ALPHA = 0.92
os.makedirs(OUTDIR, exist_ok=True)

d = pickle.load(gzip.open(PKL, "rb"))
fd = d["frame_data"]
faces = np.asarray(d["mano_faces"])
# The pkl's frame_data['img_rgb'] has the hand keypoint/skeleton viz baked in.
# Use the PRISTINE ffmpeg-extracted frames instead (pkl idx i -> 00000{i+1}.jpg).
FRAMES_DIR = os.path.join(os.path.dirname(os.path.abspath(PKL)),
                          "extract_frames", "frames")
fx = fy = float(d["dp_focal"])
cx = float(d["cx"]); cy = float(d["cy"])
N = len(fd)

# OpenCV camera frame -> pyrender/OpenGL camera pose (look down +Z, Y down)
CAM_POSE = np.array([[1, 0, 0, 0],
                     [0, -1, 0, 0],
                     [0, 0, -1, 0],
                     [0, 0, 0, 1]], dtype=float)
# white meshes for both hands (shading from the light still gives 3D form)
COLORS = {True: (1.0, 1.0, 1.0), False: (1.0, 1.0, 1.0)}


def decode_rgb(x):
    if isinstance(x, (bytes, bytearray)):
        return cv2.imdecode(np.frombuffer(x, np.uint8), cv2.IMREAD_COLOR)[..., ::-1]
    return np.ascontiguousarray(x)


renderer = None
frames = [ONLY] if ONLY >= 0 else range(N)
for fi in frames:
    fr = fd[fi]
    clean = os.path.join(FRAMES_DIR, f"{fi + 1:06d}.jpg")
    if os.path.isfile(clean):
        rgb = cv2.imread(clean)[..., ::-1]            # pristine frame, no joints
    else:
        rgb = decode_rgb(fr["img_rgb"])               # fallback (has baked-in viz)
    H, W = rgb.shape[:2]
    if renderer is None:
        renderer = pyrender.OffscreenRenderer(W, H)

    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.5, 0.5, 0.5])
    verts_list = fr.get("vertices_3d") or []
    isr = fr.get("hand_is_right") or []
    n_hand = 0
    for hi, vv in enumerate(verts_list):
        vv = np.asarray(vv)
        if vv.shape[0] != 778:
            continue
        right = bool(isr[hi]) if hi < len(isr) else True
        tm = trimesh.Trimesh(vertices=vv, faces=faces, process=False)
        mat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(*COLORS[right], 1.0),
            metallicFactor=0.1, roughnessFactor=0.6)
        scene.add(pyrender.Mesh.from_trimesh(tm, material=mat, smooth=True))
        n_hand += 1

    out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    if n_hand:
        cam = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.05, zfar=20.0)
        scene.add(cam, pose=CAM_POSE)
        scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=4.0), pose=CAM_POSE)
        color, depth = renderer.render(scene)        # color RGB, depth (H,W)
        mask = depth > 0
        col_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        out[mask] = (ALPHA * col_bgr[mask] + (1 - ALPHA) * out[mask]).astype(np.uint8)

    cv2.putText(out, f"XP Robotics  |  MANO hand mesh  |  frame {fi}/{N-1}",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(f"{OUTDIR}/{fi:06d}.png", out)

if renderer:
    renderer.delete()
print("rendered", 1 if ONLY >= 0 else N, "frame(s) ->", OUTDIR)
