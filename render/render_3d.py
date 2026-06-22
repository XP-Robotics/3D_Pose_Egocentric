#!/usr/bin/env python
"""Offscreen 3D render of the EgoInfinity reconstruction (viser-like view).

Depth point cloud (RGB-colored) + MANO hand meshes, rendered from a gently
orbiting camera over the clip frames. Headless via EGL.
"""
import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import sys, gzip, pickle
import numpy as np
import cv2
import trimesh
import pyrender

PKL = sys.argv[1]
OUTDIR = sys.argv[2]
ONLY = int(sys.argv[3]) if len(sys.argv) > 3 else -1   # render single frame idx if >=0
os.makedirs(OUTDIR, exist_ok=True)

d = pickle.load(gzip.open(PKL, "rb"))
fd = d["frame_data"]
faces = np.asarray(d["mano_faces"])
# pristine frames (pkl idx i -> {i+1:06d}.jpg); pkl img_rgb has joints baked in
FRAMES_DIR = os.path.join(os.path.dirname(os.path.abspath(PKL)),
                          "extract_frames", "frames")
f = d["dp_focal"]; cx = d["cx"]; cy = d["cy"]
N = len(fd)
W, H = 720, 540
ZMIN, ZMAX = 0.3, 3.0          # clip background beyond 3 m
STEP = 2                       # depth subsample

def decode_rgb(x):
    if isinstance(x, (bytes, bytearray)):
        return cv2.imdecode(np.frombuffer(x, np.uint8), cv2.IMREAD_COLOR)[..., ::-1]
    return np.ascontiguousarray(x)

def decode_depth(x):
    if isinstance(x, (bytes, bytearray)):
        return cv2.imdecode(np.frombuffer(x, np.uint8), cv2.IMREAD_UNCHANGED)
    return np.asarray(x)

def look_at(eye, target, up=np.array([0., -1., 0.])):
    # world: X right, Y DOWN, Z forward (camera frame). up = -Y.
    fwd = target - eye; fwd /= (np.linalg.norm(fwd) + 1e-9)
    zc = -fwd                                   # GL camera looks down -Z
    xc = np.cross(up, zc); xc /= (np.linalg.norm(xc) + 1e-9)
    yc = np.cross(zc, xc)
    P = np.eye(4); P[:3, 0] = xc; P[:3, 1] = yc; P[:3, 2] = zc; P[:3, 3] = eye
    return P

# scene target: median of near foreground points across a mid frame
def foreground_centroid(fi):
    depth = decode_depth(fd[fi]["depth_png"]).astype(np.float32) / 1000.0
    m = (depth > ZMIN) & (depth < ZMAX)
    ys, xs = np.where(m)
    z = depth[ys, xs]
    X = (xs - cx) * z / f; Y = (ys - cy) * z / f
    P = np.stack([X, Y, z], 1)
    return np.median(P, 0)

target = foreground_centroid(N // 2)
R = 1.4                                          # orbit radius
renderer = pyrender.OffscreenRenderer(W, H)

def unproject(depth, rgb):
    dm = depth.astype(np.float32) / 1000.0
    m = (dm > ZMIN) & (dm < ZMAX)
    ys, xs = np.where(m)
    ys = ys[::STEP]; xs = xs[::STEP]
    z = dm[ys, xs]
    X = (xs - cx) * z / f; Y = (ys - cy) * z / f
    pts = np.stack([X, Y, z], 1)
    cols = rgb[ys, xs].astype(np.float32) / 255.0
    return pts, cols

frames = [ONLY] if ONLY >= 0 else range(N)
for fi in frames:
    fr = fd[fi]
    _clean = os.path.join(FRAMES_DIR, f"{fi + 1:06d}.jpg")
    rgb = (cv2.imread(_clean)[..., ::-1] if os.path.isfile(_clean)
           else decode_rgb(fr["img_rgb"]))
    depth = decode_depth(fr["depth_png"])
    if rgb.shape[:2] != depth.shape:
        rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]))
    pts, cols = unproject(depth, rgb)

    scene = pyrender.Scene(bg_color=[0.05, 0.05, 0.07, 1.0],
                           ambient_light=[0.4, 0.4, 0.4])
    # point cloud
    pc = pyrender.Mesh.from_points(pts, colors=cols)
    scene.add(pc)
    # hand meshes
    hand_cols = [(0.25, 0.65, 1.0), (1.0, 0.65, 0.25)]
    for hi, vv in enumerate(fr.get("vertices_3d") or []):
        vv = np.asarray(vv)
        if vv.shape[0] != 778:
            continue
        tm = trimesh.Trimesh(vertices=vv, faces=faces, process=False)
        mat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(*hand_cols[hi % 2], 1.0), metallicFactor=0.1,
            roughnessFactor=0.7)
        scene.add(pyrender.Mesh.from_trimesh(tm, material=mat, smooth=True))

    # SAM3 detected/tracked objects: highlight each mask region as a bright,
    # distinctly-colored point cloud (nudged toward the camera so it draws on
    # top of the gray scene cloud). Makes the detections visible in 3D.
    obj_cols = [(1.0, 0.25, 0.25), (0.25, 1.0, 0.35), (0.30, 0.55, 1.0),
                (1.0, 0.85, 0.20), (1.0, 0.30, 0.90), (0.20, 1.0, 1.0),
                (1.0, 0.55, 0.10)]
    dm_m = depth.astype(np.float32) / 1000.0
    od = fr.get("sam3_obj_data") or {}
    for oi, oid in enumerate(sorted(od.keys())):
        ov = od[oid]
        mp, ms = ov.get("mask_packed"), ov.get("mask_shape")
        if mp is None:
            continue
        m = np.unpackbits(np.asarray(mp, np.uint8))[:int(ms[0]) * int(ms[1])]
        m = m.reshape(int(ms[0]), int(ms[1])).astype(bool)
        if m.shape != dm_m.shape:
            m = cv2.resize(m.astype(np.uint8), (dm_m.shape[1], dm_m.shape[0]),
                           0, 0, cv2.INTER_NEAREST).astype(bool)
        sel = m & (dm_m > ZMIN) & (dm_m < ZMAX)
        ys, xs = np.where(sel)
        if len(xs) < 10:
            continue
        z = dm_m[ys, xs]
        opts = np.stack([(xs - cx) * z / f, (ys - cy) * z / f, z - 0.006], 1)
        ocol = np.tile(np.array(obj_cols[oi % len(obj_cols)], np.float32),
                       (len(opts), 1))
        scene.add(pyrender.Mesh.from_points(opts, colors=ocol))

    # gently orbiting camera (+/-22 deg over the clip)
    frac = fi / max(N - 1, 1)
    az = np.deg2rad(28 * np.sin(2 * np.pi * frac))
    eye = target + R * np.array([np.sin(az), -0.35, -np.cos(az)])
    cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(50), aspectRatio=W / H)
    scene.add(cam, pose=look_at(eye, target))
    # a couple of directional lights from the camera-ish side
    dl = pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.0)
    scene.add(dl, pose=look_at(eye, target))

    color, _ = renderer.render(scene)
    img = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    cv2.putText(img, f"XP Robotics 3D recon | frame {fi}/{N-1}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(f"{OUTDIR}/{fi:06d}.png", img)

renderer.delete()
print("rendered", len(list(frames)) if ONLY < 0 else 1, "frame(s) ->", OUTDIR)
