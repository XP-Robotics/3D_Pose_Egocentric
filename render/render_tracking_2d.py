#!/usr/bin/env python
"""2D tracking overlay: hand skeletons + tracked-object masks on the input frames.

Usage: render_tracking_2d.py <pkl> <frames_dir> <out_png_dir>
  <frames_dir> = pristine extracted frames (pkl idx i -> {i+1:06d}.jpg).
Writes PNG frames; the pipeline encodes them to mp4.
"""
import sys
import os
import gzip
import pickle
import numpy as np
import cv2

PKL, FRAMES_DIR, OUTDIR = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUTDIR, exist_ok=True)
d = pickle.load(gzip.open(PKL, "rb"))
fd = d["frame_data"]
mapping = d.get("sam3_prompt_mapping", [])

# MANO 21-keypoint skeleton edges (wrist=0 + 5 fingers x4)
EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
         (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
         (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
COLS = [(0, 90, 255), (0, 220, 0), (255, 100, 0), (0, 210, 255),
        (255, 0, 200), (170, 0, 255)]


def clean_rgb(fi, fallback):
    p = os.path.join(FRAMES_DIR, f"{fi + 1:06d}.jpg")
    if os.path.isfile(p):
        return cv2.imread(p)
    x = fallback
    if isinstance(x, (bytes, bytearray)):
        return cv2.imdecode(np.frombuffer(x, np.uint8), cv2.IMREAD_COLOR)
    return np.ascontiguousarray(x[..., ::-1])


for fi, f in enumerate(fd):
    ov = clean_rgb(fi, f["img_rgb"])
    H, W = ov.shape[:2]
    # objects
    od = f.get("sam3_obj_data") or {}
    for i, oid in enumerate(sorted(od.keys())):
        v = od[oid]
        mp, ms = v.get("mask_packed"), v.get("mask_shape")
        if mp is None:
            continue
        m = np.unpackbits(np.asarray(mp, np.uint8))[:int(ms[0]) * int(ms[1])]
        m = m.reshape(int(ms[0]), int(ms[1])).astype(bool)
        if m.shape != (H, W):
            m = cv2.resize(m.astype(np.uint8), (W, H), 0, 0,
                           cv2.INTER_NEAREST).astype(bool)
        if not m.any():
            continue
        c = COLS[i % len(COLS)]
        ov[m] = (0.4 * np.array(c) + 0.6 * ov[m]).astype(np.uint8)
        ys, xs = np.where(m)
        lbl = mapping[oid]["prompt"] if oid < len(mapping) else str(oid)
        cv2.rectangle(ov, (xs.min(), ys.min()), (xs.max(), ys.max()), c, 1)
        cv2.putText(ov, lbl, (xs.min(), max(11, ys.min() - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1, cv2.LINE_AA)
    # hands
    j2 = f.get("joints_2d_pred") or []
    isr = f.get("hand_is_right") or []
    for hi, jj in enumerate(j2):
        jj = np.asarray(jj)
        if jj.ndim != 2 or jj.shape[0] < 21:
            continue
        hc = (60, 200, 255) if (hi < len(isr) and isr[hi]) else (255, 200, 60)
        for a, b in EDGES:
            cv2.line(ov, tuple(jj[a, :2].astype(int)),
                     tuple(jj[b, :2].astype(int)), hc, 1, cv2.LINE_AA)
        for x, y in jj[:, :2]:
            cv2.circle(ov, (int(x), int(y)), 2, (255, 255, 255), -1)
    cv2.putText(ov, f"XP Robotics  |  tracking  |  frame {fi}/{len(fd) - 1}",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.imwrite(f"{OUTDIR}/{fi:06d}.png", ov)

print("rendered", len(fd), "frames ->", OUTDIR)
