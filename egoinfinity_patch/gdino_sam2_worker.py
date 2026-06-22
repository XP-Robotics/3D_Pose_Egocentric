#!/usr/bin/env python
"""GroundingDINO + SAM2 drop-in replacement for scripts/sam3_worker.py.

Speaks the IDENTICAL line-delimited-JSON Unix-socket protocol that
``egoinfinity.pipeline.sam3_client.run_sam3_detect`` expects, so the pipeline
talks to it with ZERO code changes — it just sees a "SAM3 worker" on the
socket.  Use this when SAM3.1 weights are unavailable (gated).

  Detector : IDEA-Research/grounding-dino-base  (open-vocab, ungated HF)
  Box->mask: vendored SAM2 via egoinfinity.pipeline.object_tracker.ObjectTracker

Protocol (same as sam3_worker.py):
  Request  : {"image","prompts":[...],"out_dir","min_score","max_per_prompt"}
  Response : {"type":"done","result_file":".../result.json"}  | {"type":"error","msg"}
  result.json: {"image_path","image_size":[W,H],"min_score",
                "per_prompt":[{"prompt","n_raw","n_kept",
                               "results":[{"mask_file","box":[x1,y1,x2,y2],
                                           "score","area"}]}]}
  Plus a "<socket>.ready" status file on startup.

Run:
  EGOINFINITY env active, then:
  python scripts/gdino_sam2_worker.py --socket /tmp/egoinfinity_sam3_$USER.sock
"""
import argparse
import json
import os
import signal
import socket
import sys
import time
import traceback

import numpy as np
import torch
from PIL import Image

GDINO_MODEL = os.environ.get("GDINO_MODEL", "IDEA-Research/grounding-dino-base")
BOX_THR = float(os.environ.get("GDINO_BOX_THRESHOLD", "0.30"))
TEXT_THR = float(os.environ.get("GDINO_TEXT_THRESHOLD", "0.25"))


def _log(msg: str) -> None:
    print(f"[gdino_worker] {msg}", file=sys.stderr, flush=True)


def _recv_line(conn: socket.socket, max_bytes: int = 1 << 20) -> str:
    buf = bytearray()
    while len(buf) < max_bytes:
        chunk = conn.recv(4096)
        if not chunk:
            break
        buf.extend(chunk)
        if b"\n" in chunk:
            break
    return buf.decode("utf-8", errors="replace").split("\n", 1)[0]


def _send_line(conn: socket.socket, obj: dict) -> None:
    conn.sendall((json.dumps(obj) + "\n").encode("utf-8"))


def _detect_boxes(model, processor, image, prompt, device):
    """GroundingDINO open-vocab detection for ONE prompt -> (boxes_xyxy, scores)."""
    text = prompt.strip().lower()
    if not text.endswith("."):
        text += "."
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        outputs = model(**inputs)
    W, H = image.size
    res = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        threshold=BOX_THR, text_threshold=TEXT_THR,
        target_sizes=[(H, W)])[0]
    boxes = res["boxes"].detach().cpu().numpy()         # xyxy in pixels
    scores = res["scores"].detach().cpu().numpy()
    return boxes, scores


def _box_to_mask(tracker, frame_rgb, box):
    """Independent SAM2 segmentation of one box.

    load_first_frame() rebuilds a fresh condition_state each call (see
    sam2_camera_predictor._init_state), so each box is segmented on a clean
    slate — no reset_state() needed (and calling it before the first
    load_first_frame would KeyError on point_inputs_per_obj).
    """
    pred = tracker.predictor
    pred.load_first_frame(frame_rgb)
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    bbox_arr = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
    _, _, masks = pred.add_new_prompt(frame_idx=0, obj_id=1, bbox=bbox_arr)
    m = (masks[0] > 0.0).squeeze().detach().cpu().numpy().astype(bool)
    return m


def _handle_request(model, processor, tracker, device, req: dict) -> dict:
    image_path = req["image"]
    prompts = [s.strip() for s in req.get("prompts", []) if s and s.strip()]
    out_dir = req["out_dir"]
    min_score = float(req.get("min_score", 0.4))
    max_per_prompt = int(req.get("max_per_prompt", 10))
    if not prompts:
        return {"type": "error", "msg": "no valid prompts"}
    if not os.path.isfile(image_path):
        return {"type": "error", "msg": f"image not found: {image_path}"}
    os.makedirs(out_dir, exist_ok=True)

    # GroundingDINO scores run lower than SAM3's; honor the LOWER of the
    # request's min_score and our own box threshold so we don't over-filter.
    eff_thr = min(min_score, BOX_THR)

    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    frame_rgb = np.asarray(img)

    per_prompt = []
    t_det = time.time()
    for pi, prompt in enumerate(prompts):
        t_p = time.time()
        boxes, scores = _detect_boxes(model, processor, img, prompt, device)
        order = np.argsort(-scores)
        kept = []
        for rank, idx in enumerate(order):
            if float(scores[idx]) < eff_thr:
                continue
            if len(kept) >= max_per_prompt:
                break
            box = [float(v) for v in boxes[idx].tolist()]
            try:
                mask = _box_to_mask(tracker, frame_rgb, box)
            except Exception as e:
                _log(f"  box->mask failed ({e}); skipping det")
                continue
            if mask is None or not mask.any():
                continue
            mask_file = f"masks_{pi}_{len(kept)}.npz"
            np.savez_compressed(os.path.join(out_dir, mask_file), mask=mask)
            kept.append({
                "mask_file": mask_file,
                "box": box,
                "score": float(scores[idx]),
                "area": int(mask.sum()),
            })
        per_prompt.append({
            "prompt": prompt, "n_raw": int(len(scores)),
            "n_kept": len(kept), "results": kept,
        })
        _log(f"  [{pi}] '{prompt}' -> {len(scores)} raw, {len(kept)} kept "
             f"({(time.time()-t_p)*1000:.0f}ms)")

    result = {
        "image_path": image_path, "image_size": [W, H],
        "min_score": min_score, "per_prompt": per_prompt,
        "timings_ms": {"detect_total": int((time.time() - t_det) * 1000)},
        "detector": "groundingdino+sam2",
    }
    result_file = os.path.join(out_dir, "result.json")
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)
    return {"type": "done", "result_file": result_file}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", required=True)
    args = ap.parse_args()

    try:
        if os.path.exists(args.socket):
            os.unlink(args.socket)
    except Exception:
        pass

    device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.time()
    _log(f"loading GroundingDINO ({GDINO_MODEL}) ...")
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    processor = AutoProcessor.from_pretrained(GDINO_MODEL)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_MODEL).to(device).eval()
    _log(f"GroundingDINO loaded ({time.time()-t0:.1f}s); loading SAM2 ...")

    # ObjectTracker builds the vendored SAM2 camera predictor (and enters a
    # global bf16 autocast). Build it AFTER GroundingDINO; detector inference
    # explicitly disables autocast (see _detect_boxes).
    from egoinfinity.pipeline.object_tracker import ObjectTracker
    tracker = ObjectTracker(device=device)
    load_s = time.time() - t0
    _log(f"models ready ({load_s:.1f}s)")

    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(args.socket)
    os.chmod(args.socket, 0o600)
    srv.listen(4)
    with open(args.socket + ".ready", "w") as f:
        json.dump({"pid": os.getpid(), "version": "groundingdino+sam2",
                   "load_time_s": round(load_s, 1)}, f)
    _log(f"listening on {args.socket} (pid {os.getpid()})")

    stop = {"flag": False}
    def _sig(_s, _f):
        _log(f"signal {_s}, shutting down")
        stop["flag"] = True
        try: srv.close()
        except Exception: pass
    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    while not stop["flag"]:
        try:
            conn, _ = srv.accept()
        except OSError:
            break
        try:
            line = _recv_line(conn)
            if not line:
                continue
            try:
                req = json.loads(line)
            except Exception as e:
                _send_line(conn, {"type": "error", "msg": f"bad json: {e}"})
                continue
            try:
                resp = _handle_request(model, processor, tracker, device, req)
            except Exception:
                traceback.print_exc(file=sys.stderr)
                resp = {"type": "error", "msg": traceback.format_exc().splitlines()[-1]}
            try:
                _send_line(conn, resp)
            except (BrokenPipeError, ConnectionResetError, OSError) as e:
                _log(f"  client disconnected before response ({e})")
        finally:
            try: conn.close()
            except Exception: pass

    for suffix in ("", ".ready"):
        try: os.unlink(args.socket + suffix)
        except Exception: pass
    _log("exited cleanly")


if __name__ == "__main__":
    main()
