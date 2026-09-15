#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Isolated PaddleOCR text-DETECTION worker (C3 experiment, D23).

Same isolation rationale as paddle_rec_worker.py: paddle and torch ship
clashing cudnn DLLs, so PaddleOCR runs in its own process.

IO contract:
  --manifest IN.jsonl : {"key": str, "path": str} per line
  --out      OUT.jsonl: {"key": str, "boxes": [[[x,y],...4], ...]} per line
"""
import argparse
import json
import sys
import types
from pathlib import Path


def _block_torch():
    ms = types.ModuleType("modelscope")
    ms.__path__ = []
    sys.modules["modelscope"] = ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-name", default="PP-OCRv5_mobile_det")
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--thresh", type=float, default=0.3)
    ap.add_argument("--box-thresh", type=float, default=0.6)
    ap.add_argument("--unclip-ratio", type=float, default=1.5)
    args = ap.parse_args()

    import paddle  # noqa: F401  (import first so its DLL dirs register)
    _block_torch()
    import numpy as np
    from PIL import Image
    from paddleocr import TextDetection

    det = TextDetection(model_name=args.model_name, device=args.device,
                        thresh=args.thresh, box_thresh=args.box_thresh,
                        unclip_ratio=args.unclip_ratio)

    items = []
    with open(args.manifest, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fo:
        for it in items:
            boxes = []
            try:
                arr = np.array(Image.open(it["path"]).convert("RGB"))[:, :, ::-1]
                for res in det.predict(arr):
                    polys = res.get("dt_polys")
                    # dt_polys is a numpy array: `or []` would raise "truth value
                    # of an array is ambiguous", so test for None explicitly.
                    if polys is None:
                        continue
                    for poly in polys:
                        boxes.append([[float(p[0]), float(p[1])] for p in poly])
            except Exception as exc:
                sys.stderr.write(f"[det_worker] {it['key']}: {type(exc).__name__}: {exc}\n")
            fo.write(json.dumps({"key": it["key"], "boxes": boxes}, ensure_ascii=False) + "\n")
    sys.stderr.write(f"[det_worker] detected on {len(items)} images\n")


if __name__ == "__main__":
    main()
