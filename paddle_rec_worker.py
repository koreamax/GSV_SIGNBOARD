#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paddle_rec_worker.py

Isolated PaddleOCR text-recognition worker.

WHY a separate process: paddlepaddle-gpu (cu118) and torch (cu118) both ship
cudnn_*_8.dll with the SAME base names but different ABIs. Windows loads a DLL
once by base name, so whichever framework imports second fails with WinError 127.
run_ocr_only.py already holds torch (TrOCR + EasyOCR), so PaddleOCR cannot live
in the same process. This worker imports ONLY paddle/paddleocr (torch is blocked
by stubbing `modelscope`, which paddlex would otherwise use to pull in torch).

IO contract (line-delimited JSON):
  --manifest IN.jsonl : each line {"key": str, "path": str}
  --out      OUT.jsonl: each line {"key": str, "text": str, "score": float}
"""
import argparse
import json
import sys
import types
from pathlib import Path


def _block_torch():
    """Stub modelscope so paddlex/paddleocr does not import torch (DLL clash)."""
    ms = types.ModuleType("modelscope")
    ms.__path__ = []  # mark as a package so submodule imports resolve to the stub
    sys.modules["modelscope"] = ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-name", default="korean_PP-OCRv5_mobile_rec")
    ap.add_argument("--model-dir", default=None,
                    help="Exported inference dir of a fine-tuned model. "
                         "Omit for zero-shot pretrained model_name.")
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    # Order matters: import paddle FIRST so its DLL dirs register, then block torch.
    import paddle  # noqa: F401
    _block_torch()
    import numpy as np
    from PIL import Image
    from paddleocr import TextRecognition

    kw = dict(model_name=args.model_name, device=args.device)
    if args.model_dir:
        kw["model_dir"] = args.model_dir
    rec = TextRecognition(**kw)

    items = []
    with open(args.manifest, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    results = {}
    bs = max(1, int(args.batch_size))
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        arrs, keys = [], []
        for it in chunk:
            try:
                arr = np.array(Image.open(it["path"]).convert("RGB"))[:, :, ::-1]  # RGB->BGR
            except Exception:
                results[it["key"]] = ("", 0.0)
                continue
            arrs.append(arr)
            keys.append(it["key"])
        if not arrs:
            continue
        try:
            preds = rec.predict(arrs)
            for k, r in zip(keys, preds):
                txt = str(r.get("rec_text", "") or "")
                sc = float(r.get("rec_score", 0.0) or 0.0)
                results[k] = (txt, sc)
        except Exception:
            for k in keys:
                results.setdefault(k, ("", 0.0))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for it in items:
            txt, sc = results.get(it["key"], ("", 0.0))
            f.write(json.dumps({"key": it["key"], "text": txt, "score": sc},
                               ensure_ascii=False) + "\n")
    sys.stderr.write(f"[paddle_worker] recognized {len(results)}/{len(items)} crops\n")


if __name__ == "__main__":
    main()
