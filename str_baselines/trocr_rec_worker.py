#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TrOCR 라인 인식 워커 — paddle_rec_worker 와 같은 IO 계약.

  --manifest IN.jsonl : {"key": str, "path": str}   key 는 "<region>::<crop>::<line>"
  --out      OUT.jsonl: {"key": str, "text": str, "score": float}

기본 체크포인트는 signboard_v3 로 미세조정한 `artifacts/ocr_training/signboard_v3/trocr_model`
로, 다른 미세조정 행과 같은 학습 데이터입니다(signboard_full 로 더 오래 학습한 별도
체크포인트가 있지만 학습 데이터가 달라 통제 비교에 쓰면 안 됩니다).
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
DEFAULT_M = HERE / "artifacts" / "ocr_training" / "signboard_v3" / "trocr_model"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=str(DEFAULT_M))
    ap.add_argument("--processor", default="")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    src = args.processor or args.model
    try:
        proc = TrOCRProcessor.from_pretrained(src)
    except Exception as e:
        print(f"[trocr] processor from {src} failed ({e}); falling back to microsoft/trocr-small-stage1")
        proc = TrOCRProcessor.from_pretrained("microsoft/trocr-small-stage1")
    model = VisionEncoderDecoderModel.from_pretrained(args.model).to(args.device).eval()

    items = [json.loads(l) for l in open(args.manifest, encoding="utf-8") if l.strip()]
    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(0, len(items), args.batch):
            chunk = items[i:i + args.batch]
            imgs = [Image.open(it["path"]).convert("RGB") for it in chunk]
            px = proc(images=imgs, return_tensors="pt").pixel_values.to(args.device)
            with torch.inference_mode():
                out = model.generate(px, max_new_tokens=args.max_new_tokens,
                                     output_scores=True, return_dict_in_generate=True)
            texts = proc.batch_decode(out.sequences, skip_special_tokens=True)
            try:                                      # 평균 토큰 로그확률 → [0,1] 신뢰도
                tr = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
                confs = [float(torch.exp(row[torch.isfinite(row)].mean())) if torch.isfinite(row).any() else 0.0
                         for row in tr]
            except Exception:
                confs = [1.0] * len(chunk)
            for it, tx, sc in zip(chunk, texts, confs):
                f.write(json.dumps({"key": it["key"], "text": " ".join(str(tx).split()),
                                    "score": sc}, ensure_ascii=False) + "\n")
            if (i // args.batch) % 25 == 0:
                print(f"[trocr] {min(i + args.batch, len(items))}/{len(items)}", flush=True)
    print(f"[trocr] done {len(items)} -> {args.out}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
