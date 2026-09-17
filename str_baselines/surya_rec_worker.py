#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Surya OCR(0.14.x, 인식 모델 surya_rec2) 라인 인식 워커 — paddle_rec_worker 와 같은 IO 계약.

실행 환경: .venv_surya (CPU torch 2.14 + transformers 4.57; 메인 .venv 와 버전 충돌이라 분리).
스트립 1장을 이미지 전체 bbox 하나로 인식시킵니다(검출은 우리 파이프라인의 CRAFT∪DB 결과를 사용).
  --manifest / --out : {"key","path"} → {"key","text","score"}
"""
import argparse
import json
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    from PIL import Image
    from surya.recognition import RecognitionPredictor
    rec = RecognitionPredictor()

    items = [json.loads(l) for l in open(args.manifest, encoding="utf-8") if l.strip()]
    with open(args.out, "w", encoding="utf-8") as f:
        for s in range(0, len(items), args.batch_size):
            chunk = items[s:s + args.batch_size]
            imgs = [Image.open(it["path"]).convert("RGB") for it in chunk]
            bboxes = [[[0, 0, im.width, im.height]] for im in imgs]
            res = rec(imgs, bboxes=bboxes, recognition_batch_size=args.batch_size)
            for it, r in zip(chunk, res):
                lines = getattr(r, "text_lines", []) or []
                txt = " ".join(tl.text.strip() for tl in lines if tl.text and tl.text.strip())
                sc = float(sum(tl.confidence for tl in lines) / len(lines)) if lines else 0.0
                f.write(json.dumps({"key": it["key"], "text": " ".join(txt.split()), "score": sc},
                                   ensure_ascii=False) + "\n")
            if (s // args.batch_size) % 10 == 0:
                print(f"[surya] {min(s + args.batch_size, len(items))}/{len(items)}", flush=True)
    print(f"[surya] done {len(items)} -> {args.out}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
