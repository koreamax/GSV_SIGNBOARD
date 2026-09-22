#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EasyOCR(CRNN) 라인 인식 워커 — paddle_rec_worker 와 같은 IO 계약.

  --manifest IN.jsonl : {"key": str, "path": str}   key 는 "<region>::<crop>::<line>"
  --out      OUT.jsonl: {"key": str, "text": str, "score": float}

검출은 하지 않습니다. 라인 스트립 1장 = 한 줄이므로 EasyOCR 의 인식기만 호출합니다
(recognize(horizontal_list=None) = 이미지 전체를 한 줄로 읽기). 그래야 다른 인식기와
같은 조건 — 같은 검출기가 만든 같은 스트립 — 이 됩니다.

기본 모델은 signboard_v3 로 미세조정한 사용자 인식망 `signboard_v3_custom`
(~/.EasyOCR/user_network) 으로, 다른 미세조정 행과 같은 학습 데이터입니다.
"""
import argparse
import json
import sys

import numpy as np
from PIL import Image

MIN_H = 32          # CRNN 은 저해상 입력에서 급격히 나빠집니다


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--recog-network", default="signboard_v3_custom")
    ap.add_argument("--langs", default="ko,en")
    ap.add_argument("--gpu", type=int, default=1)
    args = ap.parse_args()

    import easyocr
    reader = easyocr.Reader([l for l in args.langs.split(",") if l],
                            recog_network=args.recog_network, gpu=bool(args.gpu))

    items = [json.loads(l) for l in open(args.manifest, encoding="utf-8") if l.strip()]
    with open(args.out, "w", encoding="utf-8") as f:
        for i, it in enumerate(items, 1):
            img = Image.open(it["path"]).convert("L")
            if img.height < MIN_H:
                s = MIN_H / img.height
                img = img.resize((max(1, round(img.width * s)), MIN_H), Image.BICUBIC)
            try:
                res = reader.recognize(np.array(img), horizontal_list=None, free_list=None, detail=1)
            except Exception as e:                       # 극단적으로 가는 스트립에서 간헐 실패
                print(f"[easyocr] {it['key']}: {e}", flush=True)
                res = []
            text = " ".join(str(r[1]) for r in res).strip()
            score = float(res[0][2]) if res else 0.0
            f.write(json.dumps({"key": it["key"], "text": " ".join(text.split()),
                                "score": score}, ensure_ascii=False) + "\n")
            if i % 200 == 0:
                print(f"[easyocr] {i}/{len(items)}", flush=True)
    print(f"[easyocr] done {len(items)} -> {args.out}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
