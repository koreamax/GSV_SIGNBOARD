#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OpenOCR(SVTRv2 등) 라인 인식 워커 — paddle_rec_worker 와 같은 IO 계약.

  --config  OpenOCR yml (예: external/OpenOCR/configs/rec/svtrv2/svtrv2_rctc_signboard.yml)
  --weights 학습된 best.pth (Global.pretrained_model 을 이걸로 덮어씀)
  --manifest / --out : {"key","path"} → {"key","text","score"}
실행: .venv (torch cu118). 반드시 PYTHONUTF8=1 로 실행(OpenOCR 가 yml/dict 를 로케일 인코딩으로 읽음).
"""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
OPENOCR = HERE / "external" / "OpenOCR"
sys.path.insert(0, str(OPENOCR))
os.chdir(OPENOCR)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    import numpy as np
    from PIL import Image
    from tools.engine.config import Config
    from tools.infer_rec import OpenRecognizer

    cfg = Config(str(Path(args.config).resolve())).cfg
    cfg["Global"]["pretrained_model"] = str(Path(args.weights).resolve())
    cfg["Global"]["checkpoints"] = None
    rec = OpenRecognizer(config=cfg, backend="torch", use_gpu="true")

    items = [json.loads(l) for l in open(args.manifest, encoding="utf-8") if l.strip()]
    with open(args.out, "w", encoding="utf-8") as f:
        for s in range(0, len(items), args.batch_size):
            chunk = items[s:s + args.batch_size]
            # OpenRecognizer 의 img_numpy_list 경로는 디코드 op 만 건너뛰고 이후 op(RatioRecTVReisze 등)가
            # PIL 이미지(img.size)를 기대하므로 PIL 객체를 그대로 넘긴다.
            imgs = [Image.open(it["path"]).convert("RGB") for it in chunk]
            # batch_num=1: OpenRecognizer 는 배치 내 폭을 0 패딩으로 맞추는데, 학습(RatioSampler)은 같은 비율끼리만
            # 묶어 패딩이 없었음 → 패딩 영역이 CTC 끝 글자 중복('CAFE'→'CAFEE')을 만들어 1장씩 추론한다.
            res = []
            for im in imgs:
                res += rec(img_numpy_list=[im], batch_num=1)
            for it, r in zip(chunk, res):
                f.write(json.dumps({"key": it["key"], "text": str(r.get("text", "")),
                                    "score": float(r.get("score", 0.0))}, ensure_ascii=False) + "\n")
            if (s // args.batch_size) % 20 == 0:
                print(f"[openocr] {min(s + args.batch_size, len(items))}/{len(items)}", flush=True)
    print(f"[openocr] done {len(items)} -> {args.out}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
