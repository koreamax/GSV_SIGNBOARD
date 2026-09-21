#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""학습된 YOLO26x 단어(word) 탐지기 워커 — paddle_det_worker.py 와 같은 IO 계약.

  --manifest IN.jsonl : {"key": str, "path": str}
  --out      OUT.jsonl: {"key": str, "boxes": [[[x,y],...4], ...]}

가중치는 artifacts/signboard_text_holdout 으로 학습한 artifacts/yolo26x_text_holdout/run/weights/best.pt.
배포 검출기(CRAFT ∪ PaddleOCR-DB)를 이 탐지기로 갈아끼웠을 때의 파이프라인 성능을 재기 위한 워커입니다
(D38). 박스는 축정렬 사각형을 4점 폴리곤으로 내보내 뒤쪽 라인 병합이 그대로 동작합니다.

Usage:
  .venv/Scripts/python.exe str_baselines/yolo_text_det_worker.py --manifest in.jsonl --out out.jsonl --conf 0.25
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
DEFAULT_W = HERE / "artifacts" / "yolo26x_text_holdout" / "run" / "weights" / "best.pt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default=str(DEFAULT_W))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5, help="NMS IoU")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    from ultralytics import YOLO

    model = YOLO(args.weights)
    items = [json.loads(l) for l in open(args.manifest, encoding="utf-8") if l.strip()]
    n_box = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for s in range(0, len(items), args.batch):
            chunk = items[s:s + args.batch]
            res = model.predict([it["path"] for it in chunk], conf=args.conf, iou=args.iou,
                                imgsz=args.imgsz, max_det=args.max_det, device=args.device,
                                verbose=False)
            for it, r in zip(chunk, res):
                boxes = []
                for b in r.boxes.xyxy.tolist():
                    x1, y1, x2, y2 = b
                    boxes.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
                n_box += len(boxes)
                f.write(json.dumps({"key": it["key"], "boxes": boxes}) + "\n")
    print(f"[yolo_det] {len(items)} images, {n_box} boxes (conf {args.conf}) -> {args.out}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
