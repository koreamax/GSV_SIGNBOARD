#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""텍스트(word) 박스 탐지 — hold-out test 통합 평가 (4모델, 단일 AP@0.5 구현).

signboard_v3 test 소스 이미지(=OCR 인식기의 test)에서 각 모델의 best 가중치로 추론하고,
eval_det_unified 와 같은 `compute_ap50`(train_frcnn_kfold) 하나로 채점합니다. 모델별
입력 해상도는 각자의 학습 설정(YOLO 640 / FRCNN min640 / EffDet 512)을 따릅니다.

Usage: .venv/Scripts/python.exe eval_text_holdout.py [--models yolo26x,yolov5x,frcnn,effdet] [--limit N]
출력: artifacts/gt/text_holdout_ap50.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from PIL import Image

from train_frcnn_kfold import compute_ap50
import eval_det_unified as U

HERE = Path(__file__).resolve().parent
BASE = HERE / "artifacts" / "signboard_text_holdout"
MODELS = {
    "yolo26x": ("ultra", "artifacts/yolo26x_text_holdout/run/weights/best.pt", 640),
    "yolov5x": ("ultra", "artifacts/yolov5x_text_holdout/run/weights/best.pt", 640),
    "frcnn":   ("frcnn", "artifacts/frcnn_text_holdout/best_frcnn_text.pth", None),
    "effdet":  ("effdet", "artifacts/effdet_text_holdout/best_effdet_text.pth", 512),
}


def load_test(limit: int | None):
    items = []
    for ip in sorted((BASE / "images" / "test").glob("*.jpg"))[:limit]:
        W, H = Image.open(ip).size
        boxes = []
        lp = BASE / "labels" / "test" / f"{ip.stem}.txt"
        for line in lp.read_text().strip().splitlines():
            _, xc, yc, w, h = (float(v) for v in line.split()[:5])
            boxes.append(((xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H))
        items.append((ip, boxes, None))
    return items


def predict_frcnn_text(weights: Path, items):
    from train_frcnn_text_kfold import get_frcnn        # min_size 640 / max 1066 (텍스트용 구성)
    import torchvision.transforms.functional as TF
    m = get_frcnn(); m.load_state_dict(torch.load(weights, map_location="cpu")); m.to(U.DEVICE).eval()
    preds = []
    with torch.no_grad():
        for ip, _, _ in items:
            out = m([TF.to_tensor(Image.open(ip).convert("RGB")).to(U.DEVICE)])[0]
            b = out["boxes"].cpu().tolist(); s = out["scores"].cpu().tolist()
            preds.append([(bb[0], bb[1], bb[2], bb[3], sc) for bb, sc in zip(b, s)])
    return preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--score-thr", type=float, default=0.0)
    ap.add_argument("--out", default="artifacts/gt/text_holdout_ap50.csv")
    ap.add_argument("--weights", default="",
                    help="가중치 경로 덮어쓰기: name=path,name=path (스모크 테스트용)")
    args = ap.parse_args()
    for kv in filter(None, args.weights.split(",")):
        n, p = kv.split("=", 1)
        MODELS[n] = (MODELS[n][0], p, MODELS[n][2])

    items = load_test(args.limit)
    gts = [g for _, g, _ in items]
    print(f"[test] {len(items)} images, {sum(len(g) for g in gts)} word boxes")
    rows = []
    for name in args.models.split(","):
        kind, wpath, res = MODELS[name]
        wpath = HERE / wpath
        if not wpath.exists():
            print(f"[{name}] 가중치 없음: {wpath}"); continue
        if kind == "ultra":
            preds = U.predict_ultra(wpath, items, imgsz=res)
        elif kind == "frcnn":
            preds = predict_frcnn_text(wpath, items)
        else:
            preds = U.predict_effdet(wpath, items, img_size=res)
        a = compute_ap50(preds, gts, score_thr=args.score_thr)
        rows.append((name, a, sum(len(p) for p in preds)))
        print(f"[{name}] test AP@0.5 = {a:.4f}", flush=True)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["model", "test_AP50", "n_pred_boxes", "n_test_images", "n_gt_boxes"])
        for name, a, n in rows:
            w.writerow([name, f"{a:.4f}", n, len(items), sum(len(g) for g in gts)])
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
