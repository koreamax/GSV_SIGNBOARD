#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T2: all detection models under ONE protocol (single-class AP@0.5, same folds).

Table 2 previously mixed two things that make the models not directly comparable:

  1. two evaluators — YOLO via Ultralytics val(), FRCNN/EffDet via their own
     VOC all-points AP (measured gap ~0.04 on identical weights, 4.12);
  2. "peak over epochs" reporting — every trainer logged the BEST epoch's score,
     which is optimistic. For FRCNN/EffDet the saved checkpoint IS that epoch, but
     Ultralytics saves best.pt by *fitness* (0.1*mAP50 + 0.9*mAP50-95), so the
     YOLO row reported a score its saved weights never had (0.9006 vs 0.8785).

This script re-scores every model from its SAVED best checkpoint, on the SAME
fold val splits, with ONE AP implementation (imported from train_frcnn_kfold so
the FRCNN/EffDet numbers stay on their original footing).

Per-region breakdown comes free (fold images are named total__{region}__{n}.jpg).

Usage:
  .venv/Scripts/python.exe eval_det_unified.py                    # all models
  .venv/Scripts/python.exe eval_det_unified.py --models yolo26x,frcnn
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from train_frcnn_kfold import compute_ap50   # single shared AP@0.5

HERE = Path(__file__).resolve().parent
FOLD_ROOT = HERE / os.environ.get("FOLD_ROOT", "artifacts/yolo11x_kfold") / "total_fold{i}" / "dataset"  # D33
REGIONS = ["gangnam", "brooklyn", "suwon"]
NAME_RE = re.compile(r"^total__([a-z]+)__(\d+)$", re.I)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# D33: DET_TAG 환경변수로 재학습 가중치 세트를 고릅니다 (예: "_grouped"). 기본 "" = 기존.
_TAG = os.environ.get("DET_TAG", "")
MODELS = {
    "yolo26x": ("ultra", "artifacts/yolo26x_kfold" + _TAG + "/fold{i}/weights/best.pt"),
    "yolov5x": ("ultra", "artifacts/yolov5x_kfold" + _TAG + "/fold{i}/weights/best.pt"),
    "frcnn":   ("frcnn", "artifacts/frcnn_kfold_v2" + _TAG + "/best_frcnn_fold{i}.pth"),
    "effdet":  ("effdet", "artifacts/effdet_kfold" + _TAG + "/best_effdet_fold{i}.pth"),
}


def load_fold_val(i: int):
    """[(image_path, [gt xyxy in image pixels], region)] for fold i's val split."""
    root = Path(str(FOLD_ROOT).format(i=i))
    out = []
    for ip in sorted((root / "images" / "val").glob("*.jpg")):
        m = NAME_RE.match(ip.stem)
        region = m.group(1).lower() if m else "?"
        W, H = Image.open(ip).size
        boxes = []
        lp = root / "labels" / "val" / f"{ip.stem}.txt"
        if lp.exists():
            for line in lp.read_text().strip().split("\n"):
                if not line.strip():
                    continue
                _, xc, yc, w, h = [float(x) for x in line.split()]
                boxes.append(((xc - w / 2) * W, (yc - h / 2) * H,
                              (xc + w / 2) * W, (yc + h / 2) * H))
        out.append((ip, boxes, region))
    return out


def predict_ultra(weights: Path, items, imgsz=960, conf=0.001):
    from ultralytics import YOLO
    model = YOLO(str(weights))
    preds = []
    for ip, _, _ in items:
        r = model.predict(source=str(ip), conf=conf, iou=0.7, imgsz=imgsz, verbose=False)[0]
        p = []
        if r.boxes is not None:
            for b in r.boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                p.append((x1, y1, x2, y2, float(b.conf[0])))
        preds.append(p)
    return preds


def predict_frcnn(weights: Path, items, res: int | None = None):
    from train_frcnn_kfold import get_frcnn
    model = get_frcnn(2)
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    if res:      # torchvision resizes to min_size, capped by max_size
        model.transform.min_size = (res,)
        model.transform.max_size = int(res * 1333 / 800)   # keep the default ratio
    model.to(DEVICE).eval()
    import torchvision.transforms.functional as TF
    preds = []
    with torch.no_grad():
        for ip, _, _ in items:
            img = TF.to_tensor(Image.open(ip).convert("RGB")).to(DEVICE)
            out = model([img])[0]
            b = out["boxes"].cpu().tolist(); s = out["scores"].cpu().tolist()
            preds.append([(bb[0], bb[1], bb[2], bb[3], sc) for bb, sc in zip(b, s)])
    return preds


def predict_effdet(weights: Path, items, img_size=512):
    """Must mirror train_effdet_kfold exactly: letterbox (not plain resize) and
    to_tensor with NO ImageNet normalisation — that is what the net was fed."""
    from effdet import create_model, DetBenchPredict
    from eval_map_only import letterbox_pil
    import torchvision.transforms.functional as TF

    net = create_model("tf_efficientdet_d0", bench_task="", num_classes=1,
                       pretrained=False, image_size=(img_size, img_size))
    net.load_state_dict(torch.load(weights, map_location="cpu"))
    pb = DetBenchPredict(net).to(DEVICE).eval()
    preds = []
    with torch.no_grad():
        for ip, _, _ in items:
            im = Image.open(ip).convert("RGB")
            img_lb, scale, pl, pt = letterbox_pil(im, img_size)
            out = pb(TF.to_tensor(img_lb).unsqueeze(0).to(DEVICE))[0].detach().cpu().tolist()
            # undo letterbox: subtract padding, then divide by the resize scale
            preds.append([((d[0] - pl) / scale, (d[1] - pt) / scale,
                           (d[2] - pl) / scale, (d[3] - pt) / scale, d[4]) for d in out])
    return preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--res", type=int, default=None,
                    help="override inference resolution for every selected model "
                         "(ultra: imgsz, frcnn: min_size, effdet: image_size). "
                         "Omit to use each model's training resolution.")
    ap.add_argument("--score-thr", type=float, default=0.0,
                    help="min score kept before AP. 0.0 = full PR curve (fair "
                         "across models); 0.05 reproduces the legacy FRCNN/EffDet runs")
    ap.add_argument("--cache", default="artifacts/gt/det_unified_preds.json",
                    help="prediction cache; re-scoring at another threshold is instant")
    ap.add_argument("--refresh", action="store_true", help="ignore the cache")
    ap.add_argument("--out", default="artifacts/gt/det_unified_protocol.csv")
    args = ap.parse_args()
    names = [m.strip() for m in args.models.split(",") if m.strip()]

    folds = {i: load_fold_val(i) for i in range(args.folds)}
    for i, it in folds.items():
        print(f"[fold{i}] {len(it)} val images, {sum(len(b) for _, b, _ in it)} GT boxes")

    import json
    cache_path = HERE / args.cache
    cache = {}
    if cache_path.exists() and not args.refresh:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"[cache] {cache_path.name}: {len(cache)} (model,fold) entries")

    rows, table = [], {}
    for name in names:
        kind, pat = MODELS[name]
        fold_aps, reg_pred, reg_gt = [], {r: [] for r in REGIONS}, {r: [] for r in REGIONS}
        for i in range(args.folds):
            w = HERE / pat.format(i=i)
            if not w.exists():
                print(f"[skip] {name} fold{i}: weights missing ({w})")
                fold_aps.append(None)
                continue
            items = folds[i]
            res = args.res            # None = each model's own training resolution
            ck = f"{name}::{i}" + (f"::r{res}" if res else "")
            if ck in cache:
                preds = [[tuple(p) for p in im] for im in cache[ck]]
            else:
                if kind == "ultra":
                    preds = predict_ultra(w, items, imgsz=res or args.imgsz)
                elif kind == "frcnn":
                    preds = predict_frcnn(w, items, res=res)
                else:
                    preds = predict_effdet(w, items, img_size=res or 512)
                cache[ck] = [[list(p) for p in im] for im in preds]
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(cache), encoding="utf-8")
            gts = [b for _, b, _ in items]
            ap50 = compute_ap50(preds, gts, score_thr=args.score_thr)
            fold_aps.append(ap50)
            print(f"[{name}] fold{i}: AP@0.5 = {ap50:.4f}", flush=True)
            for (ipath, gb, region), pr in zip(items, preds):
                if region in reg_pred:
                    reg_pred[region].append(pr); reg_gt[region].append(gb)
        ok = [a for a in fold_aps if a is not None]
        mean = float(np.mean(ok)) if ok else 0.0
        per_reg = {r: (compute_ap50(reg_pred[r], reg_gt[r], score_thr=args.score_thr)
                       if reg_pred[r] else 0.0) for r in REGIONS}
        table[name] = (fold_aps, mean, per_reg)
        rows.append([name, *[f"{a:.4f}" if a is not None else "" for a in fold_aps],
                     f"{mean:.4f}", *[f"{per_reg[r]:.4f}" for r in REGIONS]])

    print(f"\n=== 통합 프로토콜: 저장된 best 체크포인트 + 단일 AP@0.5 (동일 5-Fold) ===")
    print(f"{'model':<10} " + " ".join(f"{'f'+str(i):>7}" for i in range(args.folds)) +
          f" {'MEAN':>8} | " + " ".join(f"{r[:5]:>7}" for r in REGIONS))
    for name in names:
        fa, mean, pr = table[name]
        print(f"{name:<10} " + " ".join(f"{a:7.4f}" if a is not None else f"{'-':>7}" for a in fa) +
              f" {mean:8.4f} | " + " ".join(f"{pr[r]:7.4f}" for r in REGIONS))

    out = Path(args.out)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", *[f"fold{i}" for i in range(args.folds)], "mean",
                    *[f"AP50_{r}" for r in REGIONS]])
        w.writerows(rows)
    print(f"[saved] {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
