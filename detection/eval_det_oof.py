#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_det_oof.py

Honest per-region mAP@0.5 under 5-fold CV via OUT-OF-FOLD prediction:
each GSV image is predicted ONLY by the fold model for which it was in the
validation split (so we never evaluate on an image the model trained on).

Reconstructs the identical fold split used in training by importing
get_all_region_pairs() + make_kfold_splits() from one_click_finetune_yolo11
(seed=42), so it matches whatever was trained into <kfold-root>/runs/total_fold{i}.

GT: artifacts/gt/gt_{region}.csv (QUAD -> axis-aligned bbox).
Single-class AP@0.5 (conf-sorted, VOC all-points). Per-region + pooled.

Usage:
  .venv/Scripts/python.exe eval_det_oof.py --kfold-root artifacts/yolov5_kfold --imgsz 1280
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from eval_det_per_region import iou_xyxy, voc_ap, load_gt  # reuse helpers
from one_click_finetune_yolo11 import get_all_region_pairs, make_kfold_splits

HERE = Path(__file__).resolve().parents[1]
REGIONS = ["gangnam", "brooklyn", "suwon"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kfold-root", required=True,
                    help="e.g. artifacts/yolov5_kfold (expects runs/total_fold{i}/weights/best.pt)")
    ap.add_argument("--prefix", default="total")
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    from ultralytics import YOLO

    pairs = get_all_region_pairs(HERE / "artifacts")
    splits = make_kfold_splits(pairs, k=args.kfold, seed=args.seed)

    # GT per region
    gt = {}
    for region in REGIONS:
        g, _, _ = load_gt(region)
        gt.update(g)

    # collect OOF predictions: for each fold, predict its val images with that fold's model
    preds = []  # (conf, image_key, [x1,y1,x2,y2])
    kfold_root = Path(args.kfold_root)
    for fi in range(args.kfold):
        model_path = kfold_root / "runs" / f"{args.prefix}_fold{fi}" / "weights" / "best.pt"
        if not model_path.exists():
            print(f"[WARN] fold {fi} model missing: {model_path}")
            continue
        model = YOLO(str(model_path))
        val_pairs = splits[fi]["val"]
        for (jpg, js, region) in val_pairs:
            key = f"{region}__{jpg.name}"
            res = model.predict(source=str(jpg), conf=args.conf, iou=0.5,
                                imgsz=args.imgsz, verbose=False)[0]
            if res.boxes is None:
                continue
            for b in res.boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                preds.append((float(b.conf[0]), key, [x1, y1, x2, y2]))

    def ap_for(keys_subset):
        sub_gt = {k: v for k, v in gt.items() if k in keys_subset}
        sub_preds = [p for p in preds if p[1] in keys_subset]
        total_gt = sum(len(v) for v in sub_gt.values())
        sub_preds.sort(key=lambda t: -t[0])
        matched = {k: [False] * len(v) for k, v in sub_gt.items()}
        tp = np.zeros(len(sub_preds)); fp = np.zeros(len(sub_preds))
        for i, (cf, key, pbox) in enumerate(sub_preds):
            best_iou, best_j = 0.0, -1
            for j, gbox in enumerate(sub_gt.get(key, [])):
                if matched[key][j]:
                    continue
                v = iou_xyxy(pbox, gbox)
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_iou >= args.iou and best_j >= 0:
                matched[key][best_j] = True; tp[i] = 1
            else:
                fp[i] = 1
        cum_tp = np.cumsum(tp); cum_fp = np.cumsum(fp)
        rec = cum_tp / (total_gt + 1e-9)
        prec = cum_tp / (cum_tp + cum_fp + 1e-9)
        return (voc_ap(rec, prec) if len(sub_preds) else 0.0), total_gt, len(sub_preds)

    print(f"\n=== OOF per-region signboard detection mAP@{args.iou:g}  "
          f"(5-fold, {args.kfold_root}) {args.tag} ===")
    print(f"{'region':<10} {'GT':>5} {'preds':>7} {'mAP@.5':>8}")
    aps = []
    for region in REGIONS:
        keys = {f"{region}__{jpg.name}" for (jpg, js, r) in pairs if r == region}
        a, g, p = ap_for(keys)
        aps.append(a)
        print(f"{region:<10} {g:>5} {p:>7} {a:>8.4f}")
    all_keys = set(gt.keys())
    ga, gg, gp = ap_for(all_keys)
    print("-" * 34)
    print(f"{'POOLED':<10} {gg:>5} {gp:>7} {ga:>8.4f}   (micro)")
    print(f"{'MEAN':<10} {'':>5} {'':>7} {float(np.mean(aps)):>8.4f}   (macro)")


if __name__ == "__main__":
    main()
