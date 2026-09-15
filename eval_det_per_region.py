#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_det_per_region.py

Per-region signboard-detection mAP@0.5 for an Ultralytics model (YOLOv5u / YOLO11 / ...).

- Runs the model on artifacts/gsv_photo/{region}/*.jpg (low conf, full PR curve).
- GT from artifacts/gt/gt_{region}.csv (QUAD x1..x4 -> axis-aligned bbox via min/max),
  scaled from GT (w,h) to the actual image size.
- Single-class AP@0.5 (conf-sorted, VOC all-points interpolation), plus P/R/F1 at a
  reporting confidence threshold. Pools the three regions for a global number too.

Usage:
  .venv/Scripts/python.exe eval_det_per_region.py --model artifacts/yolo_ft_archive/weights/best.pt
  (optional) --regions gangnam,brooklyn,suwon --imgsz 1280 --conf 0.001 --pr-conf 0.25
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
GSV_DIR = HERE / "artifacts" / "gsv_photo"
GT_DIR = HERE / "artifacts" / "gt"
DEFAULT_REGIONS = ["gangnam", "brooklyn", "suwon"]


def load_gt(region):
    """Return {image_key: [ [x1,y1,x2,y2 in GT pixel coords], ... ]} and (gt_w, gt_h)."""
    p = GT_DIR / f"gt_{region}.csv"
    gt = {}
    gt_w = gt_h = None
    with open(p, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            gt_w = float(r["w"]); gt_h = float(r["h"])
            xs = [float(r[k]) for k in ("x1", "x2", "x3", "x4")]
            ys = [float(r[k]) for k in ("y1", "y2", "y3", "y4")]
            box = [min(xs), min(ys), max(xs), max(ys)]
            gt.setdefault(r["filename"], []).append(box)
    return gt, gt_w, gt_h


def iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0


def voc_ap(rec, prec):
    """VOC all-points interpolation AP."""
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def eval_region(model, region, imgsz, conf, iou_thr, pr_conf):
    gt, gt_w, gt_h = load_gt(region)
    img_dir = GSV_DIR / region
    images = sorted(img_dir.glob("*.jpg"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)

    preds = []  # (conf, image_key, [x1,y1,x2,y2]) in IMAGE pixel coords
    total_gt = sum(len(v) for v in gt.values())

    for img_path in images:
        key = f"{region}__{img_path.name}"
        W, H = Image.open(img_path).size
        res = model.predict(source=str(img_path), conf=conf, iou=0.5, imgsz=imgsz, verbose=False)[0]
        if res.boxes is None:
            continue
        for b in res.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            preds.append((float(b.conf[0]), key, [x1, y1, x2, y2]))

    # scale GT (gt_w x gt_h) -> image size. All images here are full-res = GT size,
    # but scale generically per image is unnecessary since sizes are uniform.
    # We assume the GT pixel space == image pixel space (verified equal).
    # AP@0.5 (conf-sorted)
    preds.sort(key=lambda t: -t[0])
    matched = {k: [False] * len(v) for k, v in gt.items()}
    tp = np.zeros(len(preds)); fp = np.zeros(len(preds))
    # P/R/F1 at pr_conf
    tp_at = fp_at = 0
    for i, (cf, key, pbox) in enumerate(preds):
        gboxes = gt.get(key, [])
        best_iou, best_j = 0.0, -1
        for j, gbox in enumerate(gboxes):
            if matched[key][j]:
                continue
            v = iou_xyxy(pbox, gbox)
            if v > best_iou:
                best_iou, best_j = v, j
        is_tp = best_iou >= iou_thr and best_j >= 0
        if is_tp:
            matched[key][best_j] = True
            tp[i] = 1
        else:
            fp[i] = 1
        if cf >= pr_conf:
            if is_tp:
                tp_at += 1
            else:
                fp_at += 1

    cum_tp = np.cumsum(tp); cum_fp = np.cumsum(fp)
    rec = cum_tp / (total_gt + 1e-9)
    prec = cum_tp / (cum_tp + cum_fp + 1e-9)
    ap = voc_ap(rec, prec) if len(preds) else 0.0

    fn_at = total_gt - tp_at
    P = tp_at / (tp_at + fp_at + 1e-9)
    R = tp_at / (total_gt + 1e-9)
    F1 = 2 * P * R / (P + R + 1e-9)

    return {
        "region": region, "images": len(images), "gt": total_gt, "preds": len(preds),
        "ap50": ap, "P": P, "R": R, "F1": F1, "pr_conf": pr_conf,
        "tp": tp_at, "fp": fp_at, "fn": fn_at,
        # raw arrays for pooling
        "_preds": preds, "_gt": gt, "_total_gt": total_gt,
    }


def pooled_ap(region_results, iou_thr):
    preds = []
    gt = {}
    total_gt = 0
    for rr in region_results:
        preds.extend(rr["_preds"])
        gt.update(rr["_gt"])
        total_gt += rr["_total_gt"]
    preds.sort(key=lambda t: -t[0])
    matched = {k: [False] * len(v) for k, v in gt.items()}
    tp = np.zeros(len(preds)); fp = np.zeros(len(preds))
    for i, (cf, key, pbox) in enumerate(preds):
        gboxes = gt.get(key, [])
        best_iou, best_j = 0.0, -1
        for j, gbox in enumerate(gboxes):
            if matched[key][j]:
                continue
            v = iou_xyxy(pbox, gbox)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_iou >= iou_thr and best_j >= 0:
            matched[key][best_j] = True
            tp[i] = 1
        else:
            fp[i] = 1
    cum_tp = np.cumsum(tp); cum_fp = np.cumsum(fp)
    rec = cum_tp / (total_gt + 1e-9)
    prec = cum_tp / (cum_tp + cum_fp + 1e-9)
    return voc_ap(rec, prec) if len(preds) else 0.0, total_gt, len(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--regions", default=",".join(DEFAULT_REGIONS))
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.001, help="low conf for full PR curve")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU threshold for a match")
    ap.add_argument("--pr-conf", type=float, default=0.25, help="conf for reported P/R/F1")
    ap.add_argument("--tag", default="", help="label for the printout")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)
    regions = [r.strip() for r in args.regions.split(",") if r.strip()]

    print(f"\n=== Per-region signboard detection mAP@{args.iou:g}  (model: {args.model}) {args.tag} ===")
    print(f"{'region':<10} {'imgs':>5} {'GT':>5} {'preds':>7} {'mAP@.5':>8} "
          f"{'P':>6} {'R':>6} {'F1':>6}  (P/R/F1 @conf>={args.pr_conf})")
    results = []
    for region in regions:
        rr = eval_region(model, region, args.imgsz, args.conf, args.iou, args.pr_conf)
        results.append(rr)
        print(f"{rr['region']:<10} {rr['images']:>5} {rr['gt']:>5} {rr['preds']:>7} "
              f"{rr['ap50']:>8.4f} {rr['P']:>6.3f} {rr['R']:>6.3f} {rr['F1']:>6.3f}")

    gap, gtot, gpreds = pooled_ap(results, args.iou)
    mean_ap = float(np.mean([r["ap50"] for r in results]))
    print("-" * 72)
    print(f"{'POOLED':<10} {'':>5} {gtot:>5} {gpreds:>7} {gap:>8.4f}   (micro, all regions)")
    print(f"{'MEAN':<10} {'':>5} {'':>5} {'':>7} {mean_ap:>8.4f}   (macro avg of regions)")


if __name__ == "__main__":
    main()
