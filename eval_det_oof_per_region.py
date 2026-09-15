#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A1/T4: leakage-free per-region signboard-detection mAP@0.5 (out-of-fold).

The GSV photos ARE the 5-fold training data (folds live in
artifacts/yolo11x_kfold/total_fold{i}/dataset), so scoring one fold's weights on
all 300 photos would report memorisation. Here every photo is predicted ONLY by
the fold model that held it out:

  fold i  ->  images in total_fold{i}/dataset/images/val  ->  <kfold>/fold{i}/weights/best.pt

The val splits partition the photo set, so the union is a genuine out-of-fold
prediction for every image, and per-region AP@0.5 follows directly.

GT: artifacts/gt/gt_{region}.csv (QUAD -> axis-aligned bbox), same loader and
VOC all-points AP as eval_det_per_region.py.

Usage:
  .venv/Scripts/python.exe eval_det_oof_per_region.py --kfold artifacts/yolo26x_kfold
  .venv/Scripts/python.exe eval_det_oof_per_region.py --kfold artifacts/yolo11x_kfold --name-pat "total_fold{i}"
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from eval_det_per_region import load_gt, iou_xyxy, voc_ap


def coco_ap(rec, prec):
    """101-point interpolated AP — the convention Ultralytics val() reports, so
    these numbers line up with the 5-fold table (VOC all-points runs ~0.04 lower)."""
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([1.0], prec, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.trapz(np.interp(x, mrec, mpre), x))

HERE = Path(__file__).resolve().parent
FOLD_DATA = HERE / "artifacts" / "yolo11x_kfold"      # canonical fold definition
REGIONS = ["gangnam", "brooklyn", "suwon"]
NAME_RE = re.compile(r"^total__([a-z]+)__(\d+)\.jpg$", re.I)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kfold", required=True,
                    help="weights root, e.g. artifacts/yolo26x_kfold")
    ap.add_argument("--name-pat", default="fold{i}",
                    help="run dir name inside --kfold (default 'fold{i}')")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.5, help="IoU for a TP match")
    ap.add_argument("--nms-iou", type=float, default=0.7,
                    help="NMS IoU; 0.7 matches Ultralytics val() so the numbers "
                         "are comparable to the 5-fold table")
    ap.add_argument("--pr-conf", type=float, default=0.25)
    ap.add_argument("--out", default="artifacts/gt/det_oof_per_region.csv")
    args = ap.parse_args()

    from ultralytics import YOLO

    # GT boxes are stored in EACH image's own pixel space (the csv's w,h column
    # varies per row — the photo set mixes 8192x4828 and 2197x1295), and we
    # predict on that same file, so no rescaling is applied.
    gts = {r: load_gt(r)[0] for r in REGIONS}

    # region -> list of (conf, image_key, box) ; every image predicted once, OOF
    preds: dict[str, list] = {r: [] for r in REGIONS}
    seen: dict[str, set] = {r: set() for r in REGIONS}
    fold_of: dict[str, int] = {}      # image_key -> fold that held it out

    for fi in range(args.folds):
        val_dir = FOLD_DATA / f"total_fold{fi}" / "dataset" / "images" / "val"
        wpath = Path(args.kfold) / args.name_pat.format(i=fi) / "weights" / "best.pt"
        if not val_dir.is_dir():
            sys.exit(f"[ABORT] fold split missing: {val_dir}")
        if not wpath.exists():
            sys.exit(f"[ABORT] weights missing: {wpath}")
        imgs = sorted(val_dir.glob("*.jpg"))
        print(f"[fold{fi}] {len(imgs)} held-out images  <- {wpath}")
        model = YOLO(str(wpath))
        for p in imgs:
            m = NAME_RE.match(p.name)
            if not m:
                continue
            region, idx = m.group(1).lower(), m.group(2)
            if region not in gts:
                continue
            key = f"{region}__{idx}.jpg"
            if key in seen[region]:      # each image must be scored exactly once
                continue
            seen[region].add(key)
            fold_of[key] = fi
            res = model.predict(source=str(p), conf=args.conf, iou=args.nms_iou,
                                imgsz=args.imgsz, verbose=False)[0]
            if res.boxes is None:
                continue
            for b in res.boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                preds[region].append((float(b.conf[0]), key, [x1, y1, x2, y2]))

    def ap_of(subset_keys):
        """AP@0.5 over an arbitrary image subset (spanning regions)."""
        gt_sub, pr = {}, []
        for r in REGIONS:
            for k, v in gts[r].items():
                if k in subset_keys:
                    gt_sub[k] = v
            pr += [p for p in preds[r] if p[1] in subset_keys]
        n_gt = sum(len(v) for v in gt_sub.values())
        pr.sort(key=lambda t: -t[0])
        matched = {k: [False] * len(v) for k, v in gt_sub.items()}
        tp = np.zeros(len(pr)); fp = np.zeros(len(pr))
        for i, (cf, key, pbox) in enumerate(pr):
            best_iou, best_j = 0.0, -1
            for j, gbox in enumerate(gt_sub.get(key, [])):
                if matched[key][j]:
                    continue
                v = iou_xyxy(pbox, gbox)
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_iou >= args.iou and best_j >= 0:
                matched[key][best_j] = True; tp[i] = 1
            else:
                fp[i] = 1
        rec = np.cumsum(tp) / (n_gt + 1e-9)
        prec = np.cumsum(tp) / np.maximum(np.cumsum(tp) + np.cumsum(fp), 1e-9)
        return (coco_ap(rec, prec) if len(pr) else 0.0), n_gt

    rows, pooled = [], []
    tot_gt = 0
    print(f"\n=== Out-of-fold per-region signboard detection (mAP@{args.iou:g}, "
          f"{args.kfold}) ===")
    print(f"{'region':<10} {'imgs':>5} {'GT':>5} {'preds':>7} {'mAP@.5':>8} "
          f"{'(VOC)':>7} {'P':>7} {'R':>7} {'F1':>7}")
    for r in REGIONS:
        gt = gts[r]
        # only score images that were actually predicted (OOF coverage)
        gt = {k: v for k, v in gt.items() if k in seen[r]}
        n_gt = sum(len(v) for v in gt.values())
        tot_gt += n_gt
        pr = sorted(preds[r], key=lambda t: -t[0])
        matched = {k: [False] * len(v) for k, v in gt.items()}
        tp = np.zeros(len(pr)); fp = np.zeros(len(pr))
        tp_at = fp_at = 0
        for i, (cf, key, pbox) in enumerate(pr):
            gboxes = gt.get(key, [])
            best_iou, best_j = 0.0, -1
            for j, gbox in enumerate(gboxes):
                if matched[key][j]:
                    continue
                v = iou_xyxy(pbox, gbox)
                if v > best_iou:
                    best_iou, best_j = v, j
            is_tp = best_iou >= args.iou and best_j >= 0
            if is_tp:
                matched[key][best_j] = True
                tp[i] = 1
            else:
                fp[i] = 1
            if cf >= args.pr_conf:
                tp_at += int(is_tp); fp_at += int(not is_tp)
        rec = np.cumsum(tp) / (n_gt + 1e-9)
        prec = np.cumsum(tp) / np.maximum(np.cumsum(tp) + np.cumsum(fp), 1e-9)
        ap50 = coco_ap(rec, prec) if len(pr) else 0.0
        ap50_voc = voc_ap(rec, prec) if len(pr) else 0.0
        P = tp_at / max(tp_at + fp_at, 1)
        R = tp_at / max(n_gt, 1)
        F1 = 2 * P * R / max(P + R, 1e-9)
        print(f"{r:<10} {len(seen[r]):>5} {n_gt:>5} {len(pr):>7} {ap50:>8.4f} "
              f"{ap50_voc:>7.4f} {P:>7.4f} {R:>7.4f} {F1:>7.4f}")
        rows.append([r, len(seen[r]), n_gt, len(pr), f"{ap50:.4f}",
                     f"{ap50_voc:.4f}", f"{P:.4f}", f"{R:.4f}", f"{F1:.4f}"])
        pooled.append(ap50)

    print("-" * 64)
    print(f"{'MEAN':<10} {'':>5} {tot_gt:>5} {'':>7} {np.mean(pooled):>8.4f}   "
          f"(macro avg of regions)")
    all_keys = set(fold_of)
    ap_all, _ = ap_of(all_keys)
    print(f"{'POOLED':<10} {'':>5} {'':>5} {'':>7} {ap_all:>8.4f}   (all 3 regions in one AP)")

    # Same predictions, aggregated per fold instead of per region — this is the
    # partition Ultralytics val() uses, so it isolates aggregation from method.
    fold_aps = []
    for fi in range(args.folds):
        a, n = ap_of({k for k, f in fold_of.items() if f == fi})
        fold_aps.append(a)
        print(f"  fold{fi} (같은 예측, fold 단위 집계): AP@0.5 = {a:.4f}  (GT {n})")
    print(f"  fold 평균 = {np.mean(fold_aps):.4f}   <- Ultralytics val() 집계와 동일 단위")
    out = Path(args.out)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["region", "images", "gt_boxes", "preds", "mAP50_coco101",
                    "mAP50_voc", "P", "R", "F1"])
        w.writerows(rows)
        w.writerow(["MEAN", "", tot_gt, "", f"{np.mean(pooled):.4f}", "", "", "", ""])
    print(f"[saved] {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
