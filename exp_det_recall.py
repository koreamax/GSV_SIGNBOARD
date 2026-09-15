#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""탐지 recall 개선 탐색 — 연쇄 손실(4.16)의 전부가 탐지 누락에서 오므로.

4.16에서 GT 크롭 78.6% ↔ 연쇄 53.0% 의 25.6%p 차이가 전적으로 탐지가 놓친
간판(FN 85개)에서 왔습니다. 인식을 개선해도 이 손실은 줄지 않으므로 탐지 recall
자체를 올려야 합니다.

**이미 배제된 길**: 입력 해상도 증가 — 960 → 1280 → 1600 에서 AP가 오히려
떨어집니다(0.832 → 0.814 → 0.747, `artifacts/gt/det_res{1280,1600}.csv`).

여기서는 GPU 재추론 없이 캐시된 OOF 예측(`det_unified_preds.json`, conf 0.001,
4모델 × 5fold)만으로 두 가지를 탐색합니다:
  1. **임계값 조정** — recall/precision 교환 곡선. 공짜지만 FP가 늘어납니다.
  2. **모델 union** — D23에서 OCR 검출기 union(CRAFT∪DB)이 통했던 것과 같은 발상.
     실패 모드가 다른 모델을 합치면 recall이 오르는지.

Usage:
  .venv/Scripts/python.exe exp_det_recall.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import eval_det_unified as U

HERE = Path(__file__).resolve().parent
CACHE = HERE / "artifacts" / "gt" / "det_unified_preds.json"


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


def nms(boxes, thr=0.6):
    """모델 간 중복 제거. 신뢰도 내림차순 greedy."""
    out = []
    for b in sorted(boxes, key=lambda x: -x[4]):
        if all(iou(b, k) < thr for k in out):
            out.append(b)
    return out


def score(per_image, conf, match_iou=0.5):
    """(recall, precision, TP, FP, FN) — 이미지별 그리디 1:1 매칭."""
    tp = fp = fn = 0
    for boxes, gts in per_image:
        pl = sorted([b for b in boxes if b[4] >= conf], key=lambda x: -x[4])
        used = set()
        for b in pl:
            best_j, best_v = -1, 0.0
            for j, g in enumerate(gts):
                if j in used:
                    continue
                v = iou(b, g)
                if v > best_v:
                    best_v, best_j = v, j
            if best_v >= match_iou:
                used.add(best_j); tp += 1
            else:
                fp += 1
        fn += len(gts) - len(used)
    r = tp / max(tp + fn, 1)
    p = tp / max(tp + fp, 1)
    return r, p, tp, fp, fn


def build(cache, models, folds=5, nms_thr=0.6):
    """모델 목록을 합쳐 이미지별 (예측박스, GT박스) 리스트를 만듭니다."""
    per_image = []
    for i in range(folds):
        items = U.load_fold_val(i)
        for idx, (ip, gts, region) in enumerate(items):
            boxes = []
            for m in models:
                key = f"{m}::{i}"
                if key in cache and idx < len(cache[key]):
                    boxes.extend(cache[key][idx])
            if len(models) > 1:
                boxes = nms(boxes, nms_thr)
            per_image.append((boxes, gts, region))
    return per_image


def emit(cache, models, conf, tag, nms_thr, match_iou=0.5) -> None:
    """고른 구성을 크롭 입력 CSV + 매칭표로 내보냅니다 (e2e_det_boxes.py 와 동일 스키마).

    캐시를 쓰므로 GPU 재추론이 없습니다. 크롭 번호는 신뢰도 내림차순이라
    `make_crops_from_gt_polygon.py` 가 매기는 순번과 일치합니다."""
    GT_DIR = HERE / "artifacts" / "gt"
    by_region: dict[str, dict[str, tuple]] = {}
    for i in range(5):
        for idx, (ip, gts, region) in enumerate(U.load_fold_val(i)):
            boxes = []
            for m in models:
                k = f"{m}::{i}"
                if k in cache and idx < len(cache[k]):
                    boxes.extend(cache[k][idx])
            boxes = [b for b in boxes if b[4] >= conf]
            if len(models) > 1:
                boxes = nms(boxes, nms_thr)
            boxes.sort(key=lambda x: -x[4])
            parts = ip.stem.split("__")          # total__region__id
            key = f"{parts[1]}__{parts[2]}"
            by_region.setdefault(region, {})[key] = (boxes, gts)

    for region, photos in by_region.items():
        det_rows, match_rows = [], []
        for key in sorted(photos):
            boxes, gts = photos[key]
            used = set()
            for n, b in enumerate(boxes, 1):
                x1, y1, x2, y2 = b[:4]
                det_rows.append({"filename": f"{key}.jpg", "w": 0, "h": 0,
                                 "x1": x1, "y1": y1, "x2": x2, "y2": y1,
                                 "x3": x2, "y3": y2, "x4": x1, "y4": y2,
                                 "class": "signboard"})
                bj, bv = -1, 0.0
                for j, g in enumerate(gts):
                    if j in used:
                        continue
                    v = iou(b, g)
                    if v > bv:
                        bv, bj = v, j
                row = {"det_crop": f"{key}__crop_{n:03d}", "photo": key,
                       "iou": round(bv, 4), "conf": round(b[4], 4)}
                if bv >= match_iou:
                    used.add(bj)
                    match_rows.append({**row, "gt_index": bj + 1, "status": "TP"})
                else:
                    match_rows.append({**row, "gt_index": "", "status": "FP"})
            for j in range(len(gts)):
                if j not in used:
                    match_rows.append({"det_crop": "", "photo": key, "gt_index": j + 1,
                                       "iou": 0.0, "conf": "", "status": "FN"})
        with (GT_DIR / f"gt_{region}_det{tag}.csv").open("w", encoding="utf-8",
                                                         newline="") as f:
            w = csv.DictWriter(f, fieldnames=["filename", "w", "h", "x1", "y1", "x2",
                                              "y2", "x3", "y3", "x4", "y4", "class"])
            w.writeheader(); w.writerows(det_rows)
        with (GT_DIR / f"e2e_match_{region}{tag}.csv").open("w", encoding="utf-8",
                                                            newline="") as f:
            w = csv.DictWriter(f, fieldnames=["det_crop", "photo", "gt_index", "iou",
                                              "conf", "status"])
            w.writeheader(); w.writerows(match_rows)
        n_tp = sum(1 for r in match_rows if r["status"] == "TP")
        n_fn = sum(1 for r in match_rows if r["status"] == "FN")
        print(f"  [{region}] 크롭 {len(det_rows)}  TP {n_tp} FN {n_fn} "
              f"→ gt_{region}_det{tag}.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nms", type=float, default=0.6,
                    help="모델 union 시 중복 제거 IoU")
    ap.add_argument("--emit", nargs=3, metavar=("MODELS", "CONF", "TAG"),
                    help="구성을 크롭 입력으로 내보냄. 예: "
                         "--emit yolo26x,frcnn 0.25 _union")
    args = ap.parse_args()

    if args.emit:
        cache = json.loads(CACHE.read_text(encoding="utf-8"))
        models, conf, tag = args.emit[0].split(","), float(args.emit[1]), args.emit[2]
        print(f"[emit] models={models} conf={conf} tag={tag}")
        emit(cache, models, conf, tag, args.nms)
        return

    cache = json.loads(CACHE.read_text(encoding="utf-8"))
    combos = [
        ("yolo26x (현행)", ["yolo26x"]),
        ("yolo26x ∪ frcnn", ["yolo26x", "frcnn"]),
        ("yolo26x ∪ yolov5x", ["yolo26x", "yolov5x"]),
        ("yolo26x ∪ frcnn ∪ yolov5x", ["yolo26x", "frcnn", "yolov5x"]),
    ]
    confs = [0.25, 0.15, 0.10, 0.05, 0.02]

    print("탐지 recall/precision 교환 (OOF, IoU 0.5 매칭, GT 463개)\n")
    print(f"{'구성':28s} {'conf':>5s} {'recall':>8s} {'prec':>7s} "
          f"{'TP':>5s} {'FP':>5s} {'FN':>4s} {'크롭수':>7s}")
    base = None
    for label, models in combos:
        per = [(b, g) for b, g, _ in build(cache, models, nms_thr=args.nms)]
        for c in confs:
            r, p, tp, fp, fn = score(per, c)
            mark = ""
            if label.startswith("yolo26x (") and c == 0.25:
                base = (r, p); mark = "  ← 4.16 기준"
            elif base and r >= base[0] + 0.05 and p >= base[1] - 0.10:
                mark = "  ★"
            print(f"{label:28s} {c:5.2f} {r:8.3f} {p:7.3f} "
                  f"{tp:5d} {fp:5d} {fn:4d} {tp+fp:7d}{mark}")
        print()

    print("★ = 현행 대비 recall +5%p 이상이면서 precision 하락 10%p 이내")
    print("\n※ recall이 올라도 그 박스가 읽히지 않으면 연쇄 성능은 안 오릅니다 —")
    print("  후보 구성을 골라 crop→OCR 까지 돌려야 실제 이득이 확인됩니다.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
