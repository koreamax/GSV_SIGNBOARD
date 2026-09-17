#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""연쇄(end-to-end) 평가 1단계 — 탐지기가 실제로 낸 박스를 크롭 입력으로 내보냅니다.

**왜 필요한가:** 지금까지 OCR(4.4·4.14)과 태깅(4.13·4.15) 평가는 전부 **GT 폴리곤에서
자른 크롭**으로 했습니다. 즉 탐지가 놓친 간판은 평가에 아예 들어오지 않아, 모듈별
점수를 곱한 값이 실제 파이프라인 성능이라는 보장이 없습니다. 파이프라인 논문에서
반드시 나오는 질문이므로 탐지 출력으로 다시 자른 크롭에서 전 단계를 재측정합니다.

**누수 통제**: GSV 사진 300장이 곧 5-Fold 학습 데이터이므로, 4.12와 동일하게
**out-of-fold** 예측만 씁니다 — 각 사진은 그 사진을 val로 held-out 했던 fold의
가중치로만 예측합니다.

산출물:
  artifacts/gt/gt_{region}_det.csv    탐지 박스 (GT CSV와 같은 폴리곤 형식 →
                                      make_crops_from_gt_polygon.py 를 그대로 재사용)
  artifacts/gt/e2e_match_{region}.csv 예측↔GT 매칭표 (IoU 기준)
                                      det_crop / gt_crop / iou / status
                                      status ∈ {TP, FP, FN}

Usage:
  .venv/Scripts/python.exe e2e_det_boxes.py --conf 0.25
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
GT_DIR = HERE / "artifacts" / "gt"
FOLD_DATA = HERE / "artifacts" / "yolo11x_kfold"     # fold 정의(4.12와 동일)
WEIGHTS = HERE / "artifacts" / "yolo26x_kfold"       # 주 탐지 모델
REGIONS = ("gangnam", "brooklyn", "suwon")


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def load_gt_boxes() -> dict[str, list]:
    """total_gt.csv → {photo_key: [(x1,y1,x2,y2), ...]}  (폴리곤의 외접 사각형)"""
    out: dict[str, list] = defaultdict(list)
    with (GT_DIR / "total_gt.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            xs = [float(r[f"x{i}"]) for i in range(1, 5)]
            ys = [float(r[f"y{i}"]) for i in range(1, 5)]
            key = Path(r["filename"]).stem          # e.g. brooklyn__1
            out[key].append((min(xs), min(ys), max(xs), max(ys)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.25,
                    help="배포 운용점. 4.12의 P/R 보고에 쓰인 값과 동일(0.25)")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--nms-iou", type=float, default=0.7)
    ap.add_argument("--match-iou", type=float, default=0.5,
                    help="예측↔GT를 같은 간판으로 볼 IoU 하한")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    from ultralytics import YOLO

    gt_boxes = load_gt_boxes()
    # region -> photo_key -> [(box, conf)]
    preds: dict[str, dict[str, list]] = {r: defaultdict(list) for r in REGIONS}

    for fi in range(args.folds):
        val_dir = FOLD_DATA / f"total_fold{fi}" / "dataset" / "images" / "val"
        wt = WEIGHTS / f"fold{fi}" / "weights" / "best.pt"
        imgs = sorted(val_dir.glob("*.jpg"))
        print(f"[fold{fi}] {len(imgs)}장 ← {wt.relative_to(HERE)}", flush=True)
        model = YOLO(str(wt))
        for i in range(0, len(imgs), 8):
            batch = imgs[i:i + 8]
            res = model.predict([str(p) for p in batch], imgsz=args.imgsz,
                                conf=args.conf, iou=args.nms_iou, verbose=False)
            for p, r in zip(batch, res):
                # total__brooklyn__12 → brooklyn / brooklyn__12
                parts = p.stem.split("__")
                region, key = parts[1], f"{parts[1]}__{parts[2]}"
                for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                    preds[region][key].append((tuple(b), float(c)))

    for region in REGIONS:
        # ---- 크롭 입력 CSV (GT와 동일 스키마 → 크롭 스크립트 재사용) ----
        det_rows, match_rows = [], []
        n_tp = n_fp = n_fn = 0
        for key in sorted(set(list(preds[region]) + [k for k in gt_boxes
                                                     if k.startswith(region + "__")])):
            pl = sorted(preds[region].get(key, []), key=lambda t: -t[1])
            gl = gt_boxes.get(key, [])
            used = set()
            for idx, (box, conf) in enumerate(pl, 1):
                # GT 크롭 번호와 헷갈리지 않게 det 크롭은 같은 규칙(crop_00N)으로
                # 매기되 별도 디렉터리에 저장합니다.
                name = f"{key}__crop_{idx:03d}"
                x1, y1, x2, y2 = box
                det_rows.append({"filename": f"{key}.jpg", "w": 0, "h": 0,
                                 "x1": x1, "y1": y1, "x2": x2, "y2": y1,
                                 "x3": x2, "y3": y2, "x4": x1, "y4": y2,
                                 "class": "signboard"})
                best_j, best_i = -1, 0.0
                for j, g in enumerate(gl):
                    if j in used:
                        continue
                    v = iou(box, g)
                    if v > best_i:
                        best_i, best_j = v, j
                if best_i >= args.match_iou:
                    used.add(best_j)
                    n_tp += 1
                    match_rows.append({"det_crop": name, "photo": key,
                                       "gt_index": best_j + 1, "iou": round(best_i, 4),
                                       "conf": round(conf, 4), "status": "TP"})
                else:
                    n_fp += 1
                    match_rows.append({"det_crop": name, "photo": key, "gt_index": "",
                                       "iou": round(best_i, 4), "conf": round(conf, 4),
                                       "status": "FP"})
            for j in range(len(gl)):
                if j not in used:
                    n_fn += 1
                    match_rows.append({"det_crop": "", "photo": key,
                                       "gt_index": j + 1, "iou": 0.0, "conf": "",
                                       "status": "FN"})

        with (GT_DIR / f"gt_{region}_det.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["filename", "w", "h", "x1", "y1", "x2",
                                              "y2", "x3", "y3", "x4", "y4", "class"])
            w.writeheader(); w.writerows(det_rows)
        with (GT_DIR / f"e2e_match_{region}.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["det_crop", "photo", "gt_index", "iou",
                                              "conf", "status"])
            w.writeheader(); w.writerows(match_rows)
        rec = n_tp / max(n_tp + n_fn, 1)
        prec = n_tp / max(n_tp + n_fp, 1)
        print(f"[{region}] 예측 {len(det_rows)}개  TP {n_tp} / FP {n_fp} / FN {n_fn}  "
              f"recall {rec:.3f}  precision {prec:.3f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
