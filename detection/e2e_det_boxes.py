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
# 기본값은 옛 구성(비-group-aware)입니다. 표 1 의 0.860 은 group-aware 인
# yolo26x_kfold_grouped + kfold_grouped 폴드이므로, 연쇄 측정은 반드시
# --fold-data artifacts/kfold_grouped --weights artifacts/yolo26x_kfold_grouped 로 돌립니다.
FOLD_DATA = HERE / "artifacts" / "yolo11x_kfold"
WEIGHTS = HERE / "artifacts" / "yolo26x_kfold"
REGIONS = ("gangnam", "brooklyn", "suwon")
# 크롭 스크립트(make_crops_from_gt_polygon.py)의 --gt-size large 기준 좌표계.
# 사진 크기가 2197/8192 두 종류라 좌표를 이 기준으로 맞춰 적어 두면 크롭 때 추측이 필요 없습니다.
LARGE_W, LARGE_H = 8192, 4828


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
    ap.add_argument("--fold-data", default=str(FOLD_DATA),
                    help="폴드 정의(total_fold{i}/dataset/images/val). 표 1 과 같게: artifacts/kfold_grouped")
    ap.add_argument("--weights", default=str(WEIGHTS),
                    help="폴드별 가중치 루트. 표 1 과 같게: artifacts/yolo26x_kfold_grouped")
    ap.add_argument("--tag", default="",
                    help="출력 파일 접미사 (gt_{region}_det{tag}.csv, e2e_match_{region}{tag}.csv)")
    args = ap.parse_args()
    fold_data = Path(args.fold_data)
    weights_root = Path(args.weights)
    fold_data = fold_data if fold_data.is_absolute() else HERE / fold_data
    weights_root = weights_root if weights_root.is_absolute() else HERE / weights_root

    from ultralytics import YOLO

    gt_boxes = load_gt_boxes()
    # region -> photo_key -> [(box, conf)]
    preds: dict[str, dict[str, list]] = {r: defaultdict(list) for r in REGIONS}
    sizes: dict[str, tuple[int, int]] = {}        # photo_key -> (W, H)

    for fi in range(args.folds):
        val_dir = fold_data / f"total_fold{fi}" / "dataset" / "images" / "val"
        wt = weights_root / f"fold{fi}" / "weights" / "best.pt"
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
                oh, ow = r.orig_shape
                sizes[key] = (int(ow), int(oh))
                preds[region].setdefault(key, [])      # 예측 0개인 사진도 기록
                for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                    preds[region][key].append((tuple(b), float(c)))

    for region in REGIONS:
        # ---- 크롭 입력 CSV (GT와 동일 스키마 → 크롭 스크립트 재사용) ----
        det_rows, match_rows = [], []
        n_tp = n_fp = n_fn = 0
        # 어느 fold 의 val 에도 없는 사진(brooklyn__38, gangnam__35)은 out-of-fold 예측이 없습니다.
        # 합집합으로 돌면 그 사진의 GT 간판이 전부 FN 이 되어, 검출기가 보지도 않은 사진을
        # 놓친 것으로 셉니다. 표 1·2 와 같은 298장만 채점합니다.
        skipped = sorted(k for k in gt_boxes if k.startswith(region + "__") and k not in preds[region])
        if skipped:
            print(f"[{region}] fold 밖 사진 {len(skipped)}장 제외 (GT 간판 "
                  f"{sum(len(gt_boxes[k]) for k in skipped)}개): {', '.join(skipped)}")
        for key in sorted(preds[region]):
            pl = sorted(preds[region].get(key, []), key=lambda t: -t[1])
            gl = gt_boxes.get(key, [])
            used = set()
            for idx, (box, conf) in enumerate(pl, 1):
                # GT 크롭 번호와 헷갈리지 않게 det 크롭은 같은 규칙(crop_00N)으로
                # 매기되 별도 디렉터리에 저장합니다.
                name = f"{key}__crop_{idx:03d}"
                x1, y1, x2, y2 = box
                # 크롭 스크립트는 좌표 크기로 2197/8192 기준을 '추측'하는데, 8K 사진의
                # 왼쪽 위 간판은 작은 사진으로 오인해 엉뚱한 곳을 자릅니다(기존 502개 중 9개).
                # 그래서 모든 좌표를 8192x4828 기준으로 환산해 적고 --gt-size large 로 자릅니다.
                W, H = sizes[key]
                sx, sy = LARGE_W / W, LARGE_H / H
                det_rows.append({"filename": f"{key}.jpg", "w": LARGE_W, "h": LARGE_H,
                                 "x1": x1 * sx, "y1": y1 * sy, "x2": x2 * sx, "y2": y1 * sy,
                                 "x3": x2 * sx, "y3": y2 * sy, "x4": x1 * sx, "y4": y2 * sy,
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

        with (GT_DIR / f"gt_{region}_det{args.tag}.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["filename", "w", "h", "x1", "y1", "x2",
                                              "y2", "x3", "y3", "x4", "y4", "class"])
            w.writeheader(); w.writerows(det_rows)
        with (GT_DIR / f"e2e_match_{region}{args.tag}.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["det_crop", "photo", "gt_index", "iou",
                                              "conf", "status"])
            w.writeheader(); w.writerows(match_rows)
        rec = n_tp / max(n_tp + n_fn, 1)
        prec = n_tp / max(n_tp + n_fp, 1)
        print(f"[{region}] 예측 {len(det_rows)}개  TP {n_tp} / FP {n_fp} / FN {n_fn}  "
              f"recall {rec:.3f}  precision {prec:.3f}")

    # 단어 박스 연쇄 AP 가 같은 간판 검출 결과를 쓰도록 원본 좌표 그대로 남깁니다.
    import json
    out = {k: {"size": list(sizes[k]),
               "boxes": [[*b, c] for b, c in sorted(preds[k.split("__")[0]].get(k, []),
                                                   key=lambda t: -t[1])]}
           for k in sorted(sizes)}
    jp = GT_DIR / f"det_oof{args.tag}.json"
    jp.write_text(json.dumps(out), encoding="utf-8")
    print(f"[saved] {jp.name}: 사진 {len(out)}장 · 간판 박스 {sum(len(v['boxes']) for v in out.values())}개")
    print(f"[다음] 크롭은 반드시 --gt-size large 로: make_crops_from_gt_polygon.py "
          f"--gt-csv artifacts/gt/gt_{{region}}_det{args.tag}.csv --gt-size large")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
