#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""단어 박스 탐지를 '단독'이 아니라 '간판 검출 → 단어 검출' 연쇄로 AP@0.5 채점합니다.

eval_text_holdout.py 는 단어 검출기에 원본 사진을 통째로 넣어 AP 를 잽니다. 배포
파이프라인은 그렇게 돌지 않습니다 — 간판 검출기가 먼저 자르고, 단어 검출기는 그
크롭만 봅니다. 그래서 같은 단위(AP@0.5)로 두 값을 나란히 놓으면 **1단계가 2단계에
얼마를 깎아먹는지**가 그대로 나옵니다.

간판 검출기의 출력 자체는 채점하지 않으므로(중간 단계) 간판 GT 는 필요 없습니다.
AI Hub hold-out 은 단어 GT 가 있으니 연쇄 AP 를 여기서 잴 수 있습니다. GSV 에는
단어 GT 가 없어 같은 측정이 불가능합니다 — 거기서는 라인 회수율로 봅니다.

채점기·전처리·해상도는 eval_text_holdout.py 와 동일하게 재사용합니다. 바뀌는 것은
단어 검출기가 보는 입력(원본 사진 → 간판 크롭)뿐입니다.

Usage:
  .venv/Scripts/python.exe detection/eval_text_chain.py [--models yolo26x,yolov5x,frcnn,effdet] [--limit N]
출력: artifacts/gt/text_chain_ap50.csv
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from PIL import Image

from train_frcnn_kfold import compute_ap50
import eval_det_unified as U
from eval_text_holdout import MODELS, load_test, predict_frcnn_text

HERE = Path(__file__).resolve().parents[1]
SIGN_W = HERE / "artifacts" / "yolo26x_kfold" / "fold0" / "weights" / "best.pt"


def detect_signboards(items, conf: float, imgsz: int, pad: float, device: str = "0"):
    """배포 연쇄와 같은 간판 검출기·같은 설정. 사진당 [(x0,y0,x1,y1,score), ...]."""
    from ultralytics import YOLO
    model = YOLO(str(SIGN_W))
    out = []
    B = 8
    paths = [ip for ip, _, _ in items]
    for i in range(0, len(paths), B):
        batch = paths[i:i + B]
        res = model.predict([str(p) for p in batch], imgsz=imgsz, conf=conf,
                            iou=0.7, device=device, verbose=False)
        for p, r in zip(batch, res):
            W, H = Image.open(p).size
            boxes = []
            for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                x0, y0, x1, y1 = b
                dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
                boxes.append((max(0.0, x0 - dx), max(0.0, y0 - dy),
                              min(float(W), x1 + dx), min(float(H), y1 + dy), c))
            out.append(boxes)
        if (i // B) % 25 == 0:
            print(f"  [sign] {min(i + B, len(paths))}/{len(paths)}", flush=True)
    return out


def nms(boxes, iou_thr=0.6):
    """크롭이 겹치면 같은 단어가 두 번 나옵니다 — 원본 좌표계에서 한 번 정리합니다."""
    boxes = sorted(boxes, key=lambda b: -b[4])
    keep = []
    for b in boxes:
        ok = True
        for k in keep:
            ix0, iy0 = max(b[0], k[0]), max(b[1], k[1])
            ix1, iy1 = min(b[2], k[2]), min(b[3], k[3])
            iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
            inter = iw * ih
            if inter <= 0:
                continue
            ua = (b[2] - b[0]) * (b[3] - b[1]) + (k[2] - k[0]) * (k[3] - k[1]) - inter
            if ua > 0 and inter / ua > iou_thr:
                ok = False
                break
        if ok:
            keep.append(b)
    return keep


def words_in_crops(name, kind, wpath, res, items, sign_boxes, tmp: Path):
    """간판 크롭을 파일로 떨어뜨린 뒤 기존 예측 함수를 그대로 태웁니다(추론 설정 동일 보장)."""
    crop_items, owner = [], []          # owner[i] = (사진 index, crop 원점 x0,y0)
    for pi, ((ip, _, _), sbs) in enumerate(zip(items, sign_boxes)):
        if not sbs:
            continue
        im = Image.open(ip).convert("RGB")
        for ci, (x0, y0, x1, y1, _sc) in enumerate(sbs):
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            cp = tmp / f"{pi:06d}_{ci:03d}.jpg"
            im.crop((int(x0), int(y0), int(x1), int(y1))).save(cp, quality=95)
            crop_items.append((cp, [], None))
            owner.append((pi, x0, y0))
    print(f"  [{name}] {len(crop_items)} crops from {len(items)} photos", flush=True)
    if not crop_items:
        return [[] for _ in items]
    if kind == "ultra":
        cpreds = U.predict_ultra(wpath, crop_items, imgsz=res)
    elif kind == "frcnn":
        cpreds = predict_frcnn_text(wpath, crop_items)
    else:
        cpreds = U.predict_effdet(wpath, crop_items, img_size=res)
    per_photo = [[] for _ in items]
    for (pi, ox, oy), pr in zip(owner, cpreds):
        for (x0, y0, x1, y1, s) in pr:
            per_photo[pi].append((x0 + ox, y0 + oy, x1 + ox, y1 + oy, s))
    return [nms(p) for p in per_photo]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--score-thr", type=float, default=0.0)
    ap.add_argument("--sign-conf", type=float, default=0.25, help="배포 연쇄와 동일")
    ap.add_argument("--sign-imgsz", type=int, default=960, help="배포 연쇄와 동일")
    ap.add_argument("--pad", type=float, default=0.02, help="간판 크롭 여유")
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default="artifacts/gt/text_chain_ap50.csv")
    args = ap.parse_args()

    items = load_test(args.limit)
    gts = [g for _, g, _ in items]
    print(f"[test] {len(items)} images, {sum(len(g) for g in gts)} word boxes")
    print(f"[sign] {SIGN_W.relative_to(HERE)} conf={args.sign_conf} imgsz={args.sign_imgsz}")
    sign_boxes = detect_signboards(items, args.sign_conf, args.sign_imgsz, args.pad, args.device)
    n_sb = sum(len(s) for s in sign_boxes)
    n_cov = sum(1 for s in sign_boxes if s)
    print(f"[sign] {n_sb} signboards on {n_cov}/{len(items)} photos "
          f"({len(items) - n_cov} photos yield NO crop and therefore no word box at all)")

    rows = []
    for name in args.models.split(","):
        kind, wp, res = MODELS[name]
        wpath = HERE / wp
        if not wpath.exists():
            print(f"[{name}] 가중치 없음: {wpath}")
            continue
        tmp = Path(tempfile.mkdtemp(prefix=f"chain_{name}_"))
        try:
            preds = words_in_crops(name, kind, wpath, res, items, sign_boxes, tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        a = compute_ap50(preds, gts, score_thr=args.score_thr)
        rows.append((name, a, sum(len(p) for p in preds)))
        print(f"[{name}] chained AP@0.5 = {a:.4f}", flush=True)

    with open(HERE / args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "chain_AP50", "n_pred_boxes", "n_test_images", "n_gt_boxes",
                    "n_signboards", "photos_with_crop"])
        for name, a, n in rows:
            w.writerow([name, f"{a:.4f}", n, len(items), sum(len(g) for g in gts), n_sb, n_cov])
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
