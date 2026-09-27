#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GSV 단어 박스 AP@0.5 — 표 2 의 'Word-box AP (GSV)' 와 'Chained AP' 를 한 코드로 잽니다.

두 열의 차이가 **간판 검출이 단어 검출에 끼치는 손실**이 되려면, 간판 박스를 어디서
가져오느냐 말고는 모든 것이 같아야 합니다. 그래서 한 스크립트에서 두 출처를 똑같이 처리합니다.

  source=gt  : 사람이 그린 간판 폴리곤의 외접 사각형 (463개)    → 'Word-box AP (GSV)'
  source=det : 표 1 의 검출기(YOLO26x, group-aware 5-fold)의
               out-of-fold 예측, conf 0.25 (det_oof_grp.json)  → 'Chained AP'

두 출처 공통 처리:
  간판 박스 → 6% 여유 · 긴 변 1,400px(최대 4배) 컬러 크롭 (라벨링 화면과 동일 규칙)
  → 단어 검출기(모델별 학습 해상도, conf 0.001) → 원본 사진 좌표로 되돌림
  → 사진 단위 NMS(IoU 0.6) → 단어 GT(원본 좌표, 1,483박스) 와 AP@0.5

채점 대상 사진은 표 1 과 같은 **298장 전부**입니다. 단어 라벨이 없는 사진은 GT 0개로 두어,
허위 간판에서 나온 단어 예측이 오검출로 잡히게 합니다. 간판을 하나도 못 찾은 사진의 GT 는
전부 미검출로 계산하고(배포 거동), 그 사진들을 뺀 값도 따로 냅니다.

이전 eval_text_chain.py --dataset gsv 는 간판 검출에 fold0 가중치 하나를 전 사진에 써서
학습 때 본 사진 80% 가 섞였습니다. 이 스크립트가 그것을 대체합니다.

Usage:
  .venv/Scripts/python.exe detection/eval_wordbox_chain_gsv.py [--models ...] [--limit N]
출력: artifacts/gt/wordbox_chain_gsv_ap50.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "data"))

from train_frcnn_kfold import compute_ap50            # noqa: E402
import eval_det_unified as U                          # noqa: E402
from eval_text_holdout import MODELS, predict_frcnn_text   # noqa: E402
from eval_text_chain import nms                       # noqa: E402
import prep_wordbox_label as PREP                     # noqa: E402  (크롭 규칙을 그대로 공유)

GT = HERE / "artifacts" / "gt"
PHOTO = HERE / "artifacts" / "gsv_photo"
LABELS = GT / "wordbox" / "yolo" / "labels"
REGIONS = ("gangnam", "brooklyn", "suwon")


def crop_box(x0, y0, x1, y1, W, H):
    """라벨링 작업 이미지와 같은 규칙: 6% 여유, 긴 변 1,400px(최대 4배)."""
    bw, bh = x1 - x0, y1 - y0
    cx0 = max(0.0, x0 - bw * PREP.PAD)
    cy0 = max(0.0, y0 - bh * PREP.PAD)
    cx1 = min(float(W), x1 + bw * PREP.PAD)
    cy1 = min(float(H), y1 + bh * PREP.PAD)
    ix0, iy0, ix1, iy1 = int(cx0), int(cy0), int(cx1), int(cy1)
    scale = min(PREP.MAX_UP, PREP.TARGET / max(ix1 - ix0, iy1 - iy0))
    return ix0, iy0, ix1, iy1, scale


def load_photos(det_json: Path, limit: int | None):
    det = json.loads(det_json.read_text(encoding="utf-8"))
    keys = sorted(det)[:limit]
    gt_sb: dict[str, list] = {k: [] for k in keys}
    for r in REGIONS:
        for row in csv.DictReader(open(GT / f"gt_{r}.csv", encoding="utf-8-sig")):
            k = row["filename"].replace(".jpg", "")
            if k in gt_sb:
                xs = [float(row[f"x{i}"]) for i in (1, 2, 3, 4)]
                ys = [float(row[f"y{i}"]) for i in (1, 2, 3, 4)]
                gt_sb[k].append((min(xs), min(ys), max(xs), max(ys)))
    photos = []
    for k in keys:
        W, H = det[k]["size"]
        words = []
        lp = LABELS / f"{k}.txt"
        if lp.exists():
            for line in lp.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    _, xc, yc, w, h = (float(v) for v in line.split()[:5])
                    words.append(((xc - w / 2) * W, (yc - h / 2) * H,
                                  (xc + w / 2) * W, (yc + h / 2) * H))
        photos.append({"key": k, "region": k.split("__")[0], "W": W, "H": H,
                       "words": words, "gt": gt_sb[k],
                       "det": [tuple(b[:4]) for b in det[k]["boxes"]]})
    return photos


def build_crops(photos, tmp: Path):
    """두 출처의 크롭을 한 번에 만들고, 각 크롭이 어느 사진·출처·원점·배율인지 기록."""
    items, owner = [], []
    for pi, p in enumerate(photos):
        im = None
        for source in ("gt", "det"):
            for bi, (x0, y0, x1, y1) in enumerate(p[source]):
                ix0, iy0, ix1, iy1, s = crop_box(x0, y0, x1, y1, p["W"], p["H"])
                if ix1 - ix0 < 4 or iy1 - iy0 < 4:
                    continue
                if im is None:
                    region, n = p["key"].split("__")
                    im = Image.open(PHOTO / region / f"{n}.jpg").convert("RGB")
                nw, nh = max(1, round((ix1 - ix0) * s)), max(1, round((iy1 - iy0) * s))
                cp = tmp / f"{pi:04d}_{source}_{bi:02d}.jpg"
                im.crop((ix0, iy0, ix1, iy1)).resize((nw, nh), Image.LANCZOS).save(cp, quality=92)
                items.append((cp, [], None))
                owner.append((pi, source, ix0, iy0, (ix1 - ix0) / nw, (iy1 - iy0) / nh))
    return items, owner


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--det-json", default=str(GT / "det_oof_grp.json"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="artifacts/gt/wordbox_chain_gsv_ap50.csv")
    args = ap.parse_args()

    photos = load_photos(Path(args.det_json), args.limit)
    n_words = sum(len(p["words"]) for p in photos)
    n_gt_sb = sum(len(p["gt"]) for p in photos)
    n_det_sb = sum(len(p["det"]) for p in photos)
    no_det = [i for i, p in enumerate(photos) if not p["det"]]
    print(f"[GSV] 사진 {len(photos)}장 · 단어 GT {n_words}박스 · 간판 GT {n_gt_sb}개 · "
          f"간판 검출 {n_det_sb}개 · 간판 0개 검출 사진 {len(no_det)}장 · 장치 {U.DEVICE}")

    tmp = Path(tempfile.mkdtemp(prefix="wbchain_"))
    try:
        items, owner = build_crops(photos, tmp)
        print(f"[crops] {len(items)}개 (gt {sum(o[1] == 'gt' for o in owner)} / "
              f"det {sum(o[1] == 'det' for o in owner)})", flush=True)
        gts = [p["words"] for p in photos]
        rows = []
        for name in args.models.split(","):
            kind, wp, res = MODELS[name]
            wpath = HERE / wp
            if kind == "ultra":
                cpreds = U.predict_ultra(wpath, items, imgsz=res, conf=0.001)
            elif kind == "frcnn":
                cpreds = predict_frcnn_text(wpath, items)
            else:
                cpreds = U.predict_effdet(wpath, items, img_size=res)
            per = {s: [[] for _ in photos] for s in ("gt", "det")}
            for (pi, source, ox, oy, sx, sy), pr in zip(owner, cpreds):
                for (x0, y0, x1, y1, sc) in pr:
                    per[source][pi].append((ox + x0 * sx, oy + y0 * sy,
                                            ox + x1 * sx, oy + y1 * sy, sc))
            res_row = {"model": name}
            for source in ("gt", "det"):
                preds = [nms(pp) for pp in per[source]]
                res_row[f"{source}_all"] = compute_ap50(preds, gts, score_thr=0.0)
                for r in REGIONS:
                    idx = [i for i, p in enumerate(photos) if p["region"] == r]
                    res_row[f"{source}_{r}"] = compute_ap50([preds[i] for i in idx],
                                                            [gts[i] for i in idx], score_thr=0.0)
                if source == "det":
                    keep = [i for i in range(len(photos)) if i not in set(no_det)]
                    res_row["det_excl"] = compute_ap50([preds[i] for i in keep],
                                                       [gts[i] for i in keep], score_thr=0.0)
            rows.append(res_row)
            print(f"[{name}] 단어 AP(GT 간판) {res_row['gt_all']:.4f} "
                  f"(강남 {res_row['gt_gangnam']:.4f} · 브루클린 {res_row['gt_brooklyn']:.4f} · "
                  f"수원 {res_row['gt_suwon']:.4f})", flush=True)
            print(f"[{name}] 연쇄 AP(검출 간판) {res_row['det_all']:.4f} "
                  f"(강남 {res_row['det_gangnam']:.4f} · 브루클린 {res_row['det_brooklyn']:.4f} · "
                  f"수원 {res_row['det_suwon']:.4f}) · 간판 0개 사진 {len(no_det)}장 제외 "
                  f"{res_row['det_excl']:.4f}", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    cols = (["model", "gt_all"] + [f"gt_{r}" for r in REGIONS] + ["det_all"]
            + [f"det_{r}" for r in REGIONS] + ["det_excl"])
    with open(HERE / args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols + ["n_photos", "n_word_gt", "n_gt_signboards", "n_det_signboards",
                           "photos_no_det"])
        for r in rows:
            w.writerow([r["model"]] + [f"{r[c]:.4f}" for c in cols[1:]]
                       + [len(photos), n_words, n_gt_sb, n_det_sb, len(no_det)])
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
