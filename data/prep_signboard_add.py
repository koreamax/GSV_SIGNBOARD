#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""누락 간판 보강 라벨링 준비 — 표 1 과 같은 사진 298장, 기존 정답 + 검출기 후보.

RAG 점검(09-30)에서 태그가 붙은 FP 69개 중 59개가 육안으로 실제 가게 간판이었고, 51개는 어떤 정답 간판과도
겹치지 않았습니다. 간판 정답에 빠진 간판을 채우려고 사진 전체를 다시 봅니다.

  작업 이미지 : 원본을 긴 변 2,400px 로 줄인 사진 (artifacts/gt/signboard_add/img/)
  고정 표시   : 기존 정답 간판 (표 1 의 fold 라벨, 수정 불가)
  후보(씨앗)  : 4개 검출기(YOLO26x · YOLOv5x · Faster R-CNN · EfficientDet-D0)의 out-of-fold 예측 중
                conf ≥ 0.25, 모델 간 IoU ≥ 0.5 로 묶고, 기존 정답과 IoU ≥ 0.3 이거나 후보의 70% 이상이
                정답 박스 안에 있으면(정답 간판의 일부) 뺀 것. `by` = 그 후보를 낸 모델 목록.

후보는 검출기가 찾은 것뿐이라, 네 모델이 모두 놓친 간판은 라벨러가 직접 그려야 합니다.

Usage:
  .venv/Scripts/python.exe data/prep_signboard_add.py
  .venv/Scripts/python.exe data/label_signboard.py      # → http://127.0.0.1:8778
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "detection"))
os.environ.setdefault("FOLD_ROOT", "artifacts/kfold_grouped")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import eval_det_unified as U  # noqa: E402

OUT = HERE / "artifacts" / "gt" / "signboard_add"
LONG = 2400
CONF = 0.25
MODELS = ["yolo26x", "yolov5x", "frcnn", "effdet"]
PREDS = ["det_unified_preds_grouped.json", "det_unified_preds_grouped_effdet.json",
         "det_unified_preds_grouped_v5x.json"]


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def inside(a, g) -> float:
    """a 의 면적 중 g 안에 들어가는 비율."""
    ix = max(0.0, min(a[2], g[2]) - max(a[0], g[0]))
    iy = max(0.0, min(a[3], g[3]) - max(a[1], g[1]))
    return ix * iy / max((a[2] - a[0]) * (a[3] - a[1]), 1e-9)


def main() -> None:
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    P = {}
    for f in PREDS:
        P.update(json.loads((HERE / "artifacts" / "gt" / f).read_text(encoding="utf-8")))
    items, seeds = [], {}
    for i in range(5):
        for j, (ip, gts, region) in enumerate(U.load_fold_val(i)):
            pid = ip.stem.replace("total__", "")                  # gangnam__12
            im = Image.open(ip).convert("RGB")
            W, H = im.size
            s = LONG / max(W, H)
            name = f"{pid}.jpg"
            if not (OUT / "img" / name).exists():
                im.resize((round(W * s), round(H * s)), Image.LANCZOS).save(OUT / "img" / name, quality=88)
            norm = lambda b: [round(b[0] / W, 6), round(b[1] / H, 6), round(b[2] / W, 6), round(b[3] / H, 6)]
            cands = []
            for m in MODELS:
                for b in P[f"{m}::{i}"][j]:
                    if b[4] >= CONF:
                        cands.append((list(b[:4]), b[4], m))
            cands.sort(key=lambda t: -t[1])
            clusters = []                                          # [box, conf, {models}]
            for box, c, m in cands:
                for cl in clusters:
                    if iou(box, cl[0]) >= 0.5:
                        cl[2].add(m)
                        break
                else:
                    clusters.append([box, c, {m}])
            keep = [cl for cl in clusters
                    if all(iou(cl[0], g) < 0.3 and inside(cl[0], g) < 0.7 for g in gts)]
            seeds[pid] = [{"box": norm(cl[0]), "by": sorted(cl[2]), "conf": round(cl[1], 3)} for cl in keep]
            items.append({"id": pid, "region": region, "img": name, "W": W, "H": H, "fold": i,
                          "fixed": [norm(g) for g in gts], "has_ocr_gt": True, "gt_text": ""})
    items.sort(key=lambda it: (it["region"], int(it["id"].split("__")[1])))
    (OUT / "items.json").write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    (OUT / "seeds.json").write_text(json.dumps(seeds, ensure_ascii=False), encoding="utf-8")
    n = sum(len(v) for v in seeds.values())
    by = {}
    for v in seeds.values():
        for s in v:
            by[len(s["by"])] = by.get(len(s["by"]), 0) + 1
    print(f"사진 {len(items)}장 · 기존 정답 {sum(len(it['fixed']) for it in items)}개 · 후보 {n}개 "
          f"(모델 수별 {dict(sorted(by.items()))}) → {OUT}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
