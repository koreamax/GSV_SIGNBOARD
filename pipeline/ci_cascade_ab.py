#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""연쇄 구성 A(YOLO26x 단어 검출) vs B(YOLOv5x) 차이의 95% 신뢰구간 — 사진 단위 페어 부트스트랩.

A 와 B 는 간판 검출(같은 470 크롭 · 같은 매칭표)을 공유하고 단어 검출기만 다르므로, 같은 사진에서 나온
결과끼리 짝지어 비교합니다. 한 사진의 간판·라인은 서로 독립이 아니라서 **사진 단위로** 재표집합니다
(298장, 2,000회, percentile). 채점은 eval_e2e_cascade 의 score_crop(표 3·4 와 같은 채점기)과
eval_e2e_tagging 과 같은 규칙(eval_tag=1 만, 기권은 오답)을 그대로 씁니다.

  OCR · GT 크롭   : run 131 gtfixA vs run 132 gtfixB (사람이 그린 간판, 검출 완벽)
  OCR · TP 위     : run 134 g26fix vs run 136 g5fix  (제대로 찾은 간판 374개)
  OCR · 연쇄 recall: 위와 같은 분자, 분모에 놓친 간판의 라인 포함
  태깅 · TP 위 / 연쇄 recall / 허위 POI : e2e_tagging_final_{A,B}.csv

Usage:
  .venv/Scripts/python.exe pipeline/ci_cascade_ab.py [--n-boot 2000]
"""
from __future__ import annotations

from pathlib import Path as _P
import sys as _sys; _sys.path[:0] = [str(_P(__file__).resolve().parents[1] / _d) for _d in ("pipeline", "vlm")]

import argparse
import csv
import random
import sys
from collections import defaultdict

import eval_e2e_cascade as C

GT_DIR, OCR_DIR, REGIONS = C.GT_DIR, C.OCR_DIR, C.REGIONS
CFG = {"A": dict(gt=("131", "gtfixA"), det=("134", "g26fix"), tag="e2e_tagging_final_A.csv"),
       "B": dict(gt=("132", "gtfixB"), det=("136", "g5fix"), tag="e2e_tagging_final_B.csv")}


def load_tag_gt() -> dict[str, dict]:
    """eval_e2e_tagging 과 같은 로더 — 정답 태그를 canon() 으로 정규화합니다(원문 비교는 틀림)."""
    import eval_vlm_tagging as T
    return {r["image_name"]: r for r in T.load_gt()}


def per_photo() -> dict[str, dict]:
    """사진별 분자·분모. P[photo][metric] = [A분자, B분자, 분모] (허위 POI 는 [A건수, B건수, FP 수])."""
    E = C.scorer()
    tg = load_tag_gt()
    P = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    for region in REGIONS:
        gold = E.load_csv_map(GT_DIR / f"ocr_{region}_gt.csv")
        match = C.load_match(region, "_grp")
        photos = {m["photo"] for m in match}
        pred = {k: {"gt": E.load_csv_map(OCR_DIR / f"ocr_{region}_{v['gt'][0]}_{v['gt'][1]}.csv"),
                    "det": E.load_csv_map(OCR_DIR / f"ocr_{region}_{v['det'][0]}_{v['det'][1]}.csv")}
                for k, v in CFG.items()}
        # OCR · GT 크롭 (같은 298장)
        for key, g in gold.items():
            ph = key.split("__crop_")[0]
            if ph not in photos:
                continue
            (a, n), (b, _) = (C.score_crop(pred[c]["gt"].get(key, ""), g, region) for c in "AB")
            P[ph]["ocr_gt"][0] += a; P[ph]["ocr_gt"][1] += b; P[ph]["ocr_gt"][2] += n
        # OCR · TP 위 / 연쇄 recall
        for m in match:
            ph = m["photo"]
            if m["status"] == "TP":
                g = gold.get(C.gt_crop_name(ph, int(m["gt_index"])), "")
                (a, n), (b, _) = (C.score_crop(pred[c]["det"].get(m["det_crop"], ""), g, region) for c in "AB")
                for met in ("ocr_tp", "ocr_rec"):
                    P[ph][met][0] += a; P[ph][met][1] += b; P[ph][met][2] += n
            elif m["status"] == "FN":
                g = gold.get(C.gt_crop_name(ph, int(m["gt_index"])), "")
                _, n = C.score_crop("", g, region)
                P[ph]["ocr_rec"][2] += n
                t = tg.get(C.gt_crop_name(ph, int(m["gt_index"])))
                if t and t["eval_tag"] == "1":
                    P[ph]["tag_rec"][2] += 1
    # 태깅
    rows = {c: {r["det_crop"]: r for r in csv.DictReader((GT_DIR / v["tag"]).open(encoding="utf-8"))}
            for c, v in CFG.items()}
    for region in REGIONS:
        for m in C.load_match(region, "_grp"):
            ph = m["photo"]
            if m["status"] == "TP":
                t = tg.get(C.gt_crop_name(ph, int(m["gt_index"])))
                if not (t and t["eval_tag"] == "1"):
                    continue
                ok = [int(rows[c][m["det_crop"]]["pred_tag"] == t["tag"]) for c in "AB"]
                for met in ("tag_tp", "tag_rec"):
                    P[ph][met][0] += ok[0]; P[ph][met][1] += ok[1]; P[ph][met][2] += 1
            elif m["status"] == "FP":
                tagged = [int(bool(rows[c][m["det_crop"]]["pred_tag"]) and
                              rows[c][m["det_crop"]]["pred_tag"] != "unknown") for c in "AB"]
                P[ph]["poi"][0] += tagged[0]; P[ph]["poi"][1] += tagged[1]; P[ph]["poi"][2] += 1
    return P


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260929)
    args = ap.parse_args()

    P = per_photo()
    photos = sorted(P)
    names = [("ocr_gt", "OCR line exact · GT 간판"), ("ocr_tp", "OCR line exact · TP 위"),
             ("ocr_rec", "OCR line exact · 연쇄 recall"), ("tag_tp", "태깅 정확도 · TP 위"),
             ("tag_rec", "태깅 정확도 · 연쇄 recall"), ("poi", "허위 POI 비율 (FP 중)")]
    rng = random.Random(args.seed)
    boots = [[rng.choice(photos) for _ in photos] for _ in range(args.n_boot)]
    print(f"사진 {len(photos)}장 · 사진 단위 페어 부트스트랩 {args.n_boot}회 · 차이 = A − B (%p)\n")
    print(f"{'지표':28s} {'A':>7s} {'B':>7s} {'분모':>5s} {'A−B':>7s}   95% CI           판정")
    out = []
    for met, label in names:
        def rate(sample, i):
            num = sum(P[p][met][i] for p in sample)
            den = sum(P[p][met][2] for p in sample)
            return num / den * 100 if den else 0.0
        a, b = rate(photos, 0), rate(photos, 1)
        den = sum(P[p][met][2] for p in photos)
        diffs = sorted(rate(s, 0) - rate(s, 1) for s in boots)
        lo, hi = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs)) - 1]
        verdict = "유의" if lo > 0 or hi < 0 else "미판정"
        print(f"{label:28s} {a:6.1f}% {b:6.1f}% {den:5d} {a - b:+6.1f}   [{lo:+.1f}, {hi:+.1f}]   {verdict}")
        out.append([met, f"{a:.2f}", f"{b:.2f}", den, f"{a - b:+.2f}", f"{lo:+.2f}", f"{hi:+.2f}", verdict])
    path = GT_DIR / "ci_cascade_ab.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "A", "B", "denominator", "A_minus_B", "ci_lo", "ci_hi", "verdict"])
        w.writerows(out)
    print(f"\n[saved] {path}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
