#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""연쇄(표 5) OCR 행의 line exact 와 CER — 구성 A·B, 지역별 + 전체.

eval_e2e_cascade 는 line exact 만 냅니다. 여기서는 같은 매칭표(298장)·같은 출력으로 표 3·4 와 **같은 채점기**
(eval_ocr_v2.eval_engine: --mask-phone, 브루클린 영어 전용, FP 라인 편집 포함)를 돌려 CER 까지 냅니다.
전체 값은 표 3·4 처럼 세 지역의 라인·글자를 합산한 micro 평균입니다.

  GT 간판 : 사람이 그린 간판 crop 의 교정 출력 (A run 131 gtfixA, B run 132 gtfixB), 매칭표에 있는 298장
  TP      : 검출 간판 crop 의 교정 출력 (A run 134 g26fix, B run 136 g5fix) ↔ 매칭된 정답 간판의 정답
  연쇄 recall : TP + 놓친 간판(FN, 출력 없음 = 그 라인 글자 전부 삭제로 셈)

Usage:
  .venv/Scripts/python.exe pipeline/cascade_cer.py
출력: artifacts/gt/cascade_cer.csv
"""
from __future__ import annotations

from pathlib import Path as _P
import sys as _sys; _sys.path[:0] = [str(_P(__file__).resolve().parents[1] / "pipeline")]

import csv
import sys

import eval_e2e_cascade as C

CFG = {"A": dict(gt=("131", "gtfixA"), det=("134", "g26fix")),
       "B": dict(gt=("132", "gtfixB"), det=("136", "g5fix"))}
FIELDS = ("edit", "fp_edit", "chars", "exact", "lines")


def add(tot, a):
    for f in FIELDS:
        tot[f] = tot.get(f, 0) + getattr(a, f)


def fmt(t):
    return t["exact"] / t["lines"] * 100, (t["edit"] + t["fp_edit"]) / t["chars"]


def main() -> None:
    E = C.scorer()
    assert E.ARGS.keep_fp, "표 3·4 와 같게 FP 편집 포함이어야 합니다"
    rows = []
    for cfg, v in CFG.items():
        tot = {k: {} for k in ("gt", "tp", "rec")}
        for region in C.REGIONS:
            E.EN_ONLY = region in E.EN_ONLY_REGIONS
            gold = E.load_csv_map(C.GT_DIR / f"ocr_{region}_gt.csv")
            gtp = E.load_csv_map(C.OCR_DIR / f"ocr_{region}_{v['gt'][0]}_{v['gt'][1]}.csv")
            det = E.load_csv_map(C.OCR_DIR / f"ocr_{region}_{v['det'][0]}_{v['det'][1]}.csv")
            match = C.load_match(region, "_grp")
            photos = {m["photo"] for m in match}
            g_same = {k: x for k, x in gold.items() if k.split("__crop_")[0] in photos}
            tp_gold, tp_pred, rec_pred = {}, {}, {}
            for m in match:
                if m["status"] not in ("TP", "FN"):
                    continue
                key = C.gt_crop_name(m["photo"], int(m["gt_index"]))
                if key not in gold:
                    continue
                if m["status"] == "TP":
                    tp_gold[key] = gold[key]
                    tp_pred[key] = det.get(m["det_crop"], "")
                    rec_pred[key] = tp_pred[key]
                else:
                    rec_pred[key] = ""
            rec_gold = {k: gold[k] for k in rec_pred}
            per = {"gt": E.eval_engine(g_same, gtp, []), "tp": E.eval_engine(tp_gold, tp_pred, []),
                   "rec": E.eval_engine(rec_gold, rec_pred, [])}
            for k, a in per.items():
                add(tot[k], a)
                ex, cer = fmt({f: getattr(a, f) for f in FIELDS})
                rows.append([cfg, k, region, a.lines, f"{ex:.1f}", f"{cer:.3f}"])
        for k in ("gt", "tp", "rec"):
            ex, cer = fmt(tot[k])
            rows.append([cfg, k, "ALL", tot[k]["lines"], f"{ex:.1f}", f"{cer:.3f}"])

    label = {"gt": "GT 간판", "tp": "TP", "rec": "연쇄 recall"}
    print(f"{'구성':4s} {'행':10s} {'강남':>14s} {'브루클린':>14s} {'수원':>14s} {'전체':>14s}   (line exact % / CER)")
    for cfg in CFG:
        for k in ("tp", "gt", "rec"):
            cells = {r[2]: f"{r[4]} / {r[5]}" for r in rows if r[0] == cfg and r[1] == k}
            print(f"{cfg:4s} {label[k]:10s} " + " ".join(f"{cells[x]:>14s}" for x in
                                                          ("gangnam", "brooklyn", "suwon", "ALL")))
    out = C.GT_DIR / "cascade_cer.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "row", "region", "lines", "line_exact", "CER"])
        w.writerows(rows)
    print(f"[saved] {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
