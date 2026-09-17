#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""연쇄 평가 G — 탐지 크롭에서 의미 태깅까지 (3모듈 전체 연쇄).

4.15의 태깅 평가는 GT 크롭 411개 기준이라 탐지 손실이 빠져 있습니다. 여기서는
`e2e_det_boxes.py` 가 만든 탐지 크롭(502개)의 OCR 결과로 태깅을 돌리고,
`e2e_match_{region}.csv` 로 정답과 맞춰 채점합니다.

**채점 규칙** (4.16 OCR 연쇄와 동일한 철학):
  TP → 대응 GT 크롭의 tagging_gt 정답과 비교 (eval_tag=1 인 건만)
  FN → 탐지가 놓쳐 태그를 못 붙임 → **오답**
  FP → 정답이 없으므로 정확도 분모에서 제외하되, **"태그가 붙은 FP" = 허위 POI
       생성 건수**로 따로 보고. 지도 구축에서는 이게 누락보다 나쁠 수 있습니다.

  연쇄 tag recall = (TP 중 정답) / (전체 GT 태깅 대상)
  TP 위 정확도    = (TP 중 정답) / (TP 중 태깅 대상)   ← GT 크롭 평가와 비교 가능

Usage:
  .venv/Scripts/python.exe eval_e2e_tagging.py --ocr-run 31 --model gemma3:12b
"""
from __future__ import annotations

from pathlib import Path as _P
import sys as _sys; _sys.path[:0] = [str(_P(__file__).resolve().parents[1] / _d) for _d in ("pipeline", "ocr", "vlm", "str_baselines", "detection", "data")]  # 형제 폴더 모듈 import (로컬 import 보다 먼저)

import argparse
import csv
import sys
import time
from pathlib import Path

import eval_vlm_tagging as T

HERE = Path(__file__).resolve().parents[1]

import sys as _sys; _sys.path[:0] = [str(HERE / _d) for _d in ("pipeline", "ocr", "vlm", "str_baselines", "detection", "data")]  # 폴더 재편 후 형제 모듈 import 경로
GT_DIR = HERE / "artifacts" / "gt"
OCR_DIR = HERE / "artifacts" / "ocr_gt"
REGIONS = ("gangnam", "brooklyn", "suwon")
BS_N = chr(92) + "n"


def load_ocr_run(run: str, engine: str = "paddle") -> dict[str, str]:
    out = {}
    for region in REGIONS:
        p = OCR_DIR / f"ocr_{region}_{run}_{engine}.csv"
        if not p.exists():
            continue
        for r in csv.DictReader(p.open(encoding="utf-8-sig")):
            out[r["image_name"]] = (r.get("gt_text") or "").replace(BS_N, " / ")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr-run", default="31", help="탐지 크롭 OCR run (하이브리드 합성본)")
    ap.add_argument("--model", default="gemma3:12b")
    ap.add_argument("--retriever", choices=["lexical", "hybrid"], default="hybrid")
    ap.add_argument("--rag-k", type=int, default=5)
    ap.add_argument("--no-abstain", action="store_true", default=True)
    ap.add_argument("--out", default="artifacts/gt/e2e_tagging_results.csv")
    args = ap.parse_args()

    if args.retriever == "hybrid":
        import rag_hybrid as HINT
    else:
        import rag_retrieve as HINT

    gt_rows = {r["image_name"]: r for r in T.load_gt()}
    tags = sorted({r["tag"] for r in gt_rows.values() if r["eval_tag"] == "1"})
    ocr = load_ocr_run(args.ocr_run)

    detail, t0 = [], time.time()
    stat = {r: {"tp_ok": 0, "tp_n": 0, "fn_n": 0, "fp": 0, "fp_tagged": 0}
            for r in REGIONS}

    work = []
    for region in REGIONS:
        for m in csv.DictReader((GT_DIR / f"e2e_match_{region}.csv").open(encoding="utf-8")):
            work.append((region, m))
    print(f"[E2E-TAG] {len(work)}건 (모델={args.model}, 검색={args.retriever}, "
          f"OCR run={args.ocr_run})")

    for i, (region, m) in enumerate(work, 1):
        st = m["status"]
        gt_name = (f"{m['photo']}__crop_{int(m['gt_index']):03d}"
                   if m["gt_index"] else "")
        g = gt_rows.get(gt_name)

        if st == "FN":
            # 탐지가 놓친 간판 — 예측 자체가 없으므로 오답
            if g and g["eval_tag"] == "1":
                stat[region]["fn_n"] += 1
            continue

        text = ocr.get(m["det_crop"], "")
        hints = ""
        if text:
            hints = HINT.get(region).hint_block(
                [l for l in text.split(" / ") if l.strip()],
                k=args.rag_k, allowed_tags=set(tags))
        try:
            resp = T.ask(args.model, T.build_prompt(tags, text, hints,
                                                    args.no_abstain), None)
            pn, pt = T.parse(resp)
        except Exception:
            pn, pt = "", ""

        if st == "TP" and g and g["eval_tag"] == "1":
            stat[region]["tp_n"] += 1
            if pt == g["tag"]:
                stat[region]["tp_ok"] += 1
        elif st == "FP":
            stat[region]["fp"] += 1
            if pt and pt != "unknown":
                stat[region]["fp_tagged"] += 1

        detail.append({"region": region, "det_crop": m["det_crop"], "status": st,
                       "gt_crop": gt_name, "gt_tag": g["tag"] if g else "",
                       "pred_name": pn, "pred_tag": pt, "ocr_text": text[:120]})
        if i % 50 == 0:
            el = time.time() - t0
            print(f"  {i}/{len(work)}  {el/i:.1f}s/건  "
                  f"ETA {(len(work)-i)*el/i/60:.0f}분", flush=True)

    print(f"\n{'지역':10s} {'GT대상':>7s} {'TP':>5s} {'FN':>5s} {'FP':>5s} | "
          f"{'연쇄 recall':>11s} {'TP위 정확도':>11s} {'허위POI':>8s}")
    T_ok = T_n = T_fn = T_fp = T_ft = 0
    for region in REGIONS:
        s = stat[region]
        denom = s["tp_n"] + s["fn_n"]
        print(f"{region:10s} {denom:7d} {s['tp_n']:5d} {s['fn_n']:5d} {s['fp']:5d} | "
              f"{s['tp_ok']/max(denom,1)*100:10.1f}% "
              f"{s['tp_ok']/max(s['tp_n'],1)*100:10.1f}% "
              f"{s['fp_tagged']:4d}/{s['fp']:<4d}")
        T_ok += s["tp_ok"]; T_n += s["tp_n"]; T_fn += s["fn_n"]
        T_fp += s["fp"]; T_ft += s["fp_tagged"]
    D = T_n + T_fn
    print(f"{'전체':10s} {D:7d} {T_n:5d} {T_fn:5d} {T_fp:5d} | "
          f"{T_ok/max(D,1)*100:10.1f}% {T_ok/max(T_n,1)*100:10.1f}% "
          f"{T_ft:4d}/{T_fp:<4d}")
    print(f"\n허위 POI: 탐지 오류 {T_fp}건 중 {T_ft}건에 업종 태그가 붙었습니다 "
          f"({T_ft/max(T_fp,1)*100:.0f}%) — 지도에 없는 가게가 등록될 수 있는 건수입니다.")

    out = Path(args.out)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["region", "det_crop", "status", "gt_crop",
                                          "gt_tag", "pred_name", "pred_tag", "ocr_text"])
        w.writeheader(); w.writerows(detail)
    print(f"[saved] {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
