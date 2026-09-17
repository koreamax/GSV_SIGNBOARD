#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""연쇄(end-to-end) 채점 — 탐지 출력으로 자른 크롭에서 OCR/태깅을 재측정합니다.

모듈별 점수(탐지 0.88 · OCR 78.6% · 태깅 75.7%)는 전부 **GT 크롭** 기준이라
탐지 실패가 전파되는 효과가 빠져 있습니다. 여기서는 `e2e_det_boxes.py` 가 만든
out-of-fold 탐지 박스로 자른 크롭(`artifacts/gt/crop_det`)의 결과를 GT에 맞춰
채점합니다.

**채점 규칙 (연쇄에서 1:1 대응이 깨지므로 명시가 필요합니다):**
  TP 크롭 (예측∩GT, IoU≥0.5) → 대응하는 GT 크롭의 정답과 비교. 정상 채점.
  FN     (탐지가 놓친 GT)     → 예측이 아예 없으므로 **오답 처리**. 연쇄 손실의 핵심.
  FP     (GT에 없는 예측)     → 정답이 없어 정확도 분모에서는 빼고, **별도로 개수와
                               '내용이 있는 FP' 비율을 보고**합니다. 지도 구축
                               관점에서 FP는 "없는 가게를 만들어내는" 비용이라
                               정확도에 섞지 않고 따로 읽는 편이 정확합니다.

  → recall_e2e = (TP 중 정답) / (전체 GT)   ... 파이프라인이 실제로 건진 비율
     acc_on_TP = (TP 중 정답) / (TP 개수)   ... 탐지가 성공했을 때의 성능
                                              (= 기존 GT 크롭 평가와 비교 가능)

Usage:
  .venv/Scripts/python.exe eval_e2e_cascade.py --ocr-run 30
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
GT_DIR = HERE / "artifacts" / "gt"
OCR_DIR = HERE / "artifacts" / "ocr_gt"
REGIONS = ("gangnam", "brooklyn", "suwon")
BS_N = chr(92) + "n"


def nk(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    return re.sub(r"[^0-9a-z가-힣]", "", s)


def load_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    enc = "utf-8-sig"
    return {r["image_name"]: (r.get("gt_text") or "")
            for r in csv.DictReader(path.open(encoding=enc))}


def gt_crop_name(photo: str, idx: int) -> str:
    return f"{photo}__crop_{idx:03d}"


def load_match(region: str, tag: str = "") -> list[dict]:
    p = GT_DIR / f"e2e_match_{region}{tag}.csv"
    return list(csv.DictReader(p.open(encoding="utf-8")))


def split_lines(txt: str) -> list[str]:
    return [l for l in (txt or "").replace(BS_N, "\n").split("\n") if l.strip()]


def line_score(pred: str, gold: str) -> tuple[int, int]:
    """라인 단위 그리디 1:1 매칭으로 (정확히 맞은 라인 수, 채점 대상 라인 수).

    eval_ocr_v2 와 같은 철학이지만 연쇄 채점에 필요한 최소 구현입니다
    (### don't-care 제외, 정규화 후 완전일치)."""
    g = [l for l in split_lines(gold) if "###" not in l and nk(l)]
    p = [nk(l) for l in split_lines(pred) if nk(l)]
    used, ok = set(), 0
    for gl in g:
        k = nk(gl)
        for i, pl in enumerate(p):
            if i in used:
                continue
            if pl == k:
                used.add(i); ok += 1
                break
    return ok, len(g)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr-run", default="30", help="탐지 크롭으로 돌린 OCR run 번호")
    ap.add_argument("--gt-run", default="24", help="GT 크롭 기준 비교용 run")
    ap.add_argument("--engine", default="paddle")
    ap.add_argument("--tag", default="",
                    help="탐지 구성 태그 (예: _lowconf, _union). 매칭표 선택")
    args = ap.parse_args()

    print(f"{'지역':10s} {'GT라인':>7s} {'TP크롭':>7s} {'FN크롭':>7s} {'FP크롭':>7s} | "
          f"{'연쇄 recall':>11s} {'TP위 정확도':>11s} {'GT크롭 기준':>11s}")
    TOT = defaultdict(int)
    for region in REGIONS:
        gt_txt = load_map(GT_DIR / f"ocr_{region}_gt.csv")
        det_pred = load_map(OCR_DIR / f"ocr_{region}_{args.ocr_run}_{args.engine}.csv")
        gtc_pred = load_map(OCR_DIR / f"ocr_{region}_{args.gt_run}_{args.engine}.csv")
        match = load_match(region, args.tag)

        ok_tp = n_tp_lines = 0          # TP 크롭에서 맞은 라인 / 그 크롭들의 GT 라인
        fn_lines = 0                    # 탐지가 놓쳐 통째로 잃은 GT 라인
        n_tp = n_fp = n_fn = 0
        fp_with_text = 0
        for m in match:
            if m["status"] == "TP":
                n_tp += 1
                g = gt_txt.get(gt_crop_name(m["photo"], int(m["gt_index"])), "")
                ok, n = line_score(det_pred.get(m["det_crop"], ""), g)
                ok_tp += ok; n_tp_lines += n
            elif m["status"] == "FN":
                n_fn += 1
                g = gt_txt.get(gt_crop_name(m["photo"], int(m["gt_index"])), "")
                _, n = line_score("", g)
                fn_lines += n
            else:
                n_fp += 1
                if split_lines(det_pred.get(m["det_crop"], "")):
                    fp_with_text += 1

        gt_lines = n_tp_lines + fn_lines
        # GT 크롭 기준(기존 프로토콜) 재계산 — 같은 채점기로 재서 비교 가능하게
        ok_g = n_g = 0
        for name, g in gt_txt.items():
            ok, n = line_score(gtc_pred.get(name, ""), g)
            ok_g += ok; n_g += n
        print(f"{region:10s} {gt_lines:7d} {n_tp:7d} {n_fn:7d} {n_fp:7d} | "
              f"{ok_tp/max(gt_lines,1)*100:10.1f}% {ok_tp/max(n_tp_lines,1)*100:10.1f}% "
              f"{ok_g/max(n_g,1)*100:10.1f}%")
        for k, v in (("gt_lines", gt_lines), ("ok_tp", ok_tp), ("tp_lines", n_tp_lines),
                     ("tp", n_tp), ("fp", n_fp), ("fn", n_fn),
                     ("fp_text", fp_with_text), ("ok_g", ok_g), ("n_g", n_g)):
            TOT[k] += v

    print(f"{'전체':10s} {TOT['gt_lines']:7d} {TOT['tp']:7d} {TOT['fn']:7d} "
          f"{TOT['fp']:7d} | {TOT['ok_tp']/max(TOT['gt_lines'],1)*100:10.1f}% "
          f"{TOT['ok_tp']/max(TOT['tp_lines'],1)*100:10.1f}% "
          f"{TOT['ok_g']/max(TOT['n_g'],1)*100:10.1f}%")
    print(f"\nFP 크롭 {TOT['fp']}개 중 텍스트가 나온 것 {TOT['fp_text']}개 "
          f"({TOT['fp_text']/max(TOT['fp'],1)*100:.0f}%) — 지도에 없는 항목을 "
          f"만들어낼 수 있는 건수입니다(정확도 분모에는 미포함).")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
