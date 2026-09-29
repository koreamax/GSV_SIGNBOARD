#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VLM 교정 재현성 — 같은 입력·같은 설정(temperature 0)으로 두 번 돌린 결과가 얼마나 같은지.

표 4 의 Gemma 4 · fix(ocr_*_vlmfix_svtr_gemma4_31b.csv)와, 같은 명령에 출력 이름만 바꿔 다시 돌린
ocr_*_vlmfix_svtr_gemma4_31b_rep2.csv 를 크롭·라인 단위로 비교하고, 둘 다 표 3·4 채점기로 잽니다.
출력 차이가 연쇄 A vs B 차이(1~2%p)와 비슷하면, 그 차이는 VLM 자체의 흔들림으로도 설명됩니다.

Usage:
  .venv/Scripts/python.exe pipeline/vlm_repro_check.py [--a SUFFIX] [--b SUFFIX]
"""
from __future__ import annotations

from pathlib import Path as _P
import sys as _sys; _sys.path[:0] = [str(_P(__file__).resolve().parents[1] / "pipeline")]

import argparse
import sys

import eval_e2e_cascade as C

VDIR = C.OCR_DIR / "ab_v4_spacecat"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="vlmfix_svtr_gemma4_31b")
    ap.add_argument("--b", default="vlmfix_svtr_gemma4_31b_rep2")
    args = ap.parse_args()
    E = C.scorer()
    tot = dict(crops=0, same_crop=0, lines=0, ok_a=0, ok_b=0, flip_ab=0, flip_ba=0)
    for region in C.REGIONS:
        gold = E.load_csv_map(C.GT_DIR / f"ocr_{region}_gt.csv")
        A = E.load_csv_map(VDIR / f"ocr_{region}_{args.a}.csv")
        B = E.load_csv_map(VDIR / f"ocr_{region}_{args.b}.csv")
        if not B:
            raise SystemExit(f"[오류] 재실행 결과 없음: ocr_{region}_{args.b}.csv")
        r = dict(crops=0, same_crop=0, lines=0, ok_a=0, ok_b=0)
        for key, g in gold.items():
            ea, n = C.score_crop(A.get(key, ""), g, region)
            eb, _ = C.score_crop(B.get(key, ""), g, region)
            if n == 0:
                continue
            r["crops"] += 1
            r["same_crop"] += int(C.nk(A.get(key, "")) == C.nk(B.get(key, "")))
            r["lines"] += n; r["ok_a"] += ea; r["ok_b"] += eb
            tot["flip_ab"] += max(0, ea - eb); tot["flip_ba"] += max(0, eb - ea)
        for k, v in r.items():
            tot[k] += v
        print(f"[{region}] 크롭 {r['crops']} 중 출력 동일 {r['same_crop']} · line exact "
              f"{r['ok_a'] / r['lines'] * 100:.1f}% vs {r['ok_b'] / r['lines'] * 100:.1f}%")
    print(f"\n[전체] 크롭 {tot['crops']} 중 출력이 글자 그대로 같은 것 {tot['same_crop']} "
          f"({tot['same_crop'] / tot['crops'] * 100:.1f}%)")
    print(f"[전체] line exact {tot['ok_a'] / tot['lines'] * 100:.1f}% (원본) vs "
          f"{tot['ok_b'] / tot['lines'] * 100:.1f}% (재실행) · {tot['lines']} line")
    print(f"[전체] 원본만 맞은 line {tot['flip_ab']} · 재실행만 맞은 line {tot['flip_ba']} "
          f"(크롭 단위 순증감 합)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
