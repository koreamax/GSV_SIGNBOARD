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

import argparse
import csv
import sys
import time
from pathlib import Path

import eval_vlm_tagging as T

HERE = Path(__file__).resolve().parents[1]

import sys as _sys; _sys.path[:0] = [str(HERE / _d) for _d in ("pipeline", "ocr", "vlm", "str_baselines", "detection", "data")]  # 폴더 재편 후 형제 모듈 import 경로
