#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Official line-merge OCR pipeline (D19/D20/D23 — adopted protocol).

Per-box slicing was the dominant bottleneck (4.9): merging each detected line's
boxes into ONE strip and recognizing it whole lifts Paddle from exact 47.6% to
~65% global. Detector union (D23) adds another ~+2%p:

  detection : EasyOCR(CRAFT), same params/filters as run_ocr_only run12
              UNION PaddleOCR DB boxes that overlap no CRAFT box (--no-det-union
              disables). CRAFT and DB fail on different things — DB alone is
              worse, the union is better in all three regions.
  merge     : per line (y-center clustering), union bbox + pad 0.12 -> one strip
  recognize : PaddleOCR via isolated worker (paddle_rec_worker.py)
              gangnam/suwon -> v5_lines (real line crops + spacecat — 4.10)
              brooklyn      -> en_PP-OCRv5_mobile_rec (pretrained English)

Emits artifacts/ocr_gt/ocr_{region}_{run}_paddle.csv (official run files).
EasyOCR/TrOCR remain per-box engines (run_ocr_only) and are kept only as
reference/oracle candidates — the deployed output is this file alone (D20).

Usage: .venv/Scripts/python.exe run_ocr_line.py --run 22 [--regions ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

HERE = Path(__file__).resolve().parents[1]

import sys as _sys; _sys.path[:0] = [str(HERE / _d) for _d in ("pipeline", "ocr", "vlm", "str_baselines", "detection", "data")]  # 폴더 재편 후 형제 모듈 import 경로
