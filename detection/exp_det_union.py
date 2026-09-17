#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C3/D23: does a second detector (PaddleOCR DB) recover Suwon's lost lines?

Suwon's residual failures are lines that appear NOWHERE in the prediction
(vertical stacks, embossed low-contrast, calligraphy — 4.9/D22). CRAFT
threshold sweeps were negative, so try a structurally different detector.

Variants (all recognized by the SAME model, line-merged the same way):
  craft      : current deployed detection (baseline)
  db         : PaddleOCR DB boxes only
  union      : CRAFT boxes + DB boxes that don't overlap any CRAFT box (IoU-ish)

Line grouping/merging reuses run_ocr_only.easyocr_detect_lines' geometry rules
via a local reimplementation (boxes come from two sources, so grouping runs on
plain polygons here).

Usage: .venv/Scripts/python.exe exp_det_union.py [--region suwon]
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parents[1]

import sys as _sys; _sys.path[:0] = [str(HERE / _d) for _d in ("pipeline", "ocr", "vlm", "str_baselines", "detection", "data")]  # 폴더 재편 후 형제 모듈 import 경로
