#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Line-strip tuning experiments on the deployed pipeline (C-track follow-ups).

Detection (CRAFT ∪ DB, D23) is expensive and identical across these
experiments, so it is run once and cached to artifacts/ocr_gt/det_cache.json.

  --mode pad       : strip padding sweep (targets trailing-character truncation
                     — 'BBQ'->'BB', 'MLB'->'B' in the 1-2 char error bucket)
  --mode ensemble  : LINE-level multi-model recognition + rec_score selection.
                     Unlike the failed crop-level T7/C1 (candidates had
                     different granularity), every candidate here reads the SAME
                     line strip, so agreement/confidence are meaningful.

Usage:
  .venv/Scripts/python.exe exp_line_tuning.py --mode pad
  .venv/Scripts/python.exe exp_line_tuning.py --mode ensemble [--pad 0.18]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parents[1]

import sys as _sys; _sys.path[:0] = [str(HERE / _d) for _d in ("pipeline", "ocr", "vlm", "str_baselines", "detection", "data")]  # 폴더 재편 후 형제 모듈 import 경로
