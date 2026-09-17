#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D19: line-merge pipeline experiment — feed whole detected LINES to Paddle.

The per-box pipeline slices signs into single-word CRAFT boxes, so the v4
spacecat model's multi-word/space ability never fires (4.8). Here each detected
line's boxes are merged into ONE strip (union bbox, pad 0.12 — same detector,
same filters as run_ocr_only) and recognized whole:

  A) per-box Paddle v3      = official run12 files (baseline, no recompute)
  B) line-merge Paddle v3   -> expects space-less multi-word outputs
  C) line-merge Paddle v4   -> spaces should appear (trained with gap+space)

Korean regions only (gangnam, suwon) — brooklyn uses the English model.
Outputs: artifacts/ocr_gt/ab_v4_spacecat/ocr_{region}_linemerge_{v3,v4}.csv
Eval: official eval_ocr_v2 code path (module import), exact/CER/WAR/contain.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

HERE = Path(__file__).resolve().parents[1]

import sys as _sys; _sys.path[:0] = [str(HERE / _d) for _d in ("pipeline", "ocr", "vlm", "str_baselines", "detection", "data")]  # 폴더 재편 후 형제 모듈 import 경로
