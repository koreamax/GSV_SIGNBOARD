#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T7: GT-free crop-level ensemble re-selection (agreement + dictionary bonus).

The deployed ensemble picks per-box by raw cross-engine confidence — but the
three confidences live on different scales, and per-box scores were not
persisted for past runs. This selector re-chooses AT CROP LEVEL among the
already-saved engine outputs (easyocr / trocr / paddle / conf-ensemble),
using only deployable signals:

  score(c) = mean pairwise agreement with the 3 engine outputs
             + LAMBDA_DB * dictionary hit ratio (region OCR DB, exact key)

Agreement = 1 - normalized edit distance between whitespace/punct-collapsed
texts. No GT anywhere; weights are fixed a priori (no GSV tuning = no leakage).

Outputs a comparison table (official eval protocol via eval_ocr_v2 import):
  paddle-only / conf-ensemble raw / +snap (current official) /
  T7 variants / crop-level oracle / line-level oracle.

Usage:  python ocr_ensemble_select.py [--run 12] [--emit-run N]
        --emit-run N writes ocr_{region}_N_ensemble.csv with the winning
        deployable variant (T7 selection + gangnam/suwon guarded snap).
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]

import sys as _sys; _sys.path[:0] = [str(HERE / _d) for _d in ("pipeline", "ocr", "vlm", "str_baselines", "detection", "data")]  # 폴더 재편 후 형제 모듈 import 경로
