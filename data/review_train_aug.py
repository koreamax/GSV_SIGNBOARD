#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""review_train_aug.py — inspect what each OCR engine ACTUALLY saw in training.

Sibling of review_crops.py (which reviews the raw crops/labels). This app
re-runs each engine's real training-time pipeline on signboard_v3 train rows
so augmentation problems can be checked one by one:

  * TrOCR  : train_textinthewild_ocr.build_trocr_augment()  (the real code path)
             + the 384x384 square resize the ViT processor applies.
  * Paddle : DecodeImage(BGR) -> RecConAug(prob .5, ext 2, label += NO SPACE)
             -> RecAug (tia/crop/blur/hsv/jitter/noise/REVERSE 40%)
             -> final 48x320 network view. Real ppocr classes are imported from
             the venv PaddleOCR repo; if that import fails only RecConAug is
             reproduced locally (faithful port) and RecAug variants are skipped.
  * EasyOCR: no augmentation in signboard_v3.yaml (contrast_adjust 0.0) —
             shown as the trainer input: grayscale, keep-AR resize to H=64,
             right-pad to 600 (NormalizePAD).

Usage:
  .venv/Scripts/python.exe review_train_aug.py            # -> http://localhost:8124
  .venv/Scripts/python.exe review_train_aug.py --scan     # dataset audit only (no UI)
  .venv/Scripts/python.exe review_train_aug.py --selftest # render one sample to scratch, no server
  .venv/Scripts/python.exe review_train_aug.py --export   # aug_review_bad.csv from marks

Verdicts append to artifacts/ocr_training/signboard_v3/aug_review_marks.csv.
Keys: G/→ good  B/X bad  S skip  ← prev  Z undo  R re-roll augmentation.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
import sys
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parents[1]

import sys as _sys; _sys.path[:0] = [str(HERE / _d) for _d in ("pipeline", "ocr", "vlm", "str_baselines", "detection", "data")]  # 폴더 재편 후 형제 모듈 import 경로
