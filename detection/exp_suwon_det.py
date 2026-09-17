#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Suwon detection-bottleneck sweep (D22 candidate) — CPU only (GPU is training).

Suwon line-merge failures: 29/181 GT lines appear NOWHERE in the prediction
(complete detection/recognition loss). Sweep detection-side remedies on the
line-merge pipeline and score with the official eval code:

  variants: 1x baseline / 2x upscale (LANCZOS) / low_text 0.3 / 2x + low_text 0.3

Recognition = v4_spacecat via paddle worker (--device cpu).
Usage: .venv/Scripts/python.exe exp_suwon_det.py [--region suwon]
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
