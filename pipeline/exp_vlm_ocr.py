#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VLM×OCR 하이브리드가 OCR 지표(exact/CER/WAR)를 올리는가.

두 변형을 배포 OCR(run24, 69.9%/0.163/0.648)과 같은 프로토콜로 채점합니다:

  --mode solo   : gemma가 이미지만 보고 라인별로 읽음 (VLM 단독 OCR 기준선)
  --mode fix    : 배포 OCR 라인 + 이미지를 함께 주고 **오독 글자만 교정**.
                  라인 수·순서 유지, 확신 없으면 원문 유지, 새 라인 추가 금지
                  — 1~2글자 오독 버킷(전체 라인의 16%)을 겨냥하고,
                  라인 구조를 보존해 기존 라인 매칭 평가를 그대로 쓰기 위함.

출력: artifacts/ocr_gt/ab_v4_spacecat/ocr_{region}_vlm{mode}.csv (공식 run과 분리)
채점: eval_ocr_v2 모듈 임포트 (공식 프로토콜 동일).

Usage: .venv/Scripts/python.exe exp_vlm_ocr.py --mode fix [--model gemma3:12b]
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]

import sys as _sys; _sys.path[:0] = [str(HERE / _d) for _d in ("pipeline", "ocr", "vlm", "str_baselines", "detection", "data")]  # 폴더 재편 후 형제 모듈 import 경로
