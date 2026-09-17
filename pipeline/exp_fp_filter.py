#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""허위 탐지(FP) 되거르기 — union 으로 올린 recall 을 지키면서 허위 POI 를 줄인다.

4.16/D32 에서 모델 union 은 탐지 recall 을 0.825 → 0.932 로 올렸지만 FP 크롭이
124 → 335 개가 됐고, FP 의 90% 가 업종 태그까지 받아 **허위 POI** 가 됩니다.
문제의 본질은 탐지기가 아니라 **걸러낼 단계가 파이프라인에 없다는 것**입니다.

인식 단계에서 이미 얻은 신호로 사후 필터를 검증합니다 (추가 GPU 없음):
  conf   : 탐지 신뢰도
  chars  : OCR 이 뽑은 글자 수 (간판이 아니면 글자가 안 나오거나 부스러기만 남음)
  lines  : 인식된 라인 수
  rag    : 지역 상호 사전에서 유사 상호가 검색되는가 (허위 크롭의 잡음 문자열은
           실제 상호와 닮지 않음)

각 규칙에 대해 **연쇄 recall 을 얼마나 잃고 허위 POI 를 얼마나 줄이는지**를 함께
봅니다. 필터가 TP 를 같이 버리면 의미가 없기 때문입니다.
"""
from __future__ import annotations
import csv, sys
from pathlib import Path
import eval_e2e_cascade as C
import rag_retrieve as RAG

HERE = Path(__file__).resolve().parents[1]

import sys as _sys; _sys.path[:0] = [str(HERE / _d) for _d in ("pipeline", "ocr", "vlm", "str_baselines", "detection", "data")]  # 폴더 재편 후 형제 모듈 import 경로
