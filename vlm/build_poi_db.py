#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RAG 검색용 상호명 사전(L3) 구축 — OSM POI만으로는 커버리지가 너무 낮아서.

측정된 문제(2026-08 기준): GT 라인 중 **OSM 상호명에 존재하는 비율이 강남 9.0% /
브루클린 16.1% / 수원 1.2%** 입니다. 사전에 없는 상호는 RAG로 못 고치므로,
검색 코퍼스를 넓히는 것이 RAG 도입의 선행조건입니다.

레이어 (전부 배포 시점에 합법적으로 쓸 수 있는 공개 데이터 — DB_RULES.md L1 규칙 준수):
  L1 osm       : OpenStreetMap POI          (build_ocr_db.py 가 이미 생성)
  L3 kr_sangga : 소상공인시장진흥공단 상가(상권)정보  → 강남·수원  [수동 다운로드 필요]
  L3 nyc_lob   : NYC Legally Operating Businesses     → 브루클린   [자동 수집]
  L2 vocab     : signboard train 어휘        (일반명사 — 검색 기본값에선 제외)

**금지(변함없음)**: 평가 GT(ocr_{region}_gt.csv, tagging_gt.csv)에서 어휘 추출.

Usage:
  .venv/Scripts/python.exe build_poi_db.py --nyc                    # 브루클린 자동
  .venv/Scripts/python.exe build_poi_db.py --kr-csv "D:/소상공인_서울.csv" "D:/소상공인_경기.csv"
  .venv/Scripts/python.exe build_poi_db.py --merge                  # rag_{region}.csv 생성
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from build_ocr_db import BBOX, norm_key   # 지역 bbox·정규화는 기존 정의를 그대로 씁니다

HERE = Path(__file__).resolve().parents[1]

import sys as _sys; _sys.path[:0] = [str(HERE / _d) for _d in ("pipeline", "ocr", "vlm", "str_baselines", "detection", "data")]  # 폴더 재편 후 형제 모듈 import 경로
