#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PARSeq·SVTRv2 와 같은 학습 목록(단어+라인 75,612 / val 9,609)을 EasyOCR·TrOCR 형식으로 옮깁니다.

표 3 에서 EasyOCR·TrOCR-base 는 단어 크롭 62,464 만으로 학습돼 있었습니다. 다른 미세조정 행과
학습 데이터를 맞추려고, 같은 목록(`artifacts/str_baselines/lists/{train,val}.txt`)을 그대로
각 트레이너의 입력 형식으로 씁니다. 크롭 파일은 새로 만들지 않고 목록의 절대경로를 씁니다.

  TrOCR  : artifacts/ocr_training/signboard_v4_wl/labels.csv   (train / val / test=test_word.txt)
  EasyOCR: external/EasyOCR/trainer/all_data_signboard_v4_wl/{train,val}/labels.csv

Usage:
  .venv/Scripts/python.exe ocr/make_wordline_manifests.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
LISTS = HERE / "artifacts" / "str_baselines" / "lists"
TROCR_DIR = HERE / "artifacts" / "ocr_training" / "signboard_v4_wl"
EASY_DIR = HERE / "external" / "EasyOCR" / "trainer" / "all_data_signboard_v4_wl"


def read_list(name: str) -> list[tuple[str, str]]:
    rows = []
    for line in (LISTS / name).open(encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            continue
        path, text = line.split("\t", 1)
        if not Path(path).exists():
            raise FileNotFoundError(path)
        rows.append((path, text))
    return rows


def main() -> None:
    splits = {"train": read_list("train.txt"), "val": read_list("val.txt"),
              "test": read_list("test_word.txt")}
    for s, r in splits.items():
        print(f"[{s}] {len(r)} (공백 포함 라인 {sum(' ' in t for _, t in r)})")

    TROCR_DIR.mkdir(parents=True, exist_ok=True)
    with (TROCR_DIR / "labels.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "text", "split", "source_image", "source_image_id",
                    "annotation_id", "class", "x", "y", "w", "h"])
        for s, rows in splits.items():
            for p, t in rows:
                stem = Path(p).stem
                w.writerow([p, t, s, stem.split("__")[0], "", stem, "line" if " " in t else "word",
                            0, 0, 0, 0])
    print(f"[trocr] {TROCR_DIR / 'labels.csv'}")

    # EasyOCR 트레이너는 labels.csv 를 정규식 "^([^,]+)," 로 나누므로 파일명에 쉼표가 없어야 합니다.
    for s in ("train", "val"):
        d = EASY_DIR / s
        d.mkdir(parents=True, exist_ok=True)
        with (d / "labels.csv").open("w", encoding="utf-8", newline="") as f:
            f.write("filename,words\n")
            for p, t in splits[s]:
                assert "," not in p, p
                f.write(f"{p},{t}\n")
        print(f"[easyocr] {d / 'labels.csv'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
