#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STR 비교군(PARSeq·SVTRv2) 학습용 데이터 준비 — signboard_v3 분할 그대로.

Paddle v5_lines 와 **동일한 학습 혼합**(단어 train 62,464 + 실라인 train 13,148)을 쓰고,
val(단어+라인)로 best 를 고르며, test 는 단어/라인을 따로 둡니다. 분할은 labels.csv /
lines_*.txt 의 split 을 그대로 상속하므로 새 분할·증강 파일은 만들지 않습니다(분할→학습).

출력 (artifacts/str_baselines/):
  lists/{train,val,test_word,test_line}.txt   "<abs path>\t<label>"
  parseq_data/train/real, val, test/word, test/line   표준 LMDB(image-%09d / label-%09d /
                                              num-samples) — PARSeq root_dir 레이아웃, OpenOCR 도 같은 디렉터리 사용
  dict/korean_v5_dict.txt                     PP-OCRv5 한국어 사전 복사본(SVTRv2·PARSeq 공용)
  charset_train.txt                           train 라벨에 실제 등장한 문자(감사용)
  audit.txt                                   분할 교차 0 검증 결과

Usage: .venv/Scripts/python.exe prep_str_baselines_data.py [--skip-lmdb]
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
V3 = HERE / "artifacts" / "ocr_training" / "signboard_v3"
OUT = HERE / "artifacts" / "str_baselines"
DICT_SRC = (HERE / ".venv/Lib/site-packages/paddlex/repo_manager/repos/PaddleOCR/ppocr/utils/dict/"
            "ppocrv5_korean_dict.txt")
MAX_LEN = 25


def read_words() -> dict[str, list[tuple[Path, str]]]:
    out = {"train": [], "val": [], "test": []}
    with (V3 / "labels.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["split"]].append((V3 / r["image_path"], r["text"]))
    return out


def read_lines(split: str) -> list[tuple[Path, str]]:
    rows = []
    with (V3 / f"lines_{split}.txt").open(encoding="utf-8") as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if not ln:
                continue
            p, t = ln.split("\t", 1)
            rows.append((V3 / p, t))
    return rows


def write_list(name: str, rows: list[tuple[Path, str]]) -> None:
    (OUT / "lists").mkdir(parents=True, exist_ok=True)
    with (OUT / "lists" / f"{name}.txt").open("w", encoding="utf-8") as f:
        for p, t in rows:
            f.write(f"{p.resolve().as_posix()}\t{t}\n")


def build_lmdb(name: str, rows: list[tuple[Path, str]]) -> None:
    import lmdb
    d = OUT / "parseq_data" / {"train": "train/real", "val": "val", "test_word": "test/word",
                                 "test_line": "test/line"}[name]
    if (d / "data.mdb").exists():
        print(f"[lmdb] {name}: 존재, 건너뜀"); return
    d.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(d), map_size=8 * 1024 ** 3)
    cache, n = {}, 0
    for p, t in rows:
        b = p.read_bytes()
        if not b:
            continue
        n += 1
        cache[f"image-{n:09d}".encode()] = b
        cache[f"label-{n:09d}".encode()] = t.encode("utf-8")
        if n % 2000 == 0:
            with env.begin(write=True) as txn:
                for k, v in cache.items():
                    txn.put(k, v)
            cache = {}
            print(f"[lmdb] {name}: {n}/{len(rows)}", flush=True)
    cache[b"num-samples"] = str(n).encode()
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)
    env.close()
    print(f"[lmdb] {name}: done {n}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-lmdb", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    words = read_words()
    lines = {s: read_lines(s) for s in ("train", "val", "test")}
    sets = {
        "train": words["train"] + lines["train"],
        "val": words["val"] + lines["val"],
        "test_word": words["test"],
        "test_line": lines["test"],
    }
    # 길이 필터(라벨 25자 초과는 두 모델 모두 max_text_length 25 → 학습에서만 제외, test 는 유지)
    drop = [(p, t) for p, t in sets["train"] if len(t) > MAX_LEN]
    sets["train"] = [(p, t) for p, t in sets["train"] if len(t) <= MAX_LEN]

    # ---- 감사: 원천 이미지 stem 이 split 을 넘나들지 않는지 (단어/라인 모두) ----
    def src_stem(p: Path) -> str:
        return p.stem.split("__")[0]
    stems = {k: {src_stem(p) for p, _ in v} for k, v in sets.items()}
    audit = []
    for a, b in (("train", "val"), ("train", "test_word"), ("train", "test_line"),
                 ("val", "test_word"), ("val", "test_line")):
        x = len(stems[a] & stems[b]); audit.append(f"{a} ∩ {b} (source image) = {x}")
    audit.append(f"test_word ∩ test_line = {len(stems['test_word'] & stems['test_line'])} (같은 test 원천 — 정상)")
    ok = all(int(s.split("= ")[1].split()[0]) == 0 for s in audit[:-1])
    audit.append(f"train 길이>{MAX_LEN} 제외: {len(drop)}")
    for k, v in sets.items():
        audit.append(f"{k}: {len(v)} samples")
    audit.append("AUDIT " + ("PASS" if ok else "FAIL"))
    (OUT / "audit.txt").write_text("\n".join(audit) + "\n", encoding="utf-8")
    print("\n".join(audit))
    if not ok:
        sys.exit(1)

    for k, v in sets.items():
        write_list(k, v)
    (OUT / "dict").mkdir(exist_ok=True)
    shutil.copy(DICT_SRC, OUT / "dict" / "korean_v5_dict.txt")
    ch = Counter("".join(t for _, t in sets["train"]))
    (OUT / "charset_train.txt").write_text("".join(sorted(ch)), encoding="utf-8")
    dict_chars = set((OUT / "dict" / "korean_v5_dict.txt").read_text(encoding="utf-8").split("\n"))
    missing = sorted(c for c in ch if c not in dict_chars and c != " ")
    print(f"[dict] v5 korean dict {len(dict_chars)} chars; train charset {len(ch)}; "
          f"dict 에 없는 train 문자 {len(missing)}: {''.join(missing)[:80]}")

    if not args.skip_lmdb:
        for k, v in sets.items():
            build_lmdb(k, v)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
