#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""데이터 섞임(누수) 전수 감사 — 분할→증강 순서와 원천 단위 분리를 숫자로 검증합니다.

규칙: 분할(train/val/test)이 먼저, 증강은 그 다음 **학습 루프 안에서만**. 증강 사본이
디스크에 있으면 안 되고, 같은 원천(사진·소스이미지·가게)이 두 분할에 있으면 안 됩니다.

검사 항목 (하나라도 위반이면 exit 1):
  탐지 fold (artifacts/kfold_grouped)
    D1  val 이 전체 사진을 정확히 분할 (fold 간 val 중복 0, fold 내 train∩val 0)
    D2  fold 안의 모든 jpg 가 원본 GSV 사진과 md5 동일 (증강 사본·변형 0)
    D3  다른 fold 사진과 같은 가게(간판 텍스트) 공유 0
    D4  fold 디렉터리에 증강 산출물(*aug*, *.npy) 0
  OCR 학습 (artifacts/ocr_training/signboard_v3)
    O1  단어 크롭 train/val/test 간 원천 이미지(파일명 해시) 공유 0
    O2  라인 크롭 lines_{split} 의 원천이 단어 {split} 과 일치 (다른 분할 원천 0)
    O3  6개 크롭 폴더의 디스크 파일 수 == 리스트 수 (리스트 밖 사본 0)
    O4  Paddle yml 의 Eval 변환에 RecConAug/RecAug 없음 (증강은 Train 블록에만)

Usage:  .venv/Scripts/python.exe audit_split_leakage.py
"""
from __future__ import annotations

import csv
import hashlib
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
DET = HERE / "artifacts" / "kfold_grouped"
GSV = HERE / "artifacts" / "gsv_photo"
GT = HERE / "artifacts" / "gt"
OCR = HERE / "artifacts" / "ocr_training" / "signboard_v3"
PADDLE_YML = HERE / "configs/paddle_signboard_rec_v5_lines.yml"
REGIONS = ("gangnam", "brooklyn", "suwon")
GENERIC = {
    "coffee", "cafe", "chicken", "pizza", "food", "deli", "market", "pharmacy", "nail",
    "nails", "salon", "bar", "grill", "hotel", "bank", "restaurant", "shop", "store",
    "open", "sale", "tel", "since", "inc", "total", "available", "laundromat", "beauty",
    "hair", "wine", "liquor", "corp", "llc", "center", "노래방", "치킨", "커피", "식당",
    "약국", "편의점", "미용실", "부동산", "호프", "카페", "마트", "병원", "의원", "학원",
}

fails: list[str] = []


def check(cond: bool, code: str, msg: str) -> None:
    print(f"  [{'OK ' if cond else 'FAIL'}] {code}  {msg}")
    if not cond:
        fails.append(code)


def nk(s: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]", "", unicodedata.normalize("NFKC", s or "").lower())


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def audit_detection(k: int = 5) -> None:
    print("탐지 fold:", DET.relative_to(HERE).as_posix())
    vals = [{p.stem for p in (DET / f"total_fold{i}/dataset/images/val").glob("*.jpg")} for i in range(k)]
    trains = [{p.stem for p in (DET / f"total_fold{i}/dataset/images/train").glob("*.jpg")} for i in range(k)]
    allv = set().union(*vals)
    dup = sum(len(vals[i] & vals[j]) for i in range(k) for j in range(i + 1, k))
    inner = sum(len(trains[i] & vals[i]) for i in range(k))
    sizes = {len(trains[i] | vals[i]) for i in range(k)}
    check(dup == 0 and inner == 0 and sizes == {len(allv)}, "D1",
          f"val 합집합 {len(allv)}장, fold간 val 중복 {dup}, fold내 train∩val {inner}, train∪val 크기 {sorted(sizes)}")

    orig = {f"total__{reg}__{p.stem}": md5(p) for reg in REGIONS for p in (GSV / reg).glob("*.jpg")}
    n = extra = mism = 0
    for i in range(k):
        for split in ("train", "val"):
            for p in (DET / f"total_fold{i}/dataset/images/{split}").glob("*.jpg"):
                n += 1
                if p.stem not in orig:
                    extra += 1
                elif md5(p) != orig[p.stem]:
                    mism += 1
    check(extra == 0 and mism == 0, "D2", f"{n}개 파일 중 원본에 없는 것 {extra}, 원본과 바이트가 다른 것 {mism}")

    keys: dict[str, set[str]] = defaultdict(set)
    for reg in REGIONS:
        for r in csv.DictReader((GT / f"ocr_{reg}_gt.csv").open(encoding="utf-8")):
            ph = "total__" + "__".join(r["image_name"].split("__")[:2])
            for line in r["gt_text"].replace(chr(92) + "n", "\n").split("\n"):
                kk = nk(line)
                if "###" not in line and len(kk) >= 4 and kk not in GENERIC:
                    keys[ph].add(kk)
    fold_of = {s: i for i in range(k) for s in vals[i]}
    stems = sorted(fold_of)
    leak = sum(1 for a in stems for b in stems
               if a < b and a.split("__")[1] == b.split("__")[1]
               and keys[a] & keys[b] and fold_of[a] != fold_of[b])
    check(leak == 0, "D3", f"다른 fold 사진과 가게(간판 텍스트)를 공유하는 쌍 {leak}")

    junk = len(list(DET.rglob("*aug*"))) + len(list(DET.rglob("*.npy")))
    check(junk == 0, "D4", f"fold 디렉터리 내 증강 산출물 {junk}개")


def audit_ocr() -> None:
    print("OCR 학습 데이터:", OCR.relative_to(HERE).as_posix())

    def srcs(txt: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for line in (OCR / txt).read_text(encoding="utf-8").splitlines():
            if "\t" in line:
                h = Path(line.split("\t", 1)[0]).name.split("__")[0]
                out[h] = out.get(h, 0) + 1
        return out

    W = {s: srcs(f"{s}.txt") for s in ("train", "val", "test")}
    L = {s: srcs(f"lines_{s}.txt") for s in ("train", "val", "test")}
    ov = {f"{a}∩{b}": len(set(W[a]) & set(W[b])) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    check(all(v == 0 for v in ov.values()), "O1",
          f"단어 크롭 원천 공유 {ov} (원천 수 {len(W['train'])}/{len(W['val'])}/{len(W['test'])})")
    bad = {}
    for s in ("train", "val", "test"):
        others = set().union(*(set(W[o]) for o in W if o != s))
        bad[s] = len(set(L[s]) & others)
    check(all(v == 0 for v in bad.values()), "O2", f"라인 크롭이 다른 단어 분할 원천을 쓰는 건수 {bad}")

    dirs = [("crops/train", "train.txt"), ("crops/val", "val.txt"), ("crops/test", "test.txt"),
            ("crops_lines/train", "lines_train.txt"), ("crops_lines/val", "lines_val.txt"),
            ("crops_lines/test", "lines_test.txt")]
    diffs = {}
    for d, lst in dirs:
        n_disk = sum(1 for _ in (OCR / d).glob("*.jpg"))
        n_list = sum(1 for l in (OCR / lst).read_text(encoding="utf-8").splitlines() if "\t" in l)
        diffs[d] = n_disk - n_list
    check(all(v == 0 for v in diffs.values()), "O3", f"디스크−리스트 파일 수 차이 {diffs}")

    if PADDLE_YML.exists():
        block, eval_aug = None, []
        for line in PADDLE_YML.read_text(encoding="utf-8").splitlines():
            if re.match(r"^(Train|Eval):", line):
                block = line.split(":")[0]
            if block == "Eval" and re.search(r"RecConAug|RecAug", line) and "null" not in line:
                eval_aug.append(line.strip())
        check(not eval_aug, "O4", f"Paddle Eval 블록의 증강 항목 {eval_aug or 0}")


def main() -> None:
    audit_detection()
    audit_ocr()
    print()
    if fails:
        print(f"위반 {len(fails)}건: {fails}")
        sys.exit(1)
    print("전 항목 통과 — train/val/test 간 섞임 0, 디스크 증강 사본 0.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
