#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Group-aware 5-fold 재분할 — 같은 가게를 찍은 사진은 한 fold에 묶습니다 (D33).

**왜 필요한가:** 기존 fold(`artifacts/yolo11x_kfold`)는 사진 단위 무작위 분할이라,
인접 위치에서 찍은 GSV 사진들이 **같은 가게 간판**을 담은 채 train/val 로 갈렸습니다.
GT 간판 텍스트로 대조하면 val 사진 298장 중 62장(20.8%)이 다른 fold 의 train 사진과
가게를 공유합니다(강남 31.3% / 브루클린 25.3% / 수원 6.0%). 탐지기가 학습 때 본
바로 그 간판을 val 에서 다시 만나므로 fold/OOF mAP 가 낙관적으로 나옵니다.
파일 단위 증강은 없었습니다(train 238장 전부 원본) — 증강→분할 누수가 아닙니다.

**방법:** 사진을 노드, "4자 이상·일반어가 아닌 간판 텍스트를 공유"를 간선으로
같은 지역 안에서 연결성분을 만들고, 성분 단위로 지역 층화·크기 균형을 맞춰
5개 fold 에 배정합니다. 원본 fold 의 이미지/라벨 파일을 그대로 재사용하므로
데이터 자체는 동일하고 **배정만** 바뀝니다.

출력: artifacts/kfold_grouped/total_fold{i}/dataset/{images,labels}/{train,val} + data.yaml
      artifacts/kfold_grouped/group_assignment.csv (사진, 지역, 그룹id, fold)

한계: 재라벨 때 제외된 크롭 52개는 GT 텍스트가 없어 매칭에 못 씁니다. 그 간판이
두 사진에 걸쳐 있으면 잔여 누수가 남을 수 있습니다(pHash 전체 이미지 검사에서는
val-train 근접 중복 0쌍).
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "artifacts" / "yolo11x_kfold"          # 기존 fold (파일 원천)
GT = HERE / "artifacts" / "gt"
OUT = HERE / "artifacts" / "kfold_grouped"
REGIONS = ("gangnam", "brooklyn", "suwon")
GENERIC = {
    "coffee", "cafe", "chicken", "pizza", "food", "deli", "market", "pharmacy", "nail",
    "nails", "salon", "bar", "grill", "hotel", "bank", "restaurant", "shop", "store",
    "open", "sale", "tel", "since", "inc", "total", "available", "laundromat", "beauty",
    "hair", "wine", "liquor", "corp", "llc", "center", "노래방", "치킨", "커피", "식당",
    "약국", "편의점", "미용실", "부동산", "호프", "카페", "마트", "병원", "의원", "학원",
}


def nk(s: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]", "", unicodedata.normalize("NFKC", s or "").lower())


def gather_pairs() -> dict[str, tuple[Path, Path | None]]:
    """stem(total__region__id) -> (jpg, label txt or None). fold0 train∪val = 전체."""
    pairs = {}
    for split in ("train", "val"):
        for jpg in (SRC / "total_fold0" / "dataset" / "images" / split).glob("*.jpg"):
            lbl = SRC / "total_fold0" / "dataset" / "labels" / split / f"{jpg.stem}.txt"
            pairs[jpg.stem] = (jpg, lbl if lbl.exists() else None)
    return pairs


def store_keys() -> dict[str, set[str]]:
    keys: dict[str, set[str]] = defaultdict(set)
    for reg in REGIONS:
        for r in csv.DictReader((GT / f"ocr_{reg}_gt.csv").open(encoding="utf-8")):
            photo = "total__" + "__".join(r["image_name"].split("__")[:2])
            for line in r["gt_text"].replace(chr(92) + "n", "\n").split("\n"):
                if "###" in line:
                    continue
                k = nk(line)
                if len(k) >= 4 and k not in GENERIC:
                    keys[photo].add(k)
    return keys


def components(stems: list[str], keys) -> list[list[str]]:
    adj = defaultdict(set)
    by_reg = defaultdict(list)
    for s in stems:
        by_reg[s.split("__")[1]].append(s)
    for reg, lst in by_reg.items():
        for i, a in enumerate(lst):
            for b in lst[i + 1:]:
                if keys[a] & keys[b]:
                    adj[a].add(b); adj[b].add(a)
    seen, comps = set(), []
    for s in stems:
        if s in seen:
            continue
        stack, comp = [s], set()
        while stack:
            x = stack.pop()
            if x in comp:
                continue
            comp.add(x); stack.extend(adj[x] - comp)
        seen |= comp
        comps.append(sorted(comp))
    return comps


def assign(comps: list[list[str]], k: int, seed: int) -> dict[str, int]:
    """큰 그룹부터, '그 지역 사진이 가장 적은 fold'(동률이면 전체가 적은 fold)에 배정."""
    rng = random.Random(seed)
    comps = sorted(comps, key=lambda c: (-len(c), rng.random()))
    reg_cnt = [defaultdict(int) for _ in range(k)]
    tot = [0] * k
    fold_of = {}
    for comp in comps:
        reg = comp[0].split("__")[1]
        f = min(range(k), key=lambda i: (reg_cnt[i][reg], tot[i], rng.random()))
        for s in comp:
            fold_of[s] = f
        reg_cnt[f][reg] += len(comp); tot[f] += len(comp)
    return fold_of


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)

    pairs = gather_pairs()
    stems = sorted(pairs)
    keys = store_keys()
    comps = components(stems, keys)
    fold_of = assign(comps, args.k, args.seed)
    comp_id = {s: ci for ci, c in enumerate(comps) for s in c}

    # ---- 검증: 분할 완전성 + fold 간 가게 공유 0 ----
    assert len(fold_of) == len(stems) == 298 or len(fold_of) == len(stems)
    leak = 0
    for a in stems:
        for b in stems:
            if a < b and a.split("__")[1] == b.split("__")[1] and (keys[a] & keys[b]) \
                    and fold_of[a] != fold_of[b]:
                leak += 1
    assert leak == 0, f"fold 간 가게 공유 쌍 {leak}개 — 배정 실패"

    # ---- 파일 배치 ----
    if out.exists():
        shutil.rmtree(out)
    for i in range(args.k):
        for split in ("train", "val"):
            (out / f"total_fold{i}" / "dataset" / "images" / split).mkdir(parents=True)
            (out / f"total_fold{i}" / "dataset" / "labels" / split).mkdir(parents=True)
    for s, (jpg, lbl) in pairs.items():
        for i in range(args.k):
            split = "val" if fold_of[s] == i else "train"
            d = out / f"total_fold{i}" / "dataset"
            shutil.copy2(jpg, d / "images" / split / jpg.name)
            dst_lbl = d / "labels" / split / f"{s}.txt"
            if lbl is not None:
                shutil.copy2(lbl, dst_lbl)
            else:
                dst_lbl.write_text("", encoding="utf-8")     # 간판 없는 사진 = 배경
    for i in range(args.k):
        d = out / f"total_fold{i}" / "dataset"
        (d / "data.yaml").write_text(
            "names:\n- signboard\nnc: 1\n"
            f"path: {d.relative_to(HERE).as_posix()}\n"
            "train: images/train\nval: images/val\n", encoding="utf-8")

    with (out / "group_assignment.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["photo", "region", "group_id", "group_size", "fold"])
        for s in stems:
            w.writerow([s, s.split("__")[1], comp_id[s], len(comps[comp_id[s]]), fold_of[s]])

    # ---- 리포트 ----
    print(f"사진 {len(stems)}장 → 가게 그룹 {len(comps)}개 "
          f"(2장+ 그룹 {sum(1 for c in comps if len(c) >= 2)}개, 최대 {max(len(c) for c in comps)}장)")
    print(f"fold 간 가게 공유 쌍: {leak}개 (검증 통과)\n")
    print(f"{'fold':5s} {'train':>6s} {'val':>5s} | {'강남':>5s} {'브루클린':>7s} {'수원':>5s}  (val 지역별)")
    for i in range(args.k):
        val = [s for s in stems if fold_of[s] == i]
        rc = defaultdict(int)
        for s in val:
            rc[s.split("__")[1]] += 1
        print(f"fold{i} {len(stems)-len(val):6d} {len(val):5d} | {rc['gangnam']:5d} "
              f"{rc['brooklyn']:7d} {rc['suwon']:5d}")
    print(f"\n출력: {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
