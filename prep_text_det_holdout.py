#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""텍스트(word) 박스 탐지용 hold-out 데이터셋 — OCR 학습데이터 **전부**, 인식기와 **같은 분할**.

이전 `prep_text_det.py`는 5,000장 표본 + 5-fold 였습니다. 여기서는 signboard_data_info.json
의 word 박스가 있는 이미지 전부(27,519장)를 쓰되, 분할은 새로 만들지 않고 **OCR 인식기
(signboard_v3)의 소스 이미지 분할을 그대로 상속**합니다:
  train.txt / val.txt / test.txt 의 크롭 파일명 앞 해시 == JSON file_name 의 stem (1:1 확인됨)

그래야 (1) 같은 test 이미지 위에서 "탐지 모델 X + 인식기" 파이프라인 비교가 성립하고,
(2) 분할→증강 규칙이 지켜지며(증강은 학습 루프 안에서만), (3) 탐지·인식 어느 쪽에서도
test 소스가 train 에 새지 않습니다. v3 분할 어디에도 없는 이미지는 제외합니다.

출력: artifacts/signboard_text_holdout/{images,labels}/{train,val,test}/  (이미지는 하드링크)
      data.yaml (Ultralytics), split_source.csv (파일명, 분할, 박스 수)
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
JSON = HERE / "artifacts" / "signboard_data_info.json"
SRC = HERE / "artifacts" / "Signboard"
V3 = HERE / "artifacts" / "ocr_training" / "signboard_v3"
OUT = HERE / "artifacts" / "signboard_text_holdout"
SPLITS = ("train", "val", "test")


def v3_split_of() -> dict[str, str]:
    """소스 이미지 해시 -> split (인식기 분할 상속)."""
    m: dict[str, str] = {}
    for s in SPLITS:
        for line in (V3 / f"{s}.txt").read_text(encoding="utf-8").splitlines():
            if "\t" in line:
                h = Path(line.split("\t", 1)[0]).name.split("__")[0]
                assert m.get(h, s) == s, f"해시 {h} 가 두 분할에 있음"
                m[h] = s
    return m


def main() -> None:
    d = json.loads(JSON.read_text(encoding="utf-8"))
    imgs = {im["id"]: im for im in d["images"]}
    words: dict[str, list] = defaultdict(list)
    for a in d["annotations"]:
        attrs = a["attributes"] if isinstance(a["attributes"], dict) else json.loads(
            a["attributes"].replace("'", '"'))
        if attrs.get("class") == "word":
            bb = a["bbox"] if isinstance(a["bbox"], list) else json.loads(a["bbox"])
            words[a["image_id"]].append(bb)
    split_of = v3_split_of()

    if OUT.exists():
        shutil.rmtree(OUT)
    for s in SPLITS:
        (OUT / "images" / s).mkdir(parents=True)
        (OUT / "labels" / s).mkdir(parents=True)

    stat = {s: [0, 0] for s in SPLITS}          # images, boxes
    skipped = {"no_v3_split": 0, "missing_file": 0, "no_word": 0}
    rows = []
    for iid, im in imgs.items():
        fn = im["file_name"]; stem = Path(fn).stem
        boxes = words.get(iid, [])
        if not boxes:
            skipped["no_word"] += 1; continue
        s = split_of.get(stem)
        if s is None:
            skipped["no_v3_split"] += 1; continue
        src = SRC / fn
        if not src.exists():
            skipped["missing_file"] += 1; continue
        W, H = float(im["width"]), float(im["height"])
        lines = []
        for x, y, w, h in boxes:
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(W, x + w), min(H, y + h)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            lines.append(f"0 {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
        if not lines:
            skipped["no_word"] += 1; continue
        dst = OUT / "images" / s / fn
        try:
            os.link(src, dst)                      # 하드링크: 디스크 사본 없음
        except OSError:
            shutil.copy2(src, dst)
        (OUT / "labels" / s / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        stat[s][0] += 1; stat[s][1] += len(lines)
        rows.append((fn, s, len(lines)))

    (OUT / "data.yaml").write_text(
        "names:\n- text\nnc: 1\n"
        f"path: {OUT.relative_to(HERE).as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n", encoding="utf-8")
    with (OUT / "split_source.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["file_name", "split", "n_boxes"]); w.writerows(rows)

    # 검증: 분할 간 소스 공유 0 (구성상 보장되지만 숫자로 남김)
    stems = {s: {p.stem for p in (OUT / "images" / s).glob("*.jpg")} for s in SPLITS}
    assert not (stems["train"] & stems["val"]) and not (stems["train"] & stems["test"]) \
        and not (stems["val"] & stems["test"])
    print(f"{'split':6s} {'images':>7s} {'boxes':>8s}")
    for s in SPLITS:
        print(f"{s:6s} {stat[s][0]:7d} {stat[s][1]:8d}")
    print(f"제외: {skipped}")
    print(f"분할 간 소스 공유: 0 (검증 통과)  → {OUT}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
