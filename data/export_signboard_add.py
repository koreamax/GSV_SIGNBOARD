#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""누락 간판 보강 라벨 → 원본 사진 좌표.

`data/label_signboard.py` 로 라벨한 `artifacts/gt/signboard_add/labels.json`(작업 이미지 기준 0~1)을
원본 사진 픽셀 좌표로 바꿔 `artifacts/gt/signboard_add/added_boxes.json` 에 씁니다.

  {"gangnam__12": {"size": [W, H], "added": [[x0, y0, x1, y1], ...], "status": "done"}, ...}

완료(done) 표시가 안 된 사진은 status 가 todo 로 남으며, 재채점 스크립트는 기본적으로 298장이 모두
done 일 때만 돌도록 되어 있습니다(일부만 보강된 GT 로 표를 다시 내지 않기 위해).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "artifacts" / "gt" / "signboard_add"


def main() -> None:
    items = {it["id"]: it for it in json.loads((OUT / "items.json").read_text(encoding="utf-8"))}
    labels = json.loads((OUT / "labels.json").read_text(encoding="utf-8")) if (OUT / "labels.json").exists() else {}
    res, n_add, n_done = {}, 0, 0
    for pid, it in items.items():
        rec = labels.get(pid, {"boxes": [], "status": "todo"})
        W, H = it["W"], it["H"]
        boxes = [[round(b[0] * W, 1), round(b[1] * H, 1), round(b[2] * W, 1), round(b[3] * H, 1)]
                 for b in rec["boxes"] if b[2] > b[0] and b[3] > b[1]]
        res[pid] = {"size": [W, H], "added": boxes, "status": rec["status"]}
        n_add += len(boxes)
        n_done += rec["status"] == "done"
    (OUT / "added_boxes.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"완료 {n_done}/{len(items)}장 · 추가 간판 {n_add}개 → {OUT / 'added_boxes.json'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
