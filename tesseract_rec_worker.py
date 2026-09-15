#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tesseract 5 라인 인식 워커 — paddle_rec_worker 와 같은 IO 계약.

  --manifest IN.jsonl : {"key": str, "path": str}   key 는 "<region>::<crop>::<line>"
  --out      OUT.jsonl: {"key": str, "text": str, "score": float}

라인 스트립 1장 = 한 줄이므로 --psm 7. 언어는 key 의 region 으로 고릅니다
(brooklyn → eng, 그 외 → kor+eng). 이미지는 높이 48px 미만이면 확대(Tesseract 는 저해상에 약함).
conda env `tess`(tesseract 5.5.3, conda-forge) 의 실행 파일과 tessdata 를 사용합니다.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

TESS_HOME = Path(os.environ.get("TESS_HOME", str(Path.home() / "anaconda3" / "envs" / "tess")))
TESS = TESS_HOME / "Library" / "bin" / "tesseract.exe"
TESSDATA = TESS_HOME / "share" / "tessdata"
MIN_H = 48


def lang_for(key: str, en_regions: set[str]) -> str:
    region = key.split("::", 1)[0]
    return "eng" if region in en_regions else "kor+eng"


def run_one(item: dict, en_regions: set[str], psm: int, tmpdir: Path) -> dict:
    img = Image.open(item["path"]).convert("RGB")
    if img.height < MIN_H:
        s = MIN_H / img.height
        img = img.resize((max(1, round(img.width * s)), MIN_H), Image.BICUBIC)
    p = tmpdir / (str(abs(hash(item["key"]))) + ".png")
    img.save(p)
    env = dict(os.environ, TESSDATA_PREFIX=str(TESSDATA))
    r = subprocess.run([str(TESS), str(p), "stdout", "-l", lang_for(item["key"], en_regions),
                        "--psm", str(psm)], capture_output=True, env=env)
    txt = r.stdout.decode("utf-8", errors="replace").strip().replace("\n", " ")
    p.unlink(missing_ok=True)
    return {"key": item["key"], "text": " ".join(txt.split()), "score": 1.0 if txt else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--psm", type=int, default=7)
    ap.add_argument("--en-regions", default="brooklyn")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    en = {r for r in args.en_regions.split(",") if r}
    items = [json.loads(l) for l in open(args.manifest, encoding="utf-8") if l.strip()]
    tmpdir = Path(tempfile.mkdtemp(prefix="tess_"))
    with ThreadPoolExecutor(args.workers) as ex, open(args.out, "w", encoding="utf-8") as f:
        for i, res in enumerate(ex.map(lambda it: run_one(it, en, args.psm, tmpdir), items), 1):
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            if i % 200 == 0:
                print(f"[tesseract] {i}/{len(items)}", flush=True)
    print(f"[tesseract] done {len(items)} -> {args.out}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
