#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Suwon detection-bottleneck sweep (D22 candidate) — CPU only (GPU is training).

Suwon line-merge failures: 29/181 GT lines appear NOWHERE in the prediction
(complete detection/recognition loss). Sweep detection-side remedies on the
line-merge pipeline and score with the official eval code:

  variants: 1x baseline / 2x upscale (LANCZOS) / low_text 0.3 / 2x + low_text 0.3

Recognition = v4_spacecat via paddle worker (--device cpu).
Usage: .venv/Scripts/python.exe exp_suwon_det.py [--region suwon]
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

HERE = Path(__file__).resolve().parents[1]

import sys as _sys; _sys.path[:0] = [str(HERE / _d) for _d in ("pipeline", "ocr", "vlm", "str_baselines", "detection", "data")]  # 폴더 재편 후 형제 모듈 import 경로
CROP = HERE / "artifacts" / "gt" / "crop"
PY = HERE / ".venv" / "Scripts" / "python.exe"
WORKER = HERE / "str_baselines/paddle_rec_worker.py"
MODEL = HERE / "output" / "paddle_signboard_rec_v4_spacecat" / "inference"
FILTER_ARGS = SimpleNamespace(filt_min_conf=0.10, filt_numeric_digits=7,
                              filt_numeric_ratio=0.6, filt_min_h_ratio=0.0,
                              filt_min_h_frac=0.0, no_filter=False)

VARIANTS = [
    ("1x_base",   1, 0.4),
    ("2x_up",     2, 0.4),
    ("1x_lt0.3",  1, 0.3),
    ("2x_lt0.3",  2, 0.3),
]


def natural_key(s: str):
    import re
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="suwon")
    args = ap.parse_args()
    region = args.region

    import easyocr
    import run_ocr_only as RO
    reader = easyocr.Reader(["ko", "en"], gpu=False, recog_network="signboard_v3_custom")

    files = sorted((CROP / region).glob("*.jpg"), key=lambda p: natural_key(p.stem))
    tmp = Path(tempfile.mkdtemp(prefix="suwon_det_"))
    results = {}

    for tag, scale, low_text in VARIANTS:
        manifest, layout = [], {}
        for img_path in files:
            img = Image.open(img_path).convert("RGB")
            if scale != 1:
                img = img.resize((img.width * scale, img.height * scale), Image.LANCZOS)
            lines = RO.easyocr_detect_lines(
                reader, img, line_y_tol=0.06, craft_text_threshold=0.7,
                craft_link_threshold=0.4, craft_low_text=low_text)
            lines, _ = RO.filter_boxes(lines, img.height, FILTER_ARGS)
            n = 0
            for li, line in enumerate(lines):
                pts = [p for b in line for p in b["bbox"]]
                if not pts:
                    continue
                strip = RO.crop_box(img, pts, pad=0.12)
                p = tmp / f"{tag}_{img_path.stem}_{li}.png"
                strip.save(p)
                manifest.append({"key": f"{img_path.stem}::{li}", "path": str(p)})
                n += 1
            layout[img_path.stem] = n
        man, out = tmp / f"man_{tag}.jsonl", tmp / f"out_{tag}.jsonl"
        with man.open("w", encoding="utf-8") as f:
            for m in manifest:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        subprocess.run([str(PY), str(WORKER), "--manifest", str(man), "--out", str(out),
                        "--model-dir", str(MODEL), "--device", "cpu"], check=True)
        preds = {}
        with out.open(encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                preds[d["key"]] = d["text"]
        pmap = {}
        for name, n in layout.items():
            parts = [preds.get(f"{name}::{li}", "").strip() for li in range(n)]
            pmap[name] = "\n".join(p for p in parts if p)
        results[tag] = (pmap, len(manifest))
        print(f"[{tag}] strips={len(manifest)}")

    saved = sys.argv
    sys.argv = ["pipeline/eval_ocr_v2.py", "--mask-phone", "--en-only-regions", "brooklyn"]
    import eval_ocr_v2 as E
    sys.argv = saved
    E.EN_ONLY = region in E.EN_ONLY_REGIONS
    gt = E.load_csv_map(HERE / "artifacts" / "gt" / f"ocr_{region}_gt.csv")

    print(f"\n{'variant':10s} {'strips':>6s} {'exact':>7s} {'CER':>7s} {'WAR':>7s} {'contain':>8s} {'NOT-READ':>9s}")
    for tag, (pmap, nstrips) in results.items():
        det = []
        a = E.eval_engine(gt, pmap, det)
        nr = 0
        for key, gtext in gt.items():
            gl, ndc = E.split_gt_lines(gtext)
            whole = E.norm_cer(pmap.get(key, ""))
            for g in gl:
                G = E.norm_cer(g)
                if G and G not in whole:
                    pl = E.split_lines(pmap.get(key, "")) if pmap.get(key, "").strip() else []
                    assign, _ = E.match_lines(gl, pl, infix=(ndc > 0))
                    i = gl.index(g)
                    if assign[i][1] >= assign[i][2]:
                        nr += 1
        print(f"{tag:10s} {nstrips:6d} {a.exact_rate()*100:6.1f}% {a.cer():7.3f} "
              f"{a.war():7.3f} {a.contain_rate()*100:7.1f}% {nr:9d}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
