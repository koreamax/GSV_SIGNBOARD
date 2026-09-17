#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D19: line-merge pipeline experiment — feed whole detected LINES to Paddle.

The per-box pipeline slices signs into single-word CRAFT boxes, so the v4
spacecat model's multi-word/space ability never fires (4.8). Here each detected
line's boxes are merged into ONE strip (union bbox, pad 0.12 — same detector,
same filters as run_ocr_only) and recognized whole:

  A) per-box Paddle v3      = official run12 files (baseline, no recompute)
  B) line-merge Paddle v3   -> expects space-less multi-word outputs
  C) line-merge Paddle v4   -> spaces should appear (trained with gap+space)

Korean regions only (gangnam, suwon) — brooklyn uses the English model.
Outputs: artifacts/ocr_gt/ab_v4_spacecat/ocr_{region}_linemerge_{v3,v4}.csv
Eval: official eval_ocr_v2 code path (module import), exact/CER/WAR/contain.
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
OUT_DIR = HERE / "artifacts" / "ocr_gt" / "ab_v4_spacecat"
PY = HERE / ".venv" / "Scripts" / "python.exe"
WORKER = HERE / "str_baselines/paddle_rec_worker.py"

ARGS = argparse.Namespace()          # set in main()
REGIONS: list[str] = []
MODELS: dict[str, Path | None] = {}  # tag -> inference dir (None = pretrained en)

FILTER_ARGS = SimpleNamespace(filt_min_conf=0.10, filt_numeric_digits=7,
                              filt_numeric_ratio=0.6, filt_min_h_ratio=0.0,
                              filt_min_h_frac=0.0, no_filter=False)


def natural_key(s: str):
    import re
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def detect_strips(tmp: Path):
    """Same detector + line grouping + filters as run_ocr_only run12."""
    import torch
    import easyocr
    import run_ocr_only as RO

    if ARGS.en:      # brooklyn: pretrained English reader, same as run12 (D15)
        reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
    else:
        reader = easyocr.Reader(["ko", "en"], gpu=torch.cuda.is_available(),
                                recog_network="signboard_v3_custom")
    manifest, layout = [], {}
    for region in REGIONS:
        layout[region] = {}
        files = sorted((CROP / region).glob("*.jpg"), key=lambda p: natural_key(p.stem))
        print(f"[DETECT] {region}: {len(files)} crops")
        for img_path in files:
            img = Image.open(img_path).convert("RGB")
            lines = RO.easyocr_detect_lines(
                reader, img, line_y_tol=0.06, craft_text_threshold=0.7,
                craft_link_threshold=0.4, craft_low_text=0.4)
            lines, _ = RO.filter_boxes(lines, img.height, FILTER_ARGS)
            n = 0
            for li, line in enumerate(lines):
                pts = [p for b in line for p in b["bbox"]]
                if not pts:
                    continue
                strip = RO.crop_box(img, pts, pad=0.12)
                key = f"{region}::{img_path.stem}::{li}"
                p = tmp / f"{region}_{img_path.stem}_{li}.png"
                strip.save(p)
                manifest.append({"key": key, "path": str(p)})
                n += 1
            layout[region][img_path.stem] = n
    return manifest, layout


def run_worker(model_dir: Path | None, manifest: list[dict], tmp: Path, tag: str) -> dict[str, str]:
    man, out = tmp / f"man_{tag}.jsonl", tmp / f"out_{tag}.jsonl"
    with man.open("w", encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    cmd = [str(PY), str(WORKER), "--manifest", str(man), "--out", str(out),
           "--device", "gpu"]
    if model_dir is not None:
        cmd += ["--model-dir", str(model_dir)]
    else:
        cmd += ["--model-name", "en_PP-OCRv5_mobile_rec"]
    subprocess.run(cmd, check=True)
    preds = {}
    with out.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            preds[d["key"]] = d["text"]
    return preds


def assemble(layout, preds) -> dict[str, dict[str, str]]:
    maps = {}
    for region, imgs in layout.items():
        m = {}
        for name, n in imgs.items():
            parts = [preds.get(f"{region}::{name}::{li}", "").strip() for li in range(n)]
            m[name] = "\n".join(p for p in parts if p)
        maps[region] = m
    return maps


def load_eval():
    saved = sys.argv
    sys.argv = ["pipeline/eval_ocr_v2.py", "--mask-phone", "--en-only-regions", "brooklyn"]
    import eval_ocr_v2 as E
    sys.argv = saved
    return E


def main() -> None:
    global REGIONS, MODELS
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", default="gangnam,suwon")
    ap.add_argument("--en", action="store_true",
                    help="brooklyn mode: en reader + pretrained en_PP-OCRv5_mobile_rec")
    ap.parse_args(namespace=ARGS)
    REGIONS = [r.strip() for r in ARGS.regions.split(",") if r.strip()]
    if ARGS.en:
        MODELS = {"en": None}
    else:
        MODELS = {"v3": HERE / "output" / "paddle_signboard_rec_v3" / "inference",
                  "v4": HERE / "output" / "paddle_signboard_rec_v4_spacecat" / "inference"}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="linemerge_"))
    manifest, layout = detect_strips(tmp)
    print(f"[DETECT] line strips: {len(manifest)}  tmp={tmp}")

    variant_maps = {}
    for tag, mdir in MODELS.items():
        print(f"[REC] paddle {tag} on line strips ...")
        preds = run_worker(mdir, manifest, tmp, tag)
        variant_maps[f"linemerge_{tag}"] = assemble(layout, preds)
        for region, m in variant_maps[f"linemerge_{tag}"].items():
            path = OUT_DIR / f"ocr_{region}_linemerge_{tag}.csv"
            with path.open("w", encoding="utf-8", newline="") as f:
                w = csv.writer(f, quoting=csv.QUOTE_ALL)
                w.writerow(["image_name", "gt_text"])
                for k in sorted(m):
                    w.writerow([k, m[k].replace("\n", "\\n")])
    print(f"[SAVE] -> {OUT_DIR}\\ocr_*_linemerge_*.csv")

    E = load_eval()
    baseline = {r: E.load_csv_map(HERE / "artifacts" / "ocr_gt" / f"ocr_{r}_12_paddle.csv")
                for r in REGIONS}
    gt = {r: E.load_csv_map(HERE / "artifacts" / "gt" / f"ocr_{r}_gt.csv") for r in REGIONS}

    def acc_of(region, pred_map):
        E.EN_ONLY = region in E.EN_ONLY_REGIONS
        det = []
        a = E.eval_engine(gt[region], pred_map, det)
        E.EN_ONLY = False
        return a

    rows = [("per-box v3 (run12 공식)", baseline)]
    rows += [(f"line-merge {t}", variant_maps[f"linemerge_{t}"]) for t in MODELS]
    print(f"\n{'variant':26s} {'region':9s} {'exact':>7s} {'CER':>7s} {'WAR':>7s} {'contain':>8s}")
    for name, maps in rows:
        tot = None
        for region in REGIONS:
            a = acc_of(region, maps[region])
            print(f"{name:26s} {region:9s} {a.exact_rate()*100:6.1f}% {a.cer():7.3f} "
                  f"{a.war():7.3f} {a.contain_rate()*100:7.1f}%")
            if tot is None:
                tot = a
            else:
                for fld in ("edit", "fp_edit", "chars", "word_lcs", "gt_words",
                            "exact", "recalled", "lines", "contain", "empty_pred", "crops"):
                    setattr(tot, fld, getattr(tot, fld) + getattr(a, fld))
        print(f"{name:26s} {'gn+sw':9s} {tot.exact_rate()*100:6.1f}% {tot.cer():7.3f} "
              f"{tot.war():7.3f} {tot.contain_rate()*100:7.1f}%")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
