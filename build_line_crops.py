#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build REAL line-level training crops from signboard word annotations (D21).

v4_spacecat learned multi-word lines only from synthetic concatenation (4.8).
The source dataset has per-word boxes on the original photos, so real lines —
with true background continuity, kerning and spacing — can be cropped directly:

  group words per source image into horizontal lines (y-center clustering,
  split at big horizontal gaps) -> union bbox + 4% pad -> crop from the ORIGINAL
  photo -> label = words joined with single spaces (left-to-right).

Split is inherited from labels.csv (source-image-level split -> no leakage).
Only multi-word lines are emitted (single words already exist in v3 crops).

Output (under signboard_v3 root so PaddleOCR data_dir stays unchanged):
  crops_lines/{split}/{source}__line_{i}.jpg
  lines_train.txt / lines_val.txt / lines_test.txt   (path<TAB>label)

Usage: .venv/Scripts/python.exe build_line_crops.py [--preview 8]
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
V3 = HERE / "artifacts" / "ocr_training" / "signboard_v3"
SRC_IMG = HERE / "artifacts" / "Signboard"
MAX_LABEL = 25          # paddle max_text_length
MAX_RATIO = 12.0        # drop absurdly wide lines
MIN_H = 12              # px
MAX_OUT_H = 96          # save downscaled: training only ever resizes to h<=64,
                        # and native-res line crops made the DataLoader 3.6x slower


def cluster_lines(words: list[dict]) -> list[list[dict]]:
    """Line = words whose y-intervals overlap strongly AND have similar heights
    (center-distance alone merges a small top strip with huge letters below).
    Then split clusters at big horizontal gaps."""
    ws = sorted(words, key=lambda w: w["y"] + w["h"] / 2)
    rows: list[list[dict]] = []
    for w in ws:
        placed = False
        for row in rows:
            ry0 = sum(r["y"] for r in row) / len(row)
            ry1 = sum(r["y"] + r["h"] for r in row) / len(row)
            rh = max(ry1 - ry0, 1.0)
            ov = min(ry1, w["y"] + w["h"]) - max(ry0, w["y"])
            if ov <= 0 or ov / min(rh, w["h"]) < 0.5:
                continue
            if max(rh, w["h"]) / max(min(rh, w["h"]), 1.0) > 2.5:
                continue                     # very different letter heights
            row.append(w)
            placed = True
            break
        if not placed:
            rows.append([w])
    out = []
    for row in rows:
        row.sort(key=lambda w: w["x"])
        med_h = sorted(w["h"] for w in row)[len(row) // 2]
        seg = [row[0]]
        for a, b in zip(row, row[1:]):
            gap = b["x"] - (a["x"] + a["w"])
            if gap > 3.0 * med_h or gap < -0.5 * a["w"]:   # far apart / heavy overlap
                out.append(seg)
                seg = [b]
            else:
                seg.append(b)
        out.append(seg)
    return [s for s in out if len(s) >= 2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", type=int, default=0)
    args = ap.parse_args()

    by_img: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with (V3 / "labels.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            t = (r["text"] or "").strip()
            if r.get("class") != "word" or not t or " " in t:
                continue
            try:
                x, y, w, h = (float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"]))
            except ValueError:
                continue
            if w <= 0 or h <= 0:
                continue
            by_img[(r["source_image"], r["split"])].append(
                {"text": t, "x": x, "y": y, "w": w, "h": h})

    txts = {s: [] for s in ("train", "val", "test")}
    stats = defaultdict(int)
    previews = []
    for (src_name, split), words in by_img.items():
        img_path = SRC_IMG / src_name
        if not img_path.exists():
            stats["no_src_img"] += 1
            continue
        lines = cluster_lines(words)
        if not lines:
            continue
        img = None
        for li, line in enumerate(lines):
            label = " ".join(w["text"] for w in line)
            if len(label) > MAX_LABEL:
                stats["too_long"] += 1
                continue
            x0 = min(w["x"] for w in line)
            y0 = min(w["y"] for w in line)
            x1 = max(w["x"] + w["w"] for w in line)
            y1 = max(w["y"] + w["h"] for w in line)
            bw, bh = x1 - x0, y1 - y0
            if bh < MIN_H or bw / max(bh, 1) > MAX_RATIO:
                stats["bad_geom"] += 1
                continue
            if img is None:
                img = Image.open(img_path).convert("RGB")
            px, py = bw * 0.04, bh * 0.04
            crop = img.crop((max(0, int(x0 - px)), max(0, int(y0 - py)),
                             min(img.width, int(x1 + px)), min(img.height, int(y1 + py))))
            if crop.height > MAX_OUT_H:
                crop = crop.resize((max(1, round(crop.width * MAX_OUT_H / crop.height)),
                                    MAX_OUT_H), Image.LANCZOS)
            rel = f"crops_lines/{split}/{Path(src_name).stem}__line_{li:02d}.jpg"
            out_path = V3 / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            crop.save(out_path, quality=95)
            txts[split].append(f"{rel}\t{label}")
            stats[f"lines_{split}"] += 1
            stats[f"words_in_lines"] += len(line)
            if args.preview and len(previews) < args.preview and split == "train":
                previews.append((rel, label))

    for split, rows in txts.items():
        p = V3 / f"lines_{split}.txt"
        p.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        print(f"[OUT] {p.name}: {len(rows)} lines")
    print(f"[STATS] {dict(stats)}")
    for rel, label in previews:
        print(f"  preview: {rel}  '{label}'")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
