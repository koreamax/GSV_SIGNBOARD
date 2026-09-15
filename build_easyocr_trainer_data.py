#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build EasyOCR trainer (deep-text-recognition style) data layout from a prepared
Text-in-the-Wild crop manifest (labels.csv produced by train_textinthewild_ocr.py).

Output layout (consumed by external/EasyOCR/trainer):
  <trainer_data>/train/labels.csv   header: filename,words   rows: <abs_crop_path>,<text>
  <trainer_data>/val/labels.csv

Images are referenced by absolute path (no copy). dataset.OCRDataset does
os.path.join(root, filename); an absolute filename is returned unchanged.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def build(manifest: Path, out_dir: Path) -> None:
    rows = list(csv.DictReader(manifest.open("r", encoding="utf-8", newline="")))
    crop_root = manifest.resolve().parent  # crops/<split>/... are relative to here

    charset: set[str] = set()
    counts = {"train": 0, "val": 0, "test": 0}
    for split in ("train", "val", "test"):
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        labels_path = split_dir / "labels.csv"
        with labels_path.open("w", encoding="utf-8", newline="\n") as f:
            f.write("filename,words\n")
            for r in rows:
                if r["split"] != split:
                    continue
                text = r["text"]
                abs_img = (crop_root / r["image_path"]).resolve()
                # custom regex sep splits on first comma only; path has no comma.
                f.write(f"{abs_img.as_posix()},{text}\n")
                counts[split] += 1
                charset.update(ch for ch in text if not ch.isspace())

    charset_str = "".join(sorted(charset))
    (out_dir / "charset.txt").write_text(charset_str, encoding="utf-8")
    print(f"[DONE] train={counts['train']} val={counts['val']} test={counts['test']} chars={len(charset)}")
    print(f"[DONE] layout -> {out_dir}")
    print(f"[DONE] charset -> {out_dir / 'charset.txt'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True, help="labels.csv from prepare step")
    ap.add_argument("--out-dir", type=Path, required=True, help="trainer all_data dir")
    args = ap.parse_args()
    build(args.manifest, args.out_dir)


if __name__ == "__main__":
    main()
