#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnose data leakage / overfitting for the signboard recognizers.

Checks:
  1) Split integrity: are train and val DISJOINT by source image and by crop path?
  2) Label memorization: what fraction of val texts also occur verbatim in train,
     and how does each model do on SEEN-text vs UNSEEN-text val subsets?
     (A big seen >> unseen gap means headline accuracy is inflated by memorizing
      frequent words rather than truly generalizing.)
"""
from __future__ import annotations
import argparse, csv
from collections import Counter
from pathlib import Path
import torch
from nltk.metrics.distance import edit_distance

from eval_ocr import load_val_rows, eval_trocr, eval_easyocr


def read_manifest(manifest: Path):
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    return rows


def subset_metrics(pairs):
    cd = ct = ex = 0
    for pred, gt in pairs:
        cd += edit_distance(pred, gt); ct += max(1, len(gt)); ex += int(pred == gt)
    n = len(pairs)
    return {"n": n, "CER": cd / ct if ct else 0, "exact": ex / n if n else 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--trocr", type=Path, default=None)
    ap.add_argument("--easyocr-pth", type=Path, default=None)
    ap.add_argument("--config", default="config_files/signboard_full.yaml")
    args = ap.parse_args()

    rows = read_manifest(args.manifest)
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]

    # 1) split integrity
    tr_imgs = {r["source_image_id"] for r in train}
    va_imgs = {r["source_image_id"] for r in val}
    img_overlap = tr_imgs & va_imgs
    tr_paths = {r["image_path"] for r in train}
    va_paths = {r["image_path"] for r in val}
    path_overlap = tr_paths & va_paths

    print("=== 1) SPLIT INTEGRITY ===")
    print(f"train crops={len(train)} val crops={len(val)}")
    print(f"train source-images={len(tr_imgs)} val source-images={len(va_imgs)}")
    print(f"source-image overlap (LEAK if >0): {len(img_overlap)}")
    print(f"crop-path overlap   (LEAK if >0): {len(path_overlap)}")

    # 2) label memorization
    train_text = Counter(r["text"] for r in train)
    val_texts = [r["text"] for r in val]
    seen_flags = [t in train_text for t in val_texts]
    n_seen = sum(seen_flags)
    print("\n=== 2) LABEL OVERLAP ===")
    print(f"val texts that appear verbatim in train: {n_seen}/{len(val)} = {n_seen/len(val):.3f}")
    print(f"unique train texts={len(train_text)}  unique val texts={len(set(val_texts))}")
    uniq_val_unseen = sum(1 for t in set(val_texts) if t not in train_text)
    print(f"unique val texts NOT in train: {uniq_val_unseen}/{len(set(val_texts))}")

    # 3) per-model seen vs unseen accuracy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_rows = load_val_rows(args.manifest)  # list[(path, text)]
    seen_set = set(t for t in val_texts if t in train_text)

    def split_seen(pairs):
        seen = [(p, g) for p, g in pairs if g in seen_set]
        unseen = [(p, g) for p, g in pairs if g not in seen_set]
        return seen, unseen

    if args.trocr:
        print("\n=== 3a) TrOCR seen-vs-unseen ===")
        pairs = eval_trocr(val_rows, args.trocr, device)
        seen, unseen = split_seen(pairs)
        print(f"  ALL    {subset_metrics(pairs)}")
        print(f"  SEEN   {subset_metrics(seen)}")
        print(f"  UNSEEN {subset_metrics(unseen)}")

    if args.easyocr_pth:
        print("\n=== 3b) EasyOCR seen-vs-unseen ===")
        pairs = eval_easyocr(val_rows, args.config, args.easyocr_pth, device)
        seen, unseen = split_seen(pairs)
        print(f"  ALL    {subset_metrics(pairs)}")
        print(f"  SEEN   {subset_metrics(seen)}")
        print(f"  UNSEEN {subset_metrics(unseen)}")


if __name__ == "__main__":
    main()
