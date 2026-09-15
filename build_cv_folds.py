#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build K-fold cross-validation splits for the signboard recognizer, partitioned
BY SOURCE IMAGE so that all crops of one signboard image stay in a single fold
(no leakage / no mixing across folds).

From signboard_full/labels.csv it writes, per fold k:
  <cv_root>/fold{k}/train/labels.csv   (trainer format: filename,words)
  <cv_root>/fold{k}/val/labels.csv
  <cv_root>/fold{k}/manifest.csv       (image_path[abs],text,split) for eval_ocr
And a per-fold trainer config:
  <trainer>/config_files/cv_fold{k}.yaml

Verifies that no source image appears in more than one fold.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


CONFIG_TEMPLATE = """number: '0123456789'
symbol: "!\\"#$%&'()*+,-./:;<=>?@[\\\\]^_`{{|}}~ €"
lang_char: 'None'
experiment_name: 'cv_fold{k}'
train_data: '{cv_rel}/fold{k}'
valid_data: '{cv_rel}/fold{k}/val'
manualSeed: 1111
workers: 0
batch_size: 32
num_iter: {num_iter}
valInterval: {val_interval}
saved_model: ''
FT: False
optim: False
lr: 1.
beta1: 0.9
rho: 0.95
eps: 0.00000001
grad_clip: 5
select_data: 'train'
batch_ratio: '1'
total_data_usage_ratio: 1.0
batch_max_length: 64
imgH: 64
imgW: 600
rgb: False
contrast_adjust: 0.0
sensitive: True
PAD: True
data_filtering_off: False
augment: {augment}
Transformation: 'None'
FeatureExtraction: 'VGG'
SequenceModeling: 'BiLSTM'
Prediction: 'CTC'
num_fiducial: 20
input_channel: 1
output_channel: 256
hidden_size: 256
decode: 'greedy'
new_prediction: False
freeze_FeatureFxtraction: False
freeze_SequenceModeling: False
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True, help="signboard_full/labels.csv")
    ap.add_argument("--cv-root", type=Path, required=True, help="output trainer dir for folds")
    ap.add_argument("--config-dir", type=Path, required=True, help="trainer config_files dir")
    ap.add_argument("--cv-rel", required=True, help="cv-root path relative to trainer cwd (for yaml)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-iter", type=int, default=30000)
    ap.add_argument("--val-interval", type=int, default=5000)
    ap.add_argument("--augment", default="False")
    args = ap.parse_args()

    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8")))
    crop_root = args.manifest.resolve().parent

    # group crops by source image, assign each image to one fold
    images = sorted({r["source_image_id"] for r in rows})
    rng = random.Random(args.seed)
    rng.shuffle(images)
    fold_of = {img: (i % args.k) for i, img in enumerate(images)}

    # bucket rows by fold
    by_fold = {k: [] for k in range(args.k)}
    for r in rows:
        by_fold[fold_of[r["source_image_id"]]].append(r)

    args.cv_root.mkdir(parents=True, exist_ok=True)
    args.config_dir.mkdir(parents=True, exist_ok=True)

    print(f"total crops={len(rows)} images={len(images)} k={args.k}")
    # integrity: each image in exactly one fold
    seen_imgs = {}
    for k in range(args.k):
        for r in by_fold[k]:
            sid = r["source_image_id"]
            if sid in seen_imgs and seen_imgs[sid] != k:
                raise RuntimeError(f"LEAK: image {sid} in folds {seen_imgs[sid]} and {k}")
            seen_imgs[sid] = k

    def abs_path(r):
        return (crop_root / r["image_path"]).resolve().as_posix()

    for k in range(args.k):
        val_rows = by_fold[k]
        train_rows = [r for kk in range(args.k) if kk != k for r in by_fold[kk]]
        fold_dir = args.cv_root / f"fold{k}"
        (fold_dir / "train").mkdir(parents=True, exist_ok=True)
        (fold_dir / "val").mkdir(parents=True, exist_ok=True)

        # trainer-format labels.csv
        for name, rws in (("train", train_rows), ("val", val_rows)):
            with (fold_dir / name / "labels.csv").open("w", encoding="utf-8", newline="\n") as f:
                f.write("filename,words\n")
                for r in rws:
                    f.write(f"{abs_path(r)},{r['text']}\n")

        # eval manifest (CropRow-ish: image_path absolute, text, split)
        with (fold_dir / "manifest.csv").open("w", encoding="utf-8", newline="\n") as f:
            w = csv.writer(f)
            w.writerow(["image_path", "text", "split"])
            for r in train_rows:
                w.writerow([abs_path(r), r["text"], "train"])
            for r in val_rows:
                w.writerow([abs_path(r), r["text"], "val"])

        # per-fold config
        cfg = CONFIG_TEMPLATE.format(
            k=k, cv_rel=args.cv_rel, num_iter=args.num_iter,
            val_interval=args.val_interval, augment=args.augment,
        )
        (args.config_dir / f"cv_fold{k}.yaml").write_text(cfg, encoding="utf-8")

        val_imgs = len({r["source_image_id"] for r in val_rows})
        print(f"fold{k}: train_crops={len(train_rows)} val_crops={len(val_rows)} val_imgs={val_imgs}")

    print("[OK] folds built, no image spans two folds")


if __name__ == "__main__":
    main()
