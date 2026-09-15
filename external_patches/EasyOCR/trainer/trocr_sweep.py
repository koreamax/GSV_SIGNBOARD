#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sweep TrOCR checkpoints x decode settings; report CER/WER/exact to pick best."""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import torch
from PIL import Image
from nltk.metrics.distance import edit_distance
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


def load_val(manifest: Path, limit: int):
    root = manifest.resolve().parent
    rows = [(root / r["image_path"], r["text"])
            for r in csv.DictReader(manifest.open(encoding="utf-8")) if r["split"] == "val"]
    return rows[:limit] if limit else rows


def metrics(pairs):
    cd = ct = wd = wt = ex = 0
    for pred, gt in pairs:
        cd += edit_distance(pred, gt); ct += max(1, len(gt))
        wd += edit_distance(pred.split(), gt.split()); wt += max(1, len(gt.split()))
        ex += int(pred == gt)
    return cd / ct, wd / wt, ex / len(pairs)


@torch.no_grad()
def run(rows, model_dir, device, beams, bs=16, max_len=64):
    proc = TrOCRProcessor.from_pretrained(model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir).eval().to(device)
    pairs = []
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        imgs = [Image.open(p).convert("RGB") for p, _ in chunk]
        pv = proc(images=imgs, return_tensors="pt").pixel_values.to(device)
        kw = {"max_length": max_len}
        if beams > 1:
            kw.update(num_beams=beams, early_stopping=True)
        ids = model.generate(pv, **kw)
        preds = proc.batch_decode(ids, skip_special_tokens=True)
        pairs += [(pr.strip(), gt) for (_, gt), pr in zip(chunk, preds)]
    del model
    torch.cuda.empty_cache()
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--ckpt-root", type=Path, required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--beams", nargs="+", type=int, default=[1, 4])
    ap.add_argument("--limit", type=int, default=500)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_val(args.manifest, args.limit)
    print(f"val={len(rows)} device={device}")
    print(f"{'ckpt':12} {'beams':>5} {'CER':>8} {'WER':>8} {'exact':>8}")
    for ck in args.ckpts:
        mdir = args.ckpt_root / ck if ck != "final" else args.ckpt_root.parent / "trocr_model"
        for b in args.beams:
            cer, wer, ex = metrics(run(rows, mdir, device, b))
            print(f"{ck:12} {b:>5} {cer:>8.4f} {wer:>8.4f} {ex:>8.4f}")


if __name__ == "__main__":
    main()
