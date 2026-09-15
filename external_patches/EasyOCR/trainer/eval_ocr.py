#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate CER / WER / exact-match for the trained TrOCR and EasyOCR (CRNN-CTC)
recognizers on the same validation split.

Run from the trainer/ directory (it imports trainer modules):
  python eval_ocr.py \
    --manifest ../../../artifacts/ocr_training/signboard/labels.csv \
    --trocr ../../../artifacts/ocr_training/signboard/trocr_model \
    --easyocr-pth saved_models/signboard/best_accuracy.pth \
    --config config_files/signboard.yaml
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
from nltk.metrics.distance import edit_distance

from utils import CTCLabelConverter
from model import Model
from dataset import AlignCollate
from run_easyocr_train import get_config


def load_val_rows(manifest: Path, split: str = "val"):
    root = manifest.resolve().parent
    rows = []
    for r in csv.DictReader(manifest.open("r", encoding="utf-8")):
        if r["split"] == split:
            rows.append((root / r["image_path"], r["text"]))
    return rows


def metrics(pairs):
    """pairs: list of (pred, gt). Returns CER, WER, exact-acc."""
    char_dist = char_tot = 0
    word_dist = word_tot = 0
    exact = 0
    for pred, gt in pairs:
        char_dist += edit_distance(pred, gt)
        char_tot += max(1, len(gt))
        gw, pw = gt.split(), pred.split()
        word_dist += edit_distance(pw, gw)
        word_tot += max(1, len(gw))
        exact += int(pred == gt)
    n = len(pairs)
    return {
        "CER": char_dist / char_tot,
        "WER": word_dist / word_tot,
        "exact_acc": exact / n,
        "n": n,
    }


@torch.no_grad()
def eval_trocr(rows, model_dir: Path, device, batch_size=16, max_len=64):
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    proc = TrOCRProcessor.from_pretrained(model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir).eval().to(device)
    pairs = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        imgs = [Image.open(p).convert("RGB") for p, _ in chunk]
        pv = proc(images=imgs, return_tensors="pt").pixel_values.to(device)
        ids = model.generate(pv, max_length=max_len)
        preds = proc.batch_decode(ids, skip_special_tokens=True)
        for (p, gt), pred in zip(chunk, preds):
            pairs.append((pred.strip(), gt))
    return pairs


@torch.no_grad()
def eval_easyocr(rows, config_path: str, pth: Path, device, batch_size=32):
    opt = get_config(config_path)
    converter = CTCLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    model = torch.nn.DataParallel(model).to(device)
    model.load_state_dict(torch.load(pth, map_location=device, weights_only=False))
    model.eval()

    align = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD,
                         contrast_adjust=opt.contrast_adjust)
    pairs = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        pil = [(Image.open(p).convert("L"), gt) for p, gt in chunk]
        image_tensors, _ = align([(img, gt) for img, gt in pil])
        image = image_tensors.to(device)
        bs = image.size(0)
        text_for_pred = torch.LongTensor(bs, opt.batch_max_length + 1).fill_(0).to(device)
        preds = model(image, text_for_pred)
        preds_size = torch.IntTensor([preds.size(1)] * bs)
        _, preds_index = preds.max(2)
        preds_str = converter.decode_greedy(preds_index.view(-1).data, preds_size.data)
        for (p, gt), pred in zip(chunk, preds_str):
            pairs.append((pred, gt))
    return pairs


def show_examples(name, pairs, k=8):
    print(f"\n  [{name}] examples (GT -> PRED):")
    for pred, gt in pairs[:k]:
        flag = "OK " if pred == gt else "XX "
        print(f"    {flag}{gt!r:18} -> {pred!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--trocr", type=Path, default=None)
    ap.add_argument("--easyocr-pth", type=Path, default=None)
    ap.add_argument("--config", default="config_files/signboard.yaml")
    ap.add_argument("--limit", type=int, default=0, help="cap val rows (0=all)")
    ap.add_argument("--split", default="val", help="manifest split to evaluate (val|test)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_val_rows(args.manifest, args.split)
    if args.limit:
        rows = rows[:args.limit]
    print(f"[INFO] val rows: {len(rows)}  device: {device}")

    if args.trocr:
        print("\n[INFO] evaluating TrOCR ...")
        pairs = eval_trocr(rows, args.trocr, device)
        m = metrics(pairs)
        print(f"  TrOCR    CER={m['CER']:.4f}  WER={m['WER']:.4f}  exact_acc={m['exact_acc']:.4f}  (n={m['n']})")
        show_examples("TrOCR", pairs)

    if args.easyocr_pth:
        print("\n[INFO] evaluating EasyOCR (CRNN-CTC) ...")
        pairs = eval_easyocr(rows, args.config, args.easyocr_pth, device)
        m = metrics(pairs)
        print(f"  EasyOCR  CER={m['CER']:.4f}  WER={m['WER']:.4f}  exact_acc={m['exact_acc']:.4f}  (n={m['n']})")
        show_examples("EasyOCR", pairs)


if __name__ == "__main__":
    main()
