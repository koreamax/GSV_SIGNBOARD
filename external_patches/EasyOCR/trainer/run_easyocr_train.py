#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Driver for the EasyOCR (deep-text-recognition style) recognition trainer.

Mirrors trainer.ipynb's get_config + train(opt), but as a plain script with a
__main__ guard so it works on Windows. Run from the trainer/ directory:

  python run_easyocr_train.py --config config_files/traffic_sign_small.yaml
"""
from __future__ import annotations

import argparse
import os

import pandas as pd
import torch.backends.cudnn as cudnn
import yaml

from train import train
from utils import AttrDict


def get_config(file_path: str) -> AttrDict:
    with open(file_path, "r", encoding="utf8") as stream:
        opt = yaml.safe_load(stream)
    opt = AttrDict(opt)
    if opt.lang_char == "None":
        # Derive charset from BOTH train (select_data) AND validation labels, so
        # the CTC converter covers every character that can appear at eval time.
        # Otherwise a val-only character raises KeyError during converter.encode.
        csv_paths = [
            os.path.join(opt["train_data"], data, "labels.csv")
            for data in opt["select_data"].split("-")
        ]
        csv_paths.append(os.path.join(opt["valid_data"], "labels.csv"))
        charset = set()
        for csv_path in csv_paths:
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(
                csv_path,
                sep="^([^,]+),",
                engine="python",
                usecols=["filename", "words"],
                keep_default_na=False,
            )
            charset.update("".join(df["words"]))
        opt.character = "".join(sorted(charset))
    else:
        opt.character = opt.number + opt.symbol + opt.lang_char
    os.makedirs(f"./saved_models/{opt.experiment_name}", exist_ok=True)
    # persist the resolved charset so the EasyOCR plug-in yaml can reuse it
    with open(f"./saved_models/{opt.experiment_name}/character.txt", "w", encoding="utf8") as f:
        f.write(opt.character)
    return opt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--amp", action="store_true", help="mixed precision (GPU)")
    args = ap.parse_args()

    cudnn.benchmark = True
    cudnn.deterministic = False

    opt = get_config(args.config)
    print(f"[INFO] num classes (chars incl. CTC blank handled in converter): {len(opt.character)}")
    train(opt, amp=args.amp)


if __name__ == "__main__":
    main()
