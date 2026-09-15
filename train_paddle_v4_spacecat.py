#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A/B experiment: retrain Paddle rec with space-aware RecConAug (v4_spacecat).

Hypothesis (D18): v3's RecConAug concatenates crops with `label += ext_label`
(NO space, ~36% of samples/epoch), teaching the model to join adjacent words —
matching the GSV symptom of space-less multi-word outputs. v4 inserts a visual
gap (0.15~0.40 x h, edge-blended) AND a " " in the label at each join.

Mechanics: the patch lives INSIDE the venv PaddleOCR repo
(ppocr/data/imaug/rec_img_aug.py, marked "[GSV spacecat patch]", original kept
as rec_img_aug.py.orig_v3) and is gated by env RECCON_SPACECAT=1 — a monkeypatch
would not survive Windows DataLoader worker re-imports. Default env keeps stock
behavior, so v3 reproduction is unaffected.

Everything else (data, splits, pretrained init, epochs, lr, yml) is identical
to v3: A = output/paddle_signboard_rec_v3 (already trained), B = this run.

Usage:
  .venv/Scripts/python.exe train_paddle_v4_spacecat.py --preview   # eyeball merges, no GPU
  .venv/Scripts/python.exe train_paddle_v4_spacecat.py --train     # ~2h40m on this GPU
  .venv/Scripts/python.exe train_paddle_v4_spacecat.py --export    # best_accuracy -> inference/
"""
from __future__ import annotations

import argparse
import os
import random
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE / ".venv/Lib/site-packages/paddlex/repo_manager/repos/PaddleOCR"
YML = HERE / "paddle_signboard_rec_v4_spacecat.yml"
V3DATA = HERE / "artifacts" / "ocr_training" / "signboard_v3"


def check_patch() -> None:
    src = (REPO / "ppocr/data/imaug/rec_img_aug.py").read_text(encoding="utf-8")
    if "[GSV spacecat patch]" not in src:
        sys.exit("[ABORT] rec_img_aug.py lacks the spacecat patch — re-apply it first.")
    print("[OK] spacecat patch present, RECCON_SPACECAT=1")


def preview(n: int = 6) -> None:
    """Standalone replica of the patched merge on real train crops (no paddle)."""
    import cv2
    import numpy as np

    rows = []
    with (V3DATA / "train.txt").open(encoding="utf-8") as f:
        for line in f:
            if "\t" in line:
                p, t = line.rstrip("\n").split("\t", 1)
                rows.append((p, t))
    rng = random.Random(42)
    out_dir = V3DATA / "preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    made = 0
    while made < n:
        (pa, ta), (pb, tb) = rng.choice(rows), rng.choice(rows)
        ia, ib = cv2.imread(str(V3DATA / pa)), cv2.imread(str(V3DATA / pb))
        if ia is None or ib is None or len(ta) + len(tb) + 1 > 25:
            continue
        if ia.shape[1] / ia.shape[0] + ib.shape[1] / ib.shape[0] > 320 / 48:
            continue
        h = 48
        a = cv2.resize(ia, (round(ia.shape[1] / ia.shape[0] * h), h))
        b = cv2.resize(ib, (round(ib.shape[1] / ib.shape[0] * h), h))
        gap_w = max(2, int(h * rng.uniform(0.15, 0.40)))
        left, right = a[:, -1:, :].astype(np.float32), b[:, :1, :].astype(np.float32)
        t = np.linspace(0, 1, gap_w, dtype=np.float32)[None, :, None]
        gap = (left * (1 - t) + right * t).astype(a.dtype)
        old = np.concatenate([a, b], axis=1)
        new = np.concatenate([a, gap, b], axis=1)
        cv2.imwrite(str(out_dir / f"spacecat_{made}_v3old.jpg"), old)
        cv2.imwrite(str(out_dir / f"spacecat_{made}_v4new.jpg"), new)
        print(f"  [{made}] v3 label: '{ta}{tb}'   ->   v4 label: '{ta} {tb}'  (gap {gap_w}px)")
        made += 1
    print(f"[PREVIEW] {n} pairs -> {out_dir}\\spacecat_*.jpg")


def run_tool(tool: str, argv: list[str]) -> None:
    check_patch()
    sys.path.insert(0, str(REPO))
    os.chdir(HERE)      # yml save paths are relative — resolve them to main/, not the repo
    sys.argv = [tool] + argv
    runpy.run_path(str(REPO / "tools" / tool), run_name="__main__")


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--preview", action="store_true")
    g.add_argument("--train", action="store_true")
    g.add_argument("--export", action="store_true")
    ap.add_argument("--config", type=Path, default=YML,
                    help="training yml (default: v4 spacecat)")
    ap.add_argument("--stock", action="store_true",
                    help="do NOT enable the spacecat patch (stock v3 recipe — "
                         "e.g. the v3b run-to-run variance rerun)")
    args = ap.parse_args()

    # env is inherited by DataLoader worker processes
    os.environ["RECCON_SPACECAT"] = "0" if args.stock else "1"
    print(f"[CFG] config={args.config.name}  RECCON_SPACECAT={os.environ['RECCON_SPACECAT']}")

    import yaml
    out = Path(yaml.safe_load(args.config.read_text(encoding="utf-8"))["Global"]["save_model_dir"])
    out = out if out.is_absolute() else HERE / out

    if args.preview:
        check_patch()
        preview()
    elif args.train:
        run_tool("train.py", ["-c", str(args.config)])
    else:
        run_tool("export_model.py", [
            "-c", str(args.config),
            "-o", f"Global.pretrained_model={out / 'best_accuracy'}",
            f"Global.save_inference_dir={out / 'inference'}",
        ])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
