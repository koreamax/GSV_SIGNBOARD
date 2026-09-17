#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A1/T4: per-region detection mAP@0.5 using Ultralytics val() itself.

Why not just compute AP ourselves: a hand-rolled AP on model.predict() output
lands ~0.03-0.04 below Ultralytics val() on identical weights/images. The gap is
NOT the AP convention (101-pt == VOC here), the NMS IoU, the GT boxes, or the
aggregation unit — all were checked — so it comes from val()'s own inference
path (rectangular batching / letterbox). Rather than approximate it, this script
calls val() directly so Table 3's per-region column uses exactly the protocol
that produced the 5-fold numbers in Table 2.

Leakage control is unchanged: every photo is scored only by the fold model that
held it out. Each (fold, region) val subset becomes its own tiny dataset dir.

Usage:
  .venv/Scripts/python.exe eval_det_oof_ultra.py --kfold artifacts/yolo26x_kfold
"""
from __future__ import annotations

import argparse
import os
import csv
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
FOLD_DATA = HERE / os.environ.get("FOLD_ROOT", "artifacts/yolo11x_kfold")   # D33: group-aware fold 지원
REGIONS = ["gangnam", "brooklyn", "suwon"]
NAME_RE = re.compile(r"^total__([a-z]+)__(\d+)\.jpg$", re.I)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kfold", required=True)
    ap.add_argument("--name-pat", default="fold{i}")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--out", default="artifacts/gt/det_oof_per_region_ultra.csv")
    args = ap.parse_args()

    from ultralytics import YOLO

    tmp = Path(tempfile.mkdtemp(prefix="oof_ultra_"))
    # region -> list of (fold, mAP50, n_instances)
    per_region: dict[str, list] = {r: [] for r in REGIONS}
    per_fold_all: list[float] = []

    for fi in range(args.folds):
        vald = FOLD_DATA / f"total_fold{fi}" / "dataset" / "images" / "val"
        labd = FOLD_DATA / f"total_fold{fi}" / "dataset" / "labels" / "val"
        wpath = Path(args.kfold) / args.name_pat.format(i=fi) / "weights" / "best.pt"
        if not vald.is_dir() or not wpath.exists():
            sys.exit(f"[ABORT] missing fold data or weights for fold{fi}")
        model = YOLO(str(wpath))

        for region in REGIONS:
            imgs = sorted(p for p in vald.glob("*.jpg")
                          if (m := NAME_RE.match(p.name)) and m.group(1).lower() == region)
            if not imgs:
                continue
            root = tmp / f"f{fi}_{region}"
            (root / "images" / "val").mkdir(parents=True, exist_ok=True)
            (root / "labels" / "val").mkdir(parents=True, exist_ok=True)
            n_inst = 0
            for p in imgs:
                shutil.copy2(p, root / "images" / "val" / p.name)
                lp = labd / (p.stem + ".txt")
                if lp.exists():
                    shutil.copy2(lp, root / "labels" / "val" / lp.name)
                    n_inst += len([l for l in lp.read_text().strip().split("\n") if l.strip()])
            (root / "data.yaml").write_text(
                f"path: {root.as_posix()}\ntrain: images/val\nval: images/val\n"
                f"nc: 1\nnames:\n- signboard\n", encoding="utf-8")
            res = model.val(data=str(root / "data.yaml"), imgsz=args.imgsz,
                            batch=args.batch, verbose=False, plots=False)
            m50 = float(res.box.map50)
            per_region[region].append((fi, m50, n_inst))
            print(f"[fold{fi}/{region}] imgs={len(imgs)} inst={n_inst} mAP50={m50:.4f}",
                  flush=True)

    print(f"\n=== Per-region OOF mAP@0.5 (Ultralytics val(), {args.kfold}) ===")
    print(f"{'region':<10} {'folds':>6} {'inst':>6} {'mAP@.5 (가중)':>14} {'단순평균':>10}")
    rows = []
    for r in REGIONS:
        entries = per_region[r]
        inst = sum(n for _, _, n in entries)
        wavg = sum(m * n for _, m, n in entries) / max(inst, 1)
        savg = float(np.mean([m for _, m, _ in entries])) if entries else 0.0
        print(f"{r:<10} {len(entries):>6} {inst:>6} {wavg:>14.4f} {savg:>10.4f}")
        rows.append([r, len(entries), inst, f"{wavg:.4f}", f"{savg:.4f}"])
    all_inst = sum(sum(n for _, _, n in per_region[r]) for r in REGIONS)
    all_w = sum(m * n for r in REGIONS for _, m, n in per_region[r]) / max(all_inst, 1)
    print("-" * 52)
    print(f"{'ALL':<10} {'':>6} {all_inst:>6} {all_w:>14.4f}")
    out = Path(args.out)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["region", "folds", "instances", "mAP50_inst_weighted", "mAP50_simple_mean"])
        w.writerows(rows)
        w.writerow(["ALL", "", all_inst, f"{all_w:.4f}", ""])
    print(f"[saved] {out}")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
