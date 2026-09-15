#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""High-resolution 5-Fold retraining for YOLO26x and Faster R-CNN (T2 후속).

Both models are retrained on the SAME folds as before, changing only the input
resolution, so the comparison stays like-for-like. Everything else (optimiser,
lr, patience, fold split) matches the original recipes.

  yolo26x : train_yolo26x_kfold.py 설정 + --imgsz
  frcnn   : train_frcnn_kfold.py 설정 + transform.min_size/max_size

`--probe` measures peak VRAM for one short step at the requested resolution and
batch size, so a 10GB card can be pushed to its limit without a mid-run OOM.

Usage:
  .venv/Scripts/python.exe train_hires_kfold.py --model yolo26x --imgsz 1280 --batch 1 --probe
  .venv/Scripts/python.exe train_hires_kfold.py --model yolo26x --imgsz 1280 --batch 1
  .venv/Scripts/python.exe train_hires_kfold.py --model frcnn --res 1024 --batch 1
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
FOLD_ROOT = "artifacts/yolo11x_kfold/total_fold{i}/dataset"
FOLDS = 5


# ----------------------------------------------------------------- YOLO26x
def run_yolo(args) -> None:
    from ultralytics import YOLO
    out = HERE / "artifacts" / f"yolo26x_kfold_r{args.imgsz}"
    results = []
    for i in range(args.fold_start, FOLDS):
        data = FOLD_ROOT.format(i=i) + "/data.yaml"
        t0 = time.time()
        YOLO("yolo26x.pt").train(
            data=data, epochs=args.epochs, patience=args.patience,
            batch=args.batch, imgsz=args.imgsz,
            optimizer="Adam", lr0=1e-4, device=0, seed=42,
            project=str(out), name=f"fold{i}", exist_ok=True,
            cache=False,           # hi-res RAM cache would thrash
            workers=8, plots=False, verbose=False,
        )
        rows = list(csv.DictReader(open(out / f"fold{i}" / "results.csv")))
        col = [c for c in rows[0] if "mAP50(B)" in c and "95" not in c][0]
        peak = max(float(r[col]) for r in rows)
        results.append((i, peak))
        print(f"[yolo26x@{args.imgsz}] fold{i} peak mAP50={peak:.4f} "
              f"({(time.time()-t0)/3600:.1f}h)", flush=True)
    _summary(out / "kfold_summary_map50.csv", results, "yolo26x")


# ----------------------------------------------------------------- FRCNN
def run_frcnn(args) -> None:
    import train_frcnn_kfold as T
    from torch.utils.data import DataLoader

    out = HERE / "artifacts" / f"frcnn_kfold_r{args.res}"
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for i in range(args.fold_start, FOLDS):
        root = FOLD_ROOT.format(i=i)
        tr = DataLoader(T.YoloFRCNN(f"{root}/images/train", f"{root}/labels/train"),
                        batch_size=args.batch, shuffle=True, num_workers=0,
                        collate_fn=T.collate)
        va = DataLoader(T.YoloFRCNN(f"{root}/images/val", f"{root}/labels/val"),
                        batch_size=1, shuffle=False, num_workers=0, collate_fn=T.collate)
        model = T.get_frcnn(2)
        model.transform.min_size = (args.res,)
        model.transform.max_size = int(args.res * 1333 / 800)
        model = model.to(T.DEVICE)
        opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                              lr=T.LR, momentum=0.9, weight_decay=5e-4)
        best_ap, best_ep, no_imp = -1.0, 0, 0
        ckpt = out / f"best_frcnn_fold{i}.pth"
        t0 = time.time()
        for ep in range(1, args.epochs + 1):
            model.train()
            for images, targets in tr:
                images = [im.to(T.DEVICE) for im in images]
                targets = [{k: v.to(T.DEVICE) for k, v in t.items()} for t in targets]
                if sum(len(t["boxes"]) for t in targets) == 0:
                    continue
                loss = sum(model(images, targets).values())
                opt.zero_grad(); loss.backward(); opt.step()
            ap = T.evaluate(model, va)
            if ap > best_ap + 1e-6:
                best_ap, best_ep, no_imp = ap, ep, 0
                torch.save(model.state_dict(), ckpt)
            else:
                no_imp += 1
                if no_imp >= args.patience:
                    break
        results.append((i, best_ap))
        print(f"[frcnn@{args.res}] fold{i} best AP@0.5={best_ap:.4f} (ep{best_ep}, "
              f"{(time.time()-t0)/3600:.1f}h)", flush=True)
    _summary(out / "kfold_summary_ap50.csv", results, "frcnn")


def _summary(path: Path, results, tag: str) -> None:
    if not results:
        return
    avg = sum(v for _, v in results) / len(results)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["fold", "best"])
        for i, v in results:
            w.writerow([i, f"{v:.4f}"])
        w.writerow(["AVG", f"{avg:.4f}"])
    print(f"[{tag}] DONE avg={avg:.4f} -> {path}", flush=True)


# ----------------------------------------------------------------- VRAM probe
def probe(args) -> None:
    """One forward+backward at the requested size; reports peak VRAM."""
    torch.cuda.reset_peak_memory_stats()
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    if args.model == "yolo26x":
        # A synthetic forward/backward cannot reproduce Ultralytics' grad setup,
        # so run ONE real epoch on fold0 — it also exercises the dataloader and
        # AMP exactly as the full run would.
        import tempfile
        from ultralytics import YOLO
        YOLO("yolo26x.pt").train(
            data=FOLD_ROOT.format(i=0) + "/data.yaml", epochs=1, batch=args.batch,
            imgsz=args.imgsz, optimizer="Adam", lr0=1e-4, device=0, seed=42,
            project=tempfile.mkdtemp(prefix="probe_"), name="p", exist_ok=True,
            cache=False, workers=4, plots=False, verbose=False, val=False,
        )
        label = f"yolo26x imgsz={args.imgsz}"
    else:
        import train_frcnn_kfold as T
        m = T.get_frcnn(2)
        m.transform.min_size = (args.res,); m.transform.max_size = int(args.res * 1333 / 800)
        m = m.to("cuda").train()
        imgs = [torch.randn(3, args.res, int(args.res * 1.7), device="cuda")
                for _ in range(args.batch)]
        tgts = [{"boxes": torch.tensor([[10.0, 10.0, 200.0, 200.0]], device="cuda"),
                 "labels": torch.ones(1, dtype=torch.int64, device="cuda")}
                for _ in range(args.batch)]
        sum(m(imgs, tgts).values()).backward()
        label = f"frcnn min_size={args.res}"
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"[probe] {label} batch={args.batch}: peak {peak:.2f} GiB / {total:.1f} GiB "
          f"({'OK' if peak < total * 0.85 else '위험 — batch 낮추거나 해상도 축소'})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["yolo26x", "frcnn"], required=True)
    ap.add_argument("--imgsz", type=int, default=1280, help="yolo26x 입력 해상도")
    ap.add_argument("--res", type=int, default=1024, help="frcnn min_size")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--fold-start", type=int, default=0, help="이어서 학습할 fold 인덱스")
    ap.add_argument("--probe", action="store_true", help="VRAM만 측정하고 종료")
    args = ap.parse_args()
    if args.probe:
        probe(args)
    elif args.model == "yolo26x":
        run_yolo(args)
    else:
        run_frcnn(args)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
