#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tune YOLOv11x using ONLY artifacts/archive dataset.
- archive는 이미 YOLO 포맷(train/valid/test + labels txt)이라 변환 불필요
- data.yml 그대로 사용
- GPU 강제 사용
"""

import shutil
from pathlib import Path
from ultralytics import YOLO
import torch


# ============================================================
# 평가용 mAP@0.5 (GPU 강제)
# ============================================================
def eval_map50(model_path: Path, data_yaml: Path, imgsz=640, device=0):
    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        verbose=False,
        device=device
    )
    return float(metrics.box.map50)


# ============================================================
# Fine-tune on archive only
# ============================================================
def finetune_archive_only(
        artifacts_dir="artifacts",
        archive_dir="artifacts/archive",
        out_dir="artifacts/yolo_ft_archive",
        epochs=50,
        batch=8,
        imgsz=1280,
        lr=1e-4,
        device=0
):
    BASE_DIR = Path(__file__).resolve().parents[1]  # ✅ main 폴더
    artifacts_dir = (BASE_DIR / artifacts_dir).resolve()
    archive_dir   = (BASE_DIR / archive_dir).resolve()
    out_dir       = (BASE_DIR / out_dir).resolve()

    best_model_path = artifacts_dir / "yolo" / "best_yolo.pt"
    if not best_model_path.exists():
        raise RuntimeError("best_yolo.pt not found in artifacts/yolo/")

    data_yaml = archive_dir / "data.yaml"
    if not data_yaml.exists():
        raise RuntimeError(f"data.yaml not found: {data_yaml}")

    # ✅ GPU 강제 체크
    print("\n[GPU CHECK]")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. GPU 강제 사용 불가.")
    print("CUDA available: True")
    print("GPU:", torch.cuda.get_device_name(device))

    print("\n[1] Using ARCHIVE dataset only")
    print("data.yaml =", data_yaml)

    # --- old model eval ---
    print("\n[2] Evaluate old model on ARCHIVE (GPU)...")
    old_map = eval_map50(best_model_path, data_yaml, imgsz, device=device)
    print(f"[MY-OLD] Old best mAP@0.5 = {old_map:.6f}")

    # --- fine-tune ---
    print("\n[3] Fine-tuning on ARCHIVE (GPU)...")
    model = YOLO(str(best_model_path))
    run = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        lr0=lr,
        pretrained=True,
        optimizer="Adam",
        verbose=True,
        save=True,
        device=device,
        project=str(out_dir.parent),
        name=out_dir.name
    )

    new_best = Path(run.save_dir) / "weights" / "best.pt"

    # --- new model eval ---
    print("\n[4] Evaluate new model on ARCHIVE (GPU)...")
    new_map = eval_map50(new_best, data_yaml, imgsz, device=device)
    print(f"[MY-NEW] Fine-tuned model mAP@0.5 = {new_map:.6f}")

    # debug
    print(f"\n[DEBUG] old_map raw={old_map!r}")
    print(f"[DEBUG] new_map raw={new_map!r}")
    print(f"[DEBUG] new-old = {new_map - old_map}")

    # --- update rule ---
    if new_map > old_map:
        shutil.copy2(new_best, best_model_path)
        print("[UPDATE] New model is better → best_yolo.pt replaced.")
    else:
        print("[KEEP] Old model is better → keep old best_yolo.pt.")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", type=int, default=0, help="GPU index. default=0")

    # archive 경로 커스텀 가능하게만 열어둠(기본은 artifacts/archive)
    ap.add_argument("--archive_dir", default="artifacts/archive")
    ap.add_argument("--out_dir", default="artifacts/yolo_ft_archive")
    args = ap.parse_args()

    finetune_archive_only(
        archive_dir=args.archive_dir,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr=args.lr,
        device=args.device
    )
