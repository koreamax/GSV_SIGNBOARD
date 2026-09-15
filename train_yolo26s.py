"""
YOLO26s signboard detection training (larger variant to fairly beat YOLOv5s).
- Pretrained yolo26s.pt fine-tuned on GSV signboard data (artifacts/yolo_ft_total).
- best.pt auto-updated whenever val mAP improves; early stopping via patience=100.
- imgsz 1280 for comparability with YOLOv5/YOLO11x.
"""
import os
from pathlib import Path
from ultralytics import YOLO

PROJECT = str(Path("artifacts/yolo26s_signboard").resolve())  # absolute -> avoid ~/runs prefix


def main():
    model = YOLO("yolo26s.pt")
    results = model.train(
        data="artifacts/yolo_ft_total/data.yaml",
        epochs=400,
        patience=100,         # increased early stopping
        batch=8,              # yolo26s heavier than nano -> safer batch at imgsz1280
        imgsz=1280,
        optimizer="Adam",
        lr0=1e-4,
        device=0,
        seed=42,
        project=PROJECT,
        name="train",
        exist_ok=True,
        save=True,
        plots=True,
        workers=4,
        verbose=True,
    )
    best_pt = Path(model.trainer.best)  # actual best.pt path
    print(f"[YOLO26s] best.pt -> {best_pt}")
    best = YOLO(str(best_pt))
    m = best.val(data="artifacts/yolo_ft_total/data.yaml", imgsz=1280,
                 device=0, workers=0, verbose=False)
    print(f"[YOLO26s] FINAL best.pt mAP@0.5 = {float(m.box.map50):.4f}", flush=True)


if __name__ == "__main__":
    main()
