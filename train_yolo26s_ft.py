"""
YOLO26s fine-tune from the plateaued best.pt (mAP50=0.8697) to push past YOLOv5s (0.875).
Low LR + full cosine decay anneal — same trick that lifted YOLO11x on resume.
"""
from pathlib import Path
from ultralytics import YOLO

BEST = "artifacts/yolo26s_signboard/train/weights/best.pt"
PROJECT = str(Path("artifacts/yolo26s_signboard").resolve())


def main():
    model = YOLO(BEST)
    model.train(
        data="artifacts/yolo_ft_total/data.yaml",
        epochs=150,
        patience=80,
        batch=8,
        imgsz=1280,
        optimizer="Adam",
        lr0=5e-5,           # gentle fine-tune
        lrf=0.01,           # decay to ~5e-7 by the end (cosine)
        cos_lr=True,
        close_mosaic=20,    # disable mosaic for last 20 ep -> cleaner convergence
        device=0,
        seed=42,
        project=PROJECT,
        name="finetune",
        exist_ok=True,
        save=True,
        plots=True,
        workers=4,
        verbose=True,
    )
    best_pt = Path(model.trainer.best)
    m = YOLO(str(best_pt)).val(data="artifacts/yolo_ft_total/data.yaml", imgsz=1280,
                               device=0, workers=0, verbose=False)
    print(f"[YOLO26s-ft] FINAL best.pt mAP@0.5 = {float(m.box.map50):.4f} -> {best_pt}", flush=True)


if __name__ == "__main__":
    main()
