"""
YOLO26n signboard detection training.
- Pretrained yolo26n.pt fine-tuned on GSV signboard data (artifacts/yolo_ft_total).
- best.pt is auto-updated by Ultralytics whenever val mAP improves.
- Early stopping via patience.
Mirrors the YOLO11x finetune config (imgsz=1280, Adam, lr=1e-4) for comparability.
"""
from ultralytics import YOLO


def main():
    model = YOLO("yolo26n.pt")
    model.train(
        data="artifacts/yolo_ft_total/data.yaml",
        epochs=300,
        patience=50,          # early stopping
        batch=16,
        imgsz=1280,
        optimizer="Adam",
        lr0=1e-4,
        device=0,
        seed=42,
        project="artifacts/yolo26n_signboard",
        name="train",
        exist_ok=True,
        save=True,            # keep best.pt + last.pt
        plots=True,
        workers=4,
        verbose=True,
    )
    # final val with best.pt
    best = YOLO("artifacts/yolo26n_signboard/train/weights/best.pt")
    metrics = best.val(data="artifacts/yolo_ft_total/data.yaml", imgsz=1280,
                       device=0, workers=0, verbose=False)
    print(f"[YOLO26n] FINAL best.pt mAP@0.5 = {float(metrics.box.map50):.4f}")


if __name__ == "__main__":
    main()
