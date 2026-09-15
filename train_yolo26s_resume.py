"""
Resume the interrupted YOLO26s run to its natural end (early-stop when no more best).
Continues from last.pt (ep163) toward epochs=400 with patience=100.
"""
from pathlib import Path
from ultralytics import YOLO

LAST = "artifacts/yolo26s_signboard/train/weights/last.pt"


def main():
    model = YOLO(LAST)
    model.train(resume=True)
    best_pt = Path(model.trainer.best)
    m = YOLO(str(best_pt)).val(data="artifacts/yolo_ft_total/data.yaml", imgsz=1280,
                               device=0, workers=0, verbose=False)
    print(f"[YOLO26s-resume] FINAL best.pt mAP@0.5 = {float(m.box.map50):.4f} -> {best_pt}", flush=True)


if __name__ == "__main__":
    main()
