"""
YOLO text(word) detection 5-Fold. Usage: train_yolo_text_kfold.py <model.pt> <project_dir_name>
e.g. yolo26x.pt yolo26x_text_kfold  |  yolov5xu.pt yolov5x_text_kfold
imgsz 640, batch 4 (x models on 10GB), patience 50, same 5 folds (signboard_text/fold{i}.yaml).
"""
import sys, csv
from pathlib import Path
from ultralytics import YOLO

MODEL = sys.argv[1]
TAG = sys.argv[2]
PROJ = str(Path(f"artifacts/{TAG}").resolve())


def peak(rd: Path) -> float:
    rows = list(csv.DictReader(open(rd / "results.csv")))
    c = [x for x in rows[0] if "mAP50(B)" in x][0]
    return max(float(r[c]) for r in rows)


def main():
    res = []
    for i in range(5):
        m = YOLO(MODEL)
        m.train(data=f"artifacts/signboard_text/fold{i}.yaml", epochs=60, patience=15,
                batch=4, imgsz=640, optimizer="Adam", lr0=1e-4, device=0, seed=42,
                project=PROJ, name=f"fold{i}", exist_ok=True, workers=8, plots=False, verbose=False)
        p = peak(Path(PROJ) / f"fold{i}"); res.append((i, p))
        print(f"[{TAG}] fold{i} best mAP@0.5 = {p:.4f}", flush=True)
    avg = sum(p for _, p in res) / len(res)
    out = Path(PROJ) / "kfold_summary_map50.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["fold", "best_mAP50"])
        for i, p in res:
            w.writerow([i, f"{p:.4f}"])
        w.writerow(["AVG", f"{avg:.4f}"])
    print(f"[{TAG}] DONE per-fold={[round(p,4) for _,p in res]} AVG mAP@0.5={avg:.4f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
