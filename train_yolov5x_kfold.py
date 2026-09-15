"""
YOLOv5x 5-Fold training — SAME setup as YOLO26x for a fair X-vs-X comparison.
- folds: artifacts/yolo11x_kfold/total_fold{i}/dataset/data.yaml
- imgsz 960, batch 2 (97M params, 10GB GPU), Adam lr0 1e-4, patience 100 (full convergence)
- per-fold best mAP@0.5 (Ultralytics val) -> 5-fold average.
"""
import csv
import os
from pathlib import Path
from ultralytics import YOLO

# D33: 환경변수로 fold 원천/출력 경로 교체 (group-aware 재학습용). 기본값 = 기존.
FOLD_ROOT = os.environ.get("FOLD_ROOT", "artifacts/yolo11x_kfold")
PROJECT = str(Path(os.environ.get("DET_PROJECT", "artifacts/yolov5x_kfold")).resolve())
FOLDS = 5
BATCH = 2
IMGSZ = 960


def fold_peak_map50(run_dir: Path) -> float:
    rows = list(csv.DictReader(open(run_dir / "results.csv")))
    col = [c for c in rows[0] if "mAP50(B)" in c][0]
    return max(float(r[col]) for r in rows)


def main():
    results = []
    for i in range(FOLDS):
        data = f"{FOLD_ROOT}/total_fold{i}/dataset/data.yaml"
        model = YOLO("yolov5xu.pt")
        model.train(
            data=data, epochs=400, patience=100, batch=BATCH, imgsz=IMGSZ,
            optimizer="Adam", lr0=1e-4, device=0, seed=42,
            project=PROJECT, name=f"fold{i}", exist_ok=True,
            cache="ram", workers=8, plots=False, verbose=False,
        )
        peak = fold_peak_map50(Path(PROJECT) / f"fold{i}")
        results.append((i, peak))
        print(f"[YOLOv5x-5fold] fold{i} best mAP@0.5 = {peak:.4f}", flush=True)

    avg = sum(p for _, p in results) / len(results)
    out = Path(PROJECT) / "kfold_summary_map50.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["fold", "best_mAP50"])
        for i, p in results:
            w.writerow([i, f"{p:.4f}"])
        w.writerow(["AVG", f"{avg:.4f}"])
    print(f"[YOLOv5x-5fold] DONE per-fold={[round(p,4) for _,p in results]} AVG mAP@0.5={avg:.4f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
