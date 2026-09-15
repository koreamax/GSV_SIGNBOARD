"""
YOLO26s 5-Fold training — SAME folds & protocol as YOLO11x for fair comparison.
- folds: artifacts/yolo11x_kfold/total_fold{i}/dataset/data.yaml (train 238/val 60 each)
- imgsz 1280, batch 8, Adam, lr0 1e-4, patience 50 (matches YOLO11x finetune)
- per-fold best mAP@0.5 (Ultralytics val) -> 5-fold average.
"""
import csv
from pathlib import Path
from ultralytics import YOLO

PROJECT = str(Path("artifacts/yolo26s_kfold").resolve())
FOLDS = 5


def fold_peak_map50(run_dir: Path) -> float:
    rows = list(csv.DictReader(open(run_dir / "results.csv")))
    col = [c for c in rows[0] if "mAP50(B)" in c][0]
    return max(float(r[col]) for r in rows)


def main():
    results = []
    for i in range(FOLDS):
        data = f"artifacts/yolo11x_kfold/total_fold{i}/dataset/data.yaml"
        model = YOLO("yolo26s.pt")
        model.train(
            data=data, epochs=200, patience=50, batch=8, imgsz=1280,
            optimizer="Adam", lr0=1e-4, device=0, seed=42,
            project=PROJECT, name=f"fold{i}", exist_ok=True,
            cache="ram", workers=8, plots=False, verbose=False,
        )
        peak = fold_peak_map50(Path(PROJECT) / f"fold{i}")
        results.append((i, peak))
        print(f"[YOLO26s-5fold] fold{i} best mAP@0.5 = {peak:.4f}", flush=True)

    avg = sum(p for _, p in results) / len(results)
    out = Path(PROJECT) / "kfold_summary_map50.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold", "best_mAP50"])
        for i, p in results:
            w.writerow([i, f"{p:.4f}"])
        w.writerow(["AVG", f"{avg:.4f}"])
    print(f"[YOLO26s-5fold] DONE per-fold={[round(p,4) for _,p in results]} AVG mAP@0.5={avg:.4f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
