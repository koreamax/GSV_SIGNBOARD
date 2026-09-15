"""
YOLO text(word) detection — hold-out (train/val/test = signboard_v3 source split, 전체 데이터).
Usage: train_yolo_text_holdout.py <model.pt> <project_dir>
  e.g. yolo26x.pt artifacts/yolo26x_text_holdout  |  yolov5xu.pt artifacts/yolov5x_text_holdout
하이퍼파라미터는 5k·5-fold 실험(train_yolo_text_kfold.py)과 동일: imgsz 640, batch 4,
Adam lr0 1e-4, epochs 60, patience 15. 증강은 Ultralytics 학습 루프 안에서만(mosaic 등).
평가는 eval_text_holdout.py 가 test 에서 통합 AP@0.5 로 합니다(여기서는 val 로 조기종료만).
SMOKE=1 이면 1 에폭·소량으로 배선만 확인합니다.
"""
import os
import sys
from pathlib import Path

from ultralytics import YOLO

MODEL, PROJECT = sys.argv[1], sys.argv[2]
DATA = "artifacts/signboard_text_holdout/data.yaml"
SMOKE = os.environ.get("SMOKE") == "1"

if __name__ == "__main__":
    data = DATA
    if SMOKE:   # 소량 리스트로 대체한 임시 yaml
        root = Path("artifacts/signboard_text_holdout")
        tr = sorted((root / "images/train").glob("*.jpg"))[:64]
        va = sorted((root / "images/val").glob("*.jpg"))[:32]
        (root / "smoke_train.txt").write_text("\n".join(p.resolve().as_posix() for p in tr))
        (root / "smoke_val.txt").write_text("\n".join(p.resolve().as_posix() for p in va))
        (root / "smoke.yaml").write_text("names:\n- text\nnc: 1\n"
                                         f"path: {root.resolve().as_posix()}\n"
                                         "train: smoke_train.txt\nval: smoke_val.txt\n")
        data = str(root / "smoke.yaml")
    m = YOLO(MODEL)
    m.train(data=data, epochs=1 if SMOKE else 60, patience=15, batch=4, imgsz=640,
            optimizer="Adam", lr0=1e-4, device=0, seed=42,
            project=str(Path(PROJECT).resolve()), name="run", exist_ok=True,
            workers=8, plots=False, verbose=False)
    print(f"[yolo_text_holdout] DONE -> {Path(PROJECT) / 'run' / 'weights' / 'best.pt'}", flush=True)
