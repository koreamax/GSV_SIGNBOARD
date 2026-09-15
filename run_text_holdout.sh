#!/usr/bin/env bash
# 텍스트(word) 박스 탐지 4모델 hold-out 학습 → test 통합 평가. 순차, 재개 가능.
# 데이터: artifacts/signboard_text_holdout (OCR 인식기와 같은 소스 분할, 전체 27k장)
set -u
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
LOG=artifacts/signboard_text_holdout/logs; mkdir -p "$LOG"
STATUS="$LOG/status.txt"
step() {  # name, done-file, command...
  local name=$1 done_file=$2; shift 2
  if [ -f "$done_file" ]; then echo "[$(date '+%m-%d %H:%M')] $name: 완료본 존재, 건너뜀" | tee -a "$STATUS"; return; fi
  echo "[$(date '+%m-%d %H:%M')] $name: 시작" | tee -a "$STATUS"
  "$@" > "$LOG/$name.log" 2>&1
  echo "[$(date '+%m-%d %H:%M')] $name: 종료 (exit $?)" | tee -a "$STATUS"
}
step yolo26x artifacts/yolo26x_text_holdout/run/weights/best.pt "$PY" train_yolo_text_holdout.py yolo26x.pt artifacts/yolo26x_text_holdout
step yolov5x artifacts/yolov5x_text_holdout/run/weights/best.pt "$PY" train_yolo_text_holdout.py yolov5xu.pt artifacts/yolov5x_text_holdout
step effdet  artifacts/effdet_text_holdout/summary.csv            "$PY" train_effdet_text_holdout.py
step frcnn   artifacts/frcnn_text_holdout/summary.csv             "$PY" train_frcnn_text_holdout.py
echo "[$(date '+%m-%d %H:%M')] test 통합 평가 시작" | tee -a "$STATUS"
"$PY" eval_text_holdout.py > "$LOG/eval_test.log" 2>&1
echo "[$(date '+%m-%d %H:%M')] 전체 완료 (exit $?)" | tee -a "$STATUS"
