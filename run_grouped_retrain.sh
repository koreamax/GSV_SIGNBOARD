#!/usr/bin/env bash
# D33: group-aware 분할로 탐지 4모델 5-fold 전부 재학습 → 통합 프로토콜 재평가.
# 순차 실행(10GB GPU에 x-모델 2개 동시 불가). 각 단계는 요약 CSV가 있으면 건너뜀(재개 가능).
# 학습 하이퍼파라미터는 기존과 동일 — 바뀐 것은 fold 배정뿐이라 누수 효과만 분리됨.
set -u
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
export FOLD_ROOT=artifacts/kfold_grouped
LOG=artifacts/kfold_grouped/logs; mkdir -p "$LOG"
STATUS="$LOG/status.txt"
step() {  # name, summary-file, project, script
  local name=$1 done_file=$2 proj=$3 script=$4
  if [ -f "$done_file" ]; then echo "[$(date '+%m-%d %H:%M')] $name: 완료본 존재, 건너뜀" | tee -a "$STATUS"; return; fi
  echo "[$(date '+%m-%d %H:%M')] $name: 시작" | tee -a "$STATUS"
  DET_PROJECT="$proj" "$PY" "$script" > "$LOG/$name.log" 2>&1
  echo "[$(date '+%m-%d %H:%M')] $name: 종료 (exit $?)" | tee -a "$STATUS"
}
step yolo26x artifacts/yolo26x_kfold_grouped/kfold_summary_map50.csv artifacts/yolo26x_kfold_grouped train_yolo26x_kfold.py
step yolov5x artifacts/yolov5x_kfold_grouped/kfold_summary_map50.csv artifacts/yolov5x_kfold_grouped train_yolov5x_kfold.py
step effdet  artifacts/effdet_kfold_grouped/kfold_summary_ap50.csv    artifacts/effdet_kfold_grouped  train_effdet_kfold.py
step frcnn   artifacts/frcnn_kfold_v2_grouped/kfold_summary_ap50.csv  artifacts/frcnn_kfold_v2_grouped train_frcnn_kfold.py

echo "[$(date '+%m-%d %H:%M')] 통합 프로토콜 재평가 시작" | tee -a "$STATUS"
DET_TAG=_grouped "$PY" eval_det_unified.py --refresh \
  --cache artifacts/gt/det_unified_preds_grouped.json \
  --out artifacts/gt/det_unified_protocol_grouped.csv > "$LOG/eval_unified.log" 2>&1
echo "[$(date '+%m-%d %H:%M')] 지역별 OOF(YOLO26x, Ultralytics val) 시작" | tee -a "$STATUS"
"$PY" eval_det_oof_ultra.py --kfold artifacts/yolo26x_kfold_grouped   --out artifacts/gt/det_oof_per_region_ultra_grouped.csv > "$LOG/eval_oof.log" 2>&1
echo "[$(date '+%m-%d %H:%M')] 전체 완료 (exit $?)" | tee -a "$STATUS"
