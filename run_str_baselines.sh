#!/usr/bin/env bash
# STR 비교군 학습 러너 — SVTRv2(OpenOCR) → PARSeq(공식 repo), 순차·재개 가능.
# 데이터: artifacts/str_baselines/parseq_data (signboard_v3 분할 그대로, prep_str_baselines_data.py)
set -u
cd "$(dirname "$0")"
MAIN=$(pwd -W 2>/dev/null || pwd)
PY="$MAIN/.venv/Scripts/python.exe"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
OUT=artifacts/str_baselines; LOG=$OUT/logs; ST=$OUT/status; mkdir -p "$LOG" "$ST"
STATUS="$LOG/status.txt"
step() {  # name, command...  (done flag = status/<name>.done)
  local name=$1; shift
  if [ -f "$ST/$name.done" ]; then echo "[$(date '+%m-%d %H:%M')] $name: 완료본 존재, 건너뜀" | tee -a "$STATUS"; return; fi
  echo "[$(date '+%m-%d %H:%M')] $name: 시작" | tee -a "$STATUS"
  "$@" > "$LOG/$name.log" 2>&1; rc=$?
  echo "[$(date '+%m-%d %H:%M')] $name: 종료 (exit $rc)" | tee -a "$STATUS"
  [ $rc -eq 0 ] && touch "$ST/$name.done"
}
svtrv2_train() { (cd external/OpenOCR && "$PY" tools/train_rec.py -c configs/rec/svtrv2/svtrv2_rctc_signboard.yml); }
parseq_train() { (cd external/parseq && "$PY" train_parseq_signboard.py charset=korean_v5 dataset=real \
    data.root_dir="$MAIN/artifacts/str_baselines/parseq_data" data.remove_whitespace=false data.normalize_unicode=false \
    data.num_workers=0 model.batch_size=128 model.lr=3e-4 trainer.devices=1 trainer.max_epochs=20 \
    trainer.val_check_interval=1.0 pretrained=parseq hydra.run.dir="$MAIN/artifacts/str_baselines/parseq"); }
step svtrv2 svtrv2_train
step parseq parseq_train
echo "[$(date '+%m-%d %H:%M')] 학습 전체 완료" | tee -a "$STATUS"
