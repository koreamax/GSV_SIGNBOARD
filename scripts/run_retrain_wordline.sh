#!/bin/bash
# 표 3 학습 데이터 통일: EasyOCR · TrOCR-base 를 PARSeq·SVTRv2·PaddleOCR 와 같은 단어+라인 75,612 로 재학습.
#
#   기존 행은 단어 크롭 62,464 만으로 학습했습니다(README 5절). 모델·하이퍼파라미터는 기존 그대로 두고
#   학습 목록만 artifacts/str_baselines/lists/{train,val}.txt 로 바꿉니다.
#     EasyOCR : config_files/signboard_v4_wl.yaml (= signboard_v3.yaml 에서 데이터 경로만 교체)
#     TrOCR   : trocr-base-printed, 10 epoch, bs 4, lr 2e-5, warmup 500, linear decay, augment (기존 v2 와 동일)
#   추론·채점은 표 3 프로토콜 그대로 (YOLO26x 스트립 conf 0.01, --mask-phone, 브루클린 영어만).
#   Tesseract 는 재학습하지 않습니다 — 단어/여러 단어 라인 exact 차이가 +2.2%p 로 라인 데이터가 부족해
#   생긴 격차가 보이지 않습니다.
#
# 새 run 번호 138(yeasywl) · 139(ytrocrbwl). 기존 산출물은 건드리지 않습니다.
# 설정 B 태깅(run_final_measure)이 GPU 를 비울 때까지 기다렸다 시작합니다.
# 루트에서 스냅샷으로 실행: cp scripts/run_retrain_wordline.sh artifacts/retrain_wl/logs/run_snapshot.sh && bash ...
cd "$(git rev-parse --show-toplevel)" || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 HF_HUB_OFFLINE=1
PY=.venv/Scripts/python.exe
L=artifacts/retrain_wl/logs; mkdir -p "$L"
ST=$L/status.txt
ROOT=$(pwd -W 2>/dev/null || pwd)
DET="--text-detector yolo --yolo-det-conf 0.01"
say() { echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$ST"; }
step() {
  local name=$1; shift
  if [ -f "$L/$name.done" ]; then say "$name: 완료본, 건너뜀"; return 0; fi
  say "$name: 시작"; "$@" > "$L/$name.log" 2>&1; local rc=$?
  say "$name: 종료 (exit $rc)"; [ $rc -eq 0 ] && touch "$L/$name.done"; return $rc
}

# ---------- 설정 B 종료 대기 ----------
FM=artifacts/final_measure/logs
if [ ! -f "$FM/casB_tag.done" ]; then
  say "대기: 설정 B 태깅 종료를 기다립니다"
  while [ ! -f "$FM/casB_tag.done" ] && ! grep -q "exit [1-9]" "$FM/status.txt" 2>/dev/null; do sleep 120; done
  say "대기 해제"
fi
ollama stop gemma4:31b >/dev/null 2>&1; ollama stop bge-m3:latest >/dev/null 2>&1; sleep 20

step prep $PY ocr/make_wordline_manifests.py || exit 1

# ---------- EasyOCR ----------
ET=external/EasyOCR/trainer
step easy_train bash -c "cd $ET && ../../../$PY run_easyocr_train.py --config config_files/signboard_v4_wl.yaml" || exit 1
step easy_pack $PY ocr/make_easyocr_plugin.py --name signboard_v4wl_custom \
    --pth $ET/saved_models/signboard_v4_wl/best_accuracy.pth \
    --character $ET/saved_models/signboard_v4_wl/character.txt || exit 1
step easy_rec $PY pipeline/run_ocr_line.py --run 138 $DET \
    --worker str_baselines/easyocr_rec_worker.py --worker-args "--recog-network signboard_v4wl_custom" \
    --engine-tag yeasywl || exit 1
step easy_eval $PY pipeline/eval_ocr_v2.py --mask-phone --en-only-regions brooklyn \
    --out-dir artifacts/retrain_wl --engines yeasy,yeasywl,cysvtrv2 || exit 1
grep -E "^\s*\[(yeasy|yeasywl)" "$L/easy_eval.log" | tail -n 8 | tee -a "$ST"

# ---------- TrOCR-base ----------
TB=artifacts/ocr_training/signboard_v4_wl
step trocr_train $PY ocr/train_textinthewild_ocr.py train-trocr --out-dir "$TB" \
    --trocr-model microsoft/trocr-base-printed \
    --epochs 10 --batch-size 4 --lr 2e-5 --warmup-steps 500 --lr-decay linear --augment || exit 1
# val 이 개선될 때만 저장되므로 마지막 [SAVE] 가 best-val 체크포인트입니다(경로는 슬래시로).
CK=$(grep -E "^\[SAVE\]" "$L/trocr_train.log" | tail -n 1 | sed 's/^\[SAVE\] //' | tr -d '\r' | sed 's#\\#/#g')
[ -f "$CK/model.safetensors" ] || { say "TrOCR best 체크포인트 없음 ($CK)"; exit 1; }
say "TrOCR 체크포인트: $CK"
step trocr_rec $PY pipeline/run_ocr_line.py --run 139 $DET \
    --worker str_baselines/trocr_rec_worker.py --worker-args "--model $CK --processor $CK" \
    --engine-tag ytrocrbwl || exit 1
step trocr_eval $PY pipeline/eval_ocr_v2.py --mask-phone --en-only-regions brooklyn \
    --out-dir artifacts/retrain_wl --engines yeasy,yeasywl,ytrocrb2,ytrocrbwl,cysvtrv2 || exit 1
grep -E "^\s*\[(yeasy|yeasywl|ytrocrb2|ytrocrbwl)" "$L/trocr_eval.log" | tail -n 16 | tee -a "$ST"
say "완료 (retrain_wl)"
