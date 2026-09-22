#!/bin/bash
# 배포 인식기를 SVTRv2-B 로 확정한 뒤의 전체 재측정 (D40).
#
#   chain : 간판 검출 → 단어 검출 연쇄 AP@0.5 (4 아키텍처) — 단독 AP 와 같은 단위
#   rec   : EasyOCR · TrOCR 를 같은 YOLO26x 라인 스트립에서 실행 (Table 3 채우기)
#   vlm   : SVTRv2 배포 기준선 위에서 Gemma 4 31B × Qwen3-VL 32B × fixtext/fix/fixcand (Table 4)
#
# 앞 작업(SVTRv2 연쇄 재측정)이 GPU 를 비울 때까지 기다렸다 시작합니다.
# 기존 산출물은 덮어쓰지 않습니다(run 112~115, 태그 _svtr).
# 루트에서: bash scripts/run_svtr_pipeline.sh [chain|rec|vlm|all]
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
PY=.venv/Scripts/python.exe
L=artifacts/svtr_pipeline/logs; mkdir -p "$L"
ST=$L/status.txt
DET="--text-detector yolo --yolo-det-conf 0.01"
say() { echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$ST"; }
step() {
  local name=$1; shift
  if [ -f "$L/$name.done" ]; then say "$name: 완료본, 건너뜀"; return; fi
  say "$name: 시작"; "$@" > "$L/$name.log" 2>&1; local rc=$?
  say "$name: 종료 (exit $rc)"; [ $rc -eq 0 ] && touch "$L/$name.done"
}
STEP=${1:-all}

W=artifacts/svtr_deploy/logs
if [ ! -f "$W/cas_eval_tag.done" ]; then
  say "대기: 앞 작업(SVTRv2 연쇄) 종료를 기다립니다"
  while [ ! -f "$W/cas_eval_tag.done" ] && ! grep -q "exit [1-9]" "$W/status.txt" 2>/dev/null; do sleep 120; done
  say "대기 해제"
fi

# ---------- 1) 연쇄 AP@0.5 — 간판 검출 실패가 그대로 넘어가는 진짜 연쇄 ----------
if [ "$STEP" = "chain" ] || [ "$STEP" = "all" ]; then
  step "chain_ap" $PY detection/eval_text_chain.py --models yolo26x,yolov5x,frcnn,effdet
  grep -E "chained AP|signboards on|NO crop" "$L/chain_ap.log" 2>/dev/null | tee -a "$ST"
fi

# ---------- 2) Table 3 — EasyOCR (이미 미세조정됨) ----------
if [ "$STEP" = "rec" ] || [ "$STEP" = "all" ]; then
  step "rec_easyocr" $PY pipeline/run_ocr_line.py --run 112 $DET \
      --worker str_baselines/easyocr_rec_worker.py --engine-tag yeasy
fi

# ---------- 3) TrOCR-base 미세조정 (small 대신 5.4배 큰 모델) → 같은 스트립에서 실행 ----------
# 크롭·분할·토크나이저는 signboard_v3 프로토콜 그대로, base 모델만 바꿉니다.
TB=artifacts/ocr_training/signboard_v3_base
if [ "$STEP" = "trocr" ] || [ "$STEP" = "all" ]; then
  step "train_trocr_base" $PY ocr/train_textinthewild_ocr.py train-trocr \
      --out-dir "$TB" \
      --trocr-model microsoft/trocr-base-printed \
      --tokenizer-dir artifacts/ocr_training/signboard_v3/trocr_model \
      --epochs 10 --batch-size 4 --lr 5e-5 --augment
  step "rec_trocr_base" $PY pipeline/run_ocr_line.py --run 113 $DET \
      --worker str_baselines/trocr_rec_worker.py \
      --worker-args "--model $TB/trocr_model --processor $TB/trocr_model" \
      --engine-tag ytrocrb
fi

# ---------- 3) Table 4 — SVTRv2 배포 기준선 × 2 VLM × 3 모드 ----------
# 후보 3종(y5/y4/ypre)은 이미 새 스트립 기준으로 만들어져 있습니다.
if [ "$STEP" = "vlm" ] || [ "$STEP" = "all" ]; then
  for M in gemma4:31b qwen3-vl:32b-instruct; do
    TAG=$(echo "$M" | tr ':.-' '___')
    for MODE in fixtext fix; do
      step "vlm_${TAG}_${MODE}" $PY pipeline/exp_vlm_ocr.py --mode $MODE --name $MODE --model "$M" \
          --deploy-run 83 --deploy-engine cysvtrv2 --out-suffix "_svtr_${TAG}"
      grep -E "^\[전역" "$L/vlm_${TAG}_${MODE}.log" | tail -n 1 | tee -a "$ST"
    done
    step "vlm_${TAG}_fixcand" $PY pipeline/exp_vlm_ocr.py --mode fixcand --cand-tags y5,y4,ypre \
        --name fixcand --model "$M" --deploy-run 83 --deploy-engine cysvtrv2 \
        --out-suffix "_svtr_${TAG}"
    grep -E "^\[전역" "$L/vlm_${TAG}_fixcand.log" | tail -n 1 | tee -a "$ST"
  done
fi
say "완료 ($STEP)"
