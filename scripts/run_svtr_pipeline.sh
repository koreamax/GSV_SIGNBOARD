#!/bin/bash
# 배포 인식기를 SVTRv2-B 로 확정한 뒤의 전체 재측정 (D40).
#
#   (chain : 단어 박스 GT 라벨링 대기 — 보류)
#   rec   : EasyOCR · TrOCR 를 같은 YOLO26x 라인 스트립에서 실행 (Table 3 채우기)
#   str   : ABINet · MAERec 미세조정 + 추론 (Table 3)
#   vlm   : SVTRv2 배포 기준선 위에서 Gemma 4 31B × Qwen3-VL 32B × fixtext/fix/fixcand (Table 4)
#
# 앞 작업(SVTRv2 연쇄 재측정)이 GPU 를 비울 때까지 기다렸다 시작합니다.
# 기존 산출물은 덮어쓰지 않습니다(run 112~115, 태그 _svtr).
# 루트에서: bash scripts/run_svtr_pipeline.sh [rec|trocr|str|vlm|all]
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

# ---------- 1) 연쇄 AP@0.5 — 보류 ----------
# GSV 지역별 단어 박스 GT(930개)를 만든 뒤에 돌립니다. 그 전에는 AI Hub 에서만 잴 수 있어
# 지역 칸을 못 채우므로, 라벨이 준비될 때까지 실행하지 않습니다.
#   .venv/Scripts/python.exe detection/eval_text_chain.py --models yolo26x,yolov5x,frcnn,effdet

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

# ---------- ABINet · MAERec 미세조정 (SVTRv2 와 같은 하네스·같은 데이터) ----------
# 사전학습 가중치가 external/OpenOCR/pretrained/{abinet,maerec}/best.pth 에 있으면 그것으로 시작하고
# (strict=False 로 로드되어 분류층만 무작위), 없으면 스크래치로 학습합니다 — 표에 그 사실을 적어야 합니다.
ROOT=$(pwd -W 2>/dev/null || pwd)
if [ "$STEP" = "str" ] || [ "$STEP" = "all" ]; then
  step "train_abinet" bash -c 'cd external/OpenOCR && ../../.venv/Scripts/python.exe tools/train_rec.py -c configs/rec/abinet/abinet_signboard.yml'
  step "rec_abinet" $PY pipeline/run_ocr_line.py --run 114 $DET       --worker str_baselines/openocr_rec_worker.py       --worker-args "--config $ROOT/external/OpenOCR/configs/rec/abinet/abinet_signboard.yml --weights $ROOT/artifacts/str_baselines/abinet/best.pth"       --engine-tag yabinet
  step "train_maerec" bash -c 'cd external/OpenOCR && ../../.venv/Scripts/python.exe tools/train_rec.py -c configs/rec/maerec/maerec_signboard.yml'
  step "rec_maerec" $PY pipeline/run_ocr_line.py --run 115 $DET       --worker str_baselines/openocr_rec_worker.py       --worker-args "--config $ROOT/external/OpenOCR/configs/rec/maerec/maerec_signboard.yml --weights $ROOT/artifacts/str_baselines/maerec/best.pth"       --engine-tag ymaerec
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
