#!/bin/bash
# 표 2·3·4 의 빈칸을 같은 조건에서 채웁니다 (A안).
#   Table 3 : EasyOCR·TrOCR 를 YOLO26x 라인 스트립에서 재실행 (옛 per-box 값은 조건이 달라 못 씀)
#   Table 2 : 간판 검출 → 단어 검출 연쇄 AP@0.5 (지금까지 잰 적 없음)
#   Table 4 : PaddleOCR + VLM 의 fixtext / fix / fixcand 를 YOLO26x 스트립에서 재실행
# 앞서 돌던 SVTRv2 실험이 끝난 뒤 시작합니다. 기존 산출물은 하나도 덮어쓰지 않습니다.
# 루트에서: bash scripts/run_plan_a.sh [rec|chain|vlm|all]
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
PY=.venv/Scripts/python.exe
L=artifacts/plan_a/logs; mkdir -p "$L"
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

# 앞 작업(SVTRv2 배포 재측정)이 GPU 를 비울 때까지 대기
W=artifacts/svtr_deploy/logs
if [ ! -f "$W/cas_eval_tag.done" ]; then
  say "대기: SVTRv2 실험 종료를 기다립니다"
  while [ ! -f "$W/cas_eval_tag.done" ] && ! grep -q "exit [1-9]" "$W/status.txt" 2>/dev/null; do sleep 120; done
  say "대기 해제"
fi

# ---------- 1) Table 3 — EasyOCR·TrOCR (라인 스트립, 같은 검출기) ----------
if [ "$STEP" = "rec" ] || [ "$STEP" = "all" ]; then
  step "rec_easyocr" $PY pipeline/run_ocr_line.py --run 112 $DET \
      --worker str_baselines/easyocr_rec_worker.py --engine-tag yeasy
  step "rec_trocr"   $PY pipeline/run_ocr_line.py --run 113 $DET \
      --worker str_baselines/trocr_rec_worker.py --engine-tag ytrocr
fi

# ---------- 2) Table 2 — 간판 검출 → 단어 검출 연쇄 AP@0.5 ----------
if [ "$STEP" = "chain" ] || [ "$STEP" = "all" ]; then
  step "chain_ap" $PY detection/eval_text_chain.py --models yolo26x,yolov5x,frcnn,effdet
  grep -E "chained AP|\[sign\] [0-9]+ signboards" "$L/chain_ap.log" 2>/dev/null | tee -a "$ST"
fi

# ---------- 3) Table 4 — VLM 세 모드 (PaddleOCR 기준선, YOLO26x 스트립) ----------
if [ "$STEP" = "vlm" ] || [ "$STEP" = "all" ]; then
  for m in fixtext fix; do
    step "vlm_$m" $PY pipeline/exp_vlm_ocr.py --mode $m --name $m --model gemma4:31b \
        --deploy-run 72 --deploy-engine yolo01 --out-suffix "_yolodet"
    grep -E "^\[전역" "$L/vlm_$m.log" | tail -n 1 | tee -a "$ST"
  done
  step "vlm_fixcand" $PY pipeline/exp_vlm_ocr.py --mode fixcand --cand-tags y5,y4,ypre \
      --name fixcand --model gemma4:31b --deploy-run 72 --deploy-engine yolo01 --out-suffix "_yolodet"
  grep -E "^\[전역" "$L/vlm_fixcand.log" | tail -n 1 | tee -a "$ST"
fi
say "완료 ($STEP)"
