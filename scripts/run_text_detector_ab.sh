#!/bin/bash
# D38 — 텍스트 검출기 교체 A/B: CRAFT / CRAFT∪DB(배포) / YOLO26x / YOLO26x∪CRAFT
# 검출 단계만 갈아끼우고 라인 병합(y_tol 0.04)·패딩(0.04)·인식기·전처리는 전부 동일합니다.
# run 번호: 60~63 = GT 크롭, 65~68 = 탐지 크롭(연쇄). 기존 run(24/30/47/51/52)과 겹치지 않습니다.
# 루트에서: bash scripts/run_text_detector_ab.sh [sweep|final|cascade|all]
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
PY=.venv/Scripts/python.exe
L=artifacts/text_det_ab/logs; mkdir -p "$L"
ST=$L/status.txt
say() { echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$ST"; }
step() {
  local name=$1; shift
  if [ -f "$L/$name.done" ]; then say "$name: 완료본, 건너뜀"; return; fi
  say "$name: 시작"
  "$@" > "$L/$name.log" 2>&1; local rc=$?
  say "$name: 종료 (exit $rc)"
  [ $rc -eq 0 ] && touch "$L/$name.done"
}
STEP=${1:-all}

# ---------- 1) conf 훑기 (GT 크롭) ----------
if [ "$STEP" = "sweep" ] || [ "$STEP" = "all" ]; then
  for c in 0.15 0.25 0.35; do
    r=$(echo "$c" | sed 's/0\.//')            # 15 / 25 / 35
    step "sweep_yolo_c$r" $PY pipeline/run_ocr_line.py --run "6$((r/10))" \
        --text-detector yolo --yolo-det-conf "$c" --engine-tag "yolo$r"
  done
  say "훑기 채점"
  $PY pipeline/eval_ocr_v2.py --mask-phone --en-only-regions brooklyn \
      --engines yolo15,yolo25,yolo35 > "$L/eval_sweep.log" 2>&1
  grep -A 6 "^GLOBAL" "$L/eval_sweep.log" | tee -a "$ST"
fi

# ---------- 2) 확정 구성 (GT 크롭) ----------
if [ "$STEP" = "final" ] || [ "$STEP" = "all" ]; then
  BEST=${BEST_CONF:-0.25}
  step "final_craft"      $PY pipeline/run_ocr_line.py --run 63 --text-detector craft --engine-tag craftonly
  step "final_yolo_union" $PY pipeline/run_ocr_line.py --run 64 --text-detector yolo_union \
      --yolo-det-conf "$BEST" --engine-tag yolounion
  say "최종 채점 (4구성)"
  $PY pipeline/eval_ocr_v2.py --mask-phone --en-only-regions brooklyn \
      --engines paddle,craftonly,yolo15,yolo25,yolo35,yolounion > "$L/eval_final.log" 2>&1
  grep -A 8 "^GLOBAL" "$L/eval_final.log" | tee -a "$ST"
fi

# ---------- 3) 연쇄 (탐지 크롭 502개) ----------
if [ "$STEP" = "cascade" ]; then
  BEST=${BEST_CONF:-0.25}
  DET=${BEST_DET:-yolo}
  step "cascade_ocr" $PY pipeline/run_ocr_line.py --run 65 --crop-dir artifacts/gt/crop_det \
      --text-detector "$DET" --yolo-det-conf "$BEST" --engine-tag paddle
  step "cascade_eval" $PY pipeline/eval_e2e_cascade.py --ocr-run 65 --engine paddle
  grep -E "^(지역|강남|브루클린|수원|전체)" "$L/cascade_eval.log" | tee -a "$ST"
fi
say "완료 ($STEP)"
