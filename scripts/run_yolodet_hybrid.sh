#!/bin/bash
# D38 후속 — YOLO26x 단어 탐지기(conf 0.01) 위에서 후보 재생성 → VLM 교정 → 연쇄 재측정.
# 검출기만 바뀐 구성이라 후보 파일도 새 스트립 기준으로 다시 만들어야 인덱스가 맞습니다.
# 기존 CRAFT∪DB 산출물은 하나도 건드리지 않습니다(태그 분리: cand_y*, _yolodet).
# 루트에서: bash scripts/run_yolodet_hybrid.sh [cand|vlm|cascade|all]
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
PY=.venv/Scripts/python.exe
L=artifacts/text_det_ab/logs; mkdir -p "$L"
AB=artifacts/ocr_gt/ab_v4_spacecat
ST=$L/status.txt
CONF=0.01
DET="--text-detector yolo --yolo-det-conf $CONF"
V5=output/paddle_signboard_rec_v5_lines/inference
V4=output/paddle_signboard_rec_v4_spacecat/inference
SV=artifacts/str_baselines/svtrv2/best.pth
SVCFG=external/OpenOCR/configs/rec/svtrv2/svtrv2_rctc_signboard.yml
say() { echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$ST"; }
step() {
  local name=$1; shift
  if [ -f "$L/$name.done" ]; then say "$name: 완료본, 건너뜀"; return; fi
  say "$name: 시작"; "$@" > "$L/$name.log" 2>&1; local rc=$?
  say "$name: 종료 (exit $rc)"; [ $rc -eq 0 ] && touch "$L/$name.done"
}
cand_from_run() {   # cand_from_run <run> <engine-tag> <cand-tag>
  local r=$1 e=$2 t=$3 reg
  for reg in gangnam brooklyn suwon; do
    cp "artifacts/ocr_gt/ocr_${reg}_${r}_${e}.csv" "$AB/ocr_${reg}_cand_${t}.csv" || return 1
  done
  echo "[CAND] cand_$t <- run $r/$e"
}
STEP=${1:-all}

# ---------- 1) 후보 5종 (새 스트립 기준) ----------
if [ "$STEP" = "cand" ] || [ "$STEP" = "all" ]; then
  step "cand_v5"  $PY pipeline/run_ocr_line.py --run 80 $DET --no-vote --ko-model-dir "$V5" --engine-tag cy5
  step "cand_v4"  $PY pipeline/run_ocr_line.py --run 81 $DET --no-vote --ko-model-dir "$V4" --engine-tag cy4
  step "cand_pre" $PY pipeline/run_ocr_line.py --run 82 $DET --no-vote --engine-tag cypre
  step "cand_svtrv2" $PY pipeline/run_ocr_line.py --run 83 $DET \
      --worker str_baselines/openocr_rec_worker.py --worker-args "--config $SVCFG --weights $SV" --engine-tag cysvtrv2
  step "cand_parseq" $PY pipeline/run_ocr_line.py --run 84 $DET \
      --worker str_baselines/parseq_rec_worker.py --engine-tag cyparseq
  for spec in "80 cy5 y5" "81 cy4 y4" "82 cypre ypre" "83 cysvtrv2 ysvtrv2" "84 cyparseq yparseq"; do
    set -- $spec; cand_from_run "$1" "$2" "$3" | tee -a "$ST"
  done
fi

# ---------- 2) VLM 교정 (Gemma 4 fixcand5) ----------
if [ "$STEP" = "vlm" ] || [ "$STEP" = "all" ]; then
  step "vlm_fixcand5_yolodet" $PY pipeline/exp_vlm_ocr.py --mode fixcand \
      --cand-tags y5,y4,ypre,ysvtrv2,yparseq --name fixcand5 --model gemma4:31b \
      --deploy-run 72 --deploy-engine yolo01 --out-suffix "_yolodet"
  grep -E "^\[전역|^\[(gangnam|brooklyn|suwon)\]" "$L/vlm_fixcand5_yolodet.log" | tail -n 4 | tee -a "$ST"
fi

# ---------- 3) 연쇄 (탐지 크롭 502개) ----------
if [ "$STEP" = "cascade" ] || [ "$STEP" = "all" ]; then
  step "cas_ocr"  $PY pipeline/run_ocr_line.py --run 85 --crop-dir artifacts/gt/crop_det $DET --engine-tag yolodet
  step "cas_fix"  $PY pipeline/exp_vlm_ocr.py --mode fix --model gemma4:31b \
      --crop-dir artifacts/gt/crop_det --deploy-run 85 --deploy-engine yolodet \
      --out-suffix "_det_yolodet" --no-score
  step "cas_emit" bash -c 'for r in gangnam brooklyn suwon; do cp "artifacts/ocr_gt/ab_v4_spacecat/ocr_${r}_vlmfix_det_yolodet.csv" "artifacts/ocr_gt/ocr_${r}_86_vlmfix.csv"; done; echo emitted'
  step "cas_eval_ocr"  $PY pipeline/eval_e2e_cascade.py --ocr-run 86 --engine vlmfix
  step "cas_eval_tag"  $PY pipeline/eval_e2e_tagging.py --ocr-run 86 --engine vlmfix --model gemma4:31b \
      --out artifacts/gt/e2e_tagging_results_yolodet.csv
  grep -E "^(지역|강남|브루클린|수원|전체)" "$L/cas_eval_ocr.log" "$L/cas_eval_tag.log" 2>/dev/null | tee -a "$ST"
fi
say "완료 ($STEP)"
