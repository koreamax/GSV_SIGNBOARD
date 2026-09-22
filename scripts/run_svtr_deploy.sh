#!/bin/bash
# D40 — 배포 인식기를 3-way vote(PP-OCRv5)에서 SVTRv2-B 단일로 교체하고 전 구간 재측정.
# 검출기는 YOLO26x @conf 0.01 로 고정(D38). 바뀌는 것은 인식기 하나뿐입니다.
# GT 크롭 인식은 run 83(cysvtrv2)이 이미 같은 구성이라 재사용합니다.
# 기존 산출물은 건드리지 않습니다(태그 분리: _svtrdet, run 110/111).
# 루트에서: bash scripts/run_svtr_deploy.sh [vlm|cascade|all]
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
PY=.venv/Scripts/python.exe
L=artifacts/svtr_deploy/logs; mkdir -p "$L"
ST=$L/status.txt
CONF=0.01
DET="--text-detector yolo --yolo-det-conf $CONF"
ROOT=$(pwd -W 2>/dev/null || pwd)          # OpenOCR 워커는 chdir 하므로 모델 경로는 절대경로여야 합니다
SV="$ROOT/artifacts/str_baselines/svtrv2/best.pth"
SVCFG="$ROOT/external/OpenOCR/configs/rec/svtrv2/svtrv2_rctc_signboard.yml"
REC="--worker str_baselines/openocr_rec_worker.py --worker-args --config $SVCFG --weights $SV"
say() { echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$ST"; }
step() {
  local name=$1; shift
  if [ -f "$L/$name.done" ]; then say "$name: 완료본, 건너뜀"; return; fi
  say "$name: 시작"; "$@" > "$L/$name.log" 2>&1; local rc=$?
  say "$name: 종료 (exit $rc)"; [ $rc -eq 0 ] && touch "$L/$name.done"
}
STEP=${1:-all}

# ---------- 1) VLM 교정 (GT 크롭) — 배포 기준선만 SVTRv2 로 바뀝니다 ----------
if [ "$STEP" = "vlm" ] || [ "$STEP" = "all" ]; then
  step "vlm_fixcand5_svtr" $PY pipeline/exp_vlm_ocr.py --mode fixcand \
      --cand-tags y5,y4,ypre,ysvtrv2,yparseq --name fixcand5 --model gemma4:31b \
      --deploy-run 83 --deploy-engine cysvtrv2 --out-suffix "_svtrdet"
  grep -E "^\[전역|^\[(gangnam|brooklyn|suwon)\]" "$L/vlm_fixcand5_svtr.log" | tail -n 4 | tee -a "$ST"
fi

# ---------- 2) 연쇄 (탐지 크롭 502개) ----------
if [ "$STEP" = "cascade" ] || [ "$STEP" = "all" ]; then
  step "cas_ocr"  $PY pipeline/run_ocr_line.py --run 110 --crop-dir artifacts/gt/crop_det $DET \
      --worker str_baselines/openocr_rec_worker.py --worker-args "--config $SVCFG --weights $SV" \
      --engine-tag svtrdet
  step "cas_fix"  $PY pipeline/exp_vlm_ocr.py --mode fix --model gemma4:31b \
      --crop-dir artifacts/gt/crop_det --deploy-run 110 --deploy-engine svtrdet \
      --out-suffix "_det_svtrdet" --no-score
  step "cas_emit" bash -c 'for r in gangnam brooklyn suwon; do cp "artifacts/ocr_gt/ab_v4_spacecat/ocr_${r}_vlmfix_det_svtrdet.csv" "artifacts/ocr_gt/ocr_${r}_111_vlmfix.csv"; done; echo emitted'
  step "cas_eval_ocr"  $PY pipeline/eval_e2e_cascade.py --ocr-run 111 --engine vlmfix
  step "cas_eval_tag"  $PY pipeline/eval_e2e_tagging.py --ocr-run 111 --engine vlmfix --model gemma4:31b \
      --out artifacts/gt/e2e_tagging_results_svtrdet.csv
  grep -E "^(지역|강남|브루클린|수원|전체)" "$L/cas_eval_ocr.log" "$L/cas_eval_tag.log" 2>/dev/null | tee -a "$ST"
fi
say "완료 ($STEP)"
