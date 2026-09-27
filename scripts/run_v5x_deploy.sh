#!/bin/bash
# ⚠ 일부 대체됨 (D42) — 연쇄(4) 부분은 scripts/run_final_measure.sh 로 대체됐습니다.
#   여기 연쇄는 비-group-aware 간판 검출 크롭(crop_det, 좌표 오인 9개), 간이 채점기, 기권 금지
#   태깅, fixcand(PaddleOCR 후보 3회 추가)로 돌아서 결과를 쓰지 않습니다.
#   이 스크립트에서 **지금도 쓰이는 산출물**:
#     run 120 ytrocrb2  — 표 3 의 TrOCR-base 행 (0 단계)
#     run 121 v5svtrv2  — B(YOLOv5x) 의 GT 크롭 SVTRv2, run_final_measure 의 gtB_fix 입력 (1 단계)
#
# 새 배포 구성 전 구간 측정 (D41):
#   YOLO26x 간판 검출 → YOLOv5x 단어 검출 (conf 0.01) → 라인 병합 → SVTRv2-B → Gemma 4 31B fixcand
#
# 기존 배포(단어 검출기 YOLO26x)와 다른 것은 **단어 검출기 하나**입니다. Table 2 에서 YOLOv5x 가
# GSV 두 측정 모두 0.3~0.5%p 앞서서(유의하진 않음) 이 구성으로도 전 구간을 잽니다.
#
#   0) TrOCR-base 추론·채점 (Table 3 마지막 빈칸)      — YOLO26x 스트립, 표 3 프로토콜 그대로
#   1) GT 크롭: YOLOv5x 스트립 → SVTRv2 (run 121)      — 인식기 단독 (표 3 조건)
#   2) GT 크롭: 후보 3종 (PP-OCRv5 ft / v4 ft / pre)    — fixcand 재료, 같은 스트립에서 다시 뽑음
#   3) GT 크롭: Gemma 4 fixcand                          — 표 4 조건
#   4) 탐지 크롭(502개): 1)~3) 반복 → 연쇄 OCR·태깅 채점 — 7절 조건
#
# 새 run 번호 120~130, 새 엔진 태그(v5*). 기존 산출물은 건드리지 않습니다.
# GPU 를 쓰는 TrOCR 학습이 끝날 때까지 기다렸다 시작합니다.
# 루트에서: bash scripts/run_v5x_deploy.sh [trocr|gt|cascade|all]
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 HF_HUB_OFFLINE=1
PY=.venv/Scripts/python.exe
L=artifacts/v5x_deploy/logs; mkdir -p "$L"
ST=$L/status.txt
ROOT=$(pwd -W 2>/dev/null || pwd)
say() { echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$ST"; }
step() {
  local name=$1; shift
  if [ -f "$L/$name.done" ]; then say "$name: 완료본, 건너뜀"; return 0; fi
  say "$name: 시작"; "$@" > "$L/$name.log" 2>&1; local rc=$?
  say "$name: 종료 (exit $rc)"; [ $rc -eq 0 ] && touch "$L/$name.done"; return $rc
}
STEP=${1:-all}

# ---------- 구성 ----------
V5W="$ROOT/artifacts/yolov5x_text_holdout/run/weights/best.pt"
DET26="--text-detector yolo --yolo-det-conf 0.01"                         # 표 3 프로토콜 (TrOCR 용)
DET5="--text-detector yolo --yolo-det-conf 0.01 --yolo-det-weights $V5W"  # 새 배포 구성
SV="$ROOT/artifacts/str_baselines/svtrv2/best.pth"
SVCFG="$ROOT/external/OpenOCR/configs/rec/svtrv2/svtrv2_rctc_signboard.yml"
# --worker-args 는 한 문자열로 넘겨야 하므로 워커와 인자를 따로 두고 호출부에서 따옴표로 묶습니다.
W_SVTR=str_baselines/openocr_rec_worker.py;  A_SVTR="--config $SVCFG --weights $SV"
W_PAD=str_baselines/paddle_rec_worker.py
A_V5="--model-dir $ROOT/output/paddle_signboard_rec_v5_lines/inference"
A_V4="--model-dir $ROOT/output/paddle_signboard_rec_v4_spacecat/inference"
A_PRE="--model-name korean_PP-OCRv5_mobile_rec"
CAND_DIR=artifacts/ocr_gt/ab_v4_spacecat
VLM=gemma4:31b
DETCROP=artifacts/gt/crop_det

# ---------- TrOCR 학습이 GPU 를 비울 때까지 대기 ----------
TL=artifacts/svtr_pipeline/logs/train_trocr_base_v2.log
if [ -f "$TL" ] && ! grep -qE "^\[TEST\]|Traceback" "$TL"; then
  say "대기: TrOCR-base 학습 종료를 기다립니다 (예상 09-25 04:00경)"
  while ! grep -qE "^\[TEST\]|Traceback" "$TL"; do sleep 300; done
  say "대기 해제 — GPU 확보"
fi

# ---------- 0) TrOCR-base 추론 + 채점 (표 3) ----------
if [ "$STEP" = "trocr" ] || [ "$STEP" = "all" ]; then
  # val 이 개선될 때만 저장되므로 마지막 [SAVE] 가 best-val 체크포인트입니다.
  # 로그의 경로는 백슬래시인데 run_ocr_line 이 --worker-args 를 shlex 로 나누면서 백슬래시를
  # 이스케이프로 먹어 버립니다(C:UsersDANHA... 가 되어 실패). 슬래시로 바꿔 넘깁니다.
  CK=$(grep -E "^\[SAVE\]" "$TL" | tail -n 1 | sed 's/^\[SAVE\] //' | tr -d '\r' | sed 's#\\#/#g')
  if [ -z "$CK" ] || [ ! -f "$CK/model.safetensors" ]; then
    say "TrOCR best 체크포인트를 못 찾았습니다 ($CK) — 0) 건너뜀"
  else
    say "TrOCR 체크포인트: $CK"
    step "trocr_rec" $PY pipeline/run_ocr_line.py --run 120 $DET26 \
        --worker str_baselines/trocr_rec_worker.py --worker-args "--model $CK --processor $CK" \
        --engine-tag ytrocrb2
    step "trocr_eval" $PY pipeline/eval_ocr_v2.py --mask-phone --en-only-regions brooklyn \
        --engines ytrocrb2,cysvtrv2
    grep -E "^\s*\[ytrocrb2" "$L/trocr_eval.log" | tail -n 4 | tee -a "$ST"
  fi
fi

# ---------- 1)~3) GT 크롭 ----------
if [ "$STEP" = "gt" ] || [ "$STEP" = "all" ]; then
  step "gt_svtr"   $PY pipeline/run_ocr_line.py --run 121 $DET5 --worker $W_SVTR --worker-args "$A_SVTR" --engine-tag v5svtrv2
  step "gt_cand5"  $PY pipeline/run_ocr_line.py --run 122 $DET5 --worker $W_PAD  --worker-args "$A_V5"   --engine-tag v5cy5
  step "gt_cand4"  $PY pipeline/run_ocr_line.py --run 123 $DET5 --worker $W_PAD  --worker-args "$A_V4"   --engine-tag v5cy4
  step "gt_candp"  $PY pipeline/run_ocr_line.py --run 124 $DET5 --worker $W_PAD  --worker-args "$A_PRE"  --engine-tag v5cypre
  step "gt_cand_link" bash -c "for r in gangnam brooklyn suwon; do
      cp artifacts/ocr_gt/ocr_\${r}_122_v5cy5.csv   $CAND_DIR/ocr_\${r}_cand_v5y5.csv &&
      cp artifacts/ocr_gt/ocr_\${r}_123_v5cy4.csv   $CAND_DIR/ocr_\${r}_cand_v5y4.csv &&
      cp artifacts/ocr_gt/ocr_\${r}_124_v5cypre.csv $CAND_DIR/ocr_\${r}_cand_v5ypre.csv || exit 1; done; echo linked"
  step "gt_eval_rec" $PY pipeline/eval_ocr_v2.py --mask-phone --en-only-regions brooklyn \
      --engines v5svtrv2,v5cy5,v5cy4,v5cypre,cysvtrv2
  grep -E "^\s*\[(v5svtrv2|v5cy5|v5cy4|v5cypre|cysvtrv2)" "$L/gt_eval_rec.log" | tail -n 5 | tee -a "$ST"
  step "gt_fixcand" $PY pipeline/exp_vlm_ocr.py --mode fixcand --name fixcand --model $VLM \
      --cand-tags v5y5,v5y4,v5ypre --deploy-run 121 --deploy-engine v5svtrv2 --out-suffix "_v5x"
  grep -E "^\[전역|^\[(gangnam|brooklyn|suwon)\]" "$L/gt_fixcand.log" | tail -n 4 | tee -a "$ST"
  # GT 크롭 교정 결과를 run 130 으로 내보내 연쇄 채점의 기준(gt-run)으로 씁니다.
  step "gt_emit" bash -c "for r in gangnam brooklyn suwon; do
      cp $CAND_DIR/ocr_\${r}_vlmfixcand_v5x.csv artifacts/ocr_gt/ocr_\${r}_130_v5vlmfixcand.csv || exit 1; done; echo emitted"
fi

# ---------- 4) 연쇄: 탐지 크롭 502개 ----------
if [ "$STEP" = "cascade" ] || [ "$STEP" = "all" ]; then
  step "cas_svtr"  $PY pipeline/run_ocr_line.py --run 125 --crop-dir $DETCROP $DET5 --worker $W_SVTR --worker-args "$A_SVTR" --engine-tag v5svtrdet
  step "cas_cand5" $PY pipeline/run_ocr_line.py --run 126 --crop-dir $DETCROP $DET5 --worker $W_PAD  --worker-args "$A_V5"   --engine-tag v5dcy5
  step "cas_cand4" $PY pipeline/run_ocr_line.py --run 127 --crop-dir $DETCROP $DET5 --worker $W_PAD  --worker-args "$A_V4"   --engine-tag v5dcy4
  step "cas_candp" $PY pipeline/run_ocr_line.py --run 128 --crop-dir $DETCROP $DET5 --worker $W_PAD  --worker-args "$A_PRE"  --engine-tag v5dcypre
  step "cas_cand_link" bash -c "for r in gangnam brooklyn suwon; do
      cp artifacts/ocr_gt/ocr_\${r}_126_v5dcy5.csv   $CAND_DIR/ocr_\${r}_cand_v5dy5.csv &&
      cp artifacts/ocr_gt/ocr_\${r}_127_v5dcy4.csv   $CAND_DIR/ocr_\${r}_cand_v5dy4.csv &&
      cp artifacts/ocr_gt/ocr_\${r}_128_v5dcypre.csv $CAND_DIR/ocr_\${r}_cand_v5dypre.csv || exit 1; done; echo linked"
  step "cas_fixcand" $PY pipeline/exp_vlm_ocr.py --mode fixcand --name fixcand --model $VLM \
      --crop-dir $DETCROP --cand-tags v5dy5,v5dy4,v5dypre \
      --deploy-run 125 --deploy-engine v5svtrdet --out-suffix "_det_v5x" --no-score
  step "cas_emit" bash -c "for r in gangnam brooklyn suwon; do
      cp $CAND_DIR/ocr_\${r}_vlmfixcand_det_v5x.csv artifacts/ocr_gt/ocr_\${r}_129_v5vlmfixcand.csv || exit 1; done; echo emitted"
  step "cas_eval_ocr" $PY pipeline/eval_e2e_cascade.py --ocr-run 129 --gt-run 130 --engine v5vlmfixcand
  step "cas_eval_tag" $PY pipeline/eval_e2e_tagging.py --ocr-run 129 --engine v5vlmfixcand --model $VLM \
      --out artifacts/gt/e2e_tagging_results_v5x.csv
  grep -E "^(지역|강남|브루클린|수원|전체)" "$L/cas_eval_ocr.log" "$L/cas_eval_tag.log" 2>/dev/null | tee -a "$ST"
fi
say "완료 ($STEP)"
