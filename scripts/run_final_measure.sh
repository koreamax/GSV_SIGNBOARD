#!/bin/bash
# 최종 재측정 (D42) — 점검에서 찾은 문제를 모두 고친 구성으로 표 2 와 연쇄(7절)를 다시 잽니다.
#
# 고친 것
#   1) 연쇄 간판 검출: 표 1 과 같은 group-aware 검출기(yolo26x_kfold_grouped)의 out-of-fold
#      예측으로 통일 (이전: 비-group-aware yolo26x_kfold / 연쇄 AP 는 fold0 하나를 전 사진에 → 80% 누수)
#   2) 연쇄 크롭 좌표계: 8192 기준으로 적고 --gt-size large 로 자름 (이전: 추측이 틀려 9개가 엉뚱한 곳)
#   3) 연쇄 OCR 채점: 표 3·4 와 같은 eval_ocr_v2 채점기 (이전: 간이 채점기라 ~10pp 차이)
#   4) 태깅: 기권(unknown) 허용 (이전: 끌 수 없는 기권 금지 → 허위 POI 가 설정의 결과였음)
#   5) VLM 단계: fix (SVTRv2 + 간판 이미지). fixcand 는 PaddleOCR 3종을 더 돌려야 해서
#      '인식 패스 1회' 라는 SVTRv2 선택 근거와 맞지 않습니다. fixcand 는 표 4 의 ablation 으로만 남깁니다.
#
#   A = 단어 검출 YOLO26x,  B = 단어 검출 YOLOv5x. 나머지(간판 검출·SVTRv2·Gemma 4 fix·태깅) 동일.
#
# 이 파일은 실행 중에 고치지 않습니다(bash 는 스크립트를 조금씩 읽어서, 실행 중 편집하면 깨집니다).
# 실제 실행은 logs/ 에 복사한 스냅샷으로 합니다.
# 루트에서: bash scripts/run_final_measure.sh
# 스냅샷(logs/ 안)에서 실행돼도 저장소 루트로 가도록 git 기준으로 찾습니다.
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 HF_HUB_OFFLINE=1
PY=.venv/Scripts/python.exe
L=artifacts/final_measure/logs; mkdir -p "$L"
ST=$L/status.txt
ROOT=$(pwd -W 2>/dev/null || pwd)
say() { echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$ST"; }
step() {
  local name=$1; shift
  if [ -f "$L/$name.done" ]; then say "$name: 완료본, 건너뜀"; return 0; fi
  say "$name: 시작"; "$@" > "$L/$name.log" 2>&1; local rc=$?
  say "$name: 종료 (exit $rc)"
  if [ $rc -ne 0 ]; then say "중단 — $name 실패. 로그: $L/$name.log"; exit 1; fi
  touch "$L/$name.done"
}

VLM=gemma4:31b
DETCROP=artifacts/gt/crop_det_grp
OUTD=artifacts/ocr_gt/ab_v4_spacecat
SV="$ROOT/artifacts/str_baselines/svtrv2/best.pth"
SVCFG="$ROOT/external/OpenOCR/configs/rec/svtrv2/svtrv2_rctc_signboard.yml"
W_SVTR=str_baselines/openocr_rec_worker.py
A_SVTR="--config $SVCFG --weights $SV"
V5W="$ROOT/artifacts/yolov5x_text_holdout/run/weights/best.pt"

# ---------- 사전 점검: 입력이 전부 있는지 ----------
for f in artifacts/gt/det_oof_grp.json artifacts/gt/e2e_match_gangnam_grp.csv \
         $OUTD/ocr_gangnam_vlmfix_svtr_gemma4_31b.csv artifacts/ocr_gt/ocr_gangnam_121_v5svtrv2.csv \
         "$SV" "$V5W"; do
  [ -e "$f" ] || { say "중단 — 입력 없음: $f"; exit 1; }
done
[ "$(ls $DETCROP/*/*.jpg | wc -l)" -eq 470 ] || { say "중단 — 탐지 크롭이 470개가 아님"; exit 1; }
say "사전 점검 통과 — 탐지 크롭 470개, 입력 파일 모두 존재"

# ---------- 1) 표 2: 단어 박스 AP (GT 간판) + 연쇄 AP (검출 간판), 4 아키텍처 ----------
step "t2_chain" $PY detection/eval_wordbox_chain_gsv.py --models yolo26x,yolov5x,frcnn,effdet
grep -E "^\[(yolo26x|yolov5x|frcnn|effdet)\]" "$L/t2_chain.log" | tee -a "$ST"

# ---------- GT 크롭 기준선(연쇄 표의 'GT크롭 기준' 열) ----------
# A 는 표 4 의 Gemma 4 · fix (YOLO26x 스트립) 출력을 그대로 씁니다.
step "gtA_emit" bash -c "for r in gangnam brooklyn suwon; do
    cp $OUTD/ocr_\${r}_vlmfix_svtr_gemma4_31b.csv artifacts/ocr_gt/ocr_\${r}_131_gtfixA.csv || exit 1; done"

# ---------- 2) 연쇄 A: 탐지 크롭 → YOLO26x 단어 → SVTRv2 → Gemma 4 fix → 태깅 ----------
step "casA_rec" $PY pipeline/run_ocr_line.py --run 133 --crop-dir $DETCROP \
    --text-detector yolo --yolo-det-conf 0.01 \
    --worker $W_SVTR --worker-args "$A_SVTR" --engine-tag g26svtr
step "casA_fix" $PY pipeline/exp_vlm_ocr.py --mode fix --model $VLM --crop-dir $DETCROP \
    --deploy-run 133 --deploy-engine g26svtr --out-suffix "_det_g26" --no-score
step "casA_emit" bash -c "for r in gangnam brooklyn suwon; do
    cp $OUTD/ocr_\${r}_vlmfix_det_g26.csv artifacts/ocr_gt/ocr_\${r}_134_g26fix.csv || exit 1; done"
step "casA_ocr" $PY pipeline/eval_e2e_cascade.py --ocr-run 134 --engine g26fix \
    --gt-run 131 --gt-engine gtfixA --tag _grp
grep -E "^(지역|gangnam|brooklyn|suwon|전체)" "$L/casA_ocr.log" | sed 's/^/[A OCR] /' | tee -a "$ST"
step "casA_tag" $PY pipeline/eval_e2e_tagging.py --ocr-run 134 --engine g26fix --model $VLM \
    --tag _grp --out artifacts/gt/e2e_tagging_final_A.csv
grep -E "^(지역|gangnam|brooklyn|suwon|전체)|허위 POI|기권" "$L/casA_tag.log" | sed 's/^/[A TAG] /' | tee -a "$ST"

# ---------- 3) 연쇄 B: 단어 검출기만 YOLOv5x ----------
step "casB_rec" $PY pipeline/run_ocr_line.py --run 135 --crop-dir $DETCROP \
    --text-detector yolo --yolo-det-conf 0.01 --yolo-det-weights "$V5W" \
    --worker $W_SVTR --worker-args "$A_SVTR" --engine-tag g5svtr
step "casB_fix" $PY pipeline/exp_vlm_ocr.py --mode fix --model $VLM --crop-dir $DETCROP \
    --deploy-run 135 --deploy-engine g5svtr --out-suffix "_det_g5" --no-score
step "casB_emit" bash -c "for r in gangnam brooklyn suwon; do
    cp $OUTD/ocr_\${r}_vlmfix_det_g5.csv artifacts/ocr_gt/ocr_\${r}_136_g5fix.csv || exit 1; done"

# ---------- 4) B 의 GT 크롭 fix (표 4 에 없는 조합 — B 의 'GT크롭 기준' 과 5단계 비교용) ----------
step "gtB_fix" $PY pipeline/exp_vlm_ocr.py --mode fix --model $VLM \
    --deploy-run 121 --deploy-engine v5svtrv2 --out-suffix "_v5x"
grep -E "^\[전역|^\[(gangnam|brooklyn|suwon)\]" "$L/gtB_fix.log" | tail -n 4 | sed 's/^/[B GT fix] /' | tee -a "$ST"
step "gtB_emit" bash -c "for r in gangnam brooklyn suwon; do
    cp $OUTD/ocr_\${r}_vlmfix_v5x.csv artifacts/ocr_gt/ocr_\${r}_132_gtfixB.csv || exit 1; done"

step "casB_ocr" $PY pipeline/eval_e2e_cascade.py --ocr-run 136 --engine g5fix \
    --gt-run 132 --gt-engine gtfixB --tag _grp
grep -E "^(지역|gangnam|brooklyn|suwon|전체)" "$L/casB_ocr.log" | sed 's/^/[B OCR] /' | tee -a "$ST"
step "casB_tag" $PY pipeline/eval_e2e_tagging.py --ocr-run 136 --engine g5fix --model $VLM \
    --tag _grp --out artifacts/gt/e2e_tagging_final_B.csv
grep -E "^(지역|gangnam|brooklyn|suwon|전체)|허위 POI|기권" "$L/casB_tag.log" | sed 's/^/[B TAG] /' | tee -a "$ST"

say "완료 (final)"
