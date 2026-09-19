#!/bin/bash
# OCR×VLM 하이브리드 — Qwen3-VL-32B-Instruct / Gemma4-31B (Ollama, 4-bit, GPU+RAM 오프로드) 전체 실험 러너 (4.20)
# 2026-09-19 방향 확정: 교정 방식은 fix(이미지+OCR 라인) 와 fixcand(이미지+다모델 후보) 두 가지만 본다.
#   A) GT 크롭(411): fix / fixcand(3후보) / fixcand5(5후보) + fixtext(텍스트 전용 교정 = 문헌 표준 기준선)
#   B) 연쇄(탐지 크롭 502): fix_det → 그대로 run 으로 배출 → eval_e2e_cascade → eval_e2e_tagging
#   C) GT 크롭 태깅(vlmrag)
# 모든 VLM 단계는 크롭 단위 체크포인트(.partial.jsonl)로 이어하기가 됩니다. 루트에서: bash scripts/run_vlm_hybrid.sh
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
PY=.venv/Scripts/python.exe
L=artifacts/vlm_hybrid/logs; mkdir -p "$L"
ST=$L/status.txt
say() { echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$ST"; }
step() { # step <name> <cmd...>
  local name=$1; shift
  if [ -f "$L/$name.done" ]; then say "$name: 완료본, 건너뜀"; return; fi
  say "$name: 시작"
  "$@" > "$L/$name.log" 2>&1; local rc=$?
  say "$name: 종료 (exit $rc)  $(grep -E '^\[전역|^\| \*\*합계|^\| \*\*전체' "$L/$name.log" | tail -n 1 | cut -c1-120)"
  [ $rc -eq 0 ] && touch "$L/$name.done"
}
emit_fix_run() { # emit_fix_run <TAG> <RUN> : fix_det 출력을 연쇄 채점용 run 파일로 복사 (compose 없음)
  local TAG=$1 RUN=$2 r
  for r in gangnam brooklyn suwon; do
    cp "artifacts/ocr_gt/ab_v4_spacecat/ocr_${r}_vlmfix_det_${TAG}.csv" "artifacts/ocr_gt/ocr_${r}_${RUN}_paddle.csv" || return 1
  done
  echo "[EMIT] run $RUN <- vlmfix_det_$TAG"
}
MODELS=("gemma4_31b|gemma4:31b|52" "qwen3vl32b|qwen3-vl:32b-instruct|51")

say "===== VLM 하이브리드 러너(v2, fix/fixcand) 시작 ====="
# ---------- A) GT 크롭 ----------
for m in "${MODELS[@]}"; do IFS='|' read -r TAG NAME RUN <<< "$m"
  step "A_${TAG}_fix"      $PY pipeline/exp_vlm_ocr.py --mode fix     --model "$NAME" --out-suffix "_$TAG"
  step "A_${TAG}_fixcand"  $PY pipeline/exp_vlm_ocr.py --mode fixcand --model "$NAME" --out-suffix "_$TAG"
  step "A_${TAG}_fixcand5" $PY pipeline/exp_vlm_ocr.py --mode fixcand --cand-tags v5,v4,pre,svtrv2,parseq --name fixcand5 --model "$NAME" --out-suffix "_$TAG"
  step "A_${TAG}_fixtext"  $PY pipeline/exp_vlm_ocr.py --mode fixtext --model "$NAME" --out-suffix "_$TAG"
done
step "A_matrix" $PY pipeline/eval_vlm_hybrid_matrix.py --models gemma4_31b,qwen3vl32b
# ---------- B) 연쇄(탐지 크롭): fix_det → run → 채점 → 태깅 ----------
for m in "${MODELS[@]}"; do IFS='|' read -r TAG NAME RUN <<< "$m"
  step "B_${TAG}_fix_det"  $PY pipeline/exp_vlm_ocr.py --mode fix --model "$NAME" --crop-dir artifacts/gt/crop_det --deploy-run 30 --out-suffix "_det_$TAG" --no-score
  step "B_${TAG}_emit"     emit_fix_run "$TAG" "$RUN"
  step "B_${TAG}_cascade"  $PY pipeline/eval_e2e_cascade.py --ocr-run "$RUN"
  step "B_${TAG}_tagging"  $PY pipeline/eval_e2e_tagging.py --ocr-run "$RUN" --model "$NAME" --out "artifacts/gt/e2e_tagging_results_$TAG.csv"
done
# ---------- C) GT 크롭 태깅 (이미지 + POI 후보) ----------
for m in "${MODELS[@]}"; do IFS='|' read -r TAG NAME RUN <<< "$m"
  step "C_${TAG}_tag_vlmrag" $PY vlm/eval_vlm_tagging.py --model "$NAME" --mode vlmrag --retriever hybrid --no-abstain --out "artifacts/gt/vlm_tagging_results_$TAG.csv"
done
say "===== 전체 완료 ====="
