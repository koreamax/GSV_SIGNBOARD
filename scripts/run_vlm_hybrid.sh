#!/bin/bash
# OCR×VLM 하이브리드 — Qwen3-VL-32B / Gemma4-31B (Ollama, 4-bit, GPU+RAM 오프로드) 전체 실험 러너 (4.20)
#   A) GT 크롭(411): solo / fix / fixcand(3후보) / fixcand5(5후보)  → eval_vlm_hybrid_matrix.py 로 compose·guard·consensus 채점
#   B) 탐지 크롭(502, 연쇄): solo_det / fix_det → compose(--use-fix) → eval_e2e_cascade → eval_e2e_tagging
#   C) GT 크롭 태깅(vlmrag)
# 모든 단계는 크롭 단위 체크포인트(.partial.jsonl)로 이어하기가 됩니다. 루트에서: bash scripts/run_vlm_hybrid.sh
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
  say "$name: 종료 (exit $rc)  $(grep -E '^\[전역|recall|TP 위' "$L/$name.log" | tail -n 1 | cut -c1-120)"
  [ $rc -eq 0 ] && touch "$L/$name.done"
}
MODELS=("qwen3vl32b|qwen3-vl:32b|51" "gemma4_31b|gemma4:31b|52")

say "===== VLM 하이브리드 러너 시작 ====="
# ---------- A) GT 크롭: 요청된 비교(fix / fixcand / compose) + 5후보 ----------
for m in "${MODELS[@]}"; do IFS='|' read -r TAG NAME RUN <<< "$m"
  step "A_${TAG}_fixcand5" $PY pipeline/exp_vlm_ocr.py --mode fixcand --cand-tags v5,v4,pre,svtrv2,parseq --name fixcand5 --model "$NAME" --out-suffix "_$TAG"
  step "A_${TAG}_solo"     $PY pipeline/exp_vlm_ocr.py --mode solo    --model "$NAME" --out-suffix "_$TAG"
  step "A_${TAG}_fixcand"  $PY pipeline/exp_vlm_ocr.py --mode fixcand --model "$NAME" --out-suffix "_$TAG"
  step "A_${TAG}_fix"      $PY pipeline/exp_vlm_ocr.py --mode fix     --model "$NAME" --out-suffix "_$TAG"
  step "A_matrix_after_${TAG}" $PY pipeline/eval_vlm_hybrid_matrix.py --models gemma3,qwen3vl32b,gemma4_31b
done
# ---------- B) 연쇄(탐지 크롭): fix + solo → compose → 채점 → 태깅 ----------
for m in "${MODELS[@]}"; do IFS='|' read -r TAG NAME RUN <<< "$m"
  step "B_${TAG}_fix_det"  $PY pipeline/exp_vlm_ocr.py --mode fix  --model "$NAME" --crop-dir artifacts/gt/crop_det --deploy-run 30 --out-suffix "_det_$TAG" --no-score
  step "B_${TAG}_solo_det" $PY pipeline/exp_vlm_ocr.py --mode solo --model "$NAME" --crop-dir artifacts/gt/crop_det --deploy-run 30 --out-suffix "_det_$TAG" --no-score
  step "B_${TAG}_compose"  $PY pipeline/exp_vlm_ocr.py --mode compose --use-fix --out-suffix "_det_$TAG" --no-score --emit-run "$RUN"
  step "B_${TAG}_cascade"  $PY pipeline/eval_e2e_cascade.py --ocr-run "$RUN"
  step "B_${TAG}_tagging"  $PY pipeline/eval_e2e_tagging.py --ocr-run "$RUN" --model "$NAME" --out "artifacts/gt/e2e_tagging_results_$TAG.csv"
done
# ---------- C) GT 크롭 태깅 (이미지 + POI 후보) ----------
for m in "${MODELS[@]}"; do IFS='|' read -r TAG NAME RUN <<< "$m"
  step "C_${TAG}_tag_vlmrag" $PY vlm/eval_vlm_tagging.py --model "$NAME" --mode vlmrag --retriever hybrid --no-abstain --out "artifacts/gt/vlm_tagging_results_$TAG.csv"
done
say "===== 전체 완료 ====="
