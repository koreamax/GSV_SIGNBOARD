#!/bin/bash
# RAG 점검(09-30) 후속 실험 — 연쇄 구성 A 의 태깅만 조건을 바꿔 다시 돌립니다.
#   norag : RAG 힌트 없이 OCR 텍스트만 (RAG 기여도 기준선)
#   gated : hybrid 힌트, OCR 라인과 같은 업소로 볼 만한 후보(유사도 ≥0.8 또는 포함)에만 태그 표시
# 입력은 기존 A 와 같음(run 134 g26fix, 매칭표 _grp 298장, Gemma 4 31B, 기권 허용). 조건당 약 5시간.
cd "$(git rev-parse --show-toplevel)" || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
PY=.venv/Scripts/python.exe
L=artifacts/rag_exp/logs; ST=$L/status.txt
say() { echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$ST"; }
step() {
  local name=$1; shift
  if [ -f "$L/$name.done" ]; then say "$name: 완료본, 건너뜀"; return 0; fi
  say "$name: 시작"; "$@" > "$L/$name.log" 2>&1; local rc=$?
  say "$name: 종료 (exit $rc)"; [ $rc -eq 0 ] && touch "$L/$name.done"; return $rc
}
COMMON="--ocr-run 134 --engine g26fix --model gemma4:31b --tag _grp"
step tag_norag $PY -u pipeline/eval_e2e_tagging.py $COMMON --retriever none \
    --out artifacts/gt/e2e_tagging_A_norag.csv || exit 1
grep -E "^(지역|gangnam|brooklyn|suwon|전체)|허위 POI|기권" "$L/tag_norag.log" | sed 's/^/[norag] /' | tee -a "$ST"
step tag_gated $PY -u pipeline/eval_e2e_tagging.py $COMMON --retriever hybrid --hint-tag-min-sim 0.8 \
    --out artifacts/gt/e2e_tagging_A_gated.csv || exit 1
grep -E "^(지역|gangnam|brooklyn|suwon|전체)|허위 POI|기권" "$L/tag_gated.log" | sed 's/^/[gated] /' | tee -a "$ST"
say "완료 (rag_exp)"
