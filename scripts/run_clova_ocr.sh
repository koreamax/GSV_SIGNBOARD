#!/bin/bash
# NAVER CLOVA OCR (General) 비교군 추가 — 유료 API 이므로 단계를 나눠 돌립니다.
#   0) 인증: 환경변수 CLOVA_OCR_URL / CLOVA_OCR_SECRET 또는
#            artifacts/str_baselines/clova_ocr/credentials.json  {"url":"...","secret":"..."}
#   1) 스모크  : 크롭 5장만 호출해 응답·라인 복원 확인          (API 5건)
#   2) GSV     : 배포 파이프라인과 동일한 라인 스트립           (API 1,342건)
#   3) in-domain 실라인 test_line                                (API 1,631건)
#   4) 표 재생성 (eval_ocr_v2 + make_ocr_baselines_report)
# 단어 크롭(test_word, 7,756건)은 비싸서 기본 제외 — 필요하면 STEP=word 로 따로 실행합니다.
# 모든 단계는 .partial.jsonl 체크포인트로 이어하기가 되며, 이미 끝난 건은 다시 호출하지 않습니다.
# 루트에서:  bash scripts/run_clova_ocr.sh [smoke|gsv|line|word|report|all]
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
PY=.venv/Scripts/python.exe
W=str_baselines/clova_rec_worker.py
L=artifacts/str_baselines/logs; mkdir -p "$L" artifacts/str_baselines/clova_ocr
STEP=${1:-all}
say() { echo "[$(date '+%m-%d %H:%M')] $*"; }

if [ "$STEP" = "smoke" ] || [ "$STEP" = "all" ]; then
  say "1) 스모크 — in-domain 실라인 5장"
  $PY - <<'PYEOF'
import json, pathlib
src = pathlib.Path("artifacts/str_baselines/lists/test_line.txt")
out = pathlib.Path("artifacts/str_baselines/clova_ocr/smoke_manifest.jsonl")
with src.open(encoding="utf-8") as f, out.open("w", encoding="utf-8") as g:
    for i, ln in enumerate(f):
        if i >= 5: break
        p, t = ln.rstrip("\n").split("\t", 1)
        g.write(json.dumps({"key": f"smoke::{i}", "path": p, "gt": t}, ensure_ascii=False) + "\n")
print("smoke manifest:", out)
PYEOF
  $PY $W --manifest artifacts/str_baselines/clova_ocr/smoke_manifest.jsonl \
         --out artifacts/str_baselines/clova_ocr/smoke_out.jsonl --max-calls 5 --workers 1 \
    || { say "스모크 실패 — 인증 정보/도메인 설정을 확인하세요"; exit 1; }
  say "스모크 결과 (GT → 예측):"
  $PY - <<'PYEOF'
import json
gt = {json.loads(l)["key"]: json.loads(l)["gt"] for l in open("artifacts/str_baselines/clova_ocr/smoke_manifest.jsonl", encoding="utf-8")}
for l in open("artifacts/str_baselines/clova_ocr/smoke_out.jsonl", encoding="utf-8"):
    d = json.loads(l)
    print(f"  {gt.get(d['key'],'')!r:30} -> {d['text']!r}  (conf {d['score']:.2f})")
PYEOF
  [ "$STEP" = "smoke" ] && exit 0
fi

if [ "$STEP" = "gsv" ] || [ "$STEP" = "all" ]; then
  say "2) GSV 라인 스트립 (API 약 1,342건)"
  $PY pipeline/run_ocr_line.py --run 47 --worker str_baselines/clova_rec_worker.py \
      --worker-args "--max-calls 1600 --workers 3" --engine-tag clova > "$L/gsv_clova.log" 2>&1
  say "GSV 종료 (exit $?)"; grep -E "EMIT|clova\] done" "$L/gsv_clova.log" | tail -n 4
fi

if [ "$STEP" = "line" ] || [ "$STEP" = "all" ]; then
  say "3) in-domain test_line (API 약 1,631건)"
  $PY str_baselines/eval_str_indomain.py --engine clova --worker str_baselines/clova_rec_worker.py \
      --worker-args "--max-calls 1800 --workers 3" --sets test_line > "$L/ind_clova.log" 2>&1
  say "in-domain 종료 (exit $?)"; grep "n=" "$L/ind_clova.log" | tail -n 2
fi

if [ "$STEP" = "word" ]; then
  say "4) in-domain test_word (API 약 7,756건 — 추가 과금)"
  $PY str_baselines/eval_str_indomain.py --engine clova --worker str_baselines/clova_rec_worker.py \
      --worker-args "--max-calls 8000 --workers 3" --sets test_word > "$L/ind_clova_word.log" 2>&1
  grep "n=" "$L/ind_clova_word.log" | tail -n 1
fi

if [ "$STEP" = "report" ] || [ "$STEP" = "all" ]; then
  say "5) 표 재생성"
  $PY pipeline/eval_ocr_v2.py --mask-phone --en-only-regions brooklyn \
      --engines easyocr,trocr,paddle,paddle1,tesseract,tesseract_ft,surya,clova,parseq,svtrv2 > "$L/eval_gsv.log" 2>&1
  grep -A 12 "^GLOBAL" "$L/eval_gsv.log"
  $PY str_baselines/make_ocr_baselines_report.py
fi
say "완료"
