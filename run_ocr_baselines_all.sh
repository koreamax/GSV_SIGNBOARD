#!/usr/bin/env bash
# OCR 비교군 전체 파이프라인 — 단일 프로세스, 완전 직렬 (GPU 작업 동시 실행 금지).
#  1) 학습 러너(run_str_baselines.sh: SVTRv2 → PARSeq)  2) GSV 인식(surya/parseq/svtrv2 [+tesseract/paddle 미완 시])
#  3) in-domain(parseq/svtrv2/paddle [+surya 미완 시])  4) eval_ocr_v2(--engines) + 표.
# 상태: artifacts/str_baselines/logs/post_status.txt, 단계 완료 플래그: artifacts/str_baselines/status/*.done
set -u
cd "$(dirname "$0")"
MAIN=$(pwd -W 2>/dev/null || pwd)
PY="$MAIN/.venv/Scripts/python.exe"; SURYA_PY="$MAIN/.venv_surya/Scripts/python.exe"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
L=artifacts/str_baselines/logs; ST=artifacts/str_baselines/status; mkdir -p "$L" "$ST"
S="$L/post_status.txt"
log() { echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$S"; }
step() { local name=$1; shift; if [ -f "$ST/$name.done" ]; then log "$name: 완료본, 건너뜀"; return; fi
  log "$name: 시작"; "$@" > "$L/$name.log" 2>&1; rc=$?; log "$name: 종료 (exit $rc)"; [ $rc -eq 0 ] && touch "$ST/$name.done"; }

log "===== 전체 파이프라인 시작 ====="
# 1) 학습 (포그라운드, 내부에서 svtrv2 → parseq 순차, 완료본은 건너뜀)
bash "$MAIN/run_str_baselines.sh"
log "학습 러너 종료 (svtrv2.done=$([ -f $ST/svtrv2.done ] && echo y || echo n), parseq.done=$([ -f $ST/parseq.done ] && echo y || echo n))"

# 2) GSV 인식 (검출은 GPU: CRAFT∪DB, 동일 설정)
PQ=$(ls -1 "$MAIN"/artifacts/str_baselines/parseq/checkpoints/epoch*.ckpt 2>/dev/null | sort -t- -k3,3n | tail -n 1)
SV="$MAIN/artifacts/str_baselines/svtrv2/best.pth"
SVCFG="$MAIN/external/OpenOCR/configs/rec/svtrv2/svtrv2_rctc_signboard.yml"
log "parseq ckpt: ${PQ:-없음} | svtrv2 ckpt: $([ -f "$SV" ] && echo "$SV" || echo 없음)"
step gsv_tesseract "$PY" run_ocr_line.py --run 40 --worker tesseract_rec_worker.py --engine-tag tesseract
step gsv_surya     "$PY" run_ocr_line.py --run 41 --worker surya_rec_worker.py --worker-py "$SURYA_PY" --engine-tag surya
step gsv_paddle    "$PY" run_ocr_line.py --run 44
[ -n "$PQ" ]   && step gsv_parseq "$PY" run_ocr_line.py --run 42 --worker parseq_rec_worker.py --worker-args "--ckpt $PQ" --engine-tag parseq
[ -f "$SV" ]   && step gsv_svtrv2 "$PY" run_ocr_line.py --run 43 --worker openocr_rec_worker.py --worker-args "--config $SVCFG --weights $SV" --engine-tag svtrv2

# 3) in-domain (signboard_v3 test_line / test_word)
[ -n "$PQ" ] && step ind_parseq "$PY" eval_str_indomain.py --engine parseq --worker parseq_rec_worker.py --worker-args "--ckpt $PQ"
[ -f "$SV" ] && step ind_svtrv2 "$PY" eval_str_indomain.py --engine svtrv2 --worker openocr_rec_worker.py --worker-args "--config $SVCFG --weights $SV"
step ind_paddle "$PY" eval_str_indomain.py --engine paddle --worker paddle_rec_worker.py --worker-args "--model-dir $MAIN/output/paddle_signboard_rec_v5_lines/inference --device gpu"
step ind_surya  "$PY" eval_str_indomain.py --engine surya --worker surya_rec_worker.py --worker-py "$SURYA_PY"

# 4) 평가 + 표
rm -f "$ST/eval_gsv.done" "$ST/report.done"
step eval_gsv "$PY" eval_ocr_v2.py --mask-phone --en-only-regions brooklyn --engines easyocr,trocr,paddle,tesseract,surya,parseq,svtrv2
step report "$PY" make_ocr_baselines_report.py
log "===== 전체 파이프라인 완료 ====="
