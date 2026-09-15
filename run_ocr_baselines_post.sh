#!/usr/bin/env bash
# OCR 비교군 후처리 오케스트레이션 (GPU 작업 직렬화).
#  1) PARSeq 학습 종료 대기 → 2) GSV 인식(tesseract/surya/paddle 재실행, GPU 검출) → 3) 학습 러너 재실행(SVTRv2)
#  4) SVTRv2 종료 대기 → 5) PARSeq/SVTRv2 GSV 인식 + in-domain(parseq/svtrv2/paddle/surya)
#  6) eval_ocr_v2(--engines) + 표 생성. 상태: artifacts/str_baselines/logs/post_status.txt
set -u
cd "$(dirname "$0")"
MAIN=$(pwd -W 2>/dev/null || pwd)
PY="$MAIN/.venv/Scripts/python.exe"; SURYA_PY="$MAIN/.venv_surya/Scripts/python.exe"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
L=artifacts/str_baselines/logs; ST=artifacts/str_baselines/status; mkdir -p "$L" "$ST"
S="$L/post_status.txt"; TR="$L/status.txt"
log() { echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$S"; }
step() { local name=$1; shift; if [ -f "$ST/$name.done" ]; then log "$name: 완료본, 건너뜀"; return; fi
  log "$name: 시작"; "$@" > "$L/$name.log" 2>&1; rc=$?; log "$name: 종료 (exit $rc)"; [ $rc -eq 0 ] && touch "$ST/$name.done"; }
waitfor() { local pat=$1; log "대기: $pat"; while ! grep -q "$pat" "$TR" 2>/dev/null; do sleep 120; done; }

# 1) PARSeq 종료 대기 (러너는 svtrv2 실패 → parseq 진행 중)
[ -f "$ST/parseq.done" ] || waitfor "parseq: 종료"
sleep 30
# 2) GSV 인식 — GPU 검출(CRAFT∪DB) + 워커. paddle 은 배포 파이프라인(3-way vote) 재실행으로 최신 run 갱신
step gsv_tesseract "$PY" run_ocr_line.py --run 40 --worker tesseract_rec_worker.py --engine-tag tesseract
step gsv_surya     "$PY" run_ocr_line.py --run 41 --worker surya_rec_worker.py --worker-py "$SURYA_PY" --engine-tag surya
step gsv_paddle    "$PY" run_ocr_line.py --run 44
# 3) 학습 러너 재실행 (svtrv2 만 남음; parseq 는 done 플래그로 건너뜀)
if [ ! -f "$ST/svtrv2.done" ]; then
  log "학습 러너 재실행 (SVTRv2)"
  powershell -NoProfile -Command "Start-Process -FilePath 'C:\Program Files\Git\bin\bash.exe' -ArgumentList '$MAIN/run_str_baselines.sh' -WindowStyle Hidden"
  sleep 300
  # 4) SVTRv2 종료 대기: status.txt 에 새 '종료' 줄이 생길 때까지
  n0=$(grep -c "svtrv2: 종료" "$TR"); while [ "$(grep -c 'svtrv2: 종료' "$TR")" -le "$n0" ]; do sleep 300; done
  log "SVTRv2 러너 종료 감지"
fi
# 5) 학습 모델 GSV + in-domain
PQ=$(ls -1 "$MAIN"/artifacts/str_baselines/parseq/checkpoints/epoch*.ckpt 2>/dev/null | sort -t- -k3,3n | tail -n 1)
SV="$MAIN/artifacts/str_baselines/svtrv2/best.pth"
SVCFG="$MAIN/external/OpenOCR/configs/rec/svtrv2/svtrv2_rctc_signboard.yml"
log "parseq ckpt: $PQ"; log "svtrv2 ckpt: $SV"
[ -n "$PQ" ] && step gsv_parseq "$PY" run_ocr_line.py --run 42 --worker parseq_rec_worker.py --worker-args "--ckpt $PQ" --engine-tag parseq
[ -f "$SV" ] && step gsv_svtrv2 "$PY" run_ocr_line.py --run 43 --worker openocr_rec_worker.py --worker-args "--config $SVCFG --weights $SV" --engine-tag svtrv2
[ -n "$PQ" ] && step ind_parseq "$PY" eval_str_indomain.py --engine parseq --worker parseq_rec_worker.py --worker-args "--ckpt $PQ"
[ -f "$SV" ] && step ind_svtrv2 "$PY" eval_str_indomain.py --engine svtrv2 --worker openocr_rec_worker.py --worker-args "--config $SVCFG --weights $SV"
step ind_paddle "$PY" eval_str_indomain.py --engine paddle --worker paddle_rec_worker.py --worker-args "--model-dir $MAIN/output/paddle_signboard_rec_v5_lines/inference --device gpu"
step ind_surya  "$PY" eval_str_indomain.py --engine surya --worker surya_rec_worker.py --worker-py "$SURYA_PY"
# 6) 평가 + 표
step eval_gsv "$PY" eval_ocr_v2.py --mask-phone --en-only-regions brooklyn --engines easyocr,trocr,paddle,tesseract,surya,parseq,svtrv2
step report "$PY" make_ocr_baselines_report.py
log "후처리 전체 완료"
