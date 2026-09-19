#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VLM×OCR 하이브리드 — 모델 × 교정 방식 채점표 (4.20).

exp_vlm_ocr.py 가 모델별(--out-suffix _<tag>)로 저장한 출력을 공식 프로토콜
(eval_ocr_v2, --mask-phone, brooklyn en-only)로 한꺼번에 채점합니다.

  fixtext   : 텍스트 전용 post-OCR 교정 (LLM 이 OCR 라인만 보고 교정, 이미지 없음) — 문헌의 고전 기준선
  fix       : 멀티모달 post-OCR 교정 (이미지 + OCR 라인 → 오독 글자만 교정)
  fixcand   : 후보 그라운딩 교정 — 이미지 + 3모델 후보(v5·v4·zero-shot Paddle)
  fixcand5  : 후보 그라운딩 교정 — 이미지 + 5모델 후보(+SVTRv2·PARSeq; 오류가 서로 다른 인식기)

Usage:
  .venv/Scripts/python.exe pipeline/eval_vlm_hybrid_matrix.py --models gemma4_31b,qwen3vl32b [--csv out.csv]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE / d) for d in ("pipeline", "ocr", "vlm", "str_baselines")]

GT = HERE / "artifacts" / "gt"
OCR_DIR = HERE / "artifacts" / "ocr_gt"
AB = OCR_DIR / "ab_v4_spacecat"
REGIONS = ["gangnam", "brooklyn", "suwon"]
MODES = ("fixtext", "fix", "fixcand", "fixcand5")
FIELDS = ("edit", "fp_edit", "chars", "word_lcs", "gt_words", "exact",
          "recalled", "lines", "contain", "empty_pred", "crops")


def load_E():
    saved = sys.argv
    sys.argv = ["pipeline/eval_ocr_v2.py", "--mask-phone", "--en-only-regions", "brooklyn"]
    import eval_ocr_v2 as E
    sys.argv = saved
    return E


def score(E, preds_by_region: dict[str, dict]) -> dict:
    tot, per = None, {}
    for region in REGIONS:
        gt = E.load_csv_map(GT / f"ocr_{region}_gt.csv")
        E.EN_ONLY = region in E.EN_ONLY_REGIONS
        a = E.eval_engine(gt, preds_by_region[region], [])
        E.EN_ONLY = False
        per[region] = (a.exact_rate() * 100, a.cer())
        if tot is None:
            tot = a
        else:
            for f_ in FIELDS:
                setattr(tot, f_, getattr(tot, f_) + getattr(a, f_))
    return {"exact": tot.exact_rate() * 100, "CER": tot.cer(), "WAR": tot.war(),
            "contain": tot.contain_rate() * 100,
            **{f"ex_{r}": per[r][0] for r in REGIONS}, **{f"cer_{r}": per[r][1] for r in REGIONS}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="출력 접미사 태그들, 쉼표 구분 (예: gemma4_31b,qwen3vl32b)")
    ap.add_argument("--deploy-run", default="24")
    ap.add_argument("--csv", default=str(HERE / "artifacts" / "gt" / "vlm_hybrid_matrix.csv"))
    args = ap.parse_args()
    E = load_E()
    tags = [t for t in args.models.split(",") if t]

    deploy = {r: E.load_csv_map(OCR_DIR / f"ocr_{r}_{args.deploy_run}_paddle.csv") for r in REGIONS}
    rows = [("deploy OCR (run%s)" % args.deploy_run, "-", score(E, deploy))]
    for tag in tags:
        for name in MODES:
            paths = {r: AB / f"ocr_{r}_vlm{name}_{tag}.csv" for r in REGIONS}
            if not all(p.exists() for p in paths.values()):
                continue
            rows.append((name, tag, score(E, {r: E.load_csv_map(p) for r, p in paths.items()})))

    print(f"\n{'mode':<12}{'model':<14}{'exact':>7}{'CER':>8}{'WAR':>8}{'contain':>9}   exact gangnam/brooklyn/suwon   CER gangnam/brooklyn/suwon")
    for name, tag, m in rows:
        print(f"{name:<12}{tag:<14}{m['exact']:>6.1f}%{m['CER']:>8.3f}{m['WAR']:>8.3f}{m['contain']:>8.1f}%   "
              + " / ".join(f"{m['ex_'+r]:.1f}" for r in REGIONS) + "   "
              + " / ".join(f"{m['cer_'+r]:.3f}" for r in REGIONS))
    with open(args.csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "model", "exact", "CER", "WAR", "contain"] + [f"exact_{r}" for r in REGIONS] + [f"CER_{r}" for r in REGIONS])
        for name, tag, m in rows:
            w.writerow([name, tag, f"{m['exact']:.2f}", f"{m['CER']:.4f}", f"{m['WAR']:.4f}", f"{m['contain']:.2f}"]
                       + [f"{m['ex_'+r]:.2f}" for r in REGIONS] + [f"{m['cer_'+r]:.4f}" for r in REGIONS])
    print(f"\nsaved {args.csv}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
