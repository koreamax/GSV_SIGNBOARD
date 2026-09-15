#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""허위 탐지(FP) 되거르기 — union 으로 올린 recall 을 지키면서 허위 POI 를 줄인다.

4.16/D32 에서 모델 union 은 탐지 recall 을 0.825 → 0.932 로 올렸지만 FP 크롭이
124 → 335 개가 됐고, FP 의 90% 가 업종 태그까지 받아 **허위 POI** 가 됩니다.
문제의 본질은 탐지기가 아니라 **걸러낼 단계가 파이프라인에 없다는 것**입니다.

인식 단계에서 이미 얻은 신호로 사후 필터를 검증합니다 (추가 GPU 없음):
  conf   : 탐지 신뢰도
  chars  : OCR 이 뽑은 글자 수 (간판이 아니면 글자가 안 나오거나 부스러기만 남음)
  lines  : 인식된 라인 수
  rag    : 지역 상호 사전에서 유사 상호가 검색되는가 (허위 크롭의 잡음 문자열은
           실제 상호와 닮지 않음)

각 규칙에 대해 **연쇄 recall 을 얼마나 잃고 허위 POI 를 얼마나 줄이는지**를 함께
봅니다. 필터가 TP 를 같이 버리면 의미가 없기 때문입니다.
"""
from __future__ import annotations
import csv, sys
from pathlib import Path
import eval_e2e_cascade as C
import rag_retrieve as RAG

HERE = Path(__file__).resolve().parent
GT_DIR = HERE / "artifacts" / "gt"
OCR_DIR = HERE / "artifacts" / "ocr_gt"
REGIONS = ("gangnam", "brooklyn", "suwon")


def collect(tag: str, run: str):
    """크롭별 (지역, status, conf, 글자수, 라인수, RAG최고유사도, 맞은라인, GT라인)."""
    rows, fn_lines_tot, gt_lines_tot = [], 0, 0
    for region in REGIONS:
        gt_txt = C.load_map(GT_DIR / f"ocr_{region}_gt.csv")
        pred = C.load_map(OCR_DIR / f"ocr_{region}_{run}_paddle.csv")
        retr = RAG.get(region)
        for m in C.load_match(region, tag):
            if m["status"] == "FN":
                g = gt_txt.get(C.gt_crop_name(m["photo"], int(m["gt_index"])), "")
                _, n = C.line_score("", g)
                fn_lines_tot += n; gt_lines_tot += n
                continue
            txt = pred.get(m["det_crop"], "")
            lines = C.split_lines(txt)
            chars = sum(len(C.nk(l)) for l in lines)
            best = 0.0
            for l in lines[:4]:
                hits = retr.retrieve(l, k=1)
                if hits:
                    best = max(best, hits[0]["sim"])
            ok = n = 0
            if m["status"] == "TP":
                g = gt_txt.get(C.gt_crop_name(m["photo"], int(m["gt_index"])), "")
                ok, n = C.line_score(txt, g)
                gt_lines_tot += n
            rows.append({"region": region, "status": m["status"],
                         "conf": float(m["conf"] or 0), "chars": chars,
                         "lines": len(lines), "rag": best, "ok": ok, "n": n})
    return rows, gt_lines_tot


def report(rows, gt_lines, label, keep):
    tp = [r for r in rows if r["status"] == "TP"]
    fp = [r for r in rows if r["status"] == "FP"]
    ktp = [r for r in tp if keep(r)]
    kfp = [r for r in fp if keep(r)]
    fake = sum(1 for r in kfp if r["chars"] > 0)
    ok = sum(r["ok"] for r in ktp)
    print(f"{label:34s} {len(ktp):4d}/{len(tp):<4d} {len(kfp):4d}/{len(fp):<4d} "
          f"{ok/max(gt_lines,1)*100:9.1f}% {fake:8d}")


def main() -> None:
    tag, run = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("_union", "33")
    rows, gt_lines = collect(tag, run)
    print(f"[{tag} / run{run}] 채점 대상 GT 라인 {gt_lines}\n")
    print(f"{'필터':34s} {'TP유지':>9s} {'FP유지':>9s} {'연쇄recall':>10s} {'허위POI':>8s}")
    report(rows, gt_lines, "없음 (union 그대로)", lambda r: True)
    for c in (0.35, 0.50, 0.65):
        report(rows, gt_lines, f"conf >= {c}", lambda r, c=c: r["conf"] >= c)
    for k in (1, 3, 5):
        report(rows, gt_lines, f"글자수 >= {k}", lambda r, k=k: r["chars"] >= k)
    for s in (0.55, 0.7):
        report(rows, gt_lines, f"RAG 유사도 >= {s}", lambda r, s=s: r["rag"] >= s)
    report(rows, gt_lines, "글자수>=3 AND conf>=0.35",
           lambda r: r["chars"] >= 3 and r["conf"] >= 0.35)
    report(rows, gt_lines, "글자수>=3 AND (conf>=0.5 OR RAG>=0.55)",
           lambda r: r["chars"] >= 3 and (r["conf"] >= 0.5 or r["rag"] >= 0.55))
    report(rows, gt_lines, "글자수>=5 AND (conf>=0.5 OR RAG>=0.55)",
           lambda r: r["chars"] >= 5 and (r["conf"] >= 0.5 or r["rag"] >= 0.55))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
