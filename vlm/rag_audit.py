#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RAG 점검 — 연쇄 태깅(구성 A)에서 검색 후보가 실제로 무엇이었고, 태깅을 도왔는지 방해했는지.

eval_e2e_tagging 과 **같은 입력**(run 134 g26fix OCR 텍스트, hybrid 검색, k=5, 평가 어휘 밖 태그 숨김)으로
크롭마다 힌트 후보를 다시 뽑고, 저장된 태깅 결과(e2e_tagging_final_A.csv)와 정답을 붙여 분류합니다.
VLM 은 부르지 않습니다(검색만 재현 — 결정적).

  정답 상호가 후보에 있었나 / 후보에 붙은 태그가 정답 태그였나 / 모델이 후보 태그를 따라갔나
  허위 POI(FP) 에 붙은 태그가 후보에서 왔나

출력: artifacts/gt/rag_audit_A.json (크롭별 상세), 콘솔 요약
"""
from __future__ import annotations

from pathlib import Path as _P
import sys as _sys; _sys.path[:0] = [str(_P(__file__).resolve().parents[1] / _d) for _d in ("vlm", "pipeline")]

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict

import eval_vlm_tagging as T
import rag_hybrid as H
import rag_retrieve as R
import eval_e2e_tagging as ET

GT_DIR = R.GT_DIR


def hint_entries(h: H.Hybrid, queries: list[str], k: int, allowed: set[str]) -> list[dict]:
    """Hybrid.hint_block 과 같은 순위·중복 제거 규칙, 문자열 대신 항목을 돌려줍니다."""
    queries = [q for q in queries if len(R.norm_key(q)) >= 2]
    if not queries:
        return []
    pool = max(k * 3, H.POOL_MIN)
    dens = h.dense_rank_batch(queries, pool) if h.emb is not None else [[] for _ in queries]
    rankings, lex_hit = [], set()
    for q, d in zip(queries, dens):
        lr = h.lex_rank(q, pool)
        lex_hit.update(lr)
        rankings.append(lr)
        if d:
            rankings.append(d)
    out, seen = [], set()
    for i in H.Hybrid.rrf(rankings, k * 3):
        e = h.lex.entries[i]
        if e["key"] in seen:
            continue
        seen.add(e["key"])
        best = max((1 - R.lev(R.norm_key(q), e["key"]) / max(len(R.norm_key(q)), len(e["key"]), 1)
                    for q in queries), default=0.0)
        out.append({"name": e["name"], "key": e["key"], "source": e["source"],
                    "tag_raw": e["tag"], "tag_shown": e["tag"] if e["tag"] in allowed else "",
                    "via_lexical": i in lex_hit, "best_sim": round(best, 3)})
        if len(out) >= k:
            break
    return out


def name_match(gt_name: str, key: str) -> bool:
    g = R.norm_key(gt_name)
    if len(g) < 2:
        return False
    return g == key or (len(g) >= 3 and (g in key or key in g)) or \
        1 - R.lev(g, key) / max(len(g), len(key)) >= 0.8


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr-run", default="134")
    ap.add_argument("--engine", default="g26fix")
    ap.add_argument("--tagging", default="e2e_tagging_final_A.csv")
    ap.add_argument("--out", default="artifacts/gt/rag_audit_A.json")
    args = ap.parse_args()

    gt = {r["image_name"]: r for r in T.load_gt()}
    raw_gt = {r["image_name"]: r for r in csv.DictReader((GT_DIR / "tagging_gt.csv").open(encoding="utf-8"))}
    tags = sorted({r["tag"] for r in gt.values() if r["eval_tag"] == "1"})
    allowed = set(tags)
    ocr = ET.load_ocr_run(args.ocr_run, args.engine)
    pred = {r["det_crop"]: r for r in csv.DictReader((GT_DIR / args.tagging).open(encoding="utf-8"))}

    rows = []
    for region in R.REGIONS:
        h = H.get(region)
        match = list(csv.DictReader((GT_DIR / f"e2e_match_{region}_grp.csv").open(encoding="utf-8")))
        work = [m for m in match if m["status"] in ("TP", "FP")]
        texts = {m["det_crop"]: ocr.get(m["det_crop"], "") for m in work}
        for m in work:
            text = texts[m["det_crop"]]
            hints = hint_entries(h, [l for l in text.split(" / ") if l.strip()], 5, allowed)
            p = pred.get(m["det_crop"], {})
            row = {"region": region, "crop": m["det_crop"], "status": m["status"], "ocr": text,
                   "pred_name": p.get("pred_name", ""), "pred_tag": p.get("pred_tag", ""), "hints": hints}
            if m["status"] == "TP":
                gname = f"{m['photo']}__crop_{int(m['gt_index']):03d}"
                g = gt.get(gname)
                if not g or g["eval_tag"] != "1":
                    continue
                row.update(gt_crop=gname, gt_name=raw_gt[gname]["name"], gt_tag=g["tag"],
                           in_db=bool(R.get(region).by_key.get(R.norm_key(raw_gt[gname]["name"]))),
                           name_in_hints=any(name_match(raw_gt[gname]["name"], x["key"]) for x in hints),
                           gt_tag_in_hints=any(x["tag_shown"] == g["tag"] for x in hints),
                           correct=p.get("pred_tag", "") == g["tag"])
            rows.append(row)

    # ------------------------------------------------------------ 요약
    def pct(a, b):
        return f"{a}/{b} ({a / b * 100:.1f}%)" if b else "0/0"

    print("=== TP (태깅 대상) ===")
    for region in list(R.REGIONS) + ["ALL"]:
        rr = [r for r in rows if r["status"] == "TP" and (region == "ALL" or r["region"] == region)]
        n = len(rr)
        print(f"[{region}] n={n} 정답 {pct(sum(r['correct'] for r in rr), n)} · "
              f"정답 상호가 사전에 있음 {pct(sum(r['in_db'] for r in rr), n)} · "
              f"정답 상호가 힌트에 {pct(sum(r['name_in_hints'] for r in rr), n)} · "
              f"힌트 없음 {pct(sum(not r['hints'] for r in rr), n)}")
    tp = [r for r in rows if r["status"] == "TP"]
    grp = defaultdict(lambda: [0, 0])
    for r in tp:
        k = ("상호 힌트 O" if r["name_in_hints"] else "상호 힌트 X")
        grp[k][0] += r["correct"]; grp[k][1] += 1
    for k, (a, b) in grp.items():
        print(f"  {k}: 태깅 정답 {pct(a, b)}")
    followed_wrong = [r for r in tp if not r["correct"] and r["pred_tag"] not in ("", "unknown")
                      and any(x["tag_shown"] == r["pred_tag"] for x in r["hints"]) and not r["name_in_hints"]]
    print(f"  오답 중 '정답 상호가 아닌' 후보의 태그를 그대로 따라간 것: {len(followed_wrong)}")

    print("\n=== FP (허위 POI) ===")
    fp = [r for r in rows if r["status"] == "FP"]
    tagged = [r for r in fp if r["pred_tag"] not in ("", "unknown")]
    from_hint = [r for r in tagged if any(x["tag_shown"] == r["pred_tag"] for x in r["hints"])]
    print(f"FP {len(fp)} · 태그 붙음 {len(tagged)} · 그 태그가 힌트 후보에 있던 태그 {len(from_hint)} · "
          f"OCR 텍스트 빈 FP {sum(not r['ocr'].strip() for r in fp)} · 힌트 없는 FP {sum(not r['hints'] for r in fp)}")
    sims = [x["best_sim"] for r in fp for x in r["hints"]]
    sims_tp = [x["best_sim"] for r in tp for x in r["hints"]]
    import statistics as st
    print(f"힌트-질의 최대 문자 유사도 중앙값: FP {st.median(sims) if sims else 0:.2f} vs TP {st.median(sims_tp):.2f}")
    print(f"dense 로만 들어온 후보 비율: FP {sum(not x['via_lexical'] for r in fp for x in r['hints'])/max(len(sims),1):.1%}"
          f" vs TP {sum(not x['via_lexical'] for r in tp for x in r['hints'])/max(len(sims_tp),1):.1%}")

    Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[saved] {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    from pathlib import Path
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
