#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T7: GT-free crop-level ensemble re-selection (agreement + dictionary bonus).

The deployed ensemble picks per-box by raw cross-engine confidence — but the
three confidences live on different scales, and per-box scores were not
persisted for past runs. This selector re-chooses AT CROP LEVEL among the
already-saved engine outputs (easyocr / trocr / paddle / conf-ensemble),
using only deployable signals:

  score(c) = mean pairwise agreement with the 3 engine outputs
             + LAMBDA_DB * dictionary hit ratio (region OCR DB, exact key)

Agreement = 1 - normalized edit distance between whitespace/punct-collapsed
texts. No GT anywhere; weights are fixed a priori (no GSV tuning = no leakage).

Outputs a comparison table (official eval protocol via eval_ocr_v2 import):
  paddle-only / conf-ensemble raw / +snap (current official) /
  T7 variants / crop-level oracle / line-level oracle.

Usage:  python ocr_ensemble_select.py [--run 12] [--emit-run N]
        --emit-run N writes ocr_{region}_N_ensemble.csv with the winning
        deployable variant (T7 selection + gangnam/suwon guarded snap).
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]

import sys as _sys; _sys.path[:0] = [str(HERE / _d) for _d in ("pipeline", "ocr", "vlm", "str_baselines", "detection", "data")]  # 폴더 재편 후 형제 모듈 import 경로
OCR = HERE / "artifacts" / "ocr_gt"
GTD = HERE / "artifacts" / "gt"

LAMBDA_DB = 0.15          # fixed a priori — do NOT tune on GSV GT (leakage)
REGIONS = ["gangnam", "suwon", "brooklyn"]
ENGINES = ["easyocr", "trocr", "paddle"]

# ---------------------------------------------------------------- selector norm
# Deployable normalization (independent of eval flags): NFKC, lower,
# keep only [0-9a-z가-힣] — same key space as the OCR DB.
def sel_norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    return re.sub(r"[^0-9a-z가-힣]", "", s)


def norm_key(s: str) -> str:
    return sel_norm(s)


def lev(a: str, b: str) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (0 if ai == b[j - 1] else 1))
        prev = cur
    return prev[m]


def sim(a: str, b: str) -> float:
    """1 - normalized edit distance on selector-normalized text."""
    A, B = sel_norm(a), sel_norm(b)
    if not A and not B:
        return 1.0
    d = lev(A, B)
    return 1.0 - d / max(len(A), len(B), 1)


# ---------------------------------------------------------------- dictionary
def load_db(region: str) -> set[str]:
    path = HERE / "artifacts" / "ocr_db" / f"db_{region}.csv"
    return {r["key"] for r in csv.DictReader(open(path, encoding="utf-8"))}


def db_hit_ratio(text: str, db: set[str]) -> float:
    """Char-weighted fraction of line/token keys that hit the DB exactly."""
    hit = tot = 0
    for line in re.split(r"\\n|\n", text or ""):
        line = line.strip()
        if not line:
            continue
        lk = norm_key(line)
        if len(lk) >= 2 and not lk.isdigit():
            tot += len(lk)
            if lk in db:
                hit += len(lk)
                continue        # whole line hit: tokens are covered
            for t in line.split():
                tk = norm_key(t)
                if len(tk) >= 2 and not tk.isdigit() and tk in db:
                    hit += len(tk)
    return hit / tot if tot else 0.0


# ---------------------------------------------------------------- data
def load_map(path: Path) -> dict[str, str]:
    return {r["image_name"].rsplit(".", 1)[0]: (r["gt_text"] or "").strip()
            for r in csv.DictReader(open(path, encoding="utf-8-sig"))}


def select_crop(cands: dict[str, str], engine_texts: list[str], db: set[str]) -> str:
    """Pick the candidate with max agreement(+db) score. cands: name->text."""
    best_name, best_score = None, -1e9
    for name, text in cands.items():
        if not (text or "").strip():
            score = -1.0        # empty: only wins if everything is empty
        else:
            ag = sum(sim(text, e) for e in engine_texts) / max(len(engine_texts), 1)
            score = ag + LAMBDA_DB * db_hit_ratio(text, db)
        if score > best_score:
            best_name, best_score = name, score
    return cands[best_name]


# ---------------------------------------------------------------- eval bridge
def load_eval_module():
    saved = sys.argv
    sys.argv = ["pipeline/eval_ocr_v2.py", "--mask-phone", "--en-only-regions", "brooklyn"]
    import eval_ocr_v2 as E
    sys.argv = saved
    return E


def eval_variant(E, gt_maps, pred_maps) -> dict:
    """Official-protocol metrics pooled over regions. pred_maps: region->map."""
    tot = None
    for region in REGIONS:
        E.EN_ONLY = region in E.EN_ONLY_REGIONS
        det = []
        a = E.eval_engine(gt_maps[region], pred_maps[region], det)
        if tot is None:
            tot = a
        else:
            for f in ("edit", "fp_edit", "chars", "word_lcs", "gt_words",
                      "exact", "recalled", "lines", "contain", "empty_pred", "crops"):
                setattr(tot, f, getattr(tot, f) + getattr(a, f))
    E.EN_ONLY = False
    return {"CER": tot.cer(), "WAR": tot.war(), "exact": tot.exact_rate(),
            "contain": tot.contain_rate(), "lines": tot.lines}


def crop_oracle_map(E, region, gt_map, cand_maps) -> dict[str, str]:
    """Best whole-candidate per crop BY GT (upper bound for crop-level selection)."""
    E.EN_ONLY = region in E.EN_ONLY_REGIONS
    out = {}
    for key, gt_text in gt_map.items():
        gl, ndc = E.split_gt_lines(gt_text)
        if not gl:
            continue
        best, best_rank = "", None
        for m in cand_maps:
            pred = m.get(key, "")
            pl = E.split_lines(pred) if pred.strip() else []
            assign, _ = E.match_lines(gl, pl, infix=(ndc > 0))
            ex = sum(1 for i, g in enumerate(gl)
                     if assign[i][0] is not None and E.norm_cer(g) and assign[i][1] == 0)
            edit = sum(a[1] for a in assign)
            rank = (-ex, edit)
            if best_rank is None or rank < best_rank:
                best, best_rank = pred, rank
        out[key] = best
    E.EN_ONLY = False
    return out


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="12", help="engine run to select from")
    ap.add_argument("--emit-run", default=None,
                    help="write winning deployable variant as ocr_{region}_N_ensemble.csv")
    args = ap.parse_args()

    E = load_eval_module()
    import ocr_db_snap as S

    gt_maps, eng_maps, ens_maps, dbs = {}, {}, {}, {}
    for region in REGIONS:
        gt_maps[region] = E.load_csv_map(GTD / f"ocr_{region}_gt.csv")
        eng_maps[region] = {e: load_map(OCR / f"ocr_{region}_{args.run}_{e}.csv")
                            for e in ENGINES}
        ens_maps[region] = load_map(OCR / f"ocr_{region}_{args.run}_ensemble.csv")
        dbs[region] = load_db(region)

    def snapped(region: str, pmap: dict[str, str]) -> dict[str, str]:
        """Guarded dictionary snap (T5) — official protocol applies it to gn/sw only."""
        if region == "brooklyn":
            return pmap
        db = S.load_db(region)
        stats = {"exact": 0, "exact_keep": 0, "fuzzy": 0, "trail_block": 0, "lines": 0}
        out = {}
        for k, text in pmap.items():
            parts = text.split("\\n") if "\\n" in text else [text]
            out[k] = "\\n".join(S.snap_line(p.strip(), db, stats) for p in parts)
        return out

    def t7_maps(include_ens: bool) -> dict[str, dict[str, str]]:
        sel = {}
        for region in REGIONS:
            db = dbs[region]
            emap = eng_maps[region]
            keys = set().union(*[m.keys() for m in emap.values()], ens_maps[region].keys())
            out = {}
            for k in keys:
                cands = {e: emap[e].get(k, "") for e in ENGINES}
                if include_ens:
                    cands["ens"] = ens_maps[region].get(k, "")
                engine_texts = [emap[e].get(k, "") for e in ENGINES]
                out[k] = select_crop(cands, engine_texts, db)
            sel[region] = out
        return sel

    rows = []

    def add(name, pred_maps):
        rows.append((name, eval_variant(E, gt_maps, pred_maps)))

    # --- baselines (전에 했던 방법들)
    add("paddle 단독", {r: eng_maps[r]["paddle"] for r in REGIONS})
    add("conf-argmax 앙상블 (raw)", dict(ens_maps))
    add("conf-argmax + 사전스냅 gn/sw (현 공식)",
        {r: snapped(r, ens_maps[r]) for r in REGIONS})

    # --- T7 variants
    t3 = t7_maps(include_ens=False)
    add("T7 합의+DB (3후보)", t3)
    t4 = t7_maps(include_ens=True)
    add("T7 합의+DB (4후보: +conf앙상블)", t4)
    t4s = {r: snapped(r, t4[r]) for r in REGIONS}
    add("T7 4후보 + 사전스냅 gn/sw", t4s)

    # --- T7b: reliability-weighted agreement. Engine weights come from the
    # IN-DOMAIN signboard_v3 test exact (Paddle .860 / TrOCR .729 / Easy .716)
    # — fixed a priori, no GSV tuning (no leakage).
    W = {"easyocr": 0.716, "trocr": 0.729, "paddle": 0.860}

    def t7_weighted(lam: float = LAMBDA_DB) -> dict[str, dict[str, str]]:
        sel = {}
        for region in REGIONS:
            db, emap = dbs[region], eng_maps[region]
            keys = set().union(*[m.keys() for m in emap.values()], ens_maps[region].keys())
            out = {}
            for k in keys:
                cands = {e: emap[e].get(k, "") for e in ENGINES}
                cands["ens"] = ens_maps[region].get(k, "")
                best_n, best_s = None, -1e9
                for name, text in cands.items():
                    if not (text or "").strip():
                        s = -1.0
                    else:
                        ag = sum(W[e] * sim(text, emap[e].get(k, "")) for e in ENGINES) \
                             / sum(W.values())
                        s = ag + lam * db_hit_ratio(text, db)
                    if s > best_s:
                        best_n, best_s = name, s
                out[k] = cands[best_n]
            sel[region] = out
        return sel

    t7b = t7_weighted()
    add("T7b 신뢰도가중 합의+DB (4후보)", t7b)
    add("T7b ablation: 합의만 (λ_DB=0)", t7_weighted(0.0))
    t7b_s = {r: snapped(r, t7b[r]) for r in REGIONS}
    add("T7b + 사전스냅 gn/sw", t7b_s)

    # --- T7c: paddle-primary (in-domain best engine), conf-ensemble fallback
    # only when paddle output is empty.
    t7c = {r: {k: (eng_maps[r]["paddle"].get(k, "").strip()
                   or ens_maps[r].get(k, ""))
               for k in set(eng_maps[r]["paddle"]) | set(ens_maps[r])}
           for r in REGIONS}
    add("T7c paddle우선 + 빈출력만 앙상블 폴백", t7c)
    add("T7c + 사전스냅 gn/sw", {r: snapped(r, t7c[r]) for r in REGIONS})

    # --- oracles
    add("crop-oracle (4후보, GT사용 상한)",
        {r: crop_oracle_map(E, r, gt_maps[r],
                            [eng_maps[r][e] for e in ENGINES] + [ens_maps[r]])
         for r in REGIONS})

    print(f"\n{'변형':44s} {'exact':>7s} {'CER':>7s} {'WAR':>7s} {'contain':>8s}")
    for name, m in rows:
        print(f"{name:44s} {m['exact']*100:6.1f}% {m['CER']:7.3f} "
              f"{m['WAR']:7.3f} {m['contain']*100:7.1f}%")
    print("(line-oracle 상한: exact 53.4% / CER 0.349 — eval_ocr_v2 oracle)")

    # per-region breakdown of the deployable pick (T7b raw — snap hurts on T7b)
    print("\n[T7b 지역별]")
    for region in REGIONS:
        E.EN_ONLY = region in E.EN_ONLY_REGIONS
        det = []
        a = E.eval_engine(gt_maps[region], t7b[region], det)
        E.EN_ONLY = False
        print(f"  {region:9s} exact={a.exact_rate()*100:5.1f}%  CER={a.cer():.3f}  "
              f"WAR={a.war():.3f}  contain={a.contain_rate()*100:5.1f}%")

    if args.emit_run:
        for region in REGIONS:
            dst = OCR / f"ocr_{region}_{args.emit_run}_ensemble.csv"
            with open(dst, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f, quoting=csv.QUOTE_ALL)
                w.writerow(["image_name", "gt_text"])
                for k in sorted(t7b[region]):
                    w.writerow([k, t7b[region][k]])
            print(f"[emit] {dst.name}: {len(t7b[region])} crops (T7b raw)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
