#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_ocr_v2.py  —  GSV out-of-domain OCR evaluation (Standard v1)

Design (locked by research decision):
  - UNIT:         line-level matching (GT split by '|' or '\n')
  - MATCHING:     greedy global-min 1:1 assignment of GT lines <-> pred lines
  - NORMALIZE:    NFKC -> lowercase -> strict keep [0-9 a-z 가-힣] (default ON)
  - CER:          MICRO (sum edit / sum GT chars), spaces removed
  - WAR:          Word Accuracy Rate = LCS(pred_words, gt_words) / gt_words  (in [0,1])
  - AGGREGATION:  micro, per-region and global
  - SELECTION:    per-engine (deployable) reported as MAIN; oracle best-of-two = UPPER BOUND only
  - EDGE CASES:   no detection / empty pred -> CER 1.0; extra pred lines -> false positives
  - DIAGNOSTIC:   containment hit = is norm(gt_line) a substring of norm(whole_pred)?
                  (separates "granularity loss" from "model can't read")

Inputs:
  GT :  artifacts/gt/ocr_{region}_gt.csv          (image_id|image_name, gt_text)
  OCR:  artifacts/ocr_gt/ocr_{region}_{run}_{engine}.csv  (image_name, gt_text=pred)
        -> latest run per (region, engine) is used.

Outputs:
  artifacts/gt/ocr_eval_v2_summary.csv
  artifacts/gt/ocr_eval_v2_details.csv
"""

import argparse
import csv
import glob
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple, Optional

HERE = Path(__file__).resolve().parents[1]
GT_DIR = HERE / "artifacts" / "gt"
OCR_DIR = HERE / "artifacts" / "ocr_gt"
REGIONS = ["gangnam", "brooklyn", "suwon"]
ENGINES = ["easyocr", "trocr", "paddle"]   # overridden by --engines after ARGS is parsed


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="GSV OCR evaluation standard v1")
    p.add_argument("--no-strict", action="store_true",
                   help="Disable strict filter (keep punctuation / other scripts).")
    p.add_argument("--mask-phone", action="store_true",
                   help="Strip phone-number patterns from BOTH GT and pred before scoring "
                        "(e.g. 02-532-0870, 010 1234 5678, 718.387.1234, 1588-1234).")
    p.add_argument("--en-only-regions", type=str, default="",
                   help="Comma-separated regions scored English-only: hangul is removed "
                        "from BOTH GT and pred before scoring (e.g. brooklyn).")
    p.add_argument("--keep-fp", action="store_true", default=True,
                   help="Count spurious (unmatched) pred lines as insertion errors in CER. Default on.")
    p.add_argument("--out-dir", type=str, default=str(GT_DIR))
    p.add_argument("--engines", type=str, default="easyocr,trocr,paddle",
                   help="Comma-separated engine tags to score (ocr_<region>_<run>_<engine>.csv). "
                        "e.g. easyocr,trocr,paddle,tesseract,surya,parseq,svtrv2")
    return p.parse_args()


ARGS = parse_args()
ENGINES = [e.strip() for e in ARGS.engines.split(",") if e.strip()]
STRICT = not ARGS.no_strict
EN_ONLY_REGIONS = {r.strip() for r in ARGS.en_only_regions.split(",") if r.strip()}
EN_ONLY = False   # set per region in main(); read by _base_norm


# ----------------------------------------------------------------------
# Normalization (applied identically to GT and pred)
# ----------------------------------------------------------------------
_STRICT_RE = re.compile(r"[^0-9a-z가-힣\s]")

# Phone-number shapes on signboards: 02-532-0870 / 02)532-0870 / 010 1234 5678 /
# 718.387.1234 / 1588-1234 / +82-10-1234-5678. Separators still present at this
# point (runs before the strict filter), so we match on them to avoid eating
# plain numbers like years or floor labels.
_PHONE_RE = re.compile(
    r"(?:\+\d{1,3}[-.)\s]?)?"          # optional country code
    r"\d{2,4}[-.)\s]\s?"               # area/prefix + separator
    r"(?:\d{3,4}[-.\s]\s?)?"           # optional middle group (1588-1234 has none)
    r"\d{4}(?!\d)"                     # last 4 digits
)


def _base_norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")   # fold fullwidth, compose hangul
    s = s.replace("​", "")                  # zero-width space
    s = s.lower()
    if ARGS.mask_phone:
        s = _PHONE_RE.sub(" ", s)            # applied to GT and pred alike
    if EN_ONLY:
        s = re.sub(r"[가-힣]", "", s)        # English-only region: drop hangul both sides
    if STRICT:
        s = _STRICT_RE.sub("", s)
    return s


def norm_cer(s: str) -> str:
    """For CER: remove ALL whitespace."""
    return re.sub(r"\s+", "", _base_norm(s))


def norm_words(s: str) -> List[str]:
    """For WAR: collapse whitespace, tokenize."""
    return _base_norm(s).split()


# ----------------------------------------------------------------------
# Edit distance + LCS
# ----------------------------------------------------------------------
def lev(a, b) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    cur = [0] * (m + 1)
    for i in range(1, n + 1):
        cur[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if ai == b[j - 1] else 1))
        prev, cur = cur, prev
    return prev[m]


def lev_infix(P, G) -> int:
    """Min edit distance between G and the best-matching SUBSTRING of P
    (approximate substring matching: P's prefix/suffix outside the match are
    free). Used for don't-care (###) crops, where pred lines may legitimately
    contain extra text the OCR read from the illegible region."""
    if not G:
        return 0
    if not P:
        return len(G)
    prev = [0] * (len(P) + 1)          # match may start anywhere in P
    for i in range(1, len(G) + 1):
        cur = [i] + [0] * len(P)
        gi = G[i - 1]
        for j in range(1, len(P) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (0 if gi == P[j - 1] else 1))
        prev = cur
    return min(prev)                    # match may end anywhere in P


def lcs_len(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prevdiag = 0
        for j in range(1, m + 1):
            tmp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prevdiag + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prevdiag = tmp
    return dp[m]


# ----------------------------------------------------------------------
# Line splitting + matching
# ----------------------------------------------------------------------
def split_lines(text: str) -> List[str]:
    s = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\\n", "\n").strip()   # GT csv stores line breaks as literal backslash-n
    if not s:
        return []
    parts = s.split("|") if "|" in s else s.split("\n")
    return [p.strip() for p in parts if p.strip()]


# "###" on its own line = ICDAR-style don't-care: text visible on the sign but
# too small/degraded for a human to read. Such lines are dropped from scoring,
# and crops containing them get no false-positive penalty for extra pred lines
# (the OCR may have partially read the illegible region).
DONTCARE = "###"


def split_gt_lines(text: str) -> Tuple[List[str], int]:
    """GT lines minus don't-care markers. Returns (scorable_lines, n_dontcare)."""
    lines = split_lines(text)
    keep = [l for l in lines if l != DONTCARE]
    return keep, len(lines) - len(keep)


def cer_pair(pred: str, gt: str) -> Tuple[int, int]:
    """return (edit_distance, gt_char_len) on CER-normalized strings."""
    P, G = norm_cer(pred), norm_cer(gt)
    return lev(P, G), len(G)


def match_lines(gt_lines: List[str], pred_lines: List[str], infix: bool = False):
    """Greedy global-min 1:1 assignment.
    infix=True (don't-care crops): score each pair by best-substring edit
    distance so extra text concatenated into a pred line costs nothing.
    Returns: assign = list over gt index -> (pred_idx or None, edit, gt_len)
             unmatched_pred = list of pred idx not used (false positives)
    """
    pairs = []
    norm_g = [norm_cer(g) for g in gt_lines]
    norm_p = [norm_cer(p) for p in pred_lines]
    for i, G in enumerate(norm_g):
        for j, P in enumerate(norm_p):
            d = lev_infix(P, G) if infix else lev(P, G)
            denom = max(len(G), 1)
            pairs.append((d / denom, d, i, j))
    pairs.sort(key=lambda x: x[0])

    assigned: Dict[int, Tuple[int, int]] = {}
    used_pred = set()
    for _, d, i, j in pairs:
        if i in assigned or j in used_pred:
            continue
        assigned[i] = (j, d)
        used_pred.add(j)

    assign = []
    for i, g in enumerate(gt_lines):
        gl = len(norm_g[i])
        if i in assigned:
            j, d = assigned[i]
            assign.append((j, d, gl))
        else:
            assign.append((None, gl, gl))  # unmatched -> all deletions (edit=gt_len)
    unmatched_pred = [j for j in range(len(pred_lines)) if j not in used_pred]
    return assign, unmatched_pred


# ----------------------------------------------------------------------
# IO
# ----------------------------------------------------------------------
def load_csv_map(path: Path) -> Dict[str, str]:
    raw = path.read_bytes()
    try:
        txt = raw.decode("utf-8-sig")   # tolerates a BOM; identical to utf-8 without one
    except UnicodeDecodeError:
        txt = raw.decode("cp949", "replace")
    r = csv.DictReader(txt.splitlines())
    fields = r.fieldnames or []
    kc = next((c for c in ["image_id", "image_name", "filename"] if c in fields), None)
    tc = next((c for c in ["gt_text", "ocr_text", "text"] if c in fields), None)
    out = {}
    if not kc or not tc:
        return out
    for row in r:
        k = (row.get(kc) or "").strip()
        if not k:
            continue
        out[k.rsplit(".", 1)[0]] = (row.get(tc) or "").strip()
    return out


def latest_ocr(region: str, engine: str) -> Optional[Path]:
    cands = list(OCR_DIR.glob(f"ocr_{region}_*_{engine}.csv"))
    if not cands:
        return None
    cands.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return cands[0]


# ----------------------------------------------------------------------
# Per-engine evaluation accumulator
# ----------------------------------------------------------------------
class Acc:
    def __init__(self):
        self.edit = 0          # CER numerator (matched + deletions)
        self.fp_edit = 0       # insertions from spurious pred lines
        self.chars = 0         # CER denominator (GT chars)
        self.word_lcs = 0
        self.word_edit = 0     # WER numerator: word-level edit distance (matched + deleted GT lines)
        self.fp_words = 0      # insertions from spurious pred lines (counted when keep_fp)
        self.gt_words = 0
        self.exact = 0
        self.recalled = 0      # lines with CER < 1.0
        self.lines = 0
        self.contain = 0       # diagnostic: gt line substring of whole pred
        self.empty_pred = 0
        self.crops = 0

    def cer(self):
        num = self.edit + (self.fp_edit if ARGS.keep_fp else 0)
        return num / self.chars if self.chars else 0.0

    def war(self):
        return self.word_lcs / self.gt_words if self.gt_words else 0.0

    def wer(self):
        num = self.word_edit + (self.fp_words if ARGS.keep_fp else 0)
        return num / self.gt_words if self.gt_words else 0.0

    def exact_rate(self):
        return self.exact / self.lines if self.lines else 0.0

    def recall(self):
        return self.recalled / self.lines if self.lines else 0.0

    def contain_rate(self):
        return self.contain / self.lines if self.lines else 0.0


def eval_engine(gt_map, pred_map, det_rows) -> Acc:
    a = Acc()
    for key, gt_text in gt_map.items():
        gt_lines, n_dontcare = split_gt_lines(gt_text)
        if not gt_lines:
            continue   # empty GT, or every line was don't-care (###)
        a.crops += 1
        pred = pred_map.get(key, "")
        if not pred.strip():
            a.empty_pred += 1
        pred_lines = split_lines(pred) if pred.strip() else []
        whole_pred_norm = norm_cer(pred)

        assign, unmatched = match_lines(gt_lines, pred_lines,
                                        infix=(n_dontcare > 0))

        for i, g in enumerate(gt_lines):
            j, edit, gl = assign[i]
            a.lines += 1
            a.edit += edit
            a.chars += gl
            # exact + recall (edit==0 covers both modes: full equality for
            # strict crops, perfect-substring for don't-care crops)
            G = norm_cer(g)
            if j is not None and G and edit == 0:
                a.exact += 1
            if gl > 0 and (edit / gl) < 1.0:
                a.recalled += 1
            # containment diagnostic (granularity-agnostic)
            if G and G in whole_pred_norm:
                a.contain += 1
            # WAR (LCS of words within this GT line vs whole pred words)
            gw = norm_words(g)
            pw = norm_words(pred_lines[j]) if j is not None else []
            a.word_lcs += lcs_len(pw, gw)
            a.word_edit += lev(pw, gw)
            a.gt_words += len(gw)

        # false-positive (spurious) pred lines -> insertion cost.
        # Skipped when the crop has don't-care (###) regions: the extra pred
        # lines may be partial readings of the illegible text.
        if n_dontcare == 0:
            for j in unmatched:
                a.fp_edit += len(norm_cer(pred_lines[j]))
                a.fp_words += len(norm_words(pred_lines[j]))

        det_rows.append((key, len(gt_lines), len(pred_lines), len(unmatched)))
    return a


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def fmt(a: Acc) -> str:
    return (f"CER={a.cer():.4f}  WER={a.wer():.4f}  WAR={a.war():.4f}  exact={a.exact_rate()*100:5.1f}%  "
            f"recall={a.recall()*100:5.1f}%  contain={a.contain_rate()*100:5.1f}%  "
            f"empty={a.empty_pred}/{a.crops}  lines={a.lines}")


def main():
    out_dir = Path(ARGS.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summ_path = out_dir / "ocr_eval_v2_summary.csv"
    det_path = out_dir / "ocr_eval_v2_details.csv"

    print("=" * 78)
    print(f"GSV OCR Evaluation Standard v1   (strict={STRICT}, unit=line, agg=micro)")
    print(f"  norm: NFKC -> lower" + (" -> keep[0-9a-z가-힣]" if STRICT else " (raw)"))
    print(f"  CER=micro char  WAR=word-LCS/gt-words  CER>1 possible (FP insertions)")
    print("=" * 78)

    summ_rows = []
    global_acc = {e: Acc() for e in ENGINES}
    global_ens = Acc()
    # oracle accumulators (line-level min CER across engines)
    g_oracle = Acc()

    with det_path.open("w", newline="", encoding="utf-8") as fdet:
        wdet = csv.writer(fdet)
        wdet.writerow(["region", "engine", "image_id", "gt_lines", "pred_lines", "fp_pred_lines"])

        for region in REGIONS:
            gtp = GT_DIR / f"ocr_{region}_gt.csv"
            if not gtp.exists():
                continue
            global EN_ONLY
            EN_ONLY = region in EN_ONLY_REGIONS
            gt_map = load_csv_map(gtp)

            engine_files = {e: latest_ocr(region, e) for e in ENGINES}
            present = [e for e in ENGINES if engine_files[e]]
            if not present:
                print(f"\n[{region}]  GT crops={len(gt_map)}  ->  NO OCR RESULTS (skip)")
                continue

            print(f"\n[{region}]  GT crops={len(gt_map)}   engines={present}")
            for e in present:
                print(f"           {e:8} <- {engine_files[e].name}")

            accs = {}
            for e in present:
                pred_map = load_csv_map(engine_files[e])
                det_rows = []
                a = eval_engine(gt_map, pred_map, det_rows)
                accs[e] = a
                for (k, gl, pl, fp) in det_rows:
                    wdet.writerow([region, e, k, gl, pl, fp])
                # accumulate global
                _merge(global_acc[e], a)
                print(f"   [{e:8}] {fmt(a)}")
                summ_rows.append([region, e, a.lines, f"{a.cer():.4f}", f"{a.war():.4f}",
                                  f"{a.exact_rate():.4f}", f"{a.recall():.4f}",
                                  f"{a.contain_rate():.4f}", a.empty_pred, a.crops, f"{a.wer():.4f}"])

            # deployable selection ensemble (if produced by run_ocr_only --box)
            ens_file = latest_ocr(region, "ensemble")
            if ens_file:
                pred_map = load_csv_map(ens_file)
                a = eval_engine(gt_map, pred_map, [])
                _merge(global_ens, a)
                print(f"   [ensemble] {fmt(a)}   <- deployable selection (conf-based)")
                summ_rows.append([region, "ensemble", a.lines, f"{a.cer():.4f}", f"{a.war():.4f}",
                                  f"{a.exact_rate():.4f}", f"{a.recall():.4f}",
                                  f"{a.contain_rate():.4f}", a.empty_pred, a.crops, f"{a.wer():.4f}"])

            # paper-style oracle (line-level min CER across ALL present engines)
            if len(present) >= 2:
                orc = _oracle(gt_map, {e: load_csv_map(engine_files[e]) for e in present})
                _merge(g_oracle, orc)
                print(f"   [ORACLE ] {fmt(orc)}   <- paper-style best-of-{len(present)} (needs GT)")
                summ_rows.append([region, "oracle", orc.lines, f"{orc.cer():.4f}", f"{orc.war():.4f}",
                                  f"{orc.exact_rate():.4f}", f"{orc.recall():.4f}",
                                  f"{orc.contain_rate():.4f}", orc.empty_pred, orc.crops, f"{orc.wer():.4f}"])

    # global summary
    print("\n" + "=" * 78)
    print("GLOBAL (all evaluated regions pooled, micro)")
    for e in ENGINES:
        if global_acc[e].lines:
            print(f"   [{e:8}] {fmt(global_acc[e])}")
    if global_ens.lines:
        print(f"   [ensemble] {fmt(global_ens)}   <- deployable selection")
    if g_oracle.lines:
        print(f"   [ORACLE ] {fmt(g_oracle)}   <- upper bound")

    with summ_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["region", "engine", "gt_lines", "CER_micro", "WAR", "exact",
                    "line_recall", "contain", "empty_pred", "crops", "WER"])
        w.writerows(summ_rows)
        for e in ENGINES:
            a = global_acc[e]
            if a.lines:
                w.writerow(["ALL", e, a.lines, f"{a.cer():.4f}", f"{a.war():.4f}",
                            f"{a.exact_rate():.4f}", f"{a.recall():.4f}",
                            f"{a.contain_rate():.4f}", a.empty_pred, a.crops, f"{a.wer():.4f}"])
        if global_ens.lines:
            a = global_ens
            w.writerow(["ALL", "ensemble", a.lines, f"{a.cer():.4f}", f"{a.war():.4f}",
                        f"{a.exact_rate():.4f}", f"{a.recall():.4f}",
                        f"{a.contain_rate():.4f}", a.empty_pred, a.crops, f"{a.wer():.4f}"])
        if g_oracle.lines:
            a = g_oracle
            w.writerow(["ALL", "oracle", a.lines, f"{a.cer():.4f}", f"{a.war():.4f}",
                        f"{a.exact_rate():.4f}", f"{a.recall():.4f}",
                        f"{a.contain_rate():.4f}", a.empty_pred, a.crops, f"{a.wer():.4f}"])

    print(f"\n[saved] {summ_path}")
    print(f"[saved] {det_path}")
    print("\nNOTE: per-engine = deployable metric (report this). oracle = upper bound only.")
    print("NOTE: low 'recall' with high 'contain' => granularity loss (sign-crop vs line-train),")
    print("      not a recognition failure. Fix by line-level cropping, then re-run.")


def _merge(dst: Acc, src: Acc):
    for f in ["edit", "fp_edit", "chars", "word_lcs", "word_edit", "fp_words", "gt_words", "exact",
              "recalled", "lines", "contain", "empty_pred", "crops"]:
        setattr(dst, f, getattr(dst, f) + getattr(src, f))


def _oracle(gt_map, pred_maps) -> Acc:
    """Line-level oracle: for each GT line, take the engine with lower CER."""
    a = Acc()
    es = list(pred_maps.keys())
    for key, gt_text in gt_map.items():
        gt_lines, n_dc = split_gt_lines(gt_text)   # oracle has no fp penalty; just drop ###
        if not gt_lines:
            continue
        a.crops += 1
        per_engine = {}
        whole = {}
        for e in es:
            pred = pred_maps[e].get(key, "")
            whole[e] = norm_cer(pred)
            pl = split_lines(pred) if pred.strip() else []
            per_engine[e] = (match_lines(gt_lines, pl, infix=(n_dc > 0)), pl)
        for i, g in enumerate(gt_lines):
            a.lines += 1
            G = norm_cer(g)
            gw = norm_words(g)
            a.gt_words += len(gw)
            best = None
            for e in es:
                (assign, _), pl = per_engine[e]
                j, edit, gl = assign[i]
                ratio = edit / gl if gl else 0.0
                P = norm_cer(pl[j]) if j is not None else ""
                pw = norm_words(pl[j]) if j is not None else []
                cand = (ratio, edit, gl, P, pw, e)
                if best is None or cand[0] < best[0]:
                    best = cand
            ratio, edit, gl, P, pw, e = best
            a.edit += edit
            a.chars += gl
            if G and edit == 0:
                a.exact += 1
            if gl > 0 and ratio < 1.0:
                a.recalled += 1
            if G and any(G in whole[x] for x in es):
                a.contain += 1
            a.word_lcs += lcs_len(pw, gw)
    return a


if __name__ == "__main__":
    main()
