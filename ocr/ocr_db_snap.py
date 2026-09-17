#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Snap OCR predictions to the OCR dictionary (rules: artifacts/ocr_db/DB_RULES.md).

Reads  artifacts/ocr_gt/ocr_{region}_{run}_{engine}.csv
Writes artifacts/ocr_gt/ocr_{region}_{run}_{engine}dict.csv  (same schema)

Conservative by design: a line is replaced ONLY when the normalized key hits the
dictionary exactly, or has a UNIQUE nearest neighbour within the edit budget.
Number-only / phone-like / very short lines are never snapped.

Guards (2026-08-07, measured on run12):
- fuzzy never deletes the prediction's trailing 1-2 chars (NAILS->NAIL, SALES->SALE,
  ollehO->olleh were pure damage; the restoring direction ELEVE->ELEVEn stays allowed).
- exact-key hit adopts the DB surface only if it does not ADD punctuation
  (Dominos->Domino's). Metric-neutral under eval's strict norm; output hygiene + WAR respacing.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
DB_DIR = HERE / "artifacts" / "ocr_db"
OCR_DIR = HERE / "artifacts" / "ocr_gt"

PHONE_RE = re.compile(r"(?:\+\d{1,3}[-.)\s]?)?\d{2,4}[-.)\s]\s?(?:\d{3,4}[-.\s]\s?)?\d{4}(?!\d)")


def norm_key(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    return re.sub(r"[^0-9a-z가-힣]", "", s)


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
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if ai == b[j - 1] else 1))
        prev = cur
    return prev[m]


def load_db(region: str) -> dict[str, str]:
    path = DB_DIR / f"db_{region}.csv"
    return {r["key"]: r["name"] for r in csv.DictReader(open(path, encoding="utf-8"))}


_PUNCT_RE = re.compile(r"[^0-9A-Za-z가-힣\s]")


def _punct(s: str) -> int:
    return len(_PUNCT_RE.findall(s))


def _match(key: str, db: dict[str, str], stats: dict, orig: str) -> str | None:
    if key in db:
        # same key = same text modulo case/space/punct. Adopt the DB surface only
        # if it does not ADD punctuation (OSM names carry apostrophes/commas the
        # sign transcription usually lacks: Dominos -> Domino's). Keeping the
        # original is exact/CER-neutral (strict norm); adoption restores spacing.
        if _punct(db[key]) <= _punct(orig):
            stats["exact"] += 1
            return db[key]
        stats["exact_keep"] += 1
        return orig
    # fuzzy only for latin/digit keys: a single hangul-syllable edit is a
    # different word entirely (정관장 vs 정관정), so Korean gets exact-match only
    if not re.fullmatch(r"[0-9a-z]+", key):
        return None
    budget = 0 if len(key) < 5 else (1 if len(key) <= 8 else 2)
    lo, hi = len(key) * 0.5, len(key) * 2.0
    best_d, best_keys = budget + 1, []
    for k in db:
        if not (lo <= len(k) <= hi) or abs(len(k) - len(key)) > budget:
            continue
        # never snap by deleting the prediction's trailing chars: that direction
        # is English inflection damage (NAILS->NAIL, SALES->SALE, ollehO->olleh).
        # The restoring direction (ELEVE->ELEVEn) remains allowed.
        if k == key[:-1] or (len(key) > 3 and k == key[:-2]):
            stats["trail_block"] += 1
            continue
        d = lev(key, k)
        if d < best_d:
            best_d, best_keys = d, [k]
        elif d == best_d:
            best_keys.append(k)
    if best_d <= budget and len(best_keys) == 1:
        stats["fuzzy"] += 1
        return db[best_keys[0]]
    return None


def snap_line(line: str, db: dict[str, str], stats: dict) -> str:
    key = norm_key(line)
    if len(key) < 2 or key.isdigit() or PHONE_RE.search(line):
        return line
    whole = _match(key, db, stats, line)
    if whole is not None:
        return whole
    # token-level: predictions are often space-joined multi-word lines
    tokens = line.split()
    if len(tokens) < 2:
        return line
    out = []
    for t in tokens:
        tk = norm_key(t)
        if len(tk) < 2 or tk.isdigit() or PHONE_RE.search(t):
            out.append(t)
            continue
        m = _match(tk, db, stats, t)
        out.append(m if m is not None else t)
    return " ".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True, choices=["gangnam", "brooklyn", "suwon"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--engine", default="ensemble")
    args = ap.parse_args()

    db = load_db(args.region)
    src = OCR_DIR / f"ocr_{args.region}_{args.run}_{args.engine}.csv"
    dst = OCR_DIR / f"ocr_{args.region}_{args.run}_{args.engine}dict.csv"
    stats = {"exact": 0, "exact_keep": 0, "fuzzy": 0, "trail_block": 0, "lines": 0}

    rows_out = []
    for row in csv.DictReader(open(src, encoding="utf-8-sig")):
        text = row["gt_text"]
        # predictions are line-joined with literal \n or single-line; handle both
        parts = text.split("\\n") if "\\n" in text else [text]
        snapped = []
        for p in parts:
            stats["lines"] += 1
            snapped.append(snap_line(p.strip(), db, stats))
        rows_out.append((row["image_name"], "\\n".join(snapped)))

    with open(dst, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["image_name", "gt_text"])
        w.writerows(rows_out)
    print(f"[SNAP] {src.name} -> {dst.name}  lines={stats['lines']} "
          f"exact_hit={stats['exact']} exact_keep={stats['exact_keep']} "
          f"fuzzy_snap={stats['fuzzy']} trail_block={stats['trail_block']}  db={len(db)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
