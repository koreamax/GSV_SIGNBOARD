#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the OCR snap dictionary (artifacts/ocr_db) — see artifacts/ocr_db/DB_RULES.md.

L1: OSM POI names per region (Overpass API, public data — matches the paper's
    OSM-enhancement framing; NEVER sourced from the GSV eval GT).
L2: signboard_v3 TRAIN-split vocabulary (freq >= --min-freq); val/test excluded.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DB_DIR = HERE / "artifacts" / "ocr_db"
TRAIN_TXT = HERE / "artifacts" / "ocr_training" / "signboard_v3" / "train.txt"

# Approximate bboxes (south, west, north, east) around the GSV shooting areas.
BBOX = {
    "gangnam": (37.488, 127.018, 37.512, 127.070),   # 강남역~대치동
    "suwon": (37.255, 127.015, 37.285, 127.050),     # 수원 인계동 일대
    "brooklyn": (40.655, -73.965, 40.695, -73.935),  # Bed-Stuy/Crown Heights (Nostrand/Franklin/Fulton)
}
POI_TAGS = ["shop", "amenity", "office", "craft", "leisure", "healthcare", "tourism"]
OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]


def norm_key(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    return re.sub(r"[^0-9a-z가-힣]", "", s)


def fetch_osm(region: str) -> list[tuple[str, str, str]]:
    s, w, n, e = BBOX[region]
    parts = []
    for t in POI_TAGS:
        parts.append(f'node["name"]["{t}"]({s},{w},{n},{e});')
        parts.append(f'way["name"]["{t}"]({s},{w},{n},{e});')
    q = f"[out:json][timeout:90];({''.join(parts)});out tags;"
    data = None
    last_err = None
    for attempt in range(6):
        url = OVERPASS_MIRRORS[attempt % len(OVERPASS_MIRRORS)]
        try:
            req = urllib.request.Request(url, data=q.encode("utf-8"),
                                         headers={"User-Agent": "gsv-ocr-db/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.loads(r.read().decode("utf-8"))
            break
        except Exception as exc:            # 504/429/timeouts: rotate mirror and retry
            last_err = exc
            print(f"[L1] {region}: {url.split('/')[2]} failed ({exc}); retrying...")
            time.sleep(5 * (attempt + 1))
    if data is None:
        raise RuntimeError(f"Overpass failed for {region}: {last_err}")
    rows = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        names = {tags.get("name", "")}
        names.add(tags.get("name:ko", ""))
        names.add(tags.get("name:en", ""))
        kind = next((f"{t}={tags[t]}" for t in POI_TAGS if t in tags), "")
        for nm in names:
            nm = (nm or "").strip()
            k = norm_key(nm)
            if len(k) >= 2:
                rows.append((nm, k, kind))
    return rows


def build_vocab(min_freq: int) -> list[tuple[str, str, int]]:
    cnt: Counter[str] = Counter()
    surface: dict[str, Counter] = {}
    with open(TRAIN_TXT, encoding="utf-8") as f:
        for line in f:
            if "\t" not in line:
                continue
            word = line.rstrip("\n").split("\t", 1)[1].strip()
            k = norm_key(word)
            if len(k) < 2:
                continue
            cnt[k] += 1
            surface.setdefault(k, Counter())[word] += 1
    out = []
    for k, c in cnt.items():
        if c >= min_freq:
            best = surface[k].most_common(1)[0][0]
            out.append((best, k, c))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-freq", type=int, default=3,
                    help="L2 vocab: keep train words appearing at least this often.")
    ap.add_argument("--skip-osm", action="store_true", help="Rebuild L2/merge only.")
    args = ap.parse_args()
    DB_DIR.mkdir(parents=True, exist_ok=True)

    # ---- L1: OSM per region
    if not args.skip_osm:
        for region in BBOX:
            rows = fetch_osm(region)
            # dedupe by key, keep first surface form
            seen, ded = set(), []
            for nm, k, kind in rows:
                if k not in seen:
                    seen.add(k)
                    ded.append((nm, k, kind))
            with open(DB_DIR / f"osm_{region}.csv", "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["name", "key", "tag"])
                w.writerows(ded)
            print(f"[L1] {region}: {len(ded)} OSM POI names")
            time.sleep(3)   # be polite to Overpass

    # ---- L2: signboard train vocab
    vocab = build_vocab(args.min_freq)
    with open(DB_DIR / "vocab_signboard.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["word", "key", "freq"])
        w.writerows(sorted(vocab, key=lambda r: -r[2]))
    print(f"[L2] signboard train vocab (freq>={args.min_freq}): {len(vocab)} entries")

    # ---- merge per region: L1 wins on key collision (real POI > generic vocab)
    for region in BBOX:
        merged: dict[str, tuple[str, str]] = {}
        for _, row in enumerate(csv.DictReader(open(DB_DIR / "vocab_signboard.csv", encoding="utf-8"))):
            merged[row["key"]] = (row["word"], "vocab")
        osm_path = DB_DIR / f"osm_{region}.csv"
        if osm_path.exists():
            for row in csv.DictReader(open(osm_path, encoding="utf-8")):
                merged[row["key"]] = (row["name"], "osm")
        with open(DB_DIR / f"db_{region}.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "key", "source"])
            for k, (nm, src) in merged.items():
                w.writerow([nm, k, src])
        print(f"[DB] db_{region}.csv: {len(merged)} entries")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
