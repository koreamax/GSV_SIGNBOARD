#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A/B eval for D18 spacecat experiment: v3 (no-space concat) vs v4 (gap+space).

Two in-domain tasks, both engines via the isolated paddle_rec_worker:

  1. single  — signboard_v3 test.txt as-is (7.7k word crops).
               v4 must NOT regress here (sanity).
  2. pairs   — N synthetic two-word strips built FROM TEST-SPLIT crops with the
               same gap recipe as v4 training (test split only -> unseen by both
               models; both models see identical images). GT = "A B".
               This directly measures the spacing hypothesis: does v4 emit the
               space between adjacent words where v3 joins them?

Metrics: exact / exact-ignoring-space / CER(nospace) / space-recall
(= among predictions whose spaceless text matches GT, how many contain a space).

Usage: .venv/Scripts/python.exe eval_paddle_ab.py [--pairs 1500] [--seed 42]
Writes: artifacts/ocr_training/signboard_v3/ab_spacecat_results.csv (+ console table)
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
V3DATA = HERE / "artifacts" / "ocr_training" / "signboard_v3"
PY = HERE / ".venv" / "Scripts" / "python.exe"
WORKER = HERE / "str_baselines/paddle_rec_worker.py"
MODELS = {
    "v3_baseline": HERE / "output" / "paddle_signboard_rec_v3" / "inference",
    "v4_spacecat": HERE / "output" / "paddle_signboard_rec_v4_spacecat" / "inference",
    "v5_lines": HERE / "output" / "paddle_signboard_rec_v5_lines" / "inference",
}
RESULTS = V3DATA / "ab_spacecat_results.csv"


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


def load_test() -> list[tuple[str, str]]:
    rows = []
    with (V3DATA / "test.txt").open(encoding="utf-8") as f:
        for line in f:
            if "\t" in line:
                p, t = line.rstrip("\n").split("\t", 1)
                rows.append((p, t.strip()))
    return rows


def build_pairs(rows: list[tuple[str, str]], n: int, seed: int, out_dir: Path) -> list[dict]:
    """Two-word strips with the v4 gap recipe, from TEST crops only."""
    import cv2

    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs, tries = [], 0
    while len(pairs) < n and tries < n * 30:
        tries += 1
        (pa, ta), (pb, tb) = rng.choice(rows), rng.choice(rows)
        if not ta or not tb or " " in ta or " " in tb:
            continue
        if len(ta) + len(tb) + 1 > 25:
            continue
        ia, ib = cv2.imread(str(V3DATA / pa)), cv2.imread(str(V3DATA / pb))
        if ia is None or ib is None:
            continue
        if ia.shape[1] / ia.shape[0] + ib.shape[1] / ib.shape[0] > 320 / 48:
            continue
        h = 48
        a = cv2.resize(ia, (max(1, round(ia.shape[1] / ia.shape[0] * h)), h))
        b = cv2.resize(ib, (max(1, round(ib.shape[1] / ib.shape[0] * h)), h))
        gap_w = max(2, int(h * rng.uniform(0.15, 0.40)))
        left = a[:, -1:, :].astype(np.float32)
        right = b[:, :1, :].astype(np.float32)
        t = np.linspace(0, 1, gap_w, dtype=np.float32)[None, :, None]
        gap = (left * (1 - t) + right * t).astype(a.dtype)
        strip = np.concatenate([a, gap, b], axis=1)
        p = out_dir / f"pair_{len(pairs):05d}.jpg"
        cv2.imwrite(str(p), strip)
        pairs.append({"path": str(p), "gt": f"{ta} {tb}"})
    return pairs


def run_worker(model_dir: Path, items: list[dict], tmp: Path, tag: str) -> dict[str, str]:
    man = tmp / f"manifest_{tag}.jsonl"
    out = tmp / f"out_{tag}.jsonl"
    with man.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps({"key": it["key"], "path": it["path"]}, ensure_ascii=False) + "\n")
    cmd = [str(PY), str(WORKER), "--manifest", str(man), "--out", str(out),
           "--model-dir", str(model_dir), "--device", "gpu"]
    subprocess.run(cmd, check=True)
    preds = {}
    with out.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            preds[d["key"]] = d["text"]
    return preds


def metrics(gts: list[str], preds: list[str]) -> dict:
    n = len(gts)
    ex = exns = edit = chars = sp_ok = sp_base = 0
    for g, p in zip(gts, preds):
        g_, p_ = g.strip(), (p or "").strip()
        gn, pn = g_.replace(" ", ""), p_.replace(" ", "")
        if p_ == g_:
            ex += 1
        if pn == gn:
            exns += 1
            if " " in g_:                      # spacing-sensitive subset
                sp_base += 1
                if " " in p_:
                    sp_ok += 1
        edit += lev(pn, gn)
        chars += len(gn)
    return {"n": n, "exact": ex / n, "exact_nospace": exns / n,
            "cer_nospace": edit / max(chars, 1),
            "space_recall": (sp_ok / sp_base) if sp_base else None,
            "space_base": sp_base}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for name, d in MODELS.items():
        if not (d / "inference.pdiparams").exists() and not (d / "inference.json").exists():
            sys.exit(f"[ABORT] {name} inference dir missing: {d}")

    rows = load_test()
    tmp = Path(tempfile.mkdtemp(prefix="ab_spacecat_"))
    print(f"[AB] test singles: {len(rows)}  |  building {args.pairs} pairs (seed {args.seed})")
    pair_items = build_pairs(rows, args.pairs, args.seed, tmp / "pairs")
    # real multi-word line crops (built by build_line_crops.py from TEST split)
    real_lines = []
    lt = V3DATA / "lines_test.txt"
    if lt.exists():
        for line in lt.open(encoding="utf-8"):
            if "\t" in line:
                p, t = line.rstrip("\n").split("\t", 1)
                real_lines.append((p, t.strip()))
    print(f"[AB] pairs built: {len(pair_items)}  real test lines: {len(real_lines)}  tmp={tmp}")

    items = ([{"key": f"s::{i}", "path": str(V3DATA / p)} for i, (p, _) in enumerate(rows)]
             + [{"key": f"p::{i}", "path": it["path"]} for i, it in enumerate(pair_items)]
             + [{"key": f"l::{i}", "path": str(V3DATA / p)} for i, (p, _) in enumerate(real_lines)])
    gt_single = [t for _, t in rows]
    gt_pair = [it["gt"] for it in pair_items]
    gt_line = [t for _, t in real_lines]

    table = []
    for name, mdir in MODELS.items():
        print(f"[AB] running {name} ...")
        preds = run_worker(mdir, items, tmp, name)
        ps = [preds.get(f"s::{i}", "") for i in range(len(rows))]
        pp = [preds.get(f"p::{i}", "") for i in range(len(pair_items))]
        pl = [preds.get(f"l::{i}", "") for i in range(len(real_lines))]
        ms, mp = metrics(gt_single, ps), metrics(gt_pair, pp)
        ml = metrics(gt_line, pl) if real_lines else None
        table.append((name, ms, mp, ml))
        for i in random.Random(0).sample(range(len(real_lines)), min(4, len(real_lines))):
            print(f"    line ex) GT='{gt_line[i]}'  pred='{pl[i]}'")

    print(f"\n{'model':14s} | {'single exact':>12s} {'CER':>6s} | {'pair exact':>10s} "
          f"{'sp-recall':>9s} | {'REAL-line exact':>15s} {'nospace':>8s} {'CER':>6s} {'sp-recall':>9s}")
    with RESULTS.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "single_exact", "single_cer", "pair_exact", "pair_space_recall",
                    "line_exact", "line_exact_nospace", "line_cer", "line_space_recall"])
        for name, ms, mp, ml in table:
            sr = "-" if mp["space_recall"] is None else f"{mp['space_recall']*100:.1f}%"
            lsr = "-" if not ml or ml["space_recall"] is None else f"{ml['space_recall']*100:.1f}%"
            lex = "-" if not ml else f"{ml['exact']*100:.1f}%"
            lns = "-" if not ml else f"{ml['exact_nospace']*100:.1f}%"
            lcer = "-" if not ml else f"{ml['cer_nospace']:.3f}"
            print(f"{name:14s} | {ms['exact']*100:11.1f}% {ms['cer_nospace']:6.3f} | "
                  f"{mp['exact']*100:9.1f}% {sr:>9s} | {lex:>15s} {lns:>8s} {lcer:>6s} {lsr:>9s}")
            w.writerow([name, f"{ms['exact']:.4f}", f"{ms['cer_nospace']:.4f}",
                        f"{mp['exact']:.4f}",
                        "" if mp["space_recall"] is None else f"{mp['space_recall']:.4f}",
                        "" if not ml else f"{ml['exact']:.4f}",
                        "" if not ml else f"{ml['exact_nospace']:.4f}",
                        "" if not ml else f"{ml['cer_nospace']:.4f}",
                        "" if not ml or ml["space_recall"] is None else f"{ml['space_recall']:.4f}"])
    print(f"[AB] -> {RESULTS}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
