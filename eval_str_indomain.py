#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""In-domain(signboard_v3 test) 인식기 채점 — 워커 IO 계약을 쓰는 모든 엔진 공통.

test_line(실라인 1,631) / test_word(단어 7,756) 목록을 manifest 로 만들어 워커를 돌리고,
eval_ocr_v2 와 같은 정규화(NFKC → lower → [0-9a-z가-힣 공백]만 유지)로
  exact(공백 제거 후 완전일치) / CER(문자 편집거리, 공백 제거) / WER(단어 편집거리) / WAR(단어 LCS)
를 계산합니다. 라인 매칭이 없는 1:1 크롭 채점이라 FP 삽입 항은 없습니다.

Usage:
  .venv/Scripts/python.exe eval_str_indomain.py --engine tesseract --worker tesseract_rec_worker.py [--worker-py ...] [--worker-args "..."] [--sets test_line,test_word] [--limit N]
출력: artifacts/str_baselines/indomain/<engine>_<set>.jsonl (예측) + artifacts/str_baselines/indomain_summary.csv (append)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
LISTS = HERE / "artifacts" / "str_baselines" / "lists"
OUT = HERE / "artifacts" / "str_baselines" / "indomain"
PY = HERE / ".venv" / "Scripts" / "python.exe"
_STRICT = re.compile(r"[^0-9a-z가-힣\s]")


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").replace("​", "").lower()
    return _STRICT.sub("", s)


def lev(a, b) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1)); cur = [0] * (m + 1)
    for i in range(1, n + 1):
        cur[0] = i; ai = a[i - 1]
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if ai == b[j - 1] else 1))
        prev, cur = cur, prev
    return prev[m]


def lcs(a, b) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def score(pairs):
    n = ex = ed = ch = wed = wl = gw = 0
    for pred, gt in pairs:
        g, p = norm(gt), norm(pred)
        gn, pn = re.sub(r"\s+", "", g), re.sub(r"\s+", "", p)
        n += 1; ex += int(gn == pn and gn != "")
        ed += lev(pn, gn); ch += len(gn)
        gwds, pwds = g.split(), p.split()
        wed += lev(pwds, gwds); wl += lcs(pwds, gwds); gw += len(gwds)
    return {"n": n, "exact": ex / n, "CER": ed / ch, "WER": wed / gw, "WAR": wl / gw}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--worker", required=True)
    ap.add_argument("--worker-py", default=str(PY))
    ap.add_argument("--worker-args", default="")
    ap.add_argument("--sets", default="test_line,test_word")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--region-key", default="gangnam", help="manifest key 의 region 자리(Tesseract 언어 선택용)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    rows = []
    for name in args.sets.split(","):
        items = []
        for ln in (LISTS / f"{name}.txt").read_text(encoding="utf-8").splitlines()[: args.limit]:
            p, t = ln.split("\t", 1)
            items.append({"key": f"{args.region_key}::{Path(p).stem}::0", "path": p, "gt": t})
        man = OUT / f"{args.engine}_{name}_manifest.jsonl"
        out = OUT / f"{args.engine}_{name}.jsonl"
        with man.open("w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps({"key": it["key"], "path": it["path"]}, ensure_ascii=False) + "\n")
        cmd = [args.worker_py, str(HERE / args.worker), "--manifest", str(man), "--out", str(out)] + shlex.split(args.worker_args)
        print(f"[{args.engine}/{name}] {len(items)} crops → {Path(args.worker).name}", flush=True)
        subprocess.run(cmd, check=True, env=env, cwd=str(HERE))
        pred = {}
        with out.open(encoding="utf-8") as f:
            for line in f:
                d = json.loads(line); pred[d["key"]] = d["text"]
        sc = score([(pred.get(it["key"], ""), it["gt"]) for it in items])
        print(f"[{args.engine}/{name}] " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in sc.items()), flush=True)
        rows.append([args.engine, name, sc["n"], f"{sc['exact']:.4f}", f"{sc['CER']:.4f}", f"{sc['WER']:.4f}", f"{sc['WAR']:.4f}"])
    summ = OUT.parent / "indomain_summary.csv"
    new = not summ.exists()
    with summ.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["engine", "set", "n", "exact", "CER", "WER", "WAR"])
        w.writerows(rows)
    print(f"[saved] {summ}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
