#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VLM 로 허위 탐지를 되거를 수 있는가 — 표본 추정.

값싼 신호(conf·글자수·RAG)는 TP/FP 분리력이 약했습니다. 파이프라인은 이미 크롭당
VLM 을 호출하고 있으므로, 같은 호출에 "이게 상점 간판인가?" 판정을 얹을 수 있는지
표본으로 먼저 확인합니다. 전수(761크롭)는 63분이라 층화 표본으로 추정합니다.

출력: 판정의 TP 유지율(간판을 간판이라 함) / FP 제거율(허위를 허위라 함).
"""
from __future__ import annotations
import csv, json, random, sys, time, base64, urllib.request
from pathlib import Path
import eval_e2e_cascade as C

HERE = Path(__file__).resolve().parents[1]
GT_DIR = HERE / "artifacts" / "gt"
OCR_DIR = HERE / "artifacts" / "ocr_gt"
OLLAMA = "http://localhost:11434/api/generate"
REGIONS = ("gangnam", "brooklyn", "suwon")

PROMPT = ("이 사진이 **상점·업소의 간판**인지 판정하라.\n"
          "- 간판이면 yes. 상호·업종이 적힌 판/현수막/유리창 시트 모두 포함한다.\n"
          "- 건물 외벽·창문·도로·차량·하늘·사람처럼 간판이 아닌 것이면 no.\n"
          "- 글자가 일부 잘렸어도 간판이면 yes 다.\n"
          '- JSON 한 줄만: {"signboard": "yes"} 또는 {"signboard": "no"}')


def ask(model: str, img: Path) -> str:
    body = {"model": model, "prompt": PROMPT, "stream": False,
            "images": [base64.b64encode(img.read_bytes()).decode()],
            "options": {"temperature": 0}}
    req = urllib.request.Request(OLLAMA, data=json.dumps(body).encode())
    with urllib.request.urlopen(req, timeout=240) as r:
        return json.loads(r.read())["response"]


def main() -> None:
    tag, run, n_sample = "_union", "33", int(sys.argv[1]) if len(sys.argv) > 1 else 200
    model = "gemma3:12b"
    crop_root = HERE / "artifacts" / "gt" / f"crop_det{tag}"

    pool = []
    for region in REGIONS:
        pred = C.load_map(OCR_DIR / f"ocr_{region}_{run}_paddle.csv")
        for m in C.load_match(region, tag):
            if m["status"] == "FN":
                continue
            p = crop_root / region / f"{m['det_crop']}.jpg"
            if p.exists():
                pool.append((region, m["status"], p,
                             sum(len(C.nk(l)) for l in C.split_lines(
                                 pred.get(m["det_crop"], "")))))
    tp = [x for x in pool if x[1] == "TP"]
    fp = [x for x in pool if x[1] == "FP"]
    rnd = random.Random(0)
    half = n_sample // 2
    sample = rnd.sample(tp, min(half, len(tp))) + rnd.sample(fp, min(half, len(fp)))
    print(f"[표본] TP {sum(1 for s in sample if s[1]=='TP')} / "
          f"FP {sum(1 for s in sample if s[1]=='FP')} (전체 TP {len(tp)} FP {len(fp)})")

    res, t0 = [], time.time()
    for i, (region, status, path, chars) in enumerate(sample, 1):
        try:
            r = ask(model, path)
            yes = '"yes"' in r.lower() or "'yes'" in r.lower()
        except Exception:
            yes = True          # 실패 시 보수적으로 유지
        res.append((status, yes, chars))
        if i % 25 == 0:
            el = time.time() - t0
            print(f"  {i}/{len(sample)}  {el/i:.1f}s/건  "
                  f"ETA {(len(sample)-i)*el/i/60:.0f}분", flush=True)

    def rate(st, keep_rule):
        sub = [r for r in res if r[0] == st]
        k = sum(1 for r in sub if keep_rule(r))
        return k, len(sub)

    print()
    for label, rule in (("VLM 판정만", lambda r: r[1]),
                        ("VLM 판정 AND 글자수>=3", lambda r: r[1] and r[2] >= 3),
                        ("VLM 판정 OR 글자수>=5", lambda r: r[1] or r[2] >= 5)):
        ktp, ntp = rate("TP", rule); kfp, nfp = rate("FP", rule)
        print(f"{label:26s} TP 유지 {ktp}/{ntp} ({ktp/max(ntp,1)*100:5.1f}%)   "
              f"FP 유지 {kfp}/{nfp} ({kfp/max(nfp,1)*100:5.1f}%)  "
              f"→ FP 제거율 {100-kfp/max(nfp,1)*100:5.1f}%")
    Path("artifacts/gt/fp_vlm_sample.json").write_text(
        json.dumps([[a, b, c] for a, b, c in res], ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
