#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""후보 제시형 VLM 교정(fixcand)용 — 라인별 3모델 후보 파일 생성.

run24 배포는 스트립별 v5·v4·zero-shot 3모델 투표인데 개별 모델 출력은 저장되지
않았습니다. 같은 검출 기하(pad 0.04 / y_tol 0.04, det_cache 재사용)로 스트립을
재생성해 모델별 라인 출력을 저장합니다. 스트립이 동일하므로 세 파일의 i번째
라인은 같은 스트립을 가리킵니다 (인덱스 정렬로 후보 구성 가능).

출력: artifacts/ocr_gt/ab_v4_spacecat/ocr_{region}_cand_{v5,v4,pre}.csv
검증: 3모델 다수결(동률 v5)이 run24 paddle과 일치하는지 exact로 확인.
"""
from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "artifacts" / "ocr_gt" / "ab_v4_spacecat"

import exp_line_tuning as X


def save(maps, tag):
    for region, m in maps.items():
        p = OUT / f"ocr_{region}_cand_{tag}.csv"
        with p.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, quoting=csv.QUOTE_ALL)
            w.writerow(["image_name", "gt_text"])
            for k in sorted(m):
                w.writerow([k, m[k].replace("\n", "\\n")])


def main() -> None:
    polys = X.build_det_cache()
    tmp = Path(tempfile.mkdtemp(prefix="cand_"))
    man, lay = X.make_strips(polys, 0.04, tmp, "cand", y_tol=0.04)
    print(f"[strips] {len(man)}")
    E = X.load_eval()
    gts = {r: E.load_csv_map(HERE / "artifacts" / "gt" / f"ocr_{r}_gt.csv")
           for r in X.REGIONS}
    votes = {}
    for tag, mdir in X.MODELS_KO.items():
        preds = X.recognize(man, tmp, f"c_{tag}", mdir)
        maps = X.assemble(lay, preds)
        votes[tag] = maps
        save(maps, tag)
        a = X.score(E, maps, gts)
        print(f"[{tag}] 단독 exact={a.exact_rate()*100:.1f}%")

    # 검증: 다수결(동률 v5) 재구성 vs run24
    import re, unicodedata

    def nk(s):
        return re.sub(r"[^0-9a-z가-힣]", "",
                      unicodedata.normalize("NFKC", s or "").lower())

    merged = {}
    for region in X.REGIONS:
        m = {}
        for name in lay[region]:
            out_lines = []
            n = lay[region][name]
            per = {t: (votes[t][region].get(name, "") or "").split("\n") for t in votes}
            for li in range(n):
                cands = [per[t][li] if li < len(per[t]) else "" for t in ("v5", "v4", "pre")]
                norms = [nk(c) for c in cands]
                pick = cands[0]
                for c, nn in zip(cands, norms):
                    if nn and norms.count(nn) >= 2:
                        pick = c
                        break
                if pick.strip():
                    out_lines.append(pick.strip())
            m[name] = "\n".join(out_lines)
        merged[region] = m
    a = X.score(E, merged, gts)
    print(f"[재구성 다수결] exact={a.exact_rate()*100:.1f}%  (run24 배포=69.9% 근접해야 정상)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
