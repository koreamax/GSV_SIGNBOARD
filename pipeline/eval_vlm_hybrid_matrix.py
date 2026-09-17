#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VLM×OCR 하이브리드 — 모델 × 모드 전체 채점표 (4.20).

exp_vlm_ocr.py 가 모델별(--out-suffix _<tag>)로 저장한 출력을 같은 프로토콜
(eval_ocr_v2, --mask-phone, brooklyn en-only)로 한꺼번에 채점하고, VLM 호출 없이
계산되는 **파생 방식**을 함께 평가합니다.

  solo       : VLM 단독 판독
  fix        : 배포 OCR 라인 + 이미지 → 오독 글자만 교정
  fixcand    : 3모델 후보(v5·v4·zero-shot) 그라운딩 교정           (D28 방식)
  fixcand5   : 5모델 후보(+SVTRv2·PARSeq) 그라운딩 교정            (신규 — 오류가 서로 다른 인식기 추가)
  compose    : fixcand + solo 의 누락 라인 병합                     (D28 배포 방식)
  compose5   : fixcand5 + solo 병합                                 (신규)
  guard5     : fixcand5 출력 중 **후보 집합에서 1글자 이내**인 라인만 채택, 아니면 배포 라인 유지
               (VLM 이 멀쩡한 라인을 다시 쓰는 환각 피해 차단)       (신규, 오프라인)
  consensus  : 두 VLM 의 fixcand5 가 **같은 교정**을 냈을 때만 채택, 아니면 배포 라인 유지
               → 그 위에 두 VLM solo 의 공통 누락 라인만 병합          (신규, 오프라인)

Usage:
  .venv/Scripts/python.exe pipeline/eval_vlm_hybrid_matrix.py --models qwen3vl32b,gemma4_31b [--csv out.csv]
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE / d) for d in ("pipeline", "ocr", "vlm", "str_baselines")]

GT = HERE / "artifacts" / "gt"
OCR_DIR = HERE / "artifacts" / "ocr_gt"
AB = OCR_DIR / "ab_v4_spacecat"
REGIONS = ["gangnam", "brooklyn", "suwon"]
CAND5 = ("v5", "v4", "pre", "svtrv2", "parseq")
FIELDS = ("edit", "fp_edit", "chars", "word_lcs", "gt_words", "exact",
          "recalled", "lines", "contain", "empty_pred", "crops")


def nk(s: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]", "", unicodedata.normalize("NFKC", s or "").lower())


def load_E():
    saved = sys.argv
    sys.argv = ["pipeline/eval_ocr_v2.py", "--mask-phone", "--en-only-regions", "brooklyn"]
    import eval_ocr_v2 as E
    sys.argv = saved
    return E


def lines_of(E, m: dict, k: str) -> list[str]:
    v = m.get(k, "")
    return E.split_lines(v) if v.strip() else []


def similar(E, a: str, b: str) -> bool:
    A, B = nk(a), nk(b)
    if not A or not B:
        return False
    if A in B or B in A:
        return True
    if abs(len(A) - len(B)) > max(len(A), len(B)) * 0.5:
        return False
    return E.lev(A, B) <= max(1, int(min(len(A), len(B)) * 0.3))


def merge_missing(E, base: list[str], extra_src: list[str]) -> list[str]:
    extra = [s for s in extra_src if len(nk(s)) >= 2 and not any(similar(E, s, f) for f in base)]
    return base + extra


def score(E, preds_by_region: dict[str, dict]) -> dict:
    tot = None
    per = {}
    for region in REGIONS:
        gt = E.load_csv_map(GT / f"ocr_{region}_gt.csv")
        E.EN_ONLY = region in E.EN_ONLY_REGIONS
        a = E.eval_engine(gt, preds_by_region[region], [])
        E.EN_ONLY = False
        per[region] = a.exact_rate() * 100
        if tot is None:
            tot = a
        else:
            for f_ in FIELDS:
                setattr(tot, f_, getattr(tot, f_) + getattr(a, f_))
    return {"exact": tot.exact_rate() * 100, "CER": tot.cer(), "WAR": tot.war(),
            "contain": tot.contain_rate() * 100, **{f"ex_{r}": per[r] for r in REGIONS}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="출력 접미사 태그들, 쉼표 구분 (예: qwen3vl32b,gemma4_31b). 'gemma3' = 접미사 없음(D28 원본)")
    ap.add_argument("--deploy-run", default="24")
    ap.add_argument("--csv", default=str(HERE / "artifacts" / "gt" / "vlm_hybrid_matrix.csv"))
    args = ap.parse_args()
    E = load_E()
    tags = [t for t in args.models.split(",") if t]

    def load(name: str, tag: str) -> dict[str, dict] | None:
        sfx = "" if tag == "gemma3" else f"_{tag}"
        out = {}
        for region in REGIONS:
            p = AB / f"ocr_{region}_vlm{name}{sfx}.csv"
            if not p.exists():
                return None
            out[region] = E.load_csv_map(p)
        return out

    deploy = {r: E.load_csv_map(OCR_DIR / f"ocr_{r}_{args.deploy_run}_paddle.csv") for r in REGIONS}
    cands = {r: {t: E.load_csv_map(AB / f"ocr_{r}_cand_{t}.csv") for t in CAND5
                 if (AB / f"ocr_{r}_cand_{t}.csv").exists()} for r in REGIONS}
    gts = {r: E.load_csv_map(GT / f"ocr_{r}_gt.csv") for r in REGIONS}
    rows = [("deploy OCR (run%s)" % args.deploy_run, "-", score(E, deploy))]

    store: dict[tuple[str, str], dict] = {}
    for tag in tags:
        for name in ("solo", "fix", "fixcand", "fixcand5"):
            m = load(name, tag)
            if m is None:
                continue
            store[(tag, name)] = m
            rows.append((name, tag, score(E, m)))
        # compose / compose5
        for base_name, out_name in (("fixcand", "compose"), ("fixcand5", "compose5"), ("fix", "compose_fix")):
            if (tag, base_name) in store and (tag, "solo") in store:
                merged = {}
                for r in REGIONS:
                    merged[r] = {k: "\n".join(merge_missing(E, lines_of(E, store[(tag, base_name)][r], k),
                                                           lines_of(E, store[(tag, "solo")][r], k)))
                                 for k in gts[r]}
                store[(tag, out_name)] = merged
                rows.append((out_name, tag, score(E, merged)))
        # guard5: VLM 라인이 후보 집합(배포 포함)에서 1글자 이내일 때만 채택
        if (tag, "fixcand5") in store:
            guarded = {}
            for r in REGIONS:
                g = {}
                for k in gts[r]:
                    base = lines_of(E, deploy[r], k)
                    vl = lines_of(E, store[(tag, "fixcand5")][r], k)
                    if len(vl) != len(base):
                        g[k] = "\n".join(base); continue
                    pool = [nk(x) for t in cands[r].values() for x in lines_of(E, t, k)] + [nk(b) for b in base]
                    out = []
                    for b, v in zip(base, vl):
                        V = nk(v)
                        ok = bool(V) and any(E.lev(V, c) <= 1 for c in pool if c)
                        out.append(v if ok else b)
                    g[k] = "\n".join(out)
                guarded[r] = g
            store[(tag, "guard5")] = guarded
            rows.append(("guard5", tag, score(E, guarded)))
            if (tag, "solo") in store:
                merged = {r: {k: "\n".join(merge_missing(E, lines_of(E, guarded[r], k), lines_of(E, store[(tag, "solo")][r], k)))
                              for k in gts[r]} for r in REGIONS}
                rows.append(("guard5+solo", tag, score(E, merged)))

    # consensus: 두 VLM 이 같은 교정을 냈을 때만 채택
    if len(tags) >= 2 and all((t, "fixcand5") in store for t in tags[:2]):
        a, b = tags[0], tags[1]
        cons, cons_solo = {}, {}
        for r in REGIONS:
            c, cs = {}, {}
            for k in gts[r]:
                base = lines_of(E, deploy[r], k)
                la, lb = lines_of(E, store[(a, "fixcand5")][r], k), lines_of(E, store[(b, "fixcand5")][r], k)
                if len(la) == len(base) == len(lb):
                    out = [x if nk(x) == nk(y) and nk(x) else z for x, y, z in zip(la, lb, base)]
                else:
                    out = base
                c[k] = "\n".join(out)
                if (a, "solo") in store and (b, "solo") in store:
                    sa, sb = lines_of(E, store[(a, "solo")][r], k), lines_of(E, store[(b, "solo")][r], k)
                    both = [x for x in sa if any(similar(E, x, y) for y in sb)]
                    cs[k] = "\n".join(merge_missing(E, out, both))
            cons[r], cons_solo[r] = c, cs
        rows.append(("consensus(fixcand5)", f"{a}&{b}", score(E, cons)))
        if all(cons_solo[r] for r in REGIONS):
            rows.append(("consensus+solo∩", f"{a}&{b}", score(E, cons_solo)))

    print(f"\n{'mode':<22}{'model':<26}{'exact':>7}{'CER':>8}{'WAR':>8}{'contain':>9}   gangnam/brooklyn/suwon")
    for name, tag, m in rows:
        print(f"{name:<22}{tag:<26}{m['exact']:>6.1f}%{m['CER']:>8.3f}{m['WAR']:>8.3f}{m['contain']:>8.1f}%   "
              + " / ".join(f"{m['ex_'+r]:.1f}" for r in REGIONS))
    with open(args.csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "model", "exact", "CER", "WAR", "contain"] + [f"exact_{r}" for r in REGIONS])
        for name, tag, m in rows:
            w.writerow([name, tag, f"{m['exact']:.2f}", f"{m['CER']:.4f}", f"{m['WAR']:.4f}", f"{m['contain']:.2f}"]
                       + [f"{m['ex_'+r]:.2f}" for r in REGIONS])
    print(f"\nsaved {args.csv}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
