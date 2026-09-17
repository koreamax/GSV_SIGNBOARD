#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RAG 검색 모듈 — 지역 POI 사전에서 OCR 라인과 닮은 상호 후보를 찾아옵니다.

T5(사전 스냅)와 결정적으로 다른 점: **여기서는 아무것도 치환하지 않습니다.**
검색 결과는 VLM 프롬프트에 "참고 후보"로만 들어가고, 최종 판단은 이미지를 보는
VLM이 합니다. D16에서 규명된 손상 모드(SPRINT→SPRING처럼 실단어를 사전 이웃으로
바꿔치기)는 편집거리 규칙으로 구분이 불가능했는데, 이미지를 보는 심판이 있으면
그 결정이 문자열 문제가 아니라 시각 확인 문제가 되기 때문입니다.

검색 대상은 `artifacts/ocr_db/rag_{region}.csv` (build_poi_db.py --merge 산출):
  name,key,tag,source,lat,lon   — source ∈ {osm, kr_sangga, nyc_lob, vocab}

Usage:
  # 단건 질의 (동작 확인)
  .venv/Scripts/python.exe rag_retrieve.py --region gangnam --query 스타벅스
  # 사전 커버리지 진단 (RAG 상한 측정 — GT는 진단에만 쓰고 사전에 넣지 않음)
  .venv/Scripts/python.exe rag_retrieve.py --coverage
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
DB_DIR = HERE / "artifacts" / "ocr_db"
GT_DIR = HERE / "artifacts" / "gt"
REGIONS = ("gangnam", "brooklyn", "suwon")

# 상호명 사전이 목적이므로 일반 어휘(L2)는 기본 제외합니다. CAFE·치킨 같은
# 보통명사는 VLM이 이미 알고 있어 후보로 넣어봐야 프롬프트만 길어집니다.
POI_SOURCES = ("osm", "kr_sangga", "nyc_lob")


def norm_key(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    return re.sub(r"[^0-9a-z가-힣]", "", s)


def lev(a: str, b: str) -> int:
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return max(n, m)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (0 if ai == b[j - 1] else 1))
        prev = cur
    return prev[m]


def bigrams(k: str) -> set[str]:
    return {k[i:i + 2] for i in range(len(k) - 1)} or {k}


class Retriever:
    """지역 POI 사전 하나에 대한 검색기.

    2단계입니다: ① 문자 bigram 역색인으로 후보를 수백 개로 줄이고
    ② 정규화 편집거리로 재순위. 사전이 수만 건이어도 라인당 1ms 수준입니다."""

    def __init__(self, region: str, sources: tuple[str, ...] = POI_SOURCES,
                 db_dir: Path = DB_DIR):
        self.region = region
        self.entries: list[dict] = []
        self.by_key: dict[str, list[int]] = defaultdict(list)
        self.index: dict[str, set[int]] = defaultdict(set)

        path = db_dir / f"rag_{region}.csv"
        if not path.exists():          # 아직 안 만들었으면 OSM 레이어만으로 동작
            path = db_dir / f"osm_{region}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} 없음 — build_poi_db.py --merge 를 먼저 실행하세요.")

        for row in csv.DictReader(path.open(encoding="utf-8")):
            src = row.get("source", "osm")
            if sources and src not in sources:
                continue
            key = row.get("key") or norm_key(row.get("name", ""))
            if len(key) < 2:
                continue
            i = len(self.entries)
            self.entries.append({"name": row.get("name", ""), "key": key,
                                 "tag": row.get("tag", ""), "source": src})
            self.by_key[key].append(i)
            for g in bigrams(key):
                self.index[g].add(i)

    def __len__(self) -> int:
        return len(self.entries)

    def retrieve(self, query: str, k: int = 5, min_sim: float = 0.55,
                 pool: int = 200) -> list[dict]:
        """query와 닮은 POI 상위 k건. 각 항목에 sim(0~1)과 exact 여부를 붙여 반환."""
        q = norm_key(query)
        if len(q) < 2:
            return []

        hits: list[dict] = []
        seen_keys: set[str] = set()

        for i in self.by_key.get(q, []):          # ① 정확 일치 우선
            e = self.entries[i]
            if e["key"] in seen_keys and e["tag"] == "":
                continue
            hits.append({**e, "sim": 1.0, "exact": True})
            seen_keys.add(e["key"])
            if len(hits) >= k:
                return hits

        qg = bigrams(q)                            # ② bigram 후보 축소
        score: dict[int, int] = defaultdict(int)
        for g in qg:
            for i in self.index.get(g, ()):
                score[i] += 1
        if not score:
            return hits
        cand = sorted(score, key=lambda i: -score[i])[:pool]

        scored = []
        for i in cand:
            e = self.entries[i]
            if e["key"] == q or e["key"] in seen_keys:
                continue
            L = max(len(q), len(e["key"]))
            if abs(len(q) - len(e["key"])) > L * 0.5:
                continue
            sim = 1.0 - lev(q, e["key"]) / L
            if sim >= min_sim:
                scored.append({**e, "sim": sim, "exact": False})
        # 같은 표기가 여러 소스에 있으면 태그를 가진 쪽을 남깁니다.
        scored.sort(key=lambda d: (-d["sim"], not d["tag"], len(d["name"])))
        for d in scored:
            if d["key"] in seen_keys:
                continue
            seen_keys.add(d["key"])
            hits.append(d)
            if len(hits) >= k:
                break
        return hits

    def hint_block(self, queries: list[str], k: int = 5, per_line: int = 3,
                   allowed_tags: set[str] | None = None,
                   show_tags: bool = True) -> str:
        """여러 라인을 한 번에 검색해 프롬프트에 넣을 힌트 블록 문자열을 만듭니다.

        `allowed_tags`를 주면 그 밖의 태그는 **표시하지 않습니다**(상호명만 남김).
        힌트가 정답 어휘 밖의 답을 제안하면 모델이 채점 불가능한 값을 내놓기
        때문입니다 — 브루클린에서 실측된 문제로, OSM 원본 태그 체계가 평가 어휘(42종)
        보다 세분화돼 있어 후보 태그의 24.1%가 어휘 밖이었고 StateFarm→office=insurance
        (정답 office=company), CLEANERS→shop=dry_cleaning(정답 shop=laundry) 식으로
        빗나갔습니다. 상호명 자체는 유용하므로 태그만 떼고 남깁니다.

        `show_tags=False`면 태그를 전부 생략합니다 (OCR 교정처럼 업종이 무의미할 때).

        빈 문자열이면 호출부에서 힌트 블록 자체를 생략하면 됩니다."""
        seen, out = set(), []
        for q in queries:
            for h in self.retrieve(q, k=per_line):
                if h["key"] in seen:
                    continue
                seen.add(h["key"])
                out.append(h)
        out.sort(key=lambda d: -d["sim"])
        out = out[:k]
        if not out:
            return ""
        lines = []
        for h in out:
            tag = h["tag"]
            if not show_tags or (allowed_tags is not None and tag not in allowed_tags):
                tag = ""
            lines.append(f"- {h['name']}" + (f" ({tag})" if tag else ""))
        return "\n".join(lines)


_CACHE: dict[tuple[str, tuple[str, ...]], Retriever] = {}


def get(region: str, sources: tuple[str, ...] = POI_SOURCES) -> Retriever:
    """지역별 검색기 캐시 — 평가 루프에서 매 크롭마다 재로딩하지 않게."""
    ck = (region, sources)
    if ck not in _CACHE:
        _CACHE[ck] = Retriever(region, sources)
    return _CACHE[ck]


# ---------------------------------------------------------------- diagnostics
def gt_lines(region: str) -> list[str]:
    """평가 GT 라인 (진단 전용 — 사전 구축에는 절대 쓰지 않습니다, DB_RULES 1절)."""
    out = []
    p = GT_DIR / f"ocr_{region}_gt.csv"
    for row in csv.DictReader(p.open(encoding="utf-8")):
        txt = str(row.get("gt_text") or "").replace(chr(92) + "n", "\n")
        for line in txt.split("\n"):
            if "###" in line:
                continue
            if len(norm_key(line)) >= 2:
                out.append(line)
    return out


def coverage() -> None:
    """RAG가 회수할 수 있는 라인의 상한. 사전에 없는 상호는 RAG로 못 고칩니다."""
    print(f"{'region':9s} {'GT':>5s} {'POI exact':>11s} {'POI top5':>10s} "
          f"{'+vocab exact':>13s} {'사전크기':>9s}")
    for region in REGIONS:
        poi = get(region, POI_SOURCES)
        allsrc = get(region, ())          # vocab 포함 전체
        lines = gt_lines(region)
        n = len(lines) or 1
        ex = sum(1 for l in lines if poi.by_key.get(norm_key(l)))
        top = sum(1 for l in lines if poi.retrieve(l, k=5))
        exa = sum(1 for l in lines if allsrc.by_key.get(norm_key(l)))
        print(f"{region:9s} {len(lines):5d} {ex/n:10.1%} {top/n:9.1%} "
              f"{exa/n:12.1%} {len(poi):9d}")
    print("\n※ POI exact = 상호명 사전에 정확히 존재 (RAG 상한의 핵심 지표)")
    print("  POI top5   = 유사 후보라도 검색됨 (VLM이 골라낼 기회가 있는 라인)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", choices=REGIONS)
    ap.add_argument("--query")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--all-sources", action="store_true",
                    help="일반 어휘(vocab)까지 포함해 검색")
    ap.add_argument("--coverage", action="store_true", help="사전 커버리지 진단")
    args = ap.parse_args()

    if args.coverage:
        coverage()
        return
    if not (args.region and args.query):
        ap.error("--region 과 --query 를 함께 주거나 --coverage 를 쓰세요.")

    r = get(args.region, () if args.all_sources else POI_SOURCES)
    print(f"[{args.region}] 사전 {len(r)}건")
    for h in r.retrieve(args.query, k=args.k):
        mark = "=" if h["exact"] else "~"
        print(f"  {mark} {h['sim']:.2f}  {h['name']:<30s} {h['tag']:<22s} {h['source']}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
