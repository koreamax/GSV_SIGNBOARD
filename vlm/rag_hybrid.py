#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""하이브리드 검색(lexical + dense) 실험과 검색 품질 오프라인 평가.

**왜 오프라인 평가부터인가:** 종단 태깅 실행은 26분/회입니다. 검색기를 바꿔도
VLM이 그 후보를 쓰지 않으면 지표가 안 움직이므로, 먼저 "정답 업소가 힌트 목록에
들어오는가"만 따로 재서 검색기끼리 비교합니다. 여기서 이기지 못하면 종단 실행을
돌릴 이유가 없습니다.

세 가지 검색기를 같은 프로토콜로 비교합니다:
  lexical : 현행 (문자 bigram 역색인 → 편집거리 재순위)  — rag_retrieve.Retriever
  dense   : bge-m3 임베딩 코사인 유사도 (다국어, 한글·영문 혼용 대응)
  hybrid  : 두 순위를 RRF(Reciprocal Rank Fusion)로 융합

평가 프로토콜 (실사용과 동일한 조건):
  질의 = **배포 OCR이 읽은 라인**(오독 포함, 정답 아님)
  정답 = 그 크롭의 GT 라인이 사전에 실제로 존재하는 항목
  지표 = hit@k — 크롭 단위로 상위 k개 힌트 안에 정답 항목이 들어왔는가
         (hint_block 이 라인별 검색 결과를 합쳐 상위 k만 넘기므로 그 동작 그대로)

Usage:
  .venv/Scripts/python.exe rag_hybrid.py --build          # 코퍼스 임베딩(1회, 캐시)
  .venv/Scripts/python.exe rag_hybrid.py --eval           # 세 검색기 비교
  .venv/Scripts/python.exe rag_hybrid.py --eval --region suwon
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

import rag_retrieve as RAG

HERE = Path(__file__).resolve().parents[1]
DB_DIR = HERE / "artifacts" / "ocr_db"
GT_DIR = HERE / "artifacts" / "gt"
OCR_DIR = HERE / "artifacts" / "ocr_gt"
OLLAMA_EMBED = "http://localhost:11434/api/embed"
EMB_MODEL = "bge-m3"
DEPLOY_RUN = "24"
REGIONS = ("gangnam", "brooklyn", "suwon")
# RRF 상수. 표준값은 60이지만 오프라인 스윕에서 10이 근소하게 나았습니다
# (가중평균 hit@5 87.6→88.7%). 다만 평가 표본이 97크롭이라 이 차이는 강남에서
# 크롭 1개가 움직인 수준입니다 — **유의미한 개선으로 인용하지 마세요.**
# 비용이 0이라 채택했을 뿐입니다.
RRF_K = 10
# 융합 전 각 검색기에서 뽑는 후보 수. 프롬프트에 나가는 개수(k)와 무관한 내부값이라
# 키워도 VLM 비용은 그대로입니다. 15→30에서 이득, 60 이상은 잡음이 늘어 손해였습니다.
POOL_MIN = 30


def _embed_chunk(chunk: list[str], model: str) -> np.ndarray:
    """한 배치를 임베딩. Ollama가 배치를 거부하면(400) 반씩 줄여 재시도합니다 —
    /api/embed 는 요청당 입력 개수 상한이 있어 256은 거부되고 128은 통과합니다."""
    body = {"model": model, "input": chunk}
    req = urllib.request.Request(OLLAMA_EMBED, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return np.asarray(json.loads(r.read())["embeddings"], dtype=np.float32)
    except urllib.error.HTTPError as exc:
        if exc.code != 400 or len(chunk) == 1:
            raise
        h = len(chunk) // 2
        return np.concatenate([_embed_chunk(chunk[:h], model),
                               _embed_chunk(chunk[h:], model)], axis=0)


def embed(texts: list[str], model: str = EMB_MODEL, batch: int = 128) -> np.ndarray:
    """Ollama 임베딩 API. L2 정규화해서 반환 → 코사인 유사도 = 내적."""
    out = []
    for i in range(0, len(texts), batch):
        out.append(_embed_chunk(texts[i:i + batch], model))
        if i and i % (batch * 20) == 0:
            print(f"    {i}/{len(texts)}", flush=True)
    m = np.concatenate(out, axis=0)
    m /= (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
    return m


def emb_path(region: str) -> Path:
    return DB_DIR / f"emb_{region}.npy"


def build(regions=REGIONS) -> None:
    """지역별 코퍼스 임베딩을 만들어 캐시합니다 (엔트리 순서 = Retriever.entries)."""
    for region in regions:
        r = RAG.get(region)
        names = [e["name"] for e in r.entries]
        print(f"[{region}] {len(names)}건 임베딩 중...", flush=True)
        t0 = time.time()
        m = embed(names)
        np.save(emb_path(region), m)
        print(f"[{region}] 저장 {emb_path(region).name}  shape={m.shape}  "
              f"{time.time()-t0:.0f}초")


class Hybrid:
    """lexical/dense/hybrid 세 방식으로 같은 코퍼스를 검색합니다."""

    def __init__(self, region: str):
        self.region = region
        self.lex = RAG.get(region)
        p = emb_path(region)
        self.emb = np.load(p) if p.exists() else None

    def dense_rank(self, query: str, k: int) -> list[int]:
        if self.emb is None:
            return []
        q = embed([query])[0]
        sims = self.emb @ q
        idx = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
        return [int(i) for i in idx[np.argsort(-sims[idx])]]

    def dense_rank_batch(self, queries: list[str], k: int) -> list[list[int]]:
        """질의를 모아 한 번에 임베딩 — 라인마다 API를 때리면 너무 느립니다."""
        if self.emb is None or not queries:
            return [[] for _ in queries]
        Q = embed(queries)
        S = self.emb @ Q.T                      # (N, Q)
        out = []
        for j in range(S.shape[1]):
            s = S[:, j]
            idx = np.argpartition(-s, min(k, len(s) - 1))[:k]
            out.append([int(i) for i in idx[np.argsort(-s[idx])]])
        return out

    def lex_rank(self, query: str, k: int) -> list[int]:
        hits = self.lex.retrieve(query, k=k)
        keys = {h["key"]: None for h in hits}
        out = []
        for key in keys:
            out.extend(self.lex.by_key.get(key, [])[:1])
        return out[:k]

    def hint_block(self, queries: list[str], k: int = 5,
                   allowed_tags: set[str] | None = None,
                   show_tags: bool = True) -> str:
        """rag_retrieve.Retriever.hint_block 과 같은 계약, 융합 순위만 다릅니다.

        라인별로 lexical·dense 순위를 각각 뽑아 RRF로 합치고 상위 k개를 냅니다.
        어휘 밖 태그를 떼는 규칙(4.15 ④)은 그대로 유지합니다."""
        queries = [q for q in queries if len(RAG.norm_key(q)) >= 2]
        if not queries:
            return ""
        pool = max(k * 3, POOL_MIN)
        dens = self.dense_rank_batch(queries, pool) if self.emb is not None else \
            [[] for _ in queries]
        rankings = []
        for q, d in zip(queries, dens):
            rankings.append(self.lex_rank(q, pool))
            if d:
                rankings.append(d)
        ranked = self.rrf(rankings, k * 3)

        lines, seen = [], set()
        for i in ranked:
            e = self.lex.entries[i]
            if e["key"] in seen:
                continue
            seen.add(e["key"])
            tag = e["tag"]
            if not show_tags or (allowed_tags is not None and tag not in allowed_tags):
                tag = ""
            lines.append(f"- {e['name']}" + (f" ({tag})" if tag else ""))
            if len(lines) >= k:
                break
        return "\n".join(lines)

    @staticmethod
    def rrf(rankings: list[list[int]], k: int) -> list[int]:
        score: dict[int, float] = {}
        for ranking in rankings:
            for rank, i in enumerate(ranking):
                score[i] = score.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
        return sorted(score, key=lambda i: -score[i])[:k]


_HCACHE: dict[str, Hybrid] = {}


def get(region: str) -> Hybrid:
    """지역별 하이브리드 검색기 캐시 — 임베딩 행렬(수십 MB) 재로딩 방지."""
    if region not in _HCACHE:
        _HCACHE[region] = Hybrid(region)
    return _HCACHE[region]


# ------------------------------------------------------------------ 평가
def load_pairs(region: str) -> list[tuple[list[str], set[str]]]:
    """크롭별 (OCR 라인들, 사전에 존재하는 정답 키들)."""
    gt: dict[str, str] = {}
    for row in csv.DictReader((GT_DIR / f"ocr_{region}_gt.csv").open(encoding="utf-8")):
        gt[row["image_name"]] = row["gt_text"] or ""
    ocr: dict[str, str] = {}
    p = OCR_DIR / f"ocr_{region}_{DEPLOY_RUN}_paddle.csv"
    for row in csv.DictReader(p.open(encoding="utf-8-sig")):
        ocr[row["image_name"]] = row["gt_text"] or ""

    lex = RAG.get(region)
    pairs = []
    for name, g in gt.items():
        gold = set()
        for line in g.replace(chr(92) + "n", "\n").split("\n"):
            if "###" in line:
                continue
            k = RAG.norm_key(line)
            if len(k) >= 2 and lex.by_key.get(k):
                gold.add(k)
        if not gold:
            continue                     # 사전에 없는 크롭은 검색기로 구분 불가
        q = [l for l in ocr.get(name, "").replace(chr(92) + "n", "\n").split("\n")
             if len(RAG.norm_key(l)) >= 2]
        if q:
            pairs.append((q, gold))
    return pairs


def evaluate(region: str, ks=(1, 3, 5, 10)) -> None:
    H = Hybrid(region)
    pairs = load_pairs(region)
    kmax = max(ks)
    key_of = [e["key"] for e in H.lex.entries]

    # dense 질의 임베딩은 전부 모아 한 번에 (API 왕복 최소화)
    flat = [q for qs, _ in pairs for q in qs]
    dense_all = H.dense_rank_batch(flat, kmax * 3) if H.emb is not None else []
    it = iter(dense_all)

    stats = {m: {k: 0 for k in ks} for m in ("lexical", "dense", "hybrid")}
    for qs, gold in pairs:
        lex_lists, den_lists = [], []
        for q in qs:
            lex_lists.append(H.lex_rank(q, kmax * 3))
            den_lists.append(next(it) if dense_all else [])
        merged = {
            "lexical": Hybrid.rrf(lex_lists, kmax),
            "dense": Hybrid.rrf(den_lists, kmax),
            "hybrid": Hybrid.rrf(lex_lists + den_lists, kmax),
        }
        for m, ranked in merged.items():
            keys = [key_of[i] for i in ranked]
            for k in ks:
                if gold & set(keys[:k]):
                    stats[m][k] += 1

    n = len(pairs)
    print(f"\n[{region}] 평가 크롭 {n}개 (정답이 사전에 있는 크롭만)")
    print(f"{'검색기':10s}" + "".join(f"{'hit@'+str(k):>9s}" for k in ks))
    for m in ("lexical", "dense", "hybrid"):
        if m == "dense" and H.emb is None:
            continue
        print(f"{m:10s}" + "".join(f"{stats[m][k]/max(n,1)*100:8.1f}%" for k in ks))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="코퍼스 임베딩 생성")
    ap.add_argument("--eval", action="store_true", help="세 검색기 비교")
    ap.add_argument("--region", choices=REGIONS)
    args = ap.parse_args()
    regions = (args.region,) if args.region else REGIONS
    if args.build:
        build(regions)
    if args.eval:
        for r in regions:
            evaluate(r)
    if not (args.build or args.eval):
        ap.error("--build 또는 --eval 을 주세요.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
