#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T6: 의미 태깅(name/amenity) — VLM(이미지) vs LLM(OCR 텍스트) 비교 평가.

논문 파이프라인의 3번째 모듈입니다. 같은 모델을 두 입력 모드로 돌려
**"OCR 모듈이 여전히 제 몫을 하는가"** 에 정량적으로 답하는 것이 핵심입니다:

  --mode vlm  : 간판 크롭 이미지 → name/tag        (OCR 우회)
  --mode text : 배포 OCR 출력 텍스트 → name/tag    (이미지 없이)
  --mode rag  : 배포 OCR 텍스트 + 지역 POI 검색 후보 → name/tag  (RAG 경로)
  --mode both : vlm + text (기본, D27 재현)
  --mode all  : vlm + text + rag

RAG 경로는 OCR 텍스트로 지역 상호 사전(artifacts/ocr_db/rag_{region}.csv)을 검색해
후보 상호와 그 업종 태그를 프롬프트에 참고자료로 붙입니다. 사전은 **치환하지 않고**
후보만 제시하며(T5 스냅 손상의 교훈, rag_retrieve.py 참고), 판단은 모델이 합니다.

**누수 방어**: RAG 경로는 사전에 이미 있는 업소에 유리하므로, 결과를 사전에
있는 간판 / 없는 간판으로 **분리 보고**합니다(in-DB / off-DB). 전자는 "기존 POI
검증·갱신", 후자는 "신규 POI 발굴" 성능이며 논문에서 같은 수치로 뭉뚱그리면 안 됩니다.

정답: artifacts/gt/tagging_gt.csv (review_tagging.py 검수본)
  eval_name=1 인 행만 name 채점, eval_tag=1 인 행만 tag 채점
  (간판만 보고 업종 판별이 불가한 건은 tag 평가에서 제외 — 사람도 못 하는 걸
   모델에 요구하지 않기 위함)

실행 엔진은 Ollama(로컬)입니다. transformers 핀(4.49, TrOCR용)과 충돌하지 않습니다.

Usage:
  .venv/Scripts/python.exe eval_vlm_tagging.py --model qwen2.5vl:7b --limit 100
  .venv/Scripts/python.exe eval_vlm_tagging.py --model gemma3:4b --mode vlm
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import random
import re
import sys
import time
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
GT = HERE / "artifacts" / "gt"
CROP = GT / "crop"
OCR_DIR = HERE / "artifacts" / "ocr_gt"
OLLAMA = "http://localhost:11434/api/generate"
DEPLOY_RUN = "24"          # 배포 OCR 산출물(run24)

# 검수 시 자유 입력으로 생긴 동의어·오타를 채점 전에 통일합니다.
# (보수적으로 명백한 것만 — 개념이 다른 태그는 합치지 않습니다.)
CANON = {
    "shop=optical_shop": "shop=optician",
    "amenity=karaoke": "amenity=karaoke_box",
    "shop=jewerly_shop": "shop=jewelry",
    "shop=jewelry_store": "shop=jewelry",
    "amenity=laundry": "shop=laundry",
    "office=PC_room": "amenity=internet_cafe",
    "shop=stationery_store": "shop=stationery",
    "shop=animal_market": "shop=pet",
    "amenity=animal_care": "amenity=veterinary",
    "amenity=rent_car": "amenity=car_rental",
    "shop=functional_food": "shop=health_food",
}


def canon(tag: str) -> str:
    t = (tag or "").strip().lower().replace(" ", "")
    return CANON.get(t, CANON.get(tag.strip(), t))


def norm_name(s: str) -> str:
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


def load_gt() -> list[dict]:
    rows = list(csv.DictReader((GT / "tagging_gt.csv").open(encoding="utf-8")))
    for r in rows:
        r["tag"] = canon(r["tag"]) if r["eval_tag"] == "1" else ""
    return rows


def load_ocr() -> dict[str, str]:
    out = {}
    for region in ("gangnam", "brooklyn", "suwon"):
        p = OCR_DIR / f"ocr_{region}_{DEPLOY_RUN}_paddle.csv"
        if not p.exists():
            continue
        for r in csv.DictReader(p.open(encoding="utf-8-sig")):
            out[r["image_name"]] = (r["gt_text"] or "").replace("\\n", " / ")
    return out


def build_hint_block(hints: str) -> str:
    """검색된 POI 후보를 '참고자료'로만 넣습니다.

    T5(사전 스냅)가 실패한 이유는 사전이 근거 없이 예측을 **치환**했기 때문입니다
    (D16: SPRINT→SPRING 류 손상). 여기서는 목록에 있다는 사실 자체가 근거가
    되지 않도록 명시적으로 금지하고, 일치 판단은 모델에 맡깁니다."""
    if not hints:
        return ""
    return (f"\n\n참고 — 이 지역에 등록된 업소 목록에서 위 텍스트와 닮은 항목들이다 "
            f"(무관한 항목이 섞여 있을 수 있다):\n{hints}\n"
            "- 간판 내용과 **명백히 같은 업소일 때만** 참고해서 표기·업종을 보정하라.\n"
            "- 목록에 있다는 이유만으로 상호명을 바꾸지 마라. 애매하면 무시하라.")


def build_prompt(tags: list[str], ocr_text: str | None, hints: str = "",
                 no_abstain: bool = False) -> str:
    """이미지 모드와 텍스트 모드는 실패 양상이 달라 프롬프트를 분리합니다.

    이미지 모드의 위험은 '안 보이는 걸 지어내는 것'이라 안티환각 문구가 필요하지만,
    텍스트 모드에 같은 문구를 주면 모델이 '주어진 텍스트에서 상호명 고르기'까지
    거부하고 빈 값을 냅니다(qwen에서 실측). 텍스트 모드는 선택 과제로 명시합니다."""
    vocab = "\n".join(f"- {t}" for t in tags)
    if ocr_text is None:
        head = "첨부한 간판 사진을 보고 이 업소의 상호명과 OSM 태그를 판별하라."
        name_rule = ("- 상호명(name)은 간판에 실제로 적힌 표기 그대로. "
                     "읽을 수 없으면 빈 문자열.\n"
                     "- **추측해서 지어내지 마라.** 근거가 없으면 name은 빈 문자열, "
                     'tag는 "unknown".')
    else:
        head = (f'간판에서 OCR로 인식된 텍스트다 (줄 구분은 " / ", '
                f'오타·워터마크·전화번호가 섞여 있을 수 있다):\n"{ocr_text}"\n'
                "이 텍스트에서 이 업소의 상호명과 OSM 태그를 판별하라.")
        name_rule = ("- 상호명(name)은 **위 텍스트 안에서** 상호명에 해당하는 부분을 "
                     "골라 그대로 옮겨 적는다. 부가정보(주소·전화번호·워터마크·업종어)는 "
                     "제외한다. 상호명으로 볼 부분이 전혀 없을 때만 빈 문자열.\n"
                     '- 태그를 판별할 근거가 없으면 "unknown".')
    # 기권 금지 조건: 채점 정책 차이를 제거하기 위한 대조군입니다. 논문 Table 3의
    # GPT-4 실험에 "근거 없으면 unknown" 제약이 있었는지 알 수 없어, 기권을 막았을
    # 때 정확도가 어디까지 오르는지를 따로 재기 위한 것입니다.
    if no_abstain:
        name_rule = name_rule.replace(
            '- 태그를 판별할 근거가 없으면 "unknown".',
            "- 태그는 확신이 없어도 반드시 하나를 고른다.")
        tag_rule = ('- 태그(tag)는 아래 목록에서 **반드시 하나를 고른다**. 확신이 '
                    '없어도 가장 가능성이 높은 것을 고르고, "unknown"은 쓰지 않는다.')
    else:
        tag_rule = ('- 태그(tag)는 아래 목록에서 정확히 하나만 고른다. '
                    '판별할 수 없으면 "unknown".')
    return f"""{head}{build_hint_block(hints)}

규칙:
{name_rule}
{tag_rule}
- 설명 없이 JSON 한 줄만 출력한다.

사용 가능한 태그:
{vocab}

출력 형식: {{"name": "...", "tag": "..."}}"""


def ask(model: str, prompt: str, img_path: Path | None, timeout=180) -> str:
    body = {"model": model, "prompt": prompt, "stream": False,
            "options": {"temperature": 0}}
    if img_path is not None:
        body["images"] = [base64.b64encode(img_path.read_bytes()).decode()]
    req = urllib.request.Request(OLLAMA, data=json.dumps(body).encode())
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["response"]


def parse(resp: str) -> tuple[str, str]:
    m = re.search(r"\{.*?\}", resp, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            return str(d.get("name", "")).strip(), canon(str(d.get("tag", "")).strip())
        except Exception:
            pass
    return "", ""


def evaluate(rows, preds, in_db: dict[str, bool] | None = None) -> dict:
    n_name = n_name_ok = 0
    edit = chars = 0
    n_tag = n_tag_ok = n_key_ok = n_unknown = 0
    halluc = 0
    strat = {True: [0, 0], False: [0, 0]}      # in_db → [정답, 전체]
    for r in rows:
        if in_db is not None and r["eval_tag"] == "1":
            b = in_db.get(r["image_name"], False)
            strat[b][1] += 1
            if preds[r["image_name"]][1] == r["tag"]:
                strat[b][0] += 1
        pn, pt = preds[r["image_name"]]
        if r["eval_name"] == "1":
            n_name += 1
            g, p = norm_name(r["name"]), norm_name(pn)
            if g and g == p:
                n_name_ok += 1
            edit += lev(p, g); chars += max(len(g), 1)
        if r["eval_tag"] == "1":
            n_tag += 1
            if pt == "unknown" or not pt:
                n_unknown += 1
            elif pt == r["tag"]:
                n_tag_ok += 1
            if pt and pt != "unknown" and pt.split("=")[0] == r["tag"].split("=")[0]:
                n_key_ok += 1
        else:
            # 사람이 "업종 불명"으로 판정한 건에 모델이 태그를 단언하면 환각 신호
            if pt and pt != "unknown":
                halluc += 1
    out = {
        "name_n": n_name, "name_exact": n_name_ok / max(n_name, 1),
        "name_cer": edit / max(chars, 1),
        "tag_n": n_tag, "tag_exact": n_tag_ok / max(n_tag, 1),
        "tag_key": n_key_ok / max(n_tag, 1),
        "tag_unknown": n_unknown / max(n_tag, 1),
        "halluc_on_unknown": halluc,
    }
    if in_db is not None:
        # 사전에 있는 업소 / 없는 업소를 나눠 봅니다. 전자는 "기존 POI 검증",
        # 후자는 "신규 POI 발굴" — RAG의 이득이 어디서 나오는지 드러납니다.
        for b, label in ((True, "indb"), (False, "offdb")):
            ok, n = strat[b]
            out[f"tag_{label}_n"] = n
            out[f"tag_{label}"] = ok / max(n, 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="ollama 모델명 (예: qwen2.5vl:7b)")
    ap.add_argument("--mode",
                    choices=["vlm", "text", "rag", "vlmrag", "both", "all"],
                    default="both",
                    help="vlmrag = 이미지 + POI 검색 후보 (검색 질의는 배포 OCR 텍스트). "
                         "영어권처럼 OCR 경로가 불리한 지역을 위한 경로")
    ap.add_argument("--rag-k", type=int, default=5,
                    help="rag 모드: 프롬프트에 넣을 POI 후보 수")
    ap.add_argument("--no-abstain", action="store_true",
                    help='"unknown" 기권을 금지하고 강제 선택시킴 (채점 정책 대조군)')
    ap.add_argument("--retriever", choices=["lexical", "hybrid"], default="lexical",
                    help="hybrid = lexical + bge-m3 dense 를 RRF 융합 "
                         "(rag_hybrid.py --build 로 임베딩을 먼저 만들어야 함)")
    ap.add_argument("--limit", type=int, default=0, help="지역별 균등 표본 수(0=전체)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="artifacts/gt/vlm_tagging_results.csv")
    args = ap.parse_args()

    rows = load_gt()
    # 후보 태그 목록은 항상 GT 전체에서 만든다. 표본에서 만들면 어휘가 줄어
    # 문제가 쉬워져 --limit 실행과 전체 실행을 비교할 수 없게 된다.
    tags = sorted({r["tag"] for r in rows if r["eval_tag"] == "1"})
    if args.limit:
        by_region: dict[str, list] = {}
        for r in rows:
            by_region.setdefault(r["region"], []).append(r)
        rng = random.Random(args.seed)
        per = max(1, args.limit // len(by_region))
        rows = [r for rs in by_region.values() for r in rng.sample(rs, min(per, len(rs)))]
    ocr = load_ocr()
    print(f"[EVAL] model={args.model} rows={len(rows)} "
          f"(name {sum(r['eval_name']=='1' for r in rows)} / tag {sum(r['eval_tag']=='1' for r in rows)}) "
          f"tags={len(tags)}")

    modes = {"both": ["vlm", "text"],
             "all": ["vlm", "text", "rag", "vlmrag"]}.get(args.mode, [args.mode])
    RAG_MODES = {"rag", "vlmrag"}

    in_db = None
    if RAG_MODES & set(modes):
        import rag_retrieve as RAG
        # 힌트 생성기만 교체합니다. in-DB 층화는 어느 검색기든 같은 사전을 보므로
        # lexical 쪽 by_key 를 그대로 씁니다 (비교 가능성 유지).
        HINT = RAG
        if args.retriever == "hybrid":
            import rag_hybrid as HINT
        # 정답 상호가 사전에 있는지 여부 — **채점 층화에만** 쓰고 모델에는 주지
        # 않습니다. 테스트셋을 나누는 것이지 입력을 만드는 것이 아닙니다.
        in_db = {}
        for r in rows:
            retr = RAG.get(r["region"])
            in_db[r["image_name"]] = bool(retr.by_key.get(norm_name(r["name"])))
        n_in = sum(in_db.values())
        print(f"[RAG] 사전 적재 완료 — 정답 상호가 사전에 있는 크롭 "
              f"{n_in}/{len(rows)} ({n_in/max(len(rows),1)*100:.1f}%)")

    summary, detail = {}, []
    for mode in modes:
        preds, t0, fails = {}, time.time(), 0
        for i, r in enumerate(rows, 1):
            img = (CROP / r["region"] / f"{r['image_name']}.jpg"
                   if mode in ("vlm", "vlmrag") else None)
            # vlmrag 는 프롬프트에 OCR 텍스트를 넣지 않습니다(이미지 경로 유지).
            # 다만 검색 질의로는 써야 하므로 따로 들고 있습니다.
            ocr_text = ocr.get(r["image_name"], "")
            text = None if mode in ("vlm", "vlmrag") else ocr_text
            hints, n_hit = "", 0
            if mode in RAG_MODES and ocr_text:
                retr = HINT.get(r["region"])
                # 정답 어휘 밖의 태그는 힌트에서 떼어냅니다(상호명은 유지).
                hints = retr.hint_block(
                    [l for l in ocr_text.split(" / ") if l.strip()],
                    k=args.rag_k, allowed_tags=set(tags))
                n_hit = len(hints.splitlines()) if hints else 0
            try:
                resp = ask(args.model, build_prompt(tags, text, hints,
                                           args.no_abstain), img)
                pn, pt = parse(resp)
            except Exception as exc:
                fails += 1
                pn, pt = "", ""
                if fails <= 3:
                    print(f"  [warn] {r['image_name']}: {type(exc).__name__}")
            preds[r["image_name"]] = (pn, pt)
            detail.append({"model": args.model, "mode": mode,
                           "image_name": r["image_name"], "region": r["region"],
                           "gt_name": r["name"], "gt_tag": r["tag"],
                           "pred_name": pn, "pred_tag": pt,
                           "rag_hits": n_hit,
                           "in_db": int(in_db[r["image_name"]]) if in_db else ""})
            if i % 20 == 0:
                el = time.time() - t0
                print(f"  [{mode}] {i}/{len(rows)}  {el/i:.1f}s/건  "
                      f"ETA {(len(rows)-i)*el/i/60:.0f}분", flush=True)
        m = evaluate(rows, preds, in_db)
        m["sec_per_item"] = (time.time() - t0) / len(rows)
        m["fails"] = fails
        summary[mode] = m

    print(f"\n{'mode':6s} {'name exact':>11s} {'name CER':>9s} | {'tag exact':>10s} "
          f"{'tag key':>8s} {'unknown':>8s} {'환각':>5s} {'s/건':>6s}")
    for mode, m in summary.items():
        print(f"{mode:6s} {m['name_exact']*100:10.1f}% {m['name_cer']:9.3f} | "
              f"{m['tag_exact']*100:9.1f}% {m['tag_key']*100:7.1f}% "
              f"{m['tag_unknown']*100:7.1f}% {m['halluc_on_unknown']:5d} "
              f"{m['sec_per_item']:6.1f}")

    if in_db is not None:
        print(f"\n[층화] 정답 상호가 사전에 있는 간판 / 없는 간판 (tag exact)")
        print(f"{'mode':6s} {'in-DB':>18s} {'off-DB':>18s}")
        for mode, m in summary.items():
            print(f"{mode:6s} {m['tag_indb']*100:12.1f}% (n={m['tag_indb_n']:3d}) "
                  f"{m['tag_offdb']*100:12.1f}% (n={m['tag_offdb_n']:3d})")
        print("※ 논문에는 이 분리 수치를 함께 실어야 합니다 — RAG 이득이 '기존 POI "
              "재확인'에서 온 것인지 '신규 발굴'에서 온 것인지가 구분되어야 하기 때문.")

    out = Path(args.out)
    if in_db is not None and out.name == "vlm_tagging_results.csv":
        out = out.with_name("vlm_tagging_results_rag.csv")   # 기존 D27 결과 보존
    fields = ["model", "mode", "image_name", "region",
              "gt_name", "gt_tag", "pred_name", "pred_tag"]
    if in_db is not None:
        fields += ["rag_hits", "in_db"]      # RAG 없는 실행은 기존 8열 형식 유지
    hdr = not out.exists()
    with out.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if hdr:
            w.writeheader()
        w.writerows(detail)
    print(f"[saved] {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
