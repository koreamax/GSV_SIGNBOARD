#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VLM×OCR 하이브리드가 OCR 지표(exact/CER/WAR)를 올리는가.

두 변형을 배포 OCR(run24, 69.9%/0.163/0.648)과 같은 프로토콜로 채점합니다:

  --mode solo   : gemma가 이미지만 보고 라인별로 읽음 (VLM 단독 OCR 기준선)
  --mode fix    : 배포 OCR 라인 + 이미지를 함께 주고 **오독 글자만 교정**.
                  라인 수·순서 유지, 확신 없으면 원문 유지, 새 라인 추가 금지
                  — 1~2글자 오독 버킷(전체 라인의 16%)을 겨냥하고,
                  라인 구조를 보존해 기존 라인 매칭 평가를 그대로 쓰기 위함.

출력: artifacts/ocr_gt/ab_v4_spacecat/ocr_{region}_vlm{mode}.csv (공식 run과 분리)
채점: eval_ocr_v2 모듈 임포트 (공식 프로토콜 동일).

Usage: .venv/Scripts/python.exe exp_vlm_ocr.py --mode fix [--model gemma3:12b]
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]

import sys as _sys; _sys.path[:0] = [str(HERE / _d) for _d in ("pipeline", "ocr", "vlm", "str_baselines", "detection", "data")]  # 폴더 재편 후 형제 모듈 import 경로
GT = HERE / "artifacts" / "gt"
CROP = GT / "crop"
OCR_DIR = HERE / "artifacts" / "ocr_gt"
OUT_DIR = OCR_DIR / "ab_v4_spacecat"
OLLAMA = "http://localhost:11434/api/generate"
REGIONS = ["gangnam", "brooklyn", "suwon"]
DEPLOY_RUN = "24"
DEPLOY_ENGINE = "paddle"


THINK = False          # Qwen3-VL/Gemma4 의 thinking 모드 — 오프로드 환경에서는 끕니다(--think 로 켬)
NUM_PREDICT = 400      # 간판 라인 JSON 은 짧음. 폭주(반복 출력) 방지 상한
NUM_CTX = 8192


def ask(model: str, prompt: str, img: Path, timeout=1500) -> str:
    body = {"model": model, "prompt": prompt, "stream": False, "think": THINK,
            "keep_alive": "2h",
            "options": {"temperature": 0, "num_predict": NUM_PREDICT, "num_ctx": NUM_CTX}}
    if img is not None:                       # fixtext(텍스트 전용 교정)는 이미지를 주지 않음
        body["images"] = [base64.b64encode(img.read_bytes()).decode()]
    last = None
    for attempt in range(3):
        try:
            b = dict(body)
            if attempt == 1:            # 구버전 서버/모델이 think 필드를 거부하는 경우
                b.pop("think", None)
            req = urllib.request.Request(OLLAMA, data=json.dumps(b).encode())
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())["response"]
        except Exception as e:          # noqa: BLE001
            last = e
            time.sleep(5)
    raise last


def parse_lines(resp: str, fallback: list[str]) -> list[str]:
    m = re.search(r"\[.*\]", resp, re.S)
    if m:
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            pass
    lines = [l.strip() for l in resp.strip().split("\n") if l.strip()]
    return lines if lines else fallback


def prompt_solo() -> str:
    return ("간판 사진의 텍스트를 읽어라.\n"
            "- 사진에서 시각적으로 한 줄인 텍스트를 배열 원소 하나로 한다.\n"
            "- 보이는 그대로 옮겨 적는다. 번역·설명·추측 금지.\n"
            "- 읽을 수 없는 부분은 건너뛴다.\n"
            '- JSON 배열만 출력한다. 예: ["첫째 줄", "둘째 줄"]')


def _nk(s: str) -> str:
    import unicodedata
    return re.sub(r"[^0-9a-z가-힣]", "",
                  unicodedata.normalize("NFKC", s or "").lower())


def _lev(a: str, b: str) -> int:
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


def line_variants(base: str, model_lines: dict[str, list[str]]) -> list[str]:
    """배포 라인 하나에 대해 각 모델 출력에서 가장 비슷한 라인을 후보로 수집.

    저장 파일은 빈 라인이 제거돼 인덱스가 어긋날 수 있으므로 유사도로 정렬합니다.
    후보는 정규화 키 기준으로 중복 제거."""
    B = _nk(base)
    out, seen = [base], {B}
    if not B:
        return out
    for lines in model_lines.values():
        best, bd = None, 10 ** 9
        for l in lines:
            L = _nk(l)
            if not L:
                continue
            d = _lev(B, L)
            if d < bd:
                bd, best = d, l
        if best is not None and bd <= max(2, int(len(B) * 0.5)):
            k = _nk(best)
            if k not in seen:
                seen.add(k)
                out.append(best)
    return out


def rag_block(hints: str) -> str:
    """검색된 지역 상호를 참고자료로만 붙입니다 (치환 금지 — T5/D16의 교훈).

    사전이 직접 예측을 바꾸면 실단어를 사전 이웃으로 망가뜨리는 사례
    (SPRINT→SPRING)와 복구 사례(istand→ISLAND)를 편집거리로 구분할 수 없었습니다.
    사진을 보는 심판에게 후보로만 넘기면 그 판단이 시각 확인 문제가 됩니다."""
    if not hints:
        return ""
    return ("\n\n참고 — 이 지역에 실제로 등록된 업소 이름 중 위 후보들과 닮은 것들이다 "
            f"(무관한 것이 섞여 있을 수 있다):\n{hints}\n"
            "- 사진의 글자와 **명백히 일치할 때만** 이 표기를 쓴다.\n"
            "- 목록에 있다는 이유만으로 후보를 바꾸지 마라. 애매하면 무시한다.")


def prompt_fixcand(cands: list[list[str]], hints: str = "") -> str:
    rows = []
    for i, vs in enumerate(cands, 1):
        rows.append(f"{i}. " + "  |  ".join(vs))
    listing = "\n".join(rows)
    return (f"간판 사진과, OCR 모델들이 각 라인을 읽은 후보들이다 "
            f"(라인마다 1~3개, '|'로 구분):\n{listing}"
            f"{rag_block(hints)}\n\n"
            "사진과 대조해 각 라인의 **올바른 텍스트**를 확정하라.\n"
            "- 후보 중 사진과 일치하는 것을 고르되, 모든 후보에 같은 오독이 있으면 "
            "사진 기준으로 그 글자만 고친다.\n"
            "- 확신이 없으면 첫 번째 후보를 그대로 쓴다.\n"
            "- 라인 개수와 순서를 유지하고, 새 라인을 추가하지 않는다.\n"
            f"- 확정된 라인 {len(cands)}개를 JSON 배열로만 출력한다.")


def prompt_fixtext(lines: list[str]) -> str:
    """텍스트 전용 post-OCR 교정 (문헌의 표준 기준선: LLM 이 OCR 출력만 보고 교정. 이미지 없음)."""
    numbered = "\n".join(f"{i+1}. {l}" for i, l in enumerate(lines))
    return (f"다음은 OCR 이 한 간판에서 읽은 텍스트 라인들이다 (사진은 제공되지 않는다):\n{numbered}\n\n"
            "언어 지식만으로 **명백한 오독 글자만** 고쳐라 (예: 상호·업종 표기에서 흔한 한글 자모 혼동, 영문 철자).\n"
            "- 라인 개수와 순서를 그대로 유지한다 (합치거나 나누지 말 것).\n"
            "- 확신이 없으면 원문 그대로 둔다. 새 텍스트를 추가하거나 라인을 삭제하지 않는다.\n"
            f'- 교정된 라인 {len(lines)}개를 JSON 배열로만 출력한다.')


def prompt_fix(lines: list[str]) -> str:
    numbered = "\n".join(f"{i+1}. {l}" for i, l in enumerate(lines))
    return (f"간판 사진과, OCR이 이 사진에서 읽은 텍스트 라인들이다:\n{numbered}\n\n"
            "각 라인을 사진과 대조해 **잘못 읽힌 글자만** 고쳐라.\n"
            "- 라인 개수와 순서를 그대로 유지한다 (합치거나 나누지 말 것).\n"
            "- 사진에서 확실히 다르게 보이는 글자만 고친다. 확신이 없으면 원문 그대로 둔다.\n"
            "- 새 텍스트를 추가하거나 라인을 삭제하지 않는다.\n"
            f'- 교정된 라인 {len(lines)}개를 JSON 배열로만 출력한다.')


def compose(args) -> None:
    """최종 하이브리드 = fixcand(후보 그라운딩 교정) + solo의 누락 라인 병합.

    fixcand는 라인 추가가 금지라 완전 전멸 버킷(~9%)을 못 건드리므로, solo가
    본 라인 중 fixcand 결과와 겹치지 않는 것만 덧붙입니다(정규화 키 유사도 기준)."""
    import unicodedata

    saved = sys.argv
    sys.argv = ["pipeline/eval_ocr_v2.py", "--mask-phone", "--en-only-regions", "brooklyn"]
    import eval_ocr_v2 as E
    sys.argv = saved

    def nk(s):
        return re.sub(r"[^0-9a-z가-힣]", "",
                      unicodedata.normalize("NFKC", s or "").lower())

    def similar(a, b):
        A, B = nk(a), nk(b)
        if not A or not B:
            return False
        if A in B or B in A:
            return True
        if abs(len(A) - len(B)) > max(len(A), len(B)) * 0.5:
            return False
        return E.lev(A, B) <= max(1, int(min(len(A), len(B)) * 0.3))

    tot = None
    for region in REGIONS:
        sfx = "_rag" if args.rag else ""
        # 탐지 크롭(4.16)에는 3모델 후보 파일이 없어 fixcand 대신 fix 를 씁니다.
        fix_name = args.fix_name or ("vlmfix" if args.use_fix else "vlmfixcand")
        fix = E.load_csv_map(OUT_DIR / f"ocr_{region}_{fix_name}{sfx}{args.out_suffix}.csv")
        solo = E.load_csv_map(OUT_DIR / f"ocr_{region}_vlmsolo{args.out_suffix}.csv")
        # GT 크롭 평가는 GT 키를 순회하지만, 탐지 크롭은 GT에 없는 이름이라
        # 예측 키 합집합을 순회해야 FP 크롭까지 합성됩니다.
        gt = (E.load_csv_map(GT / f"ocr_{region}_gt.csv") if not args.no_score
              else {k: "" for k in set(fix) | set(solo)})
        merged = {}
        for k in gt:
            fl = E.split_lines(fix.get(k, "")) if fix.get(k, "").strip() else []
            sl = E.split_lines(solo.get(k, "")) if solo.get(k, "").strip() else []
            extra = [s for s in sl if len(nk(s)) >= 2
                     and not any(similar(s, f) for f in fl)]
            merged[k] = "\n".join(fl + extra)
        if args.emit_run:
            dst = OCR_DIR / f"ocr_{region}_{args.emit_run}_paddle.csv"
            with dst.open("w", encoding="utf-8", newline="") as f:
                w = csv.writer(f, quoting=csv.QUOTE_ALL)
                w.writerow(["image_name", "gt_text"])
                for k in sorted(merged):
                    w.writerow([k, merged[k].replace("\n", "\\n")])
            print(f"[EMIT] {dst.name}")
        if args.no_score:
            print(f"[{region}] {len(merged)} 크롭 합성")
            continue
        E.EN_ONLY = region in E.EN_ONLY_REGIONS
        a = E.eval_engine(gt, merged, [])
        E.EN_ONLY = False
        print(f"[{region}] exact={a.exact_rate()*100:.1f}%  CER={a.cer():.3f}  "
              f"WAR={a.war():.3f}  contain={a.contain_rate()*100:.1f}%")
        if tot is None:
            tot = a
        else:
            for f_ in ("edit", "fp_edit", "chars", "word_lcs", "gt_words", "exact",
                       "recalled", "lines", "contain", "empty_pred", "crops"):
                setattr(tot, f_, getattr(tot, f_) + getattr(a, f_))
    if args.no_score:
        return
    print(f"[전역/compose] exact={tot.exact_rate()*100:.1f}%  CER={tot.cer():.3f}  "
          f"WAR={tot.war():.3f}  contain={tot.contain_rate()*100:.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["solo", "fix", "fixtext", "fixcand", "compose"], required=True)
    ap.add_argument("--model", default="gemma3:12b")
    ap.add_argument("--emit-run", default=None,
                    help="compose 모드: 최종 하이브리드를 ocr_{region}_{N}_ensemble.csv로 출력")
    ap.add_argument("--use-fix", action="store_true",
                    help="compose 재료로 fixcand 대신 fix 출력을 씀 "
                         "(탐지 크롭은 3모델 후보 파일이 없어 fix 사용)")
    ap.add_argument("--rag", action="store_true",
                    help="지역 POI 사전 검색 결과를 프롬프트 참고자료로 추가 "
                         "(rag_retrieve.py — 치환이 아니라 후보 제시)")
    ap.add_argument("--rag-k", type=int, default=5, help="크롭당 POI 후보 수")
    # ---- 연쇄(e2e) 평가용 (4.16) ----
    ap.add_argument("--crop-dir", default=None,
                    help="artifacts/gt/crop 대신 쓸 크롭 루트 (예: artifacts/gt/crop_det)")
    ap.add_argument("--deploy-run", default=None,
                    help="기준 OCR run (기본 24). 탐지 크롭은 30")
    ap.add_argument("--deploy-engine", default="paddle",
                    help="기준 run 파일의 엔진 태그 (D38: YOLO 검출기 구성은 yolo01). "
                         "engine 태그를 분리해야 eval_ocr_v2 의 '엔진별 최신 run' 선택과 충돌하지 않습니다")
    ap.add_argument("--out-suffix", default="",
                    help="출력 파일명 접미사 (예: _det) — GT 크롭 산출물과 분리")
    ap.add_argument("--no-score", action="store_true",
                    help="내부 채점 생략. 탐지 크롭은 이름이 GT와 달라 "
                         "eval_e2e_cascade.py 로 따로 채점합니다")
    # ---- 다중 VLM 비교(4.20) ----
    ap.add_argument("--cand-tags", default="v5,v4,pre",
                    help="fixcand 후보 모델 태그(ocr_{region}_cand_{tag}.csv). 5모델: v5,v4,pre,svtrv2,parseq")
    ap.add_argument("--name", default=None,
                    help="출력 파일의 모드 이름 override (예: fixcand5). 기본은 --mode")
    ap.add_argument("--fix-name", default=None,
                    help="compose 재료가 되는 교정 출력 이름 (예: vlmfixcand5). 기본 vlmfixcand/vlmfix")
    ap.add_argument("--think", action="store_true", help="thinking 모드 켜기(느림)")
    ap.add_argument("--limit", type=int, default=0, help="지역당 앞 N 크롭만 (속도 측정·스모크용)")
    args = ap.parse_args()

    global CROP, DEPLOY_RUN, DEPLOY_ENGINE, THINK
    THINK = args.think
    DEPLOY_ENGINE = args.deploy_engine
    if args.crop_dir:
        CROP = Path(args.crop_dir)
    if args.deploy_run:
        DEPLOY_RUN = args.deploy_run

    if args.mode == "compose":
        compose(args)
        return

    saved = sys.argv
    sys.argv = ["pipeline/eval_ocr_v2.py", "--mask-phone", "--en-only-regions", "brooklyn"]
    import eval_ocr_v2 as E
    sys.argv = saved

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tot = None
    print(f"[VLM-OCR] mode={args.mode} model={args.model}")
    for region in REGIONS:
        gt = E.load_csv_map(GT / f"ocr_{region}_gt.csv")
        deploy = E.load_csv_map(OCR_DIR / f"ocr_{region}_{DEPLOY_RUN}_{DEPLOY_ENGINE}.csv")
        cand_maps = {}
        if args.mode == "fixcand":
            for tag in [t for t in args.cand_tags.split(",") if t]:
                p = OUT_DIR / f"ocr_{region}_cand_{tag}.csv"
                if p.exists():
                    cand_maps[tag] = E.load_csv_map(p)
        retr = None
        if args.rag:
            import rag_retrieve as RAG
            retr = RAG.get(region)
            print(f"  [rag] {region}: POI 사전 {len(retr)}건")
        pred, t0 = {}, time.time()
        stem = f"ocr_{region}_vlm{args.name or args.mode}{'_rag' if args.rag else ''}{args.out_suffix}"
        ckpt = OUT_DIR / (stem + ".partial.jsonl")
        if ckpt.exists():               # 중단 후 재실행 시 끝난 크롭은 건너뜀
            for ln in ckpt.open(encoding="utf-8"):
                try:
                    d = json.loads(ln); pred[d["key"]] = d["text"]
                except Exception:       # noqa: BLE001
                    pass
            print(f"  [resume] {region}: {len(pred)} 크롭 복원")
        ck = ckpt.open("a", encoding="utf-8")
        def _save(k, t):
            ck.write(json.dumps({"key": k, "text": t}, ensure_ascii=False) + "\n"); ck.flush()
        # 탐지 크롭(4.16)은 GT에 없는 이름이라 GT 키를 돌면 안 됩니다.
        # 배포 예측(run30) 키 = 실제 탐지 크롭 목록입니다.
        keys = sorted(deploy.keys()) if args.no_score else list(gt.keys())
        if args.limit:
            keys = keys[:args.limit]
        for i, key in enumerate(keys, 1):
            if key in pred:
                continue
            img = CROP / region / f"{key}.jpg"
            base_lines = E.split_lines(deploy.get(key, "")) if deploy.get(key, "").strip() else []
            try:
                if args.mode == "solo":
                    resp = ask(args.model, prompt_solo(), img)
                    lines = parse_lines(resp, [])
                else:
                    if not base_lines:
                        pred[key] = ""
                        _save(key, "")
                        continue
                    if args.mode == "fixcand" and cand_maps:
                        ml = {t: E.split_lines(m.get(key, "")) if m.get(key, "").strip() else []
                              for t, m in cand_maps.items()}
                        cands = [line_variants(b, ml) for b in base_lines]
                        hints = retr.hint_block(base_lines, k=args.rag_k) if retr else ""
                        resp = ask(args.model, prompt_fixcand(cands, hints), img)
                    elif args.mode == "fixtext":
                        resp = ask(args.model, prompt_fixtext(base_lines), None)
                    else:
                        resp = ask(args.model, prompt_fix(base_lines), img)
                    lines = parse_lines(resp, base_lines)
                    if len(lines) != len(base_lines):   # 구조 훼손 시 원문 유지
                        lines = base_lines
            except Exception:
                lines = base_lines
            pred[key] = "\n".join(lines)
            _save(key, pred[key])
            if i % 10 == 0:
                el = time.time() - t0
                print(f"  [{region}] {i}/{len(keys)}  {el/i:.1f}s/크롭  "
                      f"ETA {(len(keys)-i)*el/i/60:.0f}분", flush=True)

        ck.close()
        dst = OUT_DIR / (stem + ".csv")
        with dst.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, quoting=csv.QUOTE_ALL)
            w.writerow(["image_name", "gt_text"])
            for k in sorted(pred):
                w.writerow([k, pred[k].replace("\n", "\\n")])

        if args.no_score:
            print(f"[{region}] {len(pred)} 크롭 저장 (채점은 eval_e2e_cascade.py)")
            continue
        E.EN_ONLY = region in E.EN_ONLY_REGIONS
        a = E.eval_engine(gt, pred, [])
        E.EN_ONLY = False
        print(f"[{region}] exact={a.exact_rate()*100:.1f}%  CER={a.cer():.3f}  "
              f"WAR={a.war():.3f}  contain={a.contain_rate()*100:.1f}%")
        if tot is None:
            tot = a
        else:
            for f_ in ("edit", "fp_edit", "chars", "word_lcs", "gt_words", "exact",
                       "recalled", "lines", "contain", "empty_pred", "crops"):
                setattr(tot, f_, getattr(tot, f_) + getattr(a, f_))

    if args.no_score:      # 탐지 크롭은 eval_e2e_cascade.py 로 따로 채점
        return
    print(f"\n[전역/{args.mode}] exact={tot.exact_rate()*100:.1f}%  CER={tot.cer():.3f}  "
          f"WAR={tot.war():.3f}  contain={tot.contain_rate()*100:.1f}%")
    print("[참고/배포 run24] exact=69.9%  CER=0.163  WAR=0.648  contain=75.2%")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
