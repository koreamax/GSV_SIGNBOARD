#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NAVER CLOVA OCR (General) 라인 인식 워커 — paddle_rec_worker 와 같은 IO 계약.

  --manifest IN.jsonl : {"key": str, "path": str}
  --out      OUT.jsonl: {"key": str, "text": str, "score": float}

Tesseract·Surya 와 같은 **범용 기성 엔진** 취급입니다. 라인 스트립 1장을 그대로 API 에
넘기고, 응답 필드(fields[])를 lineBreak 기준으로 이어 붙여 라인 텍스트를 만듭니다.

인증 정보는 코드·명령줄에 두지 않습니다. 다음 중 하나로 읽습니다(우선순위 순):
  1) 환경변수 CLOVA_OCR_URL / CLOVA_OCR_SECRET
  2) JSON 파일 (기본 artifacts/str_baselines/clova_ocr/credentials.json, git 제외):
       {"url": "https://....apigw.ntruss.com/custom/v1/.../general", "secret": "..."}

유료 API 이므로 안전장치를 둡니다:
  * 크롭 단위 체크포인트(.partial.jsonl) — 중단 후 재실행하면 끝난 건은 건너뜁니다(재과금 없음).
  * --max-calls 호출 상한(기본 3000). 넘으면 즉시 중단합니다.
  * --dry-run 은 호출 없이 건수만 셉니다.

Usage:
  .venv/Scripts/python.exe str_baselines/clova_rec_worker.py --manifest in.jsonl --out out.jsonl
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
CRED_DEFAULT = HERE / "artifacts" / "str_baselines" / "clova_ocr" / "credentials.json"
EXT2FMT = {".jpg": "jpg", ".jpeg": "jpg", ".png": "png", ".tif": "tif", ".tiff": "tif"}


def load_credentials(path: Path) -> tuple[str, str]:
    url, secret = os.environ.get("CLOVA_OCR_URL"), os.environ.get("CLOVA_OCR_SECRET")
    if url and secret:
        return url.strip(), secret.strip()
    if path.exists():
        d = json.loads(path.read_text(encoding="utf-8"))
        url, secret = str(d.get("url", "")).strip(), str(d.get("secret", "")).strip()
        if url and secret:
            return url, secret
    sys.exit(
        "CLOVA 인증 정보가 없습니다.\n"
        "  환경변수 CLOVA_OCR_URL / CLOVA_OCR_SECRET 를 설정하거나\n"
        f"  {path} 에 {{\"url\": \"...\", \"secret\": \"...\"}} 를 저장하세요.\n"
        "  (NAVER Cloud Platform → CLOVA OCR → General 도메인 → API Gateway 연동)"
    )


def preflight(url: str) -> None:
    """호출 전에 엔드포인트가 이 망에서 닿는지 확인 — 잘못된 URL 로 타임아웃만 쌓는 것을 막습니다.

    CLOVA 콘솔에는 공인(APIGW Invoke URL)과 VPC 내부용 주소가 함께 보입니다.
    내부용(`clovaocr-api-kr.ncloud.com` → 10.x.x.x)을 복사하면 외부망에서는 영영 닿지 않습니다."""
    import ipaddress
    import socket
    from urllib.parse import urlparse

    u = urlparse(url)
    host, port = u.hostname, u.port or (443 if u.scheme == "https" else 80)
    if not host:
        sys.exit(f"CLOVA URL 형식이 올바르지 않습니다: {url!r}")
    try:
        ips = sorted({ai[4][0] for ai in socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)})
    except socket.gaierror as e:
        sys.exit(f"CLOVA 호스트 DNS 조회 실패: {host} ({e})")
    if all(ipaddress.ip_address(ip).is_private for ip in ips):
        sys.exit(
            f"CLOVA 엔드포인트 {host} 가 사설 IP({', '.join(ips)})로 풀립니다 — VPC 내부 전용 주소입니다.\n"
            "  콘솔의 **APIGW Invoke URL**(공인)을 쓰세요. 형태:\n"
            "    https://<식별자>.apigw.ntruss.com/custom/v1/<도메인ID>/<키>/general\n"
            "  CLOVA OCR → 도메인 → [API Gateway 연동] 에서 확인할 수 있습니다."
        )
    s = socket.socket()
    s.settimeout(8)
    try:
        s.connect((ips[0], port))
    except OSError as e:
        sys.exit(f"CLOVA 엔드포인트에 연결할 수 없습니다: {host}:{port} ({e})")
    finally:
        s.close()


class Budget:
    """호출 상한 — 유료 API 사고 방지."""

    def __init__(self, limit: int):
        self.limit, self.used, self._lk = limit, 0, threading.Lock()

    def take(self) -> bool:
        with self._lk:
            if self.used >= self.limit:
                return False
            self.used += 1
            return True


def lines_from_fields(fields: list[dict]) -> tuple[str, float]:
    """fields[] → 라인 텍스트. V2 의 lineBreak=True 가 라인 끝을 뜻합니다.

    lineBreak 이 없으면(V1 등) boundingPoly 의 y 중심으로 묶습니다."""
    if not fields:
        return "", 0.0
    confs = [float(f.get("inferConfidence") or 0.0) for f in fields]
    score = sum(confs) / len(confs) if confs else 0.0

    if any("lineBreak" in f for f in fields):
        lines, cur = [], []
        for f in fields:
            t = (f.get("inferText") or "").strip()
            if t:
                cur.append(t)
            if f.get("lineBreak") and cur:
                lines.append(" ".join(cur))
                cur = []
        if cur:
            lines.append(" ".join(cur))
        return "\n".join(lines), score

    rows = []                                   # y 중심 기준 그룹핑 (fallback)
    for f in fields:
        t = (f.get("inferText") or "").strip()
        if not t:
            continue
        vs = (f.get("boundingPoly") or {}).get("vertices") or []
        ys = [v.get("y", 0) for v in vs] or [0]
        xs = [v.get("x", 0) for v in vs] or [0]
        rows.append((sum(ys) / len(ys), min(xs), max(ys) - min(ys), t))
    if not rows:
        return "", score
    rows.sort()
    h = max(1.0, sum(r[2] for r in rows) / len(rows))
    lines, cur, y0 = [], [], rows[0][0]
    for yc, x0, _, t in rows:
        if cur and abs(yc - y0) > h * 0.6:
            lines.append(" ".join(s for _, s in sorted(cur)))
            cur, y0 = [], yc
        cur.append((x0, t))
    if cur:
        lines.append(" ".join(s for _, s in sorted(cur)))
    return "\n".join(lines), score


def call_api(url: str, secret: str, img: Path, lang: str, timeout: int,
             retries: int = 4) -> tuple[str, float]:
    fmt = EXT2FMT.get(img.suffix.lower(), "jpg")
    body = {
        "version": "V2",
        "requestId": str(uuid.uuid4()),
        "timestamp": int(time.time() * 1000),
        "lang": lang,
        "images": [{"format": fmt, "name": img.stem,
                    "data": base64.b64encode(img.read_bytes()).decode()}],
    }
    data = json.dumps(body).encode()
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(url, data=data, method="POST", headers={
            "Content-Type": "application/json; charset=utf-8", "X-OCR-SECRET": secret})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                res = json.loads(r.read())
            im = (res.get("images") or [{}])[0]
            if im.get("inferResult") != "SUCCESS":
                return "", 0.0                      # 인식 실패는 빈 문자열(= 다른 엔진과 동일 취급)
            return lines_from_fields(im.get("fields") or [])
        except urllib.error.HTTPError as e:          # 429/5xx 만 재시도
            last = e
            if e.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise
        except Exception as e:                        # noqa: BLE001 — 타임아웃·네트워크
            last = e
            if attempt == retries - 1:
                raise
        time.sleep((2 ** attempt) + random.random())
    raise last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--credentials", default=str(CRED_DEFAULT))
    ap.add_argument("--lang", default="ko", help="ko / ja / zh-TW (영어는 ko 모델이 함께 읽습니다)")
    ap.add_argument("--workers", type=int, default=3, help="동시 호출 수 (기본 3 — API 한도 보호)")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--max-calls", type=int, default=3000, help="이번 실행의 API 호출 상한")
    ap.add_argument("--dry-run", action="store_true", help="호출 없이 필요한 건수만 출력")
    args = ap.parse_args()

    items = [json.loads(l) for l in open(args.manifest, encoding="utf-8") if l.strip()]
    ckpt = Path(args.out).with_suffix(".partial.jsonl")
    done: dict[str, dict] = {}
    if ckpt.exists():
        for l in ckpt.open(encoding="utf-8"):
            try:
                d = json.loads(l)
                done[d["key"]] = d
            except Exception:                         # noqa: BLE001
                pass
    todo = [it for it in items if it["key"] not in done]
    print(f"[clova] 전체 {len(items)} / 완료 {len(done)} / 이번 호출 대상 {len(todo)}", flush=True)

    if args.dry_run:
        print(f"[clova] dry-run — API 호출 {len(todo)}건이 필요합니다.")
        return
    if len(todo) > args.max_calls:
        print(f"[clova] 상한 초과: {len(todo)} > --max-calls {args.max_calls}. "
              f"상한을 올리거나 나눠서 실행하세요.", flush=True)
        sys.exit(2)

    url, secret = load_credentials(Path(args.credentials))
    preflight(url)
    budget = Budget(args.max_calls)
    lock = threading.Lock()
    ck = ckpt.open("a", encoding="utf-8")
    t0, n = time.time(), 0
    stop = threading.Event()

    def run_one(it: dict) -> dict:
        nonlocal n
        if stop.is_set() or not budget.take():
            stop.set()
            return {"key": it["key"], "text": "", "score": 0.0}
        try:
            text, score = call_api(url, secret, Path(it["path"]), args.lang, args.timeout)
        except Exception as e:                        # noqa: BLE001
            print(f"[clova] 호출 실패 {it['key']}: {e}", flush=True)
            stop.set()                                # 과금 낭비 방지: 연속 실패 시 중단
            return {"key": it["key"], "text": "", "score": 0.0}
        d = {"key": it["key"], "text": text, "score": score}
        with lock:
            ck.write(json.dumps(d, ensure_ascii=False) + "\n")
            ck.flush()
            n += 1
            if n % 25 == 0:
                el = time.time() - t0
                print(f"[clova] {n}/{len(todo)}  {el/n:.2f}s/건  "
                      f"ETA {(len(todo)-n)*el/n/60:.0f}분", flush=True)
        return d

    with ThreadPoolExecutor(args.workers) as ex:
        for d in ex.map(run_one, todo):
            done[d["key"]] = d
    ck.close()

    with open(args.out, "w", encoding="utf-8") as f:
        for it in items:
            d = done.get(it["key"]) or {"key": it["key"], "text": "", "score": 0.0}
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"[clova] done {len(items)} -> {args.out}  (API 호출 {budget.used}건)", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
