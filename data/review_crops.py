#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""review_crops.py — manual inspection UI for signboard_v3 training crops.

Serves a local web app (stdlib only) to review every crop one by one:
  .venv/Scripts/python.exe review_crops.py            # -> http://localhost:8123
  .venv/Scripts/python.exe review_crops.py --export   # write cleaned files from marks

Verdicts are appended to artifacts/ocr_training/signboard_v3/review_marks.csv
(append-only, last mark per image wins). --export writes:
  labels_clean.csv  train_clean.txt  (bad-marked rows removed)

Suspicion heuristics (sort "suspicious first") use the manifest bbox only:
  empty/short label, label longer than the box could plausibly hold,
  box too wide for a short label, tiny boxes, chars outside [0-9A-Za-z가-힣].
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
V3 = HERE / "artifacts" / "ocr_training" / "signboard_v3"
MANIFEST = V3 / "labels.csv"
MARKS = V3 / "review_marks.csv"

_ALLOWED = re.compile(r"[0-9A-Za-z가-힣\s]")


def suspicion(row: dict) -> tuple[int, list[str]]:
    text = row["text"]
    w, h = float(row["w"] or 0), float(row["h"] or 0)
    score, why = 0, []
    t = text.strip()
    if not t:
        score += 100; why.append("empty label")
    n = max(len(t.replace(" ", "")), 1)
    weird = sum(1 for c in t if not _ALLOWED.match(c))
    if weird:
        score += 20 + 5 * weird; why.append(f"{weird} non-standard chars")
    if h > 0 and w > 0:
        px_per_char = (w / h) / n          # width in "h units" per character
        if px_per_char < 0.22:
            score += 30; why.append(f"label too long for box ({px_per_char:.2f} h/char)")
        elif px_per_char > 4.0 and n > 1:
            score += 20; why.append(f"box too wide for label ({px_per_char:.1f} h/char)")
        if h < 14:
            score += 15; why.append(f"tiny box h={int(h)}px")
        if w < 14:
            score += 15; why.append(f"tiny box w={int(w)}px")
    else:
        score += 25; why.append("no bbox size")
    if len(t) > 20:
        score += 10; why.append(f"very long label ({len(t)} chars)")
    if len(t) == 1:
        score += 5; why.append("single char")
    return score, why


def load_items() -> list[dict]:
    items = []
    with MANIFEST.open(encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f)):
            s, why = suspicion(r)
            items.append({
                "id": i, "path": r["image_path"], "text": r["text"],
                "split": r["split"], "src": r.get("source_image", ""),
                "w": r.get("w", ""), "h": r.get("h", ""),
                "score": s, "why": why,
            })
    return items


def load_marks() -> dict[str, str]:
    marks: dict[str, str] = {}
    if MARKS.exists():
        with MARKS.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                marks[r["image_path"]] = r["verdict"]   # last wins
    return marks


def append_mark(path: str, text: str, verdict: str) -> None:
    new = not MARKS.exists()
    with MARKS.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["image_path", "text", "verdict", "ts"])
        w.writerow([path, text, verdict, time.strftime("%Y-%m-%d %H:%M:%S")])


def export() -> None:
    marks = load_marks()
    bad = {p for p, v in marks.items() if v == "bad"}
    with MANIFEST.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fields = rows[0].keys() if rows else []
    kept = [r for r in rows if r["image_path"] not in bad]
    with (V3 / "labels_clean.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader(); w.writerows(kept)
    n_tr = 0
    with (V3 / "train.txt").open(encoding="utf-8") as f, \
         (V3 / "train_clean.txt").open("w", encoding="utf-8", newline="\n") as g:
        for line in f:
            p = line.split("\t", 1)[0].strip()
            if p not in bad:
                g.write(line); n_tr += 1
    print(f"[EXPORT] bad={len(bad)}  labels_clean.csv rows={len(kept)}/{len(rows)}  "
          f"train_clean.txt lines={n_tr}")
    print(f"[EXPORT] -> {V3 / 'labels_clean.csv'}")
    print(f"[EXPORT] -> {V3 / 'train_clean.txt'}")


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>crop review</title><style>
body{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:0;display:flex;height:100vh}
#side{width:340px;min-width:340px;overflow-y:auto;border-right:1px solid #333;padding:10px}
#main{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:14px;padding:20px}
#img{max-width:90%;max-height:55vh;image-rendering:pixelated;background:#fff;border:2px solid #555}
#label{font-size:2.2em;font-weight:700}
#meta{color:#9a9;font-size:.95em;text-align:center;white-space:pre-line}
#why{color:#e6b05c}
.btn{font-size:1.2em;padding:8px 22px;margin:0 6px;border-radius:8px;border:0;cursor:pointer}
#good{background:#2d7a2d;color:#fff}#bad{background:#a32626;color:#fff}#skip{background:#555;color:#fff}
.row{padding:4px 6px;border-bottom:1px solid #222;cursor:pointer;font-size:.85em;display:flex;gap:6px;align-items:center}
.row.cur{background:#274a63}.row .v-good{color:#5fd35f}.row .v-bad{color:#ff6b6b}.row .v-skip{color:#aaa}
select,button.small{background:#222;color:#eee;border:1px solid #444;padding:4px 8px;margin:2px;border-radius:4px}
#stats{position:sticky;top:0;background:#111;padding-bottom:6px;border-bottom:1px solid #333;margin-bottom:4px}
kbd{background:#333;padding:1px 6px;border-radius:4px}</style></head><body>
<div id="side"><div id="stats"></div><div id="list"></div><button class="small" id="more">load more…</button></div>
<div id="main">
 <img id="img"><div id="label"></div><div id="meta"></div><div id="why"></div>
 <div><button class="btn" id="good">✓ 정상 (G/→)</button><button class="btn" id="bad">✗ 불량 (B/X)</button><button class="btn" id="skip">보류 (S)</button></div>
 <div style="color:#888">단축키: <kbd>G</kbd>/<kbd>→</kbd> 정상·다음 <kbd>B</kbd> 불량·다음 <kbd>S</kbd> 보류 <kbd>←</kbd> 이전 <kbd>Z</kbd> 되돌리기 — 저장은 자동</div>
 <div><select id="split"><option value="train">train (62k)</option><option value="val">val</option><option value="test">test</option><option value="all">all</option></select>
 <select id="order"><option value="suspicious">의심 우선</option><option value="seq">순서대로</option></select>
 <select id="filter"><option value="unreviewed">미검수만</option><option value="all">전체</option><option value="bad">불량만</option></select>
 <button class="small" id="reload">적용</button></div>
</div><script>
let Q=[],idx=0,shown=200,hist=[];
const $=id=>document.getElementById(id);
async function fetchQ(){const s=$("split").value,o=$("order").value,f=$("filter").value;
 const r=await fetch(`/api/items?split=${s}&order=${o}&filter=${f}`);Q=await r.json();idx=0;shown=200;render();}
function render(){const it=Q[idx];const st=$("stats");
 fetch("/api/stats").then(r=>r.json()).then(s=>{st.textContent=`전체 ${s.total} | 검수 ${s.reviewed} (정상 ${s.good} · 불량 ${s.bad} · 보류 ${s.skip}) | 대기열 ${Q.length}`});
 const L=$("list");L.innerHTML="";
 Q.slice(0,shown).forEach((q,i)=>{const d=document.createElement("div");d.className="row"+(i===idx?" cur":"");
  d.innerHTML=`<span class="v-${q.mark||''}">${q.mark?(q.mark==="good"?"✓":q.mark==="bad"?"✗":"…"):"·"}</span><span>[${q.score}]</span><span>${q.text.slice(0,18)}</span>`;
  d.onclick=()=>{idx=i;render()};L.appendChild(d);});
 if(!it){$("img").src="";$("label").textContent="큐 끝!";$("meta").textContent="";$("why").textContent="";return;}
 $("img").src="/img/"+encodeURIComponent(it.path);$("label").textContent=it.text||"(빈 라벨)";
 $("meta").textContent=`${idx+1}/${Q.length}  ${it.path}\\nsplit=${it.split}  box=${it.w}×${it.h}px  원본=${it.src}`;
 $("why").textContent=it.why.length?("의심 사유: "+it.why.join(", ")):"";
 const cur=document.querySelector(".row.cur");if(cur)cur.scrollIntoView({block:"nearest"});}
async function mark(v){const it=Q[idx];if(!it)return;hist.push({i:idx,old:it.mark});it.mark=v;
 await fetch("/api/mark",{method:"POST",body:JSON.stringify({path:it.path,text:it.text,verdict:v})});
 idx=Math.min(idx+1,Q.length);if(idx>shown-20)shown+=200;render();}
function undo(){const h=hist.pop();if(!h)return;idx=h.i;const it=Q[idx];it.mark=h.old;
 fetch("/api/mark",{method:"POST",body:JSON.stringify({path:it.path,text:it.text,verdict:h.old||"skip"})});render();}
$("good").onclick=()=>mark("good");$("bad").onclick=()=>mark("bad");$("skip").onclick=()=>mark("skip");
$("reload").onclick=fetchQ;$("more").onclick=()=>{shown+=400;render()};
document.addEventListener("keydown",e=>{if(e.key==="g"||e.key==="G"||e.key==="ArrowRight")mark("good");
 else if(e.key==="b"||e.key==="B"||e.key==="x"||e.key==="X")mark("bad");
 else if(e.key==="s"||e.key==="S")mark("skip");
 else if(e.key==="ArrowLeft"){idx=Math.max(0,idx-1);render();}
 else if(e.key==="z"||e.key==="Z")undo();});
fetchQ();</script></body></html>"""


class H(BaseHTTPRequestHandler):
    items: list[dict] = []
    marks: dict[str, str] = {}
    key: str = ""          # if set, require ?key=... once (then cookie) on every request

    def _authed(self) -> bool:
        if not self.key:
            return True
        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        if q.get("key", [""])[0] == self.key:
            return True
        cookies = self.headers.get("Cookie", "")
        return f"rk={self.key}" in cookies.replace(" ", "")

    def _deny(self):
        b = b"403: access key required (open the link with ?key=...)"
        self.send_response(403)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if not self._authed():
            self._deny(); return
        u = urllib.parse.urlparse(self.path)
        if u.path == "/":
            b = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            if self.key:
                self.send_header("Set-Cookie", f"rk={self.key}; Path=/; HttpOnly")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        elif u.path == "/api/stats":
            g = sum(1 for v in self.marks.values() if v == "good")
            b = sum(1 for v in self.marks.values() if v == "bad")
            s = sum(1 for v in self.marks.values() if v == "skip")
            self._json({"total": len(self.items), "reviewed": g + b + s,
                        "good": g, "bad": b, "skip": s})
        elif u.path == "/api/items":
            q = urllib.parse.parse_qs(u.query)
            split = q.get("split", ["train"])[0]
            order = q.get("order", ["suspicious"])[0]
            filt = q.get("filter", ["unreviewed"])[0]
            out = [i for i in self.items if split == "all" or i["split"] == split]
            for i in out:
                i["mark"] = self.marks.get(i["path"], "")
            if filt == "unreviewed":
                out = [i for i in out if not i["mark"]]
            elif filt == "bad":
                out = [i for i in out if i["mark"] == "bad"]
            if order == "suspicious":
                out = sorted(out, key=lambda i: -i["score"])
            self._json(out[:20000])
        elif u.path.startswith("/img/"):
            rel = urllib.parse.unquote(u.path[5:])
            p = (V3 / rel).resolve()
            if V3.resolve() not in p.parents or not p.exists():
                self.send_response(404); self.end_headers(); return
            b = p.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        if not self._authed():
            self._deny(); return
        if self.path == "/api/mark":
            n = int(self.headers.get("Content-Length", 0))
            d = json.loads(self.rfile.read(n).decode("utf-8"))
            self.marks[d["path"]] = d["verdict"]
            append_mark(d["path"], d.get("text", ""), d["verdict"])
            self._json({"ok": True})
        else:
            self.send_response(404); self.end_headers()

    def log_message(self, *a):   # quiet
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--host", type=str, default="127.0.0.1",
                    help="Bind address. Use 0.0.0.0 to allow remote access (LAN).")
    ap.add_argument("--key", type=str, default="",
                    help="Access key: requests must present ?key=... once (cookie after).")
    ap.add_argument("--export", action="store_true",
                    help="Write labels_clean.csv / train_clean.txt from bad marks and exit.")
    args = ap.parse_args()
    if args.export:
        export(); return
    print("[load] manifest ...")
    H.items = load_items()
    H.marks = load_marks()
    H.key = args.key
    n_bad = sum(1 for v in H.marks.values() if v == "bad")
    print(f"[load] {len(H.items)} crops, {len(H.marks)} reviewed ({n_bad} bad)")
    srv = ThreadingHTTPServer((args.host, args.port), H)
    print(f"[serve] http://{args.host}:{args.port}  (Ctrl+C to stop)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
