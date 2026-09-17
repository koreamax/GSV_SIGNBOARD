#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""의미 태깅(name/amenity) GT 검수 UI — T6 실험의 정답 데이터를 만듭니다.

build_tagging_draft.py가 만든 규칙 기반 초안을 사람이 확인·수정합니다.
초안이 약한 항목(태그 미매칭 등)이 큐 앞쪽에 오도록 확신도 오름차순이 기본.

  .venv/Scripts/python.exe review_tagging.py           # -> http://localhost:8125
  .venv/Scripts/python.exe review_tagging.py --export  # 검수본 -> tagging_gt.csv

**태그가 비는 경우를 구분해서 기록합니다.** 빈 값을 그대로 정답으로 넣으면
"태그가 없음"과 "태그를 못 정함"이 섞여 평가가 오염되므로 verdict로 나눕니다:

  ok        상호명 + 태그 확정        → name·tag 평가 both
  name_only 상호명은 알지만 업종 불명  → name만 평가, tag 평가에서 제외
            (예: `짚동가리쌩주`, `Saboten` — 간판만 보고는 업종 판별 불가)
  non       애초에 상호가 아님         → 두 평가 모두 제외
            (도로표지·임대공고·건물명·광고 배너 등)
  skip      보류(나중에 다시)

판정은 artifacts/gt/tagging_marks.csv 에 append-only로 즉시 저장(마지막 값이 유효).
단축키: Enter 저장·다음 / T 태그불명 / N 상호아님 / S 보류 / ← 이전
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parents[1]
GT = HERE / "artifacts" / "gt"
DRAFT = GT / "tagging_gt_draft.csv"
MARKS = GT / "tagging_marks.csv"
FINAL = GT / "tagging_gt.csv"
CROP = GT / "crop"

# 검수 UI 상단 퀵버튼 (자주 쓰는 순). 그 외는 자유 입력.
QUICK_TAGS = [
    "amenity=restaurant", "amenity=cafe", "amenity=fast_food", "amenity=bar",
    "shop=convenience", "shop=clothes", "shop=hairdresser", "shop=beauty",
    "amenity=pharmacy", "amenity=clinic", "amenity=bank", "office=estate_agent",
    "shop=supermarket", "shop=bakery", "amenity=school", "leisure=fitness_centre",
    "shop=mobile_phone", "shop=car_repair", "office=company", "tourism=hotel",
]


def load_draft() -> list[dict]:
    rows = []
    with DRAFT.open(encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f)):
            r["id"] = i
            r["confidence"] = int(r.get("confidence") or 0)
            r["lines"] = [l.strip() for l in (r["gt_text"] or "").replace("\\n", "\n").split("\n")
                          if l.strip() and l.strip() != "###"]
            rows.append(r)
    return rows


def load_marks() -> dict[str, dict]:
    marks: dict[str, dict] = {}
    if MARKS.exists():
        with MARKS.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                marks[r["image_name"]] = r          # 마지막 값이 유효
    return marks


def append_mark(rec: dict) -> None:
    new = not MARKS.exists()
    with MARKS.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["image_name", "region", "name", "tag", "verdict", "ts"])
        w.writerow([rec["image_name"], rec.get("region", ""), rec.get("name", ""),
                    rec.get("tag", ""), rec.get("verdict", "ok"),
                    time.strftime("%Y-%m-%d %H:%M:%S")])


def export() -> None:
    """name 평가셋과 tag 평가셋의 유효 범위가 다르므로 verdict를 함께 내보냅니다."""
    marks = load_marks()
    kept = [m for m in marks.values() if m["verdict"] in ("ok", "name_only")]
    with FINAL.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_name", "region", "name", "tag", "verdict",
                    "eval_name", "eval_tag"])
        for m in sorted(kept, key=lambda x: x["image_name"]):
            ok = m["verdict"] == "ok"
            w.writerow([m["image_name"], m["region"], m["name"],
                        m["tag"] if ok else "unknown", m["verdict"],
                        1, 1 if ok else 0])
    n_ok = sum(1 for m in kept if m["verdict"] == "ok")
    n_name = len(kept) - n_ok
    n_non = sum(1 for m in marks.values() if m["verdict"] == "non")
    n_skip = sum(1 for m in marks.values() if m["verdict"] == "skip")
    print(f"[EXPORT] -> {FINAL}")
    print(f"  name 평가 대상 : {len(kept)}건 (태그확정 {n_ok} + 업종불명 {n_name})")
    print(f"  tag  평가 대상 : {n_ok}건")
    print(f"  제외           : 상호아님 {n_non} · 보류 {n_skip}")
    if n_ok:
        from collections import Counter
        top = Counter(m["tag"] for m in kept if m["verdict"] == "ok").most_common(8)
        print("  태그 분포: " + ", ".join(f"{t}×{c}" for t, c in top))


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>의미 태깅 검수</title><style>
body{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:0;display:flex;height:100vh}
#side{width:300px;min-width:300px;overflow-y:auto;border-right:1px solid #333;padding:10px}
#main{flex:1;overflow-y:auto;padding:18px 26px}
#img{max-width:100%;max-height:38vh;background:#fff;border:2px solid #555}
.row{padding:4px 6px;border-bottom:1px solid #222;cursor:pointer;font-size:.84em;display:flex;gap:6px}
.row.cur{background:#274a63}.done{color:#5fd35f}.nameonly{color:#ffd27f}
.non{color:#ff9f6b}.pend{color:#888}
#nameonly{background:#8a6d1f;color:#fff}
h3{margin:14px 0 6px;color:#8fc7ff;font-size:.95em}
.line{display:inline-block;background:#222;border:1px solid #444;border-radius:6px;
 padding:4px 10px;margin:3px 4px 3px 0;cursor:pointer;font-size:1.05em}
.line:hover{background:#2a5b8f}
input[type=text]{background:#1b1b1b;color:#fff;border:1px solid #555;border-radius:6px;
 padding:8px 10px;font-size:1.15em;width:96%}
.tag{display:inline-block;background:#222;border:1px solid #444;border-radius:6px;
 padding:4px 9px;margin:3px 4px 3px 0;cursor:pointer;font-size:.9em}
.tag.sel{background:#2d7a2d;border-color:#3fa33f}
.btn{font-size:1.05em;padding:8px 18px;margin:10px 6px 0 0;border-radius:8px;border:0;cursor:pointer}
#save{background:#2d7a2d;color:#fff}#non{background:#a35a26;color:#fff}#skip{background:#555;color:#fff}
#stats{position:sticky;top:0;background:#111;padding-bottom:6px;border-bottom:1px solid #333}
#meta{color:#9a9;font-size:.9em}kbd{background:#333;padding:1px 6px;border-radius:4px}
</style></head><body>
<div id="side"><div id="stats"></div><div id="list"></div></div>
<div id="main">
 <div id="meta"></div>
 <img id="img">
 <h3>간판 텍스트 (클릭하면 상호명에 채워집니다)</h3><div id="lines"></div>
 <h3>상호명 (name)</h3><input type="text" id="name" autocomplete="off">
 <h3>태그 (OSM key=value) <span style="color:#888;font-size:.85em">— 직접 입력 가능</span></h3>
 <input type="text" id="tag" autocomplete="off">
 <div id="tags"></div>
 <div>
  <button class="btn" id="save">✓ 저장·다음 (Enter)</button>
  <button class="btn" id="nameonly">태그 불명 (T)</button>
  <button class="btn" id="non">상호 아님 (N)</button>
  <button class="btn" id="skip">보류 (S)</button>
 </div>
 <div id="warn" style="color:#ff9f6b;margin-top:8px;min-height:1.2em"></div>
 <div style="color:#888;margin-top:4px">
  <kbd>Enter</kbd> 저장·다음 <kbd>T</kbd> 태그불명 <kbd>N</kbd> 상호아님 <kbd>S</kbd> 보류 <kbd>←</kbd> 이전
  &nbsp;|&nbsp; 정렬:
  <select id="order"><option value="conf">확신도 낮은 순</option>
   <option value="seq">순서대로</option><option value="region">지역별</option></select>
  <select id="filter"><option value="todo">미검수만</option>
   <option value="all">전체</option><option value="done">검수완료</option></select>
  <button class="btn" style="background:#333;color:#eee;padding:4px 12px" id="reload">적용</button>
 </div>
</div><script>
let Q=[],idx=0;
const $=id=>document.getElementById(id);
async function fetchQ(){
 const r=await fetch(`/api/items?order=${$("order").value}&filter=${$("filter").value}`);
 Q=await r.json();idx=0;render();}
function render(){
 fetch("/api/stats").then(r=>r.json()).then(s=>{$("stats").textContent=
  `전체 ${s.total} | 확정 ${s.ok} · 태그불명 ${s.name_only} · 상호아님 ${s.non} `+
  `· 보류 ${s.skip} | 남음 ${s.todo}`});
 const L=$("list");L.innerHTML="";
 Q.slice(0,400).forEach((q,i)=>{const d=document.createElement("div");
  d.className="row"+(i===idx?" cur":"");
  const sym={ok:"✓",name_only:"~",non:"—",skip:"…"}[q.mark]||"·";
  const c={ok:"done",name_only:"nameonly",non:"non"}[q.mark]||"pend";
  d.innerHTML=`<span class="${c}">${sym}</span>`+
   `<span>[${q.confidence}]</span><span>${(q.name||q.draft_name||"?").slice(0,16)}</span>`;
  d.onclick=()=>{idx=i;render()};L.appendChild(d);});
 const it=Q[idx];
 if(!it){$("meta").textContent="큐 끝!";$("img").src="";$("lines").innerHTML="";return;}
 $("meta").textContent=`${idx+1}/${Q.length}  ${it.image_name}  (${it.region})`;
 $("img").src="/img/"+encodeURIComponent(it.region+"/"+it.image_name+".jpg");
 $("lines").innerHTML=it.lines.map(l=>`<span class="line">${l.replace(/</g,"&lt;")}</span>`).join("");
 [...document.querySelectorAll("#lines .line")].forEach(el=>el.onclick=()=>{
   $("name").value = $("name").value ? $("name").value+" "+el.textContent : el.textContent;});
 $("name").value=it.name||it.draft_name||"";
 $("tag").value=it.tag||it.draft_tag||"";
 paintTags();
 const cur=document.querySelector(".row.cur");if(cur)cur.scrollIntoView({block:"nearest"});}
function paintTags(){
 fetch("/api/tags").then(r=>r.json()).then(ts=>{
  $("tags").innerHTML=ts.map(t=>`<span class="tag${t===$("tag").value?" sel":""}">${t}</span>`).join("");
  [...document.querySelectorAll("#tags .tag")].forEach(el=>el.onclick=()=>{
    $("tag").value=el.textContent;paintTags();});});}
$("tag").addEventListener("input",paintTags);
async function mark(verdict){
 const it=Q[idx];if(!it)return;
 const name=$("name").value.trim(), tag=$("tag").value.trim();
 // 빈 값이 정답으로 새어들어가면 name/tag 평가가 오염되므로 여기서 막는다
 if(verdict==="ok"&&!tag){
   $("warn").textContent="태그가 비어 있습니다 — 업종을 알 수 없으면 [태그 불명 (T)], "+
     "상호가 아니면 [상호 아님 (N)]을 눌러주세요.";
   $("tag").focus();return;}
 if((verdict==="ok"||verdict==="name_only")&&!name){
   $("warn").textContent="상호명이 비어 있습니다 — 읽을 수 없으면 [상호 아님 (N)] 또는 [보류 (S)].";
   $("name").focus();return;}
 $("warn").textContent="";
 const rec={image_name:it.image_name,region:it.region,name:name,
            tag:(verdict==="ok"?tag:""),verdict:verdict};
 it.mark=verdict;it.name=rec.name;it.tag=rec.tag;
 await fetch("/api/mark",{method:"POST",body:JSON.stringify(rec)});
 idx=Math.min(idx+1,Q.length);render();}
$("save").onclick=()=>mark("ok");$("nameonly").onclick=()=>mark("name_only");
$("non").onclick=()=>mark("non");$("skip").onclick=()=>mark("skip");
$("reload").onclick=fetchQ;
document.addEventListener("keydown",e=>{
 const typing=["INPUT","SELECT"].includes(e.target.tagName);
 if(e.key==="Enter"){e.preventDefault();mark("ok");return;}
 if(typing)return;
 if(e.key==="n"||e.key==="N")mark("non");
 else if(e.key==="t"||e.key==="T")mark("name_only");
 else if(e.key==="s"||e.key==="S")mark("skip");
 else if(e.key==="ArrowLeft"){idx=Math.max(0,idx-1);render();}});
fetchQ();
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    rows: list[dict] = []
    marks: dict[str, dict] = {}

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj):
        self._send(200, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def log_message(self, *a):
        pass

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        try:
            if u.path == "/":
                self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif u.path == "/api/tags":
                self._json(QUICK_TAGS)
            elif u.path == "/api/items":
                order = q.get("order", ["conf"])[0]
                filt = q.get("filter", ["todo"])[0]
                items = []
                for r in self.rows:
                    m = self.marks.get(r["image_name"])
                    verdict = m["verdict"] if m else ""
                    if filt == "todo" and verdict:
                        continue
                    if filt == "done" and verdict != "ok":
                        continue
                    items.append({**{k: r[k] for k in
                                     ("image_name", "region", "draft_name",
                                      "draft_tag", "confidence", "lines")},
                                  "id": r["id"], "mark": verdict,
                                  "name": m["name"] if m else "",
                                  "tag": m["tag"] if m else ""})
                if order == "conf":
                    items.sort(key=lambda x: (x["confidence"], x["image_name"]))
                elif order == "region":
                    items.sort(key=lambda x: (x["region"], x["image_name"]))
                else:
                    items.sort(key=lambda x: x["id"])
                self._json(items)
            elif u.path == "/api/stats":
                vs = [m["verdict"] for m in self.marks.values()]
                self._json({"total": len(self.rows), "ok": vs.count("ok"),
                            "name_only": vs.count("name_only"),
                            "non": vs.count("non"), "skip": vs.count("skip"),
                            "todo": len(self.rows) - len(vs)})
            elif u.path.startswith("/img/"):
                rel = urllib.parse.unquote(u.path[len("/img/"):])
                p = (CROP / rel).resolve()
                if CROP.resolve() not in p.parents or not p.exists():
                    self._send(404, b"no image", "text/plain")
                    return
                self._send(200, p.read_bytes(), "image/jpeg")
            else:
                self._send(404, b"nope", "text/plain")
        except Exception as exc:
            self._send(500, f"{type(exc).__name__}: {exc}".encode("utf-8"), "text/plain")

    def do_POST(self):
        if self.path == "/api/mark":
            n = int(self.headers.get("Content-Length", 0))
            rec = json.loads(self.rfile.read(n).decode("utf-8"))
            append_mark(rec)
            self.marks[rec["image_name"]] = {
                "image_name": rec["image_name"], "region": rec.get("region", ""),
                "name": rec.get("name", ""), "tag": rec.get("tag", ""),
                "verdict": rec.get("verdict", "ok")}
            self._json({"ok": True})
        else:
            self._send(404, b"nope", "text/plain")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8125)
    ap.add_argument("--export", action="store_true")
    args = ap.parse_args()
    if args.export:
        export()
        return
    if not DRAFT.exists():
        sys.exit(f"[ABORT] 초안이 없습니다: {DRAFT}\n  먼저: python build_tagging_draft.py")
    H.rows = load_draft()
    H.marks = load_marks()
    done = len(H.marks)
    print(f"[SERVE] http://localhost:{args.port}   ({len(H.rows)}건, 검수완료 {done}건)")
    print("        확신도 낮은 항목부터 표시됩니다. 판정은 즉시 저장됩니다.")
    ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
