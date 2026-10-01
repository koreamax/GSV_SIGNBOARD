#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""태깅 정답 재검토 도구 (v2 기준, docs/tagging_guideline.md) — 로컬 웹앱.

    .venv/Scripts/python.exe data/review_tag_v2.py
    → http://127.0.0.1:8779

기본 화면(/)은 기준 v2 를 256건 전부에 적용해 고른 **의심 항목만**(`tagging_suspects_v2.json`, 이유·제안 태그
포함) 보여주고, 전체를 하나씩 보려면 /all 로 들어갑니다.

대상은 기준 문서 1~3절 업종군(convenience · supermarket · hairdresser · beauty · restaurant · fast_food · cafe · bar ·
bakery)에 속한 eval_tag=1 정답 **전부**입니다. 화면에는 간판 crop · 상호명 · 현재 태그만 나오고 모델 예측은 나오지
않습니다(평가 오염 방지). 결정은 바로 `artifacts/gt/tagging_review_v2.json` 에 저장됩니다.

    {"<image_name>": {"tag": "<v2 태그>" | "", "eval_tag": "1"|"0", "old": "<v1 태그>", "note": "...",
                      "updated": "..."}}

확정 반영은 `--apply` : tagging_gt.csv 를 tagging_gt_v1.csv 로 보존하고 결정을 덮어씁니다.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import threading
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
GT = HERE / "artifacts" / "gt"
CSV = GT / "tagging_gt.csv"
OUT = GT / "tagging_review_v2.json"
SUSPECTS = GT / "tagging_suspects_v2.json"   # 기준 v2 를 256건에 적용해 고른 의심 항목 + 이유 + 제안 태그
CROP = GT / "crop"
sys.path.insert(0, str(HERE / "vlm"))
import eval_vlm_tagging as T  # noqa: E402  (canon — 채점과 같은 정규화)

GROUPS = {"shop=convenience", "shop=supermarket", "shop=hairdresser", "shop=beauty", "amenity=restaurant",
          "amenity=fast_food", "amenity=cafe", "amenity=bar", "shop=bakery"}
_lock = threading.Lock()


def load_rows():
    rows = list(csv.DictReader(CSV.open(encoding="utf-8")))
    vocab = sorted({T.canon(r["tag"]) for r in rows if r["eval_tag"] == "1"})
    items = [dict(id=r["image_name"], region=r["region"], name=r["name"], tag=T.canon(r["tag"]))
             for r in rows if r["eval_tag"] == "1" and T.canon(r["tag"]) in GROUPS]
    items.sort(key=lambda d: (d["tag"], d["region"], d["id"]))
    return rows, vocab, items


def read_out():
    return json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}


PAGE = r"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8"><title>태깅 정답 재검토 v2</title>
<style>
:root{--bg:#15181e;--panel:#1d222b;--line:#2d3440;--fg:#e8ebf1;--dim:#9aa3b1;--acc:#5aa9ff;--ok:#5ad18b;--chg:#f2b544}
*{box-sizing:border-box}html,body{height:100%;margin:0}
body{background:var(--bg);color:var(--fg);font:14px/1.5 "Malgun Gothic",system-ui,sans-serif;display:grid;grid-template-columns:280px 1fr 360px}
aside{background:var(--panel);border-right:1px solid var(--line);overflow-y:auto}
aside h1{font-size:14px;margin:0;padding:12px 14px;border-bottom:1px solid var(--line)}
#prog{padding:10px 14px;border-bottom:1px solid var(--line);color:var(--dim)}
.it{padding:5px 14px;cursor:pointer;display:flex;justify-content:space-between;gap:8px;border-left:3px solid transparent;font-size:12.5px}
.it:hover{background:#252b36}.it.cur{background:#28344a;border-left-color:var(--acc)}
.it .t{color:var(--dim)}.it.done .t{color:var(--ok)}.it.chg .t{color:var(--chg);font-weight:700}
main{display:flex;flex-direction:column;min-width:0;padding:18px 24px;gap:14px;overflow-y:auto}
#img{max-width:100%;max-height:48vh;object-fit:contain;background:#000;border-radius:6px;align-self:flex-start}
.meta{display:grid;grid-template-columns:90px 1fr;gap:6px 12px}.meta span{color:var(--dim)}
select,input{background:#232933;color:var(--fg);border:1px solid var(--line);border-radius:5px;padding:6px 8px;font:inherit}
select{min-width:280px}input{width:100%}
.btns{display:flex;gap:8px;flex-wrap:wrap}
button{background:#262d39;color:var(--fg);border:1px solid var(--line);border-radius:5px;padding:7px 14px;cursor:pointer;font:inherit}
button.p{background:var(--acc);border-color:var(--acc);color:#0b1220;font-weight:700}
#st{color:var(--dim)}#st.chg{color:var(--chg);font-weight:700}#st.ok{color:var(--ok)}
.guide{background:var(--panel);border-left:1px solid var(--line);overflow-y:auto;padding:12px 16px;font-size:12.5px}
.guide h2{font-size:13px;margin:14px 0 4px;color:var(--acc)}.guide p{margin:0 0 6px;color:var(--dim)}.guide b{color:var(--fg)}
kbd{background:#2b313c;border:1px solid #3a424f;border-radius:4px;padding:0 5px;font-size:11px}
</style></head><body>
<aside><h1>태깅 정답 재검토 v2</h1><div id="prog"></div><div id="list"></div></aside>
<main>
  <div id="idx" style="color:var(--dim)"></div>
  <img id="img" alt="">
  <div class="meta"><span>이미지</span><b id="iid"></b><span>상호명</span><b id="nm"></b><span>현재 태그</span><b id="old"></b></div>
  <div class="meta"><span>v2 태그</span><select id="sel"></select><span>메모</span><input id="note" placeholder="바꾼 이유 (선택)"></div>
  <div class="btns"><button class="p" id="bKeep">현재 태그 유지 <kbd>Enter</kbd></button>
    <button id="bSave">선택한 태그로 저장 <kbd>S</kbd></button><button id="bEx">평가 제외 <kbd>X</kbd></button>
    <button id="bPrev">이전 <kbd>←</kbd></button><button id="bNext">다음 <kbd>→</kbd></button></div>
  <div id="st"></div>
</main>
<div class="guide">
<h2>식료품 (1절)</h2><p><b>convenience</b>: Deli, Grocery, Bodega, Mini Market, Candy &amp; Grocery, 99¢, 편의점 체인. OSM 은 bodega·corner store·mini-mart 를 convenience 로 봄.</p>
<p><b>supermarket</b>: Supermarket, Food Market 체인, 대형마트, 여러 칸 쓰는 대형 식료품점 ("카트를 쓸 만큼 큰"). "마트/Market" 만 있으면 한 칸=convenience, 두 칸 이상=supermarket.</p>
<h2>미용 (2절)</h2><p><b>hairdresser</b>: Hair, Barber, 미용실, 헤어, 이용원. "Beauty Salon" 만 있고 단서 없으면 hairdresser.</p>
<p><b>beauty</b>: Nail, 네일, Spa, 피부, 에스테틱, 왁싱, 속눈썹, 화장품·Beauty Supply.</p>
<h2>음식 (3절)</h2><p><b>restaurant</b>: 테이블에서 정식 식사 (한식·국밥·고기·일식·중식당·양식).</p>
<p><b>fast_food</b>: 카운터 선결제·포장 — 버거, 피자(조각·포장), 샌드위치, 토스트, 도넛, 부리토, 중식 Take-out, 치킨 포장·배달 전문, Dunkin'.</p>
<p><b>cafe</b>: 커피, 디저트, 주스·스무디 바(술 없음).</p>
<p><b>bar</b>: 호프, 이자카야, 포차, 주점, Bar, Pub — 술이 주 목적. 치킨호프는 더 크게 적힌 쪽.</p>
<p><b>bakery</b>: 빵집·제과점·Bakery (직접 굽는 빵). Bagel 은 카운터 위주면 fast_food.</p>
<p><b>주류 판매점</b>(Liquor, Wine &amp; Spirits): bar 아님 → <b>평가 제외</b> (OSM shop=alcohol, 어휘 밖).</p>
<h2>공통</h2><p>근거는 간판뿐. 가게가 여럿이면 가장 크게 적힌 상호. 판단 근거가 없거나 크기가 비슷한 가게가 여럿이면 <b>평가 제외</b>.</p>
</div>
<script>
let ITEMS=[],DEC={},VOCAB=[],cur=0;const $=id=>document.getElementById(id);
function prog(){const n=ITEMS.length,d=ITEMS.filter(i=>DEC[i.id]).length,c=ITEMS.filter(i=>DEC[i.id]&&(DEC[i.id].tag!==i.tag||DEC[i.id].eval_tag==='0')).length;
 $('prog').innerHTML=`${d} / ${n} 검토 · 변경 ${c}`}
function list(){$('list').innerHTML=ITEMS.map((it,i)=>{const d=DEC[it.id];const ch=d&&(d.tag!==it.tag||d.eval_tag==='0');
 return `<div class="it ${i===cur?'cur':''} ${d?'done':''} ${ch?'chg':''}" data-i="${i}"><span>${it.id.replace('__crop_','·')}</span><span class="t">${d?(d.eval_tag==='0'?'제외':d.tag.split('=')[1]):it.tag.split('=')[1]}</span></div>`}).join('');
 const e=$('list').querySelector('.cur');if(e)e.scrollIntoView({block:'nearest'})}
function show(i){cur=Math.max(0,Math.min(ITEMS.length-1,i));const it=ITEMS[cur],d=DEC[it.id];
 $('idx').textContent=`${cur+1} / ${ITEMS.length} · ${it.region}`;$('img').src=`/crop/${it.region}/${it.id}.jpg`;
 $('iid').textContent=it.id;$('nm').textContent=it.name||'(상호명 없음)';$('old').textContent=it.tag;
 $('sel').value=d&&d.eval_tag!=='0'?d.tag:it.tag;$('note').value=d?d.note||'':'';
 $('st').className=d?(d.tag!==it.tag||d.eval_tag==='0'?'chg':'ok'):'';$('st').textContent=d?(d.eval_tag==='0'?'평가 제외로 저장됨':(d.tag!==it.tag?`변경 저장됨: ${it.tag} → ${d.tag}`:'유지로 저장됨')):'아직 검토 안 함';list()}
function save(tag,ev){const it=ITEMS[cur];const body={id:it.id,tag,eval_tag:ev,old:it.tag,note:$('note').value};
 fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}).then(r=>r.json()).then(j=>{
 if(!j.ok)throw 0;DEC[it.id]=body;prog();show(cur+1)}).catch(()=>{$('st').className='chg';$('st').textContent='저장 실패 — 서버 확인'})}
$('bKeep').onclick=()=>save(ITEMS[cur].tag,'1');$('bSave').onclick=()=>save($('sel').value,'1');$('bEx').onclick=()=>save('','0');
$('bPrev').onclick=()=>show(cur-1);$('bNext').onclick=()=>show(cur+1);
$('list').onclick=e=>{const el=e.target.closest('.it');if(el)show(+el.dataset.i)};
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT'||e.target.tagName==='SELECT')return;
 if(e.key==='Enter'){e.preventDefault();$('bKeep').click()}else if(e.key==='s'||e.key==='S')$('bSave').click();
 else if(e.key==='x'||e.key==='X')$('bEx').click();else if(e.key==='ArrowRight')show(cur+1);else if(e.key==='ArrowLeft')show(cur-1)});
fetch('/api/state').then(r=>r.json()).then(s=>{ITEMS=s.items;DEC=s.dec;VOCAB=s.vocab;
 $('sel').innerHTML=VOCAB.map(v=>`<option>${v}</option>`).join('');prog();
 const first=ITEMS.findIndex(i=>!DEC[i.id]);show(first<0?0:first)});
</script></body></html>"""


SPAGE = r"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8"><title>태깅 정답 의심 항목 검토</title>
<style>
:root{--bg:#15181e;--panel:#1d222b;--line:#2d3440;--fg:#e8ebf1;--dim:#9aa3b1;--acc:#5aa9ff;--ok:#5ad18b;--chg:#f2b544;--bad:#ff7b72}
*{box-sizing:border-box}html,body{margin:0}
body{background:var(--bg);color:var(--fg);font:14px/1.6 "Malgun Gothic",system-ui,sans-serif}
header{position:sticky;top:0;z-index:2;background:var(--panel);border-bottom:1px solid var(--line);padding:10px 20px;display:flex;gap:18px;align-items:center;flex-wrap:wrap}
header h1{font-size:15px;margin:0}#prog{color:var(--dim)}
.filters{display:flex;gap:6px;margin-left:auto}.filters button{padding:4px 10px}
main{max-width:1180px;margin:0 auto;padding:18px 20px 60px;display:flex;flex-direction:column;gap:14px}
.card{display:grid;grid-template-columns:340px minmax(0,1fr);gap:18px;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}
.card.done{border-color:#2f5b45}.card img{width:100%;max-height:200px;object-fit:contain;background:#000;border-radius:6px}
.card .side{display:flex;flex-direction:column;gap:8px;min-width:0}
.top{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.id{font-family:Consolas,monospace;color:var(--dim);font-size:12.5px}
.chip{font-size:12px;padding:1px 9px;border-radius:999px;font-weight:700}
.high{background:#3b1f1f;color:var(--bad)}.mid{background:#3a2e12;color:var(--chg)}.low{background:#262d39;color:var(--dim)}
.name{font-size:16px;font-weight:700}
.tags{display:grid;grid-template-columns:70px 1fr;gap:4px 10px;font-size:13.5px}.tags span{color:var(--dim)}
code{font-family:Consolas,monospace;background:#262d39;padding:1px 6px;border-radius:4px}
code.sug{background:#173327;color:var(--ok)}code.old{text-decoration:line-through;color:var(--dim)}
.reason{background:#11151b;border-left:3px solid var(--acc);padding:8px 12px;border-radius:0 6px 6px 0;font-size:13.5px}
.btns{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
button{background:#262d39;color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:6px 12px;cursor:pointer;font:inherit;font-size:13px}
button.p{background:var(--ok);border-color:var(--ok);color:#08130d;font-weight:700}button:focus-visible{outline:2px solid var(--acc)}
select{background:#232933;color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:5px 8px;font:inherit;font-size:13px}
.st{font-size:13px;color:var(--dim)}.st.ok{color:var(--ok);font-weight:700}
.legend{color:var(--dim);font-size:12.5px}
@media(max-width:820px){.card{grid-template-columns:1fr}}
</style></head><body>
<header><h1>태깅 정답 의심 항목 검토 (기준 v2)</h1><span id="prog"></span>
<span class="legend">확신: <span class="chip high">높음</span> <span class="chip mid">중간</span> <span class="chip low">낮음</span> · 모델 예측은 판단에 쓰지 않았습니다 · 기준: docs/tagging_guideline.md</span>
<div class="filters"><button data-f="all">전체</button><button data-f="todo">남은 것</button><button data-f="done">완료</button></div></header>
<main id="list"></main>
<script>
let S=[],DEC={},VOCAB=[],F='all';const $=id=>document.getElementById(id);
const EX='__EXCLUDE__',show=t=>t===EX?'평가 제외 (eval_tag=0)':t,conf={high:'높음',mid:'중간',low:'낮음'};
function decText(d,it){if(!d)return'';if(d.eval_tag==='0')return'저장됨: 평가 제외';return d.tag===it.old?'저장됨: 현재 태그 유지':`저장됨: ${it.old} → ${d.tag}`}
function render(){const n=S.length,d=S.filter(i=>DEC[i.id]).length;$('prog').textContent=`${d} / ${n} 결정`;
 $('list').innerHTML=S.filter(i=>F==='all'||(F==='done')===!!DEC[i.id]).map(it=>{const d=DEC[it.id];
 const opt=VOCAB.map(v=>`<option ${v===it.old?'selected':''}>${v}</option>`).join('');
 return `<article class="card ${d?'done':''}" id="c-${it.id}"><img src="/crop/${it.region}/${it.id}.jpg" alt="${it.id}" loading="lazy">
 <div class="side"><div class="top"><span class="name">${it.name||'(상호명 없음)'}</span><span class="chip ${it.conf}">확신 ${conf[it.conf]}</span><span class="id">${it.id}</span></div>
 <div class="tags"><span>현재</span><div><code class="${d?'':'old'}">${it.old}</code></div>
 <span>제안</span><div><code class="sug">${show(it.suggest)}</code>${it.alt?` &nbsp;또는&nbsp; <code>${show(it.alt)}</code>`:''}</div></div>
 <div class="reason">${it.reason}</div>
 <div class="btns"><button class="p" data-a="sug" data-id="${it.id}">제안대로</button>${it.alt?`<button data-a="alt" data-id="${it.id}">대안으로</button>`:''}
 <button data-a="keep" data-id="${it.id}">현재 유지</button><button data-a="ex" data-id="${it.id}">평가 제외</button>
 <select data-id="${it.id}">${opt}</select><button data-a="pick" data-id="${it.id}">고른 태그로</button>
 <span class="st ${d?'ok':''}">${decText(d,it)}</span></div></div></article>`}).join('')}
function save(it,tag){const ev=tag===EX?'0':'1';const body={id:it.id,tag:tag===EX?'':tag,eval_tag:ev,old:it.old,note:'의심목록 v2'};
 fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}).then(r=>r.json()).then(j=>{if(!j.ok)throw 0;DEC[it.id]=body;render();
 const nx=S.find(i=>!DEC[i.id]);if(nx){const el=document.getElementById('c-'+nx.id);if(el)el.scrollIntoView({block:'center'})}}).catch(()=>{const el=document.querySelector(`#c-${it.id} .st`);if(el){el.textContent='저장 실패 — 서버가 켜져 있는지 확인하세요';el.className='st'}})}
$('list').onclick=e=>{const b=e.target.closest('button[data-a]');if(!b)return;const it=S.find(i=>i.id===b.dataset.id);
 const a=b.dataset.a;if(a==='sug')save(it,it.suggest);else if(a==='alt')save(it,it.alt);else if(a==='keep')save(it,it.old);else if(a==='ex')save(it,EX);
 else{const sel=document.querySelector(`select[data-id="${it.id}"]`);save(it,sel.value)}};
document.querySelectorAll('.filters button').forEach(b=>b.onclick=()=>{F=b.dataset.f;render()});
fetch('/api/suspects').then(r=>r.json()).then(s=>{S=s.items;DEC=s.dec;VOCAB=s.vocab;render()});
</script></body></html>
"""


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=86400" if ctype.startswith("image") else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p in ("/", "/index.html"):          # 기본: 의심 항목만 (tagging_suspects_v2.json)
            return self._send(200, SPAGE.encode("utf-8"), "text/html; charset=utf-8")
        if p == "/all":                        # 전체 256건 하나씩 보기
            return self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        if p == "/api/suspects":
            _, vocab, items = load_rows()
            by = {it["id"]: it for it in items}
            sus = json.loads(SUSPECTS.read_text(encoding="utf-8"))
            out = [dict(id=k, region=by[k]["region"], name=by[k]["name"], **v) for k, v in sus.items()]
            out.sort(key=lambda d: ({"high": 0, "mid": 1, "low": 2}[d["conf"]], d["old"], d["id"]))
            with _lock:
                dec = read_out()
            return self._send(200, json.dumps({"items": out, "dec": dec, "vocab": vocab},
                                              ensure_ascii=False).encode("utf-8"), "application/json")
        if p == "/api/state":
            _, vocab, items = load_rows()
            with _lock:
                dec = read_out()
            return self._send(200, json.dumps({"items": items, "dec": dec, "vocab": vocab},
                                              ensure_ascii=False).encode("utf-8"), "application/json")
        if p.startswith("/crop/"):
            parts = p.split("/")
            f = CROP / os.path.basename(parts[2]) / os.path.basename(parts[3])
            if f.exists() and f.suffix == ".jpg":
                return self._send(200, f.read_bytes(), "image/jpeg")
        return self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/save":
            return self._send(404, b"not found", "text/plain")
        d = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))).decode("utf-8"))
        with _lock:
            dec = read_out()
            dec[d["id"]] = {"tag": d.get("tag", ""), "eval_tag": "0" if d.get("eval_tag") == "0" else "1",
                            "old": d.get("old", ""), "note": d.get("note", ""),
                            "updated": datetime.now().isoformat(timespec="seconds")}
            tmp = OUT.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(dec, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, OUT)
        return self._send(200, b'{"ok": true}', "application/json")


def apply() -> None:
    rows, _, items = load_rows()
    dec = read_out()
    # 의심 항목(tagging_suspects_v2.json)은 전부 결정돼 있어야 합니다. 의심 목록에 없는 항목은 결정이
    # 없으면 현재 태그를 유지합니다(기준 v2 를 적용해 문제가 보이지 않은 것).
    sus = json.loads(SUSPECTS.read_text(encoding="utf-8")) if SUSPECTS.exists() else {}
    todo = [k for k in sus if k not in dec]
    if todo:
        raise SystemExit(f"아직 결정 안 된 의심 항목 {len(todo)}개 — 전부 결정한 뒤 반영하세요.")
    backup = GT / "tagging_gt_v1.csv"
    if not backup.exists():
        shutil.copy(CSV, backup)
    n = 0
    for r in rows:
        d = dec.get(r["image_name"])
        if not d:
            continue
        if d["eval_tag"] == "0":
            r["eval_tag"] = "0"; n += 1
        elif d["tag"] != T.canon(r["tag"]):
            r["tag"] = d["tag"]; n += 1
    with CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"반영: 변경 {n}건 · 원본 보존 {backup}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8779)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--apply", action="store_true", help="검토 결과를 tagging_gt.csv 에 반영")
    a = ap.parse_args()
    if a.apply:
        return apply()
    _, _, items = load_rows()
    url = f"http://127.0.0.1:{a.port}"
    print(f"검토 대상 {len(items)}건 · 결과 {OUT}\n주소: {url}   (Ctrl+C 로 종료)")
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
    except KeyboardInterrupt:
        print("\n종료했습니다. 결정은 저장돼 있습니다.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
