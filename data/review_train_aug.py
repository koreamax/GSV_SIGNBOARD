#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""review_train_aug.py — inspect what each OCR engine ACTUALLY saw in training.

Sibling of review_crops.py (which reviews the raw crops/labels). This app
re-runs each engine's real training-time pipeline on signboard_v3 train rows
so augmentation problems can be checked one by one:

  * TrOCR  : train_textinthewild_ocr.build_trocr_augment()  (the real code path)
             + the 384x384 square resize the ViT processor applies.
  * Paddle : DecodeImage(BGR) -> RecConAug(prob .5, ext 2, label += NO SPACE)
             -> RecAug (tia/crop/blur/hsv/jitter/noise/REVERSE 40%)
             -> final 48x320 network view. Real ppocr classes are imported from
             the venv PaddleOCR repo; if that import fails only RecConAug is
             reproduced locally (faithful port) and RecAug variants are skipped.
  * EasyOCR: no augmentation in signboard_v3.yaml (contrast_adjust 0.0) —
             shown as the trainer input: grayscale, keep-AR resize to H=64,
             right-pad to 600 (NormalizePAD).

Usage:
  .venv/Scripts/python.exe review_train_aug.py            # -> http://localhost:8124
  .venv/Scripts/python.exe review_train_aug.py --scan     # dataset audit only (no UI)
  .venv/Scripts/python.exe review_train_aug.py --selftest # render one sample to scratch, no server
  .venv/Scripts/python.exe review_train_aug.py --export   # aug_review_bad.csv from marks

Verdicts append to artifacts/ocr_training/signboard_v3/aug_review_marks.csv.
Keys: G/→ good  B/X bad  S skip  ← prev  Z undo  R re-roll augmentation.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
import sys
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parents[1]

import sys as _sys; _sys.path[:0] = [str(HERE / _d) for _d in ("pipeline", "ocr", "vlm", "str_baselines", "detection", "data")]  # 폴더 재편 후 형제 모듈 import 경로
V3 = HERE / "artifacts" / "ocr_training" / "signboard_v3"
TRAIN_TXT = V3 / "train.txt"
MANIFEST = V3 / "labels.csv"
MARKS = V3 / "aug_review_marks.csv"
SCAN_CSV = V3 / "aug_scan.csv"

PPOCR_REPO = HERE / ".venv/Lib/site-packages/paddlex/repo_manager/repos/PaddleOCR"
PPOCR_DICT = PPOCR_REPO / "ppocr/utils/dict/ppocrv5_korean_dict.txt"
EASY_CHARSET = HERE / "external/EasyOCR/trainer/all_data_signboard_v3/charset.txt"

PADDLE_MAXLEN = 25          # paddle_signboard_rec_v3.yml max_text_length
PADDLE_SHAPE = (48, 320)    # h, w
EASY_MAXLEN = 64            # signboard_v3.yaml batch_max_length
EASY_SHAPE = (64, 600)      # imgH, imgW (PAD=True, grayscale)


# ----------------------------------------------------------------------
# data
# ----------------------------------------------------------------------
def load_rows() -> list[dict]:
    meta = {}
    if MANIFEST.exists():
        with MANIFEST.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                meta[r["image_path"]] = r
    rows = []
    with TRAIN_TXT.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if "\t" not in line:
                continue
            p, text = line.rstrip("\n").split("\t", 1)
            m = meta.get(p, {})
            rows.append({"id": i, "path": p, "text": text,
                         "w": float(m.get("w") or 0), "h": float(m.get("h") or 0)})
    return rows


def load_charsets() -> tuple[set[str], set[str]]:
    pset, eset = set(), set()
    if PPOCR_DICT.exists():
        pset = {l.rstrip("\n") for l in PPOCR_DICT.open(encoding="utf-8") if l.rstrip("\n")}
        pset.add(" ")                      # use_space_char: true
    if EASY_CHARSET.exists():
        eset = set(EASY_CHARSET.read_text(encoding="utf-8"))
    return pset, eset


PADDLE_CHARS, EASY_CHARS = load_charsets()


# ----------------------------------------------------------------------
# suspicion (augmentation/pipeline-focused; raw-label issues -> review_crops.py)
# ----------------------------------------------------------------------
def suspicion(row: dict) -> tuple[int, list[str]]:
    t, score, why = row["text"], 0, []
    n = len(t)
    if n == 0:
        return 100, ["empty label"]
    if n > PADDLE_MAXLEN:
        score += 40; why.append(f"Paddle 드롭: 라벨 {n}자 > {PADDLE_MAXLEN}")
    if PADDLE_CHARS:
        miss = sorted({c for c in t if c not in PADDLE_CHARS})
        if miss:
            score += 30; why.append("Paddle 사전 밖 문자(라벨서 소실): " + "".join(miss))
    if EASY_CHARS:
        miss_e = sorted({c for c in t if c not in EASY_CHARS and not c.isspace()})
        if miss_e:
            score += 15; why.append("EasyOCR charset 밖: " + "".join(miss_e))
    if n > EASY_MAXLEN:
        score += 10; why.append(f"EasyOCR 드롭: >{EASY_MAXLEN}자")
    w, h = row["w"], row["h"]
    if w > 0 and h > 0:
        squash = (w / h * PADDLE_SHAPE[0]) / PADDLE_SHAPE[1]
        if squash > 1.5:
            score += 25; why.append(f"Paddle 320px 압착 ×{squash:.1f}")
        if h < 16:
            score += 15; why.append(f"원본 높이 {int(h)}px")
        if h < 20 or w < 20:
            why.append("tia 스킵 크기(<20px)")
    return score, why


# ----------------------------------------------------------------------
# pipelines
# ----------------------------------------------------------------------
_trocr_aug = None


def trocr_aug(img: Image.Image, seed: int) -> Image.Image:
    global _trocr_aug
    if _trocr_aug is None:
        import train_textinthewild_ocr as T     # light module import (main-guarded)
        _trocr_aug = T.build_trocr_augment()
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass
    return _trocr_aug(img)


_paddle = {"tried": False, "conaug": None, "recaug": None, "mode": "none", "err": ""}


# ---- faithful local ports of ppocr rec_img_aug internals (numpy/cv2 only).
# Needed because importing ppocr pulls paddle, whose CUDA DLLs conflict with
# torch in the same process on this machine (same reason paddle_rec_worker.py
# exists). tia_* come from ppocr's text_image_aug package, which is pure numpy
# and imported directly from the venv repo.
def _flag() -> int:
    return 1 if random.random() > 0.5000001 else -1


def _hsv_aug(img):
    import cv2
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    delta = 0.001 * random.random() * _flag()
    hsv[:, :, 2] = hsv[:, :, 2] * (1 + delta)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _jitter(img):
    w, h, _ = img.shape          # (ppocr swaps names; kept verbatim)
    if h > 10 and w > 10:
        thres = min(w, h)
        s = int(random.random() * thres * 0.01)
        src_img = img.copy()
        for i in range(s):
            img[i:, i:, :] = src_img[: w - i, : h - i, :]
    return img


def _gauss_noise(image, mean=0, var=0.1):
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    return np.uint8(np.clip(image + 0.5 * noise, 0, 255))


def _get_crop(image):
    h = image.shape[0]
    top_crop = min(int(random.randint(1, 8)), h - 1)
    crop_img = image.copy()
    if random.randint(0, 1):
        return crop_img[top_crop:h, :, :]
    return crop_img[0: h - top_crop, :, :]


class _LocalConAug:              # faithful port of ppocr RecConAug (prob handled by caller)
    max_wh_ratio = PADDLE_SHAPE[1] / PADDLE_SHAPE[0]

    def __call__(self, data):
        import cv2
        for ext in data["ext_data"]:
            if len(data["label"]) + len(ext["label"]) > PADDLE_MAXLEN:
                break
            ratio = (data["image"].shape[1] / data["image"].shape[0]
                     + ext["image"].shape[1] / ext["image"].shape[0])
            if ratio > self.max_wh_ratio:
                break
            h = PADDLE_SHAPE[0]
            a = cv2.resize(data["image"], (round(data["image"].shape[1] / data["image"].shape[0] * h), h))
            b = cv2.resize(ext["image"], (round(ext["image"].shape[1] / ext["image"].shape[0] * h), h))
            data["image"] = np.concatenate([a, b], axis=1)
            data["label"] += ext["label"]
        data.pop("ext_data")
        return data


class _LocalRecAug:              # faithful port of ppocr RecAug (all probs 0.4)
    def __init__(self, tia):
        import cv2
        self.tia = tia           # (tia_distort, tia_stretch, tia_perspective) or None
        self.fil = cv2.getGaussianKernel(ksize=5, sigma=1, ktype=cv2.CV_32F)

    def __call__(self, data):
        import cv2
        img = data["image"]
        h, w, _ = img.shape
        if self.tia and random.random() <= 0.4 and h >= 20 and w >= 20:
            dis, st, per = self.tia
            img = dis(img, random.randint(3, 6))
            img = st(img, random.randint(3, 6))
            img = per(img)
        h, w, _ = img.shape
        if random.random() <= 0.4 and h >= 20 and w >= 20:
            img = _get_crop(img)
        if random.random() <= 0.4:
            img = cv2.sepFilter2D(img, -1, self.fil, self.fil)
        if random.random() <= 0.4:
            img = _hsv_aug(img)
        if random.random() <= 0.4:
            img = _jitter(img)
        if random.random() <= 0.4:
            img = _gauss_noise(img)
        if random.random() <= 0.4:
            img = 255 - img      # reverse (색 반전)
        data["image"] = img
        return data


def _paddle_ops():
    """Real ppocr ops if importable; otherwise faithful local ports."""
    if _paddle["tried"]:
        return _paddle
    _paddle["tried"] = True
    try:
        sys.path.insert(0, str(PPOCR_REPO))
        from ppocr.data.imaug.rec_img_aug import RecAug, RecConAug   # noqa
        _paddle["conaug"] = RecConAug(prob=1.0, image_shape=(*PADDLE_SHAPE, 3),
                                      max_text_length=PADDLE_MAXLEN, ext_data_num=2)
        _paddle["recaug"] = RecAug()
        _paddle["mode"] = "real"
    except Exception as exc:                                        # paddle DLL conflict etc.
        _paddle["err"] = f"{type(exc).__name__}"
        tia = None
        try:
            sys.path.insert(0, str(PPOCR_REPO / "ppocr" / "data" / "imaug"))
            from text_image_aug import tia_distort, tia_stretch, tia_perspective  # noqa
            tia = (tia_distort, tia_stretch, tia_perspective)
        except Exception:
            pass
        _paddle["conaug"] = _LocalConAug()
        _paddle["recaug"] = _LocalRecAug(tia)
        _paddle["mode"] = "port" if tia else "port-notia"
    return _paddle


def paddle_final_view(bgr: np.ndarray) -> np.ndarray:
    """RecResizeImg for (3,48,320): keep-AR resize to h=48, right-pad zeros to 320."""
    import cv2
    h, w = bgr.shape[:2]
    nw = min(PADDLE_SHAPE[1], max(1, round(PADDLE_SHAPE[0] / h * w)))
    r = cv2.resize(bgr, (nw, PADDLE_SHAPE[0]))
    out = np.zeros((PADDLE_SHAPE[0], PADDLE_SHAPE[1], 3), dtype=r.dtype)
    out[:, :nw] = r
    return out


def easy_view(img: Image.Image) -> Image.Image:
    """EasyOCR trainer input: grayscale, keep-AR resize H=64, right-pad black to 600."""
    g = img.convert("L")
    w, h = g.size
    nw = min(EASY_SHAPE[1], max(1, round(EASY_SHAPE[0] / h * w)))
    g = g.resize((nw, EASY_SHAPE[0]), Image.BICUBIC)
    canvas = Image.new("L", (EASY_SHAPE[1], EASY_SHAPE[0]), 0)
    canvas.paste(g, (0, 0))
    return canvas


def jpeg(img: Image.Image, q: int = 90) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=q)
    return buf.getvalue()


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(bgr[:, :, ::-1])


# ----------------------------------------------------------------------
# per-sample render bundle
# ----------------------------------------------------------------------
def paddle_roll(row: dict, rows: list[dict], seed: int) -> dict:
    """One RecConAug(+RecAug) roll exactly like training (forced concat for display)."""
    import cv2
    ops = _paddle_ops()
    rng = random.Random(seed * 100003 + row["id"])   # per-sample partner draw
    img = cv2.imread(str(V3 / row["path"]))
    if img is None:
        return {"error": "이미지 로드 실패"}, None
    partners = [rows[rng.randrange(len(rows))] for _ in range(2)]
    ext = []
    for p in partners:
        pi = cv2.imread(str(V3 / p["path"]))
        if pi is not None:
            ext.append({"image": pi, "label": p["text"]})
    data = {"image": img, "label": row["text"], "ext_data": ext}
    random.seed(seed)
    data = ops["conaug"](data)
    con_bgr, con_label = data["image"], data["label"]
    out = {
        "partners": [p["text"] for p in partners],
        "concat_label": con_label,
        "concat_happened": con_label != row["text"],
        "dropped": len(con_label) > PADDLE_MAXLEN,
        "mode": ops["mode"],
    }
    return out, con_bgr


def render_view(row: dict, rows: list[dict], view: str, seed: int, k: int) -> bytes:
    img = Image.open(V3 / row["path"]).convert("RGB")
    if view == "orig":
        return jpeg(img)
    if view == "trocr":
        return jpeg(trocr_aug(img, seed * 101 + k))
    if view == "trocr384":
        return jpeg(trocr_aug(img, seed * 101 + k).resize((384, 384), Image.BICUBIC))
    if view == "easy":
        return jpeg(easy_view(img))
    if view in ("pcon", "pfinal", "precaug"):
        meta, con_bgr = paddle_roll(row, rows, seed)
        if con_bgr is None:
            return jpeg(img)
        if view == "pcon":
            return jpeg(bgr_to_pil(con_bgr))
        if view == "pfinal":
            return jpeg(bgr_to_pil(paddle_final_view(con_bgr)))
        ops = _paddle_ops()
        if ops["recaug"] is None:
            return jpeg(bgr_to_pil(con_bgr))
        random.seed(seed * 977 + k)
        np.random.seed((seed * 977 + k) % (2 ** 31))
        d = ops["recaug"]({"image": con_bgr.copy(), "label": ""})
        return jpeg(bgr_to_pil(d["image"]))
    raise ValueError(view)


# ----------------------------------------------------------------------
# marks / export / scan
# ----------------------------------------------------------------------
def load_marks() -> dict[str, str]:
    marks = {}
    if MARKS.exists():
        with MARKS.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                marks[r["image_path"]] = r["verdict"]
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
    bad = [(p, v) for p, v in marks.items() if v == "bad"]
    out = V3 / "aug_review_bad.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "verdict"])
        w.writerows(bad)
    print(f"[EXPORT] bad={len(bad)} -> {out}")


def scan(rows: list[dict]) -> None:
    n = len(rows)
    over25 = [r for r in rows if len(r["text"]) > PADDLE_MAXLEN]
    over64 = [r for r in rows if len(r["text"]) > EASY_MAXLEN]
    dmiss, dmiss_chars = [], {}
    for r in rows:
        miss = {c for c in r["text"] if PADDLE_CHARS and c not in PADDLE_CHARS}
        if miss:
            dmiss.append(r)
            for c in miss:
                dmiss_chars[c] = dmiss_chars.get(c, 0) + 1
    squash = [r for r in rows if r["w"] > 0 and r["h"] > 0
              and (r["w"] / r["h"] * PADDLE_SHAPE[0]) / PADDLE_SHAPE[1] > 1.5]
    tiny = [r for r in rows if 0 < r["h"] < 16]
    small_tia = [r for r in rows if 0 < r["h"] < 20 or 0 < r["w"] < 20]
    # RecConAug drop simulation: how often would concat be blocked by len/ratio?
    rng = random.Random(0)
    blocked_len = blocked_ratio = ok = 0
    for _ in range(20000):
        a, b = rows[rng.randrange(n)], rows[rng.randrange(n)]
        if len(a["text"]) + len(b["text"]) > PADDLE_MAXLEN:
            blocked_len += 1
        elif a["h"] > 0 and b["h"] > 0 and (a["w"] / a["h"] + b["w"] / b["h"]) > 320 / 48:
            blocked_ratio += 1
        else:
            ok += 1
    print(f"[SCAN] train rows: {n}")
    print(f"  Paddle 라벨>{PADDLE_MAXLEN}자 (인코딩 드롭): {len(over25)} ({len(over25)/n*100:.2f}%)")
    print(f"  Paddle 사전 밖 문자 포함 (해당 문자 라벨서 소실): {len(dmiss)} ({len(dmiss)/n*100:.2f}%)")
    if dmiss_chars:
        top = sorted(dmiss_chars.items(), key=lambda x: -x[1])[:20]
        print(f"    상위 소실 문자: {' '.join(f'{c}×{k}' for c, k in top)}")
    print(f"  EasyOCR 라벨>{EASY_MAXLEN}자: {len(over64)}")
    print(f"  Paddle 320px 압착 ×1.5 이상 (판독성 위험): {len(squash)} ({len(squash)/n*100:.2f}%)")
    print(f"  원본 높이<16px: {len(tiny)}  |  tia 스킵 크기(<20px 변): {len(small_tia)}")
    print(f"  RecConAug 시뮬(2만쌍): concat 성사 {ok/200:.1f}%  길이가드 {blocked_len/200:.1f}%  비율가드 {blocked_ratio/200:.1f}%")
    with SCAN_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "text", "score", "reasons"])
        for r in sorted(rows, key=lambda r: -suspicion(r)[0]):
            s, why = suspicion(r)
            if s > 0:
                w.writerow([r["path"], r["text"], s, " | ".join(why)])
    print(f"[SCAN] 상세 -> {SCAN_CSV}")


# ----------------------------------------------------------------------
# web app
# ----------------------------------------------------------------------
PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>train aug review</title><style>
body{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:0;display:flex;height:100vh}
#side{width:330px;min-width:330px;overflow-y:auto;border-right:1px solid #333;padding:10px}
#main{flex:1;overflow-y:auto;padding:16px 24px}
h3{margin:14px 0 6px;color:#8fc7ff;font-size:1em}
img{background:#fff;border:1px solid #555;image-rendering:pixelated;height:72px}
img.big{height:96px} img.sq{height:120px}
#label{font-size:1.7em;font-weight:700;margin:4px 0}
#meta{color:#9a9;font-size:.9em;white-space:pre-line}
#why{color:#e6b05c;margin:4px 0}
.btn{font-size:1.1em;padding:7px 20px;margin:0 5px;border-radius:8px;border:0;cursor:pointer}
#good{background:#2d7a2d;color:#fff}#bad{background:#a32626;color:#fff}#skip{background:#555;color:#fff}#roll{background:#2a5b8f;color:#fff}
.row{padding:4px 6px;border-bottom:1px solid #222;cursor:pointer;font-size:.85em;display:flex;gap:6px}
.row.cur{background:#274a63}.v-good{color:#5fd35f}.v-bad{color:#ff6b6b}.v-skip{color:#aaa}
select,button.small{background:#222;color:#eee;border:1px solid #444;padding:4px 8px;margin:2px;border-radius:4px}
#stats{position:sticky;top:0;background:#111;padding-bottom:6px;border-bottom:1px solid #333;margin-bottom:4px}
kbd{background:#333;padding:1px 5px;border-radius:4px}
.plabel{color:#ffd27f}.warn{color:#ff6b6b;font-weight:700}
.imgrow{display:flex;gap:8px;flex-wrap:wrap;align-items:flex-start}</style></head><body>
<div id="side"><div id="stats"></div><div id="list"></div><button class="small" id="more">load more…</button></div>
<div id="main">
 <div id="label"></div><div id="meta"></div><div id="why"></div>
 <div style="margin:8px 0">
  <button class="btn" id="good">✓ 정상 (G/→)</button><button class="btn" id="bad">✗ 문제 (B/X)</button>
  <button class="btn" id="skip">보류 (S)</button><button class="btn" id="roll">🎲 재추첨 (R)</button>
  <select id="order"><option value="suspicious">의심 우선</option><option value="seq">순서대로</option><option value="random">랜덤</option></select>
  <select id="filter"><option value="unreviewed">미검수만</option><option value="all">전체</option><option value="bad">문제만</option></select>
  <button class="small" id="reload">적용</button>
  <span style="color:#888">  <kbd>G</kbd>정상 <kbd>B</kbd>문제 <kbd>S</kbd>보류 <kbd>R</kbd>재추첨 <kbd>←</kbd>이전 <kbd>Z</kbd>취소</span>
 </div>
 <h3>원본</h3><div class="imgrow"><img id="orig" class="big"></div>
 <h3>TrOCR 증강 ×4 (실제 학습 코드 경로 · 회전±4° 원근 밝기 블러 저해상왕복, fill=흰색) + ViT 384² 입력</h3>
 <div class="imgrow"><img id="t0"><img id="t1"><img id="t2"><img id="t3"><img id="t384" class="sq" title="TrOCR 384x384 입력(정사각 왜곡)"></div>
 <h3>Paddle RecConAug (p=.5·최대 2개 연결 — <span class="warn">라벨 공백 없이 이어붙음</span>) → RecAug → 최종 48×320 입력</h3>
 <div id="pmeta" style="font-size:.95em"></div>
 <div class="imgrow"><img id="pcon" class="big" title="RecConAug 결과"><img id="pr0" title="RecAug roll1"><img id="pr1" title="RecAug roll2"><img id="pr2" title="RecAug roll3"><img id="pfinal" class="big" title="최종 48x320 (우측 검정 패딩)"></div>
 <h3>EasyOCR 학습 입력 (증강 없음 · 그레이 64×600 우측 패딩)</h3>
 <div class="imgrow"><img id="easy" style="height:64px;width:600px;object-fit:none;object-position:left"></div>
</div><script>
let Q=[],idx=0,shown=200,hist=[],seed=1;
const $=id=>document.getElementById(id);
async function fetchQ(){const o=$("order").value,f=$("filter").value;
 const r=await fetch(`/api/items?order=${o}&filter=${f}`);Q=await r.json();idx=0;shown=200;render();}
function render(){const it=Q[idx];
 fetch("/api/stats").then(r=>r.json()).then(s=>{$("stats").textContent=`전체 ${s.total} | 검수 ${s.reviewed} (정상 ${s.good} · 문제 ${s.bad} · 보류 ${s.skip}) | 대기열 ${Q.length}`});
 const L=$("list");L.innerHTML="";
 Q.slice(0,shown).forEach((q,i)=>{const d=document.createElement("div");d.className="row"+(i===idx?" cur":"");
  d.innerHTML=`<span class="v-${q.mark||''}">${q.mark?(q.mark==="good"?"✓":q.mark==="bad"?"✗":"…"):"·"}</span><span>[${q.score}]</span><span>${q.text.slice(0,16)}</span>`;
  d.onclick=()=>{idx=i;render()};L.appendChild(d);});
 if(!it){$("label").textContent="큐 끝!";return;}
 $("label").textContent=it.text||"(빈 라벨)";
 $("meta").textContent=`${idx+1}/${Q.length}  ${it.path}  box=${it.w}×${it.h}px  라벨 ${it.text.length}자`;
 $("why").textContent=it.why.length?("자동 감지: "+it.why.join(" · ")):"";
 const u=(v,k)=>`/img?id=${it.id}&view=${v}&seed=${seed}&k=${k||0}`;
 $("orig").src=u("orig");$("t0").src=u("trocr",0);$("t1").src=u("trocr",1);$("t2").src=u("trocr",2);$("t3").src=u("trocr",3);
 $("t384").src=u("trocr384",0);$("easy").src=u("easy");
 $("pcon").src=u("pcon");$("pr0").src=u("precaug",0);$("pr1").src=u("precaug",1);$("pr2").src=u("precaug",2);$("pfinal").src=u("pfinal");
 fetch(`/api/proll?id=${it.id}&seed=${seed}`).then(r=>r.json()).then(p=>{
  $("pmeta").innerHTML=p.error?p.error:
   `연결 파트너: [${p.partners.map(x=>`'${x}'`).join(", ")}] → 학습 라벨: <span class="plabel">'${p.concat_label}'</span> (${p.concat_label.length}자)`
   +(p.concat_happened?"":" — <i>가드로 연결 안 됨(길이/비율) — 이 샘플은 단독 학습</i>")
   +(p.dropped?" <span class=warn>⚠ 25자 초과 → 인코딩 드롭</span>":"")
   +(p.mode==="real"?"":p.mode==="port"?" <span style='color:#888'>(RecAug: 로컬 충실 포트 — paddle DLL 충돌로 원본 임포트 불가)</span>":" <span class=warn>⚠ tia 미적용 근사</span>");
 });
 const cur=document.querySelector(".row.cur");if(cur)cur.scrollIntoView({block:"nearest"});}
async function mark(v){const it=Q[idx];if(!it)return;hist.push({i:idx,old:it.mark});it.mark=v;
 await fetch("/api/mark",{method:"POST",body:JSON.stringify({path:it.path,text:it.text,verdict:v})});
 idx=Math.min(idx+1,Q.length);if(idx>shown-20)shown+=200;render();}
function undo(){const h=hist.pop();if(!h)return;idx=h.i;const it=Q[idx];it.mark=h.old;
 fetch("/api/mark",{method:"POST",body:JSON.stringify({path:it.path,text:it.text,verdict:h.old||"skip"})});render();}
$("good").onclick=()=>mark("good");$("bad").onclick=()=>mark("bad");$("skip").onclick=()=>mark("skip");
$("roll").onclick=()=>{seed=Math.floor(Math.random()*1e6);render()};
$("reload").onclick=fetchQ;$("more").onclick=()=>{shown+=400;render()};
document.addEventListener("keydown",e=>{if(e.target.tagName==="SELECT")return;
 if(e.key==="g"||e.key==="G"||e.key==="ArrowRight")mark("good");
 else if(e.key==="b"||e.key==="B"||e.key==="x"||e.key==="X")mark("bad");
 else if(e.key==="s"||e.key==="S")mark("skip");
 else if(e.key==="r"||e.key==="R"){seed=Math.floor(Math.random()*1e6);render();}
 else if(e.key==="ArrowLeft"){idx=Math.max(0,idx-1);render();}
 else if(e.key==="z"||e.key==="Z")undo();});
fetchQ();</script></body></html>"""


class H(BaseHTTPRequestHandler):
    rows: list[dict] = []
    marks: dict[str, str] = {}

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj) -> None:
        self._send(200, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def log_message(self, *a):   # quiet
        pass

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        try:
            if u.path == "/":
                self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif u.path == "/api/items":
                order = q.get("order", ["suspicious"])[0]
                filt = q.get("filter", ["unreviewed"])[0]
                items = []
                for r in self.rows:
                    s, why = suspicion(r)
                    m = self.marks.get(r["path"], "")
                    if filt == "unreviewed" and m:
                        continue
                    if filt == "bad" and m != "bad":
                        continue
                    items.append({"id": r["id"], "path": r["path"], "text": r["text"],
                                  "w": int(r["w"]), "h": int(r["h"]),
                                  "score": s, "why": why, "mark": m})
                if order == "suspicious":
                    items.sort(key=lambda x: -x["score"])
                elif order == "random":
                    random.Random(0).shuffle(items)
                self._json(items[:4000])
            elif u.path == "/api/stats":
                vs = list(self.marks.values())
                self._json({"total": len(self.rows), "reviewed": len(vs),
                            "good": vs.count("good"), "bad": vs.count("bad"),
                            "skip": vs.count("skip")})
            elif u.path == "/api/proll":
                rid = int(q["id"][0]); seed = int(q.get("seed", ["1"])[0])
                meta, _ = paddle_roll(self.rows[rid], self.rows, seed)
                self._json(meta)
            elif u.path == "/img":
                rid = int(q["id"][0]); view = q["view"][0]
                seed = int(q.get("seed", ["1"])[0]); k = int(q.get("k", ["0"])[0])
                body = render_view(self.rows[rid], self.rows, view, seed, k)
                self._send(200, body, "image/jpeg")
            else:
                self._send(404, b"nope", "text/plain")
        except Exception as exc:
            self._send(500, f"{type(exc).__name__}: {exc}".encode("utf-8"), "text/plain")

    def do_POST(self):
        if self.path == "/api/mark":
            n = int(self.headers.get("Content-Length", 0))
            d = json.loads(self.rfile.read(n).decode("utf-8"))
            append_mark(d["path"], d.get("text", ""), d["verdict"])
            self.marks[d["path"]] = d["verdict"]
            self._json({"ok": True})
        else:
            self._send(404, b"nope", "text/plain")


def selftest(rows: list[dict]) -> None:
    out = Path(sys.argv[0]).resolve().parent / "artifacts" / "ocr_training" / "signboard_v3" / "preview"
    out.mkdir(parents=True, exist_ok=True)
    row = next(r for r in rows if 3 <= len(r["text"]) <= 10)
    print(f"[SELFTEST] sample: {row['path']}  '{row['text']}'")
    for view, k in [("orig", 0), ("trocr", 0), ("trocr", 1), ("trocr384", 0),
                    ("easy", 0), ("pcon", 0), ("precaug", 0), ("precaug", 1), ("pfinal", 0)]:
        body = render_view(row, rows, view, seed=7, k=k)
        p = out / f"augtest_{view}{k}.jpg"
        p.write_bytes(body)
        print(f"  -> {p.name} ({len(body)} bytes)")
    meta, _ = paddle_roll(row, rows, 7)
    print(f"  RecConAug: partners={meta['partners']} label='{meta['concat_label']}' "
          f"mode={meta['mode']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8124)
    ap.add_argument("--scan", action="store_true", help="dataset audit only, no UI")
    ap.add_argument("--selftest", action="store_true", help="render one sample, no server")
    ap.add_argument("--export", action="store_true", help="write aug_review_bad.csv from marks")
    args = ap.parse_args()

    if args.export:
        export()
        return
    rows = load_rows()
    if not PADDLE_CHARS:
        print(f"[WARN] Paddle dict not found: {PPOCR_DICT}")
    if not EASY_CHARS:
        print(f"[WARN] EasyOCR charset not found: {EASY_CHARSET}")
    if args.scan:
        scan(rows)
        return
    if args.selftest:
        selftest(rows)
        return
    # torch와 paddle은 같은 프로세스에서 CUDA DLL이 충돌한다 (paddle_rec_worker.py가
    # 존재하는 이유). TrOCR 증강은 실제 학습 코드 경로(torchvision)가 핵심이므로
    # torch를 먼저 웜업하고, Paddle 쪽은 검증된 로컬 포트로 폴백시킨다.
    try:
        trocr_aug(Image.new("RGB", (32, 16), "white"), 0)
        print("[INIT] TrOCR 증강 파이프라인 로드 (실제 학습 코드 경로)")
    except Exception as exc:
        print(f"[WARN] TrOCR 증강 로드 실패: {exc}")
    ops = _paddle_ops()
    print(f"[INIT] Paddle RecConAug/RecAug: {ops['mode']}"
          + (" — ppocr 원본" if ops["mode"] == "real" else " — 로컬 충실 포트(소스 대조 검증)"))
    H.rows = rows
    H.marks = load_marks()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), H)
    print(f"[SERVE] http://localhost:{args.port}  (train rows: {len(rows)})")
    print("        Ctrl+C 로 종료. 판정은 aug_review_marks.csv 에 즉시 저장됩니다.")
    srv.serve_forever()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
