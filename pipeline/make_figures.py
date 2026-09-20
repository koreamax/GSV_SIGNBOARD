#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""논문/발표용 시각화 4종 — 파이프라인 단계별로 한 장씩.

  fig1  간판 바운딩 박스   : YOLO26x out-of-fold 예측 vs GT (TP/FP/FN)
  fig2  텍스트 바운딩 박스 : CRAFT ∪ PaddleOCR-DB 단어 박스 + 라인 병합 스트립
  fig3  Grad-CAM          : 간판 탐지기가 사진의 어디를 보는가 (OOF fold 모델)
  fig4  OCR + VLM 교정    : 배포 OCR → VLM 교정 전후 (글자 단위 차이 표시)

Usage:
  .venv/Scripts/python.exe pipeline/make_figures.py            # 전부
  .venv/Scripts/python.exe pipeline/make_figures.py --only 3   # 하나만
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE / d) for d in ("pipeline", "ocr", "vlm", "str_baselines")]
GT = HERE / "artifacts" / "gt"
PHOTO = HERE / "artifacts" / "gsv_photo"
OUT = Path.home() / "Desktop" / "GSV_figures"
FONT = "C:/Windows/Fonts/malgun.ttf"
FONT_B = "C:/Windows/Fonts/malgunbd.ttf"

# 색 (RGB) — 페이지·문서와 같은 의미 색을 씁니다
C_TP = (60, 190, 110)
C_FP = (235, 150, 40)
C_FN = (225, 70, 70)
C_CRAFT = (60, 150, 220)
C_DB = (235, 150, 40)
C_LINE = (60, 190, 110)
INK = (20, 28, 34)
MUTED = (110, 125, 135)
PAPER = (255, 255, 255)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_B if bold else FONT, size)


def load_boxes(path: Path) -> dict[str, list[tuple[float, float, float, float]]]:
    d: dict[str, list] = defaultdict(list)
    for r in csv.DictReader(path.open(encoding="utf-8")):
        xs = [float(r[f"x{i}"]) for i in range(1, 5)]
        ys = [float(r[f"y{i}"]) for i in range(1, 5)]
        d[Path(r["filename"]).stem].append((min(xs), min(ys), max(xs), max(ys)))
    return d


def iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    i = (x1 - x0) * (y1 - y0)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def match(pred, gts):
    """탐욕적 1:1 매칭 (IoU>=0.5). 반환: (TP 예측 idx, FP 예측 idx, FN GT idx)"""
    used, tp, fp = set(), [], []
    for pi, p in enumerate(pred):
        cand = [(iou(p, g), gi) for gi, g in enumerate(gts) if gi not in used]
        best = max(cand, default=(0.0, -1))
        if best[0] >= 0.5:
            used.add(best[1]); tp.append(pi)
        else:
            fp.append(pi)
    return tp, fp, [gi for gi in range(len(gts)) if gi not in used]


def draw_box(dr, box, color, width, label=None, fs=0):
    dr.rectangle(box, outline=color, width=width)
    if label:
        f = font(fs, True)
        tw = dr.textbbox((0, 0), label, font=f)
        pad = max(3, fs // 5)
        h = tw[3] - tw[1] + pad * 2
        y = max(0, box[1] - h)
        dr.rectangle([box[0], y, box[0] + (tw[2] - tw[0]) + pad * 2, y + h], fill=color)
        dr.text((box[0] + pad, y + pad - tw[1]), label, font=f, fill=(255, 255, 255))


BARE = True          # True 면 그림 아래 설명 띠를 붙이지 않고 이미지만 남깁니다


def caption_bar(img: Image.Image, title: str, sub: str, legend: list[tuple[str, tuple]]) -> Image.Image:
    """그림 아래 제목·설명·범례 띠를 붙입니다 (BARE=True 면 그대로 반환)."""
    if BARE:
        return img
    W = img.width
    fs_t, fs_s = max(17, W // 52), max(14, W // 68)
    pad = max(14, W // 70)
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    f_s = font(fs_s)

    lines, cur = [], ""                                         # 설명은 폭에 맞춰 줄바꿈합니다
    for word in sub.split(" "):
        trial = (cur + " " + word).strip()
        if probe.textbbox((0, 0), trial, font=f_s)[2] > W - pad * 2 and cur:
            lines.append(cur); cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)

    h = pad * 2 + fs_t + len(lines) * (fs_s + 4) + (fs_s + pad // 2 if legend else 0)
    out = Image.new("RGB", (W, img.height + h), PAPER)
    out.paste(img, (0, 0))
    dr = ImageDraw.Draw(out)
    y = img.height + pad
    dr.text((pad, y), title, font=font(fs_t, True), fill=INK)
    y += fs_t + pad // 3
    for ln in lines:
        dr.text((pad, y), ln, font=f_s, fill=MUTED)
        y += fs_s + 4
    if legend:
        y += pad // 3
        x = pad
        for text, color in legend:
            s = fs_s
            dr.rectangle([x, y + 2, x + s, y + s + 2], fill=color, outline=MUTED)
            x += s + 6
            dr.text((x, y), text, font=f_s, fill=INK)
            x += probe.textbbox((0, 0), text, font=f_s)[2] + pad
    return out


def fit(img: Image.Image, w: int) -> Image.Image:
    return img if img.width <= w else img.resize((w, round(img.height * w / img.width)), Image.LANCZOS)


# ----------------------------------------------------------------------
def fig1(stem="brooklyn__60", region="brooklyn") -> Path:
    gts = load_boxes(GT / "total_gt.csv")[stem]
    preds = load_boxes(GT / f"gt_{region}_det.csv")[stem]
    img = Image.open(PHOTO / region / f"{stem.split('__')[1]}.jpg").convert("RGB")
    tp, fp, fn = match(preds, gts)
    dr = ImageDraw.Draw(img)
    w = max(4, img.width // 380)
    fs = max(20, img.width // 70)
    for gi in fn:
        draw_box(dr, gts[gi], C_FN, w, "FN 놓침", fs)
    for pi in tp:
        draw_box(dr, preds[pi], C_TP, w, "TP", fs)
    for pi in fp:
        draw_box(dr, preds[pi], C_FP, w, "FP 허위", fs)
    img = fit(img, 1400)
    out = caption_bar(img, "① 간판 탐지 — YOLO26x out-of-fold 예측",
                      f"{region} #{stem.split('__')[1]} · 맞게 찾음 {len(tp)} · 허위 {len(fp)} · 놓침 {len(fn)} "
                      f"· 전체 mAP@0.5 0.880 (recall 0.816 / precision 0.751)",
                      [("맞게 찾음 (IoU≥0.5)", C_TP), ("허위 탐지", C_FP), ("놓친 간판", C_FN)])
    p = OUT / "fig1_signboard_boxes.jpg"
    out.save(p, quality=92)
    return p


# ----------------------------------------------------------------------
def fig2(key="brooklyn::brooklyn__18__crop_001") -> Path:
    import run_ocr_line as R
    import run_ocr_only as RO
    import easyocr
    import torch

    region, stem = key.split("::")
    crop = GT / "crop" / region / f"{stem}.jpg"
    img = Image.open(crop).convert("RGB")

    reader = easyocr.Reader(["en"] if region == "brooklyn" else ["ko", "en"],
                            gpu=torch.cuda.is_available(),
                            **({} if region == "brooklyn" else {"recog_network": "signboard_v3_custom"}))
    lines = RO.easyocr_detect_lines(reader, img, line_y_tol=0.06, craft_text_threshold=0.7,
                                    craft_link_threshold=0.4, craft_low_text=0.4)
    lines, _ = RO.filter_boxes(lines, img.height, R.FILTER_ARGS)
    craft = [[[float(c[0]), float(c[1])] for c in b["bbox"]] for ln in lines for b in ln]

    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="fig2_"))
    db_all = R.db_detect({region: [crop]}, tmp)
    db = db_all.get(f"{region}::{stem}", [])
    extra = [b for b in db if not any(R._overlaps(b, c) for c in craft)]
    polys = craft + extra

    canvas = img.copy()
    dr = ImageDraw.Draw(canvas)
    w = max(2, img.width // 500)
    for ln in R.group_lines(polys, img.height, 0.04):          # 병합된 라인 스트립
        xs = [c[0] for d in ln for c in d["poly"]]
        ys = [c[1] for d in ln for c in d["poly"]]
        dr.rectangle([min(xs), min(ys), max(xs), max(ys)], outline=C_LINE, width=w * 3)
    def bbox(poly):                                             # 단어 박스는 축정렬 사각형으로 그립니다
        xs = [c[0] for c in poly]; ys = [c[1] for c in poly]     # (CRAFT 폴리곤은 꼭짓점 수·순서가 일정하지 않음)
        return [min(xs), min(ys), max(xs), max(ys)]

    for poly in craft:                                          # CRAFT 단어 박스
        dr.rectangle(bbox(poly), outline=C_CRAFT, width=w)
    for poly in extra:                                          # DB 가 추가로 찾은 박스
        dr.rectangle(bbox(poly), outline=C_DB, width=w * 2)

    canvas = fit(canvas, 1100)
    out = caption_bar(canvas, "② 텍스트 탐지 — 검출기 합집합과 라인 병합",
                      f"단어 박스 {len(polys)}개 (CRAFT {len(craft)} + DB 추가 {len(extra)}) → 라인 스트립 "
                      f"{len(R.group_lines(polys, img.height, 0.04))}줄. 인식기는 이 스트립을 통째로 읽습니다.",
                      [("CRAFT 단어 박스", C_CRAFT), ("DB 가 추가로 찾은 박스", C_DB), ("병합된 라인 스트립", C_LINE)])
    p = OUT / "fig2_text_boxes.jpg"
    out.save(p, quality=92)
    return p


# ----------------------------------------------------------------------
def fig3(stem="brooklyn__60", region="brooklyn") -> Path:
    """Grad-CAM — 탐지 신뢰도 합을 스칼라로 역전파."""
    import warnings
    warnings.filterwarnings("ignore")
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fn
    from ultralytics import YOLO

    fold = {r["photo"]: r["fold"] for r in csv.DictReader((HERE / "artifacts" / "kfold_grouped" /
                                                           "group_assignment.csv").open(encoding="utf-8"))}
    f = fold[f"total__{stem}"]                                  # 이 사진을 학습에 쓰지 않은 fold 모델
    net = YOLO(HERE / "artifacts" / f"yolo26x_kfold_grouped/fold{f}/weights/best.pt").model
    # 추론 모드의 Detect 출력은 detach 돼 있어 기울기가 백본까지 가지 않습니다.
    # 학습 모드 순전파(원시 헤드 출력)를 쓰되 BatchNorm 은 eval 로 고정해 통계를 바꾸지 않습니다.
    net.train()
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    for p_ in net.parameters():
        p_.requires_grad_(True)

    src = Image.open(PHOTO / region / f"{stem.split('__')[1]}.jpg").convert("RGB")
    S = 640
    x = torch.from_numpy(np.array(src.resize((S, S), Image.BILINEAR))).permute(2, 0, 1).float()[None] / 255.

    LAYERS = (16, 19, 22)                                       # P3·P4·P5 특징맵
    acts, grads, hooks = {}, {}, []
    def mk(i):
        def f(m, inp, o):
            if torch.is_tensor(o):
                acts[i] = o
                if o.requires_grad:
                    o.register_hook(lambda g, i=i: grads.__setitem__(i, g))
        return f
    for i in LAYERS:
        hooks.append(net.model[i].register_forward_hook(mk(i)))

    out = net(x)
    def tensors(o):
        if torch.is_tensor(o):
            return [o]
        if isinstance(o, dict):
            return [t for v in o.values() for t in tensors(v)]
        if isinstance(o, (list, tuple)):
            return [t for v in o for t in tensors(v)]
        return []
    scores = [t for t in tensors(out) if t.requires_grad and t.dim() == 3 and t.shape[1] == net.nc]
    if not scores:
        raise RuntimeError("클래스 점수 텐서를 찾지 못했습니다")
    sc = scores[0].sigmoid().flatten()                          # 상위 앵커의 간판 신뢰도 합을 스칼라로
    score = sc.topk(min(20, sc.numel())).values.sum()
    net.zero_grad()
    score.backward()
    for h in hooks:
        h.remove()
    if not grads:
        raise RuntimeError("gradient hook 미발화")

    cams = []                                                   # 스케일별 Grad-CAM 을 정규화 후 평균
    for i in LAYERS:
        A, G = acts[i][0], grads[i][0]
        c = Fn.relu((G.mean(dim=(1, 2), keepdim=True) * A).sum(0))
        c = (c - c.min()) / (c.max() - c.min() + 1e-8)
        cams.append(Fn.interpolate(c[None, None], size=(src.height, src.width),
                                   mode="bilinear", align_corners=False)[0, 0])
    cam = torch.stack(cams).mean(0)
    cam = ((cam - cam.min()) / (cam.max() - cam.min() + 1e-8)).detach().numpy()

    import matplotlib

    # 대비 보정: 상위 1% 를 상한으로 잘라 정규화해야 히트맵이 뭉개지지 않습니다.
    hi = float(np.percentile(cam, 99))
    cam = np.clip(cam / (hi + 1e-8), 0, 1) ** 0.8

    heat = (matplotlib.colormaps["turbo"](cam)[..., :3] * 255).astype(np.uint8)
    base = np.array(src).astype(np.float32)
    alpha = (cam ** 1.1 * 0.85)[..., None]                      # 약한 곳은 원본을 그대로 보여 줍니다
    blend = Image.fromarray(np.clip(base * (1 - alpha) + heat * alpha, 0, 255).astype(np.uint8))

    dr = ImageDraw.Draw(blend)                                  # 정답 간판 위치 참고선 (검은 테두리 + 흰 선)
    w_ = max(3, src.width // 450)
    for bx in load_boxes(GT / "total_gt.csv")[stem]:
        dr.rectangle([bx[0] - w_, bx[1] - w_, bx[2] + w_, bx[3] + w_], outline=(0, 0, 0), width=w_)
        dr.rectangle(bx, outline=(255, 255, 255), width=w_)

    pair = Image.new("RGB", (src.width * 2 + 16, src.height), PAPER)
    pair.paste(src, (0, 0)); pair.paste(blend, (src.width + 16, 0))
    pair = fit(pair, 1600)
    out_img = caption_bar(pair, "③ Grad-CAM — 탐지기는 사진의 어디를 보는가",
                          f"{region} #{stem.split('__')[1]} · fold{f} 모델(이 사진을 학습하지 않음). P3·P4·P5 세 특징맵의 "
                          "Grad-CAM 을 평균했습니다. 왼쪽 원본, 오른쪽 히트맵(붉을수록 간판 판정에 기여).",
                          [("정답 간판 위치", (255, 255, 255))])
    p = OUT / "fig3_gradcam.jpg"
    out_img.save(p, quality=92)
    return p


# ----------------------------------------------------------------------
def _nk(s: str) -> str:
    import re
    return re.sub(r"[^0-9a-z가-힣]", "", unicodedata.normalize("NFKC", s or "").lower())


def fig4(keys=(("gangnam", "gangnam__37__crop_001"), ("suwon", "suwon__59__crop_001"))) -> Path:
    saved = sys.argv                                             # eval_ocr_v2 는 import 시점에 argv 를 파싱합니다
    sys.argv = ["eval_ocr_v2.py", "--mask-phone", "--en-only-regions", "brooklyn"]
    import eval_ocr_v2 as E                                      # 공식 채점 규칙으로 라인 분해
    sys.argv = saved

    W, fs, pad = 1180, 25, 22
    C_GT, C_OCR, C_VLM = INK, (185, 95, 55), (35, 135, 95)
    blocks = []
    for region, stem in keys:
        img = Image.open(GT / "crop" / region / f"{stem}.jpg").convert("RGB")
        gt = E.load_csv_map(GT / f"ocr_{region}_gt.csv")
        dep = E.load_csv_map(HERE / "artifacts" / "ocr_gt" / f"ocr_{region}_24_paddle.csv")
        fx = E.load_csv_map(HERE / "artifacts" / "ocr_gt" / "ab_v4_spacecat" /
                            f"ocr_{region}_vlmfixcand5_gemma4_31b.csv")
        rows = [("정답", [l for l in E.split_lines(gt.get(stem, "")) if l.strip() and l != "###"], C_GT),
                ("배포 OCR", E.split_lines(dep.get(stem, "")), C_OCR),
                ("+ VLM 교정 (fixcand5)", E.split_lines(fx.get(stem, "")), C_VLM)]
        gt_keys = {_nk(l) for l in rows[0][1]}

        thumb_h = 150                                            # 크롭은 높이를 맞춰 나란히 놓습니다
        thumb = img.resize((max(1, round(img.width * thumb_h / img.height)), thumb_h), Image.LANCZOS)
        thumb = thumb if thumb.width <= W - pad * 2 else fit(thumb, W - pad * 2)
        body = pad + thumb.height + pad + sum((fs + 6) + len(v) * (fs + 8) + 10 for _, v, _ in rows)
        panel = Image.new("RGB", (W, body), PAPER)
        panel.paste(thumb, (pad, pad))
        dr = ImageDraw.Draw(panel)
        y = pad + thumb.height + pad
        for label, vals, color in rows:
            dr.text((pad, y), label, font=font(fs - 6, True), fill=MUTED)
            y += fs + 6
            for v in vals:
                hit = label != "정답" and _nk(v) in gt_keys
                if hit:                                          # 정답과 완전히 일치한 라인 표시 (글리프 대신 도형)
                    r = fs // 5
                    cy = y + fs // 2
                    dr.ellipse([pad + 10, cy - r, pad + 10 + r * 2, cy + r], fill=color)
                dr.text((pad + 34, y), v, font=font(fs, hit), fill=color if hit or label == "정답" else MUTED)
                y += fs + 8
            y += 10
        blocks.append(panel)

    gapline = 2
    board = Image.new("RGB", (W, sum(b.height for b in blocks) + gapline * (len(blocks) - 1)), PAPER)
    y = 0
    for i, b in enumerate(blocks):
        board.paste(b, (0, y)); y += b.height
        if i < len(blocks) - 1:
            ImageDraw.Draw(board).rectangle([0, y, W, y + gapline], fill=(225, 231, 235)); y += gapline
    out = caption_bar(board, "④ OCR + VLM 교정 — 사진을 보고 오독 글자만 고친다",
                      "배포 OCR 라인과 크롭, 그리고 5개 인식기의 라인별 후보를 Gemma 4 31B 에 함께 제시했습니다(fixcand5). "
                      "점 표시는 정답과 완전히 일치한 라인입니다. 크롭은 파이프라인이 실제로 인식기에 넣는 전처리 결과입니다. "
                      "전역 line exact 69.9% → 80.9%.",
                      [("정답", C_GT), ("배포 OCR", C_OCR), ("+ VLM 교정", C_VLM)])
    p = OUT / "fig4_ocr_vlm.jpg"
    out.save(p, quality=92)
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="1/2/3/4 중 하나만")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [args.only] if args.only else ["1", "2", "3", "4"]
    for n in todo:
        fn = {"1": fig1, "2": fig2, "3": fig3, "4": fig4}[n]
        p = fn()
        im = Image.open(p)
        print(f"[fig{n}] {p}  {im.width}x{im.height}  {p.stat().st_size/1024:.0f} KB", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
