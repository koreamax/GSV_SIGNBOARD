#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""인식기·VLM 내부를 들여다보는 시각화 2종 (탐지 Grad-CAM 의 인식 단계 대응물).

  fig5  OCR 인식기 (SVTRv2)  : Grad-CAM + CTC 정렬 — 글자마다 이미지의 어디를 읽었는가
  fig6  VLM 교정 (Gemma 4)   : 가림(occlusion) 민감도 — 교정 결과가 어느 영역에 의존하는가
                               VLM 은 Ollama API 라 내부 attention 에 접근할 수 없어,
                               모델에 손대지 않는 표준 대체 기법(occlusion saliency)을 씁니다.

Usage:
  .venv/Scripts/python.exe pipeline/make_figures_attn.py --only 5
  .venv/Scripts/python.exe pipeline/make_figures_attn.py --only 6      # VLM 호출 ~20회, 약 10분
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.request
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parents[1]
OUT = Path.home() / "Desktop" / "GSV_figures"
CROP = HERE / "artifacts" / "gt" / "crop"
sys.path.insert(0, str(HERE / "pipeline"))
from make_figures import caption_bar, fit, font, INK, MUTED, PAPER  # noqa: E402

OLLAMA = "http://localhost:11434/api/generate"


# ----------------------------------------------------------------------
def build_strip(region: str, stem: str, index: int = 0) -> Image.Image:
    """배포 파이프라인과 동일하게 라인 스트립을 만듭니다 (CRAFT ∪ DB → 라인 병합 → 크롭).

    인식기의 실제 입력은 간판 크롭 전체가 아니라 이 스트립 한 줄입니다."""
    import tempfile
    import torch
    import easyocr
    import run_ocr_line as R
    import run_ocr_only as RO

    crop_path = CROP / region / f"{stem}.jpg"
    img = Image.open(crop_path).convert("RGB")
    reader = easyocr.Reader(["en"] if region == "brooklyn" else ["ko", "en"],
                            gpu=torch.cuda.is_available(),
                            **({} if region == "brooklyn" else {"recog_network": "signboard_v3_custom"}))
    lines = RO.easyocr_detect_lines(reader, img, line_y_tol=0.06, craft_text_threshold=0.7,
                                    craft_link_threshold=0.4, craft_low_text=0.4)
    lines, _ = RO.filter_boxes(lines, img.height, R.FILTER_ARGS)
    craft = [[[float(c[0]), float(c[1])] for c in b["bbox"]] for ln in lines for b in ln]
    tmp = Path(tempfile.mkdtemp(prefix="fig5_"))
    db = R.db_detect({region: [crop_path]}, tmp).get(f"{region}::{stem}", [])
    polys = craft + [b for b in db if not any(R._overlaps(b, c) for c in craft)]
    groups = R.group_lines(polys, img.height, 0.04)
    strip = R.crop_strip(img, [d["poly"] for d in groups[index]], pad=0.04)
    print(f"[fig5] 라인 스트립 {len(groups)}줄 중 {index}번 사용 — {strip.size}", flush=True)
    return strip


def fig5(region="gangnam", stem="gangnam__37__crop_001", index=0) -> Path:
    """SVTRv2 Grad-CAM + CTC 정렬."""
    import torch
    import torch.nn.functional as Fn
    import matplotlib

    strip = build_strip(region, stem, index)                     # 검출은 OpenOCR chdir 이전에 끝냅니다
    cwd = Path.cwd()
    openocr = HERE / "external" / "OpenOCR"
    sys.path.insert(0, str(openocr))
    os.chdir(openocr)
    try:
        from tools.engine.config import Config
        from tools.infer_rec import OpenRecognizer

        cfg = Config(str(openocr / "configs/rec/svtrv2/svtrv2_rctc_signboard.yml")).cfg
        cfg["Global"]["pretrained_model"] = str(HERE / "artifacts/str_baselines/svtrv2/best.pth")
        cfg["Global"]["checkpoints"] = None
        rec = OpenRecognizer(config=cfg, backend="torch", use_gpu="true")
        model, chars = rec.model, rec.post_process_class.character

        src = strip
        x = torch.as_tensor(rec.transform({"image": src}, rec.ops[1:])[0])[None].to(rec.device)

        model.eval()
        for p_ in model.parameters():
            p_.requires_grad_(True)
        acts, grads = {}, {}
        last = list(model.encoder.stages.children())[-1]         # 마지막 인코더 스테이지 특징맵
        def fwd(m, i, o):
            t = o[0] if isinstance(o, (list, tuple)) else o
            if torch.is_tensor(t):
                acts["a"] = t
                if t.requires_grad:
                    t.register_hook(lambda g: grads.__setitem__("g", g))
        h = last.register_forward_hook(fwd)

        logits = model(x)[0]                                     # (T, C) — CTC 시간축 T
        h.remove()
        # OpenOCR 의 CTC 헤드는 이미 softmax 를 적용해 내보냅니다. 한 번 더 걸면 11,979 클래스에
        # 확률이 고르게 퍼져 전부 0 이 됩니다. 행 합이 1 이면 그대로 씁니다.
        prob = logits if bool((logits.sum(-1) - 1).abs().max() < 1e-2) else logits.softmax(-1)
        ids = prob.argmax(-1)
        T = int(ids.shape[0])

        # 예측 글자의 로그확률 합을 스칼라로 역전파 (blank 제외)
        keep = ids != 0
        score = prob[torch.arange(T, device=ids.device)[keep], ids[keep]].log().sum()
        model.zero_grad()
        score.backward()

        A, G = acts["a"][0], grads["g"][0]                       # (C,H,W) 또는 (N,C)
        if A.dim() == 2:                                         # 토큰 시퀀스면 (H,W) 로 되돌립니다
            n, c = A.shape
            Hh = x.shape[2] // 4 or 1
            Ww = max(1, n // Hh)
            A = A.transpose(0, 1).reshape(c, Hh, Ww)
            G = G.transpose(0, 1).reshape(c, Hh, Ww)
        cam = Fn.relu((G.mean(dim=(1, 2), keepdim=True) * A).sum(0))
        cam = cam / (cam.max() + 1e-8)
        cam = Fn.interpolate(cam[None, None], size=(src.height, src.width),
                             mode="bilinear", align_corners=False)[0, 0].detach().cpu().numpy()
        cam = np.clip(cam / (np.percentile(cam, 99) + 1e-8), 0, 1) ** 0.8

        # CTC 정렬: 시간축 t 가 이미지 가로 어디에 해당하는지
        emissions = []                                           # (x 중심, 글자)
        prev = 0
        for t in range(T):
            cid = int(ids[t])
            if cid != 0 and cid != prev:
                emissions.append(((t + 0.5) / T * src.width, chars[cid],
                                  float(prob[t, cid])))
            prev = cid
    finally:
        os.chdir(cwd)

    # ---- 그리기: 원본 / Grad-CAM / CTC 정렬 3단 ----
    W = 1180
    img = fit(src, W)
    sx = img.width / src.width
    heat = (matplotlib.colormaps["turbo"](cam)[..., :3] * 255).astype(np.uint8)
    alpha = (cam ** 1.1 * 0.85)[..., None]
    cam_img = fit(Image.fromarray(np.clip(np.array(src) * (1 - alpha) + heat * alpha, 0, 255)
                                  .astype(np.uint8)), W)

    band_h = 92
    board = Image.new("RGB", (W, img.height + cam_img.height + band_h + 26), PAPER)
    board.paste(img, (0, 0))
    board.paste(cam_img, (0, img.height + 8))
    dr = ImageDraw.Draw(board)
    y0 = img.height + cam_img.height + 16
    dr.rectangle([0, y0, W, y0 + band_h], fill=(246, 248, 249))
    f = font(21, True)
    for cx, ch, pr in emissions:                                 # 글자마다 읽어낸 가로 위치에 눈금
        X = cx * sx
        dr.line([X, y0, X, y0 + 16], fill=(70, 120, 160), width=2)
        bb = dr.textbbox((0, 0), ch, font=f)
        dr.text((X - (bb[2] - bb[0]) / 2, y0 + 22), ch, font=f,
                fill=INK if pr >= 0.80 else (170, 120, 60))
    dr.text((8, y0 + band_h - 24), "CTC 시간축 → 가로 위치", font=font(14), fill=MUTED)

    out = caption_bar(board, "⑤ OCR 인식기 내부 — SVTRv2 Grad-CAM 과 CTC 정렬",
                      f"위: 인식기 입력 — 파이프라인이 만든 라인 스트립. 가운데: 예측 글자들의 로그확률 합을 역전파한 Grad-CAM. "
                      f"아래: CTC 가 각 글자를 읽어낸 가로 위치 — 읽은 결과 \"{''.join(e[1] for e in emissions)}\". "
                      "확신이 낮은 글자(확률 0.8 미만)는 주황으로 표시했습니다.",
                      [("확신 ≥ 0.8", INK), ("확신 < 0.8", (170, 120, 60))])
    p = OUT / "fig5_ocr_gradcam_ctc.jpg"
    out.save(p, quality=92)
    return p


# ----------------------------------------------------------------------
def ask_vlm(model: str, prompt: str, img: Image.Image, timeout=600) -> str:
    import io
    b = io.BytesIO(); img.save(b, "JPEG", quality=92)
    body = {"model": model, "prompt": prompt, "stream": False, "think": False, "keep_alive": "2h",
            "images": [base64.b64encode(b.getvalue()).decode()],
            "options": {"temperature": 0, "num_predict": 200, "num_ctx": 8192}}
    req = urllib.request.Request(OLLAMA, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()).get("response", "")


def _lev(a: str, b: str) -> int:
    if not a or not b:
        return max(len(a), len(b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[len(b)]


def fig6(region="gangnam", stem="gangnam__37__crop_001", model="gemma4:31b",
         cols=10, rows=2) -> Path:
    """가림 민감도 — 패치를 가리고 교정 결과가 얼마나 달라지는지 잽니다."""
    import matplotlib
    sys.path.insert(0, str(HERE / "pipeline"))
    saved = sys.argv
    sys.argv = ["eval_ocr_v2.py", "--mask-phone", "--en-only-regions", "brooklyn"]
    import eval_ocr_v2 as E
    sys.argv = saved

    src = Image.open(CROP / region / f"{stem}.jpg").convert("RGB")
    dep = E.load_csv_map(HERE / "artifacts" / "ocr_gt" / f"ocr_{region}_24_paddle.csv")
    base_lines = [l for l in E.split_lines(dep.get(stem, "")) if l.strip()]
    numbered = "\n".join(f"{i+1}. {l}" for i, l in enumerate(base_lines))
    prompt = (f"간판 사진과, OCR이 이 사진에서 읽은 텍스트 라인들이다:\n{numbered}\n\n"
              "각 라인을 사진과 대조해 **잘못 읽힌 글자만** 고쳐라.\n"
              "- 라인 개수와 순서를 그대로 유지한다.\n"
              "- 사진에서 확실히 다르게 보이는 글자만 고친다. 확신이 없으면 원문 그대로 둔다.\n"
              f'- 교정된 라인 {len(base_lines)}개를 JSON 배열로만 출력한다.')

    def norm(resp: str) -> str:
        import re
        m = re.search(r"\[.*\]", resp, re.S)
        if m:
            try:
                return " ".join(str(v) for v in json.loads(m.group(0)))
            except Exception:                                    # noqa: BLE001
                pass
        return " ".join(resp.split())

    t0 = time.time()
    ref = norm(ask_vlm(model, prompt, src))
    print(f"[fig6] 기준 출력: {ref[:90]!r}  ({time.time()-t0:.0f}s)", flush=True)

    sens = np.zeros((rows, cols), dtype=np.float32)
    pw, ph = src.width / cols, src.height / rows
    for r in range(rows):
        for c in range(cols):
            occ = src.copy()
            ImageDraw.Draw(occ).rectangle(
                [c * pw, r * ph, (c + 1) * pw, (r + 1) * ph], fill=(128, 128, 128))
            got = norm(ask_vlm(model, prompt, occ))
            sens[r, c] = _lev(ref, got) / max(len(ref), 1)
            print(f"[fig6] patch r{r}c{c}: 변화 {sens[r,c]:.2f}", flush=True)

    m = sens / (sens.max() + 1e-8)
    big = np.kron(m, np.ones((max(1, src.height // rows), max(1, src.width // cols))))
    big = np.array(Image.fromarray((big * 255).astype(np.uint8)).resize(src.size, Image.BILINEAR)) / 255.
    heat = (matplotlib.colormaps["turbo"](big)[..., :3] * 255).astype(np.uint8)
    alpha = (big * 0.7)[..., None]
    blend = Image.fromarray(np.clip(np.array(src) * (1 - alpha) + heat * alpha, 0, 255).astype(np.uint8))
    dr = ImageDraw.Draw(blend)
    for r in range(rows):
        for c in range(cols):
            dr.rectangle([c * pw, r * ph, (c + 1) * pw, (r + 1) * ph], outline=(255, 255, 255), width=1)

    W = 1180
    top, bot = fit(src, W), fit(blend, W)
    board = Image.new("RGB", (W, top.height + bot.height + 8), PAPER)
    board.paste(top, (0, 0)); board.paste(bot, (0, top.height + 8))
    out = caption_bar(board, "⑥ VLM 교정 민감도 — 어느 영역을 가리면 교정이 무너지는가",
                      f"{model} 에 같은 교정 프롬프트를 주고 격자 {rows}×{cols} 패치를 하나씩 가려 출력 변화를 "
                      f"정규화 편집거리로 쟀습니다(총 {rows*cols+1}회 호출). 붉을수록 그 영역이 교정에 결정적입니다. "
                      f"가리지 않은 출력: \"{ref[:60]}\". VLM 은 API 라 내부 attention 을 볼 수 없어 "
                      "모델을 건드리지 않는 가림 민감도로 대신했습니다.",
                      [])
    p = OUT / "fig6_vlm_occlusion.jpg"
    out.save(p, quality=92)
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--index", type=int, default=0, help="fig5: 몇 번째 라인 스트립")
    ap.add_argument("--cols", type=int, default=10)
    ap.add_argument("--rows", type=int, default=2)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for n in ([args.only] if args.only else ["5", "6"]):
        p = fig5(index=args.index) if n == "5" else fig6(cols=args.cols, rows=args.rows)
        im = Image.open(p)
        print(f"[fig{n}] {p}  {im.width}x{im.height}  {p.stat().st_size/1024:.0f} KB", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
