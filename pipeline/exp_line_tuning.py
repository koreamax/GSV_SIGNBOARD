#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Line-strip tuning experiments on the deployed pipeline (C-track follow-ups).

Detection (CRAFT ∪ DB, D23) is expensive and identical across these
experiments, so it is run once and cached to artifacts/ocr_gt/det_cache.json.

  --mode pad       : strip padding sweep (targets trailing-character truncation
                     — 'BBQ'->'BB', 'MLB'->'B' in the 1-2 char error bucket)
  --mode ensemble  : LINE-level multi-model recognition + rec_score selection.
                     Unlike the failed crop-level T7/C1 (candidates had
                     different granularity), every candidate here reads the SAME
                     line strip, so agreement/confidence are meaningful.

Usage:
  .venv/Scripts/python.exe exp_line_tuning.py --mode pad
  .venv/Scripts/python.exe exp_line_tuning.py --mode ensemble [--pad 0.18]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parents[1]

import sys as _sys; _sys.path[:0] = [str(HERE / _d) for _d in ("pipeline", "ocr", "vlm", "str_baselines", "detection", "data")]  # 폴더 재편 후 형제 모듈 import 경로
CROP = HERE / "artifacts" / "gt" / "crop"
OCR_DIR = HERE / "artifacts" / "ocr_gt"
DET_CACHE = OCR_DIR / "det_cache.json"
PY = HERE / ".venv" / "Scripts" / "python.exe"
REC_WORKER = HERE / "str_baselines/paddle_rec_worker.py"
REGIONS = ["gangnam", "suwon", "brooklyn"]
KO_REGIONS = {"gangnam", "suwon"}

MODELS_KO = {
    "v5": HERE / "output" / "paddle_signboard_rec_v5_lines" / "inference",
    "v4": HERE / "output" / "paddle_signboard_rec_v4_spacecat" / "inference",
    "pre": None,     # zero-shot korean_PP-OCRv5_mobile_rec
}
PRETRAINED_KO = "korean_PP-OCRv5_mobile_rec"
PRETRAINED_EN = "en_PP-OCRv5_mobile_rec"


def build_det_cache() -> dict:
    """CRAFT (torch) ∪ DB (paddle) polygons per crop — same recipe as run22."""
    if DET_CACHE.exists():
        print(f"[DET] cache hit: {DET_CACHE.name}")
        return json.loads(DET_CACHE.read_text(encoding="utf-8"))
    import run_ocr_line as RL
    import run_ocr_only as RO
    import torch
    import easyocr

    readers = {}

    def reader_for(kind):
        if kind not in readers:
            readers[kind] = (easyocr.Reader(["en"], gpu=torch.cuda.is_available())
                             if kind == "en" else
                             easyocr.Reader(["ko", "en"], gpu=torch.cuda.is_available(),
                                            recog_network="signboard_v3_custom"))
        return readers[kind]

    files_by_region = {r: sorted((CROP / r).glob("*.jpg"), key=lambda p: RL.natural_key(p.stem))
                       for r in REGIONS}
    polys = {}
    for region in REGIONS:
        kind = "ko" if region in KO_REGIONS else "en"
        print(f"[DET/CRAFT] {region}: {len(files_by_region[region])} crops")
        rd = reader_for(kind)
        for p in files_by_region[region]:
            img = Image.open(p).convert("RGB")
            lines = RO.easyocr_detect_lines(rd, img, line_y_tol=0.06,
                                            craft_text_threshold=0.7,
                                            craft_link_threshold=0.4, craft_low_text=0.4)
            lines, _ = RO.filter_boxes(lines, img.height, RL.FILTER_ARGS)
            polys[f"{region}::{p.stem}"] = [
                [[float(c[0]), float(c[1])] for c in b["bbox"]] for ln in lines for b in ln]
    tmp = Path(tempfile.mkdtemp(prefix="det_cache_"))
    print("[DET/DB] PaddleOCR DB detector ...")
    db = RL.db_detect(files_by_region, tmp)
    added = 0
    for k, cb in polys.items():
        extra = [b for b in db.get(k, []) if not any(RL._overlaps(b, c) for c in cb)]
        polys[k] = cb + extra
        added += len(extra)
    print(f"[DET/DB] +{added} boxes")
    DET_CACHE.write_text(json.dumps(polys), encoding="utf-8")
    print(f"[DET] cached -> {DET_CACHE.name}")
    return polys


def crop_strip_hrel(img, polys, kx: float, ky: float):
    """Padding relative to LINE HEIGHT on both axes.

    run_ocr_line.crop_strip pads by a fraction of the strip's own width, so a
    wide line gets a huge horizontal margin that drags in neighbouring text.
    Here both margins scale with the text height instead.
    """
    pts = [p for poly in polys for p in poly]
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    h = max(y1 - y0, 1.0)
    X0, Y0 = max(0, int(x0 - kx * h)), max(0, int(y0 - ky * h))
    X1 = min(img.width, int(x1 + kx * h))
    Y1 = min(img.height, int(y1 + ky * h))
    if X1 <= X0 or Y1 <= Y0:
        return None
    return img.crop((X0, Y0, X1, Y1))


def make_strips(polys: dict, pad, tmp: Path, tag: str, y_tol: float = 0.06):
    """Crop line strips. pad = float (width-relative, legacy) or (kx, ky) height-relative."""
    import run_ocr_line as RL
    manifest, layout = [], {}
    for region in REGIONS:
        layout[region] = {}
        for p in sorted((CROP / region).glob("*.jpg"), key=lambda q: RL.natural_key(q.stem)):
            img = Image.open(p).convert("RGB")
            pl = polys.get(f"{region}::{p.stem}", [])
            n = 0
            for li, ln in enumerate(RL.group_lines(pl, img.height, y_tol) if pl else []):
                pp = [d["poly"] for d in ln]
                strip = (crop_strip_hrel(img, pp, pad[0], pad[1])
                         if isinstance(pad, tuple) else RL.crop_strip(img, pp, pad=pad))
                if strip is None:
                    continue
                fp = tmp / f"{tag}_{region}_{p.stem}_{li}.png"
                strip.save(fp)
                manifest.append({"key": f"{region}::{p.stem}::{li}", "path": str(fp)})
                n += 1
            layout[region][p.stem] = n
    return manifest, layout


def recognize(manifest, tmp: Path, tag: str, ko_model, want_scores=False):
    """Run the isolated paddle worker; ko/en split by region prefix of the key."""
    out_all = {}
    for kind in ("ko", "en"):
        sub = [m for m in manifest
               if (m["key"].split("::")[0] in KO_REGIONS) == (kind == "ko")]
        if not sub:
            continue
        man, out = tmp / f"m_{tag}_{kind}.jsonl", tmp / f"o_{tag}_{kind}.jsonl"
        with man.open("w", encoding="utf-8") as f:
            for m in sub:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        cmd = [str(PY), str(REC_WORKER), "--manifest", str(man), "--out", str(out),
               "--device", "gpu"]
        if kind == "en":
            cmd += ["--model-name", PRETRAINED_EN]
        elif ko_model is None:
            cmd += ["--model-name", PRETRAINED_KO]
        else:
            cmd += ["--model-dir", str(ko_model)]
        subprocess.run(cmd, check=True)
        with out.open(encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                out_all[d["key"]] = (d["text"], float(d.get("score", 0.0)))
    return out_all if want_scores else {k: v[0] for k, v in out_all.items()}


def assemble(layout, preds):
    maps = {}
    for region, imgs in layout.items():
        m = {}
        for name, n in imgs.items():
            parts = [(preds.get(f"{region}::{name}::{li}") or "").strip() for li in range(n)]
            m[name] = "\n".join(p for p in parts if p)
        maps[region] = m
    return maps


def load_eval():
    saved = sys.argv
    sys.argv = ["pipeline/eval_ocr_v2.py", "--mask-phone", "--en-only-regions", "brooklyn"]
    import eval_ocr_v2 as E
    sys.argv = saved
    return E


def score(E, maps, gts):
    tot = None
    for r in REGIONS:
        E.EN_ONLY = r in E.EN_ONLY_REGIONS
        a = E.eval_engine(gts[r], maps[r], [])
        E.EN_ONLY = False
        if tot is None:
            tot = a
        else:
            for f in ("edit", "fp_edit", "chars", "word_lcs", "gt_words", "exact",
                      "recalled", "lines", "contain", "empty_pred", "crops"):
                setattr(tot, f, getattr(tot, f) + getattr(a, f))
    return tot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pad", "ensemble", "en-models", "tta", "ytol"],
                    required=True)
    ap.add_argument("--ytols", type=str, default="0.06,0.04,0.03,0.02,0.015",
                    help="ytol mode: line-grouping tolerances to sweep")
    ap.add_argument("--pad", type=float, default=0.12, help="ensemble mode: strip padding")
    ap.add_argument("--pads", type=str, default="0.12,0.18,0.25,0.35",
                    help="pad mode: comma-separated paddings to sweep")
    args = ap.parse_args()

    polys = build_det_cache()
    tmp = Path(tempfile.mkdtemp(prefix="line_tune_"))
    E = load_eval()
    gts = {r: E.load_csv_map(HERE / "artifacts" / "gt" / f"ocr_{r}_gt.csv") for r in REGIONS}

    if args.mode == "pad":
        # entries: "0.12" = width-relative (legacy) | "0.10x0.20" = height-relative (kx,ky)
        specs = []
        for tok in args.pads.split(","):
            tok = tok.strip()
            specs.append((tok, tuple(float(v) for v in tok.split("x"))) if "x" in tok
                         else (tok, float(tok)))
        print(f"\n{'pad':>12s} {'strips':>7s} {'exact':>7s} {'CER':>7s} {'WAR':>7s} {'contain':>8s}")
        for label, pad in specs:
            tag = "p" + label.replace(".", "").replace("x", "_")
            man, lay = make_strips(polys, pad, tmp, tag)
            preds = recognize(man, tmp, tag, MODELS_KO["v5"])
            a = score(E, assemble(lay, preds), gts)
            mark = "  <- 배포" if pad == 0.12 else ""
            kind = "(h기준)" if isinstance(pad, tuple) else "(w기준)"
            print(f"{label + kind:>12s} {len(man):7d} {a.exact_rate()*100:6.1f}% {a.cer():7.3f} "
                  f"{a.war():7.3f} {a.contain_rate()*100:7.1f}%{mark}")
        return

    if args.mode == "ytol":
        # Line grouping merges boxes whose y-centres differ by < y_tol x image
        # height. Too generous and two stacked sign lines become one strip, so
        # the eval's 1:1 line matching drops the second GT line entirely (43% of
        # complete-loss lines are this, not recognition failure).
        print(f"\n{'y_tol':>7s} {'strips':>7s} {'exact':>7s} {'CER':>7s} {'WAR':>7s} {'contain':>8s}")
        for yt in [float(x) for x in args.ytols.split(",")]:
            tag = f"y{int(yt*1000)}"
            man, lay = make_strips(polys, args.pad, tmp, tag, y_tol=yt)
            preds = recognize(man, tmp, tag, MODELS_KO["v5"])
            a = score(E, assemble(lay, preds), gts)
            mark = "  <- 배포" if abs(yt - 0.06) < 1e-9 else ""
            print(f"{yt:7.3f} {len(man):7d} {a.exact_rate()*100:6.1f}% {a.cer():7.3f} "
                  f"{a.war():7.3f} {a.contain_rate()*100:7.1f}%{mark}")
        return

    if args.mode == "tta":
        # Multi-scale test-time augmentation. Targets the 1-2 char misread bucket
        # (15.9% of GT lines): a small strip upscaled before the 48px network
        # resize can decide an ambiguous glyph differently.
        import re as _re
        import unicodedata as _ud

        def nk(s):
            return _re.sub(r"[^0-9a-z가-힣]", "",
                           _ud.normalize("NFKC", s or "").lower())

        base_man, lay = make_strips(polys, args.pad, tmp, "tta")
        variants = {}
        for scale in (1.0, 1.5, 2.0):
            if scale == 1.0:
                man = base_man
            else:                       # rescale each saved strip
                man = []
                for m in base_man:
                    img = Image.open(m["path"])
                    fp = tmp / f"s{int(scale*10)}_{Path(m['path']).name}"
                    img.resize((max(1, int(img.width * scale)),
                                max(1, int(img.height * scale))), Image.LANCZOS).save(fp)
                    man.append({"key": m["key"], "path": str(fp)})
            variants[f"v5@{scale}x"] = recognize(man, tmp, f"tta{int(scale*10)}",
                                                 MODELS_KO["v5"])
        print(f"\n{'variant':30s} {'exact':>7s} {'CER':>7s} {'WAR':>7s} {'contain':>8s}")
        for name, preds in variants.items():
            a = score(E, assemble(lay, preds), gts)
            mark = "  <- 배포 스케일" if name == "v5@1.0x" else ""
            print(f"{name:30s} {a.exact_rate()*100:6.1f}% {a.cer():7.3f} "
                  f"{a.war():7.3f} {a.contain_rate()*100:7.1f}%{mark}")
        keys = {m["key"] for m in base_man}
        vote = {}
        for k in keys:
            texts = [variants[n].get(k, "") for n in variants]
            norms = [nk(t) for t in texts]
            pick = texts[0]
            for t, nrm in zip(texts, norms):
                if nrm and norms.count(nrm) >= 2:
                    pick = t
                    break
            vote[k] = pick
        a = score(E, assemble(lay, vote), gts)
        print(f"{'TTA 다수결(동률 1.0x)':30s} {a.exact_rate()*100:6.1f}% {a.cer():7.3f} "
              f"{a.war():7.3f} {a.contain_rate()*100:7.1f}%")
        return

    if args.mode == "en-models":
        # Brooklyn is English, so the (Chinese/English) server recogniser is usable
        # zero-shot — no Korean pretrain exists for the server arch (4.9).
        man, lay = make_strips(polys, args.pad, tmp, "en")
        man = [m for m in man if m["key"].split("::")[0] == "brooklyn"]
        E.EN_ONLY = True
        gt_bk = gts["brooklyn"]
        print(f"\n{'model':26s} {'exact':>7s} {'CER':>7s} {'WAR':>7s} {'contain':>8s}")
        for name in ("en_PP-OCRv5_mobile_rec", "PP-OCRv5_server_rec",
                     "en_PP-OCRv4_mobile_rec", "PP-OCRv4_server_rec"):
            mm, oo = tmp / f"m_{name}.jsonl", tmp / f"o_{name}.jsonl"
            with mm.open("w", encoding="utf-8") as f:
                for m in man:
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")
            try:
                subprocess.run([str(PY), str(REC_WORKER), "--manifest", str(mm),
                                "--out", str(oo), "--device", "gpu",
                                "--model-name", name], check=True)
            except subprocess.CalledProcessError:
                print(f"{name:26s}   (사용 불가)")
                continue
            preds = {}
            with oo.open(encoding="utf-8") as f:
                for line in f:
                    d = json.loads(line)
                    preds[d["key"]] = d["text"]
            pm = {}
            for nm, n in lay["brooklyn"].items():
                parts = [(preds.get(f"brooklyn::{nm}::{li}") or "").strip() for li in range(n)]
                pm[nm] = "\n".join(p for p in parts if p)
            a = E.eval_engine(gt_bk, pm, [])
            mark = "  <- 배포" if name == "en_PP-OCRv5_mobile_rec" else ""
            print(f"{name:26s} {a.exact_rate()*100:6.1f}% {a.cer():7.3f} "
                  f"{a.war():7.3f} {a.contain_rate()*100:7.1f}%{mark}")
        E.EN_ONLY = False
        return

    # ---- ensemble mode: same strips, several recognizers, score-based selection
    pad = args.pad
    man, lay = make_strips(polys, pad, tmp, "ens")
    cand = {}
    for name, mdir in MODELS_KO.items():
        print(f"[REC] {name} ...")
        cand[name] = recognize(man, tmp, f"ens_{name}", mdir, want_scores=True)

    print(f"\n{'variant':28s} {'exact':>7s} {'CER':>7s} {'WAR':>7s} {'contain':>8s}")
    for name in MODELS_KO:
        a = score(E, assemble(lay, {k: v[0] for k, v in cand[name].items()}), gts)
        print(f"{'단독 ' + name:28s} {a.exact_rate()*100:6.1f}% {a.cer():7.3f} "
              f"{a.war():7.3f} {a.contain_rate()*100:7.1f}%")

    keys = {m["key"] for m in man}
    # S1: highest rec_score among the three
    sel1 = {k: max((cand[n].get(k, ("", 0.0)) for n in MODELS_KO), key=lambda t: t[1])[0]
            for k in keys}
    # S2: v5 unless its score is low AND another model is clearly more confident
    sel2 = {}
    for k in keys:
        base = cand["v5"].get(k, ("", 0.0))
        best = max((cand[n].get(k, ("", 0.0)) for n in MODELS_KO), key=lambda t: t[1])
        sel2[k] = best[0] if (base[1] < 0.80 and best[1] > base[1] + 0.15) else base[0]
    # S3: majority vote on normalized text, tie -> v5
    import re, unicodedata

    def nk(s):
        s = unicodedata.normalize("NFKC", s or "").lower()
        return re.sub(r"[^0-9a-z가-힣]", "", s)

    sel3 = {}
    for k in keys:
        texts = [cand[n].get(k, ("", 0.0))[0] for n in MODELS_KO]
        norm = [nk(t) for t in texts]
        pick = texts[0]
        for i, t in enumerate(norm):
            if t and norm.count(t) >= 2:
                pick = texts[i]
                break
        sel3[k] = pick
    # S4: rec_score selection, but never trade away a space v5 found. v5 is the
    # only model trained on real lines, so a space-less rival on a multi-word
    # line is a known failure mode (4.8) — not a genuine disagreement.
    sel4 = {}
    for k in keys:
        v5t = cand["v5"].get(k, ("", 0.0))[0]
        pick = sel1[k]
        sel4[k] = v5t if (" " in v5t and " " not in pick) else pick

    for label, sel in (("S1 최고 rec_score", sel1),
                       ("S2 v5 우선 + 저신뢰 폴백", sel2),
                       ("S3 다수결(동률 v5)", sel3),
                       ("S4 S1 + v5 공백 보호", sel4)):
        a = score(E, assemble(lay, sel), gts)
        print(f"{label:28s} {a.exact_rate()*100:6.1f}% {a.cer():7.3f} "
              f"{a.war():7.3f} {a.contain_rate()*100:7.1f}%")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
