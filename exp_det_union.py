#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C3/D23: does a second detector (PaddleOCR DB) recover Suwon's lost lines?

Suwon's residual failures are lines that appear NOWHERE in the prediction
(vertical stacks, embossed low-contrast, calligraphy — 4.9/D22). CRAFT
threshold sweeps were negative, so try a structurally different detector.

Variants (all recognized by the SAME model, line-merged the same way):
  craft      : current deployed detection (baseline)
  db         : PaddleOCR DB boxes only
  union      : CRAFT boxes + DB boxes that don't overlap any CRAFT box (IoU-ish)

Line grouping/merging reuses run_ocr_only.easyocr_detect_lines' geometry rules
via a local reimplementation (boxes come from two sources, so grouping runs on
plain polygons here).

Usage: .venv/Scripts/python.exe exp_det_union.py [--region suwon]
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
CROP = HERE / "artifacts" / "gt" / "crop"
PY = HERE / ".venv" / "Scripts" / "python.exe"
REC_WORKER = HERE / "paddle_rec_worker.py"
DET_WORKER = HERE / "paddle_det_worker.py"
V5 = HERE / "output" / "paddle_signboard_rec_v5_lines" / "inference"
EN = None


def natural_key(s: str):
    import re
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def bbox(poly):
    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def overlaps(a, b, thr=0.3):
    ax0, ay0, ax1, ay1 = bbox(a); bx0, by0, bx1, by1 = bbox(b)
    ix = max(0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    if inter <= 0:
        return False
    amin = min((ax1 - ax0) * (ay1 - ay0), (bx1 - bx0) * (by1 - by0))
    return inter / max(amin, 1) > thr


def group_lines(polys, y_tol=0.06, img_h=1):
    """Group polygons into reading-order lines (y-center clustering)."""
    items = []
    for p in polys:
        x0, y0, x1, y1 = bbox(p)
        items.append({"poly": p, "x": x0, "yc": (y0 + y1) / 2, "h": y1 - y0})
    items.sort(key=lambda d: d["yc"])
    lines = []
    for it in items:
        placed = False
        for ln in lines:
            ryc = sum(d["yc"] for d in ln) / len(ln)
            if abs(it["yc"] - ryc) <= y_tol * img_h:
                ln.append(it); placed = True; break
        if not placed:
            lines.append([it])
    for ln in lines:
        ln.sort(key=lambda d: d["x"])
    lines.sort(key=lambda ln: sum(d["yc"] for d in ln) / len(ln))
    return lines


def crop_strip(img, polys, pad=0.12):
    pts = [p for poly in polys for p in poly]
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    w, h = x1 - x0, y1 - y0
    px, py = w * pad, h * pad
    X0, Y0 = max(0, int(x0 - px)), max(0, int(y0 - py))
    X1, Y1 = min(img.width, int(x1 + px)), min(img.height, int(y1 + py))
    if X1 <= X0 or Y1 <= Y0:
        return None
    return img.crop((X0, Y0, X1, Y1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="suwon")
    ap.add_argument("--en", action="store_true",
                    help="brooklyn: English CRAFT reader + pretrained en rec model")
    args = ap.parse_args()
    region = args.region
    tmp = Path(tempfile.mkdtemp(prefix="det_union_"))
    files = sorted((CROP / region).glob("*.jpg"), key=lambda p: natural_key(p.stem))
    print(f"[C3] {region}: {len(files)} crops  tmp={tmp}")

    # ---- CRAFT boxes (torch process) ----
    craft_json = tmp / "craft_boxes.json"
    reader_expr = ('easyocr.Reader(["en"], gpu=torch.cuda.is_available())' if args.en else
                   'easyocr.Reader(["ko","en"], gpu=torch.cuda.is_available(), '
                   'recog_network="signboard_v3_custom")')
    rec_model = "" if args.en else str(V5)
    code = f'''
import json, sys
from pathlib import Path
from PIL import Image
import torch, easyocr
sys.path.insert(0, r"{HERE}")
import run_ocr_only as RO
from types import SimpleNamespace
FA = SimpleNamespace(filt_min_conf=0.10, filt_numeric_digits=7, filt_numeric_ratio=0.6,
                     filt_min_h_ratio=0.0, filt_min_h_frac=0.0, no_filter=False)
reader = {reader_expr}
out = {{}}
for p in sorted(Path(r"{CROP / region}").glob("*.jpg")):
    img = Image.open(p).convert("RGB")
    lines = RO.easyocr_detect_lines(reader, img, line_y_tol=0.06, craft_text_threshold=0.7,
                                    craft_link_threshold=0.4, craft_low_text=0.4)
    lines, _ = RO.filter_boxes(lines, img.height, FA)
    out[p.stem] = [[[float(c[0]), float(c[1])] for c in b["bbox"]] for ln in lines for b in ln]
json.dump(out, open(r"{craft_json}", "w", encoding="utf-8"))
print("craft done", len(out))
'''
    (tmp / "craft.py").write_text(code, encoding="utf-8")
    subprocess.run([str(PY), str(tmp / "craft.py")], check=True)
    craft = json.loads(craft_json.read_text(encoding="utf-8"))

    # ---- DB boxes (paddle process) ----
    man = tmp / "det_man.jsonl"
    with man.open("w", encoding="utf-8") as f:
        for p in files:
            f.write(json.dumps({"key": p.stem, "path": str(p)}) + "\n")
    det_out = tmp / "det_out.jsonl"
    subprocess.run([str(PY), str(DET_WORKER), "--manifest", str(man),
                    "--out", str(det_out), "--device", "gpu"], check=True)
    db = {}
    with det_out.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            db[d["key"]] = d["boxes"]

    n_craft = sum(len(v) for v in craft.values())
    n_db = sum(len(v) for v in db.values())
    print(f"[C3] boxes: CRAFT={n_craft}  DB={n_db}")

    # ---- build strips per variant ----
    variants = {}
    for tag in ("craft", "db", "union"):
        manifest, layout = [], {}
        for p in files:
            img = Image.open(p).convert("RGB")
            c = craft.get(p.stem, []); d = db.get(p.stem, [])
            if tag == "craft":
                polys = c
            elif tag == "db":
                polys = d
            else:
                extra = [b for b in d if not any(overlaps(b, cb) for cb in c)]
                polys = c + extra
            lines = group_lines(polys, img_h=img.height) if polys else []
            n = 0
            for li, ln in enumerate(lines):
                strip = crop_strip(img, [it["poly"] for it in ln])
                if strip is None:
                    continue
                fp = tmp / f"{tag}_{p.stem}_{li}.png"
                strip.save(fp)
                manifest.append({"key": f"{p.stem}::{li}", "path": str(fp)})
                n += 1
            layout[p.stem] = n
        variants[tag] = (manifest, layout)
        print(f"[C3] {tag}: {len(manifest)} strips")

    # ---- recognize each variant with v5 ----
    results = {}
    for tag, (manifest, layout) in variants.items():
        m, o = tmp / f"rec_man_{tag}.jsonl", tmp / f"rec_out_{tag}.jsonl"
        with m.open("w", encoding="utf-8") as f:
            for it in manifest:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        cmd = [str(PY), str(REC_WORKER), "--manifest", str(m), "--out", str(o), "--device", "gpu"]
        cmd += (["--model-name", "en_PP-OCRv5_mobile_rec"] if args.en
                else ["--model-dir", str(V5)])
        subprocess.run(cmd, check=True)
        preds = {}
        with o.open(encoding="utf-8") as f:
            for line in f:
                dd = json.loads(line)
                preds[dd["key"]] = dd["text"]
        pmap = {}
        for name, n in layout.items():
            parts = [preds.get(f"{name}::{li}", "").strip() for li in range(n)]
            pmap[name] = "\n".join(x for x in parts if x)
        results[tag] = pmap

    # ---- eval ----
    saved = sys.argv
    sys.argv = ["eval_ocr_v2.py", "--mask-phone", "--en-only-regions", "brooklyn"]
    import eval_ocr_v2 as E
    sys.argv = saved
    E.EN_ONLY = region in E.EN_ONLY_REGIONS
    gt = E.load_csv_map(HERE / "artifacts" / "gt" / f"ocr_{region}_gt.csv")
    deployed = E.load_csv_map(HERE / "artifacts" / "ocr_gt" /
                              f"ocr_{region}_{'19' if region=='brooklyn' else '21'}_paddle.csv")

    def notread(pmap):
        n = 0
        for key, gtext in gt.items():
            gl, ndc = E.split_gt_lines(gtext)
            if not gl: continue
            pl = E.split_lines(pmap.get(key, "")) if pmap.get(key, "").strip() else []
            assign, _ = E.match_lines(gl, pl, infix=(ndc > 0))
            n += sum(1 for i in range(len(gl)) if assign[i][1] >= assign[i][2])
        return n

    print(f"\n{'variant':12s} {'strips':>7s} {'exact':>7s} {'CER':>7s} {'WAR':>7s} {'contain':>8s} {'전멸라인':>8s}")
    a = E.eval_engine(gt, deployed, [])
    print(f"{'배포(run21)':12s} {'-':>7s} {a.exact_rate()*100:6.1f}% {a.cer():7.3f} "
          f"{a.war():7.3f} {a.contain_rate()*100:7.1f}% {notread(deployed):8d}")
    for tag, pmap in results.items():
        a = E.eval_engine(gt, pmap, [])
        print(f"{tag:12s} {len(variants[tag][0]):7d} {a.exact_rate()*100:6.1f}% {a.cer():7.3f} "
              f"{a.war():7.3f} {a.contain_rate()*100:7.1f}% {notread(pmap):8d}")
    E.EN_ONLY = False


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
