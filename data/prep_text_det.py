"""
Prepare Signboard TEXT (word-level) detection dataset for 5-fold CV.
- Select 5000 images (seed 42) that have >=1 word box and exist on disk.
- Convert word bboxes [x,y,w,h] -> YOLO labels (single class 0 'text').
- Hardlink images (no disk dup) into artifacts/signboard_text/images.
- Build 5-fold txt lists + per-fold data.yaml (Ultralytics derives labels via images->labels).
"""
import json, random, os, shutil
from collections import defaultdict
from pathlib import Path

random.seed(42)
SRC = Path("artifacts/Signboard")
OUT = Path("artifacts/signboard_text")
(OUT / "images").mkdir(parents=True, exist_ok=True)
(OUT / "labels").mkdir(parents=True, exist_ok=True)
N = 5000

d = json.load(open("artifacts/signboard_data_info.json", encoding="utf-8"))
imgs = {im["id"]: im for im in d["images"]}
words = defaultdict(list)
for a in d["annotations"]:
    if a["attributes"]["class"] == "word":
        words[a["image_id"]].append(a["bbox"])

cand = [iid for iid in words if (SRC / imgs[iid]["file_name"]).exists()]
cand.sort()
random.shuffle(cand)
sel = cand[:N]
print(f"candidates={len(cand)} selected={len(sel)}")

nb = 0
for iid in sel:
    im = imgs[iid]; fn = im["file_name"]; W = im["width"]; H = im["height"]
    dst = OUT / "images" / fn
    if not dst.exists():
        try:
            os.link(SRC / fn, dst)
        except OSError:
            shutil.copy(SRC / fn, dst)
    lines = []
    for (x, y, bw, bh) in words[iid]:
        cx = (x + bw / 2) / W; cy = (y + bh / 2) / H; ww = bw / W; hh = bh / H
        cx = min(max(cx, 0), 1); cy = min(max(cy, 0), 1)
        ww = min(max(ww, 0), 1); hh = min(max(hh, 0), 1)
        if ww > 0 and hh > 0:
            lines.append(f"0 {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}"); nb += 1
    (OUT / "labels" / (Path(fn).stem + ".txt")).write_text("\n".join(lines))
print(f"total word boxes written = {nb}")

# 5-fold (round-robin on a shuffled copy)
order = list(sel)
random.seed(42); random.shuffle(order)
folds = [order[i::5] for i in range(5)]
for i in range(5):
    val = set(folds[i])
    val_l = [f"./images/{imgs[iid]['file_name']}" for iid in order if iid in val]
    tr_l = [f"./images/{imgs[iid]['file_name']}" for iid in order if iid not in val]
    (OUT / f"fold{i}_val.txt").write_text("\n".join(val_l))
    (OUT / f"fold{i}_train.txt").write_text("\n".join(tr_l))
    (OUT / f"fold{i}.yaml").write_text(
        f"path: {OUT.resolve()}\ntrain: fold{i}_train.txt\nval: fold{i}_val.txt\nnc: 1\nnames: ['text']\n")
    print(f"fold{i}: train={len(tr_l)} val={len(val_l)}")
print("DONE prep ->", OUT.resolve())
