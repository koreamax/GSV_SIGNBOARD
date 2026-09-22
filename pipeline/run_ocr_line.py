#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""배포 OCR 파이프라인 — 단어 검출 → 라인 병합 → 라인 통째 인식 (D20/D38/D40).

  detection : 학습된 YOLO26x 단어 탐지기 (str_baselines/yolo_text_det_worker.py, conf 0.01)
              CRAFT ∪ PaddleOCR-DB 였던 이전 구성은 D38 에서 교체됐고 코드도 제거했습니다.
  merge     : 라인별 y-중심 클러스터링 → union bbox + pad 0.04 → 스트립 한 장
  recognize : 기본은 PaddleOCR 워커. --worker 로 교체하면 인식기만 바뀝니다
              (배포는 SVTRv2-B = openocr_rec_worker.py — D40)

박스별로 자르지 않고 라인째 인식하는 것이 핵심입니다(per-box 대비 +16.5%p, 4.7).
artifacts/ocr_gt/ocr_{region}_{run}_{engine}.csv 로 떨어집니다.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

HERE = Path(__file__).resolve().parents[1]

import sys as _sys; _sys.path[:0] = [str(HERE / _d) for _d in ("pipeline", "ocr", "vlm", "str_baselines", "detection", "data")]  # 폴더 재편 후 형제 모듈 import 경로
CROP = HERE / "artifacts" / "gt" / "crop"
OCR_DIR = HERE / "artifacts" / "ocr_gt"
PY = HERE / ".venv" / "Scripts" / "python.exe"
WORKER = HERE / "str_baselines/paddle_rec_worker.py"


# region -> (reader kind, paddle inference dir or None=pretrained en)
# Korean default = v5_lines (real line crops + spacecat, D21) — v4/v3 via --ko-model-dir.
REGION_CFG = {
    "gangnam": ("ko", HERE / "output" / "paddle_signboard_rec_v5_lines" / "inference"),
    "suwon": ("ko", HERE / "output" / "paddle_signboard_rec_v5_lines" / "inference"),
    "brooklyn": ("en", None),
}
# Korean-region line-level vote members (D24). Every candidate reads the SAME
# line strip, so agreement is meaningful — unlike the per-box era, where the
# candidates disagreed about granularity rather than about the text (4.10/C1).
VOTE_KO = [
    ("v5", HERE / "output" / "paddle_signboard_rec_v5_lines" / "inference"),
    ("v4", HERE / "output" / "paddle_signboard_rec_v4_spacecat" / "inference"),
    ("pre", None),          # zero-shot korean_PP-OCRv5_mobile_rec
]


def norm_key(s: str) -> str:
    """Vote key: NFKC + lowercase + strip everything but letters/digits."""
    import re
    import unicodedata
    return re.sub(r"[^0-9a-z가-힣]", "",
                  unicodedata.normalize("NFKC", s or "").lower())
FILTER_ARGS = SimpleNamespace(filt_min_conf=0.10, filt_numeric_digits=7,
                              filt_numeric_ratio=0.6, filt_min_h_ratio=0.0,
                              filt_min_h_frac=0.0, no_filter=False)


def natural_key(s: str):
    import re
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def _bbox(poly):
    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def group_lines(polys, img_h: int, y_tol: float = 0.06):
    """Reading-order line grouping by y-center (same geometry as run12 CRAFT
    grouping — verified to reproduce the CRAFT-only baseline exactly, D23)."""
    items = []
    for p in polys:
        x0, y0, x1, y1 = _bbox(p)
        items.append({"poly": p, "x": x0, "yc": (y0 + y1) / 2})
    items.sort(key=lambda d: d["yc"])
    lines: list[list[dict]] = []
    for it in items:
        for ln in lines:
            if abs(it["yc"] - sum(d["yc"] for d in ln) / len(ln)) <= y_tol * img_h:
                ln.append(it)
                break
        else:
            lines.append([it])
    for ln in lines:
        ln.sort(key=lambda d: d["x"])
    lines.sort(key=lambda ln: sum(d["yc"] for d in ln) / len(ln))
    return lines


def crop_strip(img: Image.Image, polys, pad: float = 0.04):
    """Strip crop with a margin of `pad` x strip width/height on each side.

    pad is deliberately small: the margin scales with the strip's own WIDTH, so
    on a wide line a large pad reaches sideways into the neighbouring sign.
    Swept 0.0~0.35 — 0.04 is the optimum (68.8% vs 66.0% at the old 0.12); see 4.10.
    """
    pts = [p for poly in polys for p in poly]
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    X0, Y0 = max(0, int(x0 - px)), max(0, int(y0 - py))
    X1, Y1 = min(img.width, int(x1 + px)), min(img.height, int(y1 + py))
    if X1 <= X0 or Y1 <= Y0:
        return None
    return img.crop((X0, Y0, X1, Y1))


def yolo_detect(files_by_region: dict[str, list[Path]], tmp: Path, conf: float) -> dict[str, list]:
    """학습된 YOLO26x 단어 탐지기 박스 (별도 프로세스 — paddle_det_worker 와 같은 계약)."""
    man, out = tmp / "ydet_man.jsonl", tmp / "ydet_out.jsonl"
    with man.open("w", encoding="utf-8") as f:
        for region, files in files_by_region.items():
            for p in files:
                f.write(json.dumps({"key": f"{region}::{p.stem}", "path": str(p)}) + "\n")
    subprocess.run([str(PY), str(HERE / "str_baselines" / "yolo_text_det_worker.py"),
                    "--manifest", str(man), "--out", str(out), "--conf", str(conf)], check=True)
    boxes = {}
    with out.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            boxes[d["key"]] = d["boxes"]
    return boxes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--crop-dir", default=None,
                    help="연쇄 평가용: artifacts/gt/crop 대신 쓸 크롭 루트 "
                         "(예: artifacts/gt/crop_det). 전처리·모델은 동일")
    ap.add_argument("--regions", default="gangnam,suwon,brooklyn")
    ap.add_argument("--ko-model-dir", type=Path, default=None,
                    help="override Korean-region inference dir (default: v5_lines)")
    ap.add_argument("--text-detector", choices=["yolo"], default="yolo",
                    help="배포 검출기는 학습된 YOLO26x 단어 탐지기 하나입니다 (D38). "
                         "이전의 CRAFT/union 경로는 제거됐습니다")
    ap.add_argument("--yolo-det-conf", type=float, default=0.25,
                    help="단어 탐지기 confidence 임계값 (배포 0.01 — 0.35~0.001 스윕의 최댓값)")
    ap.add_argument("--pad", type=float, default=0.04,
                    help="strip crop margin (fraction of strip w/h); 0.04 = swept optimum")
    ap.add_argument("--y-tol", type=float, default=0.04,
                    help="line grouping tolerance (fraction of image height); 0.04 = swept optimum")
    ap.add_argument("--no-vote", action="store_true",
                    help="Korean regions: single v5 model instead of the 3-way line vote")
    ap.add_argument("--worker", default=None,
                    help="다른 인식기 워커 스크립트(paddle_rec_worker IO 계약). 지정 시 검출·라인 병합은 "
                         "동일하고 인식만 이 워커가 모든 지역 스트립을 처리 (예: tesseract_rec_worker.py)")
    ap.add_argument("--worker-py", default=str(PY), help="워커를 실행할 python.exe (기본 .venv)")
    ap.add_argument("--worker-args", default="", help="워커 추가 인자 (공백 구분 문자열)")
    ap.add_argument("--engine-tag", default="paddle",
                    help="출력 CSV 엔진 이름: ocr_<region>_<run>_<engine-tag>.csv")
    args = ap.parse_args()
    regions = [r.strip() for r in args.regions.split(",") if r.strip()]
    if args.ko_model_dir:
        for r, (kind, mdir) in list(REGION_CFG.items()):
            if mdir is not None:
                REGION_CFG[r] = (kind, args.ko_model_dir)
        print(f"[CFG] Korean-region model: {args.ko_model_dir}")

    tmp = Path(tempfile.mkdtemp(prefix="ocr_line_"))
    manifests: dict[str, list[dict]] = {}      # model tag -> manifest
    layout: dict[str, dict[str, int]] = {}
    model_of_region: dict[str, str] = {}

    crop_root = Path(args.crop_dir) if args.crop_dir else CROP
    files_by_region = {r: sorted((crop_root / r).glob("*.jpg"), key=lambda p: natural_key(p.stem))
                       for r in regions}

    # ---- 검출: 학습된 YOLO26x 단어 탐지기 (D38) ----
    print(f"[DETECT/YOLO] trained YOLO26x word detector (conf {args.yolo_det_conf}) ...")
    boxes_by_key = yolo_detect(files_by_region, tmp, args.yolo_det_conf)
    print(f"[DETECT] boxes={sum(len(v) for v in boxes_by_key.values())}")

    # ---- line grouping + strip crops ----
    for region in regions:
        kind, mdir = REGION_CFG[region]
        tag = "en" if mdir is None else "ko"
        model_of_region[region] = tag
        manifests.setdefault(tag, [])
        layout[region] = {}
        for img_path in files_by_region[region]:
            img = Image.open(img_path).convert("RGB")
            polys = boxes_by_key.get(f"{region}::{img_path.stem}", [])
            n = 0
            for li, ln in enumerate(group_lines(polys, img.height, args.y_tol) if polys else []):
                strip = crop_strip(img, [d["poly"] for d in ln], pad=args.pad)
                if strip is None:
                    continue
                p = tmp / f"{region}_{img_path.stem}_{li}.png"
                strip.save(p)
                manifests[tag].append({"key": f"{region}::{img_path.stem}::{li}",
                                       "path": str(p)})
                n += 1
            layout[region][img_path.stem] = n

    def run_rec(manifest, mdir, label):
        man, out = tmp / f"man_{label}.jsonl", tmp / f"out_{label}.jsonl"
        with man.open("w", encoding="utf-8") as f:
            for m in manifest:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        cmd = [str(PY), str(WORKER), "--manifest", str(man), "--out", str(out),
               "--device", "gpu"]
        if label.startswith("en"):
            cmd += ["--model-name", "en_PP-OCRv5_mobile_rec"]
        elif mdir is None:
            cmd += ["--model-name", "korean_PP-OCRv5_mobile_rec"]
        else:
            cmd += ["--model-dir", str(mdir)]
        print(f"[REC] paddle worker: {label} ({len(manifest)} strips)")
        subprocess.run(cmd, check=True)
        got = {}
        with out.open(encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                got[d["key"]] = d["text"]
        return got

    preds: dict[str, str] = {}
    if args.worker:
        import shlex
        allman = [m for tag in manifests for m in manifests[tag]]
        man, out = tmp / "man_worker.jsonl", tmp / "out_worker.jsonl"
        with man.open("w", encoding="utf-8") as f:
            for m in allman:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        cmd = [args.worker_py, args.worker, "--manifest", str(man), "--out", str(out)] + shlex.split(args.worker_args)
        print(f"[REC] worker {Path(args.worker).name} ({len(allman)} strips): {' '.join(cmd[2:])}")
        env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        subprocess.run(cmd, check=True, env=env)
        with out.open(encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                preds[d["key"]] = d["text"]
    elif manifests.get("en"):
        preds.update(run_rec(manifests["en"], None, "en"))
    ko_man = [] if args.worker else (manifests.get("ko") or [])
    if ko_man:
        if args.no_vote:
            preds.update(run_rec(ko_man, REGION_CFG["gangnam"][1], "ko"))
        else:
            # D24: majority vote over line-level candidates, ties -> v5.
            votes = {name: run_rec(ko_man, mdir, f"ko_{name}")
                     for name, mdir in VOTE_KO}
            n_over = 0
            for m in ko_man:
                k = m["key"]
                texts = [votes[name].get(k, "") for name, _ in VOTE_KO]
                norms = [norm_key(t) for t in texts]
                pick = texts[0]                       # v5 = default / tiebreak
                for t, nrm in zip(texts, norms):
                    if nrm and norms.count(nrm) >= 2:
                        pick = t
                        break
                if norm_key(pick) != norms[0]:
                    n_over += 1
                preds[k] = pick
            print(f"[VOTE] v5 출력이 다수결로 교체된 라인: {n_over}/{len(ko_man)}")

    for region in regions:
        dst = OCR_DIR / f"ocr_{region}_{args.run}_{args.engine_tag}.csv"
        with dst.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, quoting=csv.QUOTE_ALL)
            w.writerow(["image_name", "gt_text"])
            for name in sorted(layout[region], key=natural_key):
                parts = [preds.get(f"{region}::{name}::{li}", "").strip()
                         for li in range(layout[region][name])]
                w.writerow([name, "\\n".join(p for p in parts if p)])
        print(f"[EMIT] {dst.name} ({len(layout[region])} crops, model={model_of_region[region]})")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
