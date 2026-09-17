#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_ocr.py  (GT: image_id,gt_text / OCR: image_name,gt_text)

- regions(brooklyn/gangnam/suwon) GT를 한 번에 평가
- crop 단위 key: suwon__1__crop_001 (확장자 없이)
- 평가 모드:
   --mode line_match (default)
     * GT를 라인( '|' 우선, 없으면 '\n')으로 분리
     * 각 GT 라인마다 pred 라인들(또는 전체 pred)과 비교하여 CER/WER 계산
     * 이미지 단위로 라인 평균 CER/WER
   --mode whole
     * GT 전체 vs Pred 전체로 CER/WER

- CER/WER:
   * CER: char-level edit distance / len(gt_chars)
   * WER: word-level edit distance / len(gt_words)
   * 둘 다 0이 best, 1 이상도 가능(삽입이 많으면)

중요:
- --ignore-space 옵션은 CER 계산에만 적용 (WER은 공백 유지하여 의미 있게)
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import chardet


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--mode", choices=["line_match", "whole"], default="line_match",
                   help="Evaluation mode. default=line_match")

    # paths
    p.add_argument("--base-dir", type=str, default=None,
                   help="Project base dir. Default: <this_file_dir>")

    # normalization
    p.add_argument("--no-lower", action="store_true", help="Disable lowercasing")
    p.add_argument("--strip-punct", action="store_true",
                   help="Strip most punctuation (keep 0-9 a-z A-Z 가-힣 and spaces). Off by default.")
    p.add_argument("--ignore-space", action="store_true",
                   help="CER only: remove ALL spaces before CER. (WER keeps spaces)")
    p.add_argument("--newline-to-space", action="store_true", default=True,
                   help="Replace '\\n' with space BEFORE normalization (default on)")

    # split rules
    p.add_argument("--split-prefer", choices=["auto", "bar", "newline"], default="auto",
                   help="How to split GT into lines for line_match. auto= '|' if present else '\\n'.")

    # output
    p.add_argument("--out-dir", type=str, default=None,
                   help="Output dir. Default: <BASE_DIR>/artifacts/gt")

    return p.parse_args()


ARGS = parse_args()


# ----------------------------
# Paths
# ----------------------------
HERE = Path(__file__).resolve().parents[1]
BASE_DIR = Path(ARGS.base_dir).resolve() if ARGS.base_dir else HERE

ARTIFACTS = BASE_DIR / "artifacts"
GT_DIR = ARTIFACTS / "gt"
OCR_DIR = ARTIFACTS / "ocr_gt"

OUT_DIR = Path(ARGS.out_dir).resolve() if ARGS.out_dir else GT_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

# GT files (너 구조)
GT_FILES = {
    "brooklyn": GT_DIR / "ocr_brooklyn_gt.csv",
    "gangnam":  GT_DIR / "ocr_gangnam_gt.csv",
    "suwon":    GT_DIR / "ocr_suwon_gt.csv",
}

# OCR result files (run_ocr_only.py 결과)
# 예: artifacts/ocr_gt/ocr_suwon_1_easyocr.csv
# run id는 평가 시점에 여러 개 있을 수 있으니, 가장 최신(파일 수정시간 기준) 1개를 사용하거나,
# 옵션으로 명시할 수도 있는데, 우선 "가장 최신"을 자동 선택하도록 함.
OCR_GLOB = re.compile(r"^ocr_(?P<region>[a-zA-Z]+)_(?P<run>\d+)_(?P<engine>easyocr|trocr)\.csv$")

OUT_IMAGELEVEL = OUT_DIR / "ocr_eval_imagelevel.csv"
OUT_DETAILS = OUT_DIR / "ocr_eval_details.csv"

print(f"[INFO] BASE_DIR = {BASE_DIR}")
print(f"[INFO] GT_DIR   = {GT_DIR}")
print(f"[INFO] OCR_DIR  = {OCR_DIR}")
print(f"[INFO] OUT_DIR  = {OUT_DIR}")
print(f"[INFO] mode     = {ARGS.mode}")


# ----------------------------
# Utilities: encoding + csv load
# ----------------------------
def detect_encoding(path: Path) -> str:
    raw = path.read_bytes()
    return chardet.detect(raw).get("encoding") or "utf-8"


def load_csv_map(path: Path) -> Dict[str, str]:
    """
    다양한 헤더를 자동 인식해서 key->text 맵으로 로드.
    Supported:
      - image_id, gt_text
      - image_name, gt_text
      - filename, ocr_text
      - filename, gt_text
      - image_id, ocr_text
    """
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    enc = detect_encoding(path)
    text = path.read_bytes().decode(enc, errors="replace")
    rdr = csv.DictReader(text.splitlines())

    if not rdr.fieldnames:
        raise ValueError(f"[ERROR] Empty header: {path}")

    fields = [f.strip() for f in rdr.fieldnames]

    key_candidates = ["image_id", "image_name", "filename"]
    text_candidates = ["gt_text", "ocr_text", "text"]

    key_col = next((c for c in key_candidates if c in fields), None)
    text_col = next((c for c in text_candidates if c in fields), None)

    if not key_col or not text_col:
        raise ValueError(f"[ERROR] {path.name} unsupported columns: {fields}")

    out: Dict[str, str] = {}
    for row in rdr:
        k = (row.get(key_col) or "").strip()
        v = (row.get(text_col) or "").strip()
        if not k:
            continue
        base_key = k.rsplit(".", 1)[0] if "." in k else k
        out[base_key] = v

    print(f"[INFO] Loaded {len(out)} rows from {path.name} (key={key_col}, text={text_col}, enc={enc})")
    return out


# ----------------------------
# Normalization + edit distance
# ----------------------------
def normalize_text(s: str, for_wer: bool) -> str:
    """
    for_wer=True: 단어 기준이므로 공백 유지 (multiple spaces -> single)
    for_wer=False: CER 용. ARGS.ignore_space면 공백 제거
    """
    s = (s or "").strip()

    if ARGS.newline_to_space:
        s = s.replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ")

    if not ARGS.no_lower:
        s = s.lower()

    if ARGS.strip_punct:
        s = re.sub(r"[^0-9a-zA-Z가-힣\s]", "", s)

    if for_wer:
        s = " ".join(s.split())
    else:
        if ARGS.ignore_space:
            s = s.replace(" ", "")
        else:
            s = " ".join(s.split())

    return s


def levenshtein_chars(a: str, b: str) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    if m < n:
        a, b = b, a
        n, m = m, n

    prev = list(range(m + 1))
    cur = [0] * (m + 1)

    for i in range(1, n + 1):
        cur[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev, cur = cur, prev

    return prev[m]


def levenshtein_words(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n

    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            tmp = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = tmp
    return dp[m]


def cer(pred: str, gt: str) -> float:
    p = normalize_text(pred, for_wer=False)
    g = normalize_text(gt, for_wer=False)
    if len(g) == 0:
        return 0.0
    return levenshtein_chars(p, g) / len(g)


def wer(pred: str, gt: str) -> float:
    p = normalize_text(pred, for_wer=True).split()
    g = normalize_text(gt, for_wer=True).split()
    if len(g) == 0:
        return 0.0
    return levenshtein_words(p, g) / len(g)


# ----------------------------
# Line split + line_match scoring
# ----------------------------
def split_lines(gt_text: str) -> List[str]:
    s = (gt_text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not s:
        return []

    mode = ARGS.split_prefer
    if mode == "bar":
        parts = s.split("|")
    elif mode == "newline":
        parts = s.split("\n")
    else:
        # auto
        parts = s.split("|") if "|" in s else s.split("\n")

    parts = [p.strip() for p in parts if p.strip()]
    return parts


def best_match_line(gt_line: str, pred_text: str) -> Tuple[float, float]:
    """
    여기서는 pred가 crop 1장의 OCR 결과라서 "후보가 1개"인 경우가 많음.
    그래도 구조를 유지해서 확장 가능하게 만든다.
    """
    return cer(pred_text, gt_line), wer(pred_text, gt_line)


def eval_line_match(gt_text: str, pred_text: str) -> Tuple[float, float, int]:
    """
    Returns: (mean_CER, mean_WER, num_lines)
    """
    gt_lines = split_lines(gt_text)
    if not gt_lines:
        return 0.0, 0.0, 0

    cers, wers = [], []
    for line in gt_lines:
        c, w = best_match_line(line, pred_text)
        cers.append(c)
        wers.append(w)

    return sum(cers) / len(cers), sum(wers) / len(wers), len(gt_lines)


def eval_whole(gt_text: str, pred_text: str) -> Tuple[float, float, int]:
    """
    whole-string CER/WER (num_lines=1)
    """
    return cer(pred_text, gt_text), wer(pred_text, gt_text), 1


# ----------------------------
# Sorting keys (region/photo/crop)
# ----------------------------
KEY_RE = re.compile(r"^(?P<region>[A-Za-z]+)__(?P<photo>\d+)__crop_(?P<crop>\d+)$")

def sort_key(k: str):
    m = KEY_RE.match(k)
    if not m:
        return ("zzz", 10**18, 10**18, k)
    return (m.group("region").lower(), int(m.group("photo")), int(m.group("crop")), k)


# ----------------------------
# OCR result file selection
# ----------------------------
def find_latest_ocr_files() -> Dict[Tuple[str, str], Path]:
    """
    return map[(region, engine)] = latest_csv_path
    """
    if not OCR_DIR.exists():
        raise FileNotFoundError(f"OCR_DIR not found: {OCR_DIR}")

    candidates: Dict[Tuple[str, str], List[Path]] = {}
    for p in OCR_DIR.glob("*.csv"):
        m = OCR_GLOB.match(p.name)
        if not m:
            continue
        region = m.group("region").lower()
        engine = m.group("engine").lower()
        candidates.setdefault((region, engine), []).append(p)

    picked: Dict[Tuple[str, str], Path] = {}
    for key, paths in candidates.items():
        paths.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        picked[key] = paths[0]

    # sanity print
    for (region, engine), path in sorted(picked.items()):
        print(f"[INFO] Using OCR file: {region}/{engine} -> {path.name}")

    return picked


# ----------------------------
# Main
# ----------------------------
def main():
    # 1) load GT (3 regions)
    gt_map: Dict[str, str] = {}
    gt_region_by_key: Dict[str, str] = {}

    for region, path in GT_FILES.items():
        if not path.exists():
            print(f"[WARN] GT missing: {path}")
            continue
        d = load_csv_map(path)
        for k, v in d.items():
            gt_map[k] = v
            gt_region_by_key[k] = region

    if not gt_map:
        raise RuntimeError("No GT loaded.")

    # 2) load OCR outputs (latest per region/engine)
    picked = find_latest_ocr_files()

    # load OCR maps
    ocr_easy: Dict[str, str] = {}
    ocr_tro: Dict[str, str] = {}

    for region in ["brooklyn", "gangnam", "suwon"]:
        ep = picked.get((region, "easyocr"))
        tp = picked.get((region, "trocr"))
        if ep:
            ocr_easy.update(load_csv_map(ep))
        else:
            print(f"[WARN] No EasyOCR csv for region={region}")
        if tp:
            ocr_tro.update(load_csv_map(tp))
        else:
            print(f"[WARN] No TrOCR csv for region={region}")

    # 3) evaluate intersection keys
    keys = sorted(gt_map.keys(), key=sort_key)

    # outputs
    with OUT_IMAGELEVEL.open("w", newline="", encoding="utf-8") as f_img, \
         OUT_DETAILS.open("w", newline="", encoding="utf-8") as f_det:

        w_img = csv.writer(f_img)
        w_det = csv.writer(f_det)

        w_img.writerow([
            "region", "image_id", "num_gt_lines",
            "easy_CER", "easy_WER",
            "trocr_CER", "trocr_WER",
            "best_engine", "best_CER", "best_WER"
        ])

        w_det.writerow([
            "engine", "region", "image_id", "gt_line_index", "gt_line_text",
            "pred_text", "CER", "WER"
        ])

        # stats
        easy_cers, easy_wers = [], []
        tro_cers, tro_wers = [], []
        best_cers, best_wers = [], []
        n_eval = 0

        for k in keys:
            region = gt_region_by_key.get(k, "unknown")
            gt_text = gt_map[k]

            pred_easy = ocr_easy.get(k, "")
            pred_tro  = ocr_tro.get(k, "")

            # mode switch
            if ARGS.mode == "whole":
                easy_c, easy_w, nlines = eval_whole(gt_text, pred_easy)
                tro_c,  tro_w,  _      = eval_whole(gt_text, pred_tro)
                gt_lines = [gt_text]
            else:
                easy_c, easy_w, nlines = eval_line_match(gt_text, pred_easy)
                tro_c,  tro_w,  _      = eval_line_match(gt_text, pred_tro)
                gt_lines = split_lines(gt_text)

            # details row (라인 단위)
            for i, line in enumerate(gt_lines):
                # Easy
                w_det.writerow(["easyocr", region, k, i, line, pred_easy, f"{cer(pred_easy, line):.4f}", f"{wer(pred_easy, line):.4f}"])
                # TrOCR
                w_det.writerow(["trocr", region, k, i, line, pred_tro,  f"{cer(pred_tro,  line):.4f}", f"{wer(pred_tro,  line):.4f}"])

            # best pick: CER 우선, tie-break WER
            if (easy_c < tro_c) or (abs(easy_c - tro_c) < 1e-12 and easy_w <= tro_w):
                best_engine = "easyocr"
                best_c, best_w = easy_c, easy_w
            else:
                best_engine = "trocr"
                best_c, best_w = tro_c, tro_w

            w_img.writerow([
                region, k, nlines,
                f"{easy_c:.4f}", f"{easy_w:.4f}",
                f"{tro_c:.4f}",  f"{tro_w:.4f}",
                best_engine, f"{best_c:.4f}", f"{best_w:.4f}"
            ])

            # collect stats (키가 둘 중 하나라도 있는 경우)
            n_eval += 1
            easy_cers.append(easy_c); easy_wers.append(easy_w)
            tro_cers.append(tro_c);   tro_wers.append(tro_w)
            best_cers.append(best_c); best_wers.append(best_w)

    def mean(xs: List[float]) -> float:
        return sum(xs)/len(xs) if xs else 0.0

    print("\n========== Global Summary ==========")
    print(f"EASY   CER={mean(easy_cers):.4f}, WER={mean(easy_wers):.4f}")
    print(f"TROCR  CER={mean(tro_cers):.4f}, WER={mean(tro_wers):.4f}")
    print(f"BEST   CER={mean(best_cers):.4f}, WER={mean(best_wers):.4f}  (per-image best engine)")
    print(f"[INFO] Evaluated keys: {n_eval}")
    print(f"[INFO] Saved:")
    print(f" - {OUT_IMAGELEVEL}")
    print(f" - {OUT_DETAILS}")


if __name__ == "__main__":
    main()
