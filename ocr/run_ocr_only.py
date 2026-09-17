#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_ocr_only.py (NO PADDLE / ORDER FIXED / CRAFT DETECTOR THRESHOLDS)

- EasyOCR ordered reading (line grouping)
- Use EasyOCR(CRAFT) detector thresholds:
    text_threshold / link_threshold / low_text
  (Top-tier style: detection-stage thresholding)
- If detector returns nothing: output "" for BOTH (EasyOCR + TrOCR)
- TrOCR + EasyOCR OCR
- OPTIONAL: strict normalize (KO/EN only) + optional top-tier postprocess

Outputs:
- artifacts/ocr_gt/ocr_{region}_{run}_trocr.csv
- artifacts/ocr_gt/ocr_{region}_{run}_easyocr.csv
"""

import argparse
import csv
import re
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from PIL import Image

import easyocr
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


BASE_DIR = Path(__file__).resolve().parents[1] / "artifacts"
CROP_DIR = BASE_DIR / "gt" / "crop"
OUT_DIR = BASE_DIR / "ocr_gt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

_num_re = re.compile(r"\d+")


# =========================================================
# Utils
# =========================================================
def natural_key(s: str):
    parts = []
    last = 0
    for m in _num_re.finditer(s):
        if m.start() > last:
            parts.append(s[last:m.start()])
        parts.append(int(m.group()))
        last = m.end()
    if last < len(s):
        parts.append(s[last:])
    return parts


def clean_text_basic(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"(.)\1{4,}", r"\1\1", text)
    return text


def normalize_text_strict(text: str) -> str:
    """Keep only digits/English/Hangul/spaces."""
    if not text:
        return ""
    text = re.sub(r"[^0-9A-Za-z가-힣\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"(.)\1{4,}", r"\1\1", text)
    return text


def is_junk_only(text: str) -> bool:
    if not text:
        return True
    t = text.strip()
    return not bool(re.search(r"[0-9A-Za-z가-힣]", t))


def remove_year_google_citations(text: str) -> str:
    """
    Remove patterns like:
      - '2022 Google', '2021 Google'
      - 'Google 2022', 'Google, 2022', 'Google (2022)'
    """
    if not text:
        return ""
    t = text
    t = re.sub(r"\b(19|20)\d{2}\s*[,()\-:]*\s*Google\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\bGoogle\s*[,()\-:]*\s*(19|20)\d{2}\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s*([,.:;])\s*", r"\1 ", t).strip()
    return t


# =========================================================
# TOP-TIER Post-processing Pack (optional; no LLM)
# =========================================================
CONFUSION_MAP_DIGIT = {
    "O": "0", "o": "0", "D": "0", "Q": "0",
    "I": "1", "l": "1", "L": "1", "|": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
    "G": "6",
    "T": "7",
}
CONFUSION_MAP_ALPHA = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "6": "G",
    "7": "T",
    "8": "B",
}

DEFAULT_EN_LEXICON = {
    "OPEN", "CLOSED", "HOURS", "HOUR", "AM", "PM", "B1", "B2", "FLOOR",
    "COFFEE", "CAFE", "MUSIC", "TOWN", "MILK", "SHOP", "STORE",
    "KOREA", "SEOUL", "GANGNAM", "SUWON", "BROOKLYN",
    "JAPANESE", "SPAGHETTI", "RESTAURANT", "BAR", "HOTEL",
    "ENTRANCE", "EXIT", "WELCOME", "THANK", "YOU",
}

def load_lexicon_txt(path: Path) -> set:
    if not path or not Path(path).exists():
        return set()
    out = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                out.add(t.upper())
    return out

def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) > len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]

def best_lexicon_match(token: str, lexicon: set, max_dist: int = 2) -> str:
    if not token or not lexicon:
        return token
    if token in lexicon:
        return token
    best = token
    best_d = 10**9
    L = len(token)
    candidates = [w for w in lexicon if abs(len(w) - L) <= max_dist]
    for w in candidates:
        d = levenshtein(token, w)
        if d < best_d:
            best_d = d
            best = w
    return best if best_d <= max_dist else token

def collapse_repeat_units(text: str) -> str:
    if not text:
        return ""
    toks = text.split()
    if not toks:
        return ""
    out = [toks[0]]
    for t in toks[1:]:
        if t == out[-1]:
            continue
        out.append(t)
    return " ".join(out)

def strip_single_char_noise(text: str, keep_digits=True) -> str:
    if not text:
        return ""
    toks = text.split()
    out = []
    for t in toks:
        if len(t) == 1:
            if keep_digits and t.isdigit():
                out.append(t)
            continue
        out.append(t)
    return " ".join(out)

def token_class(token: str) -> str:
    if not token:
        return "empty"
    has_alpha = bool(re.search(r"[A-Za-z]", token))
    has_digit = bool(re.search(r"\d", token))
    has_hangul = bool(re.search(r"[가-힣]", token))
    if has_hangul and not (has_alpha or has_digit):
        return "hangul"
    if has_alpha and not has_digit:
        return "alpha"
    if has_digit and not has_alpha:
        return "digit"
    if has_alpha and has_digit:
        return "alnum"
    return "other"

def apply_confusion_fix(token: str) -> str:
    if not token:
        return ""
    ttype = token_class(token)
    if ttype == "digit":
        return "".join(CONFUSION_MAP_DIGIT.get(ch, ch) for ch in token)
    if ttype == "alpha":
        return "".join(CONFUSION_MAP_ALPHA.get(ch, ch) for ch in token)
    if ttype == "alnum":
        out = list(token)
        for i, ch in enumerate(out):
            if ch in ("O", "o") and ((i > 0 and out[i-1].isdigit()) or (i+1 < len(out) and out[i+1].isdigit())):
                out[i] = "0"
            if ch in ("I", "l") and ((i > 0 and out[i-1].isdigit()) or (i+1 < len(out) and out[i+1].isdigit())):
                out[i] = "1"
        return "".join(out)
    return token

def fix_common_patterns_tokens(tokens: List[str]) -> List[str]:
    if not tokens:
        return tokens
    out = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if i + 1 < len(tokens) and len(t) == 1 and t.isalpha() and tokens[i + 1].isdigit():
            out.append(t + tokens[i + 1])
            i += 2
            continue
        if i + 1 < len(tokens) and tokens[i].isdigit() and tokens[i + 1].upper() in ("AM", "PM"):
            out.append(tokens[i] + tokens[i + 1].upper())
            i += 2
            continue
        out.append(t)
        i += 1
    return out

def postprocess_text_top_tier(
    text: str,
    *,
    use_lexicon: bool = False,
    lexicon: Optional[set] = None,
    max_edit_dist: int = 2,
    force_case: str = "none",
    drop_single_char: bool = True,
) -> str:
    if not text:
        return ""
    t = clean_text_basic(text)
    if not t:
        return ""

    if force_case == "upper":
        t = t.upper()
    elif force_case == "lower":
        t = t.lower()

    toks = t.split()
    toks2 = [apply_confusion_fix(tok) for tok in toks]
    toks2 = fix_common_patterns_tokens(toks2)

    if use_lexicon:
        LEX = lexicon if lexicon is not None else DEFAULT_EN_LEXICON
        corrected = []
        for tok in toks2:
            ttype = token_class(tok)
            if ttype in ("alpha", "alnum") and re.search(r"[A-Za-z]", tok):
                corrected.append(best_lexicon_match(tok.upper(), LEX, max_dist=max_edit_dist))
            else:
                corrected.append(tok)
        toks2 = corrected

    out = " ".join(toks2)
    out = collapse_repeat_units(out)

    if drop_single_char:
        out = strip_single_char_noise(out, keep_digits=True)

    return clean_text_basic(out)


# =========================================================
# EasyOCR ordered reading (with CRAFT thresholds)
# =========================================================
def _bbox_center(bbox) -> Tuple[float, float]:
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return (float(np.mean(xs)), float(np.mean(ys)))

def _bbox_minxy(bbox) -> Tuple[float, float]:
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return (min(xs), min(ys))

def _bbox_h(bbox) -> float:
    ys = [p[1] for p in bbox]
    return float(max(ys) - min(ys))

def easyocr_read_ordered(
    reader,
    pil_img: Image.Image,
    *,
    line_y_tol: float = 0.06,
    craft_text_threshold: float = 0.7,
    craft_link_threshold: float = 0.4,
    craft_low_text: float = 0.4,
    return_items: bool = False
):
    """
    return_items=True -> (ordered_text, items)
      items: List[(text, conf, h_px)]
    """
    np_img = np.array(pil_img)
    H, W = np_img.shape[:2]
    y_tol_px = max(8.0, H * line_y_tol)

    # ✅ Detection-stage thresholds (CRAFT)
    results = reader.readtext(
        np_img,
        detail=1,
        paragraph=False,
        text_threshold=float(craft_text_threshold),
        link_threshold=float(craft_link_threshold),
        low_text=float(craft_low_text),
    )

    # items: (x_min, y_center, x_center, text, conf, h_px)
    items = []
    for bbox, text, conf in results:
        if conf is None:
            continue
        text = clean_text_basic(text)
        if not text:
            continue
        x_center, y_center = _bbox_center(bbox)
        x_min, _ = _bbox_minxy(bbox)
        h_px = _bbox_h(bbox)
        items.append((x_min, y_center, x_center, text, float(conf), float(h_px)))

    if not items:
        return ("", []) if return_items else ""

    items.sort(key=lambda t: t[1])  # by y_center
    lines = []
    for it in items:
        placed = False
        for line in lines:
            y_mean = sum(x[1] for x in line) / len(line)
            if abs(it[1] - y_mean) <= y_tol_px:
                line.append(it)
                placed = True
                break
        if not placed:
            lines.append([it])

    for line in lines:
        line.sort(key=lambda t: t[2])  # by x_center

    lines.sort(key=lambda line: sum(x[1] for x in line) / len(line))

    ordered_texts = []
    ordered_items = []
    for line in lines:
        for (_, __, ___, txt, cf, hpx) in line:
            ordered_texts.append(txt)
            ordered_items.append((txt, cf, hpx))

    ordered_text = clean_text_basic(" ".join(ordered_texts))
    return (ordered_text, ordered_items) if return_items else ordered_text


# =========================================================
# TrOCR
# =========================================================
def trocr_read(processor, model, device, pil_img: Image.Image,
               max_new_tokens: int = 256, max_length: int = 0) -> str:
    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(device)
    gen_kwargs = {"max_new_tokens": int(max_new_tokens)}
    if int(max_length) > 0:
        gen_kwargs["max_length"] = int(max_length)
    with torch.inference_mode():
        outputs = model.generate(pixel_values, **gen_kwargs)
    text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return clean_text_basic(text)


def trocr_read_conf(processor, model, device, pil_img: Image.Image,
                    max_new_tokens: int = 64) -> Tuple[str, float]:
    """TrOCR read returning (text, confidence). conf = exp(mean token log-prob) in [0,1]."""
    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(device)
    if next(model.parameters()).dtype == torch.float16:
        pixel_values = pixel_values.half()
    with torch.inference_mode():
        out = model.generate(
            pixel_values,
            max_new_tokens=int(max_new_tokens),
            output_scores=True,
            return_dict_in_generate=True,
        )
    seq = out.sequences
    text = processor.batch_decode(seq, skip_special_tokens=True)[0]
    conf = 0.0
    try:
        ts = model.compute_transition_scores(seq, out.scores, normalize_logits=True)
        lp = ts[0]
        lp = lp[~torch.isinf(lp)]
        if lp.numel() > 0:
            conf = float(torch.exp(lp.mean()).clamp(0.0, 1.0))
    except Exception:
        conf = 0.0
    return clean_text_basic(text), conf


def easyocr_detect_lines(reader, pil_img: Image.Image, *,
                         line_y_tol=0.06, craft_text_threshold=0.7,
                         craft_link_threshold=0.4, craft_low_text=0.4):
    """Return reading-ordered lines of boxes:
       lines = [ [ {bbox, text, conf}, ... ] , ... ]  (top->bottom, left->right)."""
    np_img = np.array(pil_img)
    H, W = np_img.shape[:2]
    y_tol_px = max(8.0, H * line_y_tol)
    results = reader.readtext(
        np_img, detail=1, paragraph=False,
        text_threshold=float(craft_text_threshold),
        link_threshold=float(craft_link_threshold),
        low_text=float(craft_low_text),
    )
    items = []
    for bbox, text, conf in results:
        if conf is None:
            continue
        t = clean_text_basic(text)
        if not t:
            continue
        xc, yc = _bbox_center(bbox)
        xmin, _ = _bbox_minxy(bbox)
        items.append({"bbox": bbox, "text": t, "conf": float(conf),
                      "xc": xc, "yc": yc, "xmin": xmin})
    if not items:
        return []
    items.sort(key=lambda d: d["yc"])
    lines = []
    for it in items:
        placed = False
        for line in lines:
            ymean = sum(x["yc"] for x in line) / len(line)
            if abs(it["yc"] - ymean) <= y_tol_px:
                line.append(it); placed = True; break
        if not placed:
            lines.append([it])
    for line in lines:
        line.sort(key=lambda d: d["xc"])
    lines.sort(key=lambda line: sum(x["yc"] for x in line) / len(line))
    return lines


def crop_box(pil_img: Image.Image, bbox, pad: float = 0.12) -> Image.Image:
    xs = [p[0] for p in bbox]; ys = [p[1] for p in bbox]
    x0, x1 = min(xs), max(xs); y0, y1 = min(ys), max(ys)
    w, h = (x1 - x0), (y1 - y0)
    px, py = w * pad, h * pad
    X0 = max(0, int(x0 - px)); Y0 = max(0, int(y0 - py))
    X1 = min(pil_img.width, int(x1 + px)); Y1 = min(pil_img.height, int(y1 + py))
    if X1 <= X0 or Y1 <= Y0:
        return pil_img.crop((0, 0, pil_img.width, pil_img.height))
    return pil_img.crop((X0, Y0, X1, Y1))


def _is_numeric_noise(text: str, min_digits: int, ratio: float) -> bool:
    """Phone numbers / long codes: many digits or mostly-digit tokens."""
    digits = sum(c.isdigit() for c in text)
    if min_digits > 0 and digits >= min_digits:
        return True
    cleaned = re.sub(r"[^0-9A-Za-z가-힣]", "", text)
    if ratio > 0 and len(cleaned) >= 5 and digits / max(len(cleaned), 1) >= ratio:
        return True
    return False


def filter_boxes(lines, img_h: int, args):
    """Remove background/phone/low-confidence boxes BEFORE recognition.
    Returns (filtered_lines, n_dropped). Height filters are relative to the
    largest detected box (the main signboard text)."""
    if args.no_filter:
        return lines, 0
    all_h = [b["bbox"] for line in lines for b in line]
    def _h(bbox):
        ys = [p[1] for p in bbox]
        return max(ys) - min(ys)
    max_h = max((_h(b["bbox"]) for line in lines for b in line), default=0.0)

    dropped = 0
    out = []
    for line in lines:
        kept = []
        for b in line:
            t = b["text"]
            h = _h(b["bbox"])
            drop = False
            if args.filt_min_conf > 0 and b["conf"] < args.filt_min_conf:
                drop = True
            elif _is_numeric_noise(t, args.filt_numeric_digits, args.filt_numeric_ratio):
                drop = True
            elif args.filt_min_h_ratio > 0 and max_h > 0 and h < args.filt_min_h_ratio * max_h:
                drop = True
            elif args.filt_min_h_frac > 0 and img_h > 0 and h < args.filt_min_h_frac * img_h:
                drop = True
            if drop:
                dropped += 1
            else:
                kept.append(b)
        if kept:
            out.append(kept)

    # Safeguard: never let filtering empty a crop that originally had boxes.
    # Keep the single highest-confidence original box so it still yields a prediction.
    if not out and lines:
        best = max((b for line in lines for b in line), key=lambda b: b["conf"])
        out = [[best]]
        dropped = max(0, dropped - 1)
    return out, dropped


def select_box(etext, econf, ttext, tconf):
    """Deployable per-box selection: prefer non-empty; if both, higher confidence."""
    e, t = (etext or "").strip(), (ttext or "").strip()
    if not t:
        return e, "easy"
    if not e:
        return t, "trocr"
    return (t, "trocr") if tconf >= econf else (e, "easy")


def select_conf(cands):
    """N-way deployable selection: among (text, conf, name), pick highest-conf non-empty."""
    nonempty = [(txt.strip(), cf, nm) for (txt, cf, nm) in cands if txt and txt.strip()]
    if not nonempty:
        return "", "none"
    txt, cf, nm = max(nonempty, key=lambda x: x[1])
    return txt, nm


def save_csv(path: Path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "gt_text"], quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    ap.add_argument("--run", type=int, required=True)

    ap.add_argument("--line-y-tol", type=float, default=0.06)

    # ✅ CRAFT detector thresholds (Top-tier style)
    ap.add_argument("--craft-text-threshold", type=float, default=0.7,
                    help="EasyOCR(CRAFT) text_threshold. Typical: 0.6~0.8")
    ap.add_argument("--craft-link-threshold", type=float, default=0.4,
                    help="EasyOCR(CRAFT) link_threshold. Typical: 0.3~0.6")
    ap.add_argument("--craft-low-text", type=float, default=0.4,
                    help="EasyOCR(CRAFT) low_text. Typical: 0.3~0.5")

    # TrOCR decode
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-length", type=int, default=0, help="0 disables")

    # TrOCR model selection (default: fine-tuned signboard_full small, NOT base large)
    ap.add_argument("--trocr-model", type=str,
                    default=str(Path(__file__).resolve().parents[1]
                                / "artifacts" / "ocr_training" / "signboard_full" / "trocr_model_best"),
                    help="Path to fine-tuned TrOCR dir or HF id. "
                         "Default = signboard_full/trocr_model_best (in-domain val exact=84%).")
    ap.add_argument("--trocr-processor", type=str, default="microsoft/trocr-small-printed",
                    help="Fallback processor if the model dir has no processor files.")

    # Custom-trained EasyOCR recognizer (plugin via make_easyocr_plugin.py). None = default korean_g2.
    ap.add_argument("--easyocr-recog", type=str, default=None,
                    help="EasyOCR recog_network name (e.g. signboard_full_custom). Default uses pretrained.")
    ap.add_argument("--easyocr-langs", type=str, default=None,
                    help="Comma-separated EasyOCR languages (e.g. 'en' for English-only regions "
                         "like brooklyn). Default: ko,en (custom recog) / en,ko (pretrained).")

    # PaddleOCR as a 3rd engine. Box-level only. Runs in an ISOLATED subprocess
    # (paddle_rec_worker.py) because paddle and torch ship clashing cudnn DLLs.
    ap.add_argument("--paddle", action="store_true",
                    help="Add PaddleOCR as a 3rd per-box engine (isolated subprocess).")
    ap.add_argument("--paddle-model", type=str, default="korean_PP-OCRv5_mobile_rec",
                    help="PaddleOCR TextRecognition model_name (architecture).")
    ap.add_argument("--paddle-model-dir", type=str, default=None,
                    help="Exported inference dir of a FINE-TUNED PaddleOCR model "
                         "(e.g. output/paddle_signboard_rec_v2/inference). "
                         "Omit for zero-shot pretrained. Implies --paddle.")
    ap.add_argument("--paddle-device", type=str, default="gpu",
                    help="Device for the PaddleOCR worker subprocess (gpu/cpu).")

    # Box-level pipeline: EasyOCR(CRAFT) detects word boxes -> each box -> fine-tuned TrOCR
    # -> per-box selection ensemble (deployable, confidence-based). Fixes granularity mismatch.
    ap.add_argument("--box", action="store_true",
                    help="Box-level recognition: feed each EasyOCR-detected box to TrOCR.")
    ap.add_argument("--box-pad", type=float, default=0.12, help="Padding ratio around each box crop.")
    ap.add_argument("--box-trocr-tokens", type=int, default=64, help="max_new_tokens per box for TrOCR.")

    # Box filtering (remove background / phone-number / low-confidence boxes before recognition)
    ap.add_argument("--filt-min-conf", type=float, default=0.10,
                    help="Drop EasyOCR boxes with detection confidence below this. 0 disables.")
    ap.add_argument("--filt-numeric-digits", type=int, default=7,
                    help="Drop boxes whose text has >= this many digits (phone numbers/codes). 0 disables.")
    ap.add_argument("--filt-numeric-ratio", type=float, default=0.6,
                    help="Drop boxes len>=5 with digit-ratio >= this (numeric noise). 0 disables.")
    ap.add_argument("--filt-min-h-ratio", type=float, default=0.0,
                    help="Drop boxes shorter than ratio * max-box-height in the image. 0 disables (off).")
    ap.add_argument("--filt-min-h-frac", type=float, default=0.0,
                    help="Drop boxes shorter than frac * image-height (tiny distant text). 0 disables.")
    ap.add_argument("--no-filter", action="store_true", help="Disable all box filtering.")

    # normalization / postprocess (optional)
    ap.add_argument("--normalize", choices=["none", "strict"], default="none",
                    help="strict keeps only [0-9A-Za-z가-힣 ]")
    ap.add_argument("--postprocess", choices=["none", "top_tier"], default="none")
    ap.add_argument("--force-case", choices=["none", "upper", "lower"], default="none")
    ap.add_argument("--use-lexicon", action="store_true")
    ap.add_argument("--lexicon-path", type=str, default="")
    ap.add_argument("--lexicon-max-dist", type=int, default=2)
    ap.add_argument("--drop-single-char", action="store_true")

    args = ap.parse_args()

    crop_region_dir = CROP_DIR / args.region
    if not crop_region_dir.exists():
        print(f"Crop directory not found: {crop_region_dir}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = (device == "cuda")

    user_lex = load_lexicon_txt(Path(args.lexicon_path)) if args.lexicon_path else set()
    merged_lex = set(DEFAULT_EN_LEXICON)
    merged_lex.update(user_lex)

    print("Loading models...")

    # Load fine-tuned TrOCR (signboard_full). The dir carries its own processor files;
    # fall back to the small-printed processor if missing.
    print(f"  TrOCR model: {args.trocr_model}")
    try:
        processor = TrOCRProcessor.from_pretrained(args.trocr_model)
    except Exception as e:
        print(f"  [warn] processor from model dir failed ({e}); using {args.trocr_processor}")
        processor = TrOCRProcessor.from_pretrained(args.trocr_processor)
    trocr_model = VisionEncoderDecoderModel.from_pretrained(args.trocr_model).to(device)
    trocr_model.eval()
    if use_fp16:
        trocr_model.half()

    user_langs = ([l.strip() for l in args.easyocr_langs.split(",") if l.strip()]
                  if args.easyocr_langs else None)
    if args.easyocr_recog:
        print(f"  EasyOCR recog_network: {args.easyocr_recog} (custom-trained)")
        easy_reader = easyocr.Reader(user_langs or ["ko", "en"], gpu=(device == "cuda"),
                                     recog_network=args.easyocr_recog)
    else:
        langs = user_langs or ["en", "ko"]
        print(f"  EasyOCR langs: {langs} (pretrained)")
        easy_reader = easyocr.Reader(langs, gpu=(device == "cuda"))

    image_files = sorted(crop_region_dir.glob("*.jpg"), key=lambda p: natural_key(p.stem))
    print(f"Found {len(image_files)} images in {crop_region_dir}")

    # Optional 3rd engine: PaddleOCR, box-level only. Cannot share this process
    # with torch (cudnn DLL clash), so it runs in paddle_rec_worker.py. Here we
    # only decide whether it is enabled; recognition happens after all crops are
    # written to a temp dir (two-phase, see the BOX-LEVEL block below).
    use_paddle = args.box and (args.paddle or bool(args.paddle_model_dir))

    trocr_rows = []
    easy_rows = []
    ens_rows = []
    paddle_rows = []

    # =====================================================================
    # BOX-LEVEL pipeline: EasyOCR(CRAFT) boxes -> per-box TrOCR (+ PaddleOCR)
    #   -> per-box selection ensemble. Lines joined by '\n' for line_match eval.
    # =====================================================================
    if args.box:
        import json, subprocess, sys, tempfile, shutil
        print("[mode] BOX-LEVEL (EasyOCR detect -> per-box TrOCR"
              + (" + PaddleOCR" if use_paddle else "") + " -> selection ensemble)")
        if not args.no_filter:
            print(f"[filter] min_conf={args.filt_min_conf} numeric_digits={args.filt_numeric_digits} "
                  f"numeric_ratio={args.filt_numeric_ratio} h_ratio={args.filt_min_h_ratio} "
                  f"h_frac={args.filt_min_h_frac}")

        # -------- Phase A: detect, filter, crop. Cache crops in memory (for TrOCR)
        # and, when PaddleOCR is enabled, also to a temp dir (for the worker). --------
        paddle_tmp = Path(tempfile.mkdtemp(prefix="paddle_box_")) if use_paddle else None
        manifest = []  # [{"key","path"}] for the paddle worker
        records = []   # per-image: {"image_name", "lines": [[box,...],...]}
        total_dropped = 0
        for img_path in image_files:
            image_name = img_path.stem
            img = Image.open(img_path).convert("RGB")
            lines = easyocr_detect_lines(
                easy_reader, img,
                line_y_tol=args.line_y_tol,
                craft_text_threshold=args.craft_text_threshold,
                craft_link_threshold=args.craft_link_threshold,
                craft_low_text=args.craft_low_text,
            )
            lines, ndrop = filter_boxes(lines, img.height, args)
            total_dropped += ndrop

            rec_lines = []
            for li, line in enumerate(lines):
                box_recs = []
                for bi, box in enumerate(line):
                    patch = crop_box(img, box["bbox"], pad=args.box_pad)
                    key = f"{image_name}__{li}__{bi}"
                    if paddle_tmp is not None:
                        crop_path = paddle_tmp / f"{key}.png"
                        patch.save(crop_path)
                        manifest.append({"key": key, "path": str(crop_path)})
                    box_recs.append({"e_text": box["text"], "e_conf": box["conf"],
                                     "patch": patch, "key": key})
                rec_lines.append(box_recs)
            records.append({"image_name": image_name, "lines": rec_lines})

        # -------- Phase B: PaddleOCR recognition in an isolated subprocess --------
        paddle_map = {}  # key -> (text, score)
        if use_paddle:
            man_path = paddle_tmp / "manifest.jsonl"
            out_path = paddle_tmp / "results.jsonl"
            with open(man_path, "w", encoding="utf-8") as f:
                for m in manifest:
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")
            worker = Path(__file__).resolve().parents[1] / "str_baselines/paddle_rec_worker.py"
            cmd = [sys.executable, str(worker),
                   "--manifest", str(man_path), "--out", str(out_path),
                   "--model-name", args.paddle_model, "--device", args.paddle_device]
            if args.paddle_model_dir:
                cmd += ["--model-dir", args.paddle_model_dir]
                print(f"  PaddleOCR recog: FINE-TUNED {args.paddle_model_dir} (isolated, {args.paddle_device})")
            else:
                print(f"  PaddleOCR recog: {args.paddle_model} zero-shot (isolated, {args.paddle_device})")
            print(f"  [paddle] running worker over {len(manifest)} crops ...")
            subprocess.run(cmd, check=True)
            with open(out_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    paddle_map[d["key"]] = (clean_text_basic(d.get("text", "")),
                                            float(d.get("score", 0.0) or 0.0))

        # -------- Phase C: per-box TrOCR (in-process torch) + selection ensemble --------
        def _finish(line_list):
            txt = "\n".join(line_list)
            if args.normalize == "strict":
                txt = "\n".join(normalize_text_strict(l) for l in line_list)
            txt = remove_year_google_citations(txt)
            return "" if is_junk_only(txt) else txt

        for rec in records:
            image_name = rec["image_name"]
            tro_lines, easy_lines, paddle_lines, ens_lines = [], [], [], []
            for line in rec["lines"]:
                tro_toks, easy_toks, pad_toks, ens_toks = [], [], [], []
                for box in line:
                    e_text, e_conf = box["e_text"], box["e_conf"]
                    try:
                        t_text, t_conf = trocr_read_conf(
                            processor, trocr_model, device, box["patch"],
                            max_new_tokens=args.box_trocr_tokens)
                    except Exception:
                        t_text, t_conf = "", 0.0
                    p_text, p_conf = paddle_map.get(box["key"], ("", 0.0))

                    cands = [(e_text, e_conf, "easy"), (t_text, t_conf, "trocr")]
                    if use_paddle:
                        cands.append((p_text, p_conf, "paddle"))
                    sel, _src = select_conf(cands)

                    if e_text.strip():
                        easy_toks.append(e_text.strip())
                    if t_text.strip():
                        tro_toks.append(t_text.strip())
                    if p_text.strip():
                        pad_toks.append(p_text.strip())
                    if sel.strip():
                        ens_toks.append(sel.strip())
                if easy_toks:
                    easy_lines.append(" ".join(easy_toks))
                if tro_toks:
                    tro_lines.append(" ".join(tro_toks))
                if pad_toks:
                    paddle_lines.append(" ".join(pad_toks))
                if ens_toks:
                    ens_lines.append(" ".join(ens_toks))

            trocr_rows.append({"image_name": image_name, "gt_text": _finish(tro_lines)})
            easy_rows.append({"image_name": image_name, "gt_text": _finish(easy_lines)})
            ens_rows.append({"image_name": image_name, "gt_text": _finish(ens_lines)})
            if use_paddle:
                paddle_rows.append({"image_name": image_name, "gt_text": _finish(paddle_lines)})

        if paddle_tmp is not None:
            shutil.rmtree(paddle_tmp, ignore_errors=True)

        trocr_csv = OUT_DIR / f"ocr_{args.region}_{args.run}_trocr.csv"
        easy_csv = OUT_DIR / f"ocr_{args.region}_{args.run}_easyocr.csv"
        ens_csv = OUT_DIR / f"ocr_{args.region}_{args.run}_ensemble.csv"
        print(f"[filter] dropped {total_dropped} background/noise boxes")
        print("DONE (box-level)")
        print(save_csv(trocr_csv, trocr_rows))
        print(save_csv(easy_csv, easy_rows))
        print(save_csv(ens_csv, ens_rows))
        if use_paddle:
            paddle_csv = OUT_DIR / f"ocr_{args.region}_{args.run}_paddle.csv"
            print(save_csv(paddle_csv, paddle_rows))
        return

    for img_path in image_files:
        image_name = img_path.stem
        img = Image.open(img_path).convert("RGB")

        # --- EasyOCR ordered read + items (detector thresholds applied inside) ---
        try:
            easy_text_raw, items = easyocr_read_ordered(
                easy_reader,
                img,
                line_y_tol=args.line_y_tol,
                craft_text_threshold=args.craft_text_threshold,
                craft_link_threshold=args.craft_link_threshold,
                craft_low_text=args.craft_low_text,
                return_items=True
            )
        except Exception:
            items = []

        # ✅ Detection returns nothing => do not OCR (both empty)
        if not items:
            trocr_text = ""
            easy_text = ""
        else:
            # keep all detected boxes (NO conf/area post-filtering)
            easy_text = clean_text_basic(" ".join([t for (t, _, __) in [(x[0], x[1], x[2]) for x in []]]))  # placeholder

            # rebuild easy_text from items
            easy_text = clean_text_basic(" ".join([t for (t, _, __) in [(txt, cf, hpx) for (txt, cf, hpx) in items]]))

            # TrOCR
            try:
                trocr_text = trocr_read(
                    processor, trocr_model, device, img,
                    max_new_tokens=args.max_new_tokens,
                    max_length=args.max_length
                )
            except Exception:
                trocr_text = ""

            # normalize (optional)
            if args.normalize == "strict":
                easy_text = normalize_text_strict(easy_text)
                trocr_text = normalize_text_strict(trocr_text)

            # postprocess (optional)
            if args.postprocess == "top_tier":
                easy_text = postprocess_text_top_tier(
                    easy_text,
                    use_lexicon=args.use_lexicon,
                    lexicon=merged_lex,
                    max_edit_dist=args.lexicon_max_dist,
                    force_case=args.force_case,
                    drop_single_char=args.drop_single_char,
                )
                trocr_text = postprocess_text_top_tier(
                    trocr_text,
                    use_lexicon=args.use_lexicon,
                    lexicon=merged_lex,
                    max_edit_dist=args.lexicon_max_dist,
                    force_case=args.force_case,
                    drop_single_char=args.drop_single_char,
                )

            # ✅ remove "2022 Google" style noise
            easy_text = remove_year_google_citations(easy_text)
            trocr_text = remove_year_google_citations(trocr_text)

            if is_junk_only(easy_text):
                easy_text = ""
            if is_junk_only(trocr_text):
                trocr_text = ""

        trocr_rows.append({"image_name": image_name, "gt_text": trocr_text})
        easy_rows.append({"image_name": image_name, "gt_text": easy_text})

    trocr_csv = OUT_DIR / f"ocr_{args.region}_{args.run}_trocr.csv"
    easy_csv = OUT_DIR / f"ocr_{args.region}_{args.run}_easyocr.csv"

    print("DONE")
    print(save_csv(trocr_csv, trocr_rows))
    print(save_csv(easy_csv, easy_rows))


if __name__ == "__main__":
    main()