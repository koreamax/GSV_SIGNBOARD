#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_crops_from_gt_polygon.py (MASK(B) -> RECTIFY(A) + TrOCR-STRONG v2 FAST)

추가 반영:
- (B) 폴리곤 마스크로 배경 제거 후
- (A) perspective warp로 정면 보정
- GT 좌표 해상도(2197x1295 or 8192x4828) 스케일 이슈 대응:
  --gt-size auto|small|large
  auto: 폴리곤 최대 좌표 보고 자동 추정

파이프라인(고정):
0) GT->원본 스케일 보정(필요 시)
1) polygon mask (B)
2) rectification (A) with mask warp
3) tight crop by warped mask (배경 여백 제거)
4) optional padding (pixel border replicate)
5) upscale + max_side 제한
6) to GRAY
7) CLAHE on GRAY
8) denoise
9) unsharp
"""

import argparse
import csv
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np


# ---------------------------
# Args
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--debug", action="store_true",
               help="save debug overlay images with polygons")
    p.add_argument("--region", required=True, choices=["brooklyn", "gangnam", "suwon"])
    # 연쇄 평가(e2e)용: 탐지 박스 CSV로도 같은 전처리를 돌리기 위한 경로 주입.
    # 전처리 경로를 하나로 유지해야 GT 크롭 결과와 비교가 성립합니다.
    p.add_argument("--gt-csv", default=None,
                   help="기본 artifacts/gt/gt_{region}.csv 대신 쓸 박스 CSV")
    p.add_argument("--out-subdir", default="crop",
                   help="artifacts/gt/<이것>/{region} 에 저장 (기본 crop)")
    p.add_argument("--overwrite", action="store_true", help="overwrite existing crops")

    # padding ratio (quad scale padding, default OFF)
    p.add_argument("--pad", type=float, default=0.0,
                   help="quad padding ratio (e.g., 0.04 = 4%%). default=0 (OFF)")

    # after warp: optional border padding in pixels (helps TrOCR sometimes)
    p.add_argument("--border-pad", type=int, default=0,
                   help="pixel border padding after tight-crop (default=0)")

    # GT coord size handling
    p.add_argument("--gt-size", choices=["auto", "small", "large"], default="auto",
                   help="GT coordinate base size. small=2197x1295, large=8192x4828, auto=guess by coords")

    # upscale
    p.add_argument("--upscale", type=float, default=2.0,
                   help="upscale factor AFTER rectification (default=2.0). 1.0 disables")
    p.add_argument("--max_side", type=int, default=1600,
                   help="limit max side after upscale to avoid OOM (default=1600)")

    # CLAHE on GRAY
    p.add_argument("--clahe-clip", type=float, default=2.5,
                   help="CLAHE clipLimit on GRAY (default=2.5, suggested 2.0~3.5)")
    p.add_argument("--clahe-grid", type=int, default=8,
                   help="CLAHE tileGridSize (default=8, suggested 8~16)")

    # Denoise (GRAY)
    p.add_argument("--denoise", choices=["none", "median", "nlm"], default="median",
                   help="denoise method on GRAY (default=median). nlm is slower.")
    p.add_argument("--median-ksize", type=int, default=3,
                   help="medianBlur ksize (odd, default=3). 3 recommended.")
    p.add_argument("--nlm-h", type=int, default=10, help="fastNlMeansDenoising h (default=10)")
    p.add_argument("--nlm-template", type=int, default=7, help="templateWindowSize (default=7)")
    p.add_argument("--nlm-search", type=int, default=21, help="searchWindowSize (default=21)")

    # Unsharp mask (GRAY)
    p.add_argument("--unsharp-sigma", type=float, default=1.2,
                   help="unsharp sigma (default=1.2, suggested 1.0~1.4)")
    p.add_argument("--unsharp-amount", type=float, default=0.55,
                   help="unsharp amount (default=0.55, suggested 0.35~0.65)")
    p.add_argument("--unsharp-threshold", type=int, default=4,
                   help="unsharp threshold (default=4). higher -> less noise sharpened")

    return p.parse_args()


# ---------------------------
# Geometry helpers
# ---------------------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    pts: (4,2) float32
    returns ordered as [tl, tr, br, bl]
    """
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def bbox_to_quad(x1, y1, x2, y2):
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def pad_quad(quad: np.ndarray, pad_ratio: float, img_w: int, img_h: int) -> np.ndarray:
    """
    Quad를 중심 기준으로 pad_ratio만큼 확장.
    """
    if pad_ratio <= 0:
        return quad

    quad = quad.astype(np.float32)
    cx = float(np.mean(quad[:, 0]))
    cy = float(np.mean(quad[:, 1]))

    scale = 1.0 + float(pad_ratio)
    padded = np.zeros_like(quad, dtype=np.float32)
    for i in range(4):
        dx = quad[i, 0] - cx
        dy = quad[i, 1] - cy
        px = cx + dx * scale
        py = cy + dy * scale
        padded[i, 0] = clamp(px, 0, img_w - 1)
        padded[i, 1] = clamp(py, 0, img_h - 1)
    return padded


# ---------------------------
# GT scale handling (2197x1295 or 8192x4828)
# ---------------------------
SMALL_W, SMALL_H = 2197, 1295
LARGE_W, LARGE_H = 8192, 4828

def guess_gt_base(quad: np.ndarray) -> tuple[int, int]:
    """
    폴리곤 좌표 최대값 기준으로 small/large 자동 추정.
    (데이터가 더 커질 수도 있으니 여유 마진 포함)
    """
    mx = float(np.max(quad[:, 0]))
    my = float(np.max(quad[:, 1]))

    # small 범위 안이면 small로
    if mx <= SMALL_W * 1.10 and my <= SMALL_H * 1.10:
        return SMALL_W, SMALL_H
    return LARGE_W, LARGE_H


def scale_quad_to_image(quad: np.ndarray, img_w: int, img_h: int, gt_size_mode: str) -> np.ndarray:
    """
    GT 좌표계를 원본 이미지 좌표계로 스케일 변환.
    - gt_size_mode: auto|small|large
    """
    if gt_size_mode == "small":
        gw, gh = SMALL_W, SMALL_H
    elif gt_size_mode == "large":
        gw, gh = LARGE_W, LARGE_H
    else:
        gw, gh = guess_gt_base(quad)

    # 이미 같은 해상도 기준이면 scale ~1
    sx = img_w / float(gw)
    sy = img_h / float(gh)

    out = quad.astype(np.float32).copy()
    out[:, 0] *= sx
    out[:, 1] *= sy

    # clamp
    out[:, 0] = np.clip(out[:, 0], 0, img_w - 1)
    out[:, 1] = np.clip(out[:, 1], 0, img_h - 1)
    return out


# ---------------------------
# MASK(B) -> WARP(A)
# ---------------------------
def build_polygon_mask(img_h: int, img_w: int, quad_xy: np.ndarray) -> np.ndarray:
    """
    quad_xy: (4,2) in image coords
    returns mask uint8 (H,W) 0/255
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    pts = quad_xy.astype(np.int32).reshape(1, 4, 2)
    cv2.fillPoly(mask, pts, 255)
    return mask


def warp_with_mask_then_tight_crop(img_bgr: np.ndarray, quad_pts: np.ndarray):
    """
    (B) 원본에서 폴리곤 영역만 mask로 남기고
    (A) 같은 M으로 이미지+mask warp
    warp 이후 mask 기반으로 tight crop 해서 배경(검은 여백) 최소화

    returns: rectified_bgr (crop), rectified_mask (crop)
    """
    quad = order_points(quad_pts.astype(np.float32))

    wA = np.linalg.norm(quad[2] - quad[3])
    wB = np.linalg.norm(quad[1] - quad[0])
    hA = np.linalg.norm(quad[1] - quad[2])
    hB = np.linalg.norm(quad[0] - quad[3])

    out_w = int(max(wA, wB))
    out_h = int(max(hA, hB))

    out_w = max(out_w, 2)
    out_h = max(out_h, 2)

    dst = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(quad, dst)

    H, W = img_bgr.shape[:2]
    orig_mask = build_polygon_mask(H, W, quad)

    # 원본에서 먼저 배경 제거 (B)
    masked_src = cv2.bitwise_and(img_bgr, img_bgr, mask=orig_mask)

    # 이미지+마스크를 동일 transform으로 warp (A)
    warped = cv2.warpPerspective(
        masked_src, M, (out_w, out_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    warped_mask = cv2.warpPerspective(
        orig_mask, M, (out_w, out_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # tight crop by mask (검은 여백 제거)
    ys, xs = np.where(warped_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        # mask가 날아간 경우(이상 좌표 등) fallback: 그대로 반환
        return warped, warped_mask

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    # 안전 마진 1px
    x1 = max(0, x1 - 1); y1 = max(0, y1 - 1)
    x2 = min(out_w - 1, x2 + 1); y2 = min(out_h - 1, y2 + 1)

    warped = warped[y1:y2+1, x1:x2+1]
    warped_mask = warped_mask[y1:y2+1, x1:x2+1]

    return warped, warped_mask


# ---------------------------
# Preprocess (TrOCR-strong v2)
# ---------------------------
def limit_size_keep_ratio(img, max_side: int):
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    nw = int(round(w * scale))
    nh = int(round(h * scale))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def upscale(img: np.ndarray, factor: float, max_side: int):
    if factor is None or factor <= 1.0:
        return limit_size_keep_ratio(img, max_side)
    out = cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
    return limit_size_keep_ratio(out, max_side)


def clahe_gray(gray: np.ndarray, clip: float, grid: int) -> np.ndarray:
    grid = max(2, int(grid))
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(grid, grid))
    return clahe.apply(gray)


def denoise_gray(gray: np.ndarray, method: str, args) -> np.ndarray:
    if method == "none":
        return gray
    if method == "median":
        k = int(args.median_ksize)
        if k % 2 == 0:
            k += 1
        k = max(3, k)
        return cv2.medianBlur(gray, k)
    if method == "nlm":
        return cv2.fastNlMeansDenoising(
            gray, None,
            h=int(args.nlm_h),
            templateWindowSize=int(args.nlm_template),
            searchWindowSize=int(args.nlm_search)
        )
    return gray


def unsharp_gray(gray: np.ndarray, sigma: float, amount: float, threshold: int) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (0, 0), float(sigma))
    sharp = cv2.addWeighted(gray, 1.0 + float(amount), blur, -float(amount), 0)

    if threshold and threshold > 0:
        low_contrast_mask = np.absolute(gray.astype(np.int16) - blur.astype(np.int16)) < int(threshold)
        sharp[low_contrast_mask] = gray[low_contrast_mask]

    return np.clip(sharp, 0, 255).astype(np.uint8)


def preprocess_trocr_strong_v2(rectified_bgr: np.ndarray, args) -> np.ndarray:
    # 1) upscale on BGR first
    bgr = upscale(rectified_bgr, args.upscale, args.max_side)

    # 2) to gray
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # 3) CLAHE
    gray = clahe_gray(gray, clip=args.clahe_clip, grid=args.clahe_grid)

    # 4) denoise
    gray = denoise_gray(gray, args.denoise, args)

    # 5) unsharp
    gray = unsharp_gray(gray, sigma=args.unsharp_sigma, amount=args.unsharp_amount, threshold=args.unsharp_threshold)

    # 6) normalize
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return gray


# ---------------------------
# CSV parsing
# ---------------------------
def parse_row_to_polygon(row: dict):
    keys = set((k or "").strip() for k in row.keys())
    poly_need = {"x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"}
    bbox_need = {"x1", "y1", "x2", "y2"}

    if poly_need.issubset(keys):
        pts = []
        for i in range(1, 5):
            x = float(row[f"x{i}"])
            y = float(row[f"y{i}"])
            pts.append([x, y])
        return "poly", np.array(pts, dtype=np.float32)

    if bbox_need.issubset(keys):
        x1 = float(row["x1"]); y1 = float(row["y1"])
        x2 = float(row["x2"]); y2 = float(row["y2"])
        return "bbox", (x1, y1, x2, y2)

    return None, None


# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()

    BASE_DIR = Path(__file__).resolve().parent / "artifacts"
    GT_CSV = (Path(args.gt_csv) if args.gt_csv
              else BASE_DIR / "gt" / f"gt_{args.region}.csv")
    SRC_DIR = BASE_DIR / "gsv_photo" / args.region
    OUT_DIR = BASE_DIR / "gt" / args.out_subdir / args.region
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR = BASE_DIR / "gt" / "debug_overlay" / args.region
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    if not GT_CSV.exists():
        raise FileNotFoundError(f"GT CSV not found: {GT_CSV}")
    if not SRC_DIR.exists():
        raise FileNotFoundError(f"Source image dir not found: {SRC_DIR}")

    print(f"[INFO] region={args.region}")
    print(f"[INFO] GT_CSV={GT_CSV}")
    print(f"[INFO] SRC_DIR={SRC_DIR}")
    print(f"[INFO] OUT_DIR={OUT_DIR}")
    print(f"[INFO] gt_size={args.gt_size} (small={SMALL_W}x{SMALL_H}, large={LARGE_W}x{LARGE_H})")
    print(f"[INFO] pad(quad ratio)={args.pad:.4f} border_pad(px)={args.border_pad}")
    print(f"[INFO] upscale={args.upscale} max_side={args.max_side}")
    print(f"[INFO] CLAHE GRAY: clip={args.clahe_clip} grid={args.clahe_grid}")
    print(f"[INFO] Denoise GRAY: {args.denoise} (median k={args.median_ksize} / nlm h={args.nlm_h})")
    print(f"[INFO] Unsharp GRAY: sigma={args.unsharp_sigma} amount={args.unsharp_amount} thr={args.unsharp_threshold}")
    print(f"[INFO] Output: GRAY (TrOCR-strong v2) | MASK(B)->RECTIFY(A)->TIGHTCROP")

    rows_by_filename = defaultdict(list)
    with open(GT_CSV, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        if not rdr.fieldnames or "filename" not in rdr.fieldnames:
            raise ValueError(f"[ERROR] {GT_CSV.name} must include 'filename' column.")

        for row in rdr:
            fn = (row.get("filename") or "").strip()
            if not fn:
                continue
            mode, data = parse_row_to_polygon(row)
            if mode is None:
                continue
            rows_by_filename[fn].append((mode, data))

    total = 0
    skipped = 0

    for filename, shapes in rows_by_filename.items():
        stem = Path(filename).stem
        photo_id = stem.split("__")[-1] if "__" in stem else stem

        p1 = SRC_DIR / f"{photo_id}.jpg"
        p2 = SRC_DIR / filename
        img_path = p1 if p1.exists() else p2

        if not img_path.exists():
            print(f"[WARN] original not found for {filename} -> tried {p1.name} / {p2.name}")
            skipped += len(shapes)
            continue

        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"[WARN] failed to read image: {img_path}")
            skipped += len(shapes)
            continue

        img_h, img_w = img_bgr.shape[:2]

        for i, (mode, data) in enumerate(shapes, start=1):
            out_name = f"{args.region}__{photo_id}__crop_{i:03d}.jpg"
            out_path = OUT_DIR / out_name

            if out_path.exists() and not args.overwrite:
                continue

            # 1) quad 만들기 (GT 좌표계)
            if mode == "poly":
                quad_gt = data
            else:
                x1, y1, x2, y2 = data
                if x2 < x1: x1, x2 = x2, x1
                if y2 < y1: y1, y2 = y2, y1
                quad_gt = bbox_to_quad(x1, y1, x2, y2)

            # 2) GT->원본 스케일 보정
            quad = scale_quad_to_image(quad_gt, img_w, img_h, args.gt_size)

            if args.debug:
                dbg = img_bgr.copy()
                q = quad.astype(np.int32)
                cv2.polylines(dbg, [q.reshape(-1,1,2)], isClosed=True, color=(0,255,0), thickness=3)

                # corner points 표시
                for idx, (x, y) in enumerate(q):
                    cv2.circle(dbg, (int(x), int(y)), 6, (0,0,255), -1)
                    cv2.putText(dbg, str(idx+1), (int(x)+5, int(y)-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

                dbg_name = f"{args.region}__{photo_id}__dbg_{i:03d}.jpg"
                cv2.imwrite(str(DEBUG_DIR / dbg_name), dbg)

            # 3) quad padding ratio (optional)
            quad = pad_quad(quad, args.pad, img_w, img_h)

            # 4) (B)->(A): mask then warp then tight crop
            rectified_bgr, rectified_mask = warp_with_mask_then_tight_crop(img_bgr, quad)

            # 5) border padding (optional)
            if args.border_pad and args.border_pad > 0:
                bp = int(args.border_pad)
                rectified_bgr = cv2.copyMakeBorder(rectified_bgr, bp, bp, bp, bp, borderType=cv2.BORDER_REPLICATE)

            # 6) TrOCR preprocess -> GRAY
            out_gray = preprocess_trocr_strong_v2(rectified_bgr, args)

            cv2.imwrite(str(out_path), out_gray)
            total += 1

    print(f"[DONE] Saved crops: {total}")
    print(f"[DONE] Skipped boxes (missing originals / read fail): {skipped}")
    print(f"[DONE] Output dir: {OUT_DIR}")


if __name__ == "__main__":
    main()