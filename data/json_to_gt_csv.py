#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
labelme_polygon_json_to_gt_csv.py (POLYGON -> QUAD(4pt) GT)

네 상황(Polygon으로 저장됨) 기준 "완성본".

핵심:
- Labelme JSON의 signboard 폴리곤(points: N점)을 그대로 bbox(min/max)로 바꾸지 않음 ❌
- 대신 N점 폴리곤 -> cv2.minAreaRect -> 4점(quadrilateral)으로 변환 ✅
- 4점도 항상 tl,tr,br,bl로 정렬(order_points) ✅
- imageWidth/imageHeight는 JSON에 있으면 그걸 사용, 없으면 jpg 열어서 획득 ✅
- subdir prefix 규칙 유지: filename = "{subdir}__{raw_filename}" ✅
- 출력:
  - artifacts/gt/gt_{subdir}.csv  (x1..y4 포함)
  - artifacts/gt/total_gt.csv     (gt_*.csv 합침)

실행 예:
  python labelme_polygon_json_to_gt_csv.py             # 모든 subdir 처리
  python labelme_polygon_json_to_gt_csv.py --subdir brooklyn
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parents[1] / "artifacts"
GT_DIR = BASE_DIR / "gt"
GT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Utils
# ----------------------------
def order_points(pts: np.ndarray) -> np.ndarray:
    """
    pts: (4,2) float32
    return: [tl, tr, br, bl]
    """
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def dedup_points(points, eps=1e-3):
    """
    labelme polygon에 중복/거의중복 점이 섞여 있는 경우가 있어 안정성 위해 제거.
    - 연속 중복 제거
    - 마지막 점이 첫 점과 같은 경우 제거
    """
    out = []
    for x, y in points:
        if not out:
            out.append([float(x), float(y)])
            continue
        px, py = out[-1]
        if abs(float(x) - px) > eps or abs(float(y) - py) > eps:
            out.append([float(x), float(y)])

    if len(out) >= 2:
        if abs(out[0][0] - out[-1][0]) < eps and abs(out[0][1] - out[-1][1]) < eps:
            out.pop()
    return out


def polygon_to_quad(points):
    """
    points: list of [x,y], polygon (N>=3), rectangle tool (2), quad (4) 모두 처리
    return: quad (4,2) float32 ordered tl,tr,br,bl or None
    """
    if not points or len(points) < 2:
        return None

    points = dedup_points(points)
    pts = np.array(points, dtype=np.float32)

    # rectangle tool: 2 corners
    if pts.shape[0] == 2:
        (x1, y1), (x2, y2) = pts
        x1, x2 = float(min(x1, x2)), float(max(x1, x2))
        y1, y2 = float(min(y1, y2)), float(max(y1, y2))
        quad = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        return order_points(quad)

    # already 4 points
    if pts.shape[0] == 4:
        return order_points(pts)

    # polygon with N>4 (or N==3): minAreaRect -> 4 points
    if pts.shape[0] >= 3:
        rect = cv2.minAreaRect(pts)       # ((cx,cy),(w,h),angle)
        box = cv2.boxPoints(rect)         # (4,2)
        return order_points(box.astype(np.float32))

    return None


def get_image_size_from_json_or_file(data: dict, img_path: Path):
    w = data.get("imageWidth", None)
    h = data.get("imageHeight", None)
    if w is not None and h is not None:
        return int(w), int(h)

    if img_path.exists():
        try:
            ww, hh = Image.open(img_path).size
            return int(ww), int(hh)
        except Exception:
            pass

    return "", ""


# ----------------------------
# Parse one labelme json
# ----------------------------
def parse_labelme_json(json_path: Path, img_dir: Path, subdir_name: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_filename = Path(data.get("imagePath", json_path.with_suffix(".jpg").name)).name
    filename = f"{subdir_name}__{raw_filename}"

    img_path = img_dir / raw_filename
    w, h = get_image_size_from_json_or_file(data, img_path)

    rows = []
    for shape in data.get("shapes", []):
        if (shape.get("label", "") or "").strip() != "signboard":
            continue

        pts = shape.get("points", [])
        quad = polygon_to_quad(pts)
        if quad is None:
            continue

        # 너무 작은 거 제거(안전)
        wq = float(np.linalg.norm(quad[1] - quad[0]))
        hq = float(np.linalg.norm(quad[3] - quad[0]))
        if wq < 2 or hq < 2:
            continue

        rows.append({
            "filename": filename,
            "w": w,
            "h": h,
            "x1": float(quad[0, 0]), "y1": float(quad[0, 1]),
            "x2": float(quad[1, 0]), "y2": float(quad[1, 1]),
            "x3": float(quad[2, 0]), "y3": float(quad[2, 1]),
            "x4": float(quad[3, 0]), "y4": float(quad[3, 1]),
            "class": "signboard",
        })

    return rows


# ----------------------------
# Process subdir
# ----------------------------
def process_subdir(subdir_name: str):
    img_dir = BASE_DIR / "gsv_photo" / subdir_name
    out_csv = GT_DIR / f"gt_{subdir_name}.csv"

    if not img_dir.exists():
        print(f"[WARN] 이미지 폴더 없음: {img_dir}")
        return

    rows = []
    for json_path in sorted(img_dir.glob("*.json")):
        rows.extend(parse_labelme_json(json_path, img_dir, subdir_name))

    if not rows:
        print(f"[WARN] {subdir_name}: JSON에서 signboard 없음")
        return

    header = [
        "filename", "w", "h",
        "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4",
        "class"
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        wri = csv.DictWriter(f, fieldnames=header)
        wri.writeheader()
        wri.writerows(rows)

    print(f"[GT] {subdir_name} 저장 완료: {out_csv} (rows={len(rows)})")


# ----------------------------
# Merge gt_*.csv -> total_gt.csv
# ----------------------------
def make_total_gt():
    total_path = GT_DIR / "total_gt.csv"
    rows = []
    header = None

    for fpath in sorted(GT_DIR.glob("gt_*.csv")):
        if fpath.name == "total_gt.csv":
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            if header is None:
                header = rdr.fieldnames
            for r in rdr:
                rows.append(r)

    if header and rows:
        with open(total_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            w.writerows(rows)
        print(f"[TOTAL] total_gt.csv 생성 완료 → {total_path} (rows={len(rows)})")
    else:
        print("[TOTAL] gt_*.csv가 없어 total_gt.csv 생성 안 함.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subdir", default=None, help="예: brooklyn / gangnam / suwon ... (없으면 전체)")
    args = ap.parse_args()

    root_dir = BASE_DIR / "gsv_photo"
    if not root_dir.exists():
        print(f"[ERROR] gsv_photo 폴더 없음: {root_dir}")
        return

    if args.subdir:
        subdirs = [args.subdir]
    else:
        subdirs = [d.name for d in root_dir.iterdir() if d.is_dir()]

    for sd in sorted(subdirs):
        process_subdir(sd)

    make_total_gt()


if __name__ == "__main__":
    main()