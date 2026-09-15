#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent / "artifacts"
YOLO_DIR = BASE_DIR / "yolo"
YOLO_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# util: merge all yolo_*.csv → total_yolo.csv (filename 겹침 방지)
#   - filename을 "subdir__원본파일명" 형태로 통일
# ============================================================
def make_total_yolo():
    total_path = YOLO_DIR / "total_yolo.csv"
    rows = []
    header = None

    for fpath in sorted(YOLO_DIR.glob("yolo_*.csv")):
        if fpath.name == "total_yolo.csv":
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
        print(f"[TOTAL] total_yolo.csv 생성 완료 → {total_path} (rows={len(rows)})")
    else:
        print("[TOTAL] yolo_*.csv가 없어서 total_yolo.csv를 생성하지 않음.")

# ============================================================
# 필터 함수들
# ============================================================
def is_plausible_signboard(box, w, h, min_ratio=0.015):
    x1, y1, x2, y2 = box
    area = max(0, x2 - x1) * max(0, y2 - y1)
    return area >= (w * h * min_ratio)

def filter_signboard_kaggle_style(box, w, h):
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    if bw > w * 0.95:
        return False
    if bh > h * 0.95:
        return False
    return True

def is_copyright_box(box, w, h):
    x1, y1, x2, y2 = box
    return (x1 > w * 0.80 and y1 > h * 0.80)

def load_processed_filenames(csv_path: Path):
    s = set()
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            if rdr.fieldnames and "filename" in rdr.fieldnames:
                for r in rdr:
                    s.add(r["filename"])
    return s

def iter_images(root: Path):
    for ext in ("*.jpg", "*.png"):
        for p in root.glob(ext):
            yield p

# ============================================================
# 폴더 단위 YOLO 실행
#   - filename을 subdir 포함 형태로 저장해 total에서 안겹치게
#     ex) gangnam_1__12.jpg
# ============================================================
def run_yolo_on_folder(img_dir: Path, model: YOLO, args):
    out_csv = YOLO_DIR / f"yolo_{img_dir.name}.csv"
    processed = load_processed_filenames(out_csv)
    print(f"\n[YOLO][{img_dir.name}] 기존 처리 {len(processed)}개")

    rows = []
    count_new = 0

    for img_path in iter_images(img_dir):
        raw_fname = img_path.name
        fname = f"{img_dir.name}__{raw_fname}"  # ✅ subdir prefix 붙임

        if fname in processed:
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        w, h = img.size

        try:
            results = model.predict(
                source=str(img_path),
                conf=args.conf,
                iou=args.iou,
                verbose=False
            )
        except Exception as e:
            print(f"[ERROR] {raw_fname}: {e}")
            continue

        r = results[0]
        if not getattr(r, "boxes", None):
            continue

        detected = False
        for b in r.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            box = (x1, y1, x2, y2)

            if not is_plausible_signboard(box, w, h, args.min_box_area_ratio):
                continue
            if not filter_signboard_kaggle_style(box, w, h):
                continue
            if is_copyright_box(box, w, h):
                continue

            conf = float(b.conf[0])
            rows.append({
                "filename": fname,  # ✅ prefix 포함
                "w": w,
                "h": h,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "conf": conf
            })
            detected = True

        if detected:
            processed.add(fname)
            count_new += 1

    if rows:
        file_exists = out_csv.exists()
        with open(out_csv, "a", newline="", encoding="utf-8") as f:
            header = ["filename", "w", "h", "x1", "y1", "x2", "y2", "conf"]
            wri = csv.DictWriter(f, fieldnames=header)
            if not file_exists:
                wri.writeheader()
            wri.writerows(rows)

        print(f"[YOLO][{img_dir.name}] NEW {count_new}개 저장 → {out_csv}")
    else:
        print(f"[YOLO][{img_dir.name}] 탐지 없음")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="YOLO weight path")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--min-box-area-ratio", type=float, default=0.01)
    ap.add_argument("--root", default=str(BASE_DIR / "gsv_photo"))
    ap.add_argument("--prefix", default=None, help="gangnam / suwon / brooklyn 등")
    args = ap.parse_args()

    root_dir = Path(args.root)
    if not root_dir.exists():
        print(f"[ERROR] root not found: {root_dir}")
        return

    model = YOLO(args.model)

    if args.prefix:
        subdirs = [d for d in root_dir.iterdir()
                   if d.is_dir() and d.name.startswith(args.prefix)]
    else:
        subdirs = [d for d in root_dir.iterdir() if d.is_dir()]

    for d in sorted(subdirs):
        run_yolo_on_folder(d, model, args)

    make_total_yolo()

if __name__ == "__main__":
    main()
