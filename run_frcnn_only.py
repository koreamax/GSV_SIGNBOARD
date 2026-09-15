#!/usr/bin/env python3
import csv
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torchvision
import torchvision.transforms as T

MIN_SIZE = 800
MAX_SIZE = 1333

def resize_keep_ratio(img: Image.Image, min_size=MIN_SIZE, max_size=MAX_SIZE):
    """짧은 변=min_size, 긴 변<=max_size로 비율 유지 resize"""
    w, h = img.size
    short, long_ = min(w, h), max(w, h)

    scale = min_size / short
    if long_ * scale > max_size:
        scale = max_size / long_

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    img_rs = img.resize((new_w, new_h), Image.BILINEAR)
    return img_rs, scale


def load_model(weight_path: Path):
    ckpt = torch.load(weight_path, map_location="cpu")

    # ckpt에서 num_classes 자동 추출
    num_classes = ckpt["roi_heads.box_predictor.cls_score.weight"].shape[0]

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )

    model.load_state_dict(ckpt, strict=True)
    model.eval()
    return model, num_classes


def run_inference(model, num_classes, region_dir: Path, score_thr=0.05):
    region = region_dir.name
    rows = []
    to_tensor = T.ToTensor()

    # ✅ json 섞여 있어도 jpg만 뽑음 (확장자 대소문자 안전)
    jpg_files = sorted([p for p in region_dir.iterdir() if p.suffix.lower() == ".jpg"])

    for img_path in jpg_files:
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        # FRCNN 기본 resize
        img_rs, scale = resize_keep_ratio(img)

        x = to_tensor(img_rs).unsqueeze(0)
        with torch.no_grad():
            pred = model(x)[0]

        boxes = pred["boxes"].cpu().numpy()   # resized 좌표
        scores = pred["scores"].cpu().numpy()
        labels = pred["labels"].cpu().numpy()

        keep = scores >= score_thr
        if num_classes == 2:
            keep = keep & (labels == 1)

        boxes = boxes[keep]
        scores = scores[keep]

        if len(boxes) == 0:
            continue

        # resized -> 원본 좌표 복원
        boxes = boxes / scale

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

        # ✅ filename 규칙: 무조건 region__파일명
        final_name = f"{region}__{img_path.name}"

        # ✅ YOLO 포맷: conf 포함, class 제거
        for (x1, y1, x2, y2), conf in zip(boxes, scores):
            rows.append({
                "filename": final_name,
                "w": orig_w,
                "h": orig_h,
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "conf": float(conf)
            })

    return rows


def main():
    WEIGHTS = Path("artifacts/frcnn/best_frcnn.pth")
    IMG_ROOT = Path("artifacts/gsv_photo")
    OUT_CSV = Path("artifacts/frcnn/total_frcnn.csv")

    model, num_classes = load_model(WEIGHTS)
    print(f"[FRCNN] Loaded num_classes = {num_classes}")

    all_rows = []
    for region_dir in IMG_ROOT.iterdir():
        if region_dir.is_dir():
            print(f"[FRCNN] Running inference on region: {region_dir.name}")
            all_rows.extend(run_inference(model, num_classes, region_dir))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(
            f, fieldnames=["filename","w","h","x1","y1","x2","y2","conf"]
        )
        wr.writeheader()
        wr.writerows(all_rows)

    print(f"\n[FRCNN] Done. Saved total CSV → {OUT_CSV} ({len(all_rows)} boxes)")


if __name__ == "__main__":
    main()
