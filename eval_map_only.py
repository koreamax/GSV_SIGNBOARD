#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate mAP for:
  1) Ultralytics YOLO (best_yolo.pt)
  2) Torchvision Faster R-CNN (best_frcnn.pth)
  3) EfficientDet (best_effdet.pth)

Dataset root: artifacts/yolo_ft_total
  - images/val
  - labels/val (YOLO txt)
  - annotations/instances_val.json (COCO)
  - data.yml (YOLO yaml)

Outputs:
  - YOLO mAP@0.5, mAP@0.5:0.95 (ultralytics val)
  - FRCNN mAP@0.5 (pycocotools COCOeval with IoU=0.5 only)
  - EfficientDet AP@0.5 (single-class, same letterbox 512 space)
"""

import os
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------
# Paths (fixed to your project structure)
# ---------------------------------------------------------
DATA_ROOT = Path("artifacts/yolo_ft_total")
YOLO_DATA_YAML = DATA_ROOT / "data.yaml"

YOLO_CKPT = Path("artifacts/yolo/best_yolo.pt")
FRCNN_CKPT = Path("artifacts/frcnn/best_frcnn.pth")

# ✅ EfficientDet ckpt 추가
EFFDET_CKPT = Path("artifacts/efficientdet/best_effdet.pth")

COCO_VAL_JSON = DATA_ROOT / "annotations/instances_val.json"
VAL_IMG_DIR = DATA_ROOT / "images/val"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------
# 1) YOLO mAP evaluation (Ultralytics built-in val)
# ---------------------------------------------------------
def eval_yolo_map():
    print("\n==========================")
    print("[YOLO] evaluation (mAP)")
    print("==========================")

    try:
        from ultralytics import YOLO
    except Exception as e:
        raise ImportError("ultralytics not installed. pip install ultralytics") from e

    model = YOLO(str(YOLO_CKPT))

    results = model.val(
        data=str(YOLO_DATA_YAML),
        imgsz=1280,
        batch=8,
        device=0 if DEVICE == "cuda" else "cpu",
        verbose=False
    )

    map50 = float(results.box.map50)
    map5095 = float(results.box.map)

    print(f"[YOLO] mAP@0.5     = {map50:.4f}")

    return map50, map5095


# ---------------------------------------------------------
# 2) Faster R-CNN dataset wrapper for COCO val
# ---------------------------------------------------------
class CocoValDataset(Dataset):
    def __init__(self, img_dir: Path, ann_json: Path):
        from pycocotools.coco import COCO
        self.coco = COCO(str(ann_json))
        self.img_dir = img_dir
        self.img_ids = list(self.coco.imgs.keys())
        self.cat_id_to_contiguous = {cat_id: i + 1 for i, cat_id in enumerate(self.coco.getCatIds())}
        # background=0, so contiguous class starts at 1

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        img_path = self.img_dir / info["file_name"]

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        for a in anns:
            x, y, bw, bh = a["bbox"]
            boxes.append([x, y, x + bw, y + bh])
            labels.append(self.cat_id_to_contiguous[a["category_id"]])

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
        }

        img_t = F.to_tensor(img)
        return img_t, target, (w, h), info["file_name"]


def collate_fn(batch):
    imgs, targets, metas, fnames = zip(*batch)
    return list(imgs), list(targets), list(metas), list(fnames)


# ---------------------------------------------------------
# 3) Faster R-CNN mAP evaluation (COCOeval)
# ---------------------------------------------------------
@torch.no_grad()
def eval_frcnn_map(score_thr=0.001):
    print("\n==========================")
    print("[Faster R-CNN] evaluation (mAP@0.5)")
    print("==========================")

    try:
        from pycocotools.cocoeval import COCOeval
    except Exception as e:
        raise ImportError("pycocotools not installed. pip install pycocotools") from e

    dataset = CocoValDataset(VAL_IMG_DIR, COCO_VAL_JSON)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0,
                        collate_fn=collate_fn)

    model = fasterrcnn_resnet50_fpn(weights=None, num_classes=2)
    ckpt = torch.load(FRCNN_CKPT, map_location="cpu")

    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    model.to(DEVICE).eval()
    coco_gt = dataset.coco

    coco_gt.dataset.setdefault("info", {})
    coco_gt.dataset.setdefault("licenses", [])
    coco_gt.dataset.setdefault("categories", coco_gt.dataset.get("categories", []))

    coco_results = []

    for imgs, targets, metas, fnames in tqdm(loader, desc="Infer FRCNN"):
        imgs = [im.to(DEVICE) for im in imgs]
        outputs = model(imgs)

        for out, tgt in zip(outputs, targets):
            img_id = int(tgt["image_id"].item())

            boxes = out["boxes"].detach().cpu()
            scores = out["scores"].detach().cpu()
            labels = out["labels"].detach().cpu()

            for box, score, lab in zip(boxes, scores, labels):
                if float(score) < score_thr:
                    continue
                x1, y1, x2, y2 = box.tolist()
                coco_results.append({
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score)
                })

    if len(coco_results) == 0:
        print("[FRCNN] No predictions above threshold. Check score_thr.")
        return 0.0

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.iouThrs = [0.5]

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    ap50 = float(coco_eval.stats[0])
    print(f"[FRCNN] mAP@0.5 = {ap50:.4f}")
    return ap50


# =========================================================
# 4) EfficientDet mAP@0.5 evaluation 추가
# =========================================================
# Letterbox resize (keep AR + pad)
def letterbox_pil(img: Image.Image, new_size=512, color=(114, 114, 114)):
    w, h = img.size
    scale = min(new_size / w, new_size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))

    img_resized = img.resize((nw, nh), resample=Image.BILINEAR)
    new_img = Image.new("RGB", (new_size, new_size), color)
    pad_left = (new_size - nw) // 2
    pad_top  = (new_size - nh) // 2
    new_img.paste(img_resized, (pad_left, pad_top))
    return new_img, scale, pad_left, pad_top


class CocoValDatasetEffDet(Dataset):
    """
    COCO val -> EfficientDet letterbox(512) space
    single-class forced labels=0
    """
    def __init__(self, img_dir: Path, ann_json: Path, img_size=512):
        from pycocotools.coco import COCO
        self.coco = COCO(str(ann_json))
        self.img_dir = img_dir
        self.img_ids = list(self.coco.imgs.keys())
        self.img_size = img_size

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        img_path = self.img_dir / info["file_name"]

        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        for a in anns:
            x, y, bw, bh = a["bbox"]
            boxes.append([x, y, x + bw, y + bh])

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)

        # letterbox + bbox transform
        img_lb, scale, pad_left, pad_top = letterbox_pil(img, self.img_size)

        if boxes.numel() > 0:
            boxes *= scale
            boxes[:, [0, 2]] += pad_left
            boxes[:, [1, 3]] += pad_top
            boxes[:, 0::2] = boxes[:, 0::2].clamp(0, self.img_size)
            boxes[:, 1::2] = boxes[:, 1::2].clamp(0, self.img_size)

        target = {
            "boxes": boxes,
            "labels": torch.zeros((boxes.shape[0],), dtype=torch.int64),  # single class=0
            "image_id": torch.tensor([img_id]),
            "file_name": f"effdet__img{img_id}__{info['file_name']}",
        }

        return F.to_tensor(img_lb), target


def collate_fn_effdet(batch):
    images, targets = zip(*batch)
    images = torch.stack(images).float()
    boxes = [t["boxes"].float() for t in targets]
    labels = [t["labels"].long() for t in targets]

    annotations = {
        "bbox": boxes,
        "cls": labels,
        "img_size": torch.tensor([[images.shape[2], images.shape[3]]] * len(images)).float(),
        "img_scale": torch.ones((len(images), 1)).float(),
    }
    return images, annotations, targets


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union


def _load_effdet_ckpt_safely(net, ckpt_path, device="cpu"):
    """
    - ckpt가 meta dict일 수도 있어서 state_dict 자동 추출
    - module./model./net. prefix 자동 제거
    - strict=False로 안전 로딩
    """
    ckpt_raw = torch.load(ckpt_path, map_location=device)

    # 1) meta dict면 안에서 진짜 weight dict 꺼내기
    if isinstance(ckpt_raw, dict):
        for k in ["state_dict", "model_state_dict", "model", "net"]:
            if k in ckpt_raw and isinstance(ckpt_raw[k], dict):
                sd = ckpt_raw[k]
                break
        else:
            sd = ckpt_raw
    else:
        sd = ckpt_raw

    # 2) prefix 제거
    new_sd = {}
    for k, v in sd.items():
        kk = k
        for p in ["module.", "model.", "model.model.", "net."]:
            if kk.startswith(p):
                kk = kk[len(p):]
        new_sd[kk] = v

    missing, unexpected = net.load_state_dict(new_sd, strict=False)
    print("[CKPT LOAD] missing(sample) =", missing[:5])
    print("[CKPT LOAD] unexpected(sample) =", unexpected[:5])
    return net


def eval_effdet_map(score_thr=0.05, model_name="tf_efficientdet_d0", img_size=512):
    print("\n==========================")
    print("[EfficientDet] evaluation (AP@0.5)")
    print("==========================")

    try:
        from effdet import create_model, DetBenchPredict
    except Exception as e:
        raise ImportError("effdet not installed. pip install effdet") from e

    dataset = CocoValDatasetEffDet(VAL_IMG_DIR, COCO_VAL_JSON, img_size=img_size)
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn_effdet
    )

    # ✅ 1) "순수 net"으로 만들고 (bench_task="")
    net = create_model(
        model_name,
        bench_task="",           # 순수 EfficientDet Net
        num_classes=1,
        pretrained=False,
        image_size=(img_size, img_size),
    )

    # ✅ 2) ckpt 안전하게 로드 (meta/ prefix 다 처리)
    net = _load_effdet_ckpt_safely(net, EFFDET_CKPT, device="cpu")

    # ✅ 3) DetBenchPredict는 "한 번만"
    pred_bench = DetBenchPredict(net).to(DEVICE).eval()

    preds_by_file = defaultdict(list)
    gts_by_file = defaultdict(list)

    for images, _, targets in tqdm(loader, desc="Infer EffDet"):
        images = images.to(DEVICE)

        # effdet DetBenchPredict 출력:
        # batch마다 (N_i, 6) 텐서 리스트
        # [x1, y1, x2, y2, score, cls]
        outputs = pred_bench(images)

        for out, tgt in zip(outputs, targets):
            fn = tgt["file_name"]

            # preds
            for det in out.detach().cpu().tolist():
                x1, y1, x2, y2, s, cls = det
                if s >= score_thr:
                    preds_by_file[fn].append((x1, y1, x2, y2, s))

            # gts (이미 xyxy)
            for box in tgt["boxes"].cpu().tolist():
                gts_by_file[fn].append(tuple(box))

    # ---- AP50 calc (single class) ----
    all_scores = []
    all_matches = []

    for fn, pred_list in preds_by_file.items():
        gt_list = gts_by_file.get(fn, [])
        used = [False] * len(gt_list)
        pred_list.sort(key=lambda x: x[4], reverse=True)

        for (x1, y1, x2, y2, s) in pred_list:
            best_iou = 0.0
            best_j = -1
            for j, gt in enumerate(gt_list):
                if used[j]:
                    continue
                iou = iou_xyxy((x1, y1, x2, y2), gt)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_iou >= 0.5 and best_j >= 0:
                used[best_j] = True
                all_matches.append(1)
            else:
                all_matches.append(0)

            all_scores.append(s)

    total_gt = sum(len(v) for v in gts_by_file.values())
    if total_gt == 0 or len(all_scores) == 0:
        print("[EffDet] No GT or no predictions.")
        return 0.0

    sorted_idx = sorted(range(len(all_scores)), key=lambda i: all_scores[i], reverse=True)
    tp = 0
    fp = 0
    precisions = []
    recalls = []

    for i in sorted_idx:
        if all_matches[i] == 1:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / max(1, tp + fp))
        recalls.append(tp / total_gt)

    ap = 0.0
    for r_th in [i / 10 for i in range(11)]:
        p_max = 0.0
        for p, r in zip(precisions, recalls):
            if r >= r_th:
                p_max = max(p_max, p)
        ap += p_max
    ap /= 11.0

    print(f"[EfficientDet] AP@0.5 = {ap:.4f}")
    return ap


# ---------------------------------------------------------
# main
# ---------------------------------------------------------
def main():
    print(f"DEVICE={DEVICE}")

    assert YOLO_CKPT.exists(), f"YOLO ckpt not found: {YOLO_CKPT}"
    assert FRCNN_CKPT.exists(), f"FRCNN ckpt not found: {FRCNN_CKPT}"
    assert EFFDET_CKPT.exists(), f"EffDet ckpt not found: {EFFDET_CKPT}"
    assert YOLO_DATA_YAML.exists(), f"data.yaml not found: {YOLO_DATA_YAML}"
    assert COCO_VAL_JSON.exists(), f"COCO val json not found: {COCO_VAL_JSON}"
    assert VAL_IMG_DIR.exists(), f"val image dir not found: {VAL_IMG_DIR}"

    yolo_map50, yolo_map5095 = eval_yolo_map()
    frcnn_map50 = eval_frcnn_map()
    effdet_map50 = eval_effdet_map()

    print("\n==========================")
    print("Final Summary")
    print("==========================")
    print(f"YOLO   mAP@0.5      : {yolo_map50:.4f}")
    print(f"FRCNN  mAP@0.5      : {frcnn_map50:.4f}")
    print(f"EffDet AP@0.5       : {effdet_map50:.4f}")


if __name__ == "__main__":
    main()
