#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Resume MORE training (continue fine-tuning) from already-saved fold best checkpoints.

- YOLO:  artifacts/yolo11x_kfold/runs/{prefix}_fold{i}/weights/best.pt
- FRCNN: artifacts/frcnn_kfold/fold{i}/best_frcnn.pth

IMPORTANT:
- We use ONE split json (YOLO split json) for BOTH YOLO and FRCNN.
  (No new split json creation.)

Run (PowerShell):
  python .\resume_more_train_kfold_both.py --prefix total --device 0 ^
    --yolo-split-json "artifacts\resume_both\yolo_resume_kfold\splits_total_k5_seed42.json" ^
    --yolo-extra-epochs 50 --frcnn-extra-epochs 30

Run (CMD):
  python resume_more_train_kfold_both.py --prefix total --device 0 ^
    --yolo-split-json "artifacts\resume_both\yolo_resume_kfold\splits_total_k5_seed42.json" ^
    --yolo-extra-epochs 50 --frcnn-extra-epochs 30
"""

import os, json, csv, shutil
from pathlib import Path
from statistics import mean, pstdev

import torch
from ultralytics import YOLO
from PIL import Image
import yaml

import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# ============================================================
# Common utils
# ============================================================
def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def ensure_exists(p: Path, msg: str):
    if not p.exists():
        raise RuntimeError(f"{msg}: {p}")

def backup_file(src: Path):
    """Make a simple .bak backup beside the file."""
    if not src.exists():
        return
    bak = src.with_suffix(src.suffix + ".bak")
    shutil.copy2(src, bak)


# ============================================================
# YOLO dataset builder (Labelme -> YOLO txt)
# ============================================================
CLASS_NAMES = ["signboard"]
CLASS_ID = 0

def parse_labelme(json_path: Path, jpg_path: Path):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    w = int(data.get("imageWidth", 0))
    h = int(data.get("imageHeight", 0))
    if w == 0 or h == 0:
        with Image.open(jpg_path) as im:
            w, h = im.size

    boxes = []
    for sh in data.get("shapes", []):
        pts = sh.get("points", [])
        if len(pts) < 2:
            continue
        st = sh.get("shape_type")
        if st == "rectangle" and len(pts) == 2:
            boxes.append(pts)
        elif st in ["polygon", "line", "linestrip", "circle"] and len(pts) >= 2:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            boxes.append([[x1, y1], [x2, y2]])
    return w, h, boxes

def to_yolo_line(pts, w, h):
    (x1, y1), (x2, y2) = pts
    xmin, xmax = sorted([x1, x2])
    ymin, ymax = sorted([y1, y2])
    bw = xmax - xmin
    bh = ymax - ymin
    cx = xmin + bw / 2
    cy = ymin + bh / 2
    return f"{CLASS_ID} {cx/w:.6f} {cy/h:.6f} {bw/w:.6f} {bh/h:.6f}"

def get_pairs(artifacts_dir: Path, prefix: str):
    if prefix == "total":
        region_root = artifacts_dir / "gsv_photo"
        pairs = []
        for region in ["brooklyn", "gangnam", "suwon"]:
            rdir = region_root / region
            if not rdir.exists():
                continue
            for jpg in sorted(rdir.glob("*.jpg")):
                js = jpg.with_suffix(".json")
                if js.exists():
                    pairs.append((jpg, js, region))
        return pairs
    else:
        region_dir = artifacts_dir / "gsv_photo" / prefix
        if not region_dir.exists():
            raise RuntimeError(f"Region folder not found: {region_dir}")
        pairs = []
        for jpg in sorted(region_dir.glob("*.jpg")):
            js = jpg.with_suffix(".json")
            if js.exists():
                pairs.append((jpg, js, prefix))
        return pairs

def build_yolo_dataset(dataset_dir: Path, pairs, split, prefix_tag: str):
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    (dataset_dir/"images/train").mkdir(parents=True, exist_ok=True)
    (dataset_dir/"images/val").mkdir(parents=True, exist_ok=True)
    (dataset_dir/"labels/train").mkdir(parents=True, exist_ok=True)
    (dataset_dir/"labels/val").mkdir(parents=True, exist_ok=True)

    train_pairs = [pairs[i] for i in split["train"]]
    val_pairs   = [pairs[i] for i in split["val"]]

    def process(pairs_, img_dst, lbl_dst):
        for jpg, js, region in pairs_:
            name = f"{prefix_tag}__{region}__{jpg.name}"
            shutil.copy2(jpg, img_dst / name)

            w, h, boxlist = parse_labelme(js, jpg)
            lbl = lbl_dst / name.replace(".jpg", ".txt")
            if w == 0 or h == 0 or not boxlist:
                lbl.write_text("", encoding="utf-8")
                continue
            lines = [to_yolo_line(pts, w, h) for pts in boxlist]
            lbl.write_text("\n".join(lines), encoding="utf-8")

    process(train_pairs, dataset_dir/"images/train", dataset_dir/"labels/train")
    process(val_pairs,   dataset_dir/"images/val",   dataset_dir/"labels/val")

    data_yaml = {
        "path": str(dataset_dir),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": CLASS_NAMES,
    }
    (dataset_dir/"data.yaml").write_text(yaml.dump(data_yaml, allow_unicode=True), encoding="utf-8")
    return dataset_dir/"data.yaml", len(train_pairs), len(val_pairs)

def yolo_eval_map50(weight_path: Path, data_yaml: Path, imgsz: int, device: int):
    model = YOLO(str(weight_path))
    metrics = model.val(data=str(data_yaml), imgsz=imgsz, device=device, verbose=False)
    return float(metrics.box.map50)


# ============================================================
# FRCNN (COCOeval)
# ============================================================
class CocoDet(CocoDetection):
    def __init__(self, img_root, ann_file):
        super().__init__(img_root, ann_file)

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)

        boxes, labels = [], []
        for t in target:
            x, y, w, h = t["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(int(t["category_id"]))

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        coco_img_id = int(self.ids[idx])  # IMPORTANT: COCO image_id
        return F.to_tensor(img), {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([coco_img_id], dtype=torch.int64)
        }

def collate_fn(batch):
    return tuple(zip(*batch))

def get_frcnn(num_classes=2):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model

def build_coco_subset(coco: COCO, image_ids: list[int]):
    image_ids = set(int(i) for i in image_ids)
    subset = {
        "info": coco.dataset.get("info", {}),
        "licenses": coco.dataset.get("licenses", []),
        "images": [img for img in coco.dataset.get("images", []) if int(img["id"]) in image_ids],
        "annotations": [ann for ann in coco.dataset.get("annotations", []) if int(ann["image_id"]) in image_ids],
        "categories": coco.dataset.get("categories", [])
    }
    coco_sub = COCO()
    coco_sub.dataset = subset
    coco_sub.createIndex()
    return coco_sub

@torch.no_grad()
def frcnn_train_one_epoch(model, loader, optimizer, device, scaler):
    model.train()
    total_loss = 0.0
    for imgs, targets in tqdm(loader, desc="Training", leave=False):
        imgs = [i.to(device) for i in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=True):
            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item())
    return total_loss / max(1, len(loader))

@torch.no_grad()
def frcnn_predict_coco_dets(model, loader, device, conf=0.001, cat_id=1):
    model.eval()
    dets = []
    for imgs, targets in tqdm(loader, desc="Predict", leave=False):
        imgs = [i.to(device) for i in imgs]
        outputs = model(imgs)
        for pred, tgt in zip(outputs, targets):
            image_id = int(tgt["image_id"].item())
            boxes = pred["boxes"].detach().cpu().numpy()
            scores = pred["scores"].detach().cpu().numpy()
            for (x1, y1, x2, y2), s in zip(boxes, scores):
                if float(s) < conf:
                    continue
                w = float(max(0.0, x2 - x1))
                h = float(max(0.0, y2 - y1))
                dets.append({
                    "image_id": image_id,
                    "category_id": int(cat_id),
                    "bbox": [float(x1), float(y1), w, h],
                    "score": float(s)
                })
    return dets

def coco_eval_and_save_pr(coco_gt: COCO, coco_dt_list, pr_curve_path: Path):
    coco_gt.dataset.setdefault("info", {})
    coco_gt.dataset.setdefault("licenses", [])

    if not coco_dt_list:
        pr_curve_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure()
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title("Precision–Recall Curve (IoU=0.5)")
        plt.grid(True)
        plt.savefig(pr_curve_path, dpi=200, bbox_inches="tight")
        plt.close()
        return {
            "AP@0.5": 0.0, "AP@0.75": 0.0,
            "AP_small": -1.0, "AP_medium": -1.0, "AP_large": 0.0
        }

    coco_dt = coco_gt.loadRes(coco_dt_list)
    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.evaluate(); ev.accumulate(); ev.summarize()

    ap50 = float(ev.stats[1])
    ap75 = float(ev.stats[2])
    ap_s = float(ev.stats[3])
    ap_m = float(ev.stats[4])
    ap_l = float(ev.stats[5])

    iou_thrs = ev.params.iouThrs
    t_idx = int(np.where(np.isclose(iou_thrs, 0.5))[0][0]) if np.any(np.isclose(iou_thrs, 0.5)) else 0
    precision = ev.eval["precision"]
    recall_thrs = ev.params.recThrs
    pr = precision[t_idx, :, 0, 0, 2]
    valid = pr > -1
    pr_valid = pr[valid]
    rec_valid = recall_thrs[valid]

    pr_curve_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    if len(pr_valid) > 0:
        plt.plot(rec_valid, pr_valid)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision–Recall Curve (IoU=0.5)")
    plt.grid(True)
    plt.savefig(pr_curve_path, dpi=200, bbox_inches="tight")
    plt.close()

    return {"AP@0.5": ap50, "AP@0.75": ap75, "AP_small": ap_s, "AP_medium": ap_m, "AP_large": ap_l}

@torch.no_grad()
def precision_recall_at_conf(model, loader, device, conf=0.5, iou_thr=0.5):
    model.eval()
    TP = FP = FN = 0

    def iou_xyxy(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, ax2-ax1) * max(0.0, ay2-ay1)
        area_b = max(0.0, bx2-bx1) * max(0.0, by2-by1)
        union = area_a + area_b - inter + 1e-9
        return inter / union

    for imgs, targets in tqdm(loader, desc="PR@conf", leave=False):
        imgs = [i.to(device) for i in imgs]
        outputs = model(imgs)

        for pred, tgt in zip(outputs, targets):
            gt_boxes = tgt["boxes"].cpu().numpy().tolist()

            boxes = pred["boxes"].detach().cpu().numpy()
            scores = pred["scores"].detach().cpu().numpy()

            pred_boxes = []
            for (x1, y1, x2, y2), s in zip(boxes, scores):
                if float(s) >= conf:
                    pred_boxes.append([float(x1), float(y1), float(x2), float(y2)])

            matched_gt = set()
            matched_pred = set()

            for pi, pb in enumerate(pred_boxes):
                best_iou = 0.0
                best_gi = None
                for gi, gb in enumerate(gt_boxes):
                    if gi in matched_gt:
                        continue
                    v = iou_xyxy(pb, gb)
                    if v > best_iou:
                        best_iou = v
                        best_gi = gi
                if best_gi is not None and best_iou >= iou_thr:
                    matched_pred.add(pi)
                    matched_gt.add(best_gi)

            TP += len(matched_pred)
            FP += (len(pred_boxes) - len(matched_pred))
            FN += (len(gt_boxes) - len(matched_gt))

    precision = TP / (TP + FP + 1e-9)
    recall    = TP / (TP + FN + 1e-9)
    return float(precision), float(recall)


# ============================================================
# Main
# ============================================================
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="total", help="brooklyn/gangnam/suwon/total")
    ap.add_argument("--device", type=int, default=0)

    # ✅ ONE split json for BOTH
    ap.add_argument("--yolo-split-json", required=True,
                    help="Use the SAME YOLO split json for BOTH YOLO+FRCNN. "
                         r'Example: artifacts\resume_both\yolo_resume_kfold\splits_total_k5_seed42.json')

    # YOLO extra training
    ap.add_argument("--yolo-imgsz", type=int, default=1280)
    ap.add_argument("--yolo-batch", type=int, default=2)
    ap.add_argument("--yolo-lr", type=float, default=1e-4)
    ap.add_argument("--yolo-extra-epochs", type=int, default=50)
    ap.add_argument("--yolo-patience", type=int, default=50)

    # FRCNN extra training
    ap.add_argument("--frcnn-batch", type=int, default=2)
    ap.add_argument("--frcnn-lr", type=float, default=0.005)
    ap.add_argument("--frcnn-extra-epochs", type=int, default=30)
    ap.add_argument("--frcnn-patience", type=int, default=20)
    ap.add_argument("--frcnn-cat-id", type=int, default=1)

    # paths
    ap.add_argument("--artifacts-dir", default="artifacts")

    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")
    torch.cuda.set_device(args.device)
    device_str = "cuda"
    print("[GPU]", torch.cuda.get_device_name(args.device))

    artifacts_dir = Path(args.artifacts_dir)

    # split json load (ONE)
    split_path = Path(args.yolo_split_json)
    ensure_exists(split_path, "Missing split json")
    splits = load_json(split_path)["splits"]

    # ----------------------
    # YOLO
    # ----------------------
    yolo_root = artifacts_dir / "yolo11x_kfold"
    ensure_exists(yolo_root, "Missing YOLO kfold dir")

    pairs = get_pairs(artifacts_dir, args.prefix)
    if not pairs:
        raise RuntimeError("No labelme pairs found for YOLO.")

    yolo_result_rows = []

    for fi, sp in enumerate(splits):
        print("\n" + "-"*78)
        print(f"[YOLO][FOLD {fi}/5] continue from saved best.pt (runs-based)")

        run_dir = yolo_root / "runs" / f"{args.prefix}_fold{fi}"
        base_w = run_dir / "weights" / "best.pt"
        ensure_exists(base_w, "Missing YOLO fold best model (expected runs/.../weights/best.pt)")
        print(f"[YOLO] resume from: {base_w}")

        tmp_ds = yolo_root / "tmp_resume_dataset" / f"{args.prefix}_fold{fi}"
        data_yaml, ntr, nva = build_yolo_dataset(tmp_ds, pairs, sp, prefix_tag=args.prefix)
        print(f"[YOLO] Train={ntr}, Val={nva}")

        old_map50 = yolo_eval_map50(base_w, data_yaml, imgsz=args.yolo_imgsz, device=args.device)
        print(f"[YOLO] current best.pt mAP@0.5 = {old_map50:.6f}")

        model = YOLO(str(base_w))

        run = model.train(
            data=str(data_yaml),
            epochs=args.yolo_extra_epochs,
            patience=args.yolo_patience,
            batch=args.yolo_batch,
            imgsz=args.yolo_imgsz,
            lr0=args.yolo_lr,
            optimizer="Adam",
            pretrained=True,
            device=args.device,
            amp=True,
            workers=0,
            cache=False,
            project=str(yolo_root / "runs"),
            name=f"{args.prefix}_fold{fi}",
            exist_ok=True,
            save=True,
            verbose=True
        )

        new_best = Path(run.save_dir) / "weights" / "best.pt"
        ensure_exists(new_best, "[YOLO] best.pt not found after training")

        new_map50 = yolo_eval_map50(new_best, data_yaml, imgsz=args.yolo_imgsz, device=args.device)
        print(f"[YOLO] new best.pt mAP@0.5 = {new_map50:.6f}")

        if new_map50 > old_map50 + 1e-6:
            print("[YOLO] IMPROVED -> keep updated best.pt and write best_yolo.pt")
            stable = run_dir / "best_yolo.pt"
            shutil.copy2(new_best, stable)
            final_map50 = new_map50
        else:
            print("[YOLO] NOT improved -> keep existing best.pt")
            final_map50 = old_map50

        yolo_result_rows.append([args.prefix, 5, fi, final_map50])

    yolo_resume_csv = yolo_root / "resume_more_summary_map50.csv"
    with yolo_resume_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["prefix","kfold","fold","final_mAP@0.5"])
        for r in yolo_result_rows:
            w.writerow(r)

    print(f"\n[YOLO] saved: {yolo_resume_csv}")

    # ----------------------
    # FRCNN  (USE SAME SPLIT JSON)
    # ----------------------
    frcnn_root = artifacts_dir / "frcnn_kfold"
    ensure_exists(frcnn_root, "Missing FRCNN kfold dir")

    coco_root = artifacts_dir / "yolo_ft_total"
    train_images = coco_root / "images" / "train"
    train_anns = coco_root / "annotations" / "instances_train.json"
    ensure_exists(train_images, "Missing FRCNN COCO train images dir")
    ensure_exists(train_anns, "Missing FRCNN COCO train json")

    coco_train = COCO(str(train_anns))
    ds = CocoDet(str(train_images), str(train_anns))

    frcnn_rows = []

    for fi, sp in enumerate(splits):
        print("\n" + "-"*78)
        print(f"[FRCNN][FOLD {fi}/5] continue from saved best_frcnn.pth (SAME split json)")

        fold_dir = frcnn_root / f"fold{fi}"
        base_ckpt = fold_dir / "best_frcnn.pth"
        ensure_exists(base_ckpt, "Missing FRCNN fold best checkpoint")

        # ✅ YOLO split의 train/val은 "dataset index"로 사용
        train_idx = [int(i) for i in sp["train"]]
        val_idx   = [int(i) for i in sp["val"]]

        if len(train_idx) == 0 or len(val_idx) == 0:
            raise RuntimeError(f"[FRCNN] empty split at fold {fi}")

        max_idx = max(train_idx + val_idx)
        if max_idx >= len(ds):
            raise RuntimeError(
                f"[FRCNN] split index out of range at fold {fi}. "
                f"max_idx={max_idx} but dataset_len={len(ds)}. "
                f"YOLO split과 COCO dataset 순서가 다르면 이 방식은 불가."
            )

        # COCOeval용: val_idx -> 실제 COCO image_id
        val_img_ids = [int(ds.ids[i]) for i in val_idx]

        train_loader = DataLoader(torch.utils.data.Subset(ds, train_idx),
                                  batch_size=args.frcnn_batch, shuffle=True,
                                  num_workers=0, collate_fn=collate_fn)
        val_loader = DataLoader(torch.utils.data.Subset(ds, val_idx),
                                batch_size=args.frcnn_batch, shuffle=False,
                                num_workers=0, collate_fn=collate_fn)

        coco_gt_fold = build_coco_subset(coco_train, val_img_ids)

        model = get_frcnn(num_classes=2).to(device_str)
        sd = torch.load(base_ckpt, map_location="cpu")
        model.load_state_dict(sd, strict=True)

        dets0 = frcnn_predict_coco_dets(model, val_loader, device_str, conf=0.001, cat_id=args.frcnn_cat_id)
        pr_path0 = fold_dir / "pr_curve_iou0.5.png"
        base_metrics = coco_eval_and_save_pr(coco_gt_fold, dets0, pr_curve_path=pr_path0)
        base_ap50 = base_metrics["AP@0.5"]
        base_prec, base_rec = precision_recall_at_conf(model, val_loader, device_str, conf=0.5, iou_thr=0.5)
        print(f"[FRCNN] current best_frcnn.pth AP@0.5={base_ap50:.6f} P={base_prec:.6f} R={base_rec:.6f}")

        optimizer = torch.optim.SGD(model.parameters(), lr=args.frcnn_lr, momentum=0.9, weight_decay=0.0005)
        scaler = torch.cuda.amp.GradScaler()

        best_ap50 = base_ap50
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        no_improve = 0
        best_epoch = 0

        for ep in range(args.frcnn_extra_epochs):
            loss = frcnn_train_one_epoch(model, train_loader, optimizer, device_str, scaler)

            dets = frcnn_predict_coco_dets(model, val_loader, device_str, conf=0.001, cat_id=args.frcnn_cat_id)
            metrics = coco_eval_and_save_pr(coco_gt_fold, dets, pr_curve_path=pr_path0)
            ap50 = metrics["AP@0.5"]

            print(f"[FRCNN][Epoch+{ep+1}/{args.frcnn_extra_epochs}] loss={loss:.4f} AP@0.5={ap50:.6f}")

            if ap50 > best_ap50 + 1e-6:
                best_ap50 = ap50
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                best_epoch = ep + 1
                no_improve = 0
                print(f"  >>> NEW BEST (continued) AP@0.5={best_ap50:.6f}")
            else:
                no_improve += 1
                if no_improve >= args.frcnn_patience:
                    print(f"[FRCNN][EARLY STOP] no improvement for {args.frcnn_patience}")
                    break

        if best_ap50 > base_ap50 + 1e-6:
            print("[FRCNN] IMPROVED -> update fold best_frcnn.pth")
            backup_file(base_ckpt)
            torch.save(best_state, base_ckpt)
        else:
            print("[FRCNN] NOT improved -> keep existing fold best_frcnn.pth")

        model.load_state_dict(best_state, strict=True)
        dets = frcnn_predict_coco_dets(model, val_loader, device_str, conf=0.001, cat_id=args.frcnn_cat_id)
        metrics = coco_eval_and_save_pr(coco_gt_fold, dets, pr_curve_path=pr_path0)
        prec, rec = precision_recall_at_conf(model, val_loader, device_str, conf=0.5, iou_thr=0.5)

        frcnn_rows.append([
            fi,
            metrics["AP@0.5"], metrics["AP@0.75"],
            metrics["AP_small"], metrics["AP_medium"], metrics["AP_large"],
            prec, rec,
            best_epoch,
            str(base_ckpt),
            str(pr_path0)
        ])

    frcnn_resume_csv = frcnn_root / "resume_more_summary.csv"
    with frcnn_resume_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fold","AP@0.5","AP@0.75","AP_small","AP_medium","AP_large","Precision","Recall",
                    "best_epoch_in_continued","best_ckpt","PR_curve_path"])
        for r in frcnn_rows:
            w.writerow([
                r[0],
                f"{r[1]:.6f}", f"{r[2]:.6f}",
                f"{r[3]:.6f}", f"{r[4]:.6f}", f"{r[5]:.6f}",
                f"{r[6]:.6f}", f"{r[7]:.6f}",
                r[8], r[9], r[10]
            ])

    ap50s = [r[1] for r in frcnn_rows]
    print(f"\n[FRCNN] saved: {frcnn_resume_csv}")
    print(f"[FRCNN] Mean AP@0.5 = {mean(ap50s):.6f}")
    print(f"[FRCNN] Std  AP@0.5 = {pstdev(ap50s) if len(ap50s)>1 else 0.0:.6f}")

    print("\n[ALL DONE] Continued training finished.")


if __name__ == "__main__":
    main()
