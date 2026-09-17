#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5-Fold Cross Validation training for YOLOv11x with Labelme GT.
- GPU 강제 사용
- Early Stopping(patience)
- fold별 best epoch 로그(results.csv 분석)
- prefix: brooklyn / gangnam / suwon / total

Run example:
  python train_yolo11x_5fold.py --prefix total --model yolo11x.pt --kfold 5 --imgsz 1280 --batch 8 --lr 1e-4 --patience 50 --device 0
"""

import os, json, shutil, random, csv
from pathlib import Path
import yaml
from ultralytics import YOLO
from PIL import Image
import torch
from statistics import mean, pstdev
import numpy as np
import matplotlib.pyplot as plt

# pycocotools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def _load_data_yaml(data_yaml: Path):
    data = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    root = Path(data["path"])
    val_rel = data["val"]
    val_dir = root / val_rel
    lbl_val_dir = root / "labels" / "val"
    img_dir = val_dir
    return root, img_dir, lbl_val_dir

def _yolo_txt_to_coco_bbox(line: str, W: int, H: int):
    # line: "cls cx cy w h" (normalized)
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    _, cx, cy, bw, bh = parts[:5]
    cx = float(cx) * W
    cy = float(cy) * H
    bw = float(bw) * W
    bh = float(bh) * H
    x = cx - bw / 2
    y = cy - bh / 2
    return [x, y, bw, bh]

def build_coco_gt_from_yolo_val(data_yaml: Path, category_name="signboard"):
    """
    val split을 COCO GT json(dict)로 구성
    - category_id: 1
    """
    root, img_dir, lbl_dir = _load_data_yaml(data_yaml)

    images = []
    annotations = []
    ann_id = 1
    img_id = 1

    img_paths = sorted(list(img_dir.glob("*.jpg")))
    if not img_paths:
        # png도 지원하고 싶으면 여기 추가
        img_paths = sorted(list(img_dir.glob("*.png")))

    for p in img_paths:
        with Image.open(p) as im:
            W, H = im.size

        images.append({
            "id": img_id,
            "file_name": p.name,
            "width": W,
            "height": H
        })

        txt = lbl_dir / p.with_suffix(".txt").name
        if txt.exists():
            for line in txt.read_text(encoding="utf-8").splitlines():
                bbox = _yolo_txt_to_coco_bbox(line, W, H)
                if bbox is None:
                    continue
                x, y, w, h = bbox
                # COCO: bbox는 xywh, area 필요
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "area": float(max(w, 0) * max(h, 0)),
                    "iscrowd": 0
                })
                ann_id += 1

        img_id += 1

    coco_gt = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": category_name}]
    }
    return coco_gt

def predict_to_coco_dets(model: YOLO, data_yaml: Path, imgsz=640, device=0, conf=0.001):
    """
    val 이미지들에 대해 예측 → COCO det list로 반환
    - category_id: 1
    """
    root, img_dir, _ = _load_data_yaml(data_yaml)

    img_paths = sorted(list(img_dir.glob("*.jpg")))
    if not img_paths:
        img_paths = sorted(list(img_dir.glob("*.png")))

    # COCO image_id 매핑 (file_name -> id)
    # build_coco_gt_from_yolo_val과 동일 순서로 id가 붙는다고 가정
    file_to_id = {p.name: i+1 for i, p in enumerate(img_paths)}

    dets = []
    for p in img_paths:
        # ultralytics predict
        r = model.predict(
            source=str(p),
            imgsz=imgsz,
            device=device,
            conf=conf,
            verbose=False
        )[0]

        if r.boxes is None or len(r.boxes) == 0:
            continue

        # xyxy, conf
        xyxy = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()

        for (x1, y1, x2, y2), sc in zip(xyxy, scores):
            w = float(max(0.0, x2 - x1))
            h = float(max(0.0, y2 - y1))
            dets.append({
                "image_id": int(file_to_id[p.name]),
                "category_id": 1,
                "bbox": [float(x1), float(y1), w, h],  # COCO xywh
                "score": float(sc)
            })

    return dets

def coco_eval_metrics(coco_gt_dict, coco_dt_list, save_pr_curve_path: Path):
    """
    COCOeval로:
    - AP@0.5, AP@0.75
    - AP_small / AP_medium / AP_large
    추출 + PR Curve 저장( IoU=0.5, area=all, maxDets=100, 단일 클래스 )
    """
    coco_gt = COCO()
    coco_gt.dataset = coco_gt_dict
    coco_gt.dataset.setdefault("info", {})
    coco_gt.dataset.setdefault("licenses", [])
    coco_gt.dataset.setdefault("images", [])
    coco_gt.dataset.setdefault("annotations", [])
    coco_gt.dataset.setdefault("categories", [])
    coco_gt.createIndex()

    coco_dt = coco_gt.loadRes(coco_dt_list) if len(coco_dt_list) > 0 else coco_gt.loadRes([])

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # COCOeval.stats 의미:
    # 0: AP@[.5:.95], 1: AP@0.5, 2: AP@0.75, 3: AP_small, 4: AP_medium, 5: AP_large, ...
    ap50 = float(coco_eval.stats[1])
    ap75 = float(coco_eval.stats[2])
    ap_s  = float(coco_eval.stats[3])
    ap_m  = float(coco_eval.stats[4])
    ap_l  = float(coco_eval.stats[5])

    # PR Curve: precision[T, R, K, A, M]
    # IoU=0.5는 첫 번째 T(0) 인 경우가 많지만, 정확히는 iouThrs에서 0.5 인덱스를 찾자
    iou_thrs = coco_eval.params.iouThrs
    t_idx = int(np.where(np.isclose(iou_thrs, 0.5))[0][0]) if np.any(np.isclose(iou_thrs, 0.5)) else 0

    precision = coco_eval.eval["precision"]  # shape: [T, R, K, A, M]
    recall_thrs = coco_eval.params.recThrs

    # K=1, A=all(0), M=maxDets=100(2) 가 보통
    pr = precision[t_idx, :, 0, 0, 2]
    # -1은 invalid
    valid = pr > -1
    pr_valid = pr[valid]
    recall_valid = recall_thrs[valid]

    # 저장
    save_pr_curve_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    if len(pr_valid) > 0:
        plt.plot(recall_valid, pr_valid)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve (IoU=0.5)")
    plt.grid(True)
    plt.savefig(save_pr_curve_path, dpi=200, bbox_inches="tight")
    plt.close()

    return {
        "AP@0.5": ap50,
        "AP@0.75": ap75,
        "AP_small": ap_s,
        "AP_medium": ap_m,
        "AP_large": ap_l
    }

def precision_recall_at_conf(model: YOLO, data_yaml: Path, imgsz=640, device=0, conf=0.5, iou_thr=0.5):
    """
    conf 임계값 기반 Precision/Recall 계산 (단일 클래스)
    - Greedy matching IoU>=iou_thr
    """
    root, img_dir, lbl_dir = _load_data_yaml(data_yaml)

    img_paths = sorted(list(img_dir.glob("*.jpg")))
    if not img_paths:
        img_paths = sorted(list(img_dir.glob("*.png")))

    TP = 0
    FP = 0
    FN = 0

    for p in img_paths:
        # GT boxes
        with Image.open(p) as im:
            W, H = im.size

        gt_txt = lbl_dir / p.with_suffix(".txt").name
        gt_boxes = []
        if gt_txt.exists():
            for line in gt_txt.read_text(encoding="utf-8").splitlines():
                bbox = _yolo_txt_to_coco_bbox(line, W, H)  # xywh
                if bbox is None:
                    continue
                x, y, w, h = bbox
                gt_boxes.append([x, y, x+w, y+h])  # xyxy

        # Pred boxes
        r = model.predict(
            source=str(p),
            imgsz=imgsz,
            device=device,
            conf=conf,
            verbose=False
        )[0]

        pred_boxes = []
        if r.boxes is not None and len(r.boxes) > 0:
            pred_boxes = r.boxes.xyxy.cpu().numpy().tolist()

        # Matching
        matched_gt = set()
        matched_pred = set()

        def iou_xyxy(a, b):
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix1 = max(ax1, bx1)
            iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2)
            iy2 = min(ay2, by2)
            iw = max(0.0, ix2 - ix1)
            ih = max(0.0, iy2 - iy1)
            inter = iw * ih
            area_a = max(0.0, ax2-ax1) * max(0.0, ay2-ay1)
            area_b = max(0.0, bx2-bx1) * max(0.0, by2-by1)
            union = area_a + area_b - inter + 1e-9
            return inter / union

        # greedy: 각 pred에 대해 best gt를 매칭
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



CLASS_NAMES = ["signboard"]
CLASS_ID = 0

# ============================================================
# Labelme JSON → YOLO TXT 변환
# ============================================================
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


# ============================================================
# 데이터 수집 (prefix / total)
# ============================================================
def get_all_region_pairs(artifacts_dir: Path):
    region_root = artifacts_dir / "gsv_photo"
    all_pairs = []

    for region in ["brooklyn", "gangnam", "suwon"]:
        rdir = region_root / region
        if not rdir.exists():
            continue

        for jpg in sorted(rdir.glob("*.jpg")):
            js = jpg.with_suffix(".json")
            if js.exists():
                all_pairs.append((jpg, js, region))

    return all_pairs


def get_single_region_pairs(artifacts_dir: Path, prefix: str):
    region_dir = artifacts_dir / "gsv_photo" / prefix
    if not region_dir.exists():
        raise RuntimeError(f"Region folder not found: {region_dir}")

    pairs = []
    for jpg in sorted(region_dir.glob("*.jpg")):
        js = jpg.with_suffix(".json")
        if js.exists():
            pairs.append((jpg, js, prefix))

    return pairs


# ============================================================
# Fold용 dataset 생성
# ============================================================
def build_dataset_from_pairs(dataset_dir: Path, pairs, prefix_tag: str):
    """
    pairs: list of (jpg_path, json_path, region)
    dataset_dir 구조:
      images/train, images/val, labels/train, labels/val
    """
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    (dataset_dir/"images/train").mkdir(parents=True, exist_ok=True)
    (dataset_dir/"images/val").mkdir(parents=True, exist_ok=True)
    (dataset_dir/"labels/train").mkdir(parents=True, exist_ok=True)
    (dataset_dir/"labels/val").mkdir(parents=True, exist_ok=True)

    # train_pairs / val_pairs는 호출자가 분리해서 넘김
    train_pairs = [p for p in pairs if p[3] == "train"]  # (jpg, js, region, split)
    val_pairs   = [p for p in pairs if p[3] == "val"]

    def process(split_pairs, img_dst, lbl_dst):
        for jpg, js, region, _split in split_pairs:
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
        "val":   "images/val",
        "nc":    1,
        "names": CLASS_NAMES,
    }
    (dataset_dir/"data.yaml").write_text(
        yaml.dump(data_yaml, allow_unicode=True), encoding="utf-8"
    )

    return (dataset_dir/"data.yaml"), len(train_pairs), len(val_pairs)


# ============================================================
# 평가용 mAP@0.5 (GPU 강제)
# ============================================================
def eval_map50(model_path: Path, data_yaml: Path, imgsz=640, device=0):
    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        verbose=False,
        device=device
    )
    return float(metrics.box.map50)


# ============================================================
# results.csv 분석 (best epoch 로그)
# ============================================================
def analyze_training_results(results_csv: Path):
    if not results_csv.exists():
        print(f"[WARN] {results_csv} not found; skip best-epoch analysis.")
        return None

    with results_csv.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            print("[WARN] Empty results.csv")
            return None

        metric_col_candidates = [
            "metrics/mAP50(B)",
            "metrics/mAP50",
        ]

        metric_idx = None
        metric_name = None
        for cand in metric_col_candidates:
            if cand in header:
                metric_idx = header.index(cand)
                metric_name = cand
                break

        if metric_idx is None:
            print("[WARN] mAP@0.5 column not found in header: ", header)
            return None

        try:
            epoch_idx = header.index("epoch")
        except ValueError:
            epoch_idx = 0

        best = -1.0
        best_epoch = None
        improved_epochs = []

        for row in reader:
            if not row or len(row) <= max(epoch_idx, metric_idx):
                continue
            try:
                epoch_val = int(float(row[epoch_idx]))
                metric_val = float(row[metric_idx]) if row[metric_idx] != "" else None
            except ValueError:
                continue

            if metric_val is None:
                continue

            if metric_val > best:
                best = metric_val
                best_epoch = epoch_val
                improved_epochs.append((epoch_val, metric_val))

        if best_epoch is None:
            print("[WARN] Could not find any valid mAP values in results.csv")
            return None

        print("\n[TRAIN SUMMARY] Best metric history based on", metric_name)
        for ep, val in improved_epochs:
            print(f"  → New BEST at epoch {ep} (1-based {ep+1}): {val:.4f}")

        print(f"\n[FINAL BEST] epoch {best_epoch} (1-based {best_epoch+1}) with {metric_name}={best:.4f}")

        return {
            "best_epoch": best_epoch,
            "best_metric": best,
            "metric_name": metric_name,
            "history": improved_epochs,
        }


# ============================================================
# K-Fold Split
# ============================================================
def make_kfold_splits(pairs, k=5, seed=42):
    """
    pairs: list of (jpg, js, region)
    return: list of dicts: [{"train": [...], "val": [...]}] length k
    """
    rng = random.Random(seed)
    idxs = list(range(len(pairs)))
    rng.shuffle(idxs)

    folds = [[] for _ in range(k)]
    for i, idx in enumerate(idxs):
        folds[i % k].append(idx)

    splits = []
    for fi in range(k):
        val_idx = set(folds[fi])
        train = [pairs[i] for i in range(len(pairs)) if i not in val_idx]
        val   = [pairs[i] for i in range(len(pairs)) if i in val_idx]
        splits.append({"train": train, "val": val})
    return splits


# ============================================================
# 5-Fold Cross Validation Training (YOLOv11x)
# ============================================================
def finetune_kfold(prefix: str,
                   artifacts_dir="artifacts",
                   out_root="artifacts/yolo11x_kfold",
                   model_ckpt="yolo11x.pt",
                   kfold=5,
                   max_epochs=9999,
                   batch=8,
                   imgsz=1280,
                   lr=1e-4,
                   patience=50,
                   device=0,
                   seed=42,
                   export_best_to=None):
    """
    export_best_to: 지정하면, fold 중 가장 좋은 best.pt를 해당 경로로 복사
      예) artifacts/yolo/best_yolo11x_kfold.pt
    """

    artifacts_dir = Path(artifacts_dir)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # ✅ GPU 강제 체크
    print("\n[GPU CHECK]")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. GPU 강제 사용 불가. (torch cuda 설치 확인)")
    print("CUDA available: True")
    print("GPU:", torch.cuda.get_device_name(device))

    # ----- 데이터 쌍 수집 -----
    if prefix == "total":
        base_pairs = get_all_region_pairs(artifacts_dir)
        prefix_tag = "total"
    else:
        base_pairs = get_single_region_pairs(artifacts_dir, prefix)
        prefix_tag = prefix

    if not base_pairs:
        raise RuntimeError(f"No images found for prefix={prefix}")

    print(f"\n[DATA] prefix={prefix} total_pairs={len(base_pairs)} kfold={kfold}")

    # ----- k-fold split -----
    splits = make_kfold_splits(base_pairs, k=kfold, seed=seed)

    fold_map50 = []
    fold_best_paths = []

    best_overall = -1.0
    best_fold = None
    best_fold_ckpt = None

    fold_rows = []
    for fi, sp in enumerate(splits):
        print("\n" + "="*70)
        print(f"[FOLD {fi+1}/{kfold}] Building dataset...")

        # fold dataset dir
        fold_dir = out_root / f"{prefix_tag}_fold{fi}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # split 정보를 dataset builder에 맞게 태깅
        tagged = []
        for (jpg, js, region) in sp["train"]:
            tagged.append((jpg, js, region, "train"))
        for (jpg, js, region) in sp["val"]:
            tagged.append((jpg, js, region, "val"))

        data_yaml, ntr, nva = build_dataset_from_pairs(fold_dir / "dataset", tagged, prefix_tag=prefix_tag)
        print(f"[FOLD {fi}] Train={ntr} Val={nva}")

        # ----- Train -----
        print(f"\n[FOLD {fi}] Training YOLO model: {model_ckpt}")
        model = YOLO(model_ckpt)  # ✅ YOLOv11x pretrained 시작

        run = model.train(
            data=str(data_yaml),
            epochs=max_epochs,
            patience=patience,
            batch=batch,
            imgsz=imgsz,
            lr0=lr,
            pretrained=True,
            optimizer="Adam",
            verbose=True,
            save=True,
            device=device,
            name=f"{prefix_tag}_fold{fi}",
            project=str(out_root / "runs")
        )

        # results.csv 분석
        results_csv = Path(run.save_dir) / "results.csv"
        analyze_training_results(results_csv)

        # fold best checkpoint
        fold_best = Path(run.save_dir) / "weights" / "best.pt"
        fold_best_paths.append(fold_best)

        print(f"\n[FOLD {fi}] Evaluating fold best checkpoint (COCO metrics + PR curve)...")

        # 1) COCO GT 구성  
        coco_gt = build_coco_gt_from_yolo_val(data_yaml)

        # 2) 예측을 COCO dets로 생성 (conf는 낮게)
        best_model = YOLO(str(fold_best))
        coco_dt = predict_to_coco_dets(best_model, data_yaml, imgsz=imgsz, device=device, conf=0.001)

        # 3) COCOeval AP metrics + PR curve 저장
        pr_curve_path = out_root / "pr_curves" / f"{prefix_tag}_fold{fi}_pr_iou0.5.png"
        ap_metrics = coco_eval_metrics(coco_gt, coco_dt, save_pr_curve_path=pr_curve_path)

        # 4) conf 임계값 기반 Precision / Recall (원하면 conf=0.5 조정 가능)
        prec, rec = precision_recall_at_conf(best_model, data_yaml, imgsz=imgsz, device=device, conf=0.5, iou_thr=0.5)

        print(f"[FOLD {fi}] AP@0.5  = {ap_metrics['AP@0.5']:.4f}")
        print(f"[FOLD {fi}] AP@0.75 = {ap_metrics['AP@0.75']:.4f}")
        print(f"[FOLD {fi}] AP_small  = {ap_metrics['AP_small']:.4f}")
        print(f"[FOLD {fi}] AP_medium = {ap_metrics['AP_medium']:.4f}")
        print(f"[FOLD {fi}] AP_large  = {ap_metrics['AP_large']:.4f}")
        print(f"[FOLD {fi}] Precision(conf=0.5, IoU=0.5) = {prec:.4f}")
        print(f"[FOLD {fi}] Recall   (conf=0.5, IoU=0.5) = {rec:.4f}")
        print(f"[FOLD {fi}] PR curve saved: {pr_curve_path}")

        fold_result = {
            "AP@0.5": ap_metrics["AP@0.5"],
            "AP@0.75": ap_metrics["AP@0.75"],
            "AP_small": ap_metrics["AP_small"],
            "AP_medium": ap_metrics["AP_medium"],
            "AP_large": ap_metrics["AP_large"],
            "Precision": prec,
            "Recall": rec,
            "PR_curve_path": str(pr_curve_path)
        }

        fold_rows.append({"fold": fi, **fold_result})

        m50 = ap_metrics["AP@0.5"]
        fold_map50.append(m50)

        if m50 > best_overall:
            best_overall = m50
            best_fold = fi
            best_fold_ckpt = fold_best

    # ----- Summary -----
    print("\n" + "="*70)
    print("[K-FOLD SUMMARY]")
    for i, v in enumerate(fold_map50):
        print(f"  Fold {i}: mAP@0.5 = {v:.4f}")

    avg = mean(fold_map50) if fold_map50 else 0.0
    std = pstdev(fold_map50) if len(fold_map50) > 1 else 0.0
    print(f"\n  Mean mAP@0.5 = {avg:.4f}")
    print(f"  Std  mAP@0.5 = {std:.4f}")
    print(f"\n  Best Fold = {best_fold} (1-based {best_fold+1})  Best mAP@0.5 = {best_overall:.4f}")
    print(f"  Best checkpoint = {best_fold_ckpt}")

    # best 모델 export
    if export_best_to is not None and best_fold_ckpt is not None:
        export_best_to = Path(export_best_to)
        export_best_to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_fold_ckpt, export_best_to)
        print(f"\n[EXPORT] Best fold model copied to: {export_best_to}")

    # 결과 csv 저장(간단 요약)
    summary_csv = out_root / f"{prefix_tag}_kfold_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "prefix","kfold","fold",
            "AP@0.5","AP@0.75","AP_small","AP_medium","AP_large",
            "Precision","Recall",
            "PR_curve_path"
        ])
        for r in fold_rows:
            w.writerow([
                prefix_tag, kfold, r["fold"],
                f"{r['AP@0.5']:.6f}", f"{r['AP@0.75']:.6f}",
               f"{r['AP_small']:.6f}", f"{r['AP_medium']:.6f}", f"{r['AP_large']:.6f}",
               f"{r['Precision']:.6f}", f"{r['Recall']:.6f}",
                r["PR_curve_path"]
            ])
    print(f"\n[OK] Summary saved: {summary_csv}")



# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True, help="brooklyn / gangnam / suwon / total")

    ap.add_argument("--model", type=str, default="yolo11x.pt",
                    help="Ultralytics pretrained checkpoint (e.g., yolo11x.pt). If local path, provide path.")
    ap.add_argument("--kfold", type=int, default=5, help="K for K-fold cross validation (default=5)")
    ap.add_argument("--seed", type=int, default=42, help="random seed for fold split")

    ap.add_argument("--max-epochs", type=int, default=9999,
                    help="최대 epoch 수 (아주 크게 두고, early stopping으로 멈춤)")
    ap.add_argument("--batch", type=int, default=8, help="batch size")
    ap.add_argument("--imgsz", type=int, default=1280, help="이미지 입력 크기")
    ap.add_argument("--lr", type=float, default=1e-4, help="초기 learning rate")
    ap.add_argument("--patience", type=int, default=50, help="early stopping patience")
    ap.add_argument("--device", type=int, default=0, help="GPU index. default=0")

    ap.add_argument("--artifacts-dir", type=str, default="artifacts")
    ap.add_argument("--out-root", type=str, default="artifacts/yolo11x_kfold")
    ap.add_argument("--export-best-to", type=str, default=None,
                    help="If set, copy best fold best.pt to this path (e.g., artifacts/yolo/best_yolo11x_kfold.pt)")

    args = ap.parse_args()

    finetune_kfold(
        prefix=args.prefix,
        artifacts_dir=args.artifacts_dir,
        out_root=args.out_root,
        model_ckpt=args.model,
        kfold=args.kfold,
        max_epochs=args.max_epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr=args.lr,
        patience=args.patience,
        device=args.device,
        seed=args.seed,
        export_best_to=args.export_best_to
    )
