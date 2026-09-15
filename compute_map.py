#!/usr/bin/env python3
import csv
import argparse
from pathlib import Path
import numpy as np


# ==============================
# IoU
# ==============================
def iou(box1, box2):
    x1 = max(box1["x1"], box2["x1"])
    y1 = max(box1["y1"], box2["y1"])
    x2 = min(box1["x2"], box2["x2"])
    y2 = min(box1["y2"], box2["y2"])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1["x2"] - box1["x1"]) * max(0, box1["y2"] - box1["y1"])
    area2 = max(0, box2["x2"] - box2["x1"]) * max(0, box2["y2"] - box2["y1"])
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


# ==============================
# Pred CSV 로더
# ==============================
def load_pred_csv(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        fieldnames = [c.strip() for c in rdr.fieldnames]

        conf_keys = ["conf", "confS", "confidence", "score", "prob"]
        conf_key = None
        for ck in conf_keys:
            if ck in fieldnames:
                conf_key = ck
                break

        if conf_key is None:
            print("[WARN] confidence 컬럼 없음 → conf=1.0 고정")
            conf_key = None

        for r in rdr:
            r_norm = {k.strip(): v for k, v in r.items()}
            rows.append({
                "filename": r_norm["filename"],
                "w": float(r_norm.get("w", 0) or 0),
                "h": float(r_norm.get("h", 0) or 0),
                "x1": float(r_norm["x1"]),
                "y1": float(r_norm["y1"]),
                "x2": float(r_norm["x2"]),
                "y2": float(r_norm["y2"]),
                "conf": float(r_norm.get(conf_key, 1.0)) if conf_key else 1.0
            })
    return rows


# ==============================
# GT 로더 + pred 해상도에 맞게 스케일
# ==============================
def load_gt_csv_and_scale(gt_path: Path, pred_list):
    gt_rows = []
    pred_w_h_cache = {p["filename"]: (p["w"], p["h"]) for p in pred_list}

    with open(gt_path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            fname = r["filename"]
            gt_w = float(r.get("w", 0) or 0)
            gt_h = float(r.get("h", 0) or 0)

            if fname in pred_w_h_cache and gt_w > 0 and gt_h > 0:
                pred_w, pred_h = pred_w_h_cache[fname]
                scale_x = pred_w / gt_w
                scale_y = pred_h / gt_h
            else:
                scale_x = 1.0
                scale_y = 1.0

            gt_rows.append({
                "filename": fname,
                "x1": float(r["x1"]) * scale_x,
                "y1": float(r["y1"]) * scale_y,
                "x2": float(r["x2"]) * scale_x,
                "y2": float(r["y2"]) * scale_y,
            })
    return gt_rows


# ==============================
# (A) Legacy TP/FP/FN 계산 (CSV 순서 유지)
# ==============================
def legacy_stats_csv_order(pred_list, gt_list, iou_thr=0.5):
    gt_per_img = {}
    for gt in gt_list:
        gt_per_img.setdefault(gt["filename"], []).append(gt)

    tp_list, fp_list = [], []
    used_gt = {}

    for pred in pred_list:
        fname = pred["filename"]
        gts = gt_per_img.get(fname, [])

        best_iou = 0.0
        best_idx = -1

        for i, gt in enumerate(gts):
            key = f"{fname}-{i}"
            if used_gt.get(key, False):
                continue
            iou_val = iou(pred, gt)
            if iou_val > best_iou:
                best_iou = iou_val
                best_idx = i

        if best_iou >= iou_thr and best_idx >= 0:
            tp_list.append(1)
            fp_list.append(0)
            used_gt[f"{fname}-{best_idx}"] = True
        else:
            tp_list.append(0)
            fp_list.append(1)

    tp_c = np.cumsum(tp_list)
    fp_c = np.cumsum(fp_list)

    TP = int(tp_c[-1]) if len(tp_c) else 0
    FP = int(fp_c[-1]) if len(fp_c) else 0
    FN = len(gt_list) - TP

    precision = TP / (TP + FP + 1e-9)
    recall = TP / (TP + FN + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return TP, FP, FN, precision, recall, f1


# ==============================
# (B) AP 계산 (conf 정렬)
# ==============================
def ap_conf_sorted(pred_list, gt_list, iou_thr=0.5):
    gt_per_img = {}
    for gt in gt_list:
        gt_per_img.setdefault(gt["filename"], []).append(gt)

    pred_sorted = sorted(pred_list, key=lambda x: -x["conf"])

    tp_list, fp_list = [], []
    used_gt = {}

    for pred in pred_sorted:
        fname = pred["filename"]
        gts = gt_per_img.get(fname, [])

        best_iou = 0.0
        best_idx = -1
        for i, gt in enumerate(gts):
            key = f"{fname}-{i}"
            if used_gt.get(key, False):
                continue
            iou_val = iou(pred, gt)
            if iou_val > best_iou:
                best_iou = iou_val
                best_idx = i

        if best_iou >= iou_thr and best_idx >= 0:
            tp_list.append(1); fp_list.append(0)
            used_gt[f"{fname}-{best_idx}"] = True
        else:
            tp_list.append(0); fp_list.append(1)

    tp_c = np.cumsum(tp_list)
    fp_c = np.cumsum(fp_list)

    total_gt = len(gt_list)
    recall = tp_c / (total_gt + 1e-9)
    precision = tp_c / (tp_c + fp_c + 1e-9)

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return ap


def infer_region_from_gt_name(gt_path: Path):
    name = gt_path.stem
    if name.startswith("gt_"):
        return name[3:]
    if name.endswith("_gt"):
        return name[:-3]
    return name


# ==============================
# weights → 자동 pred csv 찾기
# ==============================
def find_pred_csv_from_weights(weights_path: Path):
    folder = weights_path.parent
    cands = list(folder.glob("*.csv"))

    # 가장 "의미 있어 보이는" 파일 우선순위
    priority = [
        "total", "results", "pred", "yolo_results", "efficientdet_results", "frcnn_results"
    ]

    def score(p: Path):
        lower = p.name.lower()
        for i, key in enumerate(priority):
            if key in lower:
                return i
        return len(priority) + 1

    if not cands:
        raise FileNotFoundError(f"[ERROR] {folder} 안에 pred csv가 없음")

    cands.sort(key=score)
    return cands[0]


def model_name_from_weights(weights_path: Path):
    n = weights_path.name.lower()
    if "yolo" in n:
        return "YOLO"
    if "efficientdet" in n or "effdet" in n:
        return "EfficientDet"
    if "frcnn" in n or "fasterrcnn" in n:
        return "Faster R-CNN"
    return weights_path.stem


def eval_once(pred_path: Path, gt_path: Path, weights_path: Path | None = None):
    region = infer_region_from_gt_name(gt_path)
    pred_list = load_pred_csv(pred_path)
    gt_list = load_gt_csv_and_scale(gt_path, pred_list)

    TP, FP, FN, precision, recall, f1 = legacy_stats_csv_order(pred_list, gt_list)
    apv = ap_conf_sorted(pred_list, gt_list)

    model_str = model_name_from_weights(weights_path) if weights_path else pred_path.parent.name

    print("\n====== mAP Evaluation Result ======")
    print(f"Model : {model_str}")
    print(f"Region (auto) : {region}")
    print(f"Pred CSV : {pred_path}")
    print(f"GT Count : {len(gt_list)}")
    print(f"PRED Count : {len(pred_list)}")
    print(f"TP : {TP}")
    print(f"FP : {FP}")
    print(f"FN : {FN}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"mAP@0.5   : {apv:.4f}   (conf-sorted AP)")
    print("===================================\n")

# -----------------------------
# mAP@0.5 (single-class) evaluator
# preds, gts: list of dicts
# pred dict: filename, x1,y1,x2,y2, conf(optional)
# gt dict: filename, x1,y1,x2,y2
# -----------------------------

def _iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def compute_ap_map(preds, gts, iou_thr=0.5):
    """
    Return:
      TP, FP, FN, precision, recall, f1, ap
    """
    # gts by filename
    gt_by_file = {}
    for g in gts:
        fn = g["filename"]
        gt_by_file.setdefault(fn, []).append([g["x1"], g["y1"], g["x2"], g["y2"]])

    # preds sorted by conf desc
    preds_sorted = sorted(preds, key=lambda x: x.get("conf", 1.0), reverse=True)

    tp_list = []
    fp_list = []

    # per-file matched flags
    matched = {fn: [False] * len(boxes) for fn, boxes in gt_by_file.items()}

    for p in preds_sorted:
        fn = p["filename"]
        pbox = [p["x1"], p["y1"], p["x2"], p["y2"]]

        if fn not in gt_by_file or len(gt_by_file[fn]) == 0:
            fp_list.append(1)
            tp_list.append(0)
            continue

        best_iou = 0.0
        best_j = -1
        for j, gbox in enumerate(gt_by_file[fn]):
            if matched[fn][j]:
                continue
            iou = _iou_xyxy(pbox, gbox)
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_iou >= iou_thr and best_j >= 0:
            matched[fn][best_j] = True
            tp_list.append(1)
            fp_list.append(0)
        else:
            tp_list.append(0)
            fp_list.append(1)

    TP = sum(tp_list)
    FP = sum(fp_list)
    GT = len(gts)
    FN = GT - TP

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / GT if GT > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # PR curve AP (trapezoid)
    precisions = []
    recalls = []
    cum_tp = 0
    cum_fp = 0

    for tpi, fpi in zip(tp_list, fp_list):
        cum_tp += tpi
        cum_fp += fpi
        p = cum_tp / (cum_tp + cum_fp) if (cum_tp + cum_fp) > 0 else 0.0
        r = cum_tp / GT if GT > 0 else 0.0
        precisions.append(p)
        recalls.append(r)

    ap = 0.0
    prev_r = 0.0
    for p, r in zip(precisions, recalls):
        ap += p * (r - prev_r)
        prev_r = r

    return TP, FP, FN, precision, recall, f1, ap

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="GT csv path")
    ap.add_argument("--pred", default=None, help="pred csv path (override)")
    ap.add_argument("--weights", default=None, help="best_*.pt / best_*.pth path")
    ap.add_argument("--models", default=None, choices=["yolo", "efficientdet", "frcnn", "all"],
                    help="auto run preset model weights")
    args = ap.parse_args()

    gt_path = Path(args.gt)

    # (1) --models all → 3개 모델 일괄 평가
    if args.models == "all":
        presets = [
            Path("artifacts/yolo/best_yolo.pt"),
            Path("artifacts/efficientdet/best_efficientdet.pth"),
            Path("artifacts/frcnn/best_frcnn.pth"),
        ]
        for w in presets:
            if not w.exists():
                print(f"[SKIP] weights not found: {w}")
                continue
            pred_path = find_pred_csv_from_weights(w)
            eval_once(pred_path, gt_path, w)
        return

    # (2) --models yolo/efficientdet/frcnn → 해당 weights로 자동 평가
    if args.models in ["yolo", "efficientdet", "frcnn"]:
        preset_w = {
            "yolo": Path("artifacts/yolo/best_yolo.pt"),
            "efficientdet": Path("artifacts/efficientdet/best_efficientdet.pth"),
            "frcnn": Path("artifacts/frcnn/best_frcnn.pth"),
        }[args.models]

        if not preset_w.exists():
            raise FileNotFoundError(f"[ERROR] preset weights 없음: {preset_w}")

        pred_path = Path(args.pred) if args.pred else find_pred_csv_from_weights(preset_w)
        eval_once(pred_path, gt_path, preset_w)
        return

    # (3) --weights 직접 지정한 경우
    if args.weights:
        w_path = Path(args.weights)
        if not w_path.exists():
            raise FileNotFoundError(f"[ERROR] weights not found: {w_path}")

        pred_path = Path(args.pred) if args.pred else find_pred_csv_from_weights(w_path)
        eval_once(pred_path, gt_path, w_path)
        return

    # (4) 그냥 pred + gt만 주는 기존 방식도 유지
    if not args.pred:
        raise ValueError("[ERROR] --pred 또는 --weights 또는 --models 중 하나는 필요함")

    pred_path = Path(args.pred)
    eval_once(pred_path, gt_path, None)


if __name__ == "__main__":
    main()
