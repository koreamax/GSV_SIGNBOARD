import os
import csv
import random
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
from tqdm import tqdm

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# ============================================================
# Dataset
# ============================================================
class CocoDet(CocoDetection):
    """
    COCO Detection wrapper
    - bbox: [x1,y1,x2,y2]
    - image_id: COCO의 실제 image_id를 유지 (매우 중요!)
    """
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

        # COCO의 image_id를 유지해야 COCOeval이 정확히 매칭됨
        coco_img_id = int(self.ids[idx])

        target_out = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([coco_img_id], dtype=torch.int64)
        }
        return F.to_tensor(img), target_out


def collate_fn(batch):
    return tuple(zip(*batch))


# ============================================================
# Model
# ============================================================
def get_model(num_classes=2):
    """
    Faster R-CNN (COCO pretrained)
    num_classes=2 (background + signboard)
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = \
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model


# ============================================================
# Train one epoch
# ============================================================
def train_one_epoch(model, loader, optimizer, device, scaler):
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


# ============================================================
# COCO subset builder
# ============================================================
def build_coco_subset(coco: COCO, image_ids: list[int]):
    image_ids = set(int(i) for i in image_ids)

    images = [img for img in coco.dataset.get("images", []) if int(img["id"]) in image_ids]
    ann_keep = [ann for ann in coco.dataset.get("annotations", []) if int(ann["image_id"]) in image_ids]

    subset = {
        "info": coco.dataset.get("info", {}),                # ✅ 추가
        "licenses": coco.dataset.get("licenses", []),        # ✅ 추가
        "images": images,
        "annotations": ann_keep,
        "categories": coco.dataset.get("categories", [])
    }

    coco_sub = COCO()
    coco_sub.dataset = subset
    coco_sub.createIndex()
    return coco_sub



# ============================================================
# Convert model outputs -> COCO detections
# ============================================================
@torch.no_grad()
def predict_coco_dets(model, loader, device, conf=0.001):
    """
    loader에서 (imgs, targets) 돌면서 COCO det list 생성
    det: {image_id, category_id, bbox[x,y,w,h], score}
    """
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
                    "category_id": 1,          # signboard를 1로 가정 (COCO cat id 맞춰야 함)
                    "bbox": [float(x1), float(y1), w, h],
                    "score": float(s)
                })
    return dets


# ============================================================
# COCOeval metrics + PR curve save
# ============================================================
def coco_eval_and_save_pr(coco_gt: COCO, coco_dt_list, pr_curve_path: Path):
    coco_gt.dataset.setdefault("info", {})       # ✅ 보험
    coco_gt.dataset.setdefault("licenses", [])   # ✅ 보험
    """
    COCOeval로:
      - AP@0.5, AP@0.75
      - AP_small, AP_medium, AP_large
    + PR curve (IoU=0.5, area=all) 저장
    """
    if len(coco_dt_list) == 0:
        # det 0개면 COCOeval이 깨질 수 있으니 빈 결과 처리
        metrics = {
            "AP@0.5": 0.0, "AP@0.75": 0.0,
            "AP_small": 0.0, "AP_medium": 0.0, "AP_large": 0.0
        }
        # 빈 PR curve도 저장
        pr_curve_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure()
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title("Precision–Recall Curve (IoU=0.5)")
        plt.grid(True)
        plt.savefig(pr_curve_path, dpi=200, bbox_inches="tight")
        plt.close()
        return metrics

    coco_dt = coco_gt.loadRes(coco_dt_list)

    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    # stats:
    # 1: AP@0.5, 2: AP@0.75, 3: AP_small, 4: AP_medium, 5: AP_large
    ap50 = float(ev.stats[1])
    ap75 = float(ev.stats[2])
    ap_s = float(ev.stats[3])
    ap_m = float(ev.stats[4])
    ap_l = float(ev.stats[5])

    # PR curve: precision[T, R, K, A, M]
    iou_thrs = ev.params.iouThrs
    t_idx = int(np.where(np.isclose(iou_thrs, 0.5))[0][0]) if np.any(np.isclose(iou_thrs, 0.5)) else 0
    precision = ev.eval["precision"]  # [T, R, K, A, M]
    recall_thrs = ev.params.recThrs

    # K=1, A=all(0), M=maxDets=100(2) 일반적
    pr = precision[t_idx, :, 0, 0, 2]
    valid = pr > -1
    pr_valid = pr[valid]
    rec_valid = recall_thrs[valid]

    pr_curve_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    if len(pr_valid) > 0:
        plt.plot(rec_valid, pr_valid)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve (IoU=0.5)")
    plt.grid(True)
    plt.savefig(pr_curve_path, dpi=200, bbox_inches="tight")
    plt.close()

    return {
        "AP@0.5": ap50,
        "AP@0.75": ap75,
        "AP_small": ap_s,
        "AP_medium": ap_m,
        "AP_large": ap_l
    }


# ============================================================
# Precision / Recall at a fixed conf (single-point)
# ============================================================
@torch.no_grad()
def precision_recall_at_conf(model, loader, device, conf=0.5, iou_thr=0.5):
    """
    conf 임계값 한 점에서 Precision / Recall 계산 (greedy IoU matching)
    """
    model.eval()

    TP = 0
    FP = 0
    FN = 0

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
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
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
# K-Fold split (COCO image ids)
# ============================================================
def make_kfold_image_ids(all_image_ids: list[int], k=5, seed=42):
    rng = random.Random(seed)
    ids = list(all_image_ids)
    rng.shuffle(ids)

    folds = [[] for _ in range(k)]
    for i, img_id in enumerate(ids):
        folds[i % k].append(img_id)

    splits = []
    for fi in range(k):
        val_ids = set(folds[fi])
        train_ids = [x for x in ids if x not in val_ids]
        splits.append({"train": train_ids, "val": list(val_ids)})
    return splits


# ============================================================
# Main: 5-fold CV + Early stopping + Metrics
# ============================================================
def main():
    # ------------------------------
    # Device
    # ------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "CUDA 사용 불가: PyTorch CUDA 버전 설치 확인"
    print(f"Using device: {device} / {torch.cuda.get_device_name(0)}")

    # ------------------------------
    # Data paths (너 코드 유지)
    # ------------------------------
    yolo_base = os.path.join("artifacts", "yolo_ft_total")

    train_images = os.path.join(yolo_base, "images", "train")
    train_anns   = os.path.join(yolo_base, "annotations", "instances_train.json")

    val_images = os.path.join(yolo_base, "images", "val")
    val_anns   = os.path.join(yolo_base, "annotations", "instances_val.json")

    # COCO 객체 로드
    coco_train = COCO(train_anns)
    coco_val   = COCO(val_anns)

    # 전체 image id 목록 (train/val 합쳐서 k-fold 하고 싶으면 합쳐도 됨)
    # 여기서는 "train split만" k-fold로 돌리고, val split은 사용 안 하는 방식이 정석이긴 한데,
    # 너는 이미 train/val로 나뉜 COCO가 있어서, 아래 2가지 중 선택해야 함.
    #
    # (A) train_anns만 가지고 5-fold (추천)
    # (B) train+val 합쳐서 5-fold (더 엄밀하지만 json 합치기 필요)
    #
    # 여기서는 (A)로 간다.
    all_ids = list(map(int, coco_train.getImgIds()))
    print(f"[DATA] COCO-train images = {len(all_ids)} (5-fold CV on train set)")

    # Dataset은 train json 기준 하나만 만들고, fold별로 image_id만 골라쓴다.
    ds = CocoDet(train_images, train_anns)

    # image_id -> dataset index 매핑
    id_to_idx = {int(ds.ids[i]): i for i in range(len(ds))}

    # ------------------------------
    # Hyperparams
    # ------------------------------
    kfold = 5
    seed = 42

    max_epochs = 300
    patience = 20

    batch_size = 2
    num_workers = 0  # 윈도우 안전

    out_root = Path("artifacts") / "frcnn_kfold"
    out_root.mkdir(parents=True, exist_ok=True)

    splits = make_kfold_image_ids(all_ids, k=kfold, seed=seed)

    fold_rows = []

    for fi, sp in enumerate(splits):
        print("\n" + "=" * 80)
        print(f"[FOLD {fi+1}/{kfold}] train_ids={len(sp['train'])} val_ids={len(sp['val'])}")

        train_idx = [id_to_idx[i] for i in sp["train"] if i in id_to_idx]
        val_idx   = [id_to_idx[i] for i in sp["val"]   if i in id_to_idx]

        train_subset = torch.utils.data.Subset(ds, train_idx)
        val_subset   = torch.utils.data.Subset(ds, val_idx)

        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=collate_fn
        )

        # COCOeval용 GT subset
        coco_gt_fold = build_coco_subset(coco_train, sp["val"])

        # ------------------------------
        # Model / Optim
        # ------------------------------
        model = get_model(num_classes=2).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
        scaler = torch.cuda.amp.GradScaler()

        # ------------------------------
        # Early stopping by AP@0.5
        # ------------------------------
        best_ap50 = -1.0
        best_state = None
        best_epoch = None
        no_improve = 0

        fold_dir = out_root / f"fold{fi}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        pr_curve_path = fold_dir / "pr_curve_iou0.5.png"

        for epoch in range(max_epochs):
            loss = train_one_epoch(model, train_loader, optimizer, device, scaler)

            # --- Evaluate (COCOeval 기준) ---
            dets = predict_coco_dets(model, val_loader, device, conf=0.001)
            ap_metrics = coco_eval_and_save_pr(coco_gt_fold, dets, pr_curve_path=pr_curve_path)

            # single-point Precision/Recall
            prec, rec = precision_recall_at_conf(model, val_loader, device, conf=0.5, iou_thr=0.5)

            ap50 = ap_metrics["AP@0.5"]
            print(f"[FOLD {fi}][Epoch {epoch+1}] loss={loss:.4f}  "
                  f"AP@0.5={ap50:.4f}  AP@0.75={ap_metrics['AP@0.75']:.4f}  "
                  f"P@0.5={prec:.4f} R@0.5={rec:.4f}")

            if ap50 > best_ap50 + 1e-6:
                best_ap50 = ap50
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                best_epoch = epoch
                no_improve = 0
                print(f"  >>> NEW BEST AP@0.5={best_ap50:.4f} at epoch {epoch+1}")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"[EARLY STOP] no improvement for {patience} epochs. "
                          f"best AP@0.5={best_ap50:.4f} at epoch {best_epoch+1}")
                    break

        # ------------------------------
        # Save best checkpoint for this fold
        # ------------------------------
        ckpt_path = fold_dir / "best_frcnn.pth"
        if best_state is not None:
            torch.save(best_state, ckpt_path)

        # ------------------------------
        # Final eval for fold using best ckpt
        # ------------------------------
        if best_state is not None:
            model.load_state_dict(best_state)
        dets = predict_coco_dets(model, val_loader, device, conf=0.001)
        ap_metrics = coco_eval_and_save_pr(coco_gt_fold, dets, pr_curve_path=pr_curve_path)
        prec, rec = precision_recall_at_conf(model, val_loader, device, conf=0.5, iou_thr=0.5)

        row = {
            "fold": fi,
            "AP@0.5": ap_metrics["AP@0.5"],
            "AP@0.75": ap_metrics["AP@0.75"],
            "AP_small": ap_metrics["AP_small"],
            "AP_medium": ap_metrics["AP_medium"],
            "AP_large": ap_metrics["AP_large"],
            "Precision": prec,
            "Recall": rec,
            "PR_curve_path": str(pr_curve_path),
            "best_epoch": (best_epoch + 1) if best_epoch is not None else None,
            "best_ckpt": str(ckpt_path)
        }
        fold_rows.append(row)

        print(f"\n[FOLD {fi}] FINAL METRICS")
        print(f"  AP@0.5   = {row['AP@0.5']:.4f}")
        print(f"  AP@0.75  = {row['AP@0.75']:.4f}")
        print(f"  AP_small = {row['AP_small']:.4f}")
        print(f"  AP_med   = {row['AP_medium']:.4f}")
        print(f"  AP_large = {row['AP_large']:.4f}")
        print(f"  Precision(conf=0.5, IoU=0.5) = {row['Precision']:.4f}")
        print(f"  Recall   (conf=0.5, IoU=0.5) = {row['Recall']:.4f}")
        print(f"  PR curve saved: {row['PR_curve_path']}")
        print(f"  Best epoch: {row['best_epoch']}, ckpt: {row['best_ckpt']}")

    # ------------------------------
    # Summary CSV
    # ------------------------------
    summary_csv = out_root / "frcnn_kfold_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "fold",
            "AP@0.5","AP@0.75","AP_small","AP_medium","AP_large",
            "Precision","Recall",
            "best_epoch","best_ckpt","PR_curve_path"
        ])
        for r in fold_rows:
            w.writerow([
                r["fold"],
                f"{r['AP@0.5']:.6f}", f"{r['AP@0.75']:.6f}",
                f"{r['AP_small']:.6f}", f"{r['AP_medium']:.6f}", f"{r['AP_large']:.6f}",
                f"{r['Precision']:.6f}", f"{r['Recall']:.6f}",
                r["best_epoch"], r["best_ckpt"], r["PR_curve_path"]
            ])

    # 평균/표준편차(원하면 AP@0.5 기준만 요약)
    ap50s = [r["AP@0.5"] for r in fold_rows]
    print("\n" + "=" * 80)
    print("[K-FOLD SUMMARY] (Faster R-CNN)")
    for r in fold_rows:
        print(f"  Fold {r['fold']}: AP@0.5={r['AP@0.5']:.4f}, AP@0.75={r['AP@0.75']:.4f}")
    print(f"\n  Mean AP@0.5 = {mean(ap50s):.4f}")
    print(f"  Std  AP@0.5 = {pstdev(ap50s) if len(ap50s) > 1 else 0.0:.4f}")
    print(f"\n[OK] Saved: {summary_csv}")


if __name__ == "__main__":
    main()
