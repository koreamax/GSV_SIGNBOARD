"""
Faster R-CNN (ResNet50-FPN) 5-Fold training to full convergence — same folds as YOLO26x/EffDet.
- folds: artifacts/yolo11x_kfold/total_fold{i}/dataset/{images,labels}/{train,val} (YOLO labels)
- torchvision FRCNN (COCO pretrained), SGD lr 0.005 (matches original 3.2 setup)
- patience 100 (train until best AP@0.5 stops updating); same AP@0.5 metric as EffDet.
"""
import csv
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image

from eval_map_only import iou_xyxy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, PATIENCE, BATCH, LR = 400, 100, 2, 0.005
FOLDS = 5
FOLD_ROOT = os.environ.get("FOLD_ROOT", "artifacts/yolo11x_kfold") + "/total_fold{i}/dataset"  # D33
OUT = Path(os.environ.get("DET_PROJECT", "artifacts/frcnn_kfold_v2")); OUT.mkdir(parents=True, exist_ok=True)


class YoloFRCNN(Dataset):
    def __init__(self, img_dir, lbl_dir):
        self.img_dir, self.lbl_dir = Path(img_dir), Path(lbl_dir)
        self.imgs = sorted([p for p in self.img_dir.iterdir()
                            if p.suffix.lower() in (".jpg", ".jpeg", ".png")])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        ip = self.imgs[idx]
        img = Image.open(ip).convert("RGB")
        W, H = img.size
        lp = self.lbl_dir / (ip.stem + ".txt")
        boxes = []
        if lp.exists():
            for line in lp.read_text().strip().splitlines():
                if not line.strip():
                    continue
                _, cx, cy, w, h = (float(v) for v in line.split()[:5])
                x1 = (cx - w / 2) * W; y1 = (cy - h / 2) * H
                x2 = (cx + w / 2) * W; y2 = (cy + h / 2) * H
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.ones((len(boxes),), dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        return F.to_tensor(img), {"boxes": boxes, "labels": labels}


def collate(batch):
    return [b[0] for b in batch], [b[1] for b in batch]


def get_frcnn(num_classes=2):
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
    return model


def compute_ap50(preds_by_img, gts_by_img, score_thr=0.05):
    npos = sum(len(g) for g in gts_by_img)
    used = [[False] * len(g) for g in gts_by_img]
    allp = [(p[4], i, p[:4]) for i, preds in enumerate(preds_by_img) for p in preds if p[4] >= score_thr]
    allp.sort(key=lambda x: -x[0])
    tp, fp = [], []
    for _, i, box in allp:
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gts_by_img[i]):
            if used[i][j]:
                continue
            iou = iou_xyxy(box, g)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= 0.5 and best_j >= 0:
            used[i][best_j] = True; tp.append(1); fp.append(0)
        else:
            tp.append(0); fp.append(1)
    if not allp:
        return 0.0
    tp, fp = np.cumsum(tp), np.cumsum(fp)
    rec = tp / max(npos, 1); prec = tp / np.maximum(tp + fp, 1e-9)
    mrec = np.concatenate(([0], rec, [1])); mpre = np.concatenate(([0], prec, [0]))
    for k in range(len(mpre) - 1, 0, -1):
        mpre[k - 1] = max(mpre[k - 1], mpre[k])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, gts = [], []
    for images, targets in loader:
        outs = model([im.to(DEVICE) for im in images])
        for out, tgt in zip(outs, targets):
            b = out["boxes"].cpu().tolist(); s = out["scores"].cpu().tolist()
            preds.append([(bb[0], bb[1], bb[2], bb[3], sc) for bb, sc in zip(b, s)])
            gts.append([tuple(x) for x in tgt["boxes"].tolist()])
    return compute_ap50(preds, gts)


def train_fold(i):
    root = FOLD_ROOT.format(i=i)
    tr = DataLoader(YoloFRCNN(f"{root}/images/train", f"{root}/labels/train"),
                    batch_size=BATCH, shuffle=True, num_workers=0, collate_fn=collate)
    va = DataLoader(YoloFRCNN(f"{root}/images/val", f"{root}/labels/val"),
                    batch_size=2, shuffle=False, num_workers=0, collate_fn=collate)
    model = get_frcnn(2).to(DEVICE)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=LR, momentum=0.9, weight_decay=5e-4)
    best_ap, best_ep, no_imp = -1.0, 0, 0
    ckpt = OUT / f"best_frcnn_fold{i}.pth"
    for ep in range(1, EPOCHS + 1):
        model.train()
        for images, targets in tr:
            images = [im.to(DEVICE) for im in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            if sum(len(t["boxes"]) for t in targets) == 0:
                continue
            loss = sum(model(images, targets).values())
            opt.zero_grad(); loss.backward(); opt.step()
        ap = evaluate(model, va)
        if ap > best_ap + 1e-6:
            best_ap, best_ep, no_imp = ap, ep, 0
            torch.save(model.state_dict(), ckpt)
        else:
            no_imp += 1
        if no_imp >= PATIENCE:
            break
    print(f"[FRCNN-5fold] fold{i} best AP@0.5 = {best_ap:.4f} (ep{best_ep})", flush=True)
    return best_ap


def main():
    aps = [train_fold(i) for i in range(FOLDS)]
    avg = sum(aps) / len(aps)
    with open(OUT / "kfold_summary_ap50.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["fold", "best_AP50"])
        for i, a in enumerate(aps):
            w.writerow([i, f"{a:.4f}"])
        w.writerow(["AVG", f"{avg:.4f}"])
    print(f"[FRCNN-5fold] DONE per-fold={[round(a,4) for a in aps]} AVG AP@0.5={avg:.4f}", flush=True)


if __name__ == "__main__":
    main()
