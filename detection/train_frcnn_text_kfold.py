"""
Faster R-CNN text(word) detection 5-Fold — reads signboard_text fold txt lists.
Same AP@0.5 metric as effdet_text. min_size reduced for speed on 4000 imgs/fold.
Output: artifacts/frcnn_text_kfold.
"""
import csv
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
EPOCHS, PATIENCE, BATCH, LR = 60, 15, 4, 0.005
MIN_SIZE, MAX_SIZE = 640, 1066
BASE = Path("artifacts/signboard_text")
OUT = Path("artifacts/frcnn_text_kfold"); OUT.mkdir(parents=True, exist_ok=True)


def read_list(txt):
    return [BASE / line.strip().lstrip("./") for line in open(BASE / txt) if line.strip()]


class ListFRCNN(Dataset):
    def __init__(self, img_paths):
        self.imgs = img_paths

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        ip = self.imgs[idx]
        img = Image.open(ip).convert("RGB"); W, H = img.size
        lp = BASE / "labels" / (ip.stem + ".txt")
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


def collate(b):
    return [x[0] for x in b], [x[1] for x in b]


def get_frcnn():
    m = fasterrcnn_resnet50_fpn(weights="DEFAULT", min_size=MIN_SIZE, max_size=MAX_SIZE)
    inf = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(inf, 2)
    return m


def compute_ap50(preds, gts, score_thr=0.05):
    npos = sum(len(g) for g in gts); used = [[False] * len(g) for g in gts]
    allp = [(p[4], i, p[:4]) for i, ps in enumerate(preds) for p in ps if p[4] >= score_thr]
    allp.sort(key=lambda x: -x[0]); tp, fp = [], []
    for _, i, box in allp:
        bi, bj = 0.0, -1
        for j, g in enumerate(gts[i]):
            if used[i][j]:
                continue
            v = iou_xyxy(box, g)
            if v > bi:
                bi, bj = v, j
        if bi >= 0.5 and bj >= 0:
            used[i][bj] = True; tp.append(1); fp.append(0)
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
    model.eval(); preds, gts = [], []
    for images, targets in loader:
        outs = model([im.to(DEVICE) for im in images])
        for out, tgt in zip(outs, targets):
            b = out["boxes"].cpu().tolist(); s = out["scores"].cpu().tolist()
            preds.append([(bb[0], bb[1], bb[2], bb[3], sc) for bb, sc in zip(b, s)])
            gts.append([tuple(x) for x in tgt["boxes"].tolist()])
    return compute_ap50(preds, gts)


def train_fold(i):
    tr = DataLoader(ListFRCNN(read_list(f"fold{i}_train.txt")), batch_size=BATCH, shuffle=True,
                    num_workers=4, collate_fn=collate)
    va = DataLoader(ListFRCNN(read_list(f"fold{i}_val.txt")), batch_size=4, shuffle=False,
                    num_workers=4, collate_fn=collate)
    model = get_frcnn().to(DEVICE)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=LR, momentum=0.9, weight_decay=5e-4)
    best_ap, best_ep, no_imp = -1.0, 0, 0
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
            torch.save(model.state_dict(), OUT / f"best_frcnn_text_fold{i}.pth")
        else:
            no_imp += 1
        if no_imp >= PATIENCE:
            break
    print(f"[frcnn_text] fold{i} best AP@0.5 = {best_ap:.4f} (ep{best_ep})", flush=True)
    return best_ap


def main():
    aps = [train_fold(i) for i in range(5)]
    avg = sum(aps) / len(aps)
    with open(OUT / "kfold_summary_ap50.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["fold", "best_AP50"])
        for i, a in enumerate(aps):
            w.writerow([i, f"{a:.4f}"])
        w.writerow(["AVG", f"{avg:.4f}"])
    print(f"[frcnn_text] DONE per-fold={[round(a,4) for a in aps]} AVG AP@0.5={avg:.4f}", flush=True)


if __name__ == "__main__":
    main()
