"""
EfficientDet-D0 5-Fold training — SAME folds as YOLO26s/YOLO11x (reads YOLO-format labels).
- folds: artifacts/yolo11x_kfold/total_fold{i}/dataset/{images,labels}/{train,val}
- letterbox 512; effdet boxes yxyx, class label=1; best AP@0.5 per fold -> 5-fold average.
"""
import csv
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from PIL import Image
from effdet import create_model, DetBenchPredict

from eval_map_only import letterbox_pil, iou_xyxy

IMG_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, PATIENCE, BATCH, LR = 400, 100, 8, 5e-4  # train to full convergence (best stops updating)
FOLDS = 5
FOLD_ROOT = os.environ.get("FOLD_ROOT", "artifacts/yolo11x_kfold") + "/total_fold{i}/dataset"  # D33
OUT = Path(os.environ.get("DET_PROJECT", "artifacts/effdet_kfold")); OUT.mkdir(parents=True, exist_ok=True)


class YoloEffDet(Dataset):
    """Reads YOLO-format labels (cls cx cy w h, normalized) for a fold split."""
    def __init__(self, img_dir, lbl_dir, img_size=512):
        self.img_dir, self.lbl_dir = Path(img_dir), Path(lbl_dir)
        self.imgs = sorted([p for p in self.img_dir.iterdir()
                            if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
        self.img_size = img_size

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
                boxes.append([x1, y1, x2, y2])
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
        img_lb, scale, pl, pt = letterbox_pil(img, self.img_size)
        if boxes.numel():
            boxes *= scale
            boxes[:, [0, 2]] += pl; boxes[:, [1, 3]] += pt
            boxes[:, 0::2] = boxes[:, 0::2].clamp(0, self.img_size)
            boxes[:, 1::2] = boxes[:, 1::2].clamp(0, self.img_size)
        yxyx = boxes[:, [1, 0, 3, 2]] if boxes.numel() else torch.zeros((0, 4))
        return F.to_tensor(img_lb), {"yxyx": yxyx, "xyxy": boxes}


def collate_train(batch):
    imgs = torch.stack([b[0] for b in batch]).float()
    bbox = [b[1]["yxyx"].float() for b in batch]
    cls = [torch.ones((b[1]["yxyx"].shape[0],), dtype=torch.float32) for b in batch]
    target = {"bbox": bbox, "cls": cls,
              "img_size": torch.tensor([[IMG_SIZE, IMG_SIZE]] * len(imgs)).float(),
              "img_scale": torch.ones((len(imgs), 1)).float()}
    return imgs, target


def collate_eval(batch):
    return torch.stack([b[0] for b in batch]).float(), [b[1] for b in batch]


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
def evaluate(net, loader):
    net.eval()
    pb = DetBenchPredict(net).to(DEVICE).eval()
    preds, gts = [], []
    for images, targets in loader:
        outs = pb(images.to(DEVICE))
        for out, tgt in zip(outs, targets):
            preds.append([(d[0], d[1], d[2], d[3], d[4]) for d in out.detach().cpu().tolist()])
            gts.append([tuple(b) for b in tgt["xyxy"].tolist()])
    return compute_ap50(preds, gts)


def train_fold(i):
    root = FOLD_ROOT.format(i=i)
    tr = DataLoader(YoloEffDet(f"{root}/images/train", f"{root}/labels/train", IMG_SIZE),
                    batch_size=BATCH, shuffle=True, num_workers=0, collate_fn=collate_train)
    va = DataLoader(YoloEffDet(f"{root}/images/val", f"{root}/labels/val", IMG_SIZE),
                    batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_eval)
    bench = create_model("tf_efficientdet_d0", bench_task="train", num_classes=1,
                         pretrained=True, bench_labeler=True,
                         image_size=(IMG_SIZE, IMG_SIZE)).to(DEVICE)
    net = bench.model
    opt = torch.optim.Adam(bench.parameters(), lr=LR, weight_decay=5e-5)
    scaler = torch.cuda.amp.GradScaler()
    best_ap, best_ep, no_imp = -1.0, 0, 0
    ckpt = OUT / f"best_effdet_fold{i}.pth"
    for ep in range(1, EPOCHS + 1):
        bench.train(); net.train()
        for images, target in tr:
            images = images.to(DEVICE)
            target = {k: ([t.to(DEVICE) for t in v] if isinstance(v, list) else v.to(DEVICE))
                      for k, v in target.items()}
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                loss = bench(images, target)["loss"]
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        ap = evaluate(net, va)
        if ap > best_ap + 1e-6:
            best_ap, best_ep, no_imp = ap, ep, 0
            torch.save(net.state_dict(), ckpt)
        else:
            no_imp += 1
        if no_imp >= PATIENCE:
            break
    print(f"[EffDet-5fold] fold{i} best AP@0.5 = {best_ap:.4f} (ep{best_ep})", flush=True)
    return best_ap


def main():
    aps = []
    for i in range(FOLDS):
        aps.append(train_fold(i))
    avg = sum(aps) / len(aps)
    with open(OUT / "kfold_summary_ap50.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["fold", "best_AP50"])
        for i, a in enumerate(aps):
            w.writerow([i, f"{a:.4f}"])
        w.writerow(["AVG", f"{avg:.4f}"])
    print(f"[EffDet-5fold] DONE per-fold={[round(a,4) for a in aps]} AVG AP@0.5={avg:.4f}", flush=True)


if __name__ == "__main__":
    main()
