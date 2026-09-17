"""
EfficientDet-D0 text(word) detection 5-Fold — reads signboard_text fold txt lists.
Same effdet conventions (yxyx boxes, class 1, AP@0.5). Output: artifacts/effdet_text_kfold.
"""
import csv
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
EPOCHS, PATIENCE, BATCH, LR = 60, 15, 8, 5e-4
BASE = Path("artifacts/signboard_text")
OUT = Path("artifacts/effdet_text_kfold"); OUT.mkdir(parents=True, exist_ok=True)


def read_list(txt):
    return [BASE / line.strip().lstrip("./") for line in open(BASE / txt) if line.strip()]


class ListEffDet(Dataset):
    def __init__(self, img_paths, img_size=512):
        self.imgs = img_paths; self.img_size = img_size

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
                boxes.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
        img_lb, scale, pl, pt = letterbox_pil(img, self.img_size)
        if boxes.numel():
            boxes *= scale; boxes[:, [0, 2]] += pl; boxes[:, [1, 3]] += pt
            boxes[:, 0::2] = boxes[:, 0::2].clamp(0, self.img_size)
            boxes[:, 1::2] = boxes[:, 1::2].clamp(0, self.img_size)
        yxyx = boxes[:, [1, 0, 3, 2]] if boxes.numel() else torch.zeros((0, 4))
        return F.to_tensor(img_lb), {"yxyx": yxyx, "xyxy": boxes}


def collate_train(b):
    imgs = torch.stack([x[0] for x in b]).float()
    return imgs, {"bbox": [x[1]["yxyx"].float() for x in b],
                  "cls": [torch.ones((x[1]["yxyx"].shape[0],), dtype=torch.float32) for x in b],
                  "img_size": torch.tensor([[IMG_SIZE, IMG_SIZE]] * len(imgs)).float(),
                  "img_scale": torch.ones((len(imgs), 1)).float()}


def collate_eval(b):
    return torch.stack([x[0] for x in b]).float(), [x[1] for x in b]


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
def evaluate(net, loader):
    net.eval(); pb = DetBenchPredict(net).to(DEVICE).eval(); preds, gts = [], []
    for images, targets in loader:
        outs = pb(images.to(DEVICE))
        for out, tgt in zip(outs, targets):
            preds.append([(d[0], d[1], d[2], d[3], d[4]) for d in out.detach().cpu().tolist()])
            gts.append([tuple(b) for b in tgt["xyxy"].tolist()])
    return compute_ap50(preds, gts)


def train_fold(i):
    tr = DataLoader(ListEffDet(read_list(f"fold{i}_train.txt")), batch_size=BATCH, shuffle=True,
                    num_workers=4, collate_fn=collate_train)
    va = DataLoader(ListEffDet(read_list(f"fold{i}_val.txt")), batch_size=8, shuffle=False,
                    num_workers=4, collate_fn=collate_eval)
    bench = create_model("tf_efficientdet_d0", bench_task="train", num_classes=1,
                         pretrained=True, bench_labeler=True, image_size=(IMG_SIZE, IMG_SIZE)).to(DEVICE)
    net = bench.model
    opt = torch.optim.Adam(bench.parameters(), lr=LR, weight_decay=5e-5)
    scaler = torch.cuda.amp.GradScaler()
    best_ap, best_ep, no_imp = -1.0, 0, 0
    for ep in range(1, EPOCHS + 1):
        bench.train(); net.train()
        for images, target in tr:
            images = images.to(DEVICE)
            target = {k: ([t.to(DEVICE) for t in v] if isinstance(v, list) else v.to(DEVICE)) for k, v in target.items()}
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                loss = bench(images, target)["loss"]
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        ap = evaluate(net, va)
        if ap > best_ap + 1e-6:
            best_ap, best_ep, no_imp = ap, ep, 0
            torch.save(net.state_dict(), OUT / f"best_effdet_text_fold{i}.pth")
        else:
            no_imp += 1
        if no_imp >= PATIENCE:
            break
    print(f"[effdet_text] fold{i} best AP@0.5 = {best_ap:.4f} (ep{best_ep})", flush=True)
    return best_ap


def main():
    aps = [train_fold(i) for i in range(5)]
    avg = sum(aps) / len(aps)
    with open(OUT / "kfold_summary_ap50.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["fold", "best_AP50"])
        for i, a in enumerate(aps):
            w.writerow([i, f"{a:.4f}"])
        w.writerow(["AVG", f"{avg:.4f}"])
    print(f"[effdet_text] DONE per-fold={[round(a,4) for a in aps]} AVG AP@0.5={avg:.4f}", flush=True)


if __name__ == "__main__":
    main()
