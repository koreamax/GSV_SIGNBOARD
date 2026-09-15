"""
Faster R-CNN text(word) detection — hold-out (signboard_v3 source split, 전체 데이터).
하이퍼파라미터·모델 구성은 train_frcnn_text_kfold.py 와 동일 (min_size 640 / max 1066,
SGD 0.005, batch 4, epochs 60, patience 15). val AP@0.5 로 조기종료, best 를 저장.
증강 없음(원 스크립트와 동일). 출력: artifacts/frcnn_text_holdout/best_frcnn_text.pth
SMOKE=1 이면 64/32 장으로 1 에폭.
"""
import csv
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from PIL import Image

from train_frcnn_text_kfold import get_frcnn, evaluate, collate, DEVICE, EPOCHS, PATIENCE, BATCH, LR

BASE = Path("artifacts/signboard_text_holdout")
OUT = Path(os.environ.get("TEXT_OUT", "artifacts/frcnn_text_holdout")); OUT.mkdir(parents=True, exist_ok=True)
SMOKE = os.environ.get("SMOKE") == "1"


class SplitFRCNN(Dataset):
    """train_frcnn_text_kfold.ListFRCNN 과 동일한 로딩, 라벨 경로만 split 폴더."""
    def __init__(self, split, limit=None):
        self.imgs = sorted((BASE / "images" / split).glob("*.jpg"))[:limit]
        self.lbl = BASE / "labels" / split

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        ip = self.imgs[idx]
        img = Image.open(ip).convert("RGB"); W, H = img.size
        boxes = []
        lp = self.lbl / f"{ip.stem}.txt"
        if lp.exists():
            for line in lp.read_text().strip().splitlines():
                if not line.strip():
                    continue
                _, cx, cy, w, h = (float(v) for v in line.split()[:5])
                x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H
                x2, y2 = (cx + w / 2) * W, (cy + h / 2) * H
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.ones((len(boxes),), dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        return F.to_tensor(img), {"boxes": boxes, "labels": labels}


def main():
    lim_tr, lim_va, epochs = (64, 32, 1) if SMOKE else (None, None, EPOCHS)
    tr = DataLoader(SplitFRCNN("train", lim_tr), batch_size=BATCH, shuffle=True,
                    num_workers=4, collate_fn=collate)
    va = DataLoader(SplitFRCNN("val", lim_va), batch_size=4, shuffle=False,
                    num_workers=4, collate_fn=collate)
    model = get_frcnn().to(DEVICE)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=LR, momentum=0.9, weight_decay=5e-4)
    best_ap, best_ep, no_imp = -1.0, 0, 0
    log = open(OUT / "val_log.csv", "w", newline=""); w = csv.writer(log); w.writerow(["epoch", "val_AP50"])
    for ep in range(1, epochs + 1):
        model.train()
        for images, targets in tr:
            images = [im.to(DEVICE) for im in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            if sum(len(t["boxes"]) for t in targets) == 0:
                continue
            loss = sum(model(images, targets).values())
            opt.zero_grad(); loss.backward(); opt.step()
        ap = evaluate(model, va)
        w.writerow([ep, f"{ap:.4f}"]); log.flush()
        print(f"[frcnn_text_holdout] ep{ep} val AP@0.5={ap:.4f}", flush=True)
        if ap > best_ap + 1e-6:
            best_ap, best_ep, no_imp = ap, ep, 0
            torch.save(model.state_dict(), OUT / "best_frcnn_text.pth")
        else:
            no_imp += 1
        if no_imp >= PATIENCE:
            break
    log.close()
    (OUT / "summary.csv").write_text(f"best_val_AP50,best_epoch\n{best_ap:.4f},{best_ep}\n")
    print(f"[frcnn_text_holdout] DONE best val AP@0.5={best_ap:.4f} (ep{best_ep})", flush=True)


if __name__ == "__main__":
    main()
