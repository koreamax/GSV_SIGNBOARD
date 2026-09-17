"""
EfficientDet-D0 text(word) detection — hold-out (signboard_v3 source split, 전체 데이터).
train_effdet_text_kfold.py 와 동일 구성 (512 letterbox, Adam 5e-4, batch 8, epochs 60,
patience 15, AMP). 출력: artifacts/effdet_text_holdout/best_effdet_text.pth
SMOKE=1 이면 64/32 장으로 1 에폭.
"""
import csv
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from PIL import Image
from effdet import create_model

from eval_map_only import letterbox_pil
from train_effdet_text_kfold import (collate_train, collate_eval, evaluate, DEVICE,
                                     EPOCHS, PATIENCE, BATCH, LR, IMG_SIZE)

BASE = Path("artifacts/signboard_text_holdout")
OUT = Path(os.environ.get("TEXT_OUT", "artifacts/effdet_text_holdout")); OUT.mkdir(parents=True, exist_ok=True)
SMOKE = os.environ.get("SMOKE") == "1"


class SplitEffDet(Dataset):
    """train_effdet_text_kfold.ListEffDet 과 동일한 전처리, 라벨 경로만 split 폴더."""
    def __init__(self, split, limit=None, img_size=IMG_SIZE):
        self.imgs = sorted((BASE / "images" / split).glob("*.jpg"))[:limit]
        self.lbl = BASE / "labels" / split; self.img_size = img_size

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
                boxes.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
        img_lb, scale, pl, pt = letterbox_pil(img, self.img_size)
        if boxes.numel():
            boxes *= scale; boxes[:, [0, 2]] += pl; boxes[:, [1, 3]] += pt
            boxes[:, 0::2] = boxes[:, 0::2].clamp(0, self.img_size)
            boxes[:, 1::2] = boxes[:, 1::2].clamp(0, self.img_size)
        yxyx = boxes[:, [1, 0, 3, 2]] if boxes.numel() else torch.zeros((0, 4))
        return F.to_tensor(img_lb), {"yxyx": yxyx, "xyxy": boxes}


def main():
    lim_tr, lim_va, epochs = (64, 32, 1) if SMOKE else (None, None, EPOCHS)
    tr = DataLoader(SplitEffDet("train", lim_tr), batch_size=BATCH, shuffle=True,
                    num_workers=4, collate_fn=collate_train)
    va = DataLoader(SplitEffDet("val", lim_va), batch_size=8, shuffle=False,
                    num_workers=4, collate_fn=collate_eval)
    bench = create_model("tf_efficientdet_d0", bench_task="train", num_classes=1,
                         pretrained=True, bench_labeler=True, image_size=(IMG_SIZE, IMG_SIZE)).to(DEVICE)
    net = bench.model
    opt = torch.optim.Adam(bench.parameters(), lr=LR, weight_decay=5e-5)
    scaler = torch.cuda.amp.GradScaler()
    best_ap, best_ep, no_imp = -1.0, 0, 0
    log = open(OUT / "val_log.csv", "w", newline=""); w = csv.writer(log); w.writerow(["epoch", "val_AP50"])
    for ep in range(1, epochs + 1):
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
        w.writerow([ep, f"{ap:.4f}"]); log.flush()
        print(f"[effdet_text_holdout] ep{ep} val AP@0.5={ap:.4f}", flush=True)
        if ap > best_ap + 1e-6:
            best_ap, best_ep, no_imp = ap, ep, 0
            torch.save(net.state_dict(), OUT / "best_effdet_text.pth")
        else:
            no_imp += 1
        if no_imp >= PATIENCE:
            break
    log.close()
    (OUT / "summary.csv").write_text(f"best_val_AP50,best_epoch\n{best_ap:.4f},{best_ep}\n")
    print(f"[effdet_text_holdout] DONE best val AP@0.5={best_ap:.4f} (ep{best_ep})", flush=True)


if __name__ == "__main__":
    main()
