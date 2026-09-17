"""
EfficientDet-D0 signboard detection training (single class).
- Data: artifacts/yolo_ft_total (COCO json), letterbox to 512.
- effdet conventions: gt boxes are [y0,x0,y1,x1] (yxyx); class label starts at 1.
- best checkpoint (highest val AP@0.5) saved to artifacts/efficientdet/best_effdet.pth
  (the path eval_map_only.py expects). Early stopping via patience.
"""
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from PIL import Image
from pycocotools.coco import COCO
from effdet import create_model, DetBenchPredict

from eval_map_only import letterbox_pil, iou_xyxy  # reuse, consistent with eval

DATA_ROOT = Path("artifacts/yolo_ft_total")
TRAIN_IMG, TRAIN_JSON = DATA_ROOT / "images/train", DATA_ROOT / "annotations/instances_train.json"
VAL_IMG, VAL_JSON = DATA_ROOT / "images/val", DATA_ROOT / "annotations/instances_val.json"
IMG_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path("artifacts/efficientdet"); OUT.mkdir(parents=True, exist_ok=True)
CKPT = OUT / "best_effdet.pth"

EPOCHS, PATIENCE, BATCH, LR = 300, 50, 8, 5e-4


class CocoEffDet(Dataset):
    def __init__(self, img_dir, ann_json, img_size=512):
        self.coco = COCO(str(ann_json)); self.img_dir = Path(img_dir)
        self.ids = list(self.coco.imgs.keys()); self.img_size = img_size

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]; info = self.coco.loadImgs(img_id)[0]
        img = Image.open(self.img_dir / info["file_name"]).convert("RGB")
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[img_id]))
        boxes = [[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                 for a in anns]
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
        img_lb, scale, pl, pt = letterbox_pil(img, self.img_size)
        if boxes.numel():
            boxes *= scale
            boxes[:, [0, 2]] += pl; boxes[:, [1, 3]] += pt
            boxes[:, 0::2] = boxes[:, 0::2].clamp(0, self.img_size)
            boxes[:, 1::2] = boxes[:, 1::2].clamp(0, self.img_size)
        yxyx = boxes[:, [1, 0, 3, 2]] if boxes.numel() else torch.zeros((0, 4))
        return F.to_tensor(img_lb), {"yxyx": yxyx, "xyxy": boxes, "img_id": img_id}


def collate_train(batch):
    imgs = torch.stack([b[0] for b in batch]).float()
    bbox = [b[1]["yxyx"].float() for b in batch]
    cls = [torch.ones((b[1]["yxyx"].shape[0],), dtype=torch.float32) for b in batch]  # label=1
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
    pred_bench = DetBenchPredict(net).to(DEVICE).eval()
    preds_by_img, gts_by_img = [], []
    for images, targets in loader:
        outs = pred_bench(images.to(DEVICE))
        for out, tgt in zip(outs, targets):
            preds_by_img.append([(d[0], d[1], d[2], d[3], d[4]) for d in out.detach().cpu().tolist()])
            gts_by_img.append([tuple(b) for b in tgt["xyxy"].tolist()])
    return compute_ap50(preds_by_img, gts_by_img)


def main():
    tr = DataLoader(CocoEffDet(TRAIN_IMG, TRAIN_JSON, IMG_SIZE), batch_size=BATCH, shuffle=True,
                    num_workers=0, collate_fn=collate_train)
    va = DataLoader(CocoEffDet(VAL_IMG, VAL_JSON, IMG_SIZE), batch_size=4, shuffle=False,
                    num_workers=0, collate_fn=collate_eval)

    bench = create_model("tf_efficientdet_d0", bench_task="train", num_classes=1,
                         pretrained=True, bench_labeler=True,
                         image_size=(IMG_SIZE, IMG_SIZE)).to(DEVICE)
    net = bench.model
    opt = torch.optim.Adam(bench.parameters(), lr=LR, weight_decay=5e-5)
    scaler = torch.cuda.amp.GradScaler()

    best_ap, best_ep, no_improve = -1.0, 0, 0
    for ep in range(1, EPOCHS + 1):
        bench.train(); net.train()
        tot = 0.0
        for images, target in tr:
            images = images.to(DEVICE)
            target = {k: ([t.to(DEVICE) for t in v] if isinstance(v, list) else v.to(DEVICE))
                      for k, v in target.items()}
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                out = bench(images, target)
                loss = out["loss"]
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tot += float(loss)
        ap = evaluate(net, va)
        flag = ""
        if ap > best_ap + 1e-6:
            best_ap, best_ep, no_improve = ap, ep, 0
            torch.save(net.state_dict(), CKPT)
            flag = " <- BEST saved"
        else:
            no_improve += 1
        print(f"[EffDet][Ep {ep}/{EPOCHS}] loss={tot/len(tr):.4f} valAP@0.5={ap:.4f} "
              f"best={best_ap:.4f}(ep{best_ep}){flag}", flush=True)
        if no_improve >= PATIENCE:
            print(f"[EffDet] early stop at ep{ep} (no improve {PATIENCE})", flush=True)
            break
    print(f"[EffDet] DONE best valAP@0.5={best_ap:.4f} (ep{best_ep}) -> {CKPT}", flush=True)


if __name__ == "__main__":
    main()
