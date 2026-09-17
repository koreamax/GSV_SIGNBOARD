import os
import torch
import torchvision
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
from tqdm import tqdm

from compute_map import compute_ap_map


@torch.no_grad()
def evaluate_map(model, loader, device):
    model.eval()
    preds, gts = [], []

    for imgs, targets in loader:
        imgs = [i.to(device) for i in imgs]
        outputs = model(imgs)

        for pred, tgt in zip(outputs, targets):
            file_name = tgt["image_id"].item()

            boxes = pred["boxes"].cpu().numpy()
            scores = pred["scores"].cpu().numpy()
            for (x1, y1, x2, y2), s in zip(boxes, scores):
                preds.append({
                    "filename": str(file_name),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "conf": s
                })

            for (x1, y1, x2, y2) in tgt["boxes"].cpu().numpy():
                gts.append({
                    "filename": str(file_name),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2
                })

    TP, FP, FN, precision, recall, f1, ap = compute_ap_map(preds, gts)
    return ap


class CocoDet(CocoDetection):
    """
    COCO Detection wrapper.
    ConcatDataset으로 여러 데이터 합칠 때 image_id 충돌 방지 위해 id_offset 지원.
    """
    def __init__(self, img_root, ann_file, id_offset=0):
        super().__init__(img_root, ann_file)
        self.id_offset = id_offset

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)

        boxes, labels = [], []
        for t in target:
            x, y, w, h = t["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(t["category_id"])

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            # ✅ offset을 더해서 unique id로 만듦
            "image_id": torch.tensor([idx + self.id_offset])
        }
        return F.to_tensor(img), target


def collate_fn(batch):
    return tuple(zip(*batch))


def get_model(num_classes=2):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = \
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model


def train_one_epoch(model, loader, optimizer, device, scaler):
    model.train()
    total_loss = 0.0

    for imgs, targets in tqdm(loader, desc="Training"):
        imgs = [i.to(device) for i in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "CUDA 사용 불가: PyTorch CUDA 버전 설치 확인"
    print(f"Using device: {device} / {torch.cuda.get_device_name(0)}")

    # ==============================
    # 1) archive COCO 경로
    # ==============================
    archive_base = os.path.join("artifacts", "archive")

    archive_train_images = os.path.join(archive_base, "train", "images")
    archive_train_anns   = os.path.join(archive_base, "train", "annotations.json")

    archive_valid_images = os.path.join(archive_base, "valid", "images")
    archive_valid_anns   = os.path.join(archive_base, "valid", "annotations.json")

    archive_train_ds = CocoDet(archive_train_images, archive_train_anns, id_offset=0)
    archive_val_ds   = CocoDet(archive_valid_images, archive_valid_anns, id_offset=0)

    # ==============================
    # 2) yolo_ft_total COCO 경로 추가
    # ==============================
    yolo_base = os.path.join("artifacts", "yolo_ft_total")

    yolo_train_images = os.path.join(yolo_base, "images", "train")
    yolo_train_anns   = os.path.join(yolo_base, "annotations", "instances_train.json")

    yolo_valid_images = os.path.join(yolo_base, "images", "val")
    yolo_valid_anns   = os.path.join(yolo_base, "annotations", "instances_val.json")

    # ✅ image_id 충돌 막으려고 offset을 archive 길이만큼 줌
    offset_train = len(archive_train_ds)
    offset_val   = len(archive_val_ds)

    yolo_train_ds = CocoDet(yolo_train_images, yolo_train_anns, id_offset=offset_train)
    yolo_val_ds   = CocoDet(yolo_valid_images, yolo_valid_anns, id_offset=offset_val)

    # ==============================
    # 3) 두 데이터셋 합치기
    # ==============================
    train_ds = ConcatDataset([archive_train_ds, yolo_train_ds])
    val_ds   = ConcatDataset([archive_val_ds, yolo_val_ds])

    print(f"[DATA] archive train={len(archive_train_ds)}, yolo_ft_total train={len(yolo_train_ds)}")
    print(f"[DATA] archive val  ={len(archive_val_ds)}, yolo_ft_total val  ={len(yolo_val_ds)}")
    print(f"[DATA] TOTAL train={len(train_ds)}, val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=2, shuffle=True, num_workers=0,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=2, shuffle=False, num_workers=0,
        collate_fn=collate_fn
    )

    model = get_model(num_classes=2)
    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.005, momentum=0.9, weight_decay=0.0005
    )

    scaler = torch.cuda.amp.GradScaler()

    epochs = 50

    save_dir = os.path.join("artifacts", "frcnn")
    os.makedirs(save_dir, exist_ok=True)
    old_ckpt_path = os.path.join(save_dir, "best_frcnn.pth")

    old_ap = 0.0
    if os.path.exists(old_ckpt_path):
        print(f">>> Found old checkpoint: {old_ckpt_path}")
        model.load_state_dict(torch.load(old_ckpt_path, map_location=device))
        old_ap = evaluate_map(model, val_loader, device)
        print(f">>> Old best val_mAP@0.5 = {old_ap:.4f}")
    else:
        print(">>> No old checkpoint. Training from pretrained COCO weights.")

    best_ap = old_ap
    best_state = None

    for epoch in range(epochs):
        loss = train_one_epoch(model, train_loader, optimizer, device, scaler)
        val_ap = evaluate_map(model, val_loader, device)

        print(f"[Epoch {epoch+1}] loss={loss:.4f}, val_mAP@0.5={val_ap:.4f}")

        if val_ap > best_ap:
            best_ap = val_ap
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            print(">>> NEW BEST FOUND in this run!")

    if best_state is not None and best_ap > old_ap:
        torch.save(best_state, old_ckpt_path)
        print(f">>> BEST UPDATED! Saved new best to {old_ckpt_path} (mAP={best_ap:.4f})")
    else:
        print(f">>> Old best kept. (old={old_ap:.4f}, new_best={best_ap:.4f})")


if __name__ == "__main__":
    main()
