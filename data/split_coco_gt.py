import json
from pathlib import Path
import copy

# =========================
# 분할 파라미터 (중요)
# =========================
IMG_SIZE = 768

# 이보다 큰 bbox는 분할 대상
SPLIT_W = IMG_SIZE * 0.6   # 예: 460px
SPLIT_H = IMG_SIZE * 0.6

# 분할 최소 크기 (너무 작은 조각 제거)
MIN_W = IMG_SIZE * 0.15    # 예: 115px
MIN_H = IMG_SIZE * 0.15


def split_bbox_stripe(x, y, w, h, max_parts=4):
    """
    큰 bbox를 '띠(stripe)' 형태로 적당히만 분할
    - 긴 방향으로만 2~3개 분할
    - 최대 조각 수 제한
    """
    boxes = []

    # 어떤 방향이 더 긴지 판단
    if w >= h:
        # 가로가 더 길면: 좌우로만 분할
        parts = 2 if w < 2 * SPLIT_W else 3
        parts = min(parts, max_parts)
        step = w / parts
        for i in range(parts):
            sx = x + i * step
            sw = step if i < parts - 1 else (x + w - sx)
            if sw >= MIN_W and h >= MIN_H:
                boxes.append([float(sx), float(y), float(sw), float(h)])
    else:
        # 세로가 더 길면: 상하로만 분할
        parts = 2 if h < 2 * SPLIT_H else 3
        parts = min(parts, max_parts)
        step = h / parts
        for i in range(parts):
            sy = y + i * step
            sh = step if i < parts - 1 else (y + h - sy)
            if w >= MIN_W and sh >= MIN_H:
                boxes.append([float(x), float(sy), float(w), float(sh)])

    # 너무 작아서 하나도 안 나오면 원본 유지
    if len(boxes) == 0:
        boxes = [[float(x), float(y), float(w), float(h)]]

    return boxes


def split_coco_annotations(coco):
    new_coco = {
        "images": coco["images"],
        "categories": coco["categories"],
        "annotations": []
    }

    ann_id = 1

    for ann in coco["annotations"]:
        x, y, w, h = ann["bbox"]

        # 분할 대상인지 판단
        if w > SPLIT_W or h > SPLIT_H:
            sub_boxes = split_bbox_stripe(x, y, w, h, max_parts=4)

            for sb in sub_boxes:
                new_ann = copy.deepcopy(ann)
                new_ann["id"] = ann_id
                new_ann["bbox"] = sb
                new_ann["area"] = sb[2] * sb[3]
                new_coco["annotations"].append(new_ann)
                ann_id += 1
        else:
            new_ann = copy.deepcopy(ann)
            new_ann["id"] = ann_id
            new_coco["annotations"].append(new_ann)
            ann_id += 1

    return new_coco


if __name__ == "__main__":
    # =========================
    # 경로 설정
    # =========================
    BASE = Path("artifacts/yolo_ft_total/annotations")

    IN_TRAIN = BASE / "instances_train.json"
    IN_VAL   = BASE / "instances_val.json"

    OUT_TRAIN = BASE / "instances_train_split.json"
    OUT_VAL   = BASE / "instances_val_split.json"

    print("[INFO] loading coco annotations...")
    coco_train = json.loads(IN_TRAIN.read_text(encoding="utf-8"))
    coco_val   = json.loads(IN_VAL.read_text(encoding="utf-8"))

    print("[INFO] splitting train annotations...")
    new_train = split_coco_annotations(coco_train)

    print("[INFO] splitting val annotations...")
    new_val = split_coco_annotations(coco_val)

    OUT_TRAIN.write_text(json.dumps(new_train, indent=2), encoding="utf-8")
    OUT_VAL.write_text(json.dumps(new_val, indent=2), encoding="utf-8")

    print("[OK] saved:")
    print(" -", OUT_TRAIN)
    print(" -", OUT_VAL)
    print(f"[STATS] train anns: {len(coco_train['annotations'])} → {len(new_train['annotations'])}")
    print(f"[STATS] val   anns: {len(coco_val['annotations'])} → {len(new_val['annotations'])}")
