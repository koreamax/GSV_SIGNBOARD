import json
from pathlib import Path

COCO = Path("artifacts/yolo_ft_total/annotations/instances_train.json")
IMG_DIR = Path("artifacts/yolo_ft_total/images/train")

data = json.loads(COCO.read_text(encoding="utf-8"))

imgs = {im["id"]: im for im in data["images"]}
cats = {c["id"]: c["name"] for c in data["categories"]}
print("Categories:", cats)

bad_imgid = 0
bad_catid = 0
bad_bbox = 0
missing_img = 0

for ann in data["annotations"]:
    img_id = ann["image_id"]
    cat_id = ann["category_id"]
    bbox = ann["bbox"]  # COCO: [x,y,w,h]

    if img_id not in imgs:
        bad_imgid += 1
        continue
    if cat_id not in cats:
        bad_catid += 1

    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        bad_bbox += 1
        continue

    im = imgs[img_id]
    W, H = im["width"], im["height"]
    # bbox가 이미지 밖으로 심하게 나가는지 체크
    if x < -1 or y < -1 or x + w > W + 1 or y + h > H + 1:
        bad_bbox += 1

# 이미지 파일 존재 체크
for img_id, im in imgs.items():
    fn = im["file_name"]
    if not (IMG_DIR / fn).exists():
        missing_img += 1

print("Bad image_id refs:", bad_imgid)
print("Bad category_id refs:", bad_catid)
print("Bad bbox:", bad_bbox)
print("Missing image files:", missing_img)
print("Total images:", len(imgs), "Total anns:", len(data["annotations"]))
