import json
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent / "artifacts"
SUBDIR = "gangnam"  # <-- 너 폴더명으로 바꿔

img_dir = BASE_DIR / "gsv_photo" / SUBDIR

sizes = []
missing = 0

for jp in img_dir.glob("*.json"):
    with open(jp, "r", encoding="utf-8") as f:
        data = json.load(f)

    w = data.get("imageWidth")
    h = data.get("imageHeight")

    if w is None or h is None:
        missing += 1
        continue

    sizes.append((w, h))

c = Counter(sizes)

print("==== JSON에 들어있는 GT 기준 해상도 종류 ====")
for (w, h), cnt in c.items():
    print(f"{w} x {h}  : {cnt}개")

print(f"\nimageWidth/Height 없는 JSON: {missing}개")
