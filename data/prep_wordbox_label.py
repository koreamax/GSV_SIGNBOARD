#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GSV 단어 박스 라벨링용 작업 이미지를 만듭니다 (Table 2 의 지역 칸과 연쇄 AP 용).

왜 원본 사진에서 자르나
-----------------------
`artifacts/gt/crop` 의 GT 크롭은 폴리곤 마스크 + **원근 보정(warp)** + CLAHE 를 거친
이미지라, 거기에 그린 박스는 원본 사진 좌표로 되돌릴 수 없습니다. 연쇄 AP 는
사진 → 간판 검출 → 크롭 → 단어 검출을 **원본 좌표계**에서 채점하므로, 라벨도 원본
좌표여야 합니다.

그래서 GT 폴리곤의 **축정렬 bbox**(+여유)로 원본 사진에서 잘라 작업 이미지를 만듭니다.
되돌리는 식은 단순한 평행이동·배율뿐입니다:

    사진x = crop_box[0] + 이미지u / scale

간판이 기울어져 있으면 이웃 간판 글자도 화면에 들어옵니다. 어디까지가 이 간판인지
보이도록 폴리곤 외곽선을 같이 저장해 라벨러 화면에 겹쳐 그립니다.

출력
----
    artifacts/gt/wordbox/items.json      작업 목록(좌표 되돌리기 정보 포함)
    artifacts/gt/wordbox/img/*.jpg       작업 이미지 463장

사용
----
    .venv/Scripts/python.exe data/prep_wordbox_label.py [--overwrite]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parents[1]
GT = HERE / 'artifacts' / 'gt'
PHOTO = HERE / 'artifacts' / 'gsv_photo'
OUT = GT / 'wordbox'
REGIONS = ('gangnam', 'brooklyn', 'suwon')

PAD = 0.06          # 축정렬 bbox 여유 (글자가 폴리곤 밖으로 조금 나오는 경우가 있습니다)
TARGET = 1400       # 작업 이미지 긴 변 목표
MAX_UP = 4.0        # 작은 간판이라도 4배까지만 확대 (그 이상은 뭉갭니다)


def load_ocr_gt(region: str) -> dict[str, str]:
    """크롭별 정답 텍스트 — 라벨러가 '몇 단어가 있어야 하는지' 가늠하는 힌트로 씁니다."""
    path = GT / f'ocr_{region}_gt.csv'
    if not path.exists():
        return {}
    out = {}
    with open(path, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            out[row['image_name'].replace('.jpg', '')] = row.get('gt_text', '')
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    (OUT / 'img').mkdir(parents=True, exist_ok=True)
    items, skipped = [], []

    for region in REGIONS:
        ocr_gt = load_ocr_gt(region)
        rows = list(csv.DictReader(open(GT / f'gt_{region}.csv', encoding='utf-8-sig')))
        per_photo: dict[str, int] = {}
        for row in rows:
            stem = row['filename'].split('__')[1].replace('.jpg', '')
            per_photo[stem] = per_photo.get(stem, 0) + 1
            idx = per_photo[stem]
            name = f'{region}__{stem}__crop_{idx:03d}'

            photo = PHOTO / region / f'{stem}.jpg'
            if not photo.exists():
                skipped.append((name, '사진 없음'))
                continue
            with Image.open(photo) as im:
                W, H = im.size

                pts = [(float(row[f'x{i}']), float(row[f'y{i}'])) for i in (1, 2, 3, 4)]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                bw, bh = max(xs) - min(xs), max(ys) - min(ys)
                if bw < 4 or bh < 4:
                    skipped.append((name, '폴리곤이 너무 작음'))
                    continue
                x0 = max(0.0, min(xs) - bw * PAD)
                y0 = max(0.0, min(ys) - bh * PAD)
                x1 = min(float(W), max(xs) + bw * PAD)
                y1 = min(float(H), max(ys) + bh * PAD)

                scale = min(MAX_UP, TARGET / max(x1 - x0, y1 - y0))
                iw, ih = max(1, round((x1 - x0) * scale)), max(1, round((y1 - y0) * scale))
                dst = OUT / 'img' / f'{name}.jpg'
                if args.overwrite or not dst.exists():
                    crop = im.convert('RGB').crop((int(x0), int(y0), int(x1), int(y1)))
                    crop.resize((iw, ih), Image.LANCZOS).save(dst, quality=92)

            items.append({
                'id': name,
                'region': region,
                'photo': stem,
                'crop_index': idx,
                'photo_size': [W, H],
                'crop_box': [x0, y0, x1, y1],       # 원본 사진 좌표
                'img': f'{name}.jpg',
                'img_size': [iw, ih],
                'scale': scale,                      # 작업이미지 px / 사진 px
                'poly': [[(px - x0) * scale, (py - y0) * scale] for px, py in pts],
                'has_ocr_gt': name in ocr_gt,
                'gt_text': ocr_gt.get(name, ''),
            })
        print(f'{region}: {sum(1 for i in items if i["region"] == region)}개')

    # OCR GT 가 있는 크롭(=README 의 411장)을 앞에 둬 중요한 것부터 작업하게 합니다.
    items.sort(key=lambda i: (not i['has_ocr_gt'], REGIONS.index(i['region']),
                              int(i['photo']), i['crop_index']))
    (OUT / 'items.json').write_text(
        json.dumps(items, ensure_ascii=False, indent=1), encoding='utf-8')

    n_gt = sum(i['has_ocr_gt'] for i in items)
    print(f'\n작업 대상 {len(items)}개 (OCR GT 있음 {n_gt} · 없음 {len(items) - n_gt})')
    if skipped:
        print(f'제외 {len(skipped)}: {skipped[:5]}')
    print(f'저장: {OUT / "items.json"}')


if __name__ == '__main__':
    main()
