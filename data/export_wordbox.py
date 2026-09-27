#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""라벨링 결과를 원본 사진 좌표의 YOLO 형식 데이터셋으로 내보냅니다.

라벨은 작업 이미지(간판 축정렬 크롭) 기준 0~1 로 저장돼 있습니다. 되돌리는 식은
`prep_wordbox_label.py` 가 남긴 정보로 평행이동·배율뿐입니다:

    사진x = crop_box[0] + (정규화x * 작업이미지폭) / scale

한 사진에 간판이 여러 개면 작업 이미지도 여러 개이고, 이웃 간판 글자가 양쪽에
찍혀 같은 단어가 두 번 들어올 수 있습니다. 사진 단위로 합친 뒤 IoU 로 한 번
정리합니다.

출력 (`artifacts/gt/wordbox/yolo/`)
    images/<region>__<photo>.jpg   원본 사진 (하드링크, 안 되면 복사)
    labels/<region>__<photo>.txt   `0 xc yc w h` 정규화 — eval_text_holdout 과 같은 형식
    summary.csv                    지역별 사진/박스 수

사용
----
    .venv/Scripts/python.exe data/export_wordbox.py
    .venv/Scripts/python.exe data/export_wordbox.py --allow-partial   # 작업 중 확인용
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / 'artifacts' / 'gt' / 'wordbox'
PHOTO = HERE / 'artifacts' / 'gsv_photo'
DST = OUT / 'yolo'
REGIONS = ('gangnam', 'brooklyn', 'suwon')


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def dedupe(boxes, thr):
    """간판이 겹쳐 같은 단어가 두 번 들어온 경우를 정리합니다(큰 쪽을 남깁니다)."""
    boxes = sorted(boxes, key=lambda b: -((b[2] - b[0]) * (b[3] - b[1])))
    kept = []
    for b in boxes:
        if all(iou(b, k) < thr for k in kept):
            kept.append(b)
    return kept


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)          # 8K 사진 298장을 복사하지 않습니다
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--allow-partial', action='store_true',
                    help='완료 표시가 안 된 항목이 있어도 내보냅니다')
    ap.add_argument('--dedupe-iou', type=float, default=0.6)
    args = ap.parse_args()

    items = json.loads((OUT / 'items.json').read_text(encoding='utf-8'))
    labels_path = OUT / 'labels.json'
    if not labels_path.exists():
        raise SystemExit('라벨이 없습니다. data/label_wordbox.py 로 먼저 작업하세요.')
    labels = json.loads(labels_path.read_text(encoding='utf-8'))

    todo = [it['id'] for it in items if labels.get(it['id'], {}).get('status') != 'done']
    if todo and not args.allow_partial:
        raise SystemExit(
            f'아직 완료 표시가 안 된 항목이 {len(todo)}개 있습니다 (예: {todo[:3]}).\n'
            '중간 점검이면 --allow-partial 을 붙이세요. 완료된 것만 내보냅니다.')

    # 사진 단위로 모읍니다.
    per_photo: dict[tuple[str, str], list] = defaultdict(list)
    photo_size: dict[tuple[str, str], tuple[int, int]] = {}
    used_items = 0
    for it in items:
        rec = labels.get(it['id'])
        if not rec or rec.get('status') != 'done':
            continue
        used_items += 1
        key = (it['region'], it['photo'])
        photo_size[key] = tuple(it['photo_size'])
        ox, oy = it['crop_box'][0], it['crop_box'][1]
        iw, ih = it['img_size']
        s = it['scale']
        for nx0, ny0, nx1, ny1 in rec['boxes']:
            per_photo[key].append((ox + nx0 * iw / s, oy + ny0 * ih / s,
                                   ox + nx1 * iw / s, oy + ny1 * ih / s))

    (DST / 'images').mkdir(parents=True, exist_ok=True)
    (DST / 'labels').mkdir(parents=True, exist_ok=True)

    stats = {r: {'photos': 0, 'boxes': 0, 'dropped': 0} for r in REGIONS}
    for (region, photo), boxes in sorted(per_photo.items()):
        W, H = photo_size[(region, photo)]
        kept = dedupe(boxes, args.dedupe_iou)
        stats[region]['photos'] += 1
        stats[region]['boxes'] += len(kept)
        stats[region]['dropped'] += len(boxes) - len(kept)

        name = f'{region}__{photo}'
        link_or_copy(PHOTO / region / f'{photo}.jpg', DST / 'images' / f'{name}.jpg')
        lines = []
        for x0, y0, x1, y1 in kept:
            x0, y0 = max(0.0, x0), max(0.0, y0)
            x1, y1 = min(float(W), x1), min(float(H), y1)
            if x1 - x0 < 1 or y1 - y0 < 1:
                continue
            lines.append(f'0 {((x0 + x1) / 2) / W:.6f} {((y0 + y1) / 2) / H:.6f} '
                         f'{(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}')
        (DST / 'labels' / f'{name}.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    with open(OUT / 'summary.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['region', 'photos', 'word_boxes', 'deduped_away'])
        for r in REGIONS:
            w.writerow([r, stats[r]['photos'], stats[r]['boxes'], stats[r]['dropped']])

    print(f'완료 표시된 작업 이미지 {used_items} / {len(items)}')
    for r in REGIONS:
        s = stats[r]
        print(f'  {r:9s} 사진 {s["photos"]:3d} · 단어 박스 {s["boxes"]:4d}'
              f' (중복 제거 {s["dropped"]})')
    print(f'\n저장: {DST}')
    if todo:
        print(f'주의: 완료 안 된 {len(todo)}개는 빠졌습니다 — 최종 표에 쓰지 마세요.')


if __name__ == '__main__':
    main()
