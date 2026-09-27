#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""라벨링 씨앗: 단어 검출기 **4개의 합집합**을 만듭니다.

왜 합집합인가
-------------
YOLO 검출 결과만 씨앗으로 쓰면 YOLO 가 놓친 단어는 라벨에도 안 생깁니다. 그러면 그
라벨로 잰 AP 는 YOLO 에게 유리한 쪽으로 기울고, Table 2 의 모델 비교가 무너집니다.
그래서 4개 아키텍처의 제안을 전부 올려 두고, 라벨러가 **전수 수정**하는 것을 전제로
합니다. 씨앗은 정답이 아니라 '지우고 고치라고 깔아 둔 것'입니다.

어느 모델이 제안했는지(`by`)를 같이 남기므로, 나중에 "한 모델만 찾은 박스"를 따로
추려 확인할 수 있습니다.

사용
----
    # GPU 가 놀고 있을 때 (5분 내외)
    .venv/Scripts/python.exe data/make_wordbox_seed.py
    # 다른 학습이 GPU 를 쓰고 있을 때 (CPU, 1시간 내외)
    .venv/Scripts/python.exe data/make_wordbox_seed.py --cpu

출력: artifacts/gt/wordbox/seeds.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / 'artifacts' / 'gt' / 'wordbox'


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='yolo26x,yolov5x,frcnn,effdet')
    ap.add_argument('--conf', type=float, default=0.25,
                    help='모델별 최소 점수 (낮출수록 제안이 늘고 지울 게 많아집니다)')
    ap.add_argument('--iou', type=float, default=0.55, help='모델 간 같은 단어로 묶는 IoU')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--cpu', action='store_true', help='GPU 를 쓰는 학습이 돌 때')
    return ap.parse_args()


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def merge(proposals, iou_thr):
    """점수 높은 것부터 집어 가며 겹치는 제안을 한 박스로 묶습니다(제안자 이름 누적)."""
    proposals = sorted(proposals, key=lambda p: -p['score'])
    kept = []
    for p in proposals:
        for k in kept:
            if iou(p['box'], k['box']) >= iou_thr:
                if p['by'][0] not in k['by']:
                    k['by'].append(p['by'][0])
                break
        else:
            kept.append(p)
    return kept


def main() -> None:
    args = parse_args()
    if args.cpu:
        # torch import 전에 꺼야 합니다. 빈 문자열은 무시되는 경우가 있어 '-1' 을 씁니다.
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    sys.path.insert(0, str(HERE / 'detection'))
    import eval_det_unified as U                      # noqa: E402
    from eval_text_holdout import MODELS, predict_frcnn_text   # noqa: E402

    items = json.loads((OUT / 'items.json').read_text(encoding='utf-8'))
    if args.limit:
        items = items[:args.limit]
    paths = [(OUT / 'img' / it['img'], [], None) for it in items]
    print(f'작업 이미지 {len(paths)}장 · 장치 {U.DEVICE}')

    per_item: dict[str, list] = {it['id']: [] for it in items}
    for name in args.models.split(','):
        kind, wp, res = MODELS[name]
        weights = HERE / wp
        if not weights.exists():
            print(f'[{name}] 가중치 없음: {weights} — 건너뜁니다')
            continue
        print(f'[{name}] 추론 시작 ({kind}, res={res})', flush=True)
        if kind == 'ultra':
            preds = U.predict_ultra(weights, paths, imgsz=res, conf=args.conf,
                                    device='cpu' if args.cpu else None)
        elif kind == 'frcnn':
            preds = predict_frcnn_text(weights, paths)
        else:
            preds = U.predict_effdet(weights, paths, img_size=res)

        n = 0
        for it, pr in zip(items, preds):
            iw, ih = it['img_size']
            for (x0, y0, x1, y1, s) in pr:
                if s < args.conf:
                    continue
                per_item[it['id']].append({
                    'box': [max(0.0, x0 / iw), max(0.0, y0 / ih),
                            min(1.0, x1 / iw), min(1.0, y1 / ih)],
                    'score': round(float(s), 4),
                    'by': [name],
                })
                n += 1
        print(f'[{name}] 제안 {n}개', flush=True)

    seeds = {k: merge(v, args.iou) for k, v in per_item.items()}
    for v in seeds.values():
        for s in v:
            s['box'] = [round(c, 6) for c in s['box']]
    (OUT / 'seeds.json').write_text(
        json.dumps(seeds, ensure_ascii=False, indent=1), encoding='utf-8')

    total = sum(len(v) for v in seeds.values())
    solo = sum(1 for v in seeds.values() for s in v if len(s['by']) == 1)
    print(f'\n씨앗 {total}개 (이미지당 평균 {total / max(1, len(items)):.1f})')
    print(f'  한 모델만 찾은 것 {solo}개 — 여기에 오검출과 "남들이 놓친 것"이 섞여 있습니다')
    print(f'저장: {OUT / "seeds.json"}')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    main()
