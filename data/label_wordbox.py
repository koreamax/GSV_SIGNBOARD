#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GSV 단어 박스 라벨링 도구 (로컬 웹앱).

    .venv/Scripts/python.exe data/label_wordbox.py
    → http://127.0.0.1:8777  (브라우저가 자동으로 열립니다)

먼저 `data/prep_wordbox_label.py` 로 작업 이미지를 만들어 두어야 합니다.
씨앗 박스가 있으면(`data/make_wordbox_seed.py`) 자동으로 불러옵니다.

저장 형식 — `artifacts/gt/wordbox/labels.json`
    {"<작업 id>": {"boxes": [[x0,y0,x1,y1], ...],   # 작업 이미지 기준 0~1 정규화
                   "status": "done" | "todo",
                   "updated": "2026-09-23T04:10:00"}}

박스를 그릴 때마다 서버로 바로 저장됩니다(덮어쓰기는 임시파일 → 교체라 중간에 꺼도
파일이 깨지지 않습니다). 원본 사진 좌표로의 변환은 `data/export_wordbox.py` 가 합니다.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / 'artifacts' / 'gt' / 'wordbox'
ITEMS = OUT / 'items.json'
LABELS = OUT / 'labels.json'
SEEDS = OUT / 'seeds.json'
PAGE = Path(__file__).resolve().parent / 'label_wordbox.html'

# README 의 라벨 목표치 — 진행률 표시에만 씁니다.
TARGETS = {'gangnam': 306, 'brooklyn': 383, 'suwon': 241}

_lock = threading.Lock()


def read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError) as e:
        print(f'[경고] {path.name} 을 읽지 못했습니다: {e}')
        return default


def write_labels(data: dict) -> None:
    """임시파일에 쓰고 교체 — 저장 도중 꺼져도 기존 라벨이 날아가지 않습니다."""
    tmp = LABELS.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding='utf-8')
    os.replace(tmp, LABELS)


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *a):        # 조용히
        pass

    def _send(self, code, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        # 이미지만 캐시합니다. HTML/JSON 을 캐시하면 앱을 고쳐도 옛 화면이 뜹니다.
        self.send_header('Cache-Control',
                         'max-age=86400' if ctype.startswith('image') else 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode('utf-8'),
                   'application/json; charset=utf-8')

    def do_GET(self):
        path = self.path.split('?')[0]
        if path in ('/', '/index.html'):
            if not PAGE.exists():
                return self._send(500, b'label_wordbox.html not found', 'text/plain')
            return self._send(200, PAGE.read_bytes(), 'text/html; charset=utf-8')

        if path == '/api/state':
            items = read_json(ITEMS, [])
            with _lock:
                labels = read_json(LABELS, {})
            return self._json({'items': items,
                               'labels': labels,
                               'seeds': read_json(SEEDS, {}),
                               'targets': TARGETS})

        if path.startswith('/img/'):
            name = os.path.basename(path[len('/img/'):])
            p = OUT / 'img' / name
            if p.exists() and p.suffix.lower() == '.jpg':
                return self._send(200, p.read_bytes(), 'image/jpeg')
            return self._send(404, b'not found', 'text/plain')

        return self._send(404, b'not found', 'text/plain')

    def do_POST(self):
        if self.path != '/api/save':
            return self._send(404, b'not found', 'text/plain')
        n = int(self.headers.get('Content-Length', 0))
        try:
            payload = json.loads(self.rfile.read(n).decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            return self._json({'ok': False, 'error': str(e)}, 400)

        item_id = payload.get('id')
        if not item_id:
            return self._json({'ok': False, 'error': 'id 없음'}, 400)
        boxes = [[round(float(v), 6) for v in b[:4]] for b in payload.get('boxes', [])]
        status = 'done' if payload.get('status') == 'done' else 'todo'

        with _lock:
            labels = read_json(LABELS, {})
            labels[item_id] = {'boxes': boxes, 'status': status,
                               'updated': datetime.now().isoformat(timespec='seconds')}
            write_labels(labels)
            total = sum(len(v['boxes']) for v in labels.values())
            done = sum(1 for v in labels.values() if v['status'] == 'done')
        return self._json({'ok': True, 'n_boxes': total, 'n_done': done})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8777)
    ap.add_argument('--no-browser', action='store_true')
    args = ap.parse_args()

    if not ITEMS.exists():
        raise SystemExit('작업 목록이 없습니다. 먼저 data/prep_wordbox_label.py 를 돌리세요.')

    n_items = len(read_json(ITEMS, []))
    n_seed = len(read_json(SEEDS, {}))
    url = f'http://127.0.0.1:{args.port}'
    print(f'작업 {n_items}개 · 씨앗 {n_seed}개 · 라벨 파일 {LABELS}')
    print(f'주소: {url}   (Ctrl+C 로 종료)')
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
    except KeyboardInterrupt:
        print('\n종료했습니다. 라벨은 저장돼 있습니다.')


if __name__ == '__main__':
    main()
