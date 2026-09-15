# -*- coding: utf-8 -*-
"""Diagnostic: per-region, per-language (Korean vs Latin) CER/WER for OCR engines.
Replicates eval_ocr.py line_match metric (lower=on, newline->space, spaces kept).
Only evaluates regions that have OCR result CSVs present in artifacts/ocr_gt/.
"""
import csv, re, glob
from pathlib import Path

def norm(s, for_wer):
    s = (s or '').strip().replace('\r\n', '\n').replace('\r', '\n').replace('\n', ' ')
    s = s.lower()
    s = ' '.join(s.split())
    return s

def lev(a, b):
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    prev = list(range(m + 1)); cur = [0] * (m + 1)
    for i in range(1, n + 1):
        cur[0] = i; ai = a[i - 1]
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if ai == b[j - 1] else 1))
        prev, cur = cur, prev
    return prev[m]

def cer(p, g):
    P, G = norm(p, 0), norm(g, 0)
    return lev(P, G) / len(G) if G else 0.0

def wer(p, g):
    P, G = norm(p, 1).split(), norm(g, 1).split()
    return lev(P, G) / len(G) if G else 0.0

def split_lines(t):
    s = (t or '').replace('\r\n', '\n').replace('\r', '\n').strip()
    if not s: return []
    parts = s.split('|') if '|' in s else s.split('\n')
    return [p.strip() for p in parts if p.strip()]

def has_kr(s):
    return bool(re.search('[가-힣]', s))

def load(path):
    raw = Path(path).read_bytes()
    try:
        txt = raw.decode('utf-8')
    except Exception:
        txt = raw.decode('cp949', 'replace')
    r = csv.DictReader(txt.splitlines())
    kc = 'image_id' if 'image_id' in r.fieldnames else 'image_name'
    out = {}
    for row in r:
        k = (row.get(kc) or '').strip()
        if not k: continue
        out[k.rsplit('.', 1)[0]] = (row.get('gt_text') or '').strip()
    return out

GT = {'gangnam': 'artifacts/gt/ocr_gangnam_gt.csv',
      'brooklyn': 'artifacts/gt/ocr_brooklyn_gt.csv',
      'suwon': 'artifacts/gt/ocr_suwon_gt.csv'}

def latest(region, engine):
    cands = glob.glob(f'artifacts/ocr_gt/ocr_{region}_*_{engine}.csv')
    if not cands: return None
    cands.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
    return cands[0]

print('=' * 64)
print('OCR LANGUAGE-SPLIT DIAGNOSTIC  (line_match, lower=on, spaces kept)')
print('CER/WER: 0 = perfect, higher = worse (can exceed 1)')
print('=' * 64)

for region, gtpath in GT.items():
    if not Path(gtpath).exists():
        continue
    gt = load(gtpath)
    engines = {}
    for eng in ['easyocr', 'trocr']:
        p = latest(region, eng)
        if p:
            engines[eng] = load(p)
    if not engines:
        print(f'\n[{region}]  GT={len(gt)} crops  ->  NO OCR RESULTS YET (run inference)')
        continue
    print(f'\n[{region}]  GT={len(gt)} crops   files: {list(engines.keys())}')
    print(f'  {"engine":8} {"bucket":6} {"lines":6} {"CER":>7} {"WER":>7}')
    for eng, pm in engines.items():
        buckets = {'ALL': [[], []], 'KR': [[], []], 'EN': [[], []]}
        empty = 0
        for k, g in gt.items():
            p = pm.get(k, '')
            if not p.strip(): empty += 1
            for ln in split_lines(g):
                c, w = cer(p, ln), wer(p, ln)
                b = 'KR' if has_kr(ln) else 'EN'
                buckets[b][0].append(c); buckets[b][1].append(w)
                buckets['ALL'][0].append(c); buckets['ALL'][1].append(w)
        for b in ['ALL', 'KR', 'EN']:
            cs, ws = buckets[b]
            mc = sum(cs) / len(cs) if cs else 0
            mw = sum(ws) / len(ws) if ws else 0
            print(f'  {eng:8} {b:6} {len(cs):6} {mc:7.3f} {mw:7.3f}')
        print(f'     -> empty_pred = {empty}/{len(gt)} crops')
