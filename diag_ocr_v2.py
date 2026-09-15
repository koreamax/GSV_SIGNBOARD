# -*- coding: utf-8 -*-
"""Diagnostic v2: SANE whole-string CER/WER + exact-match accuracy + samples.
Compares against the flawed line_match (each GT line vs WHOLE pred)."""
import csv, re, glob
from pathlib import Path

def norm(s):
    s = (s or '').strip().replace('\r\n', '\n').replace('\r', '\n').replace('\n', ' ')
    s = s.lower()
    return ' '.join(s.split())

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

def cer_whole(p, g):
    P, G = norm(p).replace(' ', ''), norm(g).replace(' ', '')
    return lev(P, G) / len(G) if G else 0.0

def has_kr(s): return bool(re.search('[가-힣]', s))

def load(path):
    raw = Path(path).read_bytes()
    try: txt = raw.decode('utf-8')
    except Exception: txt = raw.decode('cp949', 'replace')
    r = csv.DictReader(txt.splitlines())
    kc = 'image_id' if 'image_id' in r.fieldnames else 'image_name'
    out = {}
    for row in r:
        k = (row.get(kc) or '').strip()
        if not k: continue
        out[k.rsplit('.', 1)[0]] = (row.get('gt_text') or '').strip()
    return out

def latest(region, engine):
    c = glob.glob(f'artifacts/ocr_gt/ocr_{region}_*_{engine}.csv')
    if not c: return None
    c.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
    return c[0]

region = 'gangnam'
gt = load(f'artifacts/gt/ocr_{region}_gt.csv')
easy = load(latest(region, 'easyocr'))
tro = load(latest(region, 'trocr'))

print('=' * 70)
print(f'WHOLE-STRING metric (GT lines joined -> one string), region={region}')
print('CER capped at 1.0 for the mean (per-crop min(cer,1)); accuracy = 1-meanCER')
print('=' * 70)
for eng, pm in [('easyocr', easy), ('trocr', tro)]:
    buck = {'ALL': [], 'KR': [], 'EN': []}
    exact = {'ALL': 0, 'KR': 0, 'EN': 0}; cnt = {'ALL': 0, 'KR': 0, 'EN': 0}
    empty = 0
    for k, g in gt.items():
        p = pm.get(k, '')
        if not p.strip(): empty += 1
        gjoin = ' '.join(re.split(r'[|\n]', g))
        c = min(cer_whole(p, gjoin), 1.0)
        b = 'KR' if has_kr(g) else 'EN'
        for key in ('ALL', b):
            buck[key].append(c); cnt[key] += 1
            if norm(p).replace(' ', '') == norm(gjoin).replace(' ', '') and norm(gjoin): exact[key] += 1
    print(f'\n{eng}:  empty_pred={empty}/{len(gt)}')
    print(f'  {"bucket":6} {"crops":6} {"meanCER":>8} {"~accuracy":>10} {"exact%":>8}')
    for b in ['ALL', 'KR', 'EN']:
        xs = buck[b]; mc = sum(xs)/len(xs) if xs else 0
        acc = (1 - mc) * 100
        ex = exact[b] / cnt[b] * 100 if cnt[b] else 0
        print(f'  {b:6} {cnt[b]:6} {mc:8.3f} {acc:9.1f}% {ex:7.1f}%')

print('\n' + '=' * 70)
print('SAMPLE OUTPUTS (first 12 crops)   GT  ||  EasyOCR  ||  TrOCR')
print('=' * 70)
for i, k in enumerate(list(gt.keys())[:12]):
    g = ' / '.join(re.split(r'[|\n]', gt[k]))[:45]
    e = norm(easy.get(k, ''))[:35]
    t = norm(tro.get(k, ''))[:35]
    print(f'{k[-18:]:18} GT[{g}]')
    print(f'{"":18} EASY[{e}]  TRO[{t}]')
