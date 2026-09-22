"""OCR 인식기 통제 비교 — 단일 모델 기준 · 사진 단위 val/test 분리 · 차이의 부트스트랩 신뢰구간.

무엇을 고정했나: 검출기(CRAFT ∪ PaddleOCR-DB), 라인 병합(y_tol 0.04), 스트립 패딩(0.04),
전처리, 채점 규칙(eval_ocr_v2 --mask-phone, 브루클린 en-only). 바꾼 것은 인식기 하나뿐입니다.
미세조정군은 전부 같은 75,612 크롭 · 같은 signboard_v3 분할로 학습했습니다.

앙상블 행(배포 3-way vote)은 비교에서 제외합니다 — 모델이 아니라 시스템이라 같은 조건이 아닙니다.
"""
import os, random, sys
from pathlib import Path

os.chdir(r"C:/Users/DANHA/anaconda3/envs/GSV_YOLO_OCR/main")
sys.path[:0] = ["pipeline", "ocr", "vlm", "str_baselines"]
saved = sys.argv
sys.argv = ["eval_ocr_v2.py", "--mask-phone", "--en-only-regions", "brooklyn"]
import eval_ocr_v2 as E
sys.argv = saved

GT = Path("artifacts/gt"); OCR = Path("artifacts/ocr_gt")
REG = ["gangnam", "brooklyn", "suwon"]

ENGINES = [                                  # (태그, 표기, 학습 조건)
    ("tesseract",    "Tesseract 5.5",            "zero-shot"),
    ("surya",        "Surya 0.14",               "zero-shot"),
    ("clova",        "CLOVA OCR General",        "zero-shot (상용 API)"),
    ("tesseract_ft", "Tesseract 5.5",            "fine-tuned"),
    ("trocr",        "TrOCR-small",              "fine-tuned"),
    ("easyocr",      "EasyOCR (CRNN)",           "fine-tuned"),
    ("parseq",       "PARSeq (ViT-S)",           "fine-tuned"),
    ("paddle1",      "PP-OCRv5 rec",             "fine-tuned"),
    ("svtrv2",       "SVTRv2-B",                 "fine-tuned"),
]
REF = "paddle1"                              # 기준 = 배포 인식기의 단일 모델

# ---------- 사진 단위 val/test 분할 ----------
gt_maps = {r: E.load_csv_map(GT / f"ocr_{r}_gt.csv") for r in REG}
rng = random.Random(20260922)
split = {}
for r in REG:
    ps = sorted({k.split("__")[1] for k in gt_maps[r]})
    rng.shuffle(ps)
    h = len(ps) // 2
    split[r] = {"val": set(ps[:h]), "test": set(ps[h:])}


def per_crop(engine):
    """크롭별 (정확히 맞은 라인 수, 채점 대상 라인 수, 편집거리, GT 문자수, 지역, 사진)."""
    out = []
    for r in REG:
        cands = sorted(OCR.glob(f"ocr_{r}_*_{engine}.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not cands:
            return None
        pred = E.load_csv_map(cands[0])
        E.EN_ONLY = r in E.EN_ONLY_REGIONS
        for k, g in gt_maps[r].items():
            glines, ndc = E.split_gt_lines(g)
            if not glines:
                continue
            pl = E.split_lines(pred.get(k, ""))
            assign, _ = E.match_lines(glines, pl, infix=ndc > 0)
            ex = sum(1 for (j, d, gl) in assign if d == 0)
            ed = sum(d for (j, d, gl) in assign)
            ch = sum(gl for (j, d, gl) in assign)
            out.append((ex, len(glines), ed, ch, r, k.split("__")[1]))
        E.EN_ONLY = False
    return out


def agg(rows, which=None):
    sel = [x for x in rows if which is None or x[5] in split[x[4]][which]]
    ex = sum(x[0] for x in sel); ln = sum(x[1] for x in sel)
    ed = sum(x[2] for x in sel); ch = sum(x[3] for x in sel)
    return (ex / max(ln, 1) * 100, ed / max(ch, 1), ln)


data = {}
for eng, name, cond in ENGINES:
    rows = per_crop(eng)
    if rows is None:
        print("missing:", eng); continue
    data[eng] = rows

print("검증: 전체(=기존 공식 수치와 대조)")
for eng, name, cond in ENGINES:
    if eng in data:
        e, c, n = agg(data[eng])
        print(f"  {name:<22}{cond:<22}{e:>6.1f}%  CER {c:.3f}  (lines {n})")

print("\n사진 단위 분할:", {r: (len(split[r]['val']), len(split[r]['test'])) for r in REG})
print(f"\n{'인식기':<22}{'학습 조건':<22}{'val':>8}{'test':>8}{'test CER':>10}")
for eng, name, cond in ENGINES:
    if eng not in data: continue
    v = agg(data[eng], "val"); t = agg(data[eng], "test")
    print(f"{name:<22}{cond:<22}{v[0]:>7.1f}%{t[0]:>7.1f}%{t[1]:>10.3f}")

# ---------- 기준(PP-OCRv5 단일) 대비 차이의 부트스트랩 95% CI (test 절반, 크롭 단위 페어) ----------
print(f"\n기준 = PP-OCRv5 rec 단일 모델. test 절반에서 차이(%p)와 95% 신뢰구간 (크롭 단위 부트스트랩 2000회)")
ref_rows = {(x[4], x[5], i): x for i, x in enumerate(data[REF])}
ref_list = [x for x in data[REF] if x[5] in split[x[4]]["test"]]
B = 2000
rb = random.Random(7)
for eng, name, cond in ENGINES:
    if eng == REF or eng not in data: continue
    cur = [x for x in data[eng] if x[5] in split[x[4]]["test"]]
    if len(cur) != len(ref_list):
        print(f"{name:<22} 크롭 수 불일치 — 건너뜀"); continue
    pairs = list(zip(cur, ref_list))
    n = len(pairs)
    base = (sum(a[0] for a, b in pairs) / sum(a[1] for a, b in pairs)
            - sum(b[0] for a, b in pairs) / sum(b[1] for a, b in pairs)) * 100
    diffs = []
    for _ in range(B):
        idx = [rb.randrange(n) for _ in range(n)]
        ae = sum(pairs[i][0][0] for i in idx); al = sum(pairs[i][0][1] for i in idx)
        be = sum(pairs[i][1][0] for i in idx); bl = sum(pairs[i][1][1] for i in idx)
        diffs.append((ae / max(al, 1) - be / max(bl, 1)) * 100)
    diffs.sort()
    lo, hi = diffs[int(0.025 * B)], diffs[int(0.975 * B)]
    sig = "유의" if lo > 0 or hi < 0 else "미판정"
    print(f"{name:<22}{cond:<22}{base:>+7.1f}%p  [{lo:+.1f}, {hi:+.1f}]  {sig}")
