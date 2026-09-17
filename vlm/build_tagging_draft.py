#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T6 준비: 의미 태깅(name/amenity) GT 초안 생성.

논문 3번째 모듈(의미 태깅)의 정확도는 외부 GPT-4 결과라 저장소에 GT가 없습니다.
VLM/LLM 태깅을 측정하려면 411개 간판 크롭에 name·OSM 태그 정답이 필요한데,
OCR GT(간판에 적힌 텍스트)가 이미 있으므로 규칙으로 초안을 만들고 사람이
검수하는 편이 빠릅니다. 초안은 어디까지나 검수 대상이며, 확신도가 낮은 항목이
검수 UI에서 먼저 뜨도록 점수를 함께 남깁니다.

출력: artifacts/gt/tagging_gt_draft.csv
      image_name, region, gt_text, draft_name, draft_tag, matched_kw, confidence

검수: review_tagging.py
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
GT_DIR = HERE / "artifacts" / "gt"
REGIONS = ["gangnam", "brooklyn", "suwon"]
OUT = GT_DIR / "tagging_gt_draft.csv"

# 간판 텍스트 키워드 -> OSM 태그. 앞쪽(구체적)부터 매칭합니다.
KEYWORD_TAGS: list[tuple[tuple[str, ...], str]] = [
    # 브랜드 (가장 확실)
    (("스타벅스", "starbucks", "투썸", "twosome", "이디야", "ediya", "컴포즈",
      "메가커피", "빽다방", "dunkin", "던킨"), "amenity=cafe"),
    (("배스킨", "baskin", "31", "설빙"), "amenity=ice_cream"),
    (("롯데리아", "lotteria", "맥도날드", "mcdonald", "버거킹", "burger king",
      "kfc", "맘스터치", "subway", "써브웨이", "domino", "도미노", "피자헛",
      "pizza hut", "papa john"), "amenity=fast_food"),
    (("세븐일레븐", "7-eleven", "eleven", "gs25", "cu", "이마트24", "미니스톱",
      "bodega", "deli"), "shop=convenience"),
    (("올리브영", "oliveyoung", "아리따움", "이니스프리", "cvs", "walgreens"),
     "shop=chemist"),
    # 업종어 — 한국어
    (("약국",), "amenity=pharmacy"),
    (("병원", "의원", "치과", "한의원", "clinic", "dental", "medical"), "amenity=clinic"),
    (("동물병원", "펫", "vet"), "amenity=veterinary"),
    (("은행", "bank", "새마을금고", "신협", "credit union"), "amenity=bank"),
    (("부동산", "공인중개", "realty", "real estate", "estate"), "office=estate_agent"),
    (("학원", "어학원", "academy", "교습소"), "amenity=school"),
    (("pc방", "피시방", "internet cafe"), "amenity=internet_cafe"),
    (("노래방", "노래연습장", "karaoke"), "amenity=karaoke_box"),
    (("주유소", "gas", "fuel", "shell", "sk에너지"), "amenity=fuel"),
    (("커피", "coffee", "카페", "cafe", "espresso", "roast"), "amenity=cafe"),
    (("베이커리", "bakery", "제과", "빵", "파리바게", "뚜레쥬르", "donut", "도넛"),
     "shop=bakery"),
    # 한국 치킨 프랜차이즈는 술집보다 fast_food/restaurant에 가까워 분리
    (("치킨", "chicken", "bbq", "닭"), "amenity=fast_food"),
    (("호프", "포차", "술집", "pub", "bar", "beer", "brew", "소주", "막걸리",
      "이자카야", "와인"), "amenity=bar"),
    (("식당", "restaurant", "국밥", "갈비", "삼겹", "곱창", "찌개", "백반", "분식",
      "칼국수", "냉면", "쌈밥", "보쌈", "족발", "횟집", "초밥", "스시", "sushi",
      "ramen", "라멘", "파스타", "pasta", "grill", "kitchen", "food", "떡볶이",
      "돈까스", "감자탕", "순대", "만두", "쌀국수", "짜장", "중화"), "amenity=restaurant"),
    (("미용실", "헤어", "hair", "살롱", "salon", "barber", "이발"), "shop=hairdresser"),
    (("네일", "nail", "속눈썹", "왁싱", "피부", "스킨", "beauty", "에스테틱"),
     "shop=beauty"),
    (("안경", "optical", "optician", "eyewear"), "shop=optician"),
    (("세탁", "laundry", "크리닝", "cleaners", "dry clean"), "shop=laundry"),
    (("마트", "슈퍼", "supermarket", "grocery", "food market"), "shop=supermarket"),
    (("편의점", "convenience"), "shop=convenience"),
    (("의류", "패션", "fashion", "clothing", "boutique", "apparel", "wear"),
     "shop=clothes"),
    (("신발", "슈즈", "shoe", "sneaker"), "shop=shoes"),
    (("휴대폰", "폰", "mobile", "phone", "telecom", "kt", "skt", "lg u+", "올레",
      "verizon", "t-mobile"), "shop=mobile_phone"),
    (("철물", "hardware", "공구"), "shop=hardware"),
    (("가구", "furniture", "인테리어", "interior"), "shop=furniture"),
    (("서점", "book", "문구", "stationery"), "shop=books"),
    (("헬스", "gym", "fitness", "피트니스", "요가", "yoga", "필라테스", "pilates",
      "kickbox", "crossfit"), "leisure=fitness_centre"),
    (("호텔", "hotel", "모텔", "motel", "게스트하우스", "inn"), "tourism=hotel"),
    (("교회", "church", "성당", "사찰", "절"), "amenity=place_of_worship"),
    (("우체국", "post", "택배", "shipping", "ups", "fedex"), "amenity=post_office"),
    (("보험", "insurance", "세무", "tax", "법률", "law", "변호사", "attorney",
      "회계"), "office=company"),
    (("자동차", "카센타", "auto", "motors", "tire", "타이어", "정비", "repair",
      "bike", "자전거"), "shop=car_repair"),
    (("꽃", "flower", "florist"), "shop=florist"),
    (("정육", "butcher", "meat"), "shop=butcher"),
    (("과일", "청과", "fruit", "produce"), "shop=greengrocer"),
    (("주차", "parking"), "amenity=parking"),
    (("백화점", "department store", "mall", "plaza"), "shop=department_store"),
]

# name 후보에서 제외할 일반어(업종 표기·부가정보)
GENERIC = re.compile(
    r"^(open|close|sale|주차|영업|정품|전문|배달|포장|예약|문의|since|since\d+|"
    r"tel|phone|www|http|since|since|무료|할인|신장개업|24시|b1|1f|2f|3f|4f|5f)$",
    re.I)
DONTCARE = "###"
PHONE_RE = re.compile(r"\d{2,4}[-.\s)]\s?\d{3,4}[-.\s]?\d{4}")


def split_lines(text: str) -> list[str]:
    s = (text or "").replace("\\n", "\n")
    return [p.strip() for p in s.split("\n") if p.strip()]


def guess_tag(lines: list[str]) -> tuple[str, str]:
    """(tag, matched keyword) — 없으면 ('', '')."""
    hay = " ".join(lines).lower()
    for kws, tag in KEYWORD_TAGS:
        for kw in kws:
            if kw in hay:
                return tag, kw
    return "", ""


def guess_name(lines: list[str]) -> str:
    """상호명 후보: 일반어·전화번호·don't-care를 뺀 뒤 가장 긴 라인."""
    cands = []
    for l in lines:
        if l == DONTCARE or PHONE_RE.search(l) or GENERIC.match(l.strip()):
            continue
        if len(re.sub(r"[^0-9A-Za-z가-힣]", "", l)) < 2:
            continue
        cands.append(l)
    if not cands:
        return ""
    # 한글/영문 상호는 보통 가장 크게 적힌 첫 줄이거나 가장 긴 줄
    return max(cands, key=lambda l: len(re.sub(r"\s", "", l)))


def main() -> None:
    rows = []
    stats = {"tagged": 0, "no_tag": 0, "no_name": 0, "total": 0}
    for region in REGIONS:
        src = GT_DIR / f"ocr_{region}_gt.csv"
        for r in csv.DictReader(src.open(encoding="utf-8-sig")):
            lines = split_lines(r["gt_text"])
            scorable = [l for l in lines if l != DONTCARE]
            name = guess_name(lines)
            tag, kw = guess_tag(lines)
            # 확신도: 태그 매칭 + 이름 후보 유무 + don't-care 비중
            conf = 0
            if tag:
                conf += 2
            if name:
                conf += 1
            if scorable and len(scorable) >= 2:
                conf += 1
            if not scorable:
                conf = 0                     # 읽을 수 있는 텍스트가 없음
            rows.append({
                "image_name": r["image_name"], "region": region,
                "gt_text": r["gt_text"], "draft_name": name,
                "draft_tag": tag, "matched_kw": kw, "confidence": conf,
            })
            stats["total"] += 1
            stats["tagged" if tag else "no_tag"] += 1
            if not name:
                stats["no_name"] += 1

    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_name", "region", "gt_text",
                                          "draft_name", "draft_tag", "matched_kw",
                                          "confidence"])
        w.writeheader(); w.writerows(rows)

    print(f"[초안] {stats['total']}개 크롭 -> {OUT}")
    print(f"  태그 규칙 매칭: {stats['tagged']} ({stats['tagged']/stats['total']*100:.0f}%)")
    print(f"  태그 미매칭   : {stats['no_tag']}  ← 검수에서 직접 지정 필요")
    print(f"  이름 후보 없음: {stats['no_name']}")
    print("\n검수: .venv/Scripts/python.exe review_tagging.py")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
