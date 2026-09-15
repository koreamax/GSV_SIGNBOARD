#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RAG 검색용 상호명 사전(L3) 구축 — OSM POI만으로는 커버리지가 너무 낮아서.

측정된 문제(2026-08 기준): GT 라인 중 **OSM 상호명에 존재하는 비율이 강남 9.0% /
브루클린 16.1% / 수원 1.2%** 입니다. 사전에 없는 상호는 RAG로 못 고치므로,
검색 코퍼스를 넓히는 것이 RAG 도입의 선행조건입니다.

레이어 (전부 배포 시점에 합법적으로 쓸 수 있는 공개 데이터 — DB_RULES.md L1 규칙 준수):
  L1 osm       : OpenStreetMap POI          (build_ocr_db.py 가 이미 생성)
  L3 kr_sangga : 소상공인시장진흥공단 상가(상권)정보  → 강남·수원  [수동 다운로드 필요]
  L3 nyc_lob   : NYC Legally Operating Businesses     → 브루클린   [자동 수집]
  L2 vocab     : signboard train 어휘        (일반명사 — 검색 기본값에선 제외)

**금지(변함없음)**: 평가 GT(ocr_{region}_gt.csv, tagging_gt.csv)에서 어휘 추출.

Usage:
  .venv/Scripts/python.exe build_poi_db.py --nyc                    # 브루클린 자동
  .venv/Scripts/python.exe build_poi_db.py --kr-csv "D:/소상공인_서울.csv" "D:/소상공인_경기.csv"
  .venv/Scripts/python.exe build_poi_db.py --merge                  # rag_{region}.csv 생성
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from build_ocr_db import BBOX, norm_key   # 지역 bbox·정규화는 기존 정의를 그대로 씁니다

HERE = Path(__file__).resolve().parent
DB_DIR = HERE / "artifacts" / "ocr_db"
NYC_URL = "https://data.cityofnewyork.us/resource/w7w3-xahh.json"       # 영업 허가
# 허가 데이터(DCA)는 업종이 편중돼 있어 식당이 거의 없습니다. 태깅 GT의 최대
# 카테고리가 amenity=restaurant(113/395)이므로 보건국 식당 점검 데이터를 함께 씁니다.
NYC_REST_URL = "https://data.cityofnewyork.us/resource/43nn-pn8j.json"

# 업종 문자열 → OSM 태그. tagging_gt.csv의 47종 어휘에 맞춰 매핑합니다.
# 첫 매칭이 이기므로 구체적인 것을 위에 둡니다(예: '편의점'이 '소매'보다 먼저).
KR_TAG_RULES: list[tuple[str, str]] = [
    # ── 최우선: 아래 일반 규칙에 잡아먹히던 업종들 ──
    # 상가정보의 약국·의원 분류 문자열에 '화장품'·'미용'이 섞여 있어, 먼저 걸리는
    # 미용 규칙이 약국 453건 중 451건을 shop=beauty 로 만들고 있었습니다.
    ("약국", "amenity=pharmacy"), ("의약", "amenity=pharmacy"),
    ("치과", "amenity=clinic"), ("한의", "amenity=clinic"),
    ("의원", "amenity=clinic"), ("병원", "amenity=clinic"),
    ("동물병원", "amenity=veterinary"),
    ("학원", "amenity=school"), ("교습", "amenity=school"),
    ("편의점", "shop=convenience"),
    ("슈퍼마켓", "shop=supermarket"), ("대형마트", "shop=supermarket"),
    ("제과점", "shop=bakery"), ("빵", "shop=bakery"),
    ("커피", "amenity=cafe"), ("카페", "amenity=cafe"), ("다방", "amenity=cafe"),
    ("아이스크림", "amenity=ice_cream"), ("빙수", "amenity=ice_cream"),
    ("패스트푸드", "amenity=fast_food"), ("햄버거", "amenity=fast_food"),
    ("피자", "amenity=fast_food"), ("치킨", "amenity=fast_food"),
    ("주점", "amenity=bar"), ("호프", "amenity=bar"), ("바(BAR)", "amenity=bar"),
    ("술집", "amenity=bar"), ("맥주", "amenity=bar"),
    ("음식점", "amenity=restaurant"), ("한식", "amenity=restaurant"),
    ("중식", "amenity=restaurant"), ("일식", "amenity=restaurant"),
    ("양식", "amenity=restaurant"), ("분식", "amenity=restaurant"),
    ("뷔페", "amenity=restaurant"), ("food", "amenity=restaurant"),
    ("미용실", "shop=hairdresser"), ("헤어", "shop=hairdresser"),
    ("이용업", "shop=hairdresser"), ("이발", "shop=hairdresser"),
    ("네일", "shop=beauty"), ("피부", "shop=beauty"), ("에스테틱", "shop=beauty"),
    ("화장품", "shop=beauty"), ("미용", "shop=beauty"),
    ("약국", "amenity=pharmacy"), ("의약", "amenity=pharmacy"),
    ("의원", "amenity=clinic"), ("병원", "amenity=clinic"),
    ("치과", "amenity=clinic"), ("한의", "amenity=clinic"),
    ("동물병원", "amenity=veterinary"),
    ("부동산", "office=estate_agent"), ("공인중개", "office=estate_agent"),
    ("세탁", "shop=laundry"), ("빨래", "shop=laundry"),
    ("의복", "shop=clothes"), ("의류", "shop=clothes"), ("패션", "shop=clothes"),
    ("신발", "shop=clothes"),
    ("휴대폰", "shop=mobile_phone"), ("이동통신", "shop=mobile_phone"),
    ("통신기기", "shop=mobile_phone"),
    ("안경", "shop=optician"),
    ("귀금속", "shop=jewelry"), ("보석", "shop=jewelry"), ("시계", "shop=jewelry"),
    ("자동차수리", "shop=car_repair"), ("카센타", "shop=car_repair"),
    ("정비", "shop=car_repair"), ("타이어", "shop=car_repair"),
    ("자동차임대", "amenity=car_rental"), ("렌트카", "amenity=car_rental"),
    ("주차장", "amenity=parking"),
    ("숙박", "tourism=hotel"), ("호텔", "tourism=hotel"), ("모텔", "tourism=hotel"),
    ("여관", "tourism=hotel"),
    ("노래방", "amenity=karaoke_box"), ("노래연습", "amenity=karaoke_box"),
    ("PC방", "amenity=internet_cafe"), ("피시방", "amenity=internet_cafe"),
    ("스터디카페", "amenity=study_cafe"), ("독서실", "amenity=study_cafe"),
    ("헬스", "leisure=fitness_centre"), ("체력단련", "leisure=fitness_centre"),
    ("요가", "leisure=fitness_centre"), ("필라테스", "leisure=fitness_centre"),
    ("학원", "amenity=school"), ("교습", "amenity=school"), ("학교", "amenity=school"),
    ("은행", "amenity=bank"), ("금융", "amenity=bank"),
    ("우체국", "amenity=post_office"),
    ("서점", "shop=books"), ("도서", "shop=books"),
    ("문구", "shop=stationery"), ("사무용품", "shop=stationery"),
    ("가구", "shop=furniture"), ("침대", "shop=furniture"),
    ("철물", "shop=hardware"), ("공구", "shop=hardware"),
    ("꽃", "shop=florist"), ("화훼", "shop=florist"),
    ("담배", "shop=smoke_shop"), ("전자담배", "shop=smoke_shop"),
    ("애완", "shop=pet"), ("반려동물", "shop=pet"),
    ("건강식품", "shop=health_food"), ("건강기능", "shop=health_food"),
    ("오토바이", "shop=motorcycle"), ("이륜차", "shop=motorcycle"),
    # 전문서비스업 — 미매핑 상위를 차지하던 구간. OSM 관례상 office=company 로 묶입니다.
    ("컨설팅", "office=company"), ("광고", "office=company"),
    ("세무", "office=company"), ("회계", "office=company"),
    ("변호사", "office=company"), ("법무", "office=company"),
    ("변리", "office=company"), ("노무", "office=company"),
    ("디자인", "office=company"), ("설계", "office=company"),
    ("여행사", "office=company"), ("고용", "office=company"),
    ("인력", "office=company"), ("무역", "office=company"),
    ("소프트웨어", "office=company"), ("정보서비스", "office=company"),
    ("교육", "amenity=school"),
]

NYC_TAG_RULES: list[tuple[str, str]] = [
    ("sidewalk cafe", "amenity=cafe"),
    ("grocery", "shop=supermarket"), ("supermarket", "shop=supermarket"),
    ("bakery", "shop=bakery"),
    ("laundr", "shop=laundry"), ("dry clean", "shop=laundry"),
    ("tobacco", "shop=smoke_shop"), ("cigarette", "shop=smoke_shop"),
    ("garage", "amenity=parking"), ("parking", "amenity=parking"),
    ("pharmac", "amenity=pharmacy"), ("drug", "amenity=pharmacy"),
    ("barber", "shop=hairdresser"), ("beauty", "shop=beauty"), ("nail", "shop=beauty"),
    ("restaurant", "amenity=restaurant"), ("eating", "amenity=restaurant"),
    ("food", "amenity=restaurant"),
    ("catering", "amenity=restaurant"),
    ("garment", "shop=clothes"), ("apparel", "shop=clothes"),
    ("electronic", "shop=mobile_phone"),
    ("newsstand", "shop=convenience"), ("stoop line stand", "shop=convenience"),
    ("general vendor", "shop=convenience"),
    ("secondhand", "shop=furniture"), ("home improvement", "shop=hardware"),
    ("locksmith", "shop=hardware"),
    ("car wash", "shop=car_repair"), ("tow", "shop=car_repair"),
    ("auto", "shop=car_repair"),
    ("gas station", "amenity=fuel"),
    ("bingo", "amenity=bar"), ("pool", "leisure=fitness_centre"),
    ("health club", "leisure=fitness_centre"),
    ("hotel", "tourism=hotel"),
    ("pet", "shop=pet"),
    ("storage", "office=company"), ("employment agency", "office=company"),
    ("process serv", "office=company"), ("debt", "office=company"),
]


def map_tag(text: str, rules: list[tuple[str, str]], lower: bool = False) -> str:
    t = (text or "")
    if lower:
        t = t.lower()
    for kw, tag in rules:
        if kw in t:
            return tag
    return ""


def in_bbox(region: str, lat: float, lon: float) -> bool:
    s, w, n, e = BBOX[region]
    return s <= lat <= n and w <= lon <= e


def write_poi(region: str, rows: list[tuple[str, str, str, str, str]]) -> None:
    """rows: (name, key, tag, lat, lon) — key 중복은 태그 있는 쪽을 남깁니다."""
    best: dict[str, tuple] = {}
    for nm, k, tag, lat, lon in rows:
        cur = best.get(k)
        if cur is None or (not cur[2] and tag):
            best[k] = (nm, k, tag, lat, lon)
    out = DB_DIR / f"poi_{region}.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "key", "tag", "lat", "lon"])
        for k in sorted(best):
            w.writerow(best[k])
    tagged = sum(1 for v in best.values() if v[2])
    print(f"[L3] {out.name}: {len(best)}건 (태그 매핑 {tagged} / {len(best)})")


# ------------------------------------------------------------------ NYC (자동)
CUISINE_TAG = {
    "coffee/tea": "amenity=cafe",
    "bakery products/desserts": "shop=bakery",
    "donuts": "shop=bakery",
    "frozen desserts": "amenity=ice_cream",
    "hamburgers": "amenity=fast_food",
    "chicken": "amenity=fast_food",
    "pizza": "amenity=fast_food",
    "sandwiches": "amenity=fast_food",
    "hotdogs": "amenity=fast_food",
    "juice, smoothies, fruit salads": "amenity=fast_food",
    "bottled beverages": "shop=convenience",
    "soups/salads/sandwiches": "amenity=fast_food",
}


def _socrata(url: str, select: str, where: str, tag_of, label: str,
             unmapped: dict[str, int]) -> list[tuple]:
    """Socrata 데이터셋 하나를 bbox 조건으로 페이징 조회합니다."""
    rows, offset, limit = [], 0, 50000
    while True:
        params = {"$limit": limit, "$offset": offset,
                  "$select": select, "$where": where}
        req = urllib.request.Request(url + "?" + urllib.parse.urlencode(params),
                                     headers={"User-Agent": "gsv-ocr-db/1.0"})
        with urllib.request.urlopen(req, timeout=180) as r:
            batch = json.loads(r.read().decode("utf-8"))
        if not batch:
            break
        for d in batch:
            nm, cat, tag = tag_of(d)
            k = norm_key(nm)
            if len(k) < 2:
                continue
            if not tag and cat:
                unmapped[cat] = unmapped.get(cat, 0) + 1
            rows.append((nm, k, tag, d.get("latitude", ""), d.get("longitude", "")))
        offset += limit
        print(f"  [{label}] {offset}건 조회 (누적 후보 {len(rows)})", flush=True)
        if len(batch) < limit:
            break
        time.sleep(1)
    return rows


# 뉴욕주 데이터셋 — 좌표가 georeference(Point) 열에 들어 있어 within_box 로 거릅니다.
# 브루클린에서 사전에 없던 상호의 업종 분포(슈퍼/식료 13 · 미용 13 · 이용 7 · 바 5)를
# 보고 고른 소스입니다.
NYS_FOOD_URL = "https://data.ny.gov/resource/9a8c-vfzj.json"    # Retail Food Stores
NYS_BEAUTY_URL = "https://data.ny.gov/resource/y3u4-jbgh.json"  # 미용·이용 업소 면허
NYS_LIQUOR_URL = "https://data.ny.gov/resource/9s3h-dpkz.json"  # 주류 면허

LIQUOR_TAG = {
    "restaurant": "amenity=restaurant", "grocery store": "shop=convenience",
    "additional bar": "amenity=bar", "tavern": "amenity=bar", "club": "amenity=bar",
    "hotel": "tourism=hotel", "drug store": "amenity=pharmacy",
}


def _socrata_geo(url: str, select: str, bbox: tuple, row_of, label: str) -> list[tuple]:
    """georeference(Point) 열을 가진 데이터셋을 bbox로 페이징 조회합니다."""
    s, w, n, e = bbox
    where = f"within_box(georeference,{n},{w},{s},{e})"
    rows, offset, limit = [], 0, 50000
    while True:
        params = {"$limit": limit, "$offset": offset, "$select": select,
                  "$where": where}
        req = urllib.request.Request(url + "?" + urllib.parse.urlencode(params),
                                     headers={"User-Agent": "gsv-ocr-db/1.0"})
        with urllib.request.urlopen(req, timeout=180) as r:
            batch = json.loads(r.read().decode("utf-8"))
        if not batch:
            break
        for d in batch:
            nm, tag = row_of(d)
            k = norm_key(nm)
            if len(k) < 2:
                continue
            geo = (d.get("georeference") or {}).get("coordinates") or ["", ""]
            rows.append((nm.strip(), k, tag, geo[1], geo[0]))
        offset += limit
        if len(batch) < limit:
            break
        time.sleep(1)
    print(f"  [{label}] {len(rows)}건", flush=True)
    return rows


def fetch_nys() -> list[tuple]:
    """뉴욕주 3종 — 식품점·미용/이용·주류. 브루클린 bbox만 가져옵니다.

    ⚠️ **기본 비활성(`--nys`로만 켬).** 실측 결과 순손실이었습니다 (D29):
    커버리지는 17.6→18.1%로 올랐지만 태깅 브루클린 64.4→60.6%, 바뀐 예측 9건 중
    개선 0 / 악화 5. 원인은 이름 방해가 아니라 **태그 신뢰도**였습니다 —
    ① 태그 없이 이름만 들어간 항목(식품점)이 힌트 목록의 태그 신호를 희석하고
    ② 면허 종류로 유추한 태그(미용 vs 이용)가 경계 업종에서 오히려 오답을 유도했습니다.
    코퍼스를 키우는 것 자체가 목적이 되면 안 된다는 근거 사례로 남깁니다."""
    bbox = BBOX["brooklyn"]

    def food(d):
        # dba_name 이 곧 간판 상호입니다("GOURMET DELI"). 업종은 슈퍼/편의 경계가
        # 모호해(같은 델리를 GT가 양쪽으로 부름) **태그 없이 상호만** 넣습니다 —
        # 힌트가 틀린 답을 제안하지 않게 하는 원칙(4.15).
        return (d.get("dba_name") or d.get("entity_name") or ""), ""

    def beauty(d):
        lt = (d.get("license_type") or "").upper()
        tag = "shop=hairdresser" if "BAR" in lt else "shop=beauty"
        return (d.get("business_name") or ""), tag

    def liquor(d):
        desc = (d.get("description") or "").lower()
        return (d.get("legalname") or ""), LIQUOR_TAG.get(desc, "")

    rows = _socrata_geo(NYS_FOOD_URL, "dba_name,entity_name,georeference",
                        bbox, food, "nys-food")
    rows += _socrata_geo(NYS_BEAUTY_URL, "business_name,license_type,georeference",
                         bbox, beauty, "nys-beauty")
    rows += _socrata_geo(NYS_LIQUOR_URL, "legalname,description,georeference",
                         bbox, liquor, "nys-liquor")
    return rows


def fetch_nyc(with_nys: bool = False) -> None:
    s, w, n, e = BBOX["brooklyn"]
    where = f"latitude between {s} and {n} AND longitude between {w} and {e}"
    unmapped: dict[str, int] = {}

    def lic(d):
        # 간판에 걸리는 이름은 법인명이 아니라 상호(dba)입니다.
        nm = (d.get("dba_trade_name") or d.get("business_name") or "").strip()
        cat = d.get("business_category", "")
        return nm, cat, map_tag(cat, NYC_TAG_RULES, lower=True)

    def rest(d):
        nm = (d.get("dba") or "").strip()
        cat = (d.get("cuisine_description") or "").strip()
        tag = CUISINE_TAG.get(cat.lower(), "amenity=restaurant" if cat else "")
        return nm, cat, tag

    rows = _socrata(NYC_URL,
                    "business_name,dba_trade_name,business_category,latitude,longitude",
                    where, lic, "nyc-license", unmapped)
    rows += _socrata(NYC_REST_URL,
                     "dba,cuisine_description,latitude,longitude",
                     where, rest, "nyc-food", {})
    if with_nys:
        rows += fetch_nys()
    write_poi("brooklyn", rows)
    if unmapped:
        top = sorted(unmapped.items(), key=lambda x: -x[1])[:10]
        print("  [nyc] 태그 미매핑 상위:", ", ".join(f"{c}({n})" for c, n in top))


# ------------------------------------------------------- 소상공인 상가정보 (수동)
KR_NAME_COLS = ("상호명", "상호", "사업장명")
KR_LAT_COLS = ("위도", "lat", "y좌표")
KR_LON_COLS = ("경도", "lon", "lng", "x좌표")
KR_CAT_COLS = ("상권업종소분류명", "상권업종중분류명", "상권업종대분류명",
               "표준산업분류명", "업종명")


def _pick(header: list[str], names: tuple[str, ...]) -> str | None:
    for want in names:
        for h in header:
            if h.replace(" ", "").lower() == want.lower():
                return h
    for want in names:
        for h in header:
            if want.lower() in h.replace(" ", "").lower():
                return h
    return None


def ingest_kr(paths: list[str]) -> None:
    """소상공인 상가업소 CSV(들)을 읽어 강남·수원 bbox로 나눠 담습니다.

    시도별 파일을 그냥 다 던져주면 됩니다 — 좌표로 지역을 판정하므로 어느
    파일에 어느 지역이 있는지 신경 쓸 필요가 없습니다."""
    buckets: dict[str, list] = {"gangnam": [], "suwon": []}
    unmapped: dict[str, int] = {}
    for p in paths:
        path = Path(p)
        if not path.exists():
            print(f"[!] 파일 없음: {path}")
            continue
        enc = None
        for try_enc in ("utf-8-sig", "cp949", "utf-8"):
            try:
                with path.open(encoding=try_enc) as f:
                    f.readline()
                enc = try_enc
                break
            except UnicodeDecodeError:
                continue
        if enc is None:
            print(f"[!] 인코딩 판별 실패: {path}")
            continue

        with path.open(encoding=enc, newline="") as f:
            rd = csv.DictReader(f)
            hdr = rd.fieldnames or []
            c_nm = _pick(hdr, KR_NAME_COLS)
            c_lat = _pick(hdr, KR_LAT_COLS)
            c_lon = _pick(hdr, KR_LON_COLS)
            cats = [c for c in (_pick(hdr, (x,)) for x in KR_CAT_COLS) if c]
            if not (c_nm and c_lat and c_lon):
                print(f"[!] {path.name}: 필요한 열을 못 찾음 "
                      f"(상호명/위도/경도) — 헤더: {hdr[:12]}")
                continue
            print(f"[kr] {path.name} (enc={enc}) 상호={c_nm} 좌표={c_lat},{c_lon} "
                  f"업종={cats[:2]}")
            kept = 0
            for row in rd:
                try:
                    lat = float(row[c_lat]); lon = float(row[c_lon])
                except (TypeError, ValueError):
                    continue
                region = next((r for r in buckets if in_bbox(r, lat, lon)), None)
                if region is None:
                    continue
                nm = (row.get(c_nm) or "").strip()
                k = norm_key(nm)
                if len(k) < 2:
                    continue
                cat = " ".join((row.get(c) or "") for c in cats)
                tag = map_tag(cat, KR_TAG_RULES)
                if not tag and cat.strip():
                    key_cat = (row.get(cats[0]) or "").strip()
                    if key_cat:
                        unmapped[key_cat] = unmapped.get(key_cat, 0) + 1
                buckets[region].append((nm, k, tag, row[c_lat], row[c_lon]))
                kept += 1
            print(f"     bbox 안 {kept}건")

    for region, rows in buckets.items():
        if rows:
            write_poi(region, rows)
        else:
            print(f"[!] {region}: bbox 안에 든 행이 0건입니다 — "
                  f"파일에 해당 지역이 없거나 좌표계가 다를 수 있습니다.")
    if unmapped:
        top = sorted(unmapped.items(), key=lambda x: -x[1])[:15]
        print("[kr] 태그 미매핑 업종 상위:", ", ".join(f"{c}({n})" for c, n in top))
        print("     → 필요하면 build_poi_db.py 의 KR_TAG_RULES 에 추가하세요.")


# ------------------------------------------------------------------- merge
def merge() -> None:
    """레이어를 합쳐 검색 코퍼스 rag_{region}.csv 를 만듭니다.

    같은 표기가 여러 레이어에 있으면 **태그를 가진 쪽 > POI > 일반어휘** 순으로
    남깁니다. source 열이 남으므로 검색 시 레이어를 골라 쓸 수 있습니다."""
    vocab_path = DB_DIR / "vocab_signboard.csv"
    vocab = []
    if vocab_path.exists():
        for row in csv.DictReader(vocab_path.open(encoding="utf-8")):
            vocab.append((row["word"], row["key"], "", "vocab", "", ""))

    # 같은 표기가 여러 레이어에 있을 때의 우선순위. 일반어휘(vocab)가 POI를
    # 덮으면 그 상호가 검색에서 통째로 사라지므로(체인점 이름이 학습 어휘에도
    # 있는 경우가 많음) POI 레이어를 항상 위에 둡니다.
    RANK = {"vocab": 0, "kr_sangga": 1, "nyc_lob": 1, "osm": 2}

    for region in BBOX:
        merged: dict[str, tuple] = {}

        def put(nm, k, tag, src, lat="", lon=""):
            cur = merged.get(k)
            if cur is None or RANK[src] > RANK[cur[3]]:
                merged[k] = (nm, k, tag, src, lat, lon)
            elif RANK[src] == RANK[cur[3]] and not cur[2] and tag:
                merged[k] = (nm, k, tag, src, lat, lon)

        for nm, k, tag, src, lat, lon in vocab:
            put(nm, k, tag, src, lat, lon)
        osm_p = DB_DIR / f"osm_{region}.csv"
        if osm_p.exists():
            for row in csv.DictReader(osm_p.open(encoding="utf-8")):
                put(row["name"], row["key"], row.get("tag", ""), "osm")
        poi_p = DB_DIR / f"poi_{region}.csv"
        src = "nyc_lob" if region == "brooklyn" else "kr_sangga"
        if poi_p.exists():
            for row in csv.DictReader(poi_p.open(encoding="utf-8")):
                put(row["name"], row["key"], row.get("tag", ""), src,
                    row.get("lat", ""), row.get("lon", ""))

        out = DB_DIR / f"rag_{region}.csv"
        with out.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "key", "tag", "source", "lat", "lon"])
            for k in sorted(merged):
                w.writerow(merged[k])
        by_src: dict[str, int] = {}
        for v in merged.values():
            by_src[v[3]] = by_src.get(v[3], 0) + 1
        print(f"[RAG] {out.name}: {len(merged)}건  " +
              "  ".join(f"{s}={c}" for s, c in sorted(by_src.items())))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nyc", action="store_true", help="브루클린 POI 자동 수집")
    ap.add_argument("--nys", action="store_true",
                    help="뉴욕주 3종(식품점·미용·주류)도 포함 — 실측상 순손실이라 기본 off")
    ap.add_argument("--kr-csv", nargs="+", metavar="CSV",
                    help="소상공인 상가업소 CSV 경로(여러 개 가능)")
    ap.add_argument("--merge", action="store_true", help="rag_{region}.csv 생성")
    args = ap.parse_args()
    if not (args.nyc or args.kr_csv or args.merge):
        ap.error("--nyc / --kr-csv / --merge 중 하나 이상을 주세요.")
    DB_DIR.mkdir(parents=True, exist_ok=True)

    if args.nyc:
        fetch_nyc(with_nys=args.nys)
    if args.kr_csv:
        ingest_kr(args.kr_csv)
    if args.merge:
        merge()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
