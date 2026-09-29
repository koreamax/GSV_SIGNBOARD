# 간판 업종 태깅 기준 (v2, 2026-09-30)

`artifacts/gt/tagging_gt.csv` 의 `tag` 를 정하는 기준입니다. RAG 점검(`vlm/rag_audit.py`)에서 같은 가게의 정답 태그와
OSM 태그가 다른 사례가 반복돼(브루클린 오답 33건 중 12건), 경계가 흔들리는 업종을 OSM 위키 정의에 맞춰 문서로
고정합니다. 이 문서를 먼저 확정한 뒤, **해당 업종군에 속한 정답 전부**를 이 기준으로 다시 봅니다.

## 0. 원칙

1. **근거는 간판뿐입니다.** 간판 crop 에 보이는 글자·그림·가게 외관과, 상호명이 누구나 아는 업종을 가리키는 경우(체인
   브랜드 등)만 씁니다. 모델 예측·RAG 힌트·검색 결과는 보지 않습니다(평가 오염 방지).
2. **어휘는 그대로 42종**입니다. 새 태그를 만들지 않습니다. OSM 에 더 맞는 태그가 어휘에 없으면 아래 "어휘 밖 대응"을 따릅니다.
3. **OSM 위키 정의를 따릅니다.** 인용은 2026-09-30 에 확인한 wiki.openstreetmap.org 의 해당 Tag 페이지입니다.
4. **판단 근거가 없으면 `eval_tag=0`**(평가 제외)입니다. 억지로 고르지 않습니다.
   OSM 에 맞는 태그가 42종 어휘에 없고 가까운 대체도 없을 때도 `eval_tag=0` 입니다(예: 주류 판매점 `shop=alcohol`).
5. **간판 하나에 가게가 여럿**이면, crop 안에서 **가장 크게 적힌 상호**의 업종을 씁니다. 크기가 비슷하면 `eval_tag=0`.
6. 바꾼 태그는 전부 변경 기록에 남기고, 표는 변경 전·후 값을 둘 다 보고합니다.

## 1. 식료품 가게 — `shop=convenience` vs `shop=supermarket`

OSM: convenience 는 "A small local shop carrying a variety of everyday products", 미국에서는 bodega · corner store 로도
불리고 mini-mart 도 같은 것으로 봅니다. supermarket 은 "A large shop for groceries and other goods, including meat and
fresh produce", 구분 기준으로 "large enough to use trolleys" 를 제시합니다.

| 간판 단서 | 태그 |
|---|---|
| Deli, Grocery, Bodega, Mini Market, Candy & Grocery, 99¢ · Discount Store(식료품 위주), 편의점 체인(7-Eleven, GS25, CU) | `shop=convenience` |
| Supermarket, Food Market 체인(C-Town, Key Food 등), 이마트·홈플러스·롯데마트, 여러 칸을 쓰는 대형 식료품점 | `shop=supermarket` |
| "마트"·"Market" 만 있고 규모 판단이 안 됨 | 가게 폭이 한 칸이면 `convenience`, 두 칸 이상이면 `supermarket` |

- Deli 가 샌드위치·핫푸드만 파는 곳이 분명하면 `amenity=fast_food` 입니다(OSM `shop=deli` 는 어휘 밖).

## 2. 미용 — `shop=hairdresser` vs `shop=beauty`

OSM: beauty 는 "A non-hairdresser beauty shop, spa, nail salon, etc." 이고, "Shops which are largely given over to
haircuts and hair styling related services should be tagged shop=hairdresser even if their name suggests 'beauty'".

| 간판 단서 | 태그 |
|---|---|
| Hair, Barber, 미용실, 헤어, 이용원, Salon(머리 그림) | `shop=hairdresser` |
| Nail, 네일, Spa, 피부, 에스테틱, 왁싱, 속눈썹, Tanning, Brow | `shop=beauty` |
| "Beauty Salon" 만 있고 서비스 단서 없음 | `shop=hairdresser` (위 OSM 문장 — 이름보다 주 서비스) |
| 화장품 · Beauty Supply(판매점) | `shop=beauty` (OSM 은 `shop=cosmetics` 지만 어휘 밖) |

## 3. 음식점 — `amenity=restaurant` · `fast_food` · `cafe` · `bar` · `shop=bakery`

OSM: restaurant 는 "sells full sit-down meals with servers", fast_food 는 "A place which offers self-service and take-away
food ... typically paid for at the counter prior to consuming", cafe 는 "informal place with sit-down facilities selling
beverages and light meals", bar 는 "sells alcoholic drinks to be consumed on the premises ... usually do not sell food to be
eaten as a meal", bakery 는 "normally bake fresh bread on the premises". 두 기능을 겸하면 "decide the primary purpose".

| 간판 단서 | 태그 |
|---|---|
| 한식당·국밥·고기·찌개·일식·중식당·양식당·뷔페·Restaurant, Grill(테이블 식사) | `amenity=restaurant` |
| 버거, 피자(조각·포장), 샌드위치·Sub, 토스트, 도넛, 부리토·타코, 중식 Take-out, 치킨 **포장·배달 전문**, Dunkin' 같은 카운터 체인 | `amenity=fast_food` |
| 커피전문점, Café, 디저트 카페, 주스·스무디 바(비알코올) | `amenity=cafe` |
| 호프, 이자카야, 포차, 주점, Bar, Pub, Lounge, Wine bar — **술이 주 목적** | `amenity=bar` |
| 치킨호프 · 치킨&맥주 | 간판에서 더 크게 적힌 쪽: "호프"·"Beer" 가 주면 `bar`, "치킨" 이 주면 `fast_food`(포장 단서) 또는 `restaurant` |
| 빵집, 제과점, Bakery, Boulangerie, 베이커리 체인 | `shop=bakery` |
| Bagel 가게 | 카운터·포장 위주면 `fast_food`, 직접 굽는 빵 판매가 주면 `bakery` |

- "Juice Bar" 는 이름에 bar 가 있어도 술을 팔지 않으면 `cafe` 입니다(OSM bar 정의는 알코올).
- `amenity=pub` 은 어휘 밖이라 `bar` 로 씁니다.

## 4. 사무실 · 기타

| 간판 단서 | 태그 | 비고 |
|---|---|---|
| 공인중개사, 부동산, Realty, Real Estate | `office=estate_agent` | |
| 세무·회계·법무·변호사·노무·Legal Services·Insurance | `office=company` | OSM 은 `office=tax_advisor`·`lawyer`·`insurance` 이지만 어휘 밖 |
| 건물명 간판(○○빌딩) | `office=building` | 기존 어휘 유지 |
| 약국, Pharmacy, Drug Store | `amenity=pharmacy` | |
| 노래방, 코인노래 | `amenity=karaoke_box` | |
| 주류 판매점(Liquor, Wine & Spirits) | 평가 제외(`eval_tag=0`) | OSM `shop=alcohol`, 어휘 밖 · `bar` 는 마시는 곳이라 아님 |
| PC방 | `amenity=internet_cafe` | |

## 5. 재검토 범위와 절차

1. 대상: 정답 태그가 1~3절의 업종군(convenience · supermarket · hairdresser · beauty · restaurant · fast_food · cafe ·
   bar · bakery)인 **모든** 행(eval_tag=1, 256건). 모델이 맞힌 것도 똑같이 봅니다 — 틀린 것만 고르면 기준이 모델
   쪽으로 기웁니다.
2. 도구: `data/review_tag_v2.py` — 간판 crop, 상호명, 현재 태그만 보여주고 모델 예측은 보여주지 않습니다.
3. 결과: `artifacts/gt/tagging_review_v2.json`(변경 기록) → 확정 후 `tagging_gt.csv` 에 반영하고, 반영 전 파일은
   `tagging_gt_v1.csv` 로 보존합니다.
4. 태깅 표(표 5 태깅 행, 연쇄 태깅)는 v1·v2 기준 값을 모두 보고하고, 태깅 재실행 없이 저장된 예측으로 재채점합니다
   (`eval_e2e_tagging.py --rescore`).
