# GSV → 간판 탐지 → OCR → LLM 의미 보정·태깅 파이프라인

Google Street View(GSV) 거리 영상에서 **간판(signboard)을 탐지**하고, 탐지된 간판 영역의 **텍스트를 인식(OCR)**한 뒤, 그 결과를 **LLM으로 보정·의미 태깅(name/amenity)**하여 OSM 속성으로 환원하는 **3-모듈 파이프라인**의 연구 코드입니다. 탐지 단계는 **YOLO26x를 주 모델**로 하고 YOLOv5x·YOLO11x(버전 비교) 및 Faster R-CNN / EfficientDet(베이스라인)과 동일 5-Fold로 비교하며, 인식 단계는 **TrOCR · EasyOCR · PaddleOCR 세 엔진을 모두 간판 도메인에 미세조정**하여 비교하고, **3-way 앙상블**(박스 단위 confidence 선택)로 결합합니다. 탐지 모델은 **5-Fold 교차검증**으로, OCR은 도메인 미세조정 후 3개 지역(강남·브루클린·수원)에서 라인 매칭 기반으로 평가했습니다.

```
GSV 영상 → ①간판 탐지(Vision) → ②텍스트 인식(OCR) → ③LLM 의미 보정·태깅 → OSM 속성(name, amenity)
```

> 🔎 **저장소 범위:** 본 저장소는 3모듈 전부를 다룹니다 — ①탐지 + ②OCR 구현·학습, **③의미 태깅은 로컬 VLM(Gemma 4 31B, Ollama 4-bit)으로 구현·측정 완료**(D36, 4.13). 태깅 정답 데이터(`artifacts/gt/tagging_gt.csv`, 411크롭 검수)도 본 저장소에서 구축했습니다. 논문 본문의 태깅 수치(81.5/90.2/73.4)는 외부 GPT-4 결과로 본 측정과 프로토콜이 다릅니다.

> ⚙️ 이 저장소는 단독 프로젝트가 아니라, 아래 **0장**에 설명된 논문(_A Modular OpenStreetMap Enhancement Framework for Sustainable Smart Cities_)의 **리비전(보완) 실험**을 담당합니다. 즉 본 학습의 최종 산출물은 논문 리뷰어 지적을 해소하고 `Revised materials.docx`의 **표 1·표 3을 정량 데이터로 채우는 것**입니다.

---

## 0.0 저장소 구조 (2026-09-17 재편)

루트에 흩어져 있던 스크립트 100개를 역할별 폴더로 묶었습니다. 모든 스크립트는 **저장소 루트에서** `.venv/Scripts/python.exe <폴더>/<스크립트>.py` 로 실행하며,
내부 경로(`artifacts/`, `output/`, `external/`)는 루트 기준으로 해석됩니다(`Path(__file__).resolve().parents[1]`). 아래 본문의 스크립트 이름은 재편 전 이름 그대로이므로
이 표로 위치를 찾으면 됩니다.

| 폴더 | 내용 | 대표 스크립트 |
|---|---|---|
| `data/` | GT 변환·크롭 생성·분할(fold) 구축·누수 감사·검수 UI | `audit_split_leakage.py`, `build_line_crops.py`, `build_grouped_folds.py`, `prep_text_det*.py` |
| `detection/` | 간판/텍스트 박스 탐지 학습·평가 (YOLO, Faster R-CNN, EfficientDet, k-fold·hold-out) | `train_*_kfold.py`, `eval_det_unified.py`, `eval_text_holdout.py`, `e2e_det_boxes.py` |
| `ocr/` | PaddleOCR·EasyOCR·TrOCR 인식기 학습·A/B, OCR 사전(snap), 초기 OCR 파이프라인 | `train_textinthewild_ocr.py`, `train_paddle_v4_spacecat.py`, `run_ocr_only.py`, `build_ocr_db.py` |
| `str_baselines/` | OCR 비교군(§4.19): 데이터 준비, SVTRv2·PARSeq 초기값, 엔진별 인식 워커, in-domain 채점, 결과표 | `prep_str_baselines_data.py`, `*_rec_worker.py`, `eval_str_indomain.py`, `make_ocr_baselines_report.py` |
| `pipeline/` | 배포 파이프라인(라인 병합 OCR)과 GSV 채점, 연쇄(e2e) 평가, FP 필터·앙상블 실험 | `run_ocr_line.py`, `eval_ocr_v2.py`, `eval_e2e_cascade.py`, `ocr_ensemble_select.py` |
| `vlm/` | 의미 태깅(VLM/LLM) 평가, RAG 상호 사전·검색 | `eval_vlm_tagging.py`, `rag_retrieve.py`, `build_poi_db.py` |
| `configs/` | PaddleOCR 인식기 학습 설정(yml) | `paddle_signboard_rec_v5_lines.yml` |
| `scripts/` | 배치 러너(sh) — 루트에서 `bash scripts/xxx.sh` | `run_ocr_baselines_all.sh`, `run_grouped_retrain.sh` |
| `external_patches/` | 외부 클론(EasyOCR·OpenOCR·parseq)에 가한 수정 diff + 추가 파일 | `README.md`, `UPSTREAM.txt` |
| `artifacts/` (대부분 git 제외) | GT·평가 CSV(`gt/`), 비교군 결과(`str_baselines/`), 학습 데이터·예측·로그 | `gt/ocr_eval_v2_summary.csv`, `str_baselines/ocr_baselines_tables.md` |

루트의 `data.yaml`(YOLO 데이터셋 정의)과 `*.pt`(Ultralytics 기본 가중치, git 제외)는 경로 의존성 때문에 그대로 둡니다.
워커를 지정하는 인자는 루트 기준 상대경로로 씁니다(예: `--worker str_baselines/openocr_rec_worker.py`).

---

## 0. 연구 배경 — 논문 리비전과의 연결

### 0.1 대상 논문

| 항목 | 내용 |
|------|------|
| 제목 | **A Modular OpenStreetMap Enhancement Framework for Sustainable Smart Cities** |
| 저자 | Danha Kim, In-Nea Wang, Junho Jeong (동국대학교) |
| 투고처 | Applied Sciences (MDPI), 2025 — 리비전 진행 중 |
| 프레임워크 | OSM 건물 기하 → **방향 인식 SVI 수집**(Google Street View) → 3개 핵심 모듈 |
| 핵심 모듈 | ① 간판 탐지(Vision) → ② 텍스트 인식(OCR) → ③ LLM 의미 태깅(name/amenity) |
| 목적 | 거리뷰 영상으로 OSM 건물의 **누락된 의미 속성(이름·용도)을 자동 복원** |

**논문 원본의 한계(리뷰어가 지적한 부분):** 초판은 간판 탐지에 **YOLOv5 단일 모델**만 사용하고, 검증도 약 250개 건물에 대한 **정성적(qualitative) PoC** 수준이었습니다. 모듈별 성능은 단일 수치(YOLOv5 mAP@0.5 87.2%, OCR 정확도 78.1%, GPT-4 F1 83.1%)로만 제시되었고, **모델 간 비교·지역별 정량 비교가 부재**했습니다.

### 0.2 리뷰어 지적사항 (3인) → 본 저장소가 담당하는 부분

| # | 리뷰어 지적 (요약) | 본 비전·OCR 작업의 대응 |
|---|-------------------|------------------------|
| R1 | "**최신 YOLO 버전**을 시도했는가? 버전 간 성능을 비교하라" | 동일 5-Fold·**동일 평가 프로토콜**로 YOLOv5x(0.814) vs **YOLO26x(0.832)** — 버전 업그레이드 효과 +1.8%p 확인. 다만 통합 측정에서는 Faster R-CNN(0.874)이 최고 (4.3 / D26) |
| R1 | 방법론·**파라미터를 명시**해 재현성 확보 | 전 모델 하이퍼파라미터 표 (3장) |
| R3 | "**탐지 모델·OCR 엔진별 성능 비교표가 없다**" | 표 1: 다중 모델 벤치마크 (0.4) |
| R3 | "3개 지역 사례가 **mAP·OCR 정확도 차이를 정량화하지 않음**" | 표 3: 지역별 정량 측정 (0.4) |
| R3 | "**모듈 교체(확장성)**를 성능으로 검증하지 않음" | 탐지 3종·OCR 3종 교체 실험 (4.3) |
| R3 | "모듈 간 **결합 관계**(탐지 정확도 → OCR 영향) 불명확" | 탐지→크롭→OCR 연쇄 성능 (표 3) |

> 즉, 리뷰어들은 "단일 모델·정성 검증"을 "**다중 모델 정량 벤치마크 + 지역별 정량 비교 + 재현 가능한 파라미터**"로 끌어올리라고 요구했고, 이 저장소의 5-Fold 학습·평가가 정확히 그 데이터를 생산합니다.

### 0.3 최종 목표 — `Revised materials.docx` 채우기

리비전 문서에는 채워야 할 표 2개가 비어 있습니다. 본 저장소의 실험 결과가 이 칸들을 채웁니다.

**[표 1] 모듈별 모델 성능 비교** (각 단일 모듈의 독립 성능)

| Module type | Model | Metric | 현재 상태 (본 저장소 측정값) |
|-------------|-------|--------|------------------------------|
| Signboard Detection | **YOLOv5x** | mAP@0.5 | ✅ **89.1%** (Ultralytics val, 동일 5-Fold, yolo26x와 동일 설정) |
| Signboard Detection | Faster R-CNN | mAP@0.5 | ✅ **87.3%** (AP@0.5, 재학습·5-Fold full convergence) |
| Signboard Detection | **Faster R-CNN / YOLO26x** | mAP@0.5 | ✅ **통합 프로토콜: FRCNN 87.4% > YOLO26x 83.2%** (저장 best 체크포인트 + 단일 AP@0.5, 동일 5-Fold — 4.3·D26). 구 표기 YOLO26x 90.1%는 평가도구·peak 편향이 섞인 값 |
| Signboard Detection | EfficientDet (D0) | mAP@0.5 | ✅ **80.5%** (AP@0.5, **동일 5-Fold full convergence**) |
| OCR | TrOCR (signboard_v3 재학습, **무누수 test**) | in-domain test exact / CER | ✅ **exact 72.9% / CER 0.149** (구 84.1%는 test 누수 프로토콜 — 4.4 참고) |
| OCR | EasyOCR (CRNN, signboard_v3 재학습) | in-domain test exact / CER | ✅ **exact 71.6% / CER 0.153** (무누수 test) |
| OCR | PaddleOCR (PP-OCRv5_rec, signboard_v3 재학습) | in-domain test acc / CER | ✅ **acc 86.0% / CER ~0.073** (개별 최강, 무누수 test) |
| OCR | **하이브리드: OCR + VLM 교정 (배포: Gemma 4 31B fixcand5)** | GSV exact / WAR / CER | ✅ **exact 80.9% / WAR 0.766 / CER 0.122** (전역, 재라벨 GT — OCR 파이프라인(69.9%) 위에 이미지+5모델 후보 그라운딩 교정, 4.14) |

**[표 3] 지역별 연속 성능** (한 지역에서 탐지→OCR→태깅 연쇄 성능)

OCR은 **검출기 union + 라인 병합 Paddle(배포, run22 — D20·D21·D23)** 기준 exact/CER, 그리고 oracle best-of-3 상한을 함께 제시합니다 (CRAFT∪DB 검출 → 라인 스트립 병합 → 통째 인식 — 강남·수원 **v5_lines**(실라인 학습, 4.8), 브루클린 **영어 전용 en_PP-OCRv5** × **재라벨 GT**, `eval_ocr_v2.py --mask-phone --en-only-regions brooklyn`, 라인 매칭 micro):

| Region | Language Env. | Buildings | Signboard mAP@0.5 (%) | OCR 앙상블 exact / CER | OCR oracle CER (상한) | Semantic Tag Acc. (%) |
|--------|---------------|-----------|----------------------|------------------------|-----------------------|------------------------|
| Gangnam (Seoul) | Korean/English/Japanese | 100 | **0.853** | 82.5% / 0.147 | 0.098 | **90.4** (자체측정) |
| Brooklyn (New York) | English **(영어 전용 엔진·채점)** | 80 | **0.887** | **88.9% / 0.073** | **0.068** | **72.0** (자체측정) |
| Suwon (Ingye-dong) | Korean/Mixed Fonts | 70 | **0.899** | 70.2% / 0.206 | 0.189 | **88.2** (자체측정) |

> ⚠️ **위 표는 전부 "GT 크롭 기준", 즉 사람이 정확히 잘라준 간판에서의 성능 = 상한입니다.**
> 사진을 넣었을 때의 실제 성능(연쇄)은 **OCR 54.4% · 태깅 66.6%** 로 크게 낮습니다 (4.16).
> 논문에는 **두 표를 반드시 함께** 실으세요. OCR 열은 Gemma 4 31B fixcand5(4.14), 태깅 수치는 Gemma 4 31B vlmrag +
> 42종 완전일치 + 기권 금지 조건이며(4.13), 논문 원본의 81.5/90.2/73.4는 GPT-4 기반이라 프로토콜이 다릅니다.

> ✅ **OCR 지역별 측정 완료** (4.4, 재라벨 GT · 라인 병합 배포 기준). 지역별 oracle CER 난이도 순서 **brooklyn(0.080) < gangnam(0.109) < suwon(0.189)** — 영어권이 가장 쉽다는 논문 정성 서술과 일치합니다. 브루클린은 **채점만이 아니라 인식 모델 자체를 영어 전용**(PaddleOCR en_PP-OCRv5, per-box 보조 엔진은 english_g2·base-printed)으로 교체해 측정했습니다 (D15·D20).
> ✅ **탐지 지역별 측정 완료** (YOLO26x, **out-of-fold** — 각 사진은 그 사진을 학습에서 제외한 fold 모델로만 예측, 4.12). 의미 태깅 정확도(81.5/90.2/73.4)는 논문 Table 3 값으로 **구 OCR 기준**이라 재측정 대상입니다(T6).
>
> 🔎 **탐지·OCR 난이도가 엇갈립니다:** 수원은 **탐지가 가장 쉬운데(0.899) OCR이 가장 어렵습니다(58.6%)**. 즉 수원의 병목은 간판을 *찾는* 것이 아니라 *읽는* 것(양각·캘리그래피·저대비)이며, 이는 4.7의 실패 유형 진단과 일치합니다. 강남은 반대로 탐지가 가장 어렵습니다(0.853, 밀집 다중 간판 파사드).

### 0.4 진행 현황 — 완료된 부분 / 해야 할 부분

#### ✅ 완료된 부분 (한 부분)

| # | 작업 | 결과 | 근거 |
|---|------|------|------|
| D1 | **YOLO11x 5-Fold 벤치마크** (버전 비교 baseline, 이후 YOLO26x로 대체) | mAP@0.5 **0.862** (Ultralytics val, 이어학습) | 4.1 |
| D2 | **Faster R-CNN 베이스라인** 5-Fold 학습 | mAP@0.5 **0.864** (COCOeval, 재현율 0.894) | 4.2 |
| D2b | **YOLOv5x 5-Fold 학습** (yolov5s 대체, yolo26x와 동일 설정) | mAP@0.5 **0.891** (fold별 0.943/0.857/0.844/0.885/0.926) | 4.3 |
| D3 | **TrOCR·EasyOCR·PaddleOCR 3종 간판 도메인 미세조정** | in-domain Paddle 85.0% > TrOCR 84.1% > EasyOCR ~70% | 3.4~3.6 |
| D4 | **PaddleOCR 미세조정 + 3-way 앙상블 통합** | (구 GT) 앙상블 CER 0.632 → 재라벨 후 **0.416** | 3.7·4.4 |
| D5 | **3개 지역 OCR 라인 매칭 평가** (강남·브루클린·수원) | 표 1 exact/CER, 표 3 OCR 열 채움 | 4.4 |
| D6 | **평가기준 완화(포함 기반) 1차 측정** (구 GT 시절) | 20% → 단어 ≈50% / 간판 ≈62% — 이후 D11로 대체 | 4.5 |
| D11 | **GT 전면 재라벨링 + 평가 프로토콜 정비** | 463크롭 전수 재검수(제외 52), 라인분리 버그 수정, `###` don't-care·전화번호 마스킹·브루클린 영어 전용 채점 도입 → 앙상블 **exact 45.9% / contain 70.6% / CER 0.416** | 4.4 |
| D12 | **TrOCR 무누수 재학습 (signboard_v3)** | train/val/test 80/10/10(소스단위)+증강. 핵심 버그 발견: base 토크나이저가 한글 음절을 `<unk>`로 손상 → 바이트레벨 BPE 이식으로 해결. **test exact 72.9% / CER 0.149** (13에폭). GSV run9: 앙상블 45.8%로 유지(Paddle 지배적) | 3.4·4.4 |
| D13 | **EasyOCR·PaddleOCR v3 재학습 — 3엔진 프로토콜 통일** | 동일 v3 분할·공식 사전학습 시작(무누수). in-domain test: **Paddle 86.0%/CER 0.073 > TrOCR 72.9% > EasyOCR 71.6%**. GSV run10(전 엔진 v3): 앙상블 exact 45.4%/CER 0.415 | 3.4~3.6·4.4 |
| D14 | **OCR 사전(DB) 구축 + 스냅 1차 측정 (T5)** | OSM POI(지역별 bbox) + train 어휘, 누수 금지 규칙 명문화(`artifacts/ocr_db/DB_RULES.md`). 스냅 효과: **WAR +2.0%p**(0.230→0.251), CER 중립, exact −0.5%p — 한글 퍼지 금지 규칙 도입. 추가 이득은 선택규칙 보정(T7)에 있음 | 4.6 |
| D15 | **브루클린 영어 전용 엔진 재측정 (run12)** | 영어권 브루클린은 **인식 모델 자체를 영어 전용으로 교체**(EasyOCR english_g2 · TrOCR base-printed · Paddle en_PP-OCRv5_mobile, `--easyocr-langs en`) → 앙상블 **exact 46.7→51.8% / CER 0.383→0.369**, TrOCR exact 16.6→46.2%(한글 v3 모델의 영어 약점 해소). 사전 스냅은 브루클린서 순손실(exact −3.6%p)이라 미적용. 전역 앙상블 **exact 47.0% / CER 0.408** | 4.4 |
| D16 | **사전 스냅 손상 규명 + 가드 2차 (T5 보완)** | 손상 2대 원인 차단: ① 퍼지가 예측 **끝 1~2자를 삭제**하는 방향 금지(NAILS→NAIL·SALES→SALE·ollehO→olleh는 전부 손상, 복원 방향 ELEVE→ELEVEn은 허용) ② exact-hit 표기 치환 시 **구두점 추가 금지**(Dominos→Domino's — strict 채점에선 지표 중립, 출력 위생). 강남 exact 44.3→**44.8%**(올레 라인 복원), 수원 ±0, 전역 **47.0→47.1%**. 브루클린은 가드 후에도 −2.0%p(51.8→49.7): 잔여 손상은 **실단어→사전 이웃 치환**(SPRINT→SPRING·WINES→Wine-O·E-BIKE→BIKE·LEACH→Peach)으로 이득 케이스(istand→ISLAND·duice→JUICE)와 같은 d=1 치환이라 편집거리 규칙으로 구분 불가 → **raw 유지**. 잔여 exact 개선은 T7(사전을 후보 선택 가점으로) | 4.6 |
| D35 | **OCR 비교군 확장 — PaddleOCR 동급 인식기 (PARSeq·SVTRv2 미세조정, Tesseract·Surya 기성)** | 논문 비교군 조사(범용 엔진/학술 STR/VLM 3부류) 후 Paddle v5_lines 와 같은 75,612 크롭·같은 분할로 SVTRv2-B(Union14M 초기값)·PARSeq(공식 초기값) 미세조정, GSV 는 검출·라인 병합 동일 조건에서 인식기만 교체. **GSV line exact: SVTRv2 70.6% ≈ PP-OCRv5 69.9%(3-way vote) > PARSeq 62.0% ≫ Surya 40.2% > EasyOCR 38.5% > TrOCR 33.6% > Tesseract 30.6%**; in-domain 은 SVTRv2 가 명확히 우위(단어 94.2 vs 87.2%). Tesseract 는 같은 데이터로 미세조정해도 32.9%(→ 아키텍처 한계 확인). → PaddleOCR 선택 근거를 '동급 최상위 + 배포 편의'로 서술. WER 지표 추가. 함정: OpenOCR 배치 패딩이 CTC 끝글자 중복 유발(1장씩 추론으로 해결) | 4.19 |
| D34 | **텍스트(word) 박스 탐지 hold-out — 4 파이프라인 동일 계열 비교** | OCR 학습데이터 전체(27,132장/78,110 박스, signboard_v3 소스 분할 상속, 섞임 감사 통과)로 4모델 재학습(58h) 후 test 2,706장 단일 채점기 AP@0.5: **YOLO26x 0.822 > YOLOv5x 0.809 > FRCNN 0.715 > EffDet 0.672** (5k 5-fold 대비 +0.07~0.09, 순위 동일). 간판(FRCNN≈YOLO26x)과 달리 텍스트에서는 YOLO 계열이 0.1 이상 앞서 **두 단계 모두 최상위는 YOLO26x 뿐** — 배포 선택 근거. 배포 OCR 의 라인 검출(CRAFT∪DB)과는 단위가 달라 비교 안 함 | 4.18 |
| D33 | **탐지 fold 누수 규명 + group-aware 재분할·4모델 재학습** | "OOF 가 제일 높다"는 지적에서 출발. 증강→분할 누수는 없었고(파일 증강 0), **같은 가게 간판이 train/val 에 걸친 누수**(val 20.8%, 강남 31%)를 GT 텍스트 대조로 발견. 가게 연결성분 단위 group-aware 5-fold 로 재분할(fold 간 공유 0) 후 4모델 재학습(37.6h). **누수는 mAP 를 부풀리지 않았음**(전 모델 +0.003~+0.040). 대신 **FRCNN(0.878) ≈ YOLO26x(0.872) 로 순위 격차 소멸** — 0.05 미만 차이는 우열 아님. group-aware 분할을 정본으로 채택 | 4.17 |
| D31 | **연쇄(end-to-end) 측정 — 파이프라인 실제 성능** | 지금까지 OCR·태깅 평가가 전부 **GT 크롭** 기준이라 탐지 손실이 빠져 있었음. OOF 탐지 박스로 동일 전처리 크롭을 다시 만들어 3모듈 전체 재측정: **OCR 연쇄 54.4%** (GT 크롭 80.9%) · **태깅 연쇄 66.6%** (GT 크롭 83.5%) — D36 재측정값. 곱셈 추정(65.5%)보다 실측이 22%p 낮아 **난이도 상관을 실증**. **VLM 하이브리드가 박스 품질 저하를 흡수**(TP 위 정확도가 GT 크롭 기준을 역전: 52.1→64.0 vs 57.8). **허위 POI: FP 124건 중 112건(90%)에 업종 태그가 붙음** — 지도 구축에서 precision이 결정적인 이유. 다음 투자처는 탐지 recall(25.6%p 손실의 전부) | 4.16 |
| D30 | **하이브리드 검색 (lexical + dense RRF)** | bge-m3 임베딩을 현행 lexical 검색에 RRF 융합(`rag_hybrid.py`). 오프라인 hit@5 **강남 78.6→88.1 · 브루클린 91.4→94.3 · 수원 55.0→75.0**, dense 단독은 어디서도 lexical에 못 미치지만 융합은 둘 다를 상회(4.14와 동형의 시너지). 종단 태깅 69.4→**71.1%**, 다만 **in-DB −3.9%p / off-DB +3.7%p로 개선되는 층이 바뀜** — 신규 POI 발굴이 목적이므로 채택. 부수: 약국 451건 오태깅 수정은 예측 1건만 바꿈(간판 텍스트가 업종을 결정하면 힌트는 무시됨) | 4.15 ⑥⑦ |
| D37 | **상용 API 비교군 — NAVER CLOVA OCR General** | "상용 API 를 그냥 쓰면 되지 않는가"에 답하기 위해 추가. 검출·라인 병합 고정, 인식기만 교체(다른 워커와 동일 IO). 제로샷 **GSV 57.9%** / in-domain 실라인 67.3% / 단어 66.5% — 기성 엔진 최고(Surya 40.2)를 크게 앞서지만 **미세조정 SVTRv2(70.6/75.3/94.2)에는 미달**. API 10,734건. 유료 API 안전장치(체크포인트·호출 상한·사전 점검) 포함 | 4.19 |
| D36 | **VLM 32B급 교체 + 교정 방식 재정의 (fix / fixcand) + 태깅 재측정** | gemma3:12b 기반 결과(구 D27~D29) 전부 폐기. Gemma 4 31B·Qwen3-VL-32B-Instruct(Ollama 4-bit, GPU/RAM 오프로드)로 멀티모달 post-OCR 교정 fix / 후보 그라운딩 fixcand·fixcand5(+SVTRv2·PARSeq) 와 문헌 기준선 fixtext(텍스트 전용) 측정. **GSV exact 69.9 → fixtext 72.5 → fix 79.9 → fixcand5 80.9%**(Gemma 4; Qwen 79.7) — 교정 이득의 3/4가 이미지에서. 태깅 vlmrag **83.5%**(Gemma 4) / 80.0%(Qwen). 연쇄: OCR 54.4% · 태깅 66.6%. 배포 = Gemma 4 fixcand5 | 4.13·4.14·4.16 |
| D26 | **탐지 모델 통합 프로토콜 재측정 (T2 완료) — 순위 역전** | 4모델을 **저장 best 체크포인트 + 단일 AP@0.5 + 동일 5-Fold**로 재측정(`eval_det_unified.py`, 예측 캐시). **FRCNN 0.874 > YOLO26x 0.832 > YOLOv5x 0.814 ≈ EffDet 0.812** — 구 표의 YOLO 우위는 ① 평가도구 격차(~0.05) ② Ultralytics가 fitness 기준으로 best.pt를 골라 생긴 peak 보고의 합작이었음. 비-YOLO 수치는 기존값 재현(FRCNN 0.8736 vs 0.8734, EffDet 0.8042 vs 0.8050)으로 평가기 검증. thr 0.0/0.05 양쪽에서 순위 동일. 교란: 모델별 입력 해상도 상이 | 4.3 |
| D25 | **지역별 탐지 mAP@0.5 산출 (T4 완료)** | GSV 사진이 곧 5-Fold 학습셋이라 **out-of-fold**(각 사진 = 그 사진을 held-out한 fold 모델로만 예측)로 측정, 도구는 표 2와 동일한 Ultralytics `val()`. **강남 0.853 / 브루클린 0.887 / 수원 0.899, 전체 0.880** (fold별 평균 0.8785와 정합). 표 2의 0.9006은 **에폭 중 최고치**라 낙관 편향 — 동일 가중치의 정직한 값은 0.8785. **수원은 탐지 최고·OCR 최저**로, 병목이 인식 쪽임을 교차 확인 | 4.12 |
| D24 | **기하 파라미터 보정 + 라인 레벨 다수결 (run24, 공식)** | ① 크롭 여백이 **스트립 폭 비례**라 긴 라인일수록 옆 간판을 끌어오던 문제 — pad 0.12→**0.04** (**+2.8%p**). ② 라인 그룹핑 y_tol 0.06→**0.04**: 전멸 라인의 43%가 과잉 병합(`VIP`+`노래방`)이었음 (**+0.5%p**). ③ 한글권 v5·v4·zero-shot **라인 레벨 다수결**(767중 35라인 교체, **+0.6%p**) — 후보가 모두 같은 스트립을 읽으므로 합의가 유효. 전역 **exact 66.0→69.9% / CER 0.174→0.163 / recall 90.2→91.1%** | 4.5·4.9 |
| D23 | **검출기 union(CRAFT∪DB) 공식 채택 (run22)** | CRAFT 박스와 겹치지 않는 PaddleOCR DB 박스를 추가(537개) → 전 지역 개선: 강남 64.2→**66.5**, 수원 54.7→**56.4**, 브루클린 72.4→**74.4%**, 전멸 라인 82→58. 전역 **exact 66.0% / CER 0.174 / recall 90.2%**. **union일 때만** 이득 — 두 검출기의 실패 모드가 상보적 | 4.9 |
| D21 | **실제 라인 데이터 학습 v5_lines — 공식 채택 (run21)** | 원천 단어 박스로 실라인 크롭 13,148개 구축(`build_line_crops.py`, 96px 캡·무누수 분할 상속) + 단어 혼합 학습. **합성 concat 공백 능력은 실라인에 미전이**(v4 공백재현 15.8% ↔ v5 **85.9%**, 실라인 exact 7.9→**55.7%**, CER 0.124→0.108). GSV: exact/CER 동등(변동 내), **WAR 강남 0.302→0.610·수원 0.253→0.498** → 공식 전역 **exact 64.0% / CER 0.204 / WAR 0.440→0.605** | 4.8 |
| D19 | **라인 병합 인식 실험 — per-box 병목 발견** | 검출 박스를 라인 스트립으로 병합해 Paddle에 통째 인식: **전역 exact ≈65% / CER ≈0.21** (per-box 47.6%/0.41, T7b 앙상블 48.0% 대비 **+17%p**). 단어 박스 조각내기가 최대 병목이었음을 입증. v4(spacecat)는 이 구조에서 WAR +1.2~2.5%p (4.8 예측 적중) | 4.9 |
| D20 | **라인 병합 + v4 공식 채택 (run19/20)** | `run_ocr_line.py` 신설(강남·수원 v4_spacecat / 브루클린 en), **배포 = 라인 병합 Paddle 단독**: 전역 **exact 48.0→64.5% / CER 0.410→0.203 / recall 63.5→86.3%** (강남 65.6 / 수원 54.7 / 브루클린 72.4). 라인 병합 Paddle 단독이 3엔진 앙상블을 크게 앞서 **선택 없이 단독 배포**. 새 oracle 상한 69.4%/CER 0.151 | 4.4·4.7 |
| D18 | **RecConAug 무공백 결합 가설 A/B 검증 (v4_spacecat)** | 학습 증강 검수 환경(`review_train_aug.py`)에서 발견한 무공백 concat을 갭+공백 버전으로 고쳐 Paddle 재학습. **in-domain: 합성 2단어쌍 공백 재현율 0.0→99.8%, pair exact 0→68.9%, single 85.9→86.2%** — 가설 입증. GSV: WAR 강남+1.0·수원+0.8%p(공백 민감 지표), exact는 학습 변동 범위(강남+0.5/수원−2.3) — **v3b 동일 레시피 재학습이 수원서 동일 낙폭(−2.3%p, 43.6%)을 재현해 변동임을 실측 확정**. T7b 전역 48.0→47.3 → **공식 v3 유지**, v4는 라인 병합(4.9) 채택 시 권장 모델 | 4.8 |
| D17 | **per-box 재선택 앙상블 (run15)** | 크롭 단위 GT-free 재선택(`ocr_ensemble_select.py`): 후보 {3엔진+conf앙상블}에 in-domain 신뢰도 가중 합의 + 사전(DB) 가점 argmax → 전역 **exact 47.1→48.0%**. *(per-box 시대 기록 — D20 라인 병합 채택으로 대체됨)* | — |
| D7 | **YOLO26x 5-Fold 학습** (yolo11x·yolo26s 대체) | 학습 로그 기준 mAP@0.5 0.901(peak). **통합 프로토콜 재측정값은 0.832** (D26, 4.3) | 4.3 |
| D8 | **EfficientDet-D0 5-Fold full convergence** (patience 100) | AP@0.5 **0.805** (fold별 0.824/0.787/0.828/0.784/0.803) | 4.3 |
| D9 | **Faster R-CNN 5-Fold 재학습 full convergence** (patience 100) | AP@0.5 **0.873** (fold별 0.926/0.875/0.825/0.870/0.871) | 4.3 |
| D10 | **텍스트(word) 탐지 4모델 5-Fold** (Signboard 5000장, `*_text_kfold`) | YOLO26x **0.746** > YOLOv5x 0.737 > FRCNN 0.638 > EffDet 0.578 | 4.3.1 |

#### 🔜 해야 할 부분 (To-Do · 확장방안)

| # | 작업 | 분류 | 메모 |
|---|------|------|------|
| T2 | ~~전 탐지 모델 단일 프로토콜 통합 재측정~~ | **완료(D26)** | 저장 best 체크포인트 + 단일 AP@0.5로 통일 → **순위 역전: FRCNN 0.874 > YOLO26x 0.832 > YOLOv5x 0.814 ≈ EffDet 0.812** (4.3). 잔여: 입력 해상도가 모델마다 다름(960/1333/512) → 해상도 통제 비교는 별도 실험 |
| T4 | ~~지역별 mAP@0.5 분리 산출~~ | **완료** | out-of-fold + Ultralytics val()로 산출: 강남 0.853 / 브루클린 0.887 / 수원 0.899 (4.12) |
| T5 | ~~OCR 사전(DB) 결합~~ | **종결** (사전은 태깅 vlmrag 후보로만 사용, 4.13) | 문자열 스냅 방식은 채택 안 함(손상이 이득을 상쇄). **사전을 치환기가 아니라 VLM 후보 제공자로 바꾼 RAG 형태로 재도입** — 태깅에서 in-DB +10.8%p, off-DB 무변화 (4.15) |
| T6 | ~~③ 의미 태깅 단계 구현~~ | **완료(D36)** | 로컬 VLM(Gemma 4 31B, vlmrag)으로 구현·측정: tag exact **83.5%** (강남 90.4 / 브루클린 72.0 / 수원 88.2), 4.13 |
| T7 | ~~앙상블 선택 규칙 보정~~ | **종결** | 라인 레벨 3-way 다수결로 채택 완료(D24). 잔여 헤드룸: 배포 69.9% vs oracle 73.7% |

---

## 1. 연구 개요

| 항목 | 내용 |
|------|------|
| 목표 | 거리뷰 영상 → 간판 탐지 → 텍스트 인식 → LLM 의미 보정·태깅 (End-to-End) |
| 탐지 모델 | **YOLO26x** (주 모델), YOLOv5x · YOLO11x (버전 비교), Faster R-CNN (ResNet50-FPN), EfficientDet-D0 (베이스라인) |
| 인식 모델 | **TrOCR**(microsoft/trocr-small-printed 미세조정) · **EasyOCR**(CRNN 미세조정, en+ko) · **PaddleOCR**(korean PP-OCRv5_rec 미세조정) — **3-way 앙상블** |
| 의미 태깅 | GPT-4 등 LLM (논문 단계, **본 저장소 미구현** — 현재는 규칙 기반 후보정) |
| 검증 방식 | 탐지: 5-Fold CV (seed=42) / OCR: 도메인 미세조정 + 3개 지역 라인매칭 평가 (소스 이미지 단위 분할로 누수 방지) |
| 평가 지역 | 강남(Gangnam), 브루클린(Brooklyn), 수원(Suwon) — 지역별 각 100장 |
| 탐지 지표 | mAP@0.5, AP@0.75, Precision/Recall (IoU=0.5, conf=0.5) |
| 인식 지표 | CER (문자 오류율), WAR (단어 정확도), exact·recall·contain — 라인 매칭 기반 (`eval_ocr_v2.py`; `###` don't-care·`--mask-phone`·`--en-only-regions` 지원) |
| 태깅 지표 | F1-score (논문 Table 2: GPT-4 83.1%) |

### 파이프라인 데이터 흐름

```
GSV 이미지 (JPG + Labelme JSON)
   │
   ├─[json_to_gt_csv.py]──────────→ 폴리곤→사각형 GT CSV (artifacts/gt/)
   │
   ├─[one_click_finetune_yolo11.py]→ 공용 5-Fold 분할 생성 (artifacts/yolo11x_kfold/total_fold{0-4}/dataset)
   │                                 + YOLO11x 5-Fold 학습 (버전 비교) ─→ best.pt × 5
   ├─[train_yolo26x_kfold.py]──────→ YOLO26x 5-Fold 학습 (주 모델, imgsz 960) ─→ best.pt × 5
   ├─[train_yolov5x_kfold.py]──────→ YOLOv5x 5-Fold 학습 (버전 비교, 동일 설정)
   ├─[train_only_my_frcnn.py]──────→ Faster R-CNN 5-Fold 학습 (베이스라인)
   │
   ├─[run_yolo_only.py]───────────→ 간판 탐지 박스 CSV
   ├─[make_crops_from_gt_polygon.py]→ 원근 보정 간판 크롭 (회색조 + CLAHE)
   │
   ├─[train_textinthewild_ocr.py]──→ TrOCR 미세조정 (Text-in-the-Wild / Signboard)
   ├─[build_easyocr_trainer_data.py + external/EasyOCR/trainer]→ EasyOCR(CRNN) 미세조정
   ├─[PaddleOCR tools/train.py + export_model.py]→ PaddleOCR(rec) 미세조정 + inference export
   │
   ├─[run_ocr_only.py --box]──────→ EasyOCR(CRAFT) 박스검출 → 박스별 TrOCR/EasyOCR/PaddleOCR
   │                                 → confidence 선택 3-way 앙상블 (+ 규칙 기반 후보정)
   │     └─[paddle_rec_worker.py]── PaddleOCR 격리 subprocess (torch/paddle DLL 충돌 회피)
   │
   ├─[ ③ LLM 의미 보정·태깅 ]──────→ ⚠️ 미구현 (GPT-4 등으로 OCR 텍스트 → name/amenity)
   │
   ├─[eval_map_only.py / compute_map.py]→ 탐지 mAP 평가
   └─[eval_ocr_v2.py]─────────────→ OCR CER/WAR 평가 (라인매칭 · per-engine + ensemble + oracle)
```

---

## 2. 학습 데이터

### 2.1 탐지 학습 데이터 (간판 탐지)

GSV에서 수집한 3개 지역 거리뷰 영상에 Labelme로 간판 폴리곤을 라벨링했습니다.

| 지역 | 원본 이미지 | 라벨(JSON) | 간판 크롭 수 |
|------|------------|-----------|-------------|
| 강남 (Gangnam) | 100 | 100 | 151 |
| 브루클린 (Brooklyn) | 100 | 100 | 171 |
| 수원 (Suwon) | 100 | 100 | 141 |
| **합계** | **300** | **300** | **463** |

- COCO 변환 학습셋(`artifacts/yolo_ft_total/`): train **238장** / val **60장** (단일 클래스 `signboard`)
- 라벨은 N-점 폴리곤 → `cv2.minAreaRect()`로 4점 사각형(quad)으로 정규화하여 저장

### 2.2 OCR 학습 데이터

두 종류의 대규모 한국어/영문 텍스트 데이터셋을 사용했습니다.

| 데이터셋 | 이미지 수 | 텍스트 어노테이션 수 | 용도 |
|----------|----------|---------------------|------|
| **Signboard** (간판 특화) | 27,519 | 527,498 | 간판 텍스트 도메인 미세조정 |
| **Text-in-the-Wild** | 100,216 | 2,096,460 | 일반 자연 영상 텍스트 학습 |

학습 편의를 위해 크롭 단위로 여러 규모의 서브셋을 구축했습니다 (행 수 = 크롭 라벨 수):

| 학습 셋 (`artifacts/ocr_training/`) | 크롭 라벨 수 | 비고 |
|------------------------------------|------------|------|
| `signboard` | 20,000 | 간판 20K 프로브 |
| `signboard_full` | 78,104 | 간판 전체 |
| `textinthewild` | 80,183 | TIW 전체 |
| `textinthewild_10k` | 10,000 | 빠른 학습용 |
| `textinthewild_20k` / `_20k_probe` | 20K급 | 하이퍼파라미터 탐색 |
| `textinthewild_small` | 2,000 | 디버그용 |

- 검증 분할: **소스 이미지 단위 10%** (val_ratio=0.1) — 같은 원본 사진의 크롭이 train/val에 섞이지 않도록 누수 방지
- 전처리: bbox 패딩 4%, 최소 크롭 변 4px, 최대 텍스트 길이 64자

### 2.3 간판 크롭 전처리 (`make_crops_from_gt_polygon.py`)

탐지/GT 간판 영역을 OCR 입력으로 변환하는 9단계 향상 파이프라인:

1. GT 좌표 스케일 보정 (small/large/auto)
2. 폴리곤 마스크 생성
3. **원근 보정(perspective warp)** — 배경 제거, 정면 정규화
4. 마스크 기준 타이트 크롭
5. 경계 패딩(옵션)
6. 업스케일 ×2.0
7. 회색조 변환
8. **CLAHE** 대비 향상 (clipLimit=2.5, tile=8×8)
9. 노이즈 제거(median/NLM) + 언샤프 마스크 (sigma=1.2, amount=0.55)

---

## 3. 학습 과정 · 모델 · 하이퍼파라미터

### 3.1 YOLO26x (주 탐지 모델)

`train_yolo26x_kfold.py` — COCO 사전학습 YOLO26x(59M)를 5-Fold로 미세조정. **수렴까지 학습**(patience=100)해 mAP@0.5 **0.901**로 전 모델 1위.

| 하이퍼파라미터 | 값 |
|---------------|-----|
| 사전학습 가중치 | `yolo26x.pt` (COCO) |
| 에폭 | 최대 400, **patience=100 조기종료** |
| 배치 | 2 |
| 이미지 크기 | **960** |
| 학습률 | 1e-4 |
| 옵티마이저 | Adam |
| K-Fold | 5 (seed=42) |
| 디바이스 | CUDA (GPU 필수) |
| 결과 | `artifacts/yolo26x_kfold/fold{0-4}/weights/best.pt` + `kfold_summary_map50.csv` |

> **fold 분할 출처**: `artifacts/yolo11x_kfold/total_fold{i}/dataset/data.yaml`. 디렉터리 이름은 YOLO11x 실험 때 만들어진 것이지만 **현재 모든 탐지 모델(YOLO26x·YOLOv5x·FRCNN·EfficientDet)이 공유하는 공용 분할**이므로 삭제하면 공정 비교 프로토콜이 깨집니다.

### 3.1.1 YOLO11x · YOLOv5x (버전 비교용)

리뷰어 지적 R1("최신 YOLO 버전 간 성능을 비교하라") 대응을 위한 비교 baseline. 주 모델 아님.

| 모델 | 스크립트 | 설정 | mAP@0.5 |
|------|---------|------|---------|
| YOLOv5x | `train_yolov5x_kfold.py` | imgsz 960, batch 2, patience 100 (YOLO26x와 동일) | 0.891 |
| YOLO11x | `one_click_finetune_yolo11.py` | imgsz 1280, batch 8, Adam lr 1e-4, patience 50 | 0.862 (이어학습 후) |

**YOLO11x 이어학습(resume)** (`resume_more_train_kfold_both.py`): 각 fold의 `best.pt`에서 +50 에폭 추가 학습 (batch=2, imgsz=1280, lr=1e-4). YOLO와 Faster R-CNN이 **동일한 split JSON**을 공유해 공정 비교.

> `one_click_finetune_yolo11.py`는 위 공용 fold 분할을 생성하는 스크립트이자 `eval_det_oof.py`가 `get_all_region_pairs()`/`make_kfold_splits()`를 import하는 대상이므로 유지합니다.

### 3.2 Faster R-CNN (베이스라인)

`train_only_my_frcnn.py` — Torchvision ResNet50-FPN.

| 하이퍼파라미터 | 값 |
|---------------|-----|
| 백본 | ResNet50 + FPN (COCO 사전학습) |
| 클래스 수 | 2 (배경 + signboard) |
| 에폭 | 최대 300, patience=20 |
| 배치 | 2 |
| 학습률 | 0.005 |
| 옵티마이저 | SGD (momentum=0.9, weight_decay=5e-4) |
| K-Fold | 5 (seed=42) |
| 데이터 형식 | COCO (`instances_train.json`) |

### 3.3 EfficientDet (베이스라인, 평가 전용)

`eval_map_only.py` — `tf_efficientdet_d0`, 입력 512 letterbox, score_thr=0.05, 단일 클래스 AP@0.5 계산.

### 3.4 TrOCR *(참조 인식기 — 배포에서 제외, 3.7)*

`train_textinthewild_ocr.py` — `prepare`(크롭 추출) + `train-trocr`(미세조정).

**✅ 표준 프로토콜: signboard_v3 무누수 재학습 (2026-08).** 기존 signboard_full은 train/val 90/10에 **test가 없어** 발표치(84.1%)에 선택 편향·누수가 있었음(구 모델이 v3 test에서 98% = 암기 증거). 재학습 프로토콜:

| 항목 | 값 |
|---|---|
| 분할 | train 62,464 / val 7,884 / **test 7,756** — 소스 이미지 단위, seed 42 (`--test-ratio 0.1`) |
| 증강 | 회전±4°, 원근 0.12, 밝기/대비 ±35%, 블러, 저해상 왕복 0.6~0.9× (`--augment`) |
| **토크나이저** | **바이트레벨 BPE 이식 (`--tokenizer-dir`)** — base(trocr-small-printed)의 sentencepiece는 한글 음절 다수가 `<unk>`로 라벨을 손상시킴('정관장'→'정장'). 바이트레벨 BPE는 무손실 (78,104라벨 검증) |
| 학습 | 10에폭 @5e-5 → 어닐링 3에폭 @1e-5 (val loss 0.867 수렴) |
| **결과** | **test exact 72.9% / CER 0.149** (한글 72.0% / 비한글 75.1%) — `signboard_v3/trocr_model`, 예측: `test_predictions.csv` |

```bash
# 데이터 준비 + 학습 (3-way 분할 + 증강 + 무손실 토크나이저)
.venv/Scripts/python.exe train_textinthewild_ocr.py all \
    --json artifacts/signboard_data_info.json --image-dir artifacts/Signboard \
    --out-dir artifacts/ocr_training/signboard_v3 \
    --val-ratio 0.1 --test-ratio 0.1 --seed 42 --augment \
    --tokenizer-dir <바이트레벨 BPE 토크나이저 경로> --epochs 10 --batch-size 8
```

구(signboard_full) 학습(base `trocr-small-printed`, 5ep@5e-5, batch 8, AdamW)은 val exact 84.1%였으나 **test 분할이 없어 누수**가 있었고 위 v3 프로토콜로 대체되었습니다. 산출물: `trocr_checkpoints/epoch_*`, `trocr_model/`, `trocr_model_best/`.

> ⚠️ **transformers 버전 고정 필수:** 미세조정 TrOCR 모델은 `transformers==4.49.0`으로 저장되었습니다(`config.json`의 `transformers_version`). 더 낮은 버전(예: 4.44)으로 로드하면 디코더 시작 토큰 불일치로 **이미지를 무시한 환각 출력**이 발생하고, 더 높은 4.57은 `tokenizers` 충돌로 import가 막힙니다. → **`transformers==4.49.0` + `tokenizers 0.21.x`** 로 맞춰야 합니다.

### 3.5 EasyOCR — **검출기(CRAFT)로 배포 중** · 인식기는 참조용

EasyOCR은 두 역할이 분리됩니다:
- **CRAFT 검출기 = 배포 파이프라인의 1차 검출기**(3.7 ①). text=0.7 / link=0.4 / low_text=0.4, y-위치 라인 그룹핑 + x 정렬로 읽기 순서 복원, 구글맵 저작권 텍스트 자동 제거. 한글권은 `signboard_v3_custom` 리더, 브루클린은 영어 리더를 씁니다(리더 종류가 CRAFT 검출 결과에 영향).
- **CRNN 인식기 = 참조 후보**(배포 제외). deep-text-recognition CRNN(VGG+BiLSTM+CTC, imgH 64, imgW 600)을 `build_easyocr_trainer_data.py` + `external/EasyOCR/trainer/`로 학습, `make_easyocr_plugin.py`로 `recog_network` 플러그인 포장. **in-domain test(v3): exact 71.6%**, 5-Fold val CER ~0.157.

### 3.6 PaddleOCR (PP-OCRv5 rec) — **배포 인식기**

`korean_PP-OCRv5_mobile_rec`(SVTR_LCNet, PPLCNetV3, MultiHead CTC+NRTR)를 Signboard 데이터로 미세조정. PaddleX로 PaddleOCR repo 확보 후 `tools/train.py`를 직접 사용.

**공통 하이퍼파라미터:** 공식 사전학습 시작, 배치/샘플러 48(MultiScaleSampler first_bs=48 — 기본 128은 공유메모리 thrashing), Adam(L2 3e-5), Cosine lr, **warmup_epoch=0**.

> ⚠️ **함정:** PaddleOCR 기본 lr 스케줄은 75에폭용(warmup_epoch=5)이라 짧은 미세조정에서는 전 구간 lr이 상승해 best 이후 악화됩니다. **warmup_epoch=0**으로 막판 코사인 감쇠를 살려야 최적점에 안착합니다.

**모델 계보 (in-domain = signboard_v3 test / val):**

| 버전 | 데이터·개입 | in-domain | 상태 |
|---|---|---|---|
| v2 | signboard_full (누수 있는 구 분할) | val acc 85.0% | 폐기 |
| v3 | signboard_v3 무누수 분할, 6ep | **test exact 86.0% / CER 0.073** · val 85.6% | 기준선 |
| v3b | v3 레시피 그대로 재학습(변동 측정용) | val 85.2% | 변동 대역 근거 (4.6) |
| v4_spacecat | RecConAug 결합부에 **갭+공백** 삽입 | val 85.9% · 합성 2단어쌍 공백재현 **99.8%** | 4.6 |
| **v5_lines** | v4 + **실제 라인 크롭 13,148**(4.8) 혼합 | val 79.4%\* · **실라인 test exact 55.7% / CER 0.108** | **배포** |

\* v5의 val이 낮은 것은 val 세트에 어려운 실라인 1,725개가 추가되었기 때문으로, 단어만 보는 test exact는 85.6%로 동등합니다.

- 배포용 추론은 `tools/export_model.py`로 **inference 포맷 export** → `output/paddle_signboard_rec_v5_lines/inference/`.
- 학습 래퍼: `train_paddle_v4_spacecat.py --config <yml> [--stock]` (`--stock`은 spacecat 패치 off = 순정 v3 레시피).
- **in-domain 3엔진 순위: PaddleOCR 86.0% > TrOCR 72.9% > EasyOCR 71.6%** (동일 v3 분할).

### 3.7 배포 OCR 파이프라인 (`run_ocr_line.py`) — **현행**

간판 크롭 1장 → 텍스트 라인들. 단계별 근거는 4.5~4.9에 있습니다.

```
간판 크롭
  │
  ├─① 검출: EasyOCR(CRAFT) 박스  ∪  PaddleOCR DB 박스(CRAFT와 미겹침만)   ← 4.9
  │      · CRAFT 파라미터 text=0.7 / link=0.4 / low_text=0.4
  │      · 박스 필터: 전화번호(숫자≥7)·저신뢰(<0.10)·숫자비율 잡음 제거
  │      · DB 단독은 오히려 나쁨 — union일 때만 이득(실패 모드가 상보적)
  │
  ├─② 라인 그룹핑: y-중심 클러스터링(**y_tol 0.04**·이미지높이) → x 정렬  ← 4.7·4.9
  │      · 크면 두 줄이 한 스트립으로 합쳐져 GT 라인이 통째로 손실됨
  │
  ├─③ 라인 병합 크롭: 라인의 모든 박스를 union bbox + **pad 0.04** 로 **한 장**   ← 4.9
  │      · 핵심: 박스별로 자르지 않고 라인째 인식 (per-box 대비 +16.5%p)
  │      · 여백은 스트립 폭 비례 → 크면 옆 간판이 끼어듦 (0.12→0.04로 +2.8%p)
  │
  ├─④ 인식(PaddleOCR, 격리 subprocess):                                  ← 4.6·4.8
  │      · 강남·수원 → v5_lines·v4_spacecat·zero-shot **3-way 라인 다수결**   ← 4.9
  │                    (2개 이상 일치 → 그 텍스트, 아니면 v5)
  │      · 브루클린   → `en_PP-OCRv5_mobile_rec` (영어 전용, zero-shot 단독)
  │
  ├─⑤ 조립: 라인 텍스트를 `\n`로 이어 `ocr_{region}_{run}_paddle.csv`
  │
  └─⑥ VLM 교정 (Gemma 4 31B, Ollama 4-bit):                             ← 4.14
         · fixcand5: 라인별 5모델 후보(Paddle v5·v4·zero-shot + SVTRv2·PARSeq) + 이미지 대조로 오독 글자 확정
         → 최종 배포 출력 `artifacts/ocr_gt/ab_v4_spacecat/ocr_{region}_vlmfixcand5_gemma4_31b.csv`
```

**배포는 이 단일 경로입니다.** TrOCR·EasyOCR 인식기는 배포에서 빠졌고(3.4·3.5) oracle 상한 계산용 참조 후보로만 남습니다 — 라인 병합 Paddle이 세 엔진 앙상블(48.0%)을 단독으로 크게 앞서기 때문입니다(66.0%).

> ⚙️ **PaddleOCR 격리 (중요):** paddle(cu118)과 torch(cu118)가 **동일 이름의 cudnn_*_8.dll(다른 ABI)** 를 ship하여 한 프로세스에 공존하면 `WinError 127`이 납니다. 따라서 PaddleOCR는 별도 subprocess로 격리합니다 — 인식 `paddle_rec_worker.py`, 검출 `paddle_det_worker.py` (둘 다 `import paddle` 먼저 → `modelscope` stub으로 paddlex의 torch import 차단 → paddleocr). 파이프라인은 2-phase: 스트립을 임시 폴더에 저장 → worker가 1회 배치 처리 → 키별 병합.

```bash
.venv/Scripts/python.exe run_ocr_line.py --run 24   # 전 지역 (기본 = union + pad0.04 + y_tol0.04 + 투표)
.venv/Scripts/python.exe run_ocr_line.py --run N --no-vote --no-det-union --pad 0.12 --y-tol 0.06  # 구 동작
```

### 3.8 레거시: per-box 3-way 파이프라인 (`run_ocr_only.py --box`)

D20 이전의 배포 경로이자, 현재는 **참조 후보(easyocr/trocr) 산출 전용**입니다. 박스별로 세 엔진(TrOCR·EasyOCR·PaddleOCR)을 돌려 confidence argmax로 고르고 라인을 `\n`로 조립합니다. 단어 크롭 학습 모델을 라인/간판에 그대로 넣는 granularity 불일치를 피하려는 설계였으나, **박스 단위로 자르는 것 자체가 최대 병목**이었음이 4.7에서 밝혀졌습니다. EasyOCR CRNN 5-Fold 트레이너 설정은 `build_cv_folds.py`(batch=32, num_iter=30000, imgH=64, imgW=600, VGG+BiLSTM+CTC, 소스 이미지 단위 분할).

---

## 4. 학습 성과 (결과 지표)

### 4.1 YOLO11x 5-Fold 결과 (버전 비교 baseline — 주 모델 결과는 4.3 YOLO26x)

초기 학습 (`total_kfold_summary.csv`):

| Fold | mAP@0.5 | AP@0.75 | Precision | Recall |
|------|---------|---------|-----------|--------|
| 0 | 0.8706 | 0.7484 | 0.7634 | 0.8659 |
| 1 | 0.7852 | 0.7197 | 0.8608 | 0.6869 |
| 2 | 0.7853 | 0.6819 | 0.8140 | 0.7527 |
| 3 | 0.7826 | 0.6032 | 0.6194 | 0.8469 |
| 4 | 0.8114 | 0.6605 | 0.6893 | 0.8353 |
| **평균** | **0.8070** | **0.6947** | **0.7494** | **0.7975** |

이어학습 후 (`resume_more_summary_map50.csv`, **Ultralytics val 방식** — best.pt 기준):

| Fold | mAP@0.5 (이어학습 best.pt) |
|------|---------------------------|
| 0 | **0.8917** |
| 1 | **0.8625** |
| 2 | **0.8010** |
| 3 | **0.8780** |
| 4 | **0.8781** |
| **평균** | **0.8623** |

> ⚠️ **측정 방식 주의:** 위 초기 표(평균 0.807)는 `eval_map_only` 계열 **pycocotools COCOeval**, 이어학습 표(평균 0.862)는 **Ultralytics `model.val()`** 로 산출된 값이라 **자(척도)가 다릅니다.** 같은 Ultralytics 방식으로 보면 초기 best.pt도 평균 0.8645(`results.csv` peak)로, 이어학습(0.862)과 사실상 동등합니다. 모델 간 공정 비교를 위해서는 **전 모델을 한 방식으로 재측정**하는 것이 정확합니다(현재는 완성된 값 그대로 사용).

### 4.2 Faster R-CNN 5-Fold 결과 (`frcnn_kfold_summary.csv`)

| Fold | AP@0.5 | AP@0.75 | Precision | Recall | Best Epoch |
|------|--------|---------|-----------|--------|-----------|
| 0 | 0.9247 | 0.7119 | 0.6296 | 0.9577 | 5 |
| 1 | 0.8374 | 0.6530 | 0.7174 | 0.8800 | 8 |
| 2 | 0.8589 | 0.6762 | 0.6207 | 0.8780 | 4 |
| 3 | 0.8552 | 0.6950 | 0.6809 | 0.9014 | 32 |
| 4 | 0.8446 | 0.5652 | 0.6129 | 0.8507 | 4 |
| **평균** | **0.8642** | **0.6603** | **0.6523** | **0.8936** |

### 4.3 탐지 모델 비교 요약

**통합 프로토콜 (T2 완료, D26)** — 4개 모델 전부 **저장된 best 체크포인트**에서 재추론하여 **동일한 단일클래스 AP@0.5 구현 하나**로, **동일 5-Fold val 분할**에서 채점 (`eval_det_unified.py`):

| 모델 | **AP@0.5 (통합)** | fold별 | 지역별 (강남/브루클린/수원) | 구 표기 |
|------|------------------|--------|------------------------------|---------|
| **🥇 Faster R-CNN** | **0.874** | 0.926/0.875/0.825/0.870/0.871 | 0.813 / 0.886 / **0.912** | 0.873 |
| **YOLO26x** | **0.832** | 0.940/0.784/0.753/0.802/0.882 | 0.792 / 0.851 / 0.840 | ~~0.901~~ |
| **YOLOv5x** | **0.814** | 0.870/0.781/0.789/0.806/0.826 | 0.772 / 0.840 / 0.780 | ~~0.891~~ |
| **EfficientDet-D0** | **0.812** | 0.828/0.793/0.834/0.788/0.819 | 0.757 / 0.793 / 0.829 | 0.805 |

> **⚠️ [2026-09-11 정정, 4.17]** 아래 표의 fold 분할에는 같은 가게 누수가 있었고(수치를 부풀리진 않았음),
> group-aware 분할로 4모델을 재학습하자 **FRCNN 0.878 ≈ YOLO26x 0.872 > EffDet 0.830 ≈ YOLOv5x 0.817** 이 됐습니다.
> fold 표준편차(0.04~0.06)보다 작은 차이는 우열이 아닙니다. **논문 표 1은 4.17의 group-aware 수치를 쓰세요.**
>
> **⚠️ 통합 측정에서 순위가 바뀝니다: Faster R-CNN(0.874) > YOLO26x(0.832) > YOLOv5x(0.814) ≈ EfficientDet-D0(0.812).**
>
> **왜 바뀌었나** — 구 표의 YOLO 수치는 두 가지로 부풀려져 있었습니다:
> 1. **평가 도구 차이.** YOLO는 Ultralytics `val()`, FRCNN·EffDet은 자체 VOC AP였는데 두 도구의 격차가 **약 0.05**입니다(동일 가중치 YOLO26x: Ultralytics 0.879 vs 통합 AP 0.832 — 4.12에서 AP 적분·NMS·GT·집계를 모두 배제하고 확인).
> 2. **peak-over-epochs.** 네 모델 모두 "에폭 중 최고"를 기록했지만, FRCNN·EffDet은 **그 에폭의 가중치를 저장**해 기록값 = 저장 모델 성능인 반면, Ultralytics는 **fitness**(0.1·mAP50 + 0.9·mAP50-95) 기준으로 `best.pt`를 골라 **저장된 모델이 낸 적 없는 점수**가 보고됐습니다(YOLO26x 0.9006 → best.pt 실측 0.8785).
>
> **검증:** 통합 평가기로 잰 FRCNN 0.8736·EffDet 0.8042(thr 0.05)가 각각 기존 기록 0.8734·0.8050을 재현 — 비-YOLO 수치는 그대로이고 YOLO만 정정된 것입니다. score 임계값 0.0(전체 PR 곡선)과 0.05(구 설정) **양쪽에서 순위 동일**.
>
> **남은 교란 요인(논문에 명시 권장):** 각 모델은 자신의 학습·표준 설정에서 추론합니다 — YOLO imgsz 960, **FRCNN torchvision 기본(min 800 / max 1333)**, EffDet letterbox 512. 즉 **입력 해상도가 다릅니다.** "동일 조건 아키텍처 비교"가 아니라 **"각 모델을 제 운용점에서 동일 지표로 비교"** 로 서술하는 것이 정확하며, 해상도를 맞춘 재비교는 별도 실험이 필요합니다.
>
> *(YOLO11x·YOLO26s·YOLOv5s는 평가표 제외, 4.1의 YOLO11x 상세는 과거 기록.)*

```bash
.venv/Scripts/python.exe eval_det_unified.py                    # 통합 재측정(예측 캐시)
.venv/Scripts/python.exe eval_det_unified.py --score-thr 0.05   # 구 설정으로 재채점(즉시)
```

### 4.3.1 텍스트(word) 탐지 모델 비교 — `*_text_kfold`

위 4.3은 **간판(signboard) 영역 탐지**이고, 이 표는 **간판 위 텍스트(word) 탐지**입니다 (별도 태스크). 데이터는 **Signboard 데이터셋 5,000장**(word 박스 14,433개, 단일 클래스 `text`)을 **5-Fold(train 4000/val 1000)** 로 나눠 학습. 4개 모델을 동일 데이터·동일 설정(epochs 60, patience 15)으로 비교. 아티팩트 이름은 간판 탐지(`*_kfold`)와 구분해 **`*_text_kfold`**:

| 순위 | 모델 | AP@0.5 (5-Fold 평균) | 측정 방식 | fold별 |
|------|------|---------------------|-----------|--------|
| 🥇 | **YOLO26x** (`yolo26x_text_kfold`) | **0.746** | Ultralytics val (640) | 0.752/0.736/0.753/0.739/0.752 |
| 🥈 | **YOLOv5x** (`yolov5x_text_kfold`) | **0.737** | Ultralytics val (640) | 0.740/0.724/0.737/0.744/0.741 |
| 🥉 | **Faster R-CNN** (`frcnn_text_kfold`) | **0.638** | AP@0.5 (min 640) | 0.653/0.637/0.637/0.620/0.642 |
| | **EfficientDet-D0** (`effdet_text_kfold`) | **0.578** | AP@0.5 (512) | 0.583/0.553/0.581/0.583/0.589 |

> **해석:** 텍스트 탐지에서도 **순위가 간판 탐지와 완전히 동일** — **YOLO26x(0.746) > YOLOv5x(0.737) > FRCNN(0.638) > EfficientDet-D0(0.578)**. 최신 YOLO26x가 YOLOv5x를 다시 앞서고(같은 640·동일 설정), 작은 글자 탐지에서 YOLO 계열이 FRCNN·EffDet보다 크게 우세합니다(단어는 간판보다 작고 밀집돼 있어 격차가 더 큼). fold 간 편차도 작아(±0.01) 5,000장 규모에서 안정적입니다. 데이터가 커 간판 탐지(300장, patience 100)와 달리 patience 15로도 충분히 수렴.

> ⚠️ **대체됨 (2026-09-14):** 이 표는 YOLO=Ultralytics val, FRCNN/EffDet=자체 AP 의 **평가기 혼용**이고 데이터도 5k 뿐입니다. 논문에는 전체 27k 데이터·단일 채점기의 **4.18 hold-out 값**(YOLO26x 0.822 / YOLOv5x 0.809 / FRCNN 0.715 / EffDet 0.672)을 쓰세요.

### 4.4 OCR 평가 — 배포 파이프라인 성능 (3개 지역, `eval_ocr_v2.py`, **재라벨 GT**)

3개 지역 **411개 간판 크롭**(강남 138 · 브루클린 141 · 수원 132, 재라벨링으로 부적합 크롭 52개 제외 — `artifacts/gt/excluded_crops.csv`, GT 592 라인)을 평가합니다. GT를 라인 분리하여 **그리디 1:1 라인 매칭** 후 micro CER(공백 제거)·WAR(단어 LCS)·exact·recall·contain을 산출. **배포 = 라인 병합 Paddle 단독(3.7), 나머지 엔진은 참조·oracle 상한용.**

**평가 프로토콜 (재라벨 후 정비):**
- **GT 전면 재검수**: 463크롭 전수 재라벨링, 크롭-GT 밀림(사진 5장) 교정, 부적합 52크롭 제외
- **`###` don't-care**: 사람도 못 읽는 글씨는 GT에 `###`로 표시 → 해당 줄 채점 제외 + 그 크롭은 여분 예측 무벌점(부분문자열 정합)
- **`--mask-phone`**: 전화번호 패턴을 GT·예측 양쪽에서 제거
- **`--en-only-regions brooklyn`**: 영어권 브루클린은 한글을 양쪽에서 제거하고 채점
- 라인분리 버그 수정: 리터럴 `\n` GT가 통짜 비교되던 문제 해결

```bash
.venv/Scripts/python.exe eval_ocr_v2.py --mask-phone --en-only-regions brooklyn
```

**전역 (592 라인, micro — 배포 run22: CRAFT∪DB 검출 + 라인 병합, 강남·수원 v5_lines · 브루클린 en_PP-OCRv5):**

| 엔진 | CER ↓ | WAR ↑ | exact | recall | contain |
|----------------|------|------|-------|--------|---------|
| EasyOCR per-box (ko: signboard_v3_custom / bk: english_g2) | 0.449 | 0.290 | 38.5% | 62.8% | 56.6% |
| TrOCR per-box (ko: signboard_v3 / bk: base-printed) | 0.485 | 0.343 | 33.6% | 62.2% | 50.2% |
| **배포 = CRAFT∪DB + 라인 병합 + 3-way 투표 (run24)** | **0.163** | **0.648** | **69.9%** | **91.1%** | **75.2%** |
| (참고: 기하 파라미터·투표 이전, run22 — D23) | 0.174 | 0.618 | 66.0% | 90.2% | 72.5% |
| (참고: CRAFT 단독 검출, run21 — D21) | 0.204 | 0.605 | 64.0% | 86.1% | 70.1% |
| Oracle best-of-3 (상한: per-box easy/trocr + 라인 paddle) | 0.102 | 0.596 | 73.7% | 91.7% | 78.7% |

> - **배포는 단일 경로**(`run_ocr_line.py`) — 앙상블·후처리 없음. per-box 시대 배포(T7b 앙상블 48.0% / 0.410 / WAR 0.381) 대비 **exact +18.0%p·CER −58%·recall 63.5→90.2%**.
> - TrOCR/EasyOCR per-box가 낮은 것은 단어크롭 학습 vs 간판 박스 추론의 granularity 격차 — 라인 병합이 정확히 이 병목을 해소한 것 (contain은 ~71%로 거의 불변인데 exact만 크게 상승: "읽는 능력"은 같고 **라인 배열이 회복**된 것).
> - per-box 시대 상세 수치(conf-argmax 47.1 / 재선택 앙상블 48.0 / 구 oracle 53.4)는 git 이력 참조.

**지역별 (배포 run24 exact / CER / WAR · oracle CER):** 강남 73.1% / 0.190 / 0.649 · 0.098 — 브루클린(영어 전용 엔진·채점) **76.9% / 0.110 / 0.718** · **0.068** — 수원 58.6% / 0.255 / 0.535 · 0.189.

> **해석:**
> - **배포(CRAFT∪DB + 라인 병합 + v5): 전역 exact 66.0% / CER 0.174 / WAR 0.618 / recall 90.2%** — per-box 시대(48.0% / 0.410 / 0.381 / 63.5%) 대비 exact +18.0%p, CER −58%. 기여 분해: 라인 병합 +16.5%p(D19·D20), 실라인 학습 v5(WAR +16.5%p, D21), 검출기 union +2.0%p(D23).
> - 지역별 oracle 난이도 **brooklyn(0.080) < gangnam(0.109) < suwon(0.189)** — 영어권이 가장 쉽다는 논문 정성 서술과 일치 (라인 병합 후 강남·수원 순위는 역전: 수원은 검출 누락·아트 폰트 비중이 큼).
> - 남은 오류 구성: **1~2글자 오독**이 최대 덩어리이고, 나머지는 진짜 미검출(단일 문자·캘리그래피·양각 간판)입니다. 배포 69.9% vs oracle 73.7%.
> - 산출물: `artifacts/gt/ocr_eval_v2_summary.csv`(요약), `ocr_eval_v2_details.csv`(라인 단위). `eval_ocr_v2.py`는 (region, engine)별 **최신 run**을 자동 사용 (현재 paddle=run22 배포, ensemble=run22(=paddle 복사), easyocr/trocr=run22(per-box 복사)). 재라벨 이전 GT는 `ocr_{region}_gt.csv.bak_pre_relabel`로 보존.

### 4.12 지역별 탐지 mAP@0.5 — out-of-fold (T4/A1 완료)

표 3의 탐지 열을 채우기 위한 측정입니다. **누수 통제가 핵심**: GSV 사진 300장이 곧 5-Fold 학습 데이터(`artifacts/yolo11x_kfold/total_fold{i}/dataset`, train 238 / val 60)이므로, fold 하나의 가중치로 전체를 채점하면 암기를 측정하게 됩니다. 따라서 **각 사진을 그 사진이 val이었던 fold의 모델로만** 예측합니다(val 분할이 전체를 분할하므로 모든 사진이 정직한 out-of-fold 예측을 얻음).

**측정 도구는 Ultralytics `val()` 자체**를 씁니다(`eval_det_oof_ultra.py`) — (fold × 지역) 15개 부분집합을 각각 임시 데이터셋으로 만들어 해당 fold 가중치로 평가. 표 2를 만든 것과 동일한 프로토콜이라 두 표를 나란히 읽을 수 있습니다.

| 지역 | fold 수 | GT 인스턴스 | **mAP@0.5** (인스턴스 가중) | 단순 평균 |
|---|---|---|---|---|
| 강남 | 5 | 147 | **0.853** | 0.869 |
| 브루클린 | 5 | 173 | **0.887** | 0.885 |
| 수원 | 5 | 137 | **0.899** | 0.908 |
| **전체** | | 457 | **0.880** | |

**검증:** 지역별을 합산한 0.8799가 fold별 평균 0.8785와 사실상 일치 — 부분집합 분해가 전체와 정합합니다.

> ⚠️ **표 2의 0.9006과 다른 이유:** 표 2 수치는 `results.csv`의 **에폭 중 최고 val mAP**(peak-over-epochs)라 낙관 편향이 있습니다. 동일 가중치(`best.pt`)를 정직하게 재측정하면 **0.8785**입니다. 논문에 0.901을 쓴다면 이 점을 각주로 명시하거나, T2(프로토콜 통일)에서 전 모델을 best.pt 기준으로 재산출하는 것이 정확합니다.
>
> 참고로 자체 구현 AP(`eval_det_oof_per_region.py`, P/R/F1 병기)는 전역 0.840으로 Ultralytics 대비 ~0.04 낮습니다. AP 적분 방식(101-pt vs VOC)·NMS 임계값·GT 개수·집계 단위를 모두 배제 검증한 결과 차이는 val()의 추론 전처리(rect 배칭·letterbox)에서 오며, 따라서 **공식 수치는 Ultralytics 쪽**을 사용합니다.

```bash
.venv/Scripts/python.exe eval_det_oof_ultra.py --kfold artifacts/yolo26x_kfold
```

### 4.13 의미 태깅 — Gemma 4 31B · Qwen3-VL-32B (vlmrag, D36)

> 2026-09-19 방향 전환: 이전 gemma3:12b 기반 태깅·하이브리드·RAG 결과(구 4.13~4.15, D27~D29)는 **전부 폐기**하고
> 32B급 dense VLM 두 종으로 다시 측정했습니다. 두 모델 모두 Ollama Q4_K_M(4-bit), RTX 3080 10GB 에는 32%만 올라가고
> 68%는 RAM 오프로드라 크롭당 24~31초입니다(`scripts/run_vlm_hybrid.sh`, 크롭 단위 이어하기).

**설정.** 태깅 GT 411크롭(name 400 / tag 395, `tagging_gt.csv`), 경로는 **vlmrag**(이미지 + 지역 POI 사전 후보 5개,
하이브리드 검색 lexical+bge-m3 RRF — D30, 기권 금지). 42종 태그 완전일치가 tag exact, 대분류(key)만 맞으면 tag key.

| 모델 | name exact | name CER | **tag exact** | tag key | unknown | in-DB / off-DB (tag) | 초/건 |
|---|---|---|---|---|---|---|---|
| **Gemma 4 31B** | **63.5%** | **0.257** | **83.5%** | **91.6%** | 0.3% | 85.3% (n=102) / 82.9% (n=293) | 23.5 |
| Qwen3-VL-32B-Instruct | 57.8% | 0.330 | 80.0% | 90.4% | 1.8% | 81.4% / 79.5% | 30.7 |

**지역별 tag exact:** Gemma 4 — 강남 **90.4%** · 브루클린 72.0% · 수원 **88.2%**; Qwen — 86.0 · 73.5 · 80.3.
영어권(브루클린)만 두 모델이 비슷하고 한국어 지역은 Gemma 가 4~8%p 앞섭니다. in-DB(사전에 정답 상호가 있는 간판)와
off-DB 차이가 2~3%p 뿐이라 RAG 후보는 "재확인" 효과에 그치고, 태깅 정확도의 본체는 VLM 자체입니다.

```bash
.venv/Scripts/python.exe vlm/eval_vlm_tagging.py --model gemma4:31b --mode vlmrag --retriever hybrid --no-abstain --out artifacts/gt/vlm_tagging_results_gemma4_31b.csv
```

### 4.14 하이브리드 OCR — 멀티모달 post-OCR 교정 (fix / fixcand, D36, **공식 채택: Gemma 4 fixcand5**)

**문헌 위치.** OCR 출력을 LLM 으로 고치는 *post-OCR correction* 은 텍스트 전용 LLM 교정(CLOCR-C 계열, ICDAR 2026
HIPE-OCRepair)에서 최근 **이미지+OCR 텍스트를 함께 주는 멀티모달 교정**(arXiv 2504.00414; Springer 2025 "Enhancing OCR
Post-processing Through VLM" — PaddleOCR·EasyOCR·Tesseract·TrOCR 출력을 14개 VLM 으로 교정)으로 옮겨 왔습니다.
본 절의 **fix** 가 그 멀티모달 교정이고, **fixcand** 는 고전 post-OCR 의 *N-best 재순위* 골격에 후보를 **다중 인식기
출력**으로 만들고 VLM 이 **이미지로 고르게** 한 변형입니다. 문헌의 표준 대조군인 **텍스트 전용 교정(fixtext)** 을 함께
측정해 이미지가 기여하는 몫을 분리했습니다.

**방식 (모두 라인 수·순서 보존, 확신 없으면 원문 유지, 새 라인 추가 금지):**
- **fixtext**: OCR 라인만 제시 (이미지 없음) → 언어 지식으로 명백한 오독만 교정. *문헌 기준선*
- **fix**: 이미지 + 배포 OCR 라인 → 사진과 다르게 보이는 글자만 교정
- **fixcand**: 이미지 + 라인별 **3모델 후보**(Paddle v5·v4·zero-shot, `gen_line_candidates.py`) → 사진 대조로 확정
- **fixcand5**: 후보에 **SVTRv2·PARSeq**(4.19) 추가 = 5모델. Paddle 계열은 같은 곳을 같이 틀리므로 오류가 독립인 인식기를 넣음

**결과 — GSV 411크롭·592라인, 배포 OCR(run24) 위에서, 공식 프로토콜(`eval_ocr_v2 --mask-phone`, 브루클린 en-only):**

| 방식 | Gemma 4 31B exact / CER / WAR | Qwen3-VL-32B exact / CER / WAR |
|---|---|---|
| 배포 OCR만 (run24) | 69.9% / 0.163 / 0.648 | 동일 |
| fixtext (텍스트 전용, 문헌 기준선) | 72.5% / 0.159 / 0.691 | 71.3% / 0.160 / 0.677 |
| fix (이미지 + OCR 라인) | 79.9% / 0.130 / 0.750 | 79.7% / 0.127 / 0.743 |
| fixcand (3후보) | 79.9% / 0.127 / 0.759 | 78.9% / 0.137 / 0.741 |
| **fixcand5 (5후보)** | **80.9% / 0.122 / 0.766** | 79.7% / 0.130 / 0.747 |

**지역별 exact (Gemma 4 fixcand5):** 강남 73.1 → **82.5%** (CER 0.147) · 브루클린 76.9 → **88.9%** (0.073) · 수원 58.6 → **70.2%** (0.206).

**읽는 법:**
1. **교정 이득의 3/4 는 이미지에서 옵니다.** 텍스트만 주면 +2.6%p(72.5), 이미지를 주면 +10.0%p(79.9). 문헌이 멀티모달
   교정의 근거로 드는 바로 그 비교이며, 간판은 문서와 달리 언어 사전만으로 복원이 안 되는 고유명사가 대부분이라 격차가 큽니다.
2. **후보는 "다양할 때만" 도움이 됩니다.** Paddle 3후보(fixcand)는 fix 와 동률(79.9)이고, SVTRv2·PARSeq 를 넣은 fixcand5 만
   +1.0%p 입니다. 같은 계열 후보는 정보가 중복됩니다.
3. **두 32B 모델은 같은 체급입니다.** 모든 방식에서 Gemma 가 0.2~1.2%p 위지만 변동 대역 안이고, Qwen 은 fix 에서 수원 70.7% 로
   Gemma(68.5)를 앞섭니다. 배포는 fixcand5 최고치인 **Gemma 4 31B** 로 확정합니다.
4. 남는 오류의 대부분은 교정으로 못 건드리는 **완전 전멸 라인**(OCR 이 라인 자체를 못 낸 경우)입니다. 라인 추가를 허용하는
   변형은 방향 전환으로 제외했습니다(4.20 계획 참고).

```bash
.venv/Scripts/python.exe pipeline/exp_vlm_ocr.py --mode fix     --model gemma4:31b --out-suffix _gemma4_31b
.venv/Scripts/python.exe pipeline/exp_vlm_ocr.py --mode fixcand --cand-tags v5,v4,pre,svtrv2,parseq --name fixcand5 --model gemma4:31b --out-suffix _gemma4_31b
.venv/Scripts/python.exe pipeline/eval_vlm_hybrid_matrix.py --models gemma4_31b,qwen3vl32b       # 표 생성 (artifacts/gt/vlm_hybrid_matrix.csv)
```

**함정 기록.** ① Ollama `qwen3-vl:32b` 기본 태그는 thinking 전용이라 `think:false` 가 무시되고 초당 1.3토큰으로 추론문만 쓰다
잘림 → `qwen3-vl:32b-instruct` 사용. ② 4-bit 32B 는 VRAM 10GB 에 안 들어가 GPU/RAM 분할(32/68%) — 크롭당 25~30초, 411크롭 ×
4방식 × 2모델 = 약 30시간. ③ 두 모델 출력 파일은 `--out-suffix _<tag>` 로 분리하고 크롭 단위 `.partial.jsonl` 로 이어하기.

### 4.15 (삭제) RAG — OCR 교정 후보 제공

gemma3:12b 기반 RAG 교정 실험은 방향 전환으로 제거했습니다. POI 사전과 하이브리드 검색(D30)은 4.13 의 vlmrag 태깅 경로에서만 사용합니다.

### 4.16 연쇄(end-to-end) 측정 — 모듈 점수의 곱은 파이프라인 성능이 아니다 (D31)

**지금까지 4.4~4.15의 모든 수치는 GT 폴리곤에서 자른 크롭 기준입니다.** 즉 탐지가
놓친 간판은 평가에 들어오지도 않았습니다. "사진을 넣으면 얼마나 건지는가"를 재려면
탐지 출력으로 다시 잘라야 하고, 그것이 논문이 주장하는 시스템의 실제 성능입니다.

**설계**: 4.12와 동일한 **out-of-fold** 예측(각 사진은 그 사진을 held-out 한 fold
모델로만)으로 YOLO26x 박스를 뽑고(conf 0.25 = 4.12의 P/R 운용점), **GT 크롭과 완전히
같은 전처리 경로**로 크롭을 생성한 뒤(`make_crops_from_gt_polygon.py --gt-csv`)
전 단계를 재측정합니다 (`e2e_det_boxes.py` → `eval_e2e_cascade.py`).

**채점 규칙** — 연쇄에서는 예측·GT가 1:1이 아니므로 명시가 필요합니다:
- **TP**(IoU≥0.5) → 대응 GT 크롭의 정답과 정상 채점
- **FN**(탐지 누락) → 예측 자체가 없으므로 **오답 처리** (연쇄 손실의 핵심)
- **FP**(허위 탐지) → 정답이 없어 정확도 분모에서 제외하되 **개수와 "텍스트가 나온
  비율"을 따로 보고** — 지도 구축에서 FP는 "없는 가게를 만드는" 비용이라 정확도에
  섞으면 성격이 흐려집니다

| 지역 | GT 간판 | TP | FN | FP | recall | precision |
|---|---|---|---|---|---|---|
| 강남 | 151 | 117 | 34 | 34 | 0.775 | 0.775 |
| 브루클린 | 171 | 146 | 25 | 49 | 0.854 | 0.749 |
| 수원 | 141 | 115 | 26 | 41 | 0.816 | 0.737 |
| **합계** | **463** | **378** | **85** | **124** | — | — |

**OCR 연쇄 결과 (라인 exact).** VLM 교정(4.14) 적용 전/후를 함께 싣습니다. 탐지 크롭에는 다모델 후보 파일이
없어 교정은 `fixcand5` 대신 `fix`(이미지 + OCR 라인)를 쓰며(GT 크롭에서 80.9 vs 79.9%, 1.0%p 차이), 교정 모델은
배포와 같은 **Gemma 4 31B** 입니다(괄호는 Qwen3-VL-32B):

| 지역 | GT 라인 | 연쇄 recall | TP 위 정확도 | GT 크롭 기준 |
|---|---|---|---|---|
| | | OCR만 → **+VLM** | OCR만 → **+VLM** | (상한) |
| 강남 | 212 | 50.0 → **59.9%** (56.1) | 60.6 → **72.6%** (68.0) | 63.7% |
| 브루클린 | 199 | 46.2 → **58.3%** (55.8) | 54.8 → **69.0%** (66.1) | 61.8% |
| 수원 | 181 | 32.0 → **43.6%** (40.9) | 39.2 → **53.4%** (50.0) | 46.4% |
| **전체** | **592** | **43.2 → 54.4%** (51.4) | **52.1 → 65.6%** (61.9) | **57.8%** |

> **하이브리드가 박스 품질 저하를 흡수합니다.** VLM 적용 전에는 TP 위 정확도(52.1%)가
> GT 크롭 기준(57.8%)보다 **낮았는데**, 적용 후에는 **높아집니다**(65.6% vs 57.8%).
> 탐지 박스가 GT보다 넉넉하면 OCR 검출기에는 불리하지만(옆 간판 혼입) VLM은 사진에서
> 맥락을 더 보므로 오히려 유리합니다. 개선폭은 GT 크롭 +10.0%p(fix) / 탐지 크롭 +13.5%p 로
> 탐지 크롭에서 오히려 크며, **파이프라인 견고성의 근거**가 됩니다.
>
> 다만 **연쇄 recall 54.4%가 시스템의 실제 성능**입니다. GT 크롭 기준 80.9%(fixcand5)와의
> 차이 26.5%p는 전적으로 탐지 누락(FN 85개)에서 옵니다.

> *연쇄 recall = (TP 크롭에서 맞은 라인)/(전체 GT 라인) — 파이프라인이 실제로 건진 비율.
> TP 위 정확도 = 탐지가 성공했을 때의 성능(기존 GT 크롭 평가와 직접 비교 가능).*

**세 가지 발견:**

1. **곱셈 추정은 낙관적입니다.** 탐지 recall × OCR로 추정하면 65.5%인데 실측은
   **43.2%** 로 22%p 낮습니다. 독립 가정이 깨지는 방향이 명확합니다 —
   **탐지가 어려운 간판(작고·가려지고·비스듬한)은 OCR도 어렵습니다.**
2. **박스 품질이 IoU를 넘어 인식에 영향을 줍니다.** TP 위 정확도 52.1% vs GT 크롭
   57.8% — IoU 0.5를 넘겨 "같은 간판"으로 매칭됐는데도 **5.7%p를 잃습니다**(글자
   잘림·옆 간판 혼입). mAP만 보고하면 이 손실이 보이지 않습니다.
3. **FP 124개 중 114개(92%)가 텍스트를 냅니다.** 허위 탐지는 조용히 실패하지 않고
   그럴듯한 문자열을 만듭니다 — 지도 구축에서는 누락보다 나쁠 수 있어, **정밀도가
   응용상 왜 중요한지**에 대한 실측 근거입니다.

**태깅 연쇄 (3모듈 전체, `eval_e2e_tagging.py`):**

| 지역 | GT 대상 | TP | FN | 연쇄 recall | TP 위 정확도 | GT 크롭 기준 | 허위 POI |
|---|---|---|---|---|---|---|---|
| 강남 | 136 | 109 | 27 | 69.1% | **86.2%** | 90.4% | 34/34 |
| 브루클린 | 132 | 115 | 17 | 62.9% | 72.2% | 72.0% | 49/49 |
| 수원 | 127 | 104 | 23 | 67.7% | 82.7% | 88.2% | 41/41 |
| **전체** | **395** | **328** | **67** | **66.6%** (Qwen 65.3) | **80.2%** (78.7) | 83.5% | **124/124** |

- **허위 POI: FP 124건 전부(100%)에 업종 태그가 붙습니다**(기권 금지 조건). 존재하지 않는 가게가
  상호명과 업종을 갖춘 채 지도에 등록될 수 있다는 뜻입니다. 지도 구축 응용에서
  **recall만이 아니라 precision이 결정적**이라는 정량 근거입니다.
- 태깅은 OCR과 달리 TP 위 정확도(80.2%)가 GT 크롭 기준(83.5%)보다 낮아, 박스 품질 손실이 그대로 드러납니다.
  **선택 효과도 함께 명시해야 합니다** — 탐지가 성공한 간판은 애초에 크고 선명한
  쉬운 간판이고, 놓친 67개는 어차피 어려웠을 가능성이 높습니다. 이 수치를 순수한
  견고성 증거로만 읽으면 안 됩니다.

**파이프라인 전체 요약:**

| 관점 | 탐지 | OCR | 태깅 |
|---|---|---|---|
| 모듈별 (GT 크롭 = **상한**) | mAP@0.5 0.880 | 80.9% | 83.5% |
| **연쇄 (실제 시스템)** | recall 0.816 | **54.4%** | **66.6%** |

*채점 단위가 달라 두 연쇄 수치를 직접 비교하면 안 됩니다 — OCR은 라인 단위
완전일치(엄격), 태깅은 크롭 단위 태그 일치입니다. 읽는 법: **간판을 찾아 업종까지
맞히는 비율 66.6%**, **간판 글자를 정확히 읽어내는 비율 54.4%**.*

> **다음 투자처는 탐지 recall입니다.** GT 크롭 80.9% ↔ 연쇄 54.4%의 26.5%p 차이는
> 전적으로 탐지 누락(FN 85개)에서 오며, 인식을 아무리 개선해도 이 손실은 줄지 않습니다.

> **논문 서술 권고:** 모듈별 표(탐지 0.88 / OCR 80.9% / 태깅 83.5%)와 연쇄 표를
> **반드시 함께** 싣고, 모듈별 수치가 "GT 크롭 상한"임을 명시하세요. 연쇄 수치가
> 낮다는 사실 자체보다, 낮은 원인을 ①탐지 누락 ②박스 품질 ③상관된 난이도로
> 분해해 보여주는 것이 기여입니다.

```bash
.venv/Scripts/python.exe e2e_det_boxes.py --conf 0.25                    # OOF 박스+매칭
.venv/Scripts/python.exe make_crops_from_gt_polygon.py --region gangnam \
    --gt-csv artifacts/gt/gt_gangnam_det.csv --out-subdir crop_det       # 동일 전처리
.venv/Scripts/python.exe run_ocr_line.py --run 30 --crop-dir artifacts/gt/crop_det
.venv/Scripts/python.exe eval_e2e_cascade.py --ocr-run 30
```

### 4.17 탐지 fold 누수 — 같은 가게가 train/val 에 걸쳐 있었다 (D33)

**발단:** 표에서 배포 YOLO26x 의 OOF mAP(0.880)가 통합 프로토콜의 어느 모델보다
높게 보인다는 지적. 증강→분할 순서 때문이라는 가설로 시작했습니다.

**검증 결과 — 증강 누수는 아닙니다.** fold 의 train 은 238장 전부 원본 파일이고(증강
접미사 0개), val 은 298장을 정확히 분할하며 fold 내 train∩val = 0 입니다. 증강은
Ultralytics 가 학습 중 즉석에서 적용할 뿐 파일을 만들지 않습니다. 전체 이미지
pHash 로 본 val↔train 근접 중복도 0쌍입니다(해밍 ≤10).

**진짜 누수 — 같은 가게 간판.** 인접 위치에서 찍은 GSV 사진들이 **같은 상가 간판**을
담은 채 다른 fold 로 갈렸습니다. GT 간판 텍스트(4자 이상, 일반어 제외)로 대조하면:

| 지역 | 다른 fold 사진과 가게를 공유하는 사진 |
|---|---|
| 강남 | 31 / 99 (31.3%) |
| 브루클린 | 25 / 99 (25.3%) |
| 수원 | 6 / 100 (6.0%) |
| 전체 | **62 / 298 (20.8%)** |

예: `brooklyn__13`↔`brooklyn__15` 가 `9ROUND`·`FOOD WINE GROCER` 공유, `gangnam__34/61/62/63/64` 는
한 건물을 5지점에서 찍은 것. 탐지기는 학습에서 본 **바로 그 간판 물체**를 val 에서 다시
만나므로 fold/OOF mAP 가 낙관적입니다. (참고: "OOF 가 가장 높다"는 표 자체는 평가기
혼용 — Ultralytics val 이 통합 AP 보다 ~0.05 높음, 4.3 — 이 주원인이지만, 위 누수는 그와
별개로 모든 fold 수치에 공통으로 섞여 있습니다.)

**조치:** `build_grouped_folds.py` — 가게 공유를 간선으로 한 연결성분(249개, 최대 5장)을
단위로 지역 층화·균형 배정한 **group-aware 5-fold** (`artifacts/kfold_grouped`, fold 간
가게 공유 0쌍, val 지역별 20/20/20). 데이터·라벨·하이퍼파라미터는 기존과 동일하고
**배정만** 바뀌므로 재학습 결과의 차이 = 누수 효과입니다. 4모델 전부 재학습
(`run_grouped_retrain.sh`, 순차, 약 58h) 후 통합 프로토콜로 재평가합니다.

한계: 재라벨 때 제외된 크롭 52개는 GT 텍스트가 없어 가게 매칭에 못 씁니다 — 잔여
누수가 소량 남을 수 있습니다.

**중간 결과 — YOLO26x (재학습 완료, 2026-09-10):** 누수를 제거해도 수치가 내려가지 않았습니다.

| 지표 (YOLO26x) | 기존 fold (누수) | group-aware fold | 차이 |
|---|---|---|---|
| 에폭 최고 mAP50 (5-fold 평균) | 0.901 | 0.897 | −0.004 |
| OOF mAP@0.5 (Ultralytics val, 전체) | 0.880 | 0.883 | +0.003 |
| OOF 지역별 (강남 / 브루클린 / 수원) | 0.853 / 0.887 / 0.899 | 0.840 / 0.901 / 0.906 | 잡음 범위 |
| 통합 프로토콜 (best.pt + 단일 AP@0.5) | 0.832 | **0.872** | **+0.040** |

- **같은 가게 누수는 탐지 mAP 를 부풀리지 않았습니다.** 간판 물체를 학습에서 봤어도
  val 점수가 오르지 않았다는 뜻으로, 300장 규모에서 탐지기는 특정 간판을 암기하기보다
  "간판다움"을 일반화하는 쪽으로 학습된 것으로 보입니다.
- "OOF 가 가장 높다"는 인상은 **평가기 혼용**(Ultralytics val ≈ 통합 AP + 0.05, 4.3)이
  전부였습니다.
- 통합 프로토콜 수치가 +0.040 움직인 것은 **재분할 + 재학습 1회의 변동**입니다. 이 폭은
  표 1의 모델 간 격차(FRCNN 0.874 / YOLO26x 0.832 / YOLOv5x 0.814 / EffDet 0.812)와
  같은 크기라, **4모델 재학습이 끝나기 전에는 모델 순위를 단정할 수 없습니다.** 분해해
  보면 기존 fold 는 "에폭 최고 mAP50 ↔ best.pt 통합 AP" 격차가 fold 별 0.009~0.107
  (평균 0.068)로 들쭉날쭉했고, 새 fold 는 0.002~0.038(평균 0.025)로 안정적입니다.
  best.pt 가 저장된 에폭의 mAP50 은 두 경우 모두 최고치에 근접하므로 fitness 선택 문제가
  아니라, **통합 평가기와 Ultralytics 평가기의 격차 자체가 fold 에 따라 변한다**는 뜻입니다
  — 통합 프로토콜의 평가기 민감도는 별도 점검 항목으로 남깁니다.

**중간 결과 — YOLOv5x (재학습 완료, 2026-09-10, 7.2h):** YOLO26x 와 같은 결론입니다.

| 지표 (YOLOv5x) | 기존 fold (누수) | group-aware fold | 차이 |
|---|---|---|---|
| 에폭 최고 mAP50 (5-fold 평균) | 0.891 | 0.886 | −0.005 |
| 통합 프로토콜 (best.pt + 단일 AP@0.5) | 0.814 | 0.817 | +0.003 |
| 통합 지역별 (강남 / 브루클린 / 수원) | 0.772 / 0.840 / 0.780 | 0.815 / 0.856 / 0.744 | 지역별 ±0.04 |

두 YOLO 모두 누수 제거 후 평균은 불변이고, 지역별 수치만 ±0.04 안에서 재배열됩니다 —
**지역별 탐지 수치의 신뢰 구간이 그 정도**라는 뜻이며, 표 3의 지역 간 차이(0.853/0.887/0.899)를
해석할 때 이 폭을 감안해야 합니다.

**중간 결과 — EfficientDet-D0 (재학습 완료, 2026-09-10, 7.3h):**

| 지표 (EffDet-D0) | 기존 fold (누수) | group-aware fold | 차이 |
|---|---|---|---|
| 에폭 최고 AP@0.5 (5-fold 평균) | 0.805 | 0.822 | +0.017 |
| 통합 프로토콜 (best 체크포인트 + 단일 AP@0.5) | 0.812 | 0.830 | +0.017 |
| 통합 지역별 (강남 / 브루클린 / 수원) | 0.757 / 0.793 / 0.829 | 0.790 / 0.823 / 0.881 | 전 지역 상승 |

세 모델(YOLO26x +0.040 / YOLOv5x +0.003 / EffDet +0.017)에서 통합 프로토콜 수치가
**내려간 경우가 없습니다.** 누수 제거는 탐지 mAP 를 낮추지 않았고, 관측된 변동은
재분할·재학습에 따른 잡음(0~0.04)입니다.

**최종 결과 — 4모델 전부 재학습 완료 (2026-09-11, 총 37.6h):**

| 모델 | 통합 AP@0.5 기존 fold | **통합 AP@0.5 group-aware** | 변동 | fold 표준편차 | 강남 / 브루클린 / 수원 (group-aware) |
|---|---|---|---|---|---|
| Faster R-CNN | 0.874 | **0.878** | +0.005 | 0.045 | 0.826 / 0.871 / 0.884 |
| YOLO26x | 0.832 | **0.872** | +0.040 | 0.039 | 0.833 / 0.874 / 0.879 |
| EfficientDet-D0 | 0.812 | **0.830** | +0.017 | 0.059 | 0.790 / 0.823 / 0.881 |
| YOLOv5x | 0.814 | **0.817** | +0.003 | 0.061 | 0.815 / 0.856 / 0.744 |
| YOLO26x OOF (Ultralytics val, 배포 참고) | 0.880 | 0.883 | +0.003 | — | 0.840 / 0.901 / 0.906 |

**⚠️ 집계 방식 정정 — "mean of means" (2026-09-11).** 위 표의 통합 AP 는 **fold 별 AP 5개의
평균**이었습니다(`eval_det_unified.py` 의 `mean` 열). fold 마다 이미지·박스 수가 다르고 PR
곡선을 따로 그리므로 이는 평균의 평균입니다. 정본은 **298장 전체의 OOF 예측을 하나의 PR
곡선으로 모아 계산한 pooled AP@0.5** 입니다 (지역별 열은 원래부터 지역 내 pooled). 텍스트
탐지(4.18)는 hold-out 단일 test 라 처음부터 pooled 입니다.

| 모델 | **pooled AP@0.5 (전체 298장, 정본)** | fold 평균 (구 표기) | 기존 fold pooled (참고) |
|---|---|---|---|
| YOLO26x | **0.860** | 0.872 | 0.826 |
| Faster R-CNN | **0.854** | 0.878 | 0.868 |
| EfficientDet-D0 | **0.828** | 0.830 | 0.788 |
| YOLOv5x | **0.806** | 0.817 | 0.794 |

pooled 로 보면 YOLO26x 와 FRCNN 의 순서가 뒤집히지만(0.860 vs 0.854) 차이 0.006 은 여전히
잡음입니다 — "두 모델 동등 최상위" 결론은 그대로입니다. Ultralytics OOF 값(0.883)은 (fold×지역)
val 의 인스턴스 가중 평균이라 역시 평균의 평균이며 Ultralytics 로는 pooled 계산이 불가능하므로
**논문 표에서는 제외**하고 각주로만 남깁니다. 논문 표 1·표 3의 탐지 수치는 이 pooled 값과
지역별 pooled 값(YOLO26x: 강남 0.833 / 브루클린 0.874 / 수원 0.879)을 씁니다.

**결론:**
1. **같은 가게 누수는 탐지 mAP 를 부풀리지 않았습니다.** 4모델 모두 누수 제거 후 수치가
   내려가지 않았습니다(+0.003 ~ +0.040). 누수 자체는 실재했으므로 group-aware fold
   (`artifacts/kfold_grouped`)를 **정본 분할**로 채택하고, 위 group-aware 열이 논문 표 1의
   공식 수치입니다.
2. **모델 순위가 바뀝니다 — "FRCNN > YOLO26x" 는 더 이상 성립하지 않습니다.** 기존 격차
   0.041 이 재분할 후 **0.006** 으로 사라졌습니다. fold 표준편차가 0.04~0.06 이므로
   **0.05 미만 차이는 모델 간 우열로 서술하지 마세요.** 정확한 서술: *Faster R-CNN 과
   YOLO26x 가 동등하게 최상위(≈0.87~0.88), EfficientDet-D0 와 YOLOv5x 가 그 아래(≈0.82~0.83)*.
   4.3 의 "FRCNN 0.874 > YOLO26x 0.832" 서술은 이 결과로 대체됩니다.
3. 재분할 한 번으로 지역별 수치가 ±0.04 움직입니다(지역당 val ≈20장). 표 3의 지역 간
   차이는 이 폭 안에서 읽어야 합니다.
4. "OOF 가 가장 높다"는 인상은 평가기 혼용(Ultralytics val ≈ 통합 AP + 0.01~0.05)이었습니다.
   두 평가기 수치를 한 표에 섞지 마세요.

**섞임 전수 감사 (2026-09-11, `audit_split_leakage.py`) — 전 항목 통과.** "이번엔 진짜 안 섞였나"에
숫자로 답합니다. 재실행 가능하며 위반이 하나라도 있으면 exit 1 입니다.

| 항목 | 결과 |
|---|---|
| D1 탐지 fold — val 이 298장을 정확히 분할, fold 간 val 중복 0, fold 내 train∩val 0 | 통과 |
| D2 탐지 fold — 1,490개 jpg 전부 원본 GSV 사진과 **md5 동일** (증강 사본·변형 0) | 통과 |
| D3 탐지 fold — 다른 fold 사진과 같은 가게 공유 0쌍 | 통과 |
| D4 탐지 fold — 디렉터리 내 증강 산출물 0 (mosaic/flip/hsv 는 학습 루프 안에서만) | 통과 |
| O1 OCR 단어 크롭 — train/val/test 간 원천 이미지 공유 0 (21,714 / 2,712 / 2,706 원천) | 통과 |
| O2 OCR 라인 크롭 — 단어 분할을 그대로 상속, 교차 원천 0 | 통과 |
| O3 OCR — 6개 크롭 폴더 디스크 파일 수 == 리스트 수 (리스트 밖 사본 0) | 통과 |
| O4 Paddle — RecConAug/RecAug 는 Train 블록에만, Eval 에 없음 (TrOCR 증강도 train only) | 통과 |

정정: 앞서 "증강 후 94,645장"이라고 한 것은 오독이었습니다. 94,608 = 단어 크롭 78,104 +
**라인 크롭 16,504** 이고(나머지 37장은 사람 검수용 preview), 디스크에 증강 사본은
없습니다. 이 저장소의 모든 증강은 **분할 완료 후 학습 루프 안에서만** 적용됩니다.
EasyOCR 학습 리스트(`signboard_v3/easyocr/`)도 같은 분할을 절대경로로 참조합니다(사본 없음).

```bash
.venv/Scripts/python.exe build_grouped_folds.py          # group-aware 분할 생성·검증
bash run_grouped_retrain.sh                               # 4모델 재학습 + 재평가 (재개 가능)
FOLD_ROOT=artifacts/kfold_grouped DET_TAG=_grouped .venv/Scripts/python.exe eval_det_unified.py --cache artifacts/gt/det_unified_preds_grouped.json --out artifacts/gt/det_unified_protocol_grouped.csv
```

### 4.5 파이프라인 확립 — 무엇이 실제로 효과가 있었나

per-box 3-way 앙상블(구 배포, exact 48.0%)에서 현행 파이프라인(66.0%)까지의 기여 분해입니다. **개입은 모두 단독 ablation으로 측정**했고, 각 단계 상세는 아래 절에 있습니다.

| # | 개입 | 전역 exact | CER | 상세 |
|---|---|---|---|---|
| — | per-box 3-way 앙상블 (구 배포) | 48.0% | 0.410 | 3.8 |
| ① | **라인 병합 인식** (박스별→라인째) | **64.5%** (+16.5%p) | 0.203 | 4.7 |
| ② | **실라인 학습 v5** (합성 concat→실제 라인) | 64.0% (exact 동등, **WAR 0.440→0.605**) | 0.204 | 4.8 |
| ③ | **검출기 union** (CRAFT ∪ PaddleOCR DB) | **66.0%** (+2.0%p) | 0.174 | 4.9 |
| ④ | **기하 파라미터 보정** (패딩 0.12→0.04, y_tol 0.06→0.04) | **69.3%** (+3.3%p) | 0.165 | 4.9 |
| ⑤ | **라인 레벨 3-way 다수결** (v5·v4·zero-shot) | **69.9%** (+0.6%p) | 0.163 | 4.9 |
| ⑥ | **VLM 멀티모달 교정** (Gemma 4 31B, fix) | **79.9%** (+10.0%p) | 0.130 | 4.14 |
| ⑦ | **+ 5모델 후보 그라운딩** (fixcand5) | **80.9%** (+1.0%p) | 0.122 | 4.14 |

**한 줄 요약:** 인식 모델을 바꾸는 것보다 **텍스트를 어떤 단위로·어떻게 잘라 넣는지**가 압도적으로 중요했습니다 — 라인 단위 병합(①)과 크롭 기하 보정(④)만으로 +19.8%p. 그 다음이 **학습 데이터를 추론 단위와 일치**시키는 것(②), **검출 커버리지**(③), **후보 합의**(⑤) 순입니다.

*(②의 exact가 −0.5%p인 것은 지역 단위 재실행 변동 대역(±1~2.3%p, 4.6 v3b) 안이며, 공백 민감 지표인 WAR는 +16.5%p로 명확히 개선됩니다.)*

### 4.6 인식기 학습 개입 — RecConAug 무공백 결합과 변동 대역 (D18)

**가설**: v3 Paddle 학습의 RecConAug가 크롭들을 라벨 공백 없이 이어붙여(`label += ext_label`, 에폭당 ~36% 샘플) 인접 단어를 붙여 읽도록 학습시킴. **개입(v4)**: 연결부에 시각적 갭(높이 0.15~0.40배, 경계색 블렌드) + 라벨 공백 삽입. 나머지 전부 v3와 동일(데이터·분할·사전학습·6에폭·yml). 구현: venv PaddleOCR `rec_img_aug.py`에 **`RECCON_SPACECAT=1` 환경변수 게이트 패치**(기본=순정, 원본 `.orig_v3` 백업; Windows DataLoader 워커가 몬키패치를 무시하므로 파일 패치 필수). 스크립트: `train_paddle_v4_spacecat.py`(--preview/--train/--export), `eval_paddle_ab.py`.

**in-domain A/B (test 분할, 두 모델 동일 이미지):**

| | single exact (7,756) | single CER | 합성 2단어쌍 exact (1,500) | 쌍 공백 재현율 |
|---|---|---|---|---|
| v3 (무공백 concat) | 85.9% | 0.074 | **0.0%** | **0.0%** |
| v4 (갭+공백 concat) | **86.2%** | 0.074 | **68.9%** | **99.8%** |

**가설 입증**: v3는 인접 단어쌍의 공백을 단 한 번도 넣지 못함(`한복 고려`→`한복고려`). v4는 99.8% 재현 + 단일 단어 성능 무손실(+0.3%p, val acc 85.56→85.85%).

**GSV 실전(run16, 강남·수원 — 브루클린은 영어 전용 엔진이라 무관, EasyOCR/TrOCR 산출물은 run12와 바이트 동일 확인)**: paddle 강남 exact 44.8→45.3%/수원 45.9→43.6%(문자 변동 — 개선·악화 혼재), **WAR는 강남 +1.0%p·수원 +0.8%p 일관 개선**(유일한 공백 민감 지표). T7b 전역 48.0→47.3%. **공식 숫자는 v3 유지** (run16/17 산출물은 `artifacts/ocr_gt/ab_v4_spacecat/` 보관, v4 모델은 `output/paddle_signboard_rec_v4_spacecat/`).

**변동 대역 실측 확정 (v3b, run18)**: v3 레시피를 **그대로** 재학습(v3b, 패치 off)한 결과 — val acc 85.18%(v3 85.56/v4 85.85), GSV per-box 강남 43.9%(−0.9)·수원 **43.6%(−2.3, v4와 동일 낙폭·동일 값)**. 즉 **같은 레시피 재실행만으로 지역 exact가 ±1~2.3%p 흔들리며**(yml seed 미고정 + GPU 비결정성), D18의 수원 하락은 개입 효과가 아니라 학습 변동임이 확정. 세 모델 비교에서 v4가 in-domain 최고(85.85)·양 지역 WAR 최고(0.256/0.241)·공백 능력 유일 보유로 **실질 최선 모델**. per-box 공식 수치는 안정성 위해 v3 유지하되, 라인 병합(4.9) 채택 시 v4 사용 권장. 산출물: run18 → `ab_v4_spacecat/` 보관.

**해석 — 왜 실전 이득이 안 잡히나**: ① 박스 파이프라인은 CRAFT가 이미 단어 단위로 잘라 모델이 다단어 입력을 거의 안 받고, 라인 공백은 파이프라인의 박스 조인에서 생성 ② 평가 정규화(norm_cer)가 공백을 제거해 exact/CER에 공백 능력이 미반영 ③ ZOO COFFEE류 로고는 시각적 갭이 없어 v4 학습과 일관되게 무공백 출력. **시사점**: 라인 단위 인식(박스 병합·캘리브레이션 선택)으로 파이프라인이 진화하면 다단어 박스가 흔해져 v4가 필요해짐 — 그때 채택할 검증된 모델을 확보한 상태. seed 고정(Global.seed) 재학습으로 변동/개입 효과 분리 가능.

### 4.7 라인 병합 인식 — per-box 병목의 발견 (D19) → **공식 채택 (D20)**

4.8의 시사점("라인 단위로 가면 v4가 필요해진다")을 검증하기 위해, 동일 검출·필터(run12의 CRAFT 파라미터 그대로)에서 **각 라인의 박스들을 union bbox 스트립 하나로 병합해 Paddle에 통째로 인식**시켰습니다(`exp_line_merge.py`, 채점은 공식 eval_ocr_v2 코드 경로).

| Paddle 단독 | 강남 exact/CER | 수원 | 브루클린(en) | 전역 환산 exact |
|---|---|---|---|---|
| per-box (run12 공식) | 44.8% / 0.502 | 45.9% / 0.395 | 52.3% / 0.364 | 47.6% |
| **라인 병합 v3** | **65.6% / 0.229** | **55.8% / 0.283** | **72.4% / 0.153** | **≈64.9%** |
| 라인 병합 v4 | 65.6% / 0.229 (WAR 0.266→**0.302**) | 54.7% / 0.283 (WAR 0.241→**0.253**) | (en 무관) | ≈64.7% |

**판독:**
- **단어 박스 조각내기가 최대 병목이었음**: 라인 병합만으로 Paddle 단독이 3-엔진 T7b 앙상블(48.0%)을 +17%p 상회. CER 절반 이하(0.41→0.21대). 기존 "line-oracle 상한 53.4%"는 per-box 후보의 상한이었을 뿐 — 후보 자체를 바꾸면 그 위로 감. 원인: 통라인 입력이 인식기의 학습 분포(라인)와 평가 granularity(GT 라인)에 동시에 부합 + per-box FP 삽입 페널티 소멸.
- **v4(spacecat)의 가치가 여기서 발현**: exact/CER은 공백 무시 정규화라 v3=v4이지만, **WAR는 v4가 +2.5%p(강남)·+1.2%p(수원)** — 4.8의 예측("라인 단위 전환 시 v4 필요") 그대로. 라인 병합 채택 시 v4가 권장 모델.
- 산출물(실험): `artifacts/ocr_gt/ab_v4_spacecat/ocr_{region}_linemerge_{v3,v4,en}.csv`.

**남은 손실의 성격 (`exp_suwon_det.py` 진단):** 가장 어려운 수원 기준, 완전 손실 라인의 다수는 예측에 흔적조차 없는 검출 실패이며 유형은 **세로 배치**(`야/광`), **양각 저대비**(`포장마차`), **캘리그래피**(`7형제 감자탕`·`엽기떡볶이`)입니다. 이 유형은 CRAFT 임계값 조정으로는 회수되지 않고, 구조가 다른 검출기를 더하는 것(4.9)이 유효한 경로였습니다.

**공식 채택 (D20, run19/20):** 전용 러너 `run_ocr_line.py`(지역별 모델 자동: 강남·수원 v4_spacecat / 브루클린 en_PP-OCRv5)로 정식 run19 paddle 생성, **배포 = 라인 병합 Paddle 단독**. 공식 전역: **exact 64.5% / CER 0.203 / WAR 0.440 / recall 86.3%** (구 per-box 배포 48.0/0.410/0.381/63.5). 새 crop-oracle **68.8%**, line-oracle **69.4%/CER 0.151**.

### 4.8 실제 라인 데이터 학습 — Paddle v5_lines (D21, 공식 채택)

라인 병합 파이프라인(4.9)의 입력은 다단어 라인인데, v4는 라인을 **합성 concat**으로만 배웠고 그 공백 능력이 실제 라인에 거의 전이되지 않는다는 것이 확인됐습니다(아래 표: 실라인 공백 재현 15.8%). 원천 어노테이션(단어 박스 + 원본 사진)에서 **실제 라인 크롭**을 구축해 학습:

- `build_line_crops.py`: 수직 겹침 ≥0.5 + 높이비 ≤2.5 클러스터링(중심거리 기준은 작은 윗줄+큰 아랫줄을 오합침) → 큰 수평 갭에서 분리 → union bbox + 4% 패딩 크롭, 라벨 = 공백 결합. **높이 96px 캡 저장**(학습은 h≤64만 사용 — 원본 해상도 저장 시 DataLoader가 3.6배 느려짐). train 13,148 / val 1,725 / test 1,631 라인 (source-image 단위 분할 상속 — 무누수).
- v5 = 단어 62,464 + 실라인 13,148 혼합, 나머지 v4 레시피 동일(공식 사전학습 시작점·6에폭·spacecat). 학습 실측 3h + 재시작 오버헤드.

**in-domain (test 분할, 세 모델 동일 이미지):**

| | 단어 exact | 합성쌍 exact | **실라인 exact** | 실라인 nospace | 실라인 CER | 실라인 공백재현 |
|---|---|---|---|---|---|---|
| v3 | 85.9% | 0.0% | 0.1% | 63.9% | 0.121 | 0.1% |
| v4_spacecat | 86.2% | 68.9% | 7.9% | 63.9% | 0.124 | 15.8% |
| **v5_lines** | 85.6% | 68.4% | **55.7%** | **69.5%** | **0.108** | **85.9%** |

**핵심 발견**: 합성 concat(v4)의 공백 능력은 합성 분포에만 유효(합성쌍 99.8% ↔ 실라인 15.8%) — **실데이터 학습만이 실제 라인에 전이**됩니다. v5는 공백뿐 아니라 내용 판독도 개선(nospace +5.6%p, CER −0.013). 단어 단독 85.6%는 재실행 변동 대역(85.2~86.2, 4.8) 안.

**GSV 실전 (run21, 라인 병합):** 강남 exact 65.6→64.2%(−1.4, 변동 대역)·**WAR 0.302→0.610**, 수원 54.7% 동일·**WAR 0.253→0.498** — 공백 무시 지표(exact/CER)는 동등, **단어 단위 정확도(WAR)가 두 배**. **공식 채택**: 전역 **exact 64.0% / CER 0.204 / WAR 0.605** (v4 대비 exact −0.5%p(변동 내) / WAR **+16.5%p**). 배포 기본 모델 = v5_lines (`run_ocr_line.py` REGION_CFG). 배포 WAR(0.605)가 oracle WAR(0.568)보다 높음 — oracle은 exact 우선 선택이라 WAR 최적이 아님.

### 4.9 검출기 union — CRAFT ∪ PaddleOCR DB (D23, 공식 채택)

CRAFT 임계값 조정으로는 회수되지 않던 전멸 라인(4.7)을, 구조가 다른 **PaddleOCR DB(세그멘테이션 기반)** 검출기가 잡아냅니다. CRAFT 박스와 겹치지 않는(면적 IoU-min > 0.3 기준) DB 박스만 추가:

| 지역 | exact | CER | WAR | contain | 전멸 라인 |
|---|---|---|---|---|---|
| 강남 | 64.2 → **66.5%** | 0.239 → **0.211** | 0.610 → 0.620 | 70.3 → 72.6% | 32 → **23** |
| 수원 | 54.7 → **56.4%** | 0.278 → **0.257** | 0.498 → 0.523 | 61.3 → 65.2% | 39 → **28** |
| 브루클린 | 72.4 → **74.4%** | 0.153 → **0.117** | 0.668 → 0.676 | 77.9 → 80.9% | 11 → **7** |

**DB 단독은 오히려 나쁨**(강남 62.3 / 수원 48.1 / 브루클린 68.3%) — **union일 때만** 이득이 나므로 두 검출기가 서로 다른 실패 모드를 보완한다는 해석이 정확합니다. 공식 채택(run22, `run_ocr_line.py` 기본값 · `--no-det-union`으로 비활성): 전역 **exact 64.0→66.0% / CER 0.204→0.174 / recall 86.1→90.2%**. 격리 워커 `paddle_det_worker.py` 신설(torch/paddle DLL 충돌 회피, rec 워커와 동일 사유).

**스트립 패딩 최적화 (D24 ④, +2.8%p).** `crop_strip`의 여백은 **스트립 자체의 폭 비례**라, 가로로 긴 라인일수록 좌우로 폭의 12%씩 확장돼 **옆 간판 텍스트를 끌어옵니다**. 양방향 스윕 결과 0.04가 최적:

| pad | 0.0 | **0.04** | 0.08 | 0.12(구) | 0.18 | 0.25 | 0.35 |
|---|---|---|---|---|---|---|---|
| exact | 67.2 | **68.8** | 66.0 | 66.0 | 62.7 | 61.7 | 59.5 |
| CER | 0.170 | **0.165** | 0.171 | 0.173 | 0.187 | 0.190 | 0.204 |

여백은 **작을수록** 좋고 0.04가 최적점입니다(높이 기준 비대칭 패딩은 개선 없음).

**라인 그룹핑 y_tol 보정 (D24, +0.5%p).** 같은 성격의 두 번째 기하 결함입니다. 완전 전멸 라인 56개를 분해하니 **24개(43%)가 인식 실패가 아니라 과잉 병합**이었습니다 — GT는 `VIP`/`노래방`, `LOTTERIA`/`롯데리아`처럼 두 줄인데 y-중심 허용오차(0.06×이미지높이)가 커서 한 스트립으로 합쳐지고, 평가의 1:1 라인 매칭에서 나머지 GT 라인이 통째로 0점 처리됩니다. 스윕 결과 **0.04가 최적**(69.3%), 그 아래는 과분할로 악화(0.03: 68.1%). 나머지 32개(57%)는 진짜 미검출(`딸부자네`·`솔로몬`·단일 문자 `7`)로 그룹핑으로는 회수 불가.

> ⚠️ **튜닝 한계 (논문 명시 필요):** pad·y_tol 두 **기하 스칼라를 GSV 평가셋에서 직접 튜닝**했고 별도 홀드아웃이 없습니다. 592라인 기준 +0.5%p ≈ 3라인이라 y_tol의 이득은 노이즈와 구분이 어렵습니다. 완화 근거는 ① 두 파라미터 모두 곡선이 단봉·매끄러움 ② 패딩(+2.8%p)은 **코드 결함**(여백이 스트립 폭 비례 → 긴 라인에서 옆 간판 침범)을 고친 것이라 이득의 기전이 명확하다는 점입니다.

**라인 레벨 3-way 다수결 (D24 ⑤, +0.6%p).** 한글권 스트립을 v5·v4·zero-shot 세 모델로 읽고, 정규화 후 **2개 이상 일치하면 그 텍스트, 아니면 v5**(767라인 중 35라인 교체). **모든 후보가 동일한 라인 스트립을 읽으므로 합의가 의미를 가집니다** — per-box 시대에는 후보들이 텍스트가 아니라 granularity를 두고 불일치해 같은 규칙이 성립하지 않았습니다. 임계값 튜닝이 없는 원칙적 규칙이며 WAR도 보존(0.642→0.648)됩니다.

### 4.18 텍스트(word) 박스 탐지 hold-out — 4 파이프라인 동일 계열 비교 (D34)

**질문:** "간판 탐지에 YOLO26x 를 쓰면 텍스트 박스 탐지도 YOLO26x 로 학습한 것의 mAP 를 보여 달라" —
즉 **탐지 2단계를 같은 모델 계열로 묶은 파이프라인 4개**의 수치. 4.3.1 의 텍스트 탐지는 Signboard
5,000장 5-fold 였고 YOLO 는 Ultralytics val, FRCNN/EffDet 는 자체 AP 로 채점한 **평가기 혼용** 표였습니다.
여기서는 **OCR 학습 데이터(텍스트 박스 JSON) 전체**로 4모델을 다시 학습하고 **하나의 AP@0.5 구현**으로 채점합니다.

**데이터·분할 (`prep_text_det_holdout.py` → `artifacts/signboard_text_holdout`).** 원천은
`artifacts/signboard_data_info.json` 의 27,519 장(word 박스). **분할은 새로 만들지 않고 OCR 인식기의
signboard_v3 소스 분할을 파일 stem 으로 그대로 상속** — 인식기 train/val/test 와 같은 이미지가 같은 split 에
갑니다. 이미지는 하드링크, 라벨은 YOLO txt(단일 클래스 `text`), **디스크 증강 0** (YOLO 는 Ultralytics
학습 루프 안 mosaic/hsv 만, FRCNN·EffDet 는 무증강 — 5k 실험과 동일).

| split | 이미지 | word 박스 | 비고 |
|---|---|---|---|
| train | 21,714 | 62,469 | |
| val | 2,712 | 7,885 | 조기종료 기준 |
| test | 2,706 | 7,756 | **채점은 여기서 1회** |
| 제외 | 387 | — | v3 분할 없음 2 + word 박스 없음 385 |

**섞임 감사 (2026-09-14, 통과):** train/val/test 파일 stem 쌍별 교집합 **0/0/0**, `split_source.csv`
1행/파일(중복 0), 목록 = 디스크 이미지 수 = 라벨 수(21,714/2,712/2,706). 분할 → 학습 순서이며
증강 파일은 존재하지 않습니다(4.17 감사 O1~O4 와 같은 규칙).

**학습 (`run_text_holdout.sh`, 순차, 총 58h, 2026-09-11 22:18 → 09-14 08:36).** 하이퍼파라미터는 5k
실험과 동일: YOLO imgsz 640·batch 4·Adam 1e-4, FRCNN min 640·SGD 0.005·batch 4, EffDet 512 letterbox·
Adam 5e-4·batch 8·AMP; 전 모델 epochs 60 / patience 15 / val AP@0.5 로 best 선택.

| 모델 | 에폭 (best) | 학습 시간 | val best AP@0.5 |
|---|---|---|---|
| YOLO26x | 60 (52) | 22.4h | 0.833 |
| YOLOv5x | 60 (58) | 15.5h | 0.820 |
| Faster R-CNN | 32 조기종료 (17) | 16.9h | 0.718 |
| EfficientDet-D0 | 26 조기종료 (11) | 3.2h | 0.666 |

**결과 — 4 파이프라인 표 (`eval_text_holdout.py` → `artifacts/gt/text_holdout_ap50.csv`).** 간판 열은 4.17 의
pooled AP@0.5(298장 OOF), 텍스트 열은 test 2,706장 단일 채점(score_thr 0, `compute_ap50` 공통).

| 파이프라인 (간판 → 텍스트 박스, 같은 계열) | 간판 pooled AP@0.5 | **텍스트 박스 test AP@0.5** | 5k 5-fold (4.3.1, 참고) |
|---|---|---|---|
| **YOLO26x → YOLO26x** | 0.860 | **0.822** | 0.746 |
| YOLOv5x → YOLOv5x | 0.806 | **0.809** | 0.737 |
| Faster R-CNN → Faster R-CNN | 0.854 | **0.715** | 0.638 |
| EfficientDet-D0 → EfficientDet-D0 | 0.828 | **0.672** | 0.578 |

**해석:**
1. **텍스트 박스 순위는 YOLO26x > YOLOv5x ≫ FRCNN > EffDet** 로 5k 실험과 동일하며, 데이터 5.4배에
   전 모델 **+0.07~0.09** 올랐습니다. 이번엔 4모델을 같은 채점기로 봤으므로 이 순위는 평가기 차이가 아닙니다.
2. **간판 탐지와 텍스트 탐지의 순위가 다릅니다.** 간판(큰 객체, 장당 ~2개)에서는 FRCNN ≈ YOLO26x 가 동등 최상위지만,
   텍스트(작고 밀집, 장당 2.9개, 대부분 이미지 폭의 수 %)에서는 2-stage FRCNN(0.715)과 512 입력 EffDet(0.672)가
   YOLO 계열보다 0.1 이상 뒤집니다. **두 단계 모두에서 최상위인 모델은 YOLO26x 뿐** — 배포 파이프라인의
   YOLO26x 선택 근거는 이 표입니다.
3. val→test 차이가 −0.011 ~ +0.006 으로 작아 조기종료 기준(val)의 과적합은 없습니다.
4. **주의(논문 서술):** 이 표의 텍스트 탐지기는 "같은 계열" 비교용이고, **배포 OCR 파이프라인의 텍스트 검출은
   CRAFT ∪ PaddleOCR-DB 라인 검출(4.14)** 입니다. 둘은 단위(word 박스 AP vs 라인 recall)가 달라 직접 비교하지
   않습니다. 표 1(모델 비교)에는 이 4.18 열을, 표 3(지역별)에는 4.17 지역 pooled 값을 씁니다.
5. 5k 표(4.3.1)는 평가기 혼용 + 데이터 1/5 이므로 **논문에서는 이 hold-out 값으로 대체**합니다.

**산출물:** `artifacts/{yolo26x,yolov5x}_text_holdout/run/weights/best.pt`, `artifacts/{frcnn,effdet}_text_holdout/best_*_text.pth` + `val_log.csv`/`summary.csv`,
로그 `artifacts/signboard_text_holdout/logs/`, 워드 `Desktop/GSV_results_v5.docx` (Table 4 추가).

### 4.19 OCR 비교군 확장 — PaddleOCR 와 같은 체급의 인식기 (D35)

**배경.** 기존 OCR 비교표(4.4)의 EasyOCR·TrOCR 는 PP-OCRv5 와 체급이 다릅니다(2015년 CRNN / 인쇄 문서용 모델).
PaddleOCR 비교 논문들이 실제로 쓰는 상대는 세 부류입니다 — ① 범용 오픈소스 엔진(Tesseract·EasyOCR·Surya·MMOCR·docTR),
② 학술 STR 인식기(PARSeq·ABINet·SVTR 계열·MAERec·CLIP4STR), ③ VLM(Qwen2.5-VL·GOT-OCR 등; 우리 표 3 의 "VLM alone").
이 절은 ①에서 Tesseract·Surya, ②에서 PARSeq·SVTRv2, ④ **상용 API** 에서 NAVER CLOVA OCR General 을 추가한 결과입니다.

**학습 (PARSeq·SVTRv2, 분할→학습, 섞임 0).** 배포 Paddle v5_lines 와 **같은 학습 혼합**(단어 62,464 + 실라인 13,148 =
75,612 크롭, signboard_v3 분할 상속; `prep_str_baselines_data.py` → `artifacts/str_baselines/parseq_data` LMDB), val 9,609 으로
best 선택, test 는 단어 7,756 / 실라인 1,631 을 따로 채점. 감사: train/val/test 원천 이미지 교차 **0** (`audit.txt`). 사전은
PP-OCRv5 한국어 사전 11,945자 + 학습에 등장한 전각 문자 32자(`dict/korean_v5_plus.txt`)를 두 모델이 공유.

| 모델 | 구현 | 초기값 | 입력 | 학습 | val best |
|---|---|---|---|---|---|
| **SVTRv2-B** (ICCV 2025) | OpenOCR `svtrv2_rctc_signboard.yml` | Union14M-L SVTRv2-B 인코더(분류층 재초기화, `make_svtrv2_pretrained.py`) | h32 가변폭(max_ratio 12) | 20 ep, bs128, AdamW 2e-4 OneCycle, AMP | acc 90.1% (ep20) |
| **PARSeq** (ECCV 2022) | 공식 repo `train_parseq_signboard.py`(형상 일치 키만 로드) | 공식 parseq(영어 94자) → text_embed/head 재초기화 | 32×128 고정 | 20 ep, bs128, lr 3e-4(×bs/256), 16-mixed | exact 81.7% (ep20) |
| Tesseract 5.5.3 | conda-forge, kor+eng, `--psm 7` | 기성 | 스트립 h≥48 확대 | — | — |
| **Tesseract 미세조정** (kor_signboard) | WSL Ubuntu tesseract 4.1.1 학습 도구(lstmtraining), 추론은 Windows 5.5.3 | `tessdata_best/kor` + 문자표 병합(1,158→1,480자, 영문·한글 310자 추가, 출력층 확장) | 라인 lstmf(WordStr box), 63,301/75,612장 변환 성공 | 16.6만 반복, lr 2e-4, CPU 4h | val2k 문자오류 28.5% / 단어오류 53.7% |
| Surya 0.14.6 | 별도 venv(transformers 4.51, CPU), rec2 | 기성(학습 코드 비공개) | 스트립 전체 bbox 1개 | — | — |

**측정 조건.** GSV 는 배포 파이프라인과 **검출·라인 병합이 완전히 동일**(CRAFT ∪ PaddleOCR-DB, `run_ocr_line.py --worker …`,
DB 추가 박스 537 / 스트립 1,342 로 전 엔진 일치)하고 인식기만 교체. 채점은 `eval_ocr_v2.py --mask-phone --en-only-regions brooklyn`
(592 라인, 라인 매칭, FP 라인은 삽입으로 가산) 에 **WER(단어 편집거리/GT 단어)** 를 추가. PP-OCRv5 행은 배포 구성
(강남·수원 v5/v4/zero-shot 3-way vote, 브루클린 en_PP-OCRv5) 재실행(run 44)이고 PARSeq·SVTRv2 는 **단일 모델을 세 지역에 공통** 적용.

**GSV — line exact / CER / WER (`artifacts/str_baselines/ocr_baselines_tables.md`, 워드 Table 5):**

| 엔진 | 부류 | 강남 | 브루클린 | 수원 | **전체** | WAR |
|---|---|---|---|---|---|---|
| Tesseract 5.5 | 범용 기성 | 34.0% / 0.551 / 1.033 | 42.7% / 0.376 / 1.274 | 13.3% / 0.769 / 1.403 | 30.6% / 0.509 / 1.228 | 0.333 |
| Tesseract 5.5 (kor **미세조정**+eng) | 범용, 미세조정 | 35.9% / 0.526 / 0.971 | 42.7% / 0.370 / 1.180 | 18.8% / 0.699 / 1.245 | 32.9% / 0.484 / 1.128 | 0.287 |
| EasyOCR (v3 미세조정, per-box) | 범용 | 37.3% / 0.538 / 1.459 | 43.2% / 0.406 / 2.110 | 34.8% / 0.428 / 1.510 | 38.5% / 0.449 / 1.741 | 0.290 |
| Surya 0.14 | 범용 기성 | 41.0% / 0.502 / 1.167 | 60.3% / 0.198 / 0.760 | 17.1% / 0.650 / 1.137 | 40.2% / 0.381 / 0.991 | 0.396 |
| **CLOVA OCR General** (NAVER) | **상용 API, 제로샷** | 57.1% / 0.318 / 0.967 | 67.8% / 0.171 / 0.859 | 48.1% / 0.383 / 1.153 | **57.9% / 0.258** / 0.971 | 0.563 |
| TrOCR-small (v3 미세조정, per-box) | 문서 STR | 26.9% / 0.635 / 1.469 | 46.2% / 0.384 / 2.154 | 27.6% / 0.516 / 1.510 | 33.6% / 0.485 / 1.762 | 0.343 |
| PARSeq (미세조정) | 학술 STR | 63.7% / 0.248 / 0.652 | 70.9% / 0.151 / 0.483 | 50.3% / 0.299 / 0.830 | 62.0% / 0.210 / 0.629 | 0.619 |
| **SVTRv2-B (미세조정)** | 학술 STR | 72.6% / 0.209 / 0.593 | **77.4% / 0.096 / 0.433** | **60.8% / 0.249** / 0.743 | **70.6% / 0.161** / 0.566 | **0.682** |
| PP-OCRv5 rec (v5_lines, **단일 모델**, vote 없음) | 학술/배포 STR | 71.7% / 0.194 / 0.554 | 70.9% / 0.122 / 0.478 | 58.0% / 0.256 / 0.693 | 67.2% / 0.171 / 0.559 | 0.643 |
| PP-OCRv5 rec (v5_lines, **배포 구성**: 3-way vote + en 모델) | 배포 | **73.1% / 0.190 / 0.534** | 76.9% / 0.110 / **0.452** | 58.6% / 0.255 / **0.689** | 69.9% / 0.163 / **0.540** | 0.648 |

**In-domain (signboard_v3 test, 1:1 크롭, exact / CER / WER; `eval_str_indomain.py`, 워드 Table 6):**

| 엔진 | test_line (실라인 1,631) | test_word (단어 7,756) |
|---|---|---|
| Tesseract 5.5 | 21.0% / 0.602 / 0.815 | 34.4% / 0.571 / 0.789 |
| Tesseract 5.5 (kor 미세조정+eng) | 24.9% / 0.567 / 0.940 | 42.4% / 0.486 / 0.643 |
| Surya 0.14 | 36.4% / 0.793 / 0.802 | — (CPU 4h 소요라 생략) |
| **CLOVA OCR General** (NAVER) | **67.3% / 0.233 / 0.502** | 66.5% / 0.271 / 0.628 |
| PARSeq | 61.4% / 0.153 / 0.344 | 89.2% / 0.047 / 0.130 |
| **SVTRv2-B** | **75.3% / 0.081 / 0.256** | **94.2% / 0.023 / 0.077** |
| PP-OCRv5 rec (v5_lines) | 71.8% / 0.102 / 0.336 | 87.2% / 0.070 / 0.146 |

**해석:**
1. **같은 조건(단일 모델·같은 학습 데이터·세 지역 공통)에서는 SVTRv2 70.6% > PP-OCRv5 67.2%** (+3.4%p ≈ 20라인, CER 0.161 vs 0.171).
   배포 구성(3-way vote + 브루클린 en_PP-OCRv5)까지 얹은 Paddle 69.9% 와 비교해도 SVTRv2 단일 모델이 동급 이상입니다(차이 0.7%p 는 변동 대역 안). In-domain 에서는
   SVTRv2 가 명확히 앞섭니다(단어 94.2 vs 87.2%, 실라인 75.3 vs 71.8%). 이는 **PaddleOCR 선택이 "성능 최상"이 아니라
   "동급 최상위 중 배포 편의(PaddleX 추론 포맷·다국어 모델 교체·CPU 경량)"** 라는 서술로 바꿔야 함을 뜻합니다 — 프레임워크 논문에는
   오히려 유리한 결과(인식기 교체 가능성 실증).
2. **PARSeq 는 8%p 아래**(62.0%). 32×128 고정 입력이 W/H≈5 의 긴 한글 라인을 압축하는 구조적 손해로 보입니다(단어 test 에서는
   89.2% 로 Paddle 보다 높음 — 짧은 텍스트에서는 동급, 라인에서 하락).
3. **범용 기성 엔진(Tesseract 30.6%, Surya 40.2%)은 EasyOCR 수준**입니다. 한글 간판 도메인 데이터 없이 나오는 성능이 이 정도라는
   기준선이며, EasyOCR·TrOCR 가 낮았던 것도 "구세대·타 도메인" 부류의 문제이지 측정 오류가 아닙니다. Surya 는 브루클린(영어)
   에서 60.3% 로 범용 엔진 중 최고.
4. **Tesseract 는 같은 데이터로 미세조정해도 30.6 → 32.9%(in-domain 단어 34.4 → 42.4%)에 그칩니다.** 문자 오류는 줄지만(0.509 → 0.484)
   라인 인식기 이전 단계(텍스트 라인 검출)가 간판 크롭의 상당수에서 텍스트를 못 잡아(empty 36/411 라인 동일) 상한이 낮습니다. 즉 EasyOCR·TrOCR·Tesseract 가
   낮은 것은 학습 부족이 아니라 아키텍처(문서 OCR 전제)의 문제라는 점이 미세조정으로 확인됩니다.
5. WER 는 FP 라인 삽입을 세는 규칙 때문에 1 을 넘을 수 있습니다(CER 과 동일 규칙). 표에서는 CER 와 함께 읽어야 합니다.

**함정 기록.** ① OpenOCR 추론기의 배치 패딩(`batch_num>1`)은 학습(RatioSampler: 같은 비율끼리 묶음, 패딩 없음)과 달라 CTC 끝에
글자가 중복됐고('CAFE'→'CAFEE'; 단어 exact 65.7%) — 1장씩 추론으로 수정(94.2%). OpenOCR 자체 평가기(norm-ED 0.976)로 교차 확인.
② Windows 에서 OpenOCR DataLoader 워커는 LMDB/증강 함수 pickle 불가 → `num_workers 0` (학습 속도 절반). ③ 학습 중 같은
GPU 에 검출 작업을 올리면 SVTRv2 가 segfault — GPU 작업은 `run_ocr_baselines_all.sh` 로 완전 직렬화. ④ C: 드라이브가
두 번 가득 참(pagefile 64GB 팽창, WSL vhdx 138GB) → 체크포인트 0바이트·학습 중단; SVTRv2 는 7에폭 체크포인트에서 재개, PARSeq 는 재학습.
⑤ Tesseract 미세조정: (a) UB-Mannheim 서버 불통 → WSL Ubuntu apt tesseract 4.1.1 학습 도구 사용(모델 형식은 5.x 와 호환); (b) 4.1 의 `lstm.train` 은 `.gt.txt` 가 아니라
tesstrain 식 `WordStr` box 파일을 요구(없으면 lstmf 0개, 오류 없음); (c) `combine_lang_model --pass_through_recoder` 로 문자표를 만들면 원본 kor 의 자모 recoder 와 달라
출력층이 이어지지 않아 1시간 동안 오류율 78% 정체 → recoder 기본값(자모 분해)으로 재생성하니 정상 학습; (d) 새 traineddata 에 `kor.config`(`preserve_interword_spaces 1`)를
넣지 않으면 Windows Tesseract 5 가 한글 음절마다 띄어 써 WER 1.77 → config 내장 후 0.88. 산출물 `artifacts/str_baselines/tesseract_ft/`(traineddata, lstmtraining 로그, 병합 문자표).

**상용 API — NAVER CLOVA OCR General (D37).** Tesseract·Surya 와 같은 **범용 기성 엔진** 취급입니다. 배포 파이프라인과
동일한 라인 스트립을 그대로 API 에 넘기고(검출·라인 병합 고정, 인식기만 교체), 응답 `fields[]` 를 이어 붙여 라인을 만듭니다
(`str_baselines/clova_rec_worker.py` — 다른 워커와 같은 IO 계약). **제로샷**입니다 — 이 API 는 미세조정 경로가 없어
우리 간판 데이터를 학습시키지 못합니다. 총 API 호출 10,734건(GSV 1,342 · test_line 1,631 · test_word 7,756 · 스모크 5).

| 구간 | exact | CER | 같은 구간 최고(미세조정) |
|---|---|---|---|
| GSV 전역 592라인 | 57.9% | 0.258 | SVTRv2 70.6% / 0.161 |
| in-domain 실라인 1,631 | 67.3% | 0.233 | SVTRv2 75.3% / 0.081 |
| in-domain 단어 7,756 | 66.5% | 0.271 | SVTRv2 94.2% / 0.023 |

**해석 — 이 실험이 답하는 질문은 '상용 API 를 그냥 쓰면 되지 않는가'입니다.**
1. **기성 엔진 중에서는 압도적입니다.** GSV 57.9% 로 Surya(40.2)·EasyOCR(38.5)·Tesseract(30.6)를 17%p 이상 앞섭니다.
   한국어 간판을 학습 없이 읽는 성능만 놓고 보면 공개 엔진과 상용 API 사이에 큰 격차가 있습니다.
2. **그럼에도 도메인 미세조정 모델에는 미치지 못합니다.** 같은 스트립에서 SVTRv2 70.6% · 배포 PP-OCRv5 69.9% 로 12%p 앞서고,
   단어 크롭에서는 격차가 66.5 vs 94.2% 로 벌어집니다. **간판 도메인 데이터로 직접 학습한 것이 옳았다**는 정량 근거입니다.
3. **격차가 가장 큰 곳이 단어 크롭**인 이유는 CLOVA 가 검출+인식 일체형이라, 단어 하나만 담긴 타이트한 크롭에서 오히려
   텍스트 영역 판정이 불리해지기 때문입니다(GSV 411크롭 중 빈 출력 18건 — Paddle 3건). 라인 단위 입력에서는 이 손해가 줄어듭니다.
4. 지역 순서는 다른 엔진과 같습니다(브루클린 67.8 > 강남 57.1 > 수원 48.1). 수원의 양각·캘리그래피 간판은 상용 API 에도 어렵습니다.

**운영 메모.** 유료 API 라 워커에 크롭 단위 체크포인트(재실행 시 재과금 없음), `--max-calls` 상한, `--dry-run`,
엔드포인트 사전 점검을 넣었습니다. 인증 정보는 환경변수 또는 git 제외 파일에서만 읽습니다.

```bash
bash scripts/run_clova_ocr.sh smoke     # 5건만 호출해 응답 확인
bash scripts/run_clova_ocr.sh gsv       # GSV 라인 스트립
bash scripts/run_clova_ocr.sh line      # in-domain 실라인
bash scripts/run_clova_ocr.sh word      # in-domain 단어
bash scripts/run_clova_ocr.sh report    # 표 재생성
```

**함정 기록(D37).** ① 콘솔에는 공인 APIGW Invoke URL 과 VPC 내부 주소가 함께 보이는데, 내부 주소
(`clovaocr-api-kr.ncloud.com` → 10.x)를 쓰면 외부망에서 영영 닿지 않습니다 → 워커에 DNS/연결 사전 점검 추가.
② CLOVA 는 글자 간격이 넓으면 **같은 줄도 `lineBreak` 로 쪼개** 내놓습니다. 우리 프로토콜은 입력 1장 = 라인 1줄이라
그대로 두면 없는 라인을 삽입한 것으로 채점돼 CLOVA 만 손해를 봅니다 → Tesseract(`--psm 7`)·Surya 와 같게 한 줄로 합칩니다.
③ 드물게 정상 이미지에 HTTP 400 이 돌아오는데 **재시도하면 바로 성공**합니다(게이트웨이 순간 제한). 처음에는 한 건 실패로
전체를 중단시켜 단어 세트 4,584건이 빈 값이 됐습니다 → 400 도 재시도하고, **연속** 15건 실패에만 중단하도록 변경.
④ VLM 연쇄 실험이 결과를 `ocr_<지역>_5{1,2}_paddle.csv` 로 저장해, 엔진별 최신 파일을 고르는 `eval_ocr_v2` 가 배포 행을
탐지 크롭 결과로 잘못 채점했습니다(강남 73.1 → 37.7%) → 연쇄 산출물을 `vlmfix` 태그로 분리.

**산출물:** `artifacts/str_baselines/{svtrv2/best.pth, parseq/checkpoints/*.ckpt, indomain_summary.csv, ocr_baselines_tables.md}`,
GSV 예측 `artifacts/ocr_gt/ocr_<region>_{40 tesseract,41 surya,42 parseq,43 svtrv2,44 paddle}.csv`, 워커 `{tesseract,surya,openocr,parseq}_rec_worker.py`,
러너 `run_str_baselines.sh` / `run_ocr_baselines_all.sh`, CLOVA `artifacts/str_baselines/{indomain/clova_*.jsonl, clova_ocr/}` · GSV 예측 `artifacts/ocr_gt/ocr_<region>_47_clova.csv`,
워드 `Desktop/GSV_results_v9.docx` (Table 5·6; v7 = CLOVA 추가 전).

---

## 5. 학습 횟수 · 실험 이력

### 5.1 탐지 학습 실험 (`runs/detect/`)

| 실험 | 모델 | 에폭 | 배치 | imgsz | 디바이스 | 최종 mAP@0.5 |
|------|------|------|------|-------|---------|-------------|
| train | yolov8m | 50 | 32 | 640 | — | (미완료) |
| train2 | yolo11x | 50 | 32 | 640 | — | (results 없음) |
| train3 | yolo11n | 50 | 32 | 640 | CPU | 0.7629 (재현율 0.725) |

### 5.2 교차검증 본학습 (`artifacts/`)

- **YOLO11x**: 5 folds × (초기학습 + 이어학습 +50ep) = 10회 학습 사이클
- **Faster R-CNN**: 5 folds × 조기종료(best epoch 4~32)
- **TrOCR**: 데이터셋별(`signboard`, `signboard_full`, `textinthewild`, `_small`) × 5 에폭, 에폭별 체크포인트 저장
- **EasyOCR(CRNN)**: 5-Fold (`external/EasyOCR/trainer/saved_models/cv_fold{0-4}`, signboard/_aug/_full/_long)
- **PaddleOCR(rec)**: 1차 3ep + 이어학습 4ep (`output/paddle_signboard_rec` → `_v2`)
- 학습된 가중치 보관 위치:
  - `artifacts/yolo11x_kfold/runs/total_fold{0-4}/weights/best.pt`
  - `artifacts/frcnn_kfold/fold{0-4}/best_frcnn.pth`
  - `artifacts/ocr_training/{dataset}/trocr_model[_best]/model.safetensors`
  - `external/EasyOCR/trainer/saved_models/*/best_accuracy.pth` (+ `make_easyocr_plugin.py`로 플러그인화)
  - `output/paddle_signboard_rec_v2/best_accuracy(.pdparams)` + `inference/`(배포용 export)

---

## 6. 디렉터리 구조

```
main/
├── train_yolo26x_kfold.py           # ★ YOLO26x 5-Fold 미세조정 (주 탐지 모델)
├── train_yolov5x_kfold.py           # YOLOv5x 5-Fold (버전 비교, 동일 설정)
├── one_click_finetune_yolo11.py     # 공용 5-Fold 분할 생성 + YOLO11x 학습 (버전 비교)
├── resume_more_train_kfold_both.py  # YOLO11x + FRCNN 이어학습 (동일 split)
├── train_only_my_frcnn.py           # Faster R-CNN 5-Fold 학습
├── train_textinthewild_ocr.py       # TrOCR 데이터준비 + 미세조정
├── build_cv_folds.py                # OCR 5-Fold 분할 (누수 방지)
├── run_yolo_only.py / run_frcnn_only.py  # 탐지 추론
├── run_ocr_line.py                  # ★ 배포 OCR 파이프라인 (CRAFT∪DB 검출 + 라인 병합 인식)
├── run_ocr_only.py                  # 레거시 per-box 3-way 추론 (참조 후보 산출용)
├── paddle_rec_worker.py             # PaddleOCR 인식 격리 subprocess (torch/paddle DLL 충돌 회피)
├── paddle_det_worker.py             # PaddleOCR DB 검출 격리 subprocess
├── build_line_crops.py              # 실라인 학습 크롭 생성 (원천 단어박스 → 라인)
├── train_paddle_v4_spacecat.py      # PaddleOCR rec 학습 래퍼 (--config / --stock)
├── exp_line_tuning.py               # 라인 스트립 튜닝 실험 (패딩·라인레벨 앙상블)
├── make_easyocr_plugin.py           # 미세조정 EasyOCR(CRNN) → recog_network 플러그인
├── build_easyocr_trainer_data.py    # EasyOCR 트레이너 학습 데이터 생성
├── make_crops_from_gt_polygon.py    # 원근 보정 간판 크롭
├── json_to_gt_csv.py                # Labelme 폴리곤 → 사각형 GT
├── eval_map_only.py / compute_map.py # 탐지 mAP 평가
├── eval_ocr.py / eval_ocr_v2.py     # OCR 평가 (v2: 라인매칭 + per-engine/ensemble/oracle)
├── paddle_signboard_rec.yaml        # PaddleOCR 미세조정 config (PaddleX)
├── data.yaml / yolo26x.pt(주 모델) / yolov5xu.pt / yolo11x.pt / yolo11n.pt
├── external/EasyOCR/                # EasyOCR 트레이너 + 미세조정 가중치(saved_models)
├── output/paddle_signboard_rec_v5_lines/  # ★ 배포 인식 모델 (+ v2/v3/v3b/v4 계보)
├── runs/detect/{train,train2,train3,val}/   # YOLO 학습 로그
└── artifacts/
    ├── gsv_photo/{gangnam,brooklyn,suwon}/  # 원본 GSV + Labelme JSON
    ├── gt/                          # GT CSV, 크롭(gt/crop/{region}), OCR GT, ocr_eval_v2_*.csv
    ├── yolo_ft_total/               # COCO 변환 탐지 학습셋 (train 238 / val 60)
    ├── yolo26x_kfold/               # ★ YOLO26x 5-Fold 가중치·요약 (주 모델)
    ├── yolov5x_kfold/               # YOLOv5x 5-Fold 가중치·요약
    ├── yolo11x_kfold/               # YOLO11x 가중치·요약 + **공용 fold 분할**(total_fold{0-4}/dataset, 전 모델 공유 — 삭제 금지)
    ├── frcnn_kfold/                 # FRCNN 5-Fold 가중치·요약
    ├── ocr_training/{signboard,signboard_full,textinthewild,...}/  # OCR 학습셋·모델(TrOCR·easyocr export)
    ├── ocr_gt/                      # OCR 추론 결과 (ocr_{region}_{run}_{engine}.csv)
    ├── signboard_data_info.json     # 간판 데이터셋 메타 (27.5K img / 527K ann)
    └── textinthewild_data_info.json # TIW 메타 (100K img / 2.1M ann)
```

---

## 7. 실행 방법 (요약)

> ⚠️ **OCR 실행은 반드시 가상환경 파이썬 `.venv\Scripts\python.exe` 사용** (numpy 1.26 + easyocr + torch cu118, CUDA True). 기본 `python`(base anaconda)은 numpy ABI가 깨져 easyocr import에 실패합니다. 또한 **`transformers==4.49.0` + `tokenizers 0.21.x`** 고정(3.4 참고).

```bash
# 1) GT 준비: Labelme 폴리곤 → 사각형 CSV
python json_to_gt_csv.py

# 2) 공용 5-Fold 분할 생성 (+ YOLO11x 버전 비교 학습)
python one_click_finetune_yolo11.py --region total --kfold 5 \
       --export-best-to artifacts/yolo/best_yolo11x_kfold.pt

# 3) ★ YOLO26x 5-Fold 미세조정 (주 탐지 모델, 위 fold 분할 사용)
python train_yolo26x_kfold.py

# 3b) (선택) YOLOv5x 버전 비교 / YOLO11x 이어학습
python train_yolov5x_kfold.py
python resume_more_train_kfold_both.py

# 4) Faster R-CNN 베이스라인 학습
python train_only_my_frcnn.py

# 5) 탐지 추론 (주 모델 가중치)
python run_yolo_only.py --weights artifacts/yolo26x_kfold/fold0/weights/best.pt --region total

# 6) 간판 크롭 생성 (OCR 입력)
python make_crops_from_gt_polygon.py --gt-size auto --upscale 2.0

# 7) OCR 인식기 학습 — 배포 모델(PaddleOCR v5_lines) 기준
.venv/Scripts/python.exe build_line_crops.py                     # 실라인 학습 크롭 생성
.venv/Scripts/python.exe train_paddle_v4_spacecat.py --train  --config paddle_signboard_rec_v5_lines.yml
.venv/Scripts/python.exe train_paddle_v4_spacecat.py --export --config paddle_signboard_rec_v5_lines.yml
# (참조 인식기) TrOCR: train_textinthewild_ocr.py all ... (3.4)
# (참조 인식기) EasyOCR CRNN: build_easyocr_trainer_data.py → external/EasyOCR/trainer → make_easyocr_plugin.py

# 8) OCR 추론 — 공식(D23): CRAFT∪DB 검출 + 라인 병합 (ko: v5_lines / bk: en 자동)
.venv/Scripts/python.exe run_ocr_line.py --run 22
# (보조 per-box 엔진: run_ocr_only.py --box — easyocr/trocr 산출용)
.venv/Scripts/python.exe run_ocr_only.py --region gangnam --run 12 --box \
    --trocr-model artifacts/ocr_training/signboard_v3/trocr_model \
    --easyocr-recog signboard_v3_custom \
    --paddle --paddle-model-dir output/paddle_signboard_rec_v3/inference --paddle-device gpu

# 9) 평가
python eval_map_only.py          # 탐지 mAP
.venv/Scripts/python.exe eval_ocr_v2.py --mask-phone --en-only-regions brooklyn   # OCR 평가 (재라벨 GT 표준 프로토콜)
```

---

## 8. 핵심 설계 원칙

- **데이터 누수 방지**: 탐지·OCR 모두 소스 이미지 단위로 fold 분할 (한 사진의 크롭이 train/val에 분산되지 않음)
- **공정 비교**: 전 탐지 모델(YOLO26x·YOLOv5x·YOLO11x·FRCNN·EfficientDet)이 `artifacts/yolo11x_kfold/total_fold{0-4}/dataset` 의 **동일한 5-Fold 분할** 공유
- **재현성**: 모든 교차검증 seed 고정 (탐지 42, OCR 트레이너 1111)
- **도메인 적응**: 일반 텍스트(Text-in-the-Wild) → 간판 특화(Signboard) 순차 미세조정
- **다중 모델 비교**: 탐지 4종(**YOLO26x**/YOLOv5x/FRCNN/EfficientDet, + YOLO11x 버전 비교) × 인식 3종(TrOCR/EasyOCR/PaddleOCR)
- **granularity 정합**: 단어 단위로 학습한 OCR을 박스 단위로 추론(`--box`)하여 학습-추론 입력 단위를 일치
- **모델 격리**: 충돌하는 런타임(torch vs paddle cudnn DLL)은 별도 subprocess로 분리해 한 파이프라인에서 공존
- **앙상블**: 박스별 confidence 선택으로 배포 가능한 3-way 결합 (oracle 상한도 함께 보고)

---

*본 README는 코드 정적 분석과 학습 산출물(`artifacts/`, `output/`, `external/`, `runs/`)의 실제 결과 파일을 기반으로 작성되었습니다. 탐지 수치는 요약 CSV(`*_kfold_summary.csv`, `resume_more_summary_map50.csv`), OCR 수치는 `eval_ocr_v2.py`(run8) 및 학습 로그(`train.log`)에서 직접 추출했습니다.*
