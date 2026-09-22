# GSV 간판 파이프라인 — 간판 탐지 → 단어 탐지 → OCR → VLM 교정 → 의미 태깅

이 저장소는 **논문 표 4개**를 만드는 데 필요한 코드와 근거만 담습니다. 예전 구성(CRAFT 검출,
PaddleOCR 3-way 투표, per-box 인식, OCR 사전/스냅, 그림 생성, 각종 탐색 실험)은 **코드까지 전부
삭제**했습니다. git 이력에는 남아 있으므로 필요하면 되살릴 수 있습니다.

**원칙: 실제로 돌려서 나온 값만 표에 넣습니다.** 다른 데이터셋의 값을 환산하거나 다른 구성에서 잰 값을
옮겨 적지 않습니다. 안 돌린 조합은 빈칸으로 둡니다.

---

## 1. 배포 구성

```
GSV 사진 (298장)
  │
  ├─① 간판 검출   YOLO26x, imgsz 960, conf 0.25                        → Table 1
  │
  ├─② 단어 검출   학습된 YOLO26x 단어 탐지기, conf 0.01                 → Table 2
  │
  ├─③ 라인 구성   y-중심 클러스터링(y_tol 0.04) → union bbox + pad 0.04
  │                → 라인 하나가 스트립 한 장
  │
  ├─④ 인식        SVTRv2-B 단일 모델, 전 지역 공통                      → Table 3
  │
  ├─⑤ VLM 교정    Gemma 4 31B / Qwen3-VL-32B-instruct                   → Table 4
  │                모드 fixtext / fix / fixcand
  │
  └─⑥ 의미 태깅   Gemma 4 31B
```

**③ 이 핵심입니다.** 박스별로 자르지 않고 라인째 인식합니다. 단어 크롭 학습 모델을 라인에 넣는
granularity 불일치보다, 박스 단위로 자르는 손실이 훨씬 컸습니다.

**④ 의 근거는 성능이 아니라 비용입니다.** SVTRv2-B 는 GT 크롭에서 75.5% 로 PaddleOCR 3-way 투표
74.0% 보다 높지만, **VLM 교정을 거치면 83.6 vs 83.8% 로 동률**입니다. 통제 비교의 held-out 절반에서도
PaddleOCR 대비 +3.5%p [-0.7, +7.9] 로 **유의하지 않습니다**. 즉 "더 좋은 인식기"가 아니라
**"같은 최종 출력을 인식 패스 1회로 낸다"**(투표는 3회)가 정확한 서술입니다.

> ⚙️ **PaddleOCR 격리.** paddle(cu118)과 torch(cu118)가 같은 이름의 `cudnn_*_8.dll`(다른 ABI)을
> ship 해서 한 프로세스에 공존하면 `WinError 127` 이 납니다. 그래서 모든 인식기를 **같은 IO 계약의
> 별도 워커 프로세스**로 분리했습니다 — `--worker` 하나만 바꾸면 인식기가 교체되고, 검출·라인 병합·
> 전처리·채점은 완전히 동일하게 유지됩니다. 이 구조가 Table 3 의 통제 비교를 가능하게 합니다.

---

## 2. 데이터

| 용도 | 데이터 | 규모 | 분할 |
|---|---|---|---|
| 간판 탐지 (Table 1) | GSV 강남·브루클린·수원 | 298장 | 가게 단위 group-aware 5-fold (같은 가게가 train/val 에 걸치지 않음) |
| 단어 박스 탐지 (Table 2) | AI Hub signboard_v3 | 27,132장 / 78,110 박스 | 소스 이미지 단위 train 21,714 / val 2,712 / **test 2,706** |
| 인식기 학습 (Table 3) | AI Hub signboard_v3 | 단어 크롭 62,464 + 실라인 13,148 = **75,612** | 소스 이미지 단위, seed 42 |
| OCR 평가 (Table 3·4) | GSV GT 크롭 | 411 크롭 / **592 라인** / 930 단어 | 전수 평가 (통제 비교 시 사진 단위 val/test 반분) |

**GSV 에는 단어 박스 GT 가 없습니다.** 간판 박스와 크롭별 텍스트만 있어서 지역별 단어 박스 AP 는
현재 측정 불가입니다 (→ 6절).

---

## 3. 표 1 — 간판 검출 (AP@0.5)

전 모델 동일 5-fold 분할, 동일 AP@0.5 구현, out-of-fold 예측을 298장 전체에 대해 pooled.

| 모델 | 강남 | 브루클린 | 수원 | 전체 |
|---|---|---|---|---|
| **YOLO26x (배포)** | 0.833 | 0.874 | 0.879 | **0.860** |
| Faster R-CNN | 0.826 | 0.871 | 0.884 | 0.854 |
| EfficientDet-D0 | 0.790 | 0.823 | 0.881 | 0.828 |
| YOLOv5x | 0.815 | 0.856 | 0.744 | 0.806 |

**학습.** YOLO26x/YOLOv5x: Ultralytics, imgsz 960. Faster R-CNN: torchvision ResNet50-FPN.
EfficientDet-D0: letterbox 512, ImageNet 정규화 없음(학습과 동일). 코드는
`detection/train_{yolo26x,yolov5x,frcnn,effdet}_kfold.py`, 채점은 `detection/eval_det_unified.py`.

---

## 4. 표 2 — 단어 박스 검출과 연쇄 (AP@0.5)

세 열을 **같은 단위(AP@0.5)** 로 둬서 단계별 손실이 직접 읽히게 합니다.

| 아키텍처 | 간판 AP@0.5 | 단어 박스 AP@0.5 | 연쇄 AP@0.5 (간판→단어) |
|---|---|---|---|
| **YOLO26x (배포)** | 0.860 | **0.822** | (미측정) |
| YOLOv5x | 0.806 | 0.809 | (미측정) |
| Faster R-CNN | 0.854 | 0.715 | (미측정) |
| EfficientDet-D0 | 0.828 | 0.672 | (미측정) |

간판 AP 는 GSV, 단어 박스 AP 는 AI Hub hold-out(2,706장 / 78,110 박스)입니다. 단어 박스 GT 가
있는 데이터가 AI Hub 뿐이라 그렇습니다.

**연쇄 AP 가 재는 것.** 배포는 단어 검출기에 사진을 통째로 넣지 않습니다. 간판 검출기가 먼저 자르고
단어 검출기는 그 크롭만 봅니다. `detection/eval_text_chain.py` 는 그 경로 그대로 돕니다.

```
사진 → 간판 검출 → 크롭들 → 크롭 안에서 단어 검출
     → 박스를 원본 좌표로 되돌림 → NMS → 단어 GT 와 AP@0.5
```

간판 검출기의 출력 자체는 채점하지 않으므로(중간 단계) 간판 GT 는 필요 없습니다.

> **1단계 실패는 빼는 게 아니라 0점으로 계산됩니다.** 분모(78,110 박스)는 그대로 둡니다. 간판을
> 하나도 못 찾은 사진은 크롭이 0개 → 단어 예측 0개 → 그 사진의 GT 단어 박스가 전부 미검출로 잡힙니다.
> 반대로 허위 간판 박스는 허위 크롭을 만들고 그 안의 단어 박스가 오검출로 들어갑니다. 회수율 손실과
> 정밀도 손실이 **둘 다 전달**되며, 이것이 배포의 실제 거동입니다.

**현재 보류 중입니다.** GSV 지역별 칸을 채우려면 GSV 단어 박스 라벨이 필요하고, 라벨 작업 뒤에
한꺼번에 돌리기로 했습니다 (→ 6절).

---

## 5. 표 3 — 인식기 단일 비교 (line exact / CER)

**고정:** 단어 검출기(YOLO26x @0.01) · 라인 병합(y_tol 0.04) · 패딩(0.04) · 전처리 · 채점 규칙
(라인 매칭, `--mask-phone`, 브루클린 영어 전용 채점). **바꾼 것은 인식기 하나뿐**입니다.
미세조정 행은 전부 같은 75,612 크롭 · 같은 분할. 모든 행이 **단일 모델**입니다(앙상블 없음).
GSV GT 크롭 411개 / 592 라인.

| 인식기 | 학습 | 강남 | 브루클린 | 수원 | 전체 | CER |
|---|---|---|---|---|---|---|
| Tesseract 5.5 | 제로샷 | 31.1 / .541 | 47.7 / .306 | 14.9 / .736 | 31.8% | .464 |
| Tesseract 5.5 | 미세조정 | 35.8 / .507 | 44.2 / .316 | 20.4 / .688 | 34.0% | .449 |
| EasyOCR (CRNN) | 미세조정 | (미측정) | (미측정) | (미측정) | (미측정) | |
| TrOCR-base | 미세조정 | (학습 대기) | | | | |
| ABINet | 미세조정 | (학습 대기) | | | | |
| MAERec (ViT-S) | 미세조정 | (학습 대기) | | | | |
| Surya 0.14 | 제로샷 | 43.9 / .532 | 59.3 / .194 | 22.1 / .759 | 42.4% | .411 |
| CLOVA OCR General | 제로샷(상용 API) | 58.0 / .317 | 59.8 / .186 | 48.1 / .415 | 55.6% | .272 |
| PARSeq (ViT-S) | 미세조정 | 70.8 / .242 | 71.9 / .133 | 62.4 / .290 | 68.6% | .197 |
| PaddleOCR PP-OCRv5 rec | 미세조정 | 73.6 / .190 | 77.9 / .108 | 68.5 / .190 | 73.5% | .149 |
| **SVTRv2-B (배포)** | 미세조정 | 74.5 / .185 | 78.9 / .105 | 72.9 / .197 | **75.5%** | **.147** |

### 5.1 각 인식기를 왜 넣었나

| 인식기 | 정체 | 근거 | 체급 |
|---|---|---|---|
| Tesseract 5.5 | LSTM 라인 인식기, 오픈소스 문서 OCR 표준 | 응용 OCR 논문의 기본 기준선. 같은 데이터로 미세조정까지 되는 유일한 비-STR 엔진이라 "데이터 부족"과 "구조 문제"를 가릅니다 | 아래 |
| EasyOCR (CRNN) | CRNN 인식기 (JaidedAI) | 문헌에서 PaddleOCR 와 가장 자주 함께 벤치마크되는 상대 | 아래 |
| Surya 0.14 | 최신 다국어 문서 OCR 툴킷 (transformer) | **기성 엔진 기준선.** 학습 없이 그대로 썼을 때 얼마가 나오는지를 보는 자리로, Tesseract 제로샷의 현대판 대조군입니다. 공개된 학습 코드가 없어 미세조정이 **불가능**하므로 제로샷으로만 들어갑니다 | 비슷 |
| TrOCR-base | Transformer encoder-decoder OCR (Microsoft) | 같은 이유. small(62M) 대신 **base(334M)** 로 체급을 올림 | 비슷 |
| ABINet | 언어모델 결합 STR (CVPR 2021, 확장판 ABINet++ 는 IEEE TPAMI) | STR 비교표의 표준 기준선. **미세조정 가능**하고 SVTRv2 와 같은 학습 하네스(OpenOCR) 사용 | 비슷 |
| MAERec (ViT-S) | MAE 사전학습 ViT + NRTR 디코더 (ICCV 2023, Union14M) | 더 최신 STR 기준선. 역시 같은 하네스로 미세조정 가능 | 비슷 |
| PARSeq (ViT-S) | Permuted AR STR (ECCV 2022) | 파라미터 수가 배포 인식기와 가장 가까운 **체급 일치** 비교 | 일치 |
| PaddleOCR PP-OCRv5 rec | 이전 배포 인식기 | 교체 판단의 기준 행 | 기준 |
| **SVTRv2-B** | Single visual model + CTC, OpenOCR (ICCV 2025) | **현행 배포** | 위 |
| CLOVA OCR General | NAVER 상용 API | 상용 기준점. 미세조정이 **불가능**(가중치 비공개)하므로 "유료 범용 API 가 미세조정 모델을 대체할 수 있나"에만 답합니다 | 비공개 |

**제로샷 행 둘(Surya · CLOVA)은 미세조정군과 다른 질문에 답합니다.** 하나의 순위로 읽으면 안 됩니다.
미세조정 7종은 "같은 데이터·같은 파이프라인에서 어느 구조가 나은가"에 답하고, 제로샷 2종은
**"학습 없이 그대로 가져다 쓰면 얼마가 나오는가"**에 답합니다. 표에 학습 조건 열을 둔 이유가 이것입니다.

둘 다 미세조정이 **불가능해서** 제로샷인 것이지, 안 한 것이 아닙니다.
- **Surya**: 저장소(`datalab-to/surya`, v0.22.1)의 `surya/scripts/` 에 추론 스크립트만 있고 **학습
  스크립트가 없습니다**(GitHub API 로 디렉터리 목록 직접 확인). README 도 미세조정은 `hi@datalab.to`
  문의, 즉 자사 유료 학습 서비스로 안내합니다. 설치된 `.venv_surya`(surya-ocr 0.14.6)도 CPU 전용
  torch 라 학습 자체가 안 됩니다.
- **CLOVA OCR General**: 상용 API 로 가중치가 비공개입니다.

따라서 이 두 행으로는 **"구조가 낫다/못하다"를 주장할 수 없습니다.** 쓸 수 있는 서술은
"기성 엔진을 그대로 가져다 쓰면 이 도메인에서 이 정도"까지입니다.

> 인용 주의: Surya 는 논문이 없고(GitHub 도구) JCR Q1 저널 사용 선례도 찾지 못했습니다 — 확인된
> 사용례는 arXiv 프리프린트와 RANLP 2025 학회 논문뿐이고 전부 문서 OCR 맥락, 장면 텍스트 사례는
> 없습니다. 기성 엔진 기준선이라는 **역할**은 [A Survey of OCR Evaluation Methods](https://arxiv.org/html/2603.25761v1)
> 가 Tesseract v5·olmOCR 2 와 함께 Surya 를 묶어 비교한 용례와 같지만, 저널 선례로 인용할 수는
> 없습니다.

> 한 번 틀렸던 기록: 검색 요약만 보고 "Surya 에 `finetune_ocr.py` 가 있어 미세조정 가능"이라고 적었다가
> 저장소를 직접 열어보고 정정했습니다. 근거는 검색 요약이 아니라 원본에서 확인합니다.

### 5.2 순위가 실제로 갈리는지 — held-out 신뢰구간

298장을 **사진 단위로 반분**(같은 간판이 양쪽에 걸리지 않음)해 test 절반 296 라인에서만 차이를
계산합니다. 크롭 단위 **페어 부트스트랩 95%**(2,000회).

| 인식기 | vs PaddleOCR | 95% CI | 판정 |
|---|---|---|---|
| SVTRv2-B | +3.5 | [-0.7, +7.9] | **미판정** |
| PARSeq | -3.5 | [-8.4, +1.0] | **미판정** |
| CLOVA OCR General | -13.8 | [-19.6, -8.2] | 유의 |
| Surya 0.14 | -28.7 | [-34.4, -23.1] | 유의 |
| Tesseract (미세조정) | -39.1 | [-45.1, -33.2] | 유의 |
| Tesseract (제로샷) | -40.5 | [-45.9, -34.7] | 유의 |

구간이 0을 지나면 이 표본 크기에서 **판정 불가**이고, "이겼다"로 쓰면 안 됩니다. 미세조정 현대
인식기 세 개(SVTRv2·PaddleOCR·PARSeq)는 **한 덩어리**이고, 유일하게 갈리는 쌍은 SVTRv2 > PARSeq
(-6.9 [-10.9, -3.2]) 하나입니다.

**검출기를 바꾸면 순위가 뒤집힙니다.** 같은 비교를 이전 검출기에서 돌리면 SVTRv2 가 PaddleOCR 를
+5.2 [+1.1, +9.5] 로 유의하게 이겼고 CLOVA 는 PaddleOCR 와 구분되지 않았습니다. 인식기 격차로 보이던
것의 일부가 실은 **검출 손실**이었다는 뜻입니다. **인식기 비교는 어느 검출기 위에서 쟀는지를 반드시
밝혀야** 합니다.

**한계.** 모든 행이 공유하는 기하 파라미터(pad 0.04, y_tol 0.04)와 검출기 임계값(0.01)은 같은 GSV
데이터에서 정해졌습니다. 행 간 비교는 공정하지만 **절대 수치는 모든 행에서 낙관적**입니다.

### 5.3 인식기 학습 프로토콜

| 인식기 | 학습 | 소요 |
|---|---|---|
| PaddleOCR PP-OCRv5 rec | 실라인 13,148 + 단어 크롭, v5_lines | — |
| SVTRv2-B | OpenOCR `svtrv2_rctc_signboard.yml`, 같은 75,612 | 3h41m |
| PARSeq (ViT-S) | 같은 분할·같은 혼합 | 2h15m |
| ABINet | OpenOCR `configs/rec/abinet/abinet_signboard.yml`, 같은 조건 (예정) | ~3~4h |
| MAERec | OpenOCR `configs/rec/maerec/maerec_signboard.yml`, 같은 조건 (예정) | ~3~4h |
| TrOCR-base | `ocr/train_textinthewild_ocr.py`, base-printed, 10ep@5e-5, batch 4, 증강, **바이트레벨 BPE 토크나이저**(기본 sentencepiece 는 한글 음절을 `<unk>` 로 손상시킴) | ~11h (예정) |
| EasyOCR (CRNN) | VGG+BiLSTM+CTC, imgH 64 / imgW 600 → `make_easyocr_plugin.py` 로 `signboard_v3_custom` 플러그인화 | 완료 |
| Tesseract 5.5 | WordStr box 파일 + 자모 recoder, `preserve_interword_spaces` | 완료 |

> TrOCR-base 는 small 모델을 덮어쓰지 않도록 `artifacts/ocr_training/signboard_v3_base/` 에 따로
> 저장하고, 크롭 62,464장은 **정션으로 연결**해 복사하지 않습니다. 분할·토크나이저가 같으므로
> base 와 small 의 차이는 **모델 크기뿐**입니다.

---

## 6. 표 4 — VLM 교정 (line exact / CER)

인식기를 배포 구성(SVTRv2-B)으로 고정하고 **모델 2종 × 모드 3종**을 같은 스트립에서 돌립니다.

| 모드 | VLM 이 보는 것 | 답하는 질문 |
|---|---|---|
| fixtext | OCR 텍스트만, **이미지 없음** | 교정 이득 중 얼마가 시각 정보가 아니라 **언어 사전**에서 오는가 |
| fix | 간판 이미지 + OCR 텍스트 | 이미지를 주면 얼마가 더 붙는가 |
| fixcand | 간판 이미지 + 후보 3종 | 다른 인식기의 후보를 함께 보여주면 얼마가 더 붙는가 |

| 모델 · 모드 | 강남 | 브루클린 | 수원 | 전체 | CER |
|---|---|---|---|---|---|
| Gemma 4 31B · fixtext | (측정 중) | | | | |
| Gemma 4 31B · fix | (측정 중) | | | | |
| Gemma 4 31B · fixcand | (측정 중) | | | | |
| Qwen3-VL-32B · fixtext | (측정 중) | | | | |
| Qwen3-VL-32B · fix | (측정 중) | | | | |
| Qwen3-VL-32B · fixcand | (측정 중) | | | | |

둘 다 Ollama 4-bit, GPU/RAM 분할(RTX 3080 10GB). 크롭당 약 30초.

---

## 7. 의미 태깅과 연쇄 측정

표 1~4 는 GT 크롭 기준입니다. 실제 배포는 간판 검출기가 자른 크롭으로 돌므로, 사진 한 장을 넣어
텍스트와 업종 태그가 나오기까지를 따로 잽니다 (`pipeline/eval_e2e_cascade.py`,
`pipeline/eval_e2e_tagging.py`).

현재 SVTRv2 배포 구성으로 재측정 중입니다. 참고로 이전 구성(YOLO26x 검출 + PaddleOCR 투표 + Gemma 4)
에서는 연쇄 OCR 54.4% / TP 위 65.6%, 태깅 68.9% / TP 위 82.9% 였습니다.

> 탐지 허위 양성 124건 **전부**에 업종 태그가 붙었습니다. 지도에 없는 가게가 124곳 등록될 수 있다는
> 뜻이고, 회수율과 함께 반드시 보고해야 할 실패 모드입니다. 원인은 언어모델이 아니라 간판 검출입니다.

---

## 8. 저장소 구조

```
pipeline/    run_ocr_line.py        배포 OCR (단어 검출 → 라인 병합 → 인식)
             eval_ocr_v2.py         OCR 채점 (라인 매칭, 전화번호 마스킹, 브루클린 영어 전용)
             exp_vlm_ocr.py         VLM 교정 (fixtext / fix / fixcand)
             eval_e2e_cascade.py    연쇄 OCR 측정
             eval_e2e_tagging.py    연쇄 태깅 측정

detection/   train_*_kfold.py       간판 탐지 4모델 (Table 1)
             train_*_text_holdout.py 단어 박스 탐지 4모델 (Table 2)
             eval_det_unified.py    단일 AP@0.5 채점기
             eval_text_holdout.py   단어 박스 AP (검출기 단독)
             eval_text_chain.py     단어 박스 AP (간판→단어 연쇄)
             e2e_det_boxes.py       연쇄용 탐지 크롭 생성

str_baselines/  *_rec_worker.py     인식기 워커 8종 — 같은 IO 계약, --worker 로 교체
                                    (surya 만 .venv_surya 에서 실행: --worker-py 로 지정)
                yolo_text_det_worker.py  단어 탐지기 워커
                eval_ocr_controlled.py   통제 비교 + 부트스트랩 신뢰구간

ocr/         train_textinthewild_ocr.py  TrOCR 학습
             train_paddle_v4_spacecat.py PaddleOCR 학습
             make_easyocr_plugin.py      EasyOCR 플러그인화

vlm/         태깅 · POI 사전 · RAG

scripts/     run_svtr_deploy.sh     배포 SVTRv2 전 구간 재측정
             run_svtr_pipeline.sh   EasyOCR → TrOCR-base 학습·추론 → VLM 2×3
             run_str_baselines.sh   인식기 기준선
             run_text_holdout.sh    단어 박스 탐지 학습
             run_clova_ocr.sh       CLOVA API 실행 (체크포인트로 재과금 방지)
```

---

## 9. 실행

```bash
# 간판 탐지 (Table 1)
.venv/Scripts/python.exe detection/train_yolo26x_kfold.py
.venv/Scripts/python.exe detection/eval_det_unified.py

# 단어 박스 탐지 (Table 2)
bash scripts/run_text_holdout.sh
.venv/Scripts/python.exe detection/eval_text_holdout.py
.venv/Scripts/python.exe detection/eval_text_chain.py      # 연쇄 — GSV 라벨 준비 후

# OCR 추론 — 배포 구성
.venv/Scripts/python.exe pipeline/run_ocr_line.py --run N \
    --text-detector yolo --yolo-det-conf 0.01 \
    --worker str_baselines/openocr_rec_worker.py \
    --worker-args "--config external/OpenOCR/configs/rec/svtrv2/svtrv2_rctc_signboard.yml \
                   --weights artifacts/str_baselines/svtrv2/best.pth" \
    --engine-tag cysvtrv2
# 인식기 교체는 --worker 만 바꾸면 됩니다

# 채점
.venv/Scripts/python.exe pipeline/eval_ocr_v2.py --mask-phone --en-only-regions brooklyn
.venv/Scripts/python.exe str_baselines/eval_ocr_controlled.py    # Table 3 + 신뢰구간

# VLM 교정 (Table 4) · 연쇄
bash scripts/run_svtr_pipeline.sh all
.venv/Scripts/python.exe pipeline/eval_e2e_cascade.py --ocr-run N --engine vlmfix
.venv/Scripts/python.exe pipeline/eval_e2e_tagging.py --ocr-run N --engine vlmfix
```

---

## 10. 남은 일

| 항목 | 상태 |
|---|---|
| Table 3 — EasyOCR 라인 스트립 | 대기 (큐) |
| Table 3 — TrOCR-base 학습 + 추론 | 대기 (큐, ~11h) |
| Table 3 — ABINet 학습 + 추론 | 대기 (큐, ~3~4h) |
| Table 3 — MAERec 학습 + 추론 | 대기 (큐, ~3~4h) |
| Table 4 — VLM 2모델 × 3모드 | 대기 (큐, ~21h) |
| 연쇄 태깅 — SVTRv2 구성 | 실행 중 |
| Table 2 — 연쇄 AP@0.5 | **보류** — GSV 단어 박스 라벨 후 |

### ABINet · MAERec 사전학습 가중치

SVTRv2 는 Union14M 사전학습 가중치에서 출발했고 PARSeq 도 사전학습을 썼습니다. ABINet·MAERec 만
스크래치로 학습하면 **반대 방향의 불공정**이 됩니다. OpenOCR 모델 zoo 는 Google Drive 폴더에 있는데
스크립트로는 목록 조회가 막혀 있어 수동 다운로드가 필요합니다.

- <https://drive.google.com/drive/folders/1Po1LSBQb87DxGJuAgLNxhsJ-pdXxpIfS>
- <https://drive.google.com/drive/folders/1x1LC8C_W-Frl3sGV9i9_i_OD-bqNdodJ>

받은 파일을 `external/OpenOCR/pretrained/abinet/best.pth`, `external/OpenOCR/pretrained/maerec/best.pth`
로 두면 설정이 자동으로 집어 씁니다(`strict=False` 로 로드되어 한국어 분류층만 무작위 초기화).
없으면 스크래치로 학습되며, 그때는 **표에 "사전학습 없음"을 반드시 명시**해야 합니다.

### GSV 단어 박스 라벨링

Table 2 의 지역 칸(강남·브루클린·수원)을 채우려면 GSV 단어 박스 GT 가 필요합니다.

| 지역 | 크롭 | 라인 | 라벨할 단어 박스 |
|---|---|---|---|
| 강남 | 138 | 212 | 306 |
| 브루클린 | 141 | 199 | 383 |
| 수원 | 132 | 181 | 241 |
| **합계** | **411** | **592** | **930** |

> **주의:** 라벨 씨앗으로 YOLO 검출 결과만 쓰면 안 됩니다. YOLO 가 놓친 박스는 라벨에도 안 생겨
> YOLO 에게 유리하게 기웁니다. 백지에서 그리거나, 씨앗을 쓸 거면 **4개 검출기 결과의 합집합**을 깔고
> 전수 수정해야 공정합니다.
