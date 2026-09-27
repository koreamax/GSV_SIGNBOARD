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
  ├─⑤ VLM 교정    Gemma 4 31B, **fix** 모드 (SVTRv2 결과 + 간판 이미지)  → Table 4
  │                (fixtext / fixcand · Qwen3-VL 은 표 4 의 비교 실험)
  │
  └─⑥ 의미 태깅   Gemma 4 31B, POI 사전 RAG, "unknown" 기권 허용       → 7절 (연쇄)
```

**⑤ 는 fix 입니다(fixcand 아님).** fixcand 는 같은 스트립을 PaddleOCR 3종으로 더 읽어 후보로 주는
모드라 인식 패스가 4회(SVTRv2 + PaddleOCR 3)가 됩니다. 표 4 에서 fix → fixcand 이득이 +0.2%p 뿐이라
배포는 **인식 패스 1회**인 fix 입니다. 이것이 ④ 를 SVTRv2 로 바꾼 근거(아래)와 같은 이야기입니다.

**③ 이 핵심입니다.** 박스별로 자르지 않고 라인째 인식합니다. 단어 크롭 학습 모델을 라인에 넣는
granularity 불일치보다, 박스 단위로 자르는 손실이 훨씬 컸습니다.

**④ 의 근거는 성능이 아니라 비용입니다.** SVTRv2-B 는 GT 크롭에서 75.5% 로 PaddleOCR 3-way 투표
74.0% 보다 높지만, **VLM 교정(fixcand)을 거치면 83.6 vs 83.8% 로 동률**입니다. 통제 비교의 held-out 절반에서도
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
| 단어 박스 탐지 학습 (Table 2) | AI Hub signboard_v3 | 27,132장 / 78,110 박스 | 소스 이미지 단위 train 21,714 / val 2,712 / **test 2,706장 · 7,756 박스** |
| 단어 박스 평가 (Table 2) | **GSV 단어 박스 GT** (이번 리비전 라벨) | 간판 463개 / **1,499 박스** (5-fold 298장 기준 453개 / 1,467 박스) | 전수 |
| 인식기 학습 (Table 3) | AI Hub signboard_v3 | 단어 크롭 62,464 (+ 실라인 13,148 = 75,612) | 소스 이미지 단위, seed 42. **실라인 포함 여부가 인식기마다 다릅니다 — 5.3절** |
| OCR 평가 (Table 3·4) | GSV GT 크롭 | 411 크롭 / **592 라인** / 930 단어 | 전수 평가 (통제 비교 시 사진 단위 val/test 반분) |

GSV 단어 박스 GT 는 라벨링 도구(`data/label_wordbox.py`)로 463개 간판을 전수 라벨한 것입니다(10절).
사진 300장 중 2장(`brooklyn__38`, `gangnam__35`)은 간판 5-fold 에 들어있지 않아 out-of-fold 간판
검출이 없으므로, 연쇄 측정은 표 1 과 같은 298장(간판 453 / 단어 1,467)으로 합니다.

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

모든 열을 **같은 단위(AP@0.5)** 로 둬서 단계별 손실이 직접 읽히게 합니다. 단, **같은 데이터셋 안에서만**
비교합니다(아래 주의).

| 아키텍처 | 간판 AP (GSV) | 단어 AP (AI Hub) | 단어 AP (GSV, 정답 간판) | **연쇄 AP (GSV)** | 연쇄 (간판 0개 사진 15장 제외) |
|---|---|---|---|---|---|
| **YOLO26x (배포)** | 0.860 | 0.822 | 0.892 | **0.718** | 0.740 |
| YOLOv5x | 0.806 | 0.809 | 0.893 | 0.722 | 0.745 |
| Faster R-CNN | 0.854 | 0.715 | 0.827 | 0.675 | 0.696 |
| EfficientDet-D0 | 0.828 | 0.672 | 0.795 | 0.638 | 0.658 |

간판 AP 는 표 1 값. AI Hub 단어 AP 는 hold-out(**2,706장 / 7,756 박스**). GSV 세 열은
`detection/eval_wordbox_chain_gsv.py` 한 스크립트에서 **간판 박스의 출처만 바꿔** 계산합니다
(298장, 단어 GT 1,467 박스, 지역별 값은 `artifacts/gt/wordbox_chain_gsv_ap50.csv`).

| 열 | 간판 박스 출처 |
|---|---|
| 단어 AP (정답 간판) | 사람이 그린 간판 폴리곤의 외접 사각형 453개 |
| 연쇄 AP | 표 1 검출기(YOLO26x, group-aware 5-fold)의 **out-of-fold** 예측(conf 0.25) 470개 — 각 사진은 그 사진을 학습하지 않은 fold 의 모델로 검출 (`artifacts/gt/det_oof_grp.json`) |

두 경우 모두 같은 규칙으로 자르고(6% 여유, 긴 변 1,400px — 라벨링 화면과 동일), 같은 단어 검출기를
돌려 원본 사진 좌표로 되돌린 뒤 사진 단위 NMS(IoU 0.6) 후 채점합니다. 그래서 **두 열의 차이가 정확히
간판 검출이 깎는 몫**입니다(YOLO26x 0.892 → 0.718). 독립 검산: 단어 GT 중 검출된 간판 안에 중심이
들어오는 것이 **87.3%** 뿐이라, 0.892 × 0.873 ≈ 0.78 에 허위 간판의 오검출이 더해져 0.72 가 맞습니다.

> **정정 (D42).** 이전 판의 연쇄 AP(0.815 / 0.818 / 0.765 / 0.727)는 간판 검출에 **fold0 가중치
> 하나를 298장 전부에** 써서, 검출기가 **학습 때 본 사진 80%(238장)** 가 시험에 섞여 있었습니다. 그
> 코드는 원래 AI Hub 사진(검출기가 본 적 없음)용이었는데 GSV 옵션을 붙이면서 이 점을 놓쳤습니다.
> 약 10%p 가 외워서 맞힌 몫이었고, 위 표가 그것을 대체합니다. 그 GSV 옵션은 삭제했습니다.

> **두 단어 AP 열은 서로 비교하면 안 됩니다.** GSV 값이 더 높은 것은 검출기가 학습 분포보다
> 전이 분포에서 잘한다는 뜻이 아니라 **입력이 더 쉽기 때문**입니다. 실측한 이유 세 가지:
> (i) GSV 작업 이미지는 GT 간판의 축정렬 bbox 를 긴 변 1,400px 로 확대한 것이라 단어 하나가
> 이미지의 중앙값 **3.8%** 를 차지하는 반면 AI Hub 사진에서는 **1.8%** 입니다(면적 기준 2.2배).
> (ii) GSV 는 간판 하나만 든 크롭이라 배경이 거의 없고, AI Hub 은 사진 전체라 간판 밖 텍스트가
> 오검출을 만듭니다. (iii) GSV GT 는 4개 검출기 제안의 합집합 위에서 전수 수정해 만들었는데 최종
> 박스의 **97.3%** 가 제안과 일치하므로, 네 검출기가 모두 놓친 단어는 GT 에 거의 없습니다.
> 각 열은 **그 데이터셋 안에서의 모델 간 비교**로만 읽고, GSV 절대값은 낙관적이라고 적어야 합니다.

**두 YOLO 는 구분되지 않습니다.** AI Hub 에서는 YOLO26x(0.822) > YOLOv5x(0.809) 인데 GSV 에서는
YOLOv5x 가 0.1~0.5%p 앞섭니다. 신뢰구간 없이는 판정 불가이므로 "두 YOLO 는 구분되지 않고 FRCNN·EffDet
보다 확실히 낫다"까지만 쓸 수 있습니다. 7절의 A/B 연쇄 비교가 이 결론을 파이프라인 끝까지 확인합니다.

> **1단계 실패는 빼는 게 아니라 0점으로 계산됩니다.** 간판을 하나도 못 찾은 사진은 크롭이 0개 →
> 단어 예측 0개 → 그 사진의 GT 단어 박스가 전부 미검출. 허위 간판은 허위 크롭을 만들고 그 안의 단어
> 박스가 오검출로 들어갑니다. 라벨이 없는 사진도 298장에 그대로 포함해(GT 0개) 허위 간판에서 나온
> 단어가 오검출로 잡히게 합니다. 이것이 배포의 실제 거동이고 논문에는 이 열을 씁니다. 마지막 열은
> 간판을 하나도 못 찾은 **15장**을 뺀 값으로, 손실 분해용입니다(AP 는 사진 단위 지표라 간판을 일부만
> 찾은 사진의 손실은 양쪽에 남습니다).

---

## 5. 표 3 — 인식기 단일 비교 (line exact / CER)

**고정:** 단어 검출기(YOLO26x @0.01) · 라인 병합(y_tol 0.04) · 패딩(0.04) · 전처리 · 채점 규칙
(라인 매칭, `--mask-phone`, 브루클린 영어 전용 채점). **바꾼 것은 인식기 하나뿐**입니다.
모든 행이 **단일 모델**입니다(앙상블 없음). GSV GT 크롭 411개 / 592 라인. 전체 채점표는
`artifacts/gt/ocr_eval_v2_summary.csv`(9개 엔진).

| 인식기 | 학습 | 강남 | 브루클린 | 수원 | 전체 | CER | WAR |
|---|---|---|---|---|---|---|---|
| Tesseract 5.5 | 제로샷 | 31.1 / .541 | 47.7 / .306 | 14.9 / .736 | 31.8% | .464 | .360 |
| Tesseract 5.5 | 미세조정 · **단어만** | 35.8 / .507 | 44.2 / .316 | 20.4 / .688 | 34.0% | .449 | .285 |
| EasyOCR (CRNN) | 미세조정 · **단어만** | 54.2 / .276 | 46.7 / .208 | 51.9 / .316 | 51.0% | .251 | .199 |
| TrOCR-base | 미세조정 · **단어만** | 64.6 / .251 | 46.2 / .262 | 64.6 / .258 | 58.4% | .258 | .254 |
| ABINet | 미세조정 | (보류 — 10절) | | | | | |
| MAERec (ViT-S) | 미세조정 | (보류 — 10절) | | | | | |
| Surya 0.14 | 제로샷 | 43.9 / .532 | 59.3 / .194 | 22.1 / .759 | 42.4% | .411 | .367 |
| CLOVA OCR General | 제로샷(상용 API) | 58.0 / .317 | 59.8 / .186 | 48.1 / .415 | 55.6% | .272 | .531 |
| PARSeq (ViT-S) | 미세조정 · 단어+라인 | 70.8 / .242 | 71.9 / .133 | 62.4 / .290 | 68.6% | .197 | .672 |
| PaddleOCR PP-OCRv5 rec | 미세조정 · 단어+라인 | 73.6 / .190 | 77.9 / .108 | 68.5 / .190 | 73.5% | .149 | .669 |
| **SVTRv2-B (배포)** | 미세조정 · 단어+라인 | 74.5 / .185 | 78.9 / .105 | 72.9 / .197 | **75.5%** | **.147** | .694 |

> **학습 데이터가 두 종류로 섞여 있습니다 (D42 점검에서 확인).** 이전 판은 "미세조정 행은 전부 같은
> 75,612 크롭"이라고 적었지만 사실이 아닙니다. 실제 학습 목록을 세어 보면:
>
> | 학습 데이터 | 인식기 | 확인한 근거 |
> |---|---|---|
> | 단어 62,464 + **실라인 13,148 = 75,612** | PaddleOCR v5 · PARSeq · SVTRv2-B | LMDB `num-samples 75612` / PaddleOCR `label_file_list` 에 `lines_train.txt` |
> | **단어 크롭 62,464 만** | EasyOCR · TrOCR-base · Tesseract | EasyOCR `train.txt` 62,464줄 / TrOCR `labels.csv` train 62,464 / Tesseract lstmf 63,301개, 라인 없음 |
>
> 평가 입력은 여러 단어가 든 **라인 스트립**이라 아래 세 행은 라인 입력을 학습에서 본 적이 없습니다.
> WAR(단어 정확도)이 그대로 보여줍니다 — 단어만 학습한 세 행 **0.20~0.29**, 라인까지 학습한 세 행
> **0.67~0.69**. 글자 recall 은 92~93% 로 비슷한데(요약 CSV), 글자는 읽지만 줄 안에서 단어를 나누지
> 못합니다. 그래서 이 세 행과의 격차에는 **구조 차이와 학습 데이터 차이가 섞여** 있고, 이 표만으로
> "구조가 열세"라고 쓸 수 없습니다. 같은 75,612 로 재학습할지는 10절.

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

| 인식기 | 학습 데이터 | 학습 설정 | 소요 |
|---|---|---|---|
| PaddleOCR PP-OCRv5 rec | 단어 + 실라인 | `signboard_v3/train.txt` + `lines_train.txt`, v5_lines | — |
| SVTRv2-B | 단어 + 실라인 75,612 | OpenOCR `svtrv2_rctc_signboard.yml`, Union14M 사전학습에서 시작 | 12h (9/14 11:49 → 9/15 00:06) |
| PARSeq (ViT-S) | 단어 + 실라인 75,612 | 같은 LMDB, 사전학습에서 시작 | 2h15m |
| ABINet | (단어 + 실라인 예정) | OpenOCR `abinet_signboard.yml`, 사전학습 mmocr `abinet_20e_st-an_mj`(MJ+ST) 변환 | 보류 |
| MAERec | (단어 + 실라인 예정) | OpenOCR `maerec_signboard.yml`, 사전학습 Union14M `maerec_s_union14m` 변환 | 보류 |
| TrOCR-base | **단어 62,464 만** | `ocr/train_textinthewild_ocr.py`, base-printed, 10ep, batch 4, 증강, **lr 2e-5 · warmup 500 · linear decay · clip 1.0**, base 자체 byte-level BPE (5.4절). epoch 10 체크포인트(val 최저 0.159) | 13h |
| EasyOCR (CRNN) | **단어 62,464 만** | VGG+BiLSTM+CTC, imgH 64 / imgW 600 → `make_easyocr_plugin.py` 로 `signboard_v3_custom` 플러그인화 | — |
| Tesseract 5.5 | **단어 크롭만** (lstmf 63,301) | WordStr box 파일 + 자모 recoder, `preserve_interword_spaces` | — |

> README 이전 판의 SVTRv2 소요 "3h41m" 은 로그(9/14 11:49 → 9/15 00:06)와 맞지 않아 로그 값으로
> 고쳤습니다. TrOCR-base 는 small 을 덮어쓰지 않도록 `artifacts/ocr_training/signboard_v3_base/` 에
> 따로 저장하고, 크롭은 **정션으로 연결**해 복사하지 않습니다.

### 5.4 TrOCR-base 첫 학습이 붕괴한 이유와 수정

9시간 학습(exit 0)한 모델이 **출력을 전부 공백**으로 냈습니다. 자체 테스트셋도 `exact=0.0% CER=1.0`,
로그는 `train_loss 3.80→0.74` 인데 `val_loss` 는 1에폭부터 **11.57**(≈ ln 50265 = 10.8, 균등분포).
붕괴한 CSV(run 113 `ytrocrb`)는 `artifacts/ocr_gt/_quarantine/` 에 있습니다. 원인은 셋이고, 셋 다
`ocr/train_textinthewild_ocr.py` 에서 고쳤습니다.

**① 그래디언트 폭발 — 주원인.** 학습 루프에 클리핑이 없었습니다. 4,000스텝 대조 실험:

| 스텝 | 클리핑 없음 (val_loss / 티처포싱 정확도 / 예측 토큰 종류) | 클리핑 1.0 |
|---|---|---|
| 3000 | 0.77 / 75.0% / 31 | 1.33 / 63.0% / 27 |
| 4000 | **3.00 / 25.0% / 4** ← 붕괴 | **1.02 / 68.5% / 33** |

3,000스텝까지 잘 배우다 grad 노름이 195 → 610 으로 튄 구간에서 무너집니다. 1에폭이 15,616스텝이라
이걸 반복해서 맞고 상수 토큰만 내놓는 상태로 죽었습니다. 클리핑을 켜면 원시 노름은 1,341 까지 튀어도
학습이 이어집니다. → `--clip-grad`(기본 1.0) 추가.

**② 크기만 같은 사전 교체.** `--tokenizer-dir` 로 끼운 byte-level BPE 는 base 의 RoBERTa 사전과
vocab 크기가 **우연히 같은 50,265** 라 `resize_token_embeddings` 가 무효였고, 사전학습 임베딩이
**뒤섞인 id 에 그대로 붙은 채** 학습됐습니다(`COFFEE` → 교체본 10847 / base 6335). 이것만 고쳐도
val_loss 가 11.57 → 6.64 로 갈렸습니다. → 크기가 같아 resize 가 무효면 임베딩·출력층을 재초기화하고
로그로 알립니다.

교체 자체가 base 에는 **불필요**했습니다. small 의 XLM-R sentencepiece 는 `커피` → `''`(`<unk>`)로
한글을 날리지만, base 의 RoBERTa byte-level BPE 는 `밀크뮤직타운` → 20토큰 → 원문 복원(무손실)
입니다. 그래서 재학습은 교체 없이 돌립니다. 라벨 토큰 길이는 중앙값 8 · 최대 33 이라
`--max-target-length 64` 로 잘림 0%.

**③ 시작 토큰 덮어쓰기.** 체크포인트는 `</s>`(2)로 디코딩을 시작하는데 스크립트가 `cls`(0)로
덮어썼습니다. → 교체를 안 했으면 체크포인트 값을 유지.

**④ 고정 lr 5e-5 — 두 번째 붕괴의 원인.** ①~③ 을 고치고 lr 5e-5 그대로 돌리자 1에폭은
`val_loss 1.35` 로 정상이었다가 2에폭 **10.48**, 3에폭 **14.78** 로 다시 무너졌습니다(train 은 0.99 로
계속 하락). 클리핑은 gradient 크기만 막지 Adam 의 파라미터별 step 크기는 못 막습니다. val 을 eval 모드와
train 모드로 같이 재도록 바꿔 평가 경로 문제가 아님을 확인한 뒤 → **lr 2e-5 · warmup 500 · linear
decay** 로 재학습했습니다(`--warmup-steps`, `--lr-decay` 추가, 기본값은 기존 동작).

| 에폭 | 1 | 2 | 3 | 5 | 7 | 10 |
|---|---|---|---|---|---|---|
| val_loss (eval 모드) | 0.291 | 0.225 | 0.201 | 0.169 | 0.169 | **0.159** |

튀는 에폭 없이 수렴했고, 자체 test split(AI Hub 7,756 크롭) **exact 91.0% / CER 0.040**(붕괴본은 0.0% /
1.000). GSV 채점 결과가 표 3 의 58.4% 입니다(run 120 `ytrocrb2`).

> small 의 26.9% 는 ①~④ 가 전부 남아 있던 코드에서 나온 값입니다. TrOCR 를 "구조적으로 부적합"으로
> 서술하려면 small 도 재학습해야 하고, 5절 주의대로 단어만 학습한 조건도 함께 풀어야 합니다.

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
| (기준) SVTRv2-B 단독 | 74.5 | 78.9 | 72.9 | 75.5% | .147 |
| Gemma 4 31B · fixtext | 76.9 | 78.9 | 73.5 | 76.5% | .149 |
| Gemma 4 31B · fix | 82.5 | 88.4 | 79.0 | 83.4% | .128 |
| **Gemma 4 31B · fixcand** | 83.0 | 88.4 | 79.0 | **83.6%** | **.127** |
| Qwen3-VL-32B · fixtext | 74.5 | 82.4 | 72.4 | 76.5% | .144 |
| Qwen3-VL-32B · fix | 80.7 | 87.9 | 79.6 | 82.8% | .133 |
| Qwen3-VL-32B · fixcand | 81.1 | 90.5 | 76.8 | 82.9% | .130 |

둘 다 Ollama 4-bit, GPU/RAM 분할(RTX 3080 10GB). 실측 소요는 fixtext 약 2.5h, fix·fixcand 각 3~3.7h
(Qwen 이 Gemma 보다 조금 빠름). 6조합 전체 약 16시간.

**읽히는 것.** 이득의 대부분은 언어 사전이 아니라 **이미지**에서 옵니다 — 75.5% → 76.5%(fixtext,
+1.0%p) → 83.4%(fix, **+6.9%p**). 후보 3종을 더 줘도 +0.2%p(fixcand)뿐입니다. 두 모델은 fixtext 에서
76.5% 로 같고 fix·fixcand 에서 Gemma 가 0.6~0.7%p 앞서지만, 이 표에는 신뢰구간이 없어 1%p 안팎 차이는
판정 근거로 쓰지 않습니다.

**배포는 Gemma 4 · fix 입니다.** fixcand 의 후보는 같은 스트립을 PaddleOCR 3종(v5 미세조정 · v4 미세조정
· 사전학습)이 읽은 것이라, fixcand 를 배포하면 인식 패스가 4회가 됩니다. SVTRv2 결과는 fixcand 에서도
줄 구조와 1번 후보를 정하지만(프롬프트: "확신이 없으면 첫 번째 후보"), 후보를 만드는 비용에 비해
얻는 것이 +0.2%p 라 표 4 의 비교 실험으로만 둡니다. 7절 연쇄 측정은 fix 로 돌립니다.

---

## 7. 의미 태깅과 연쇄 측정

표 3·4 는 사람이 그린 정답 간판(GT 크롭)에서 잰 값입니다. 실제 배포는 간판 검출기가 자른 크롭으로
돌므로, **사진 한 장을 넣어 텍스트와 업종 태그가 나오기까지**를 따로 잽니다
(`scripts/run_final_measure.sh`).

```
사진 298장 → ① 간판 검출 (표 1 검출기, out-of-fold, conf 0.25) → 탐지 크롭 470개
              정답 간판 463개와 IoU≥0.5 매칭: TP 374 · FN 89(놓침) · FP 96(간판 아닌 것)
           → ② 단어 검출 → 라인 병합 → ④ SVTRv2-B → ⑤ Gemma 4 fix → ⑥ Gemma 4 태깅
```

모델은 470개 크롭을 **한 번만** 돌리고, 채점에서 분모만 달리해 세 값을 냅니다.

| 열 | 뜻 |
|---|---|
| **recall** | 놓친 간판(FN)의 정답을 0점으로 분모에 포함 — **배포 성능** |
| **TP 위** | 제대로 찾은 간판(TP)만 — 1단계 오류 제외 |
| **GT 크롭** | 같은 모델을 사람이 그린 간판에 돌린 값 — 1단계 완벽 |

허위 간판(FP)은 정답이 없으니 정확도 분모에서 빼고 **허위 POI** 로 따로 셉니다.

### 7.1 결과

| 구성 | 단어 검출 | 연쇄 OCR recall / TP 위 / GT 크롭 | 연쇄 태깅 recall / TP 위 | 허위 POI |
|---|---|---|---|---|
| **A (배포)** | YOLO26x | **63.3% / 77.2% / 83.4%** | **66.3% / 80.6%** | **69 / 96 (72%)** |
| B | YOLOv5x | (측정 중) | (측정 중) | (측정 중) |

A 지역별 — OCR: 강남 61.8 / 77.1 / 82.5 · 브루클린 65.3 / 76.5 / 88.4 · 수원 63.0 / 78.1 / 79.0,
태깅: 강남 69.9 / 87.2 (허위 26/37) · 브루클린 59.8 / 70.5 (19/26) · 수원 69.3 / 84.6 (24/33).
로그는 `artifacts/final_measure/logs/`, 태깅 상세는 `artifacts/gt/e2e_tagging_final_A.csv`.

**GT 크롭 열 83.4% 는 표 4 의 Gemma 4 · fix 와 정확히 같습니다** — 연쇄 채점기가 표 3·4 와 같은 함수라는
자체 검산입니다. OCR 손실은 이렇게 읽힙니다: 83.4 → 77.2(간판을 찾았지만 박스가 정답과 달라 크롭이
달라진 몫) → 63.3(못 찾은 간판 89개의 몫).

**허위 POI(Point of Interest — 지도에 존재하지 않는 가게가 등록되는 것) 69/96.** 간판이 아닌 크롭 96개 중
69개에 업종이 붙었습니다. 태깅 모델에게 "unknown" 을 허용해서 27개는 스스로 걸렀고, 대신 진짜 간판 325개
중 22개에도 unknown 으로 답해 오답이 됐습니다(걸러내기와 놓치기의 교환). 원인은 ① 의 허위 검출이고,
회수율과 함께 반드시 보고해야 할 실패 모드입니다.

### 7.2 이전 판 연쇄 값의 정정 (D42)

이전 판의 연쇄 값(OCR 52.5% / 63.3%, 태깅 68.6% / 82.6%, 허위 POI 124/124)은 쓰지 않습니다. 점검에서
네 가지 문제가 나왔습니다.

| 문제 | 영향 | 조치 |
|---|---|---|
| 간판 검출기가 표 1 과 다름 (비-group-aware `yolo26x_kfold`, 단독 AP 0.832) | 1단계가 표 1 의 0.860 검출기가 아니었음. 크롭 502개(FP 124) | `e2e_det_boxes.py --fold-data artifacts/kfold_grouped --weights artifacts/yolo26x_kfold_grouped --tag _grp` → 470개(FP 96) |
| 크롭 좌표계 추측 버그 — `make_crops_from_gt_polygon.py` 가 좌표 크기로 2197/8192 기준을 추측해, 8K 사진 왼쪽 위 간판을 작은 사진으로 오인 | 기존 크롭 **9개가 보도·벽을 잘라** 읽음 | 좌표를 8192 기준으로 적고 `--gt-size large` 로 자름. 470개 전부 박스와 가로세로비 일치 확인 |
| 연쇄 채점기가 간이 버전(전화번호 마스킹·브루클린 영어 전용 없음) | 같은 출력이 ~10%p 낮게 나옴 | `eval_ocr_v2.eval_engine` 을 그대로 호출. GT 크롭 열이 표 4 값을 재현 |
| 태깅의 `--no-abstain` 이 `action="store_true", default=True` — **끌 수 없는 기권 금지** | 허위 POI 124/124 가 측정이 아니라 설정의 결과 | 기본 기권 허용, 금지는 `--no-abstain` 대조군 |

같은 좌표 버그로 GT 크롭(`artifacts/gt/crop`)도 4개(`gangnam__5__crop_002`, `brooklyn__22__crop_001`,
`brooklyn__30__crop_001`, `suwon__2__crop_001`)가 잘못 잘려 있지만, 넷 다 OCR 정답이 없는 52개 쪽이라
표 3·4 채점에는 원래 들어가지 않습니다. 라벨링 작업 이미지는 원본에서 직접 잘라 영향이 없습니다.

---

## 8. 저장소 구조

```
pipeline/    run_ocr_line.py        배포 OCR (단어 검출 → 라인 병합 → 인식)
             eval_ocr_v2.py         OCR 채점 (라인 매칭, 전화번호 마스킹, 브루클린 영어 전용)
             exp_vlm_ocr.py         VLM 교정 (fixtext / fix / fixcand)
             eval_e2e_cascade.py    연쇄 OCR 측정 (eval_ocr_v2 채점기 재사용)
             eval_e2e_tagging.py    연쇄 태깅 측정 (기본 기권 허용, --tag 로 매칭표 선택)

detection/   train_*_kfold.py       간판 탐지 4모델 (Table 1)
             train_*_text_holdout.py 단어 박스 탐지 4모델 (Table 2)
             eval_det_unified.py    단일 AP@0.5 채점기
             eval_text_holdout.py   단어 박스 AP (AI Hub hold-out)
             eval_text_chain.py     단어 박스 연쇄 AP (AI Hub 전용)
             eval_wordbox_chain_gsv.py  GSV 단어 박스 AP — 정답 간판 / 연쇄 (Table 2 GSV 열)
             e2e_det_boxes.py       연쇄용 out-of-fold 간판 검출 (--fold-data/--weights/--tag)

data/        prep_wordbox_label.py  단어 박스 라벨링 작업 이미지 (GT 간판 축정렬 크롭 463장)
             make_wordbox_seed.py   라벨 씨앗 = 단어 검출기 4개 제안의 합집합
             label_wordbox.py/.html 라벨링 웹앱 (http://127.0.0.1:8777)
             export_wordbox.py      라벨 → 원본 사진 좌표 YOLO 형식
             make_crops_from_gt_polygon.py  간판 크롭 (연쇄 크롭은 --gt-size large 로)

str_baselines/  *_rec_worker.py     인식기 워커 8종 — 같은 IO 계약, --worker 로 교체
                                    (surya 만 .venv_surya 에서 실행: --worker-py 로 지정)
                yolo_text_det_worker.py  단어 탐지기 워커
                eval_ocr_controlled.py   통제 비교 + 부트스트랩 신뢰구간
                convert_pretrained_openocr.py  ABINet·MAERec 사전학습 가중치 → OpenOCR 키 변환

ocr/         train_textinthewild_ocr.py  TrOCR 학습 (--clip-grad, --warmup-steps, --lr-decay)
             train_paddle_v4_spacecat.py PaddleOCR 학습
             make_easyocr_plugin.py      EasyOCR 플러그인화

vlm/         태깅 · POI 사전(OSM · 상가정보 · NYC 인허가) · RAG

scripts/     run_final_measure.sh   ★ 최종 측정 (D42): Table 2 GSV 열 + 연쇄 A/B
             run_svtr_pipeline.sh   EasyOCR → TrOCR-base → VLM 2×3 (Table 3·4)
             run_v5x_deploy.sh      TrOCR-base 추론(run 120), B 의 GT 크롭 SVTRv2(run 121) — 연쇄 부분은 대체됨
             run_svtr_deploy.sh     (연쇄 부분 대체됨 — 7.2절)
             run_str_baselines.sh   인식기 기준선
             run_text_holdout.sh    단어 박스 탐지 학습
             run_clova_ocr.sh       CLOVA API 실행 (체크포인트로 재과금 방지)
```

> **실행 중인 셸 스크립트는 편집하지 않습니다.** bash 는 스크립트를 조금씩 읽어 가며 실행해서, 도중에
> 파일이 바뀌면 엉뚱한 위치를 읽고 죽습니다(실제로 한 번 4시간 뒤에 죽었음). 긴 작업은
> `run_final_measure.sh` 처럼 복사본(스냅샷)으로 돌립니다.

---

## 9. 실행

```bash
# 간판 탐지 (Table 1)
.venv/Scripts/python.exe detection/train_yolo26x_kfold.py
.venv/Scripts/python.exe detection/eval_det_unified.py

# 단어 박스 탐지 (Table 2)
bash scripts/run_text_holdout.sh
.venv/Scripts/python.exe detection/eval_text_holdout.py            # AI Hub 열
# GSV 열: 표 1 검출기의 out-of-fold 간판 → 크롭 → 단어 AP (정답 간판 / 연쇄)
.venv/Scripts/python.exe detection/e2e_det_boxes.py --fold-data artifacts/kfold_grouped \
    --weights artifacts/yolo26x_kfold_grouped --tag _grp
.venv/Scripts/python.exe detection/eval_wordbox_chain_gsv.py

# 연쇄 크롭 (반드시 --gt-size large)
for r in gangnam brooklyn suwon; do .venv/Scripts/python.exe data/make_crops_from_gt_polygon.py \
    --region $r --gt-csv artifacts/gt/gt_${r}_det_grp.csv --out-subdir crop_det_grp --gt-size large; done

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

# VLM 교정 (Table 4)
bash scripts/run_svtr_pipeline.sh vlm

# 최종 측정: Table 2 GSV 열 + 연쇄 A(YOLO26x 단어)·B(YOLOv5x 단어) — 스냅샷으로 실행
mkdir -p artifacts/final_measure/logs
cp scripts/run_final_measure.sh artifacts/final_measure/logs/run_snapshot.sh
bash artifacts/final_measure/logs/run_snapshot.sh
```

---

## 10. 남은 일

| 항목 | 상태 |
|---|---|
| Table 1 · 2 · 4 | **완료** (Table 2 GSV 열은 D42 재측정값) |
| Table 3 | TrOCR-base 완료(58.4%). **ABINet · MAERec 보류** — 학습 코드 오류 2건(아래), 사전학습 가중치는 변환·검증 완료 |
| Table 3 — 학습 데이터 불일치 | EasyOCR · TrOCR-base · Tesseract 는 **단어 크롭만**으로 학습(5절). 같은 75,612(단어+라인)로 재학습할지 **결정 대기** |
| 연쇄 A (배포) | **완료** (7.1절) |
| 연쇄 B (단어 검출 YOLOv5x) | **측정 중** — `artifacts/final_measure/logs/status.txt`, 9/28 새벽 완료 예정 |

### 측정 이력 (D42 점검)

| 날짜 | 무엇 | 결과 |
|---|---|---|
| 9/24 | TrOCR-base 붕괴 원인 4가지 수정 후 재학습 | 5.4절 |
| 9/27 | 연쇄 AP 누수(fold0 로 전 사진 검출, 80% 가 학습 사진) 수정 | 연쇄 AP 0.815 → **0.718** |
| 9/27 | 연쇄 간판 검출기를 표 1 과 같은 group-aware 로 통일, 크롭 좌표 버그 수정 | FP 124 → 96, 잘못 잘린 크롭 9개 제거 |
| 9/27 | 연쇄 채점기를 표 3·4 와 통일, 태깅 기권 허용, 배포 VLM 을 fix 로 | 7.2절 |
| 9/27 | 인식기 학습 데이터 전수 확인 | 3개 행이 단어만(5절) |

### ABINet · MAERec 이 멈춘 이유 (수정 전)

둘 다 `tools/train_rec.py` 가 1분 안에 죽습니다. 제가 넣은 사전학습 가중치는 정상 적재됐고(로그에
`finetune from checkpoint`), 실패는 OpenOCR 쪽입니다.

- **ABINet**: `nll_loss ... not implemented for 'Int'` — `openrec/losses/abinet_loss.py` 가 라벨을
  int32 로 넘깁니다. CrossEntropy 는 int64 가 필요합니다.
- **MAERec**: `Input image size (48*96) doesn't match model (32*128)` — ViT 는 입력 크기가 고정인데
  `maerec_signboard.yml` 의 RatioSampler 가 가변 크기를 먹입니다. 32×128 고정 리사이즈로 바꿔야 합니다.

**주의**: 첫 실행 때 `rec_abinet` 단계가 가중치 없이도 exit 0 으로 돌아 **무작위 초기화 모델의 출력**
`ocr_*_114_yabinet.csv` 를 만들었습니다. `eval_ocr_v2.py` 가 자동으로 집어가지 않도록
`artifacts/ocr_gt/_quarantine/` 으로 옮겨 뒀습니다(사유는 그 안의 README.txt).

### ABINet · MAERec 사전학습 가중치

SVTRv2 는 Union14M 사전학습 가중치에서 출발했고 PARSeq 도 사전학습을 썼습니다. ABINet·MAERec 만
스크래치로 학습하면 **반대 방향의 불공정**이 됩니다.

**OpenOCR 모델 zoo 의 Google Drive 폴더 두 개는 내려갔습니다**(로그인 없이 받아도 404). 그래서
같은 모델의 **원저자 배포본**을 받아 openrec 키 이름으로 옮겨 씁니다 —
`str_baselines/convert_pretrained_openocr.py`.

| 모델 | 받은 곳 | 학습 데이터 |
|---|---|---|
| ABINet | `https://download.openmmlab.com/mmocr/textrecog/abinet/abinet_20e_st-an_mj/abinet_20e_st-an_mj_20221005_012617-ead8c139.pth` (mmocr) | MJ+ST 합성 영어, 소문자 36자 |
| MAERec | Union14M `maerec_s_union14m.pth` ([GoogleDrive 단일 파일](https://drive.google.com/file/d/1dKLS_r3_ysWK155pSmkm7NBf5ALsEJYd/view), gdown 으로 받힘) | MAE 사전학습 ViT-S + Union14M-L |

```bash
.venv/Scripts/python.exe str_baselines/convert_pretrained_openocr.py --model abinet --src <받은 .pth>
.venv/Scripts/python.exe str_baselines/convert_pretrained_openocr.py --model maerec --src <받은 .pth>
# → external/OpenOCR/pretrained/{abinet,maerec}/best.pth (설정이 자동으로 집어 씁니다)
```

**같은 논문이라도 구현체가 다르면 이름이 달라 그냥 넣으면 한 텐서도 안 읽힙니다.** `strict=False`
는 이름이 안 맞는 텐서를 조용히 건너뛰기 때문에, 변환 없이 파일만 갖다 두면 "사전학습을 썼다"고
적어도 실제로는 스크래치입니다. 변환 스크립트는 그래서 **적재율을 반드시 출력**합니다.

| 모델 | 적재 | 무작위로 남는 것 |
|---|---|---|
| ABINet | 455 / 463 (98.3%) | 분류층 3종·`language.proj`(한국어 charset) + 사인파 위치부호 1 |
| MAERec | 270 / 273 (98.9%) | 단어 임베딩·분류층(한국어 charset) + 사인파 위치부호 1 |

변환이 맞는지는 **원본 언어로 읽혀 봤습니다**(키 개수만으로는 순서가 뒤바뀐 걸 못 잡습니다).

- ABINet: 36자 영어 사전으로 모델을 세우고 분류층 인덱스만 재배치한 뒤 합성 단어 이미지를 읽혀
  **3폰트 × 12단어 = 36/36 정확 일치**.
- MAERec: 인코더 150개 텐서는 이름만 바뀐 값 동일. 디코더는 q/k/v 를 하나로 합치는 규약이 유일한
  위험 지점이라, 합친 어텐션 출력과 원본 q/k/v 계산을 직접 비교해 **최대오차 0.00e+00**.

> 남는 차이 두 가지는 미세조정으로 흡수됩니다. (1) ABINet 원본은 **소문자 영어 36자**만 읽던
> 모델이라 언어모델 분기가 한국어를 본 적이 없습니다. (2) 원본은 ImageNet 정규화로 학습됐고
> 이 저장소 파이프라인은 openrec 기본 정규화를 씁니다.

### GSV 단어 박스 라벨링 — 완료

간판 463개 전수, **단어 박스 1,499개**(강남 485 · 브루클린 605 · 수원 409). 간판당 3.24개로,
검출기를 학습한 AI Hub signboard_v3 의 장당 2.88개와 같은 눈금입니다(단어 경계 기준이 일치).
원래 목표치 930 은 OCR 정답 텍스트의 어절 수로 잡은 추정이었고, 실제로 단어 단위로 치니 더 많았습니다.

**씨앗 편향 — 반드시 보고할 한계.** 최종 박스 1,499개 중 **1,459개(97.3%)가 검출기 제안과 일치**하고,
사람이 백지에서 새로 그린 것은 40개(2.7%)입니다. 제안 3,440개 중 57.6% 를 지워 정밀도 쪽 검수는 충실히
됐지만, **네 검출기가 모두 놓친 단어는 GT 에도 거의 없습니다.** 그래서 GSV 단어 AP 의 절대값은
낙관적이고, 표 2 GSV 열은 모델 간 비교로만 읽어야 합니다.

> **주의:** 라벨 씨앗으로 YOLO 검출 결과만 쓰면 안 됩니다. YOLO 가 놓친 박스는 라벨에도 안 생겨
> YOLO 에게 유리하게 기웁니다. 백지에서 그리거나, 씨앗을 쓸 거면 **4개 검출기 결과의 합집합**을 깔고
> 전수 수정해야 공정합니다.

#### 작업 도구

```bash
.venv/Scripts/python.exe data/prep_wordbox_label.py     # ① 작업 이미지 463장 생성 (1회)
.venv/Scripts/python.exe data/make_wordbox_seed.py      # ② 4개 검출기 합집합 씨앗 (--cpu 로 GPU 회피)
.venv/Scripts/python.exe data/label_wordbox.py          # ③ http://127.0.0.1:8777 에서 작업
.venv/Scripts/python.exe data/export_wordbox.py         # ④ 원본 사진 좌표 YOLO 형식으로 내보내기
```

**① 은 GT 크롭이 아니라 원본 사진에서 자릅니다.** `artifacts/gt/crop` 의 GT 크롭은 폴리곤 마스크 +
**원근 보정** + CLAHE 를 거친 이미지라 거기 그린 박스는 사진 좌표로 되돌릴 수 없습니다. 연쇄 AP 는
원본 좌표계에서 채점하므로, GT 폴리곤의 **축정렬 bbox**(여유 6%)로 원본에서 잘라 작업 이미지를
만듭니다. 되돌리는 식은 평행이동·배율뿐입니다. 간판이 기울어 이웃 글자가 같이 들어오므로 폴리곤
외곽선을 화면에 겹쳐 그려 경계를 보여 줍니다.

**작업 대상은 463장**입니다 — OCR GT 가 있는 411장(앞쪽에 정렬)에 더해, 간판 GT 는 있지만 OCR GT 가
없는 52장이 뒤에 붙습니다. 이 52장을 빼면 그 간판에서 나온 예측이 전부 오검출로 잡혀 분모가
비뚤어집니다. 글자가 없으면 <kbd>0</kbd> 한 번으로 "글자 없음 + 완료" 처리됩니다.

**단어 단위로 칩니다.** 검출기들은 AI Hub **단어** 박스로 학습됐고 Table 2 의 옆 칸도 AI Hub 단어
박스에서 잰 값이라, GSV 라벨을 줄 단위로 치면 같은 열에 다른 것을 재게 됩니다. 띄어쓰기로 끊습니다 —
예: `milk` / `밀크뮤직타운` / `MILK` / `MUSIC` / `TOWN` / `OPEN@6PM` / `CLOSE@10AM` / `B1`.

**③ 화면에 보이는 박스는 전부 편집 대상입니다.** 처음 여는 이미지는 제안(기본값: 2개 이상 모델이
합의한 것)이 **진짜 박스로 깔린 채** 뜹니다. 클릭해 고르고, 안쪽을 끌어 옮기고, 모서리를 끌어 크기를
바꾸고, 오른쪽 위 <kbd>✕</kbd> 나 <kbd>Del</kbd> 로 지웁니다. 제안은 줄 단위와 단어 단위가 섞여
있으므로 **단어 기준으로 쪼개거나 합치는 수정이 반드시 필요합니다.**

단축키: 빈 곳 드래그=새 박스, <kbd>Del</kbd>=삭제, <kbd>Ctrl+Z</kbd>=되돌리기, <kbd>Ctrl+S</kbd>=저장,
<kbd>←</kbd><kbd>→</kbd>=이동, <kbd>Enter</kbd>=완료 후 다음, <kbd>0</kbd>=글자 없음,
<kbd>F</kbd>=화면 맞춤, <kbd>P</kbd>=기준 미달 제안 보기, 휠=확대, <kbd>Space</kbd>+드래그=이동.
같은 동작이 상단 버튼에도 있습니다. 화면 위에 그 크롭의 **정답 텍스트**가 떠 있어 몇 단어가 있어야
하는지 바로 보입니다.

저장은 박스를 바꿀 때마다 자동으로 되고(`artifacts/gt/wordbox/labels.json`, 임시파일 교체라 중간에
꺼도 안 깨집니다) <kbd>Ctrl+S</kbd>·저장 버튼으로도 됩니다. 상단 오른쪽에 저장 상태가 표시되고,
저장 안 된 변경이 있으면 창을 닫을 때 경고합니다. **손대지 않은 이미지는 저장되지 않습니다** —
깔아만 둔 제안이 라벨로 굳지 않게 하려는 것입니다.

**제안은 정답이 아니라 지우고 고치라고 깔아 둔 것입니다.** 어느 모델이 제안했는지(`by`)가 함께
저장되고, 합의한 모델 수로 기준을 바꿀 수 있습니다(상단 드롭다운). 참고로 전체 3,440개 중
**4개 모델이 모두 합의한 것이 897개**(라벨 목표 930 과 근접), **한 모델만 찾은 것이 1,732개**로
후자에 오검출이 몰려 있습니다. 그래서 기본 기준을 2개 이상으로 두고, 기준 미달 제안은
<kbd>P</kbd> 로 흐리게 띄워 "네 모델 중 하나만 본 진짜 글자"를 놓치지 않게 합니다.

**④ 는 완료 표시된 것만** 내보냅니다(`--allow-partial` 로 중간 점검 가능). 한 사진에 간판이 여러 개면
겹치는 영역의 같은 단어가 두 번 들어올 수 있어 사진 단위로 IoU 0.6 중복 정리를 합니다. 결과는
`artifacts/gt/wordbox/yolo/{images,labels}` 에 `eval_text_holdout.py` 와 같은 YOLO 형식으로 놓입니다
(사진은 하드링크라 8K 원본을 복사하지 않습니다).
