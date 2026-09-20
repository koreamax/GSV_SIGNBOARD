### GSV 라인 채점 — line exact / CER / WER (eval_ocr_v2, --mask-phone, brooklyn en-only)

| 엔진 | 부류 | Gangnam | Brooklyn | Suwon | Total | WAR (Total) |
|---|---|---|---|---|---|---|
| Tesseract 5.5 (kor+eng) | general engine, off-the-shelf | 34.0% / 0.551 / 1.033 | 42.7% / 0.376 / 1.274 | 13.3% / 0.769 / 1.403 | 30.6% / 0.509 / 1.228 | 0.333 |
| Tesseract 5.5 (kor fine-tuned + eng) | general engine, fine-tuned, same data | 35.9% / 0.526 / 0.971 | 42.7% / 0.370 / 1.180 | 18.8% / 0.699 / 1.245 | 32.9% / 0.484 / 1.128 | 0.287 |
| EasyOCR (CRNN, fine-tuned v3) | general engine, per-box | 37.3% / 0.538 / 1.459 | 43.2% / 0.406 / 2.110 | 34.8% / 0.428 / 1.510 | 38.5% / 0.449 / 1.741 | 0.290 |
| Surya 0.14 (rec2) | general engine, off-the-shelf | 41.0% / 0.502 / 1.167 | 60.3% / 0.198 / 0.760 | 17.1% / 0.650 / 1.137 | 40.2% / 0.381 / 0.991 | 0.396 |
| CLOVA OCR General (NAVER, commercial API) | commercial API, off-the-shelf | 57.1% / 0.318 / 0.967 | 67.8% / 0.171 / 0.859 | 48.1% / 0.383 / 1.153 | 57.9% / 0.258 / 0.971 | 0.563 |
| TrOCR-small (fine-tuned v3) | document STR, per-box | 26.9% / 0.635 / 1.469 | 46.2% / 0.384 / 2.154 | 27.6% / 0.516 / 1.510 | 33.6% / 0.485 / 1.762 | 0.343 |
| PARSeq (ViT-S, fine-tuned) | academic STR, same data | 63.7% / 0.248 / 0.652 | 70.9% / 0.151 / 0.483 | 50.3% / 0.299 / 0.830 | 62.0% / 0.210 / 0.629 | 0.619 |
| SVTRv2-B (fine-tuned) | academic STR, same data | 72.6% / 0.209 / 0.593 | 77.4% / 0.096 / 0.433 | 60.8% / 0.249 / 0.743 | 70.6% / 0.161 / 0.566 | 0.682 |
| PP-OCRv5 rec (v5_lines, single model) | fine-tuned, same data, no vote | 71.7% / 0.194 / 0.554 | 70.9% / 0.122 / 0.478 | 58.0% / 0.256 / 0.693 | 67.2% / 0.171 / 0.559 | 0.643 |
| PP-OCRv5 rec (v5_lines, deployed: 3-way vote + en model) | deployed pipeline | 37.7% / 0.611 / 0.882 | 44.7% / 0.471 / 0.901 | 41.4% / 0.571 / 0.875 | 41.2% / 0.533 / 0.888 | 0.399 |

### In-domain (signboard_v3 test) — exact / CER / WER, 1:1 크롭 채점

| 엔진 | test_line (1,631 실라인) | test_word (7,756 단어) |
|---|---|---|
| Tesseract 5.5 (kor+eng) | 21.0% / 0.602 / 0.815 | 34.4% / 0.571 / 0.789 |
| Tesseract 5.5 (kor fine-tuned + eng) | 24.9% / 0.567 / 0.940 | 42.4% / 0.486 / 0.643 |
| Surya 0.14 (rec2) | 36.4% / 0.793 / 0.802 | — |
| PARSeq (ViT-S, fine-tuned) | 61.4% / 0.153 / 0.344 | 89.2% / 0.047 / 0.130 |
| SVTRv2-B (fine-tuned) | 75.3% / 0.081 / 0.256 | 94.2% / 0.023 / 0.077 |
| PP-OCRv5 rec (v5_lines, deployed: 3-way vote + en model) | 71.8% / 0.102 / 0.336 | 87.2% / 0.070 / 0.146 |
