#!/usr/bin/env bash
set -e
PY=.venv/Scripts/python.exe
echo "===== STAGE 1: pretrain YOLOv5s on external signboard dataset (artifacts/archive) ====="
$PY detection/pretrain_archive_v5.py --epochs 60 --patience 15 --imgsz 640 --batch 16 --device 0
echo "===== STAGE 2: 5-fold fine-tune on GSV (signboard-pretrained YOLOv5s base) ====="
$PY detection/one_click_finetune_yolo11.py --prefix total \
    --model artifacts/yolov5_pretrain/v5s_archive/weights/best.pt \
    --kfold 5 --imgsz 1280 --batch 8 --lr 1e-4 --patience 30 --max-epochs 150 \
    --seed 42 --device 0 \
    --out-root artifacts/yolov5_kfold \
    --export-best-to artifacts/yolo/best_yolov5_kfold.pt
echo "===== PIPELINE_DONE ====="
