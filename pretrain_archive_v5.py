#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pretrain YOLOv5s on the external single-class signboard dataset (artifacts/archive).
Produces a signboard-aware base checkpoint for GSV 5-fold fine-tuning."""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov5su.pt")
    ap.add_argument("--data", default="artifacts/archive/data.yaml")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--project", default="artifacts/yolov5_pretrain")
    ap.add_argument("--name", default="v5s_archive")
    args = ap.parse_args()

    from ultralytics import YOLO
    m = YOLO(args.model)
    m.train(data=args.data, epochs=args.epochs, patience=args.patience,
            imgsz=args.imgsz, batch=args.batch, device=args.device,
            optimizer="Adam", lr0=1e-3, pretrained=True, verbose=True,
            plots=False, project=args.project, name=args.name, exist_ok=True)
    print("PRETRAIN_DONE")


if __name__ == "__main__":
    main()
