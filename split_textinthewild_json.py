#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split the large Text-in-the-Wild data_info JSON into a single-type subset
(e.g. signboard = image type "sign") WITHOUT loading the whole file into memory.

Strategy (constant memory except for one small id-set):
  Pass 1: stream images, write those matching --type to the output's "images"
          array, and remember their ids.
  Pass 2: stream annotations, write those whose image_id is in that id-set.

Output is valid JSON: {"info":..., "images":[...], "annotations":[...]}.

Example:
  python split_textinthewild_json.py \
    --json artifacts/textinthewild_data_info.json \
    --type sign \
    --out artifacts/signboard_data_info.json \
    --image-dir artifacts/Signboard
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import ijson


def list_available_images(image_dir: Path) -> set[str]:
    exts = {".jpg", ".jpeg", ".png"}
    return {p.name for p in image_dir.iterdir() if p.suffix.lower() in exts}


def split(json_path: Path, type_name: str, out_path: Path,
          image_dir: Path | None, progress_every: int) -> None:
    available: set[str] | None = None
    if image_dir is not None:
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Image dir not found: {image_dir}")
        available = list_available_images(image_dir)
        print(f"[INFO] restricting to {len(available)} files on disk in {image_dir}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    keep_ids: set[str] = set()

    with out_path.open("w", encoding="utf-8") as out:
        out.write('{\n"info": {"name": "Text in the wild Dataset", "subset_type": ')
        out.write(json.dumps(type_name, ensure_ascii=False))
        out.write('},\n"images": [\n')

        # Pass 1: images
        scanned = 0
        written = 0
        with json_path.open("rb") as f:
            for img in ijson.items(f, "images.item"):
                scanned += 1
                if str(img.get("type", "")) != type_name:
                    continue
                file_name = str(img.get("file_name", ""))
                if available is not None and file_name not in available:
                    continue
                obj = {
                    "id": str(img.get("id", "")),
                    "width": int(img.get("width", 0)),
                    "height": int(img.get("height", 0)),
                    "file_name": file_name,
                    "type": str(img.get("type", "")),
                }
                if written:
                    out.write(",\n")
                out.write(json.dumps(obj, ensure_ascii=False))
                keep_ids.add(obj["id"])
                written += 1
                if progress_every and scanned % progress_every == 0:
                    print(f"[INFO]   pass1 scanned {scanned} images, kept {written}")
        print(f"[INFO] pass1 done: {written} images of type '{type_name}'")

        out.write('\n],\n"annotations": [\n')

        # Pass 2: annotations
        scanned = 0
        kept = 0
        with json_path.open("rb") as f:
            for ann in ijson.items(f, "annotations.item"):
                scanned += 1
                if str(ann.get("image_id", "")) not in keep_ids:
                    continue
                bbox = ann.get("bbox", []) or []
                obj = {
                    "id": str(ann.get("id", "")),
                    "image_id": str(ann.get("image_id", "")),
                    "text": ann.get("text", ""),
                    "attributes": {"class": str((ann.get("attributes") or {}).get("class", ""))},
                    "bbox": [int(v) for v in bbox] if len(bbox) == 4 else [],
                }
                if kept:
                    out.write(",\n")
                out.write(json.dumps(obj, ensure_ascii=False))
                kept += 1
                if progress_every and scanned % progress_every == 0:
                    print(f"[INFO]   pass2 scanned {scanned} annotations, kept {kept}")
        print(f"[INFO] pass2 done: {kept} annotations")

        out.write("\n]\n}\n")

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[DONE] wrote {out_path} ({size_mb:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--type", default="sign", help="image type to keep (signboard = 'sign')")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--image-dir", type=Path, default=None,
                    help="if set, keep only images present on disk here")
    ap.add_argument("--progress-every", type=int, default=200000)
    args = ap.parse_args()
    split(args.json, args.type, args.out, args.image_dir, args.progress_every)


if __name__ == "__main__":
    main()
