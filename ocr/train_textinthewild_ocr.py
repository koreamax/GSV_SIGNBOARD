#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare and fine-tune OCR data from Text in the Wild annotations.

Defaults match this repository:
  - annotations: artifacts/textinthewild_data_info.json
  - images:      artifacts/Traffic_Sign
  - output:      artifacts/ocr_training/textinthewild

Typical use:
  python train_textinthewild_ocr.py prepare
  python train_textinthewild_ocr.py train-trocr --epochs 5 --batch-size 8
  python train_textinthewild_ocr.py all --epochs 5 --batch-size 8

Notes:
  - EasyOCR does not expose a stable one-call fine-tune API. This script exports
    EasyOCR/CRNN-friendly labels, images, and charset files under easyocr/.
  - TrOCR fine-tuning is implemented directly with PyTorch + transformers.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
import shutil
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import ijson
from PIL import Image, ImageOps


HERE = Path(__file__).resolve().parents[1]
DEFAULT_JSON = HERE / "artifacts" / "textinthewild_data_info.json"
DEFAULT_IMAGE_DIR = HERE / "artifacts" / "Traffic_Sign"
DEFAULT_OUT_DIR = HERE / "artifacts" / "ocr_training" / "textinthewild"


@dataclass(frozen=True)
class CropRow:
    image_path: str
    text: str
    split: str
    source_image: str
    source_image_id: str
    annotation_id: str
    cls: str
    x: int
    y: int
    w: int
    h: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build OCR crops from Text in the Wild and train/export OCR recognizers."
    )
    p.add_argument(
        "command",
        choices=["prepare", "export-easyocr", "train-trocr", "all"],
        help="What to run. all = prepare + train-trocr.",
    )

    p.add_argument("--json", type=Path, default=DEFAULT_JSON, help="Text in the Wild JSON file.")
    p.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR, help="Directory with source jpg images.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output training directory.")
    p.add_argument(
        "--classes",
        nargs="+",
        default=["word"],
        help="Annotation classes to use. Recommended: word. Optional: character.",
    )
    p.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio by source image.")
    p.add_argument("--test-ratio", type=float, default=0.1,
                   help="Held-out test split ratio by source image (0 disables; test is never "
                        "trained on and evaluated once after training).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--padding", type=float, default=0.04, help="BBox padding ratio around each annotation.")
    p.add_argument("--min-side", type=int, default=4, help="Skip crops smaller than this.")
    p.add_argument("--max-text-len", type=int, default=64, help="Skip labels longer than this many chars.")
    p.add_argument("--max-samples", type=int, default=0, help="Debug cap. 0 means no cap.")
    p.add_argument("--overwrite", action="store_true", help="Replace existing prepared crop output.")
    p.add_argument("--preview", type=int, default=16, help="Save this many preview crops.")
    p.add_argument(
        "--image-cache",
        type=int,
        default=16,
        help="Max source images kept open at once (LRU). Annotations are grouped by image, "
        "so a small cache avoids unbounded memory while streaming.",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=200000,
        help="Log progress every N streamed entries.",
    )

    p.add_argument(
        "--trocr-model",
        default="microsoft/trocr-small-printed",
        help="Base TrOCR model name or local path.",
    )
    p.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=None,
        help="Optional tokenizer directory to swap in (e.g. a byte-level BPE that "
             "covers Korean losslessly). Decoder embeddings are resized to match. "
             "trocr-small-printed's own sentencepiece vocab maps many Hangul "
             "syllables to <unk>, silently corrupting labels.",
    )
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--num-workers", type=int, default=0, help="Windows default is 0 for fewer surprises.")
    p.add_argument("--max-target-length", type=int, default=64)
    p.add_argument("--save-every-epoch", action="store_true")
    p.add_argument("--augment", action="store_true", help="Enable train-set image augmentation (TrOCR).")

    p.add_argument(
        "--easyocr-trainer-dir",
        type=Path,
        default=None,
        help="Optional EasyOCR/deep-text-recognition trainer directory. If supplied, command export-easyocr also writes a helper command file.",
    )
    return p.parse_args()


def clean_text(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def safe_preview_stem(text: str, limit: int = 32) -> str:
    text = clean_text(text)[:limit] or "empty"
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", text)
    return text.strip(" ._") or "empty"


def safe_crop_box(
    bbox: list[float] | tuple[float, float, float, float],
    img_w: int,
    img_h: int,
    padding: float,
) -> tuple[int, int, int, int] | None:
    if not bbox or len(bbox) != 4:
        return None
    x, y, w, h = [float(v) for v in bbox]
    if w <= 0 or h <= 0:
        return None
    pad_x = w * padding
    pad_y = h * padding
    left = max(0, math.floor(x - pad_x))
    top = max(0, math.floor(y - pad_y))
    right = min(img_w, math.ceil(x + w + pad_x))
    bottom = min(img_h, math.ceil(y + h + pad_y))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def load_manifest(path: Path) -> list[CropRow]:
    rows: list[CropRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                CropRow(
                    image_path=row["image_path"],
                    text=row["text"],
                    split=row["split"],
                    source_image=row["source_image"],
                    source_image_id=row["source_image_id"],
                    annotation_id=row["annotation_id"],
                    cls=row["class"],
                    x=int(row["x"]),
                    y=int(row["y"]),
                    w=int(row["w"]),
                    h=int(row["h"]),
                )
            )
    return rows


def write_manifest(path: Path, rows: Iterable[CropRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "image_path",
        "text",
        "split",
        "source_image",
        "source_image_id",
        "annotation_id",
        "class",
        "x",
        "y",
        "w",
        "h",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "image_path": r.image_path,
                    "text": r.text,
                    "split": r.split,
                    "source_image": r.source_image,
                    "source_image_id": r.source_image_id,
                    "annotation_id": r.annotation_id,
                    "class": r.cls,
                    "x": r.x,
                    "y": r.y,
                    "w": r.w,
                    "h": r.h,
                }
            )


def choose_splits(image_ids: list[str], val_ratio: float, test_ratio: float, seed: int) -> dict[str, str]:
    """Source-image-level train/val/test split (no crop of one photo leaks across splits)."""
    rng = random.Random(seed)
    shuffled = list(image_ids)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_val = max(1, int(round(n * val_ratio))) if n > 1 and val_ratio > 0 else 0
    n_test = max(1, int(round(n * test_ratio))) if n > 2 and test_ratio > 0 else 0
    val_ids = set(shuffled[:n_val])
    test_ids = set(shuffled[n_val:n_val + n_test])
    out = {}
    for image_id in image_ids:
        out[image_id] = "val" if image_id in val_ids else ("test" if image_id in test_ids else "train")
    return out


class BoundedImageCache:
    """LRU cache of open PIL images with a hard size cap.

    Text-in-the-Wild annotations are grouped by image_id, so even a tiny cache
    keeps a high hit rate while guaranteeing we never hold more than `maxsize`
    decoded source images in memory. Evicted images are closed immediately.
    """

    def __init__(self, image_dir: Path, maxsize: int = 16):
        self.image_dir = image_dir
        self.maxsize = max(1, maxsize)
        self._cache: "OrderedDict[str, Image.Image]" = OrderedDict()

    def get(self, name: str) -> Image.Image:
        img = self._cache.get(name)
        if img is not None:
            self._cache.move_to_end(name)
            return img
        img = Image.open(self.image_dir / name).convert("RGB")
        self._cache[name] = img
        self._cache.move_to_end(name)
        while len(self._cache) > self.maxsize:
            _, evicted = self._cache.popitem(last=False)
            try:
                evicted.close()
            except Exception:
                pass
        return img

    def close_all(self) -> None:
        for img in self._cache.values():
            try:
                img.close()
            except Exception:
                pass
        self._cache.clear()


def list_available_images(image_dir: Path) -> set[str]:
    exts = {".jpg", ".jpeg", ".png"}
    return {p.name for p in image_dir.iterdir() if p.suffix.lower() in exts}


def stream_json_array(json_path: Path, prefix: str) -> Iterator[dict]:
    """Yield each object of a top-level JSON array without loading the whole file.

    prefix is an ijson path such as 'images.item' or 'annotations.item'.
    """
    with json_path.open("rb") as f:
        yield from ijson.items(f, prefix)


def build_image_index(json_path: Path, available: set[str], progress_every: int) -> dict[str, str]:
    """Pass 1: stream the images array, keep only images that exist on disk."""
    image_by_id: dict[str, str] = {}
    scanned = 0
    for img in stream_json_array(json_path, "images.item"):
        scanned += 1
        file_name = str(img.get("file_name", ""))
        if file_name in available:
            image_by_id[str(img.get("id", ""))] = file_name
        if progress_every and scanned % progress_every == 0:
            print(f"[INFO]   scanned {scanned} image entries, matched {len(image_by_id)} on disk")
    print(f"[INFO] Pass 1 done: matched {len(image_by_id)} images on disk (of {scanned} entries)")
    return image_by_id


def prepare_dataset(args: argparse.Namespace) -> list[CropRow]:
    json_path = args.json.resolve()
    image_dir = args.image_dir.resolve()
    out_dir = args.out_dir.resolve()
    crop_dir = out_dir / "crops"
    preview_dir = out_dir / "preview"
    manifest_path = out_dir / "labels.csv"

    if crop_dir.exists() and not args.overwrite:
        print(f"[INFO] Prepared crops already exist: {crop_dir}")
        print("[INFO] Use --overwrite to rebuild.")
        if manifest_path.exists():
            return load_manifest(manifest_path)
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    reset_dir(crop_dir)
    reset_dir(preview_dir)

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image dir not found: {image_dir}")
    print(f"[INFO] Indexing image files on disk: {image_dir}")
    available = list_available_images(image_dir)
    print(f"[INFO] {len(available)} image files on disk")
    if not available:
        raise RuntimeError(f"No image files found in {image_dir}")

    print(f"[INFO] Pass 1/2: streaming image index from {json_path}")
    image_by_id = build_image_index(json_path, available, args.progress_every)
    if not image_by_id:
        raise RuntimeError(f"No JSON images matched files in {image_dir}")

    split_by_image = choose_splits(list(image_by_id.keys()), args.val_ratio, args.test_ratio, args.seed)
    wanted_classes = set(args.classes)

    rows: list[CropRow] = []
    cache = BoundedImageCache(image_dir, maxsize=args.image_cache)
    skipped = {
        "missing_image": 0,
        "class": 0,
        "empty_text": 0,
        "too_long": 0,
        "bad_box": 0,
        "too_small": 0,
        "crop_error": 0,
    }

    print(f"[INFO] Pass 2/2: streaming annotations (image cache={args.image_cache})")
    scanned = 0
    try:
        for ann in stream_json_array(json_path, "annotations.item"):
            scanned += 1
            if args.progress_every and scanned % args.progress_every == 0:
                print(f"[INFO]   scanned {scanned} annotations, kept {len(rows)} crops")

            img_id = str(ann.get("image_id", ""))
            src_name = image_by_id.get(img_id)
            if not src_name:
                skipped["missing_image"] += 1
                continue

            cls = str((ann.get("attributes") or {}).get("class", ""))
            if cls not in wanted_classes:
                skipped["class"] += 1
                continue

            text = clean_text(str(ann.get("text", "")))
            if not text:
                skipped["empty_text"] += 1
                continue
            if args.max_text_len and len(text) > args.max_text_len:
                skipped["too_long"] += 1
                continue

            try:
                img = cache.get(src_name)
                img_w, img_h = img.size
                box = safe_crop_box(ann.get("bbox", []), img_w, img_h, args.padding)
                if box is None:
                    skipped["bad_box"] += 1
                    continue
                left, top, right, bottom = box
                if (right - left) < args.min_side or (bottom - top) < args.min_side:
                    skipped["too_small"] += 1
                    continue

                split = split_by_image[img_id]
                ann_id = str(ann.get("id", len(rows)))
                out_name = f"{split}/{Path(src_name).stem}__ann_{ann_id}.jpg"
                out_path = crop_dir / out_name
                out_path.parent.mkdir(parents=True, exist_ok=True)
                crop = img.crop((left, top, right, bottom))
                crop = ImageOps.exif_transpose(crop)
                crop.save(out_path, quality=95)

                if len(rows) < args.preview:
                    crop.save(preview_dir / f"{len(rows):04d}_{safe_preview_stem(text)}.jpg", quality=95)

                rows.append(
                    CropRow(
                        image_path=str(out_path.relative_to(out_dir)).replace("\\", "/"),
                        text=text,
                        split=split,
                        source_image=src_name,
                        source_image_id=img_id,
                        annotation_id=ann_id,
                        cls=cls,
                        x=left,
                        y=top,
                        w=right - left,
                        h=bottom - top,
                    )
                )
                if args.max_samples and len(rows) >= args.max_samples:
                    break
            except Exception:
                skipped["crop_error"] += 1
                continue
    finally:
        cache.close_all()

    write_manifest(manifest_path, rows)
    export_easyocr(args, rows)

    train_count = sum(1 for r in rows if r.split == "train")
    val_count = sum(1 for r in rows if r.split == "val")
    test_count = sum(1 for r in rows if r.split == "test")
    print(f"[DONE] Prepared {len(rows)} OCR crops -> {manifest_path}")
    print(f"[DONE] train={train_count}, val={val_count}, test={test_count}, skipped={skipped}")
    print(f"[DONE] EasyOCR export -> {out_dir / 'easyocr'}")
    return rows


def export_easyocr(args: argparse.Namespace, rows: list[CropRow] | None = None) -> None:
    out_dir = args.out_dir.resolve()
    manifest_path = out_dir / "labels.csv"
    if rows is None:
        rows = load_manifest(manifest_path)

    easy_dir = out_dir / "easyocr"
    easy_dir.mkdir(parents=True, exist_ok=True)

    chars = sorted({ch for r in rows for ch in r.text if not ch.isspace()})
    (easy_dir / "character.txt").write_text("".join(chars), encoding="utf-8")

    for split in ("train", "val", "test"):
        split_rows = [r for r in rows if r.split == split]
        if not split_rows and split == "test":
            continue
        label_path = easy_dir / f"{split}.txt"
        with label_path.open("w", encoding="utf-8", newline="\n") as f:
            for r in split_rows:
                f.write(f"../{r.image_path}\t{r.text}\n")

    readme = easy_dir / "README_easyocr_training.txt"
    readme.write_text(
        "\n".join(
            [
                "EasyOCR training export",
                "",
                "Files:",
                "- train.txt / val.txt: tab-separated relative_image_path<TAB>text",
                "- character.txt: charset found in this dataset",
                "",
                "EasyOCR itself is mainly an inference package. For recognition fine-tuning,",
                "use EasyOCR's trainer or a deep-text-recognition-benchmark style trainer",
                "and point it at these labels/crops, or convert these labels to LMDB.",
                "",
                "Recommended quick path:",
                "1. Train TrOCR with this script first.",
                "2. For EasyOCR, use these files as the source manifest for a CRNN/CTC trainer.",
            ]
        ),
        encoding="utf-8",
    )

    if args.easyocr_trainer_dir:
        trainer = args.easyocr_trainer_dir.resolve()
        helper = easy_dir / "run_easyocr_trainer_example.txt"
        helper.write_text(
            "\n".join(
                [
                    f"Trainer dir: {trainer}",
                    "",
                    "This is a placeholder helper because EasyOCR trainer layouts differ by repo.",
                    "Use train.txt, val.txt, and character.txt above as the data source.",
                    "If your trainer expects LMDB, convert these image/text rows to LMDB first.",
                ]
            ),
            encoding="utf-8",
        )


def build_trocr_augment():
    """GSV-flavoured, legibility-preserving augmentation for text crops (train only).

    Targets the degradations seen in street-view sign crops: oblique viewing angle
    (perspective/rotation), motion/defocus blur, exposure swings, and the
    low-resolution look of far-away signs (downscale-upscale round trip)."""
    from torchvision import transforms

    def lowres_roundtrip(img):
        # moderate: tiny word crops lose legibility fast below ~0.6x
        w, h = img.size
        f = random.uniform(0.6, 0.9)
        small = img.resize((max(8, int(w * f)), max(8, int(h * f))), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)

    return transforms.Compose(
        [
            transforms.RandomApply([transforms.RandomRotation(4, fill=255)], p=0.3),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.2)], p=0.5
            ),
            transforms.RandomApply(
                [transforms.RandomPerspective(distortion_scale=0.12, p=1.0, fill=255)], p=0.25
            ),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.5))], p=0.25),
            transforms.RandomApply([transforms.Lambda(lowres_roundtrip)], p=0.15),
        ]
    )


class TrocrDataset:
    def __init__(self, rows: list[CropRow], root: Path, processor, max_target_length: int, augment: bool = False):
        self.rows = rows
        self.root = root
        self.processor = processor
        self.max_target_length = max_target_length
        self.augment = build_trocr_augment() if augment else None

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        img = Image.open(self.root / row.image_path).convert("RGB")
        if self.augment is not None:
            img = self.augment(img)
        pixel_values = self.processor(images=img, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(
            row.text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
        ).input_ids
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
        return {"pixel_values": pixel_values, "labels": labels}


def collate_trocr(batch: list[dict]):
    import torch

    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": torch.tensor([b["labels"] for b in batch], dtype=torch.long),
    }


def train_trocr(args: argparse.Namespace) -> None:
    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    except Exception as exc:
        raise RuntimeError(
            "TrOCR dependencies failed to import. In this environment I saw a NumPy/SciPy ABI error. "
            "Fix with something like: pip install --force-reinstall \"numpy<2\" scipy transformers"
        ) from exc

    out_dir = args.out_dir.resolve()
    model_dir = out_dir / "trocr_model"
    checkpoint_dir = out_dir / "trocr_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(out_dir / "labels.csv")
    train_rows = [r for r in rows if r.split == "train"]
    val_rows = [r for r in rows if r.split == "val"]
    if not train_rows:
        raise RuntimeError("No train rows found. Run prepare first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Loading TrOCR base model: {args.trocr_model}")
    processor = TrOCRProcessor.from_pretrained(args.trocr_model)
    model = VisionEncoderDecoderModel.from_pretrained(args.trocr_model)

    if args.tokenizer_dir:
        from transformers import AutoTokenizer

        new_tok = AutoTokenizer.from_pretrained(str(args.tokenizer_dir))
        processor = TrOCRProcessor(image_processor=processor.image_processor, tokenizer=new_tok)
        n_vocab = len(new_tok)
        model.decoder.resize_token_embeddings(n_vocab)
        model.config.decoder.vocab_size = n_vocab
        print(f"[INFO] Swapped tokenizer <- {args.tokenizer_dir} "
              f"({type(new_tok).__name__}, vocab={n_vocab}); decoder embeddings resized.")

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    # transformers>=4.4x prefers generation_config at generate() time; keep it in
    # sync with the training-time config or saved models decode from the wrong
    # start token (symptom: leading syllables truncated despite low loss).
    model.generation_config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    model.generation_config.eos_token_id = processor.tokenizer.sep_token_id
    model.generation_config.bos_token_id = processor.tokenizer.cls_token_id
    model.to(device)

    train_ds = TrocrDataset(train_rows, out_dir, processor, args.max_target_length, augment=args.augment)
    val_ds = TrocrDataset(val_rows, out_dir, processor, args.max_target_length, augment=False) if val_rows else None
    if args.augment:
        print("[INFO] training-set augmentation: ON")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_trocr,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_trocr)
        if val_ds
        else None
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))   # torch 2.2-compatible

    print(f"[INFO] Training on {device} | train={len(train_rows)} val={len(val_rows)}")
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=(device.type == "cuda")):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu())
            if step % 50 == 0:
                print(f"[TRAIN] epoch={epoch} step={step}/{len(train_loader)} loss={total_loss / step:.4f}")

        train_loss = total_loss / max(1, len(train_loader))
        val_loss = None
        if val_loader:
            model.eval()
            losses = []
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    losses.append(float(model(**batch).loss.detach().cpu()))
            val_loss = sum(losses) / max(1, len(losses))

        elapsed = time.time() - t0
        msg = f"[EPOCH] {epoch}/{args.epochs} train_loss={train_loss:.4f}"
        if val_loss is not None:
            msg += f" val_loss={val_loss:.4f}"
        msg += f" time={elapsed:.1f}s"
        print(msg)

        should_save = args.save_every_epoch or val_loss is None or val_loss < best_val
        if val_loss is not None and val_loss < best_val:
            best_val = val_loss
        if should_save:
            epoch_dir = checkpoint_dir / f"epoch_{epoch:03d}"
            model.save_pretrained(epoch_dir)
            processor.save_pretrained(epoch_dir)
            print(f"[SAVE] {epoch_dir}")

    model.save_pretrained(model_dir)
    processor.save_pretrained(model_dir)
    print(f"[DONE] Final TrOCR model saved -> {model_dir}")

    test_rows = [r for r in rows if r.split == "test"]
    if test_rows:
        evaluate_trocr_split(model, processor, test_rows, out_dir, device, args, name="TEST")


def _lev(a: str, b: str) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if ai == b[j - 1] else 1))
        prev = cur
    return prev[m]


def evaluate_trocr_split(model, processor, rows: list, out_dir: Path, device, args, name: str = "TEST") -> None:
    """One-shot exact/CER on a held-out split. Predictions saved for audit."""
    import csv as _csv

    import torch

    model.eval()
    edit_sum = char_sum = exact = 0
    results = []
    with torch.no_grad():
        for i in range(0, len(rows), args.batch_size):
            chunk = rows[i:i + args.batch_size]
            imgs = [Image.open(out_dir / r.image_path).convert("RGB") for r in chunk]
            pixel_values = processor(images=imgs, return_tensors="pt").pixel_values.to(device)
            ids = model.generate(pixel_values, max_length=args.max_target_length)
            preds = processor.batch_decode(ids, skip_special_tokens=True)
            for r, pred in zip(chunk, preds):
                gt = r.text.strip()
                p = pred.strip()
                d = _lev(p, gt)
                edit_sum += d
                char_sum += len(gt)
                exact += (p == gt)
                results.append((r.image_path, gt, p, d))
    cer = edit_sum / max(1, char_sum)
    print(f"[{name}] crops={len(rows)}  exact={exact / max(1, len(rows)) * 100:.1f}%  CER={cer:.4f}")
    out_csv = out_dir / f"{name.lower()}_predictions.csv"
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["image_path", "gt", "pred", "edit"])
        w.writerows(results)
    print(f"[{name}] predictions -> {out_csv}")


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare_dataset(args)
    elif args.command == "export-easyocr":
        export_easyocr(args)
        print(f"[DONE] EasyOCR files -> {args.out_dir.resolve() / 'easyocr'}")
    elif args.command == "train-trocr":
        train_trocr(args)
    elif args.command == "all":
        prepare_dataset(args)
        train_trocr(args)


if __name__ == "__main__":
    main()
