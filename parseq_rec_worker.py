#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PARSeq(공식 strhub) 라인 인식 워커 — paddle_rec_worker 와 같은 IO 계약.

  --ckpt     Lightning 체크포인트(.ckpt) (train_parseq_signboard.py 산출 best)
  --manifest / --out : {"key","path"} → {"key","text","score"}
전처리는 학습과 동일한 SceneTextDataModule.get_transform(img_size 32x128).
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "external" / "parseq"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from strhub.data.module import SceneTextDataModule
    from strhub.models.utils import load_from_checkpoint

    model = load_from_checkpoint(str(Path(args.ckpt).resolve())).eval().to(args.device)
    tf = SceneTextDataModule.get_transform(model.hparams.img_size)
    items = [json.loads(l) for l in open(args.manifest, encoding="utf-8") if l.strip()]
    with open(args.out, "w", encoding="utf-8") as f, torch.inference_mode():
        for s in range(0, len(items), args.batch_size):
            chunk = items[s:s + args.batch_size]
            x = torch.stack([tf(Image.open(it["path"]).convert("RGB")) for it in chunk]).to(args.device)
            p = model(x).softmax(-1)
            preds, confs = model.tokenizer.decode(p)
            for it, t, c in zip(chunk, preds, confs):
                sc = float(c.mean()) if hasattr(c, "mean") and c.numel() else 0.0
                f.write(json.dumps({"key": it["key"], "text": t, "score": sc}, ensure_ascii=False) + "\n")
            if (s // args.batch_size) % 20 == 0:
                print(f"[parseq] {min(s + args.batch_size, len(items))}/{len(items)}", flush=True)
    print(f"[parseq] done {len(items)} -> {args.out}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
