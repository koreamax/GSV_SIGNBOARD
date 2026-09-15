#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Union14M-L SVTRv2-B(svtrv2_smtr_gtc_rctc) 체크포인트 → 한국어 RCTC 미세조정 초기값.

- encoder.* : 그대로
- decoder.ctc_decoder.* : 접두사를 decoder.* 로 바꿔 RCTCDecoder 에 맞춤 (분류층 fc.* 는 사전 크기가
  다르므로 제외 → 무작위 초기화)
- decoder.gtc_decoder.* : 학습 보조 분기라 버림
결과를 새 config 로 만든 모델에 strict=False 로 로드해 누락 키를 보고합니다.
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
OPENOCR = HERE / "external" / "OpenOCR"
sys.path.insert(0, str(OPENOCR))
SRC = OPENOCR / "pretrained" / "svtrv2_b" / "best.pth"
DST = HERE / "artifacts" / "str_baselines" / "svtrv2_b_union14m_init.pth"
CFG = OPENOCR / "configs" / "rec" / "svtrv2" / "svtrv2_rctc_signboard.yml"


def main() -> None:
    sd = torch.load(SRC, map_location="cpu")
    sd = sd.get("state_dict", sd)
    out, dropped = {}, []
    for k, v in sd.items():
        if k.startswith("encoder."):
            out[k] = v
        elif k.startswith("decoder.ctc_decoder."):
            nk = "decoder." + k[len("decoder.ctc_decoder."):]
            if nk in ("decoder.fc.weight", "decoder.fc.bias"):
                dropped.append(k)          # 사전 크기 불일치
            else:
                out[nk] = v
        else:
            dropped.append(k)
    print(f"kept {len(out)} keys, dropped {len(dropped)} (gtc {sum('gtc' in k for k in dropped)}, fc {sum('fc.' in k and 'ctc' in k for k in dropped)})")

    from tools.engine.config import Config
    from openrec.modeling import build_model
    from openrec.postprocess import build_post_process
    cfg = Config(str(CFG)).cfg
    post = build_post_process(cfg["PostProcess"], cfg["Global"])
    cfg["Architecture"]["Decoder"]["out_channels"] = post.get_character_num()
    model = build_model(cfg["Architecture"])
    msd = model.state_dict()
    bad = [k for k, v in out.items() if k in msd and tuple(msd[k].shape) != tuple(v.shape)]
    for k in bad:
        out.pop(k)
    res = model.load_state_dict(out, strict=False)
    print(f"model keys {len(msd)} | shape-mismatch removed {len(bad)} {bad[:5]}")
    print(f"missing (random init) {len(res.missing_keys)}: {res.missing_keys[:10]}")
    print(f"unexpected {len(res.unexpected_keys)}: {res.unexpected_keys[:10]}")
    print(f"decoder out_channels = {post.get_character_num()}")
    DST.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": out}, DST)
    print("saved", DST)


if __name__ == "__main__":
    main()
