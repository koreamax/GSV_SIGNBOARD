#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Package a trained EasyOCR-trainer recognition model into the 3-file plug-in
layout EasyOCR expects, so it can be used via recog_network=<name>.

Writes:
  <user_network_dir>/<name>.py     (Model class: None-VGG-BiLSTM-CTC)
  <user_network_dir>/<name>.yaml   (imgH, lang_list, character_list, network_params)
  <model_dir>/<name>.pth           (copied trained weights)

Defaults to ~/.EasyOCR. The trainer saves a DataParallel state_dict, which
EasyOCR.get_recognizer loads directly on GPU and de-prefixes on CPU.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

MODEL_PY = '''import torch.nn as nn
from easyocr.model.modules import VGG_FeatureExtractor, BidirectionalLSTM


class Model(nn.Module):
    def __init__(self, input_channel, output_channel, hidden_size, num_class):
        super(Model, self).__init__()
        self.FeatureExtraction = VGG_FeatureExtractor(input_channel, output_channel)
        self.FeatureExtraction_output = output_channel
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size),
        )
        self.SequenceModeling_output = hidden_size
        self.Prediction = nn.Linear(self.SequenceModeling_output, num_class)

    def forward(self, input, text=None):
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))
        visual_feature = visual_feature.squeeze(3)
        contextual_feature = self.SequenceModeling(visual_feature)
        prediction = self.Prediction(contextual_feature.contiguous())
        return prediction
'''


def build(name: str, pth: Path, character_path: Path, easyocr_dir: Path,
          imgh: int, input_channel: int, output_channel: int, hidden_size: int,
          lang_list: list[str]) -> None:
    character = character_path.read_text(encoding="utf-8")
    user_net = easyocr_dir / "user_network"
    model_dir = easyocr_dir / "model"
    user_net.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    (user_net / f"{name}.py").write_text(MODEL_PY, encoding="utf-8")

    # YAML written by hand to control character_list quoting (it can contain ':' etc.)
    yaml_lines = [
        f"imgH: {imgh}",
        "lang_list:",
        *[f"  - '{l}'" for l in lang_list],
        "network_params:",
        f"  input_channel: {input_channel}",
        f"  output_channel: {output_channel}",
        f"  hidden_size: {hidden_size}",
    ]
    # character_list as a YAML double-quoted scalar; escape backslash, quote, and
    # control chars (e.g. 0x08) which YAML forbids as raw bytes. \xXX round-trips
    # via yaml.safe_load, preserving the exact class indices the model was trained on.
    def _esc(c: str) -> str:
        o = ord(c)
        if c == "\\":
            return "\\\\"
        if c == '"':
            return '\\"'
        if o < 0x20 or o == 0x7F:
            return "\\x%02x" % o
        return c
    esc = "".join(_esc(c) for c in character)
    yaml_lines.append(f'character_list: "{esc}"')
    (user_net / f"{name}.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    shutil.copyfile(pth, model_dir / f"{name}.pth")

    print(f"[DONE] {user_net / (name + '.py')}")
    print(f"[DONE] {user_net / (name + '.yaml')}")
    print(f"[DONE] {model_dir / (name + '.pth')}")
    print(f"[DONE] chars={len(character)}  use: easyocr.Reader({lang_list}, recog_network='{name}')")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="traffic_sign_custom")
    ap.add_argument("--pth", type=Path, required=True, help="trained .pth (best_accuracy.pth)")
    ap.add_argument("--character", type=Path, required=True, help="character.txt from training")
    ap.add_argument("--easyocr-dir", type=Path, default=Path.home() / ".EasyOCR")
    ap.add_argument("--imgH", type=int, default=64)
    ap.add_argument("--input-channel", type=int, default=1)
    ap.add_argument("--output-channel", type=int, default=256)
    ap.add_argument("--hidden-size", type=int, default=256)
    ap.add_argument("--lang", nargs="+", default=["ko", "en"])
    args = ap.parse_args()
    build(args.name, args.pth, args.character, args.easyocr_dir,
          args.imgH, args.input_channel, args.output_channel, args.hidden_size, args.lang)


if __name__ == "__main__":
    main()
