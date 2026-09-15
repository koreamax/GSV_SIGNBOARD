#!/usr/bin/env python3
"""train.py 의 미세조정판 — 사전학습(parseq, 영어 94자) 가중치를 **형상이 맞는 키만** 로드.

원본 train.py 는 strict 로드라 문자 집합이 바뀌면(한국어 11,978자) 실패합니다. 여기서는
text_embed / head 처럼 출력 크기가 다른 텐서만 무작위 초기화하고 나머지(ViT 인코더·디코더
블록·pos_queries)는 그대로 씁니다. 나머지 로직은 train.py 와 동일.
Usage (cwd = external/parseq):
  python train_parseq_signboard.py +experiment=... (train.py 와 같은 hydra 오버라이드)
"""
import math
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.model_summary import summarize

from strhub.data.module import SceneTextDataModule
from strhub.models.base import BaseSystem
from strhub.models.utils import get_pretrained_weights


def _annealing_cos(start, end, pct):
    cos_out = math.cos(math.pi * pct) + 1
    return end + (start - end) / 2.0 * cos_out


def get_swa_lr_factor(warmup_pct, swa_epoch_start, div_factor=25, final_div_factor=1e4) -> float:
    total_steps = 1000
    start_step = int(total_steps * warmup_pct) - 1
    end_step = total_steps - 1
    step_num = int(total_steps * swa_epoch_start) - 1
    pct = (step_num - start_step) / (end_step - start_step)
    return _annealing_cos(1, 1 / (div_factor * final_div_factor), pct)


def load_pretrained_filtered(m: torch.nn.Module, name: str) -> None:
    sd = get_pretrained_weights(name)
    msd = m.state_dict()
    keep = {k: v for k, v in sd.items() if k in msd and tuple(msd[k].shape) == tuple(v.shape)}
    skipped = [k for k in sd if k not in keep]
    res = m.load_state_dict(keep, strict=False)
    print(f"[pretrained={name}] loaded {len(keep)}/{len(sd)} tensors; "
          f"skipped (shape/absent) {skipped}; missing {res.missing_keys}")


@hydra.main(config_path='configs', config_name='main', version_base='1.2')
def main(config: DictConfig):
    trainer_strategy = 'auto'
    with open_dict(config):
        config.data.root_dir = hydra.utils.to_absolute_path(config.data.root_dir)
        gpu = config.trainer.get('accelerator') == 'gpu'
        if gpu:
            config.trainer.precision = 'bf16-mixed' if torch.get_autocast_gpu_dtype() is torch.bfloat16 else '16-mixed'

    if config.model.get('perm_mirrored', False):
        assert config.model.perm_num % 2 == 0, 'perm_num should be even if perm_mirrored = True'

    model: BaseSystem = hydra.utils.instantiate(config.model)
    if config.pretrained is not None:
        m = model.model if config.model._target_.endswith('PARSeq') else model
        load_pretrained_filtered(m, config.pretrained)
    print(summarize(model, max_depth=2))

    datamodule: SceneTextDataModule = hydra.utils.instantiate(config.data)

    checkpoint = ModelCheckpoint(
        monitor='val_accuracy', mode='max', save_top_k=1, save_last=True,
        filename='{epoch}-{step}-{val_accuracy:.4f}-{val_NED:.4f}',
    )
    swa_epoch_start = 0.75
    swa_lr = config.model.lr * get_swa_lr_factor(config.model.warmup_pct, swa_epoch_start)
    swa = StochasticWeightAveraging(swa_lr, swa_epoch_start)
    cwd = (HydraConfig.get().runtime.output_dir if config.ckpt_path is None
           else str(Path(config.ckpt_path).parents[1].absolute()))
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, logger=TensorBoardLogger(cwd, '', '.'), strategy=trainer_strategy,
        enable_model_summary=False, callbacks=[checkpoint, swa],
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=config.ckpt_path)


if __name__ == '__main__':
    main()
