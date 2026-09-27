"""공개 ABINet / MAERec 사전학습 가중치를 OpenOCR(openrec) 키 이름으로 변환합니다.

배경
----
OpenOCR 모델 zoo 의 Google Drive 폴더 두 개가 내려갔습니다(익명 접근도 404). 그래서
같은 모델의 **원저자 배포본**을 받아 openrec 구현체의 파라미터 이름으로 옮깁니다.

| 모델   | 출처                                                              | 내용 |
|--------|-------------------------------------------------------------------|------|
| ABINet | mmocr `abinet_20e_st-an_mj` (download.openmmlab.com, MJ+ST 학습)   | ResNet45 + 비전 트랜스포머 + 언어모델 |
| MAERec | Union14M `maerec_s_union14m.pth` (MAE 사전학습 ViT-S + NRTR 디코더) | Union14M-L 학습 |

두 저장소 모두 같은 논문 구현이지만 모듈 이름이 달라서 그대로 넣으면
`load_state_dict(strict=False)` 가 **한 텐서도 읽지 않고 조용히 지나갑니다**. 게다가
분류층은 영어 charset(37 / 94)이라 한국어 charset 과 모양이 달라 예외가 납니다.
이 스크립트는 이름을 옮기고, 모양이 안 맞는 텐서(분류층·임베딩)는 **빼고** 저장합니다.

사용
----
    .venv/Scripts/python.exe str_baselines/convert_pretrained_openocr.py \
        --model maerec --src <다운로드한 .pth>

    # 검증만 (저장 안 함)
    ... --model abinet --src <...> --dry-run
"""

import argparse
import collections
import os
import sys
import types

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPENOCR = os.path.join(ROOT, 'external', 'OpenOCR')

CONFIGS = {
    'abinet': 'configs/rec/abinet/abinet_signboard.yml',
    'maerec': 'configs/rec/maerec/maerec_signboard.yml',
}


# --------------------------------------------------------------------------
# mmocr 체크포인트는 mmengine 객체를 메타데이터로 갖고 있어 그냥은 언피클이 안 됩니다.
# 텐서만 필요하므로 없는 패키지를 아무 속성이나 내주는 더미로 대체합니다.
# --------------------------------------------------------------------------
class _DummyMeta(type):

    def __getattr__(cls, name):
        if name.startswith('__'):
            raise AttributeError(name)
        return cls()


class _Dummy(metaclass=_DummyMeta):

    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {'v': state})

    def __call__(self, *a, **k):
        return self

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        return _Dummy()


class _StubModule(types.ModuleType):
    __path__ = []

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        return _Dummy


def _install_stubs(prefixes=('mmengine', 'mmcv', 'mmocr')):

    class _Finder:

        def find_module(self, fullname, path=None):
            return self if fullname.split('.')[0] in prefixes else None

        def load_module(self, fullname):
            mod = sys.modules.get(fullname) or _StubModule(fullname)
            sys.modules[fullname] = mod
            return mod

    sys.meta_path.append(_Finder())
    for p in prefixes:
        sys.modules.setdefault(p, _StubModule(p))


def load_state_dict(path):
    _install_stubs()
    ckpt = torch.load(path, map_location='cpu')
    sd = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    return {k: v for k, v in sd.items() if torch.is_tensor(v)}


# --------------------------------------------------------------------------
# 변환 규칙
# --------------------------------------------------------------------------
def convert_maerec(src):
    """Union14M MAERec-S(mmocr NRTR 계열) → openrec ViT + NRTRDecoder.

    인코더는 이름만 바뀐 1:1 대응입니다. 디코더는 q/k/v 를 따로 두는 구현을
    qkv 하나로 합친 구현에 맞춰 이어 붙입니다(순서 q, k, v — 양쪽 모두
    reshape 전에 embed 축을 헤드로 나누므로 규약이 같습니다).
    원본 어텐션은 bias 가 없으므로 합친 쪽 bias 는 0 으로 둡니다(수학적으로 동일).
    """
    out = {}
    for k, v in src.items():
        if k.startswith('backbone.'):
            out['encoder.' + k[len('backbone.'):]] = v

    n_layers = 1 + max(
        (int(k.split('.')[2]) for k in src if k.startswith('decoder.layer_stack.')),
        default=-1)
    for i in range(n_layers):
        s = f'decoder.layer_stack.{i}.'
        t = f'decoder.decoder.{i}.'
        # self-attention: 따로 있는 q/k/v 를 하나로
        out[t + 'self_attn.qkv.weight'] = torch.cat([
            src[s + 'self_attn.linear_q.weight'],
            src[s + 'self_attn.linear_k.weight'],
            src[s + 'self_attn.linear_v.weight'],
        ], dim=0)
        out[t + 'self_attn.qkv.bias'] = torch.zeros(
            out[t + 'self_attn.qkv.weight'].shape[0])
        out[t + 'self_attn.out_proj.weight'] = src[s + 'self_attn.fc.weight']
        out[t + 'self_attn.out_proj.bias'] = torch.zeros(
            src[s + 'self_attn.fc.weight'].shape[0])
        # cross-attention: q 는 그대로, k/v 만 합칩니다
        out[t + 'cross_attn.q.weight'] = src[s + 'enc_attn.linear_q.weight']
        out[t + 'cross_attn.q.bias'] = torch.zeros(
            src[s + 'enc_attn.linear_q.weight'].shape[0])
        out[t + 'cross_attn.kv.weight'] = torch.cat([
            src[s + 'enc_attn.linear_k.weight'],
            src[s + 'enc_attn.linear_v.weight'],
        ], dim=0)
        out[t + 'cross_attn.kv.bias'] = torch.zeros(
            out[t + 'cross_attn.kv.weight'].shape[0])
        out[t + 'cross_attn.out_proj.weight'] = src[s + 'enc_attn.fc.weight']
        out[t + 'cross_attn.out_proj.bias'] = torch.zeros(
            src[s + 'enc_attn.fc.weight'].shape[0])
        # FFN · LayerNorm
        out[t + 'mlp.fc1.weight'] = src[s + 'mlp.w_1.weight']
        out[t + 'mlp.fc1.bias'] = src[s + 'mlp.w_1.bias']
        out[t + 'mlp.fc2.weight'] = src[s + 'mlp.w_2.weight']
        out[t + 'mlp.fc2.bias'] = src[s + 'mlp.w_2.bias']
        for a, b in (('norm1', 'norm1'), ('norm2', 'norm2'), ('norm3', 'norm3')):
            out[t + b + '.weight'] = src[s + a + '.weight']
            out[t + b + '.bias'] = src[s + a + '.bias']

    # 단어 임베딩·분류층은 영어 charset(94)이라 한국어 설정에서는 모양이 안 맞아
    # 자동으로 빠집니다. 대응만 적어 둡니다.
    for a, b in (('decoder.trg_word_emb.weight', 'decoder.embedding.embedding.weight'),
                 ('decoder.classifier.weight', 'decoder.tgt_word_prj.weight')):
        if a in src:
            out[b] = src[a]
    return out


def convert_abinet(src):
    """mmocr ABINet → openrec ResNet45 + ABINetDecoder.

    비전 트랜스포머는 양쪽 다 qkv 가 하나로 합쳐져 있어(in_proj) 그대로 옮깁니다.
    언어모델은 원본이 합쳐진 in_proj 인데 openrec 은 q 와 kv 로 나눠 두므로 잘라 줍니다.
    conv+bn 쌍에서 원본 conv 에 bias 가 없으면 0 으로 채웁니다(뒤 BN 이 흡수합니다).
    """
    out = {}

    def put_conv(t_prefix, weight, dim):
        out[t_prefix + '.weight'] = weight
        out[t_prefix + '.bias'] = torch.zeros(dim)

    for k, v in src.items():
        if k.startswith('backbone.'):
            out['encoder.' + k[len('backbone.'):]] = v

    # ── 비전 트랜스포머 (encoder.transformer.N → decoder.encoder.N)
    n_enc = 1 + max(
        (int(k.split('.')[2]) for k in src if k.startswith('encoder.transformer.')),
        default=-1)
    for i in range(n_enc):
        s = f'encoder.transformer.{i}.'
        t = f'decoder.encoder.{i}.'
        out[t + 'self_attn.qkv.weight'] = src[s + 'attentions.0.attn.in_proj_weight']
        out[t + 'self_attn.qkv.bias'] = src[s + 'attentions.0.attn.in_proj_bias']
        out[t + 'self_attn.out_proj.weight'] = src[s + 'attentions.0.attn.out_proj.weight']
        out[t + 'self_attn.out_proj.bias'] = src[s + 'attentions.0.attn.out_proj.bias']
        out[t + 'mlp.fc1.weight'] = src[s + 'ffns.0.layers.0.0.weight']
        out[t + 'mlp.fc1.bias'] = src[s + 'ffns.0.layers.0.0.bias']
        out[t + 'mlp.fc2.weight'] = src[s + 'ffns.0.layers.1.weight']
        out[t + 'mlp.fc2.bias'] = src[s + 'ffns.0.layers.1.bias']
        for a, b in (('norms.0', 'norm1'), ('norms.1', 'norm3')):
            out[t + b + '.weight'] = src[s + a + '.weight']
            out[t + b + '.bias'] = src[s + a + '.bias']

    # ── 비전 디코더 (decoder.vision_decoder → decoder.decoder)
    vd = 'decoder.vision_decoder.'
    n_ke = 1 + max(
        (int(k.split('.')[3]) for k in src if k.startswith(vd + 'k_encoder.')), default=-1)
    for i in range(n_ke):
        w = src[f'{vd}k_encoder.{i}.conv.weight']
        put_conv(f'decoder.decoder.k_encoder.{i}.0', w, w.shape[0])
        for suf in ('weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked'):
            key = f'{vd}k_encoder.{i}.bn.{suf}'
            if key in src:
                out[f'decoder.decoder.k_encoder.{i}.1.{suf}'] = src[key]
    n_kd = 1 + max(
        (int(k.split('.')[3]) for k in src if k.startswith(vd + 'k_decoder.')), default=-1)
    for i in range(n_kd):
        w = src[f'{vd}k_decoder.{i}.1.conv.weight']
        put_conv(f'decoder.decoder.k_decoder.{i}.w.0', w, w.shape[0])
        for suf in ('weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked'):
            key = f'{vd}k_decoder.{i}.1.bn.{suf}'
            if key in src:
                out[f'decoder.decoder.k_decoder.{i}.w.1.{suf}'] = src[key]
    for suf in ('weight', 'bias'):
        out['decoder.decoder.project.' + suf] = src[vd + 'project.' + suf]
    if vd + 'pos_encoder.position_table' in src:
        out['decoder.decoder.pos_encoder.pe'] = src[vd + 'pos_encoder.position_table']

    # ── 언어모델 (decoder.language_decoder → decoder.language)
    ld = 'decoder.language_decoder.'
    n_lm = 1 + max(
        (int(k.split('.')[3]) for k in src if k.startswith(ld + 'decoder_layers.')),
        default=-1)
    for i in range(n_lm):
        s = f'{ld}decoder_layers.{i}.'
        t = f'decoder.language.decoder.{i}.'
        w = src[s + 'attentions.0.attn.in_proj_weight']
        b = src[s + 'attentions.0.attn.in_proj_bias']
        d = w.shape[1]
        out[t + 'cross_attn.q.weight'] = w[:d]
        out[t + 'cross_attn.q.bias'] = b[:d]
        out[t + 'cross_attn.kv.weight'] = w[d:]
        out[t + 'cross_attn.kv.bias'] = b[d:]
        out[t + 'cross_attn.out_proj.weight'] = src[s + 'attentions.0.attn.out_proj.weight']
        out[t + 'cross_attn.out_proj.bias'] = src[s + 'attentions.0.attn.out_proj.bias']
        out[t + 'mlp.fc1.weight'] = src[s + 'ffns.0.layers.0.0.weight']
        out[t + 'mlp.fc1.bias'] = src[s + 'ffns.0.layers.0.0.bias']
        out[t + 'mlp.fc2.weight'] = src[s + 'ffns.0.layers.1.weight']
        out[t + 'mlp.fc2.bias'] = src[s + 'ffns.0.layers.1.bias']
        for a, bb in (('norms.0', 'norm2'), ('norms.1', 'norm3')):
            out[t + bb + '.weight'] = src[s + a + '.weight']
            out[t + bb + '.bias'] = src[s + a + '.bias']
    for a, b in (('pos_encoder.position_table', 'decoder.language.pos_encoder.pe'),
                 ('token_encoder.position_table', 'decoder.language.token_encoder.pe')):
        if ld + a in src:
            out[b] = src[ld + a]

    # ── 융합 게이트와 분류층.
    # openrec 은 정렬(alignment) 쪽에 _align 접미사를 붙입니다:
    #   원본 vision_decoder.cls  → cls        (비전 분기 분류층)
    #   원본 decoder.cls / w_att → cls_align / w_att_align  (융합 분기)
    # 분류층은 charset 의존이라 한국어 설정에서는 모양이 안 맞아 자동으로 빠집니다.
    # (영어 설정으로 되돌려 검증할 때를 위해 대응만 적어 둡니다.)
    pairs = [
        (vd + 'cls.weight', 'decoder.cls.weight'),
        (vd + 'cls.bias', 'decoder.cls.bias'),
        ('decoder.cls.weight', 'decoder.cls_align.weight'),
        ('decoder.cls.bias', 'decoder.cls_align.bias'),
        ('decoder.w_att.weight', 'decoder.w_att_align.weight'),
        ('decoder.w_att.bias', 'decoder.w_att_align.bias'),
        (ld + 'cls.weight', 'decoder.language.cls.weight'),
        (ld + 'cls.bias', 'decoder.language.cls.bias'),
        (ld + 'proj.weight', 'decoder.language.proj.weight'),
    ]
    for a, b in pairs:
        if a in src:
            out[b] = src[a]
    return out


CONVERTERS = {'abinet': convert_abinet, 'maerec': convert_maerec}


# --------------------------------------------------------------------------
def build_target_model(model_name):
    """설정 파일대로 모델을 만들어 (state_dict) 를 돌려줍니다. CPU 만 씁니다."""
    sys.path.insert(0, OPENOCR)
    cwd = os.getcwd()
    os.chdir(OPENOCR)
    try:
        from tools.engine.config import Config
        from openrec.modeling import build_model
        from openrec.postprocess import build_post_process

        cfg = Config(os.path.join(OPENOCR, CONFIGS[model_name])).cfg
        post = build_post_process(cfg['PostProcess'], cfg['Global'])
        cfg['Architecture']['Decoder']['out_channels'] = post.get_character_num()
        model = build_model(cfg['Architecture'])
    finally:
        os.chdir(cwd)
    return model


def group_of(key):
    parts = key.split('.')
    if parts[0] == 'encoder':
        return 'encoder(백본)'
    if len(parts) > 1 and parts[1] in ('encoder', 'decoder', 'language'):
        return f'decoder.{parts[1]}'
    return 'decoder(기타)'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=sorted(CONVERTERS))
    ap.add_argument('--src', required=True, help='원저자 배포 체크포인트(.pth)')
    ap.add_argument('--out', default=None,
                    help='기본값 external/OpenOCR/pretrained/<model>/best.pth')
    ap.add_argument('--dry-run', action='store_true', help='저장하지 않고 검증만')
    args = ap.parse_args()

    print(f'[1/4] 원본 읽기: {args.src}')
    src = load_state_dict(args.src)
    print(f'      텐서 {len(src)}개')

    print('[2/4] 키 이름 변환')
    converted = CONVERTERS[args.model](src)
    print(f'      변환 {len(converted)}개')

    print('[3/4] 설정대로 모델을 만들어 대조')
    target = build_target_model(args.model).state_dict()

    kept, dropped_shape, dropped_absent = {}, [], []
    for k, v in converted.items():
        if k not in target:
            dropped_absent.append(k)
        elif tuple(target[k].shape) != tuple(v.shape):
            dropped_shape.append((k, tuple(v.shape), tuple(target[k].shape)))
        else:
            kept[k] = v

    filled = collections.Counter(group_of(k) for k in kept)
    total = collections.Counter(group_of(k) for k in target)
    print('\n      그룹별 적재율 (적재/전체 텐서)')
    for g in sorted(total):
        print(f'        {g:22s} {filled.get(g, 0):4d} / {total[g]:4d}')
    print(f'        {"합계":22s} {len(kept):4d} / {len(target):4d}'
          f'  ({100 * len(kept) / len(target):.1f}%)')

    if dropped_absent:
        print(f'\n      모델에 없는 이름 {len(dropped_absent)}개 (예: {dropped_absent[:3]})')
    if dropped_shape:
        print(f'      모양 불일치로 제외 {len(dropped_shape)}개 — charset 의존 층이어야 정상:')
        for k, a, b in dropped_shape[:6]:
            print(f'        {k}  원본{a} vs 모델{b}')

    missing = [k for k in target if k not in kept]
    miss_g = collections.Counter(group_of(k) for k in missing)
    print(f'\n      무작위 초기화로 남는 텐서 {len(missing)}개: '
          + ', '.join(f'{g} {c}' for g, c in miss_g.most_common()))

    if args.dry_run:
        print('\n[4/4] --dry-run 이라 저장하지 않습니다.')
        return

    out = args.out or os.path.join(OPENOCR, 'pretrained', args.model, 'best.pth')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({'state_dict': kept}, out)
    print(f'\n[4/4] 저장: {out}')
    print('      설정의 pretrained_model 경로가 이 파일을 가리킵니다.')


if __name__ == '__main__':
    main()
