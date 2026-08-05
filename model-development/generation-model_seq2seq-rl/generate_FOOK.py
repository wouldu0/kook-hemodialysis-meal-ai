# -*- coding: utf-8 -*-
"""
generate_FOOK.py — 학습된 모델로 한끼 생성 (샘플링 + 중복금지 + 특수토큰 마스킹)

train_FOOK.py로 학습한 체크포인트(results_FOOK/checkpoints)를 복원해 한끼를 뽑는다.
원본 inference()는 greedy(argmax)라 매번 같은 결과 -> 여기선 temperature 샘플링으로 다양화.
seed(실제 한끼)를 인코더 입력으로 주고, 각 슬롯을 확률적으로 샘플. 중복 메뉴/특수토큰(empty,시작,종료) 금지.

실행:
  conda activate foodbert
  set TF_USE_LEGACY_KERAS=1
  python generate_FOOK.py --n 8 --temp 0.8
"""
import os, argparse, copy
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import numpy as np
import pandas as pd
import tensorflow as tf

from util import (nutrition_preprocessor, diet_sequence_preprocessor,
                  food_to_token, diet_to_incidence, sequence_to_sentence)
from Model import Sequence_Generator

DATA = os.path.join('..', '..', '..', 'data')
CKPT = './results_FOOK/checkpoints'
SPECIAL = {0, 825, 826}   # empty, 시작, 종료

# 한 끼에 하나만 허용하는 카테고리 그룹 (중복 방지): 밥류/국류/김치
UNIQUE_GROUPS = [{'밥', '일품밥', '일품(간식)'}, {'국', '수프(간식)'}, {'김치'}]


def build_slot_masks(food_dict):
    """토큰별 Class, 슬롯별 허용 Class, 유니크그룹 반환 (슬롯 마스킹용)."""
    from collections import defaultdict
    nut = pd.read_csv(os.path.join(DATA, 'FOOK_nutrition.csv'))
    menu2class = dict(zip(nut['name'], nut['Class']))
    token_class = {i: menu2class.get(n) for i, n in food_dict.items()}
    meals = pd.read_csv(os.path.join(DATA, 'FOOK_meals_for_model.csv'))
    slot_allowed = defaultdict(set)
    for _, r in meals.iterrows():
        for j in range(5):
            c = menu2class.get(r.iloc[j])
            if isinstance(c, str):
                slot_allowed[j].add(c)

    def group_of(cls):
        for gi, g in enumerate(UNIQUE_GROUPS):
            if cls in g:
                return gi
        return None
    return token_class, dict(slot_allowed), group_of


def build():
    feature = pd.read_csv(os.path.join(DATA, 'FOOK_nutrition.csv'))
    nutrient_data, food_dict = nutrition_preprocessor(feature_data=feature)()
    diet = pd.read_csv(os.path.join(DATA, 'FOOK_meals_for_model.csv'))
    dsp = diet_sequence_preprocessor(sequence_data=diet, DB_quality='correct2', integrate=False)
    diet = dsp(nutrient_data)
    diet_np = food_to_token(diet, nutrient_data, empty_delete=True, num_empty=3)
    incidence = diet_to_incidence(diet_np, food_dict)
    return nutrient_data, food_dict, diet_np, incidence


def restore(food_dict, nutrient_data, incidence, batch_size):
    kwargs = {'fully-connected_layer': 'GRU', 'attention': True, 'embed_dim': 128, 'fc_dim': 64,
              'learning': 'off-policy', 'policy': 'greedy', 'use_beta': False, 'use_buffer': False,
              'buffer_size': 5, 'buffer_update': 5, 'num_epochs': 1, 'lr': 1e-3,
              'num_tokens': len(food_dict), 'batch_size': batch_size, 'imitation_only': True}
    gen = Sequence_Generator(food_dict, nutrient_data, incidence, **kwargs)
    ckpt = tf.train.Checkpoint(generator=gen)
    ckpt.restore(tf.train.latest_checkpoint(CKPT)).expect_partial()
    return gen


def generate(gen, seeds, temp=0.8, masks=None):
    """seeds: (B,7) int [시작,5메뉴,종료]. 각 슬롯 확률샘플 + 중복/특수토큰 금지.
    masks=(slot_ok, tok_group, group_tokens) 주면 슬롯 카테고리 + 밥/국/김치 중복 마스킹."""
    seeds = np.asarray(seeds)
    B, L = seeds.shape
    enc_hidden = tf.zeros([B, gen.encoder.units])
    enc_output, enc_hidden = gen.encoder(seeds, enc_hidden)
    dec_hidden = copy.deepcopy(enc_hidden)

    result = np.zeros((B, L), dtype=int)
    result[:, 0] = seeds[:, 0]          # 시작
    result[:, -1] = 826                 # 종료
    used = [set() for _ in range(B)]
    used_grp = [set() for _ in range(B)]

    for j in range(L - 2):              # 5개 슬롯(위치 1..5)만 샘플
        outputs, dec_hidden, _ = gen.decoder(seeds[:, j], dec_hidden, enc_output)
        probs = np.array(outputs, dtype=float)
        if probs.ndim == 1:
            probs = probs[None, :]
        for b in range(B):
            p = probs[b].copy()
            for t in SPECIAL:           # 특수토큰 금지
                p[t] = 0.0
            for t in used[b]:           # 중복 메뉴 금지
                p[t] = 0.0
            if masks is not None:
                slot_ok, tok_group, group_tokens = masks
                masked = p * slot_ok[j]                 # 슬롯 허용 카테고리만
                for gi in used_grp[b]:                  # 이미 쓴 밥/국/김치 그룹 금지
                    masked[group_tokens[gi]] = 0.0
                if masked.sum() > 0:                    # 마스킹 후 후보 있으면 적용
                    p = masked
            p = np.clip(p, 1e-12, None)
            p = p ** (1.0 / temp)       # temperature
            p /= p.sum()
            tok = int(np.random.choice(len(p), p=p))
            result[b, j + 1] = tok
            used[b].add(tok)
            if masks is not None:
                gi = masks[1].get(tok)
                if gi is not None:
                    used_grp[b].add(gi)
    return result


def make_masks(food_dict):
    """generate()에 넘길 마스크 3종 생성."""
    token_class, slot_allowed, group_of = build_slot_masks(food_dict)
    vocab = len(food_dict)
    slot_ok = {}
    for j in range(5):
        arr = np.zeros(vocab, dtype=float)
        for t, c in token_class.items():
            if isinstance(c, str) and c in slot_allowed.get(j, set()):
                arr[t] = 1.0
        slot_ok[j] = arr
    tok_group = {t: group_of(c) for t, c in token_class.items()
                 if isinstance(c, str) and group_of(c) is not None}
    group_tokens = {gi: [t for t, g in tok_group.items() if g == gi]
                    for gi in range(len(UNIQUE_GROUPS))}
    return slot_ok, tok_group, group_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--temp', type=float, default=0.8)
    ap.add_argument('--seed', type=int, default=None)
    args = ap.parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    nutrient_data, food_dict, diet_np, incidence = build()
    gen = restore(food_dict, nutrient_data, incidence, batch_size=diet_np.shape[0])
    masks = make_masks(food_dict)       # 슬롯 카테고리 + 중복 마스킹

    # 랜덤 실제 한끼를 seed로
    idx = np.random.choice(diet_np.shape[0], size=args.n, replace=False)
    seeds = diet_np.numpy()[idx]
    out = generate(gen, seeds, temp=args.temp, masks=masks)

    print(f'\n=== 생성 한끼 {args.n}개 (temp={args.temp}) ===')
    seed_txt = sequence_to_sentence(seeds, food_dict)
    gen_txt = sequence_to_sentence(out, food_dict)
    for i in range(args.n):
        s = [m for m in seed_txt[i] if m not in ('시작', '종료')]
        g = [m for m in gen_txt[i] if m not in ('시작', '종료')]
        print(f'\n[{i+1}] seed : {s}')
        print(f'    생성 : {g}')


if __name__ == '__main__':
    main()
