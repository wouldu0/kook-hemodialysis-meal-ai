# -*- coding: utf-8 -*-
"""
eval_rl_FOOK.py — 모방 vs RL 체크포인트 정면 비교 (학습보상 아님, 실제 지표)

질문: RL이 "레버가 유저 메뉴(앵커)를 덜 뜯게" 만들었나?
측정: 앵커 고정 → 나머지 4칸 생성 → 레버 → 앵커 보존율 / 전체 보존율 / 5영양 통과 / 다양성

주의: 앱(app_core)은 여기에 슬롯마스킹+resample을 더 얹는다. 여기선 정책 자체의 효과를
      분리해 보려고 순수 샘플링만 쓴다.

실행:
  conda activate foodbert; set TF_USE_LEGACY_KERAS=1
  cd .../Code
  python eval_rl_FOOK.py --n 400
"""
import os, sys, copy, argparse
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import numpy as np
import tensorflow as tf

from Model import Sequence_Generator
from train_FOOK import build_data

FINAL = os.path.abspath(os.path.join('..', '..', '..'))
sys.path.insert(0, FINAL)
SPECIAL = {0, 825, 826}


def load_gen(ckpt_dir, food_dict, nutrient_data, incidence, batch_size):
    kwargs = {'fully-connected_layer': 'GRU', 'attention': True, 'embed_dim': 128, 'fc_dim': 64,
              'learning': 'on-policy', 'policy': 'stochastic', 'use_beta': False, 'use_buffer': False,
              'buffer_size': 5, 'buffer_update': 5, 'num_epochs': 1, 'lr': 1e-3,
              'num_tokens': len(food_dict), 'batch_size': batch_size, 'imitation_only': True}
    g = Sequence_Generator(food_dict, nutrient_data, incidence, **kwargs)
    ck = tf.train.latest_checkpoint(ckpt_dir)
    if not ck:
        return None, None
    tf.train.Checkpoint(generator=g).restore(ck).expect_partial()
    return g, ck


def generate(gen, seeds, anchor_slots, food_dict, temp=1.0):
    """앵커 슬롯은 seed의 실제 토큰으로 고정, 나머지는 정책에서 샘플링."""
    n = seeds.shape[0]
    enc_hidden = tf.zeros([n, gen.encoder.units])
    enc_output, enc_hidden = gen.encoder(seeds, enc_hidden)
    dec_hidden = copy.deepcopy(enc_hidden)
    res = np.zeros((n, 5), dtype=int)
    for j in range(5):
        outputs, dec_hidden, _ = gen.decoder(seeds[:, j], dec_hidden, enc_output)
        probs = np.array(outputs, dtype=float)
        if probs.ndim == 1:
            probs = probs[None, :]
        for b in range(n):
            if anchor_slots[b] == j:
                res[b, j] = seeds[b, j + 1]        # 앵커 고정
                continue
            p = probs[b].copy()
            for t in SPECIAL:
                p[t] = 0.0
            p = np.clip(p, 1e-12, None)
            p = p ** (1.0 / temp)
            p /= p.sum()
            res[b, j] = int(np.random.choice(len(p), p=p))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=400)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--ckpts', type=str, default='',
                    help='쉼표구분 "라벨=경로" 목록. 비우면 기본 3개.')
    args = ap.parse_args()

    cwd = os.getcwd()
    os.chdir(FINAL)
    import FOOK_adjust_levers as F
    import reward_lever_FOOK as R
    print('식약청 DB 로딩...')
    F.NUT = F.load_all()
    R.init(weight=60)
    os.chdir(cwd)

    nutrient_data, food_dict, diet_np, incidence = build_data()
    bs = int(diet_np.shape[0])

    np.random.seed(args.seed)
    idx = np.random.choice(bs, size=args.n, replace=True)
    seeds_all = diet_np.numpy()[idx]
    anchors_all = np.random.randint(0, 5, size=args.n)

    if args.ckpts:
        cands = [tuple(x.split('=', 1)) for x in args.ckpts.split(',')]
    else:
        cands = [('모방 (baseline)', './results_FOOK/checkpoints'),
                 ('RL imit=0.3',     './results_rl_FOOK/checkpoints'),
                 ('RL imit=0.05',    './results_rl_FOOK/ckpt_i05')]

    print(f'\n{"":<18} {"앵커보존":>8} {"전체보존":>8} {"5영양통과":>9} {"통과영양/5":>10} {"고유메뉴":>8}')
    print('-' * 68)
    for name, d in cands:
        gen, ck = load_gen(d, food_dict, nutrient_data, incidence, bs)
        if gen is None:
            print(f'{name:<18} (체크포인트 없음: {d})')
            continue
        np.random.seed(args.seed)          # 생성 샘플링도 동일 시드
        toks = generate(gen, seeds_all, anchors_all, food_dict)
        aks, oks, pf, ok5 = [], [], [], []
        uniq = set()
        for b in range(args.n):
            menus = [food_dict[int(t)] for t in toks[b]]
            uniq.update(menus)
            anchor = menus[int(anchors_all[b])]
            _, det = R.meal_reward(menus, anchor, detail=True)
            if not det:
                continue
            aks.append(det['a_keep']); oks.append(det['o_keep'])
            pf.append(det['pass_frac']); ok5.append(det['pass_frac'] == 1.0)
        print(f'{name:<18} {np.mean(aks):>8.3f} {np.mean(oks):>8.3f} '
              f'{100*np.mean(ok5):>8.0f}% {np.mean(pf):>10.3f} {len(uniq):>8}')
    print('\n해석: 앵커보존↑ = 레버가 유저 메뉴를 덜 뜯음(RL의 목적). '
          '5영양통과가 유지되고 고유메뉴가 안 줄어야 유효.')


if __name__ == '__main__':
    main()
