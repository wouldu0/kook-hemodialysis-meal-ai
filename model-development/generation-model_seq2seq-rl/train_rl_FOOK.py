# -*- coding: utf-8 -*-
"""
train_rl_FOOK.py — RL 파인튜닝: "레버가 유저 메뉴를 덜 뜯게 하는 궁합" 학습

아이디어(중복 아님):
  영양 달성은 레버가 보장한다. RL은 레버가 못 하는 일 = "원본(유저 지정메뉴)을 덜 망치는
  반찬 궁합"을 배운다. 유저 메뉴는 앵커로 고정하고, 모델은 나머지 4칸만 고른다.
  보상 = (통과영양수/5) × (0.5 + 0.5 × 보존율)   ← reward_lever_FOOK
  근거: 5영양이 똑같이 전부 통과하는 끼들 안에서도 앵커 보존율이 0.00~1.00로 갈림(std 0.33).

설정:
  - on-policy + stochastic : CE 타깃이 '모델이 뽑은 토큰' → loss=-log π(a)×adv = REINFORCE.
                             (off-policy면 CE 타깃이 실제데이터라 정책이 보상을 못 배움)
  - use_baseline=True      : advantage = reward - 앵커슬롯별 평균 (분산감소 + 슬롯편향 제거)
  - use_beta=True          : incidence matrix(메뉴×슬롯 실제 동시출현) = 자연스러움 사전분포.
                             다양성 붕괴 방어.
  - 반드시 모방 체크포인트에서 warm-start (맨바닥 RL은 밥-국-찬 구조를 잃음)

실행 (foodbert):
  conda activate foodbert
  set TF_USE_LEGACY_KERAS=1
  cd Diet-Generation-As-Sequence-master\\Diet-Generation-As-Sequence-master\\Code
  python train_rl_FOOK.py --epochs 200 --lr 1e-5
"""
import os, sys, time, argparse
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import pandas as pd
import tensorflow as tf

from util import sequence_to_sentence
from Model import Sequence_Generator
from train_FOOK import build_data, NUT as NUT_CSV

FINAL = os.path.abspath(os.path.join('..', '..', '..'))   # E:/final
sys.path.insert(0, FINAL)
SPECIAL = {0, 825, 826}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--lr', type=float, default=1e-5)      # 파인튜닝: 모방보다 100x 작게
    ap.add_argument('--imit', type=float, default=0.3)     # 모방 닻 가중 (0=순수RL: 구조·다양성 붕괴)
    ap.add_argument('--weight', type=int, default=60)
    ap.add_argument('--init', type=str, default='./results_FOOK/checkpoints')   # 모방 ckpt
    ap.add_argument('--out', type=str, default='./results_rl_FOOK/checkpoints')
    ap.add_argument('--embed_dim', type=int, default=128)
    ap.add_argument('--fc_dim', type=int, default=64)
    ap.add_argument('--network', type=str, default='GRU')
    args = ap.parse_args()

    # 레버 + 보상 (cwd를 E:/final로 바꿔야 식약청 xlsx 상대경로가 맞음)
    cwd = os.getcwd()
    os.chdir(FINAL)
    import FOOK_adjust_levers as F
    import reward_lever_FOOK as R
    print('식약청 DB + 레시피 로딩...')
    F.NUT = F.load_all()
    R.init(weight=args.weight)
    os.chdir(cwd)

    nutrient_data, food_dict, diet_np, incidence = build_data()
    batch_size = int(diet_np.shape[0])
    print(f'데이터: {batch_size}끼, 토큰 {len(food_dict)}')

    # pred_seqs -> 메뉴명 5칸 (pred_seqs = [start, slot0..slot4, end])
    def seqs_to_menus(pred_seqs):
        out = []
        for row in np.asarray(pred_seqs)[:, 1:6]:
            out.append([food_dict[int(t)] if int(t) not in SPECIAL else None for t in row])
        return out

    def reward_fn(pred_seqs, anchor_slots):
        menus_list = seqs_to_menus(pred_seqs)
        rs = np.zeros(len(menus_list), dtype=float)
        for i, menus in enumerate(menus_list):
            if any(m is None for m in menus):   # 특수토큰이 슬롯에 나온 비정상 한끼
                rs[i] = 0.0
                continue
            a = anchor_slots[i] if anchor_slots is not None else 2
            rs[i] = R.meal_reward(menus, menus[int(a)])
        return rs

    kwargs = {
        'fully-connected_layer': args.network, 'attention': True,
        'embed_dim': args.embed_dim, 'fc_dim': args.fc_dim,
        'learning': 'on-policy',      # ★ REINFORCE 성립 조건
        'policy': 'stochastic',       # ★ 탐색
        'use_beta': True,             # 슬롯 적합성 = 자연스러움 사전분포(다양성 붕괴 방어)
        'use_buffer': False, 'buffer_size': 5, 'buffer_update': 5,
        'num_epochs': args.epochs, 'lr': args.lr,
        'num_tokens': len(food_dict), 'batch_size': batch_size,
        'imitation_only': False,      # ★ RL on
        'reward_fn': reward_fn,       # ★ 레버 조정량 보상
        'use_baseline': True,         # ★ advantage = r - 슬롯별평균
        'imit_weight': args.imit,     # ★ 모방 닻 (구조·다양성 유지)
    }

    gen = Sequence_Generator(food_dict, nutrient_data, incidence, **kwargs)
    x = tf.convert_to_tensor(diet_np)

    # ★ 모방 체크포인트에서 warm-start
    ck = tf.train.latest_checkpoint(args.init)
    if not ck:
        print(f'[중단] 모방 체크포인트 없음: {args.init}. train_FOOK.py 먼저 실행하세요.')
        return
    tf.train.Checkpoint(generator=gen).restore(ck).expect_partial()
    print(f'warm-start: {ck}')

    os.makedirs(args.out, exist_ok=True)
    ckpt = tf.train.Checkpoint(generator=gen)
    rng = np.random.default_rng(0)

    start = time.time()
    for epoch in range(args.epochs):
        # 샘플마다 유저가 고정한 슬롯을 랜덤으로 가정 (앱: 유저가 아무 슬롯 메뉴나 지정)
        anchor_slots = rng.integers(0, 5, size=batch_size)
        real_seqs, batch_loss, pred_seqs, _, _, _, _ = gen.train(x, x, anchor_slots=anchor_slots)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            r = gen.last_reward
            uniq = len(set(int(t) for t in np.asarray(pred_seqs)[:, 1:6].reshape(-1)))
            print(f'[epoch {epoch+1:4d}] loss={batch_loss:.4f}  보상 평균={r.mean():.3f} std={r.std():.3f}  '
                  f'고유메뉴={uniq}  예시={sequence_to_sentence(pred_seqs, food_dict)[0]}')
    print(f'총 시간 {time.time()-start:.1f}s | 보상캐시 {R.cache_stats()}건')
    ckpt.save(file_prefix=os.path.join(args.out, 'ckpt'))
    print(f'체크포인트 저장: {args.out}')


if __name__ == '__main__':
    main()
